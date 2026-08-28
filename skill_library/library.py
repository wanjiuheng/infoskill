"""
skill_library/library.py

SkillLibrary: loads, caches, retrieves, adds, and prunes Skills.

Retrieval:
  - Skill embeddings are pre-computed once at load time using the LLM's
    token embedding layer (frozen, mean-pooled).  See utils/embedding.py.
  - At each step, the current task description is embedded on-the-fly
    and cosine-similarity ranked against the cached skill embeddings.

Slow Module (pruning):
  - evaluate_skill() computes a per-skill MIG estimate using the
    RewardPredictor and PriorNetwork from the last-N-steps usage history.
  - prune() removes skills whose MIG ≤ 0.
  - add_skill() inserts new skills and updates the embedding cache.
"""

import json
import os
import re
import uuid
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.embedding import get_text_embedding, cosine_similarity_matrix


# ── Data structure ────────────────────────────────────────────────────────────

class Skill:
    """One skill entry from the JSON bank."""
    __slots__ = ("skill_id", "title", "principle", "when_to_apply", "category")

    def __init__(self, d: Dict, category: str = "general"):
        self.skill_id    = d.get("skill_id", d.get("mistake_id", str(uuid.uuid4())[:8]))
        if "title" not in d and "description" in d:
            # common_mistakes schema: mistake_id/description/why_it_happens/how_to_avoid
            # (no title/principle keys) — map onto the same title/principle fields
            # so grounding_text / format_for_prompt / save() need no special-casing.
            self.title       = f"Avoid: {d.get('description', '')}"
            self.principle   = d.get("how_to_avoid", "")
            self.when_to_apply = d.get("why_it_happens", "")
        else:
            self.title       = d["title"]
            self.principle   = d["principle"]
            self.when_to_apply = d.get("when_to_apply", "")
        self.category    = category   # 'general' | task-type name | 'mistake'

    @property
    def grounding_text(self) -> str:
        """Target text for GroundingDecoder: title + principle concatenated."""
        return f"{self.title}: {self.principle}"

    def to_dict(self) -> Dict:
        return {
            "skill_id":     self.skill_id,
            "title":        self.title,
            "principle":    self.principle,
            "when_to_apply": self.when_to_apply,
        }

    def __repr__(self) -> str:
        return f"Skill({self.skill_id!r}: {self.title!r})"


# ── Main class ────────────────────────────────────────────────────────────────

class SkillLibrary:
    """
    Dynamic skill bank with embedding-based cosine retrieval.

    Args:
        json_path:       Path to claude_style_skills.json (seed bank).
        model:           LLM model instance (for embedding).
        tokenizer:       Matching tokenizer.
        device:          Compute device.
        top_k_task:      Number of task-specific skills to inject per step (K).
        max_skills:      Hard cap on total skills (oldest surplus removed).
    """

    def __init__(
        self,
        json_path:     str,
        model,
        tokenizer,
        device:        torch.device,
        top_k_task:    int = 6,
        max_skills:    int = 200,
    ) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self._device    = device
        self.top_k_task    = top_k_task
        self.max_skills    = max_skills
        self._json_path    = json_path

        # Flat list of all skills (general + task-specific + mistakes)
        self._skills: List[Skill] = []
        # Embedding cache: [N, hidden_size] tensor (row i ↔ self._skills[i])
        self._embeddings: Optional[torch.Tensor] = None

        # Retrieve-result cache: (query_text, task_type) → (gen, task). Skill
        # selection is task-level (fixed per episode), so the same query is
        # re-embedded every step; this avoids recomputing the query embedding
        # and similarity ranking. Cleared whenever the skill bank changes.
        self._retrieve_cache: "OrderedDict" = OrderedDict()
        self._retrieve_cache_cap = 64

        self._load(json_path)
        self._build_embedding_cache()

    # ── Loading / saving ──────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Parse JSON into a flat Skill list."""
        assert os.path.exists(path), f"Skills JSON not found: {path}"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for s in data.get("general_skills", []):
            self._skills.append(Skill(s, category="general"))

        for cat, items in data.get("task_specific_skills", {}).items():
            for s in items:
                self._skills.append(Skill(s, category=cat))

        for s in data.get("common_mistakes", []):
            self._skills.append(Skill(s, category="mistake"))

        print(f"[SkillLibrary] Loaded {len(self._skills)} skills from {path}")

    def load(self, path: str) -> None:
        """
        用 path 中的技能库替换当前库（resume 恢复用）。
        清空现有技能，重载 checkpoint 的 skills.json 并重建 embedding cache。
        """
        self._skills = []
        self._load(path)
        self._build_embedding_cache()

    def save(self, path: Optional[str] = None) -> None:
        """Persist current skill set back to JSON."""
        out_path = path or self._json_path
        general, task_specific, mistakes = [], {}, []

        for s in self._skills:
            if s.category == "general":
                general.append(s.to_dict())
            elif s.category == "mistake":
                mistakes.append(s.to_dict())
            else:
                task_specific.setdefault(s.category, []).append(s.to_dict())

        data = {
            "general_skills":      general,
            "task_specific_skills": task_specific,
            "common_mistakes":     mistakes,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SkillLibrary] Saved {len(self._skills)} skills → {out_path}")

    # ── Embedding cache ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _build_embedding_cache(self) -> None:
        """(Re)compute embeddings for all skills using get_text_embedding()."""
        # Skill bank changed → similarity scores change → stale retrieve hits
        self._retrieve_cache.clear()
        texts = [s.grounding_text for s in self._skills]
        if not texts:
            self._embeddings = torch.zeros(
                0, self._model.config.hidden_size, device=self._device
            )
            return

        # Batch to avoid OOM on large libraries
        batch_size = 32
        emb_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = get_text_embedding(batch, self._model, self._tokenizer, self._device)
            emb_list.append(emb)
        self._embeddings = torch.cat(emb_list, dim=0)          # [N, hidden]
        print(f"[SkillLibrary] Embedding cache built: {self._embeddings.shape}")

    def _invalidate_cache(self) -> None:
        """Rebuild embedding cache after skills are added or removed."""
        self._build_embedding_cache()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def retrieve(
        self,
        query_text: str,
        task_type:  Optional[str] = None,
    ) -> Tuple[List[Skill], List[Skill]]:
        """
        Retrieve skills for a step, mirroring SkillRL.

        General skills are always returned in full (foundational guidance);
        task-specific skills are the top-K most similar within the detected
        task_type category (K = top_k_task). If the category is empty or
        task_type is unknown, fall back to cross-category top-K.

        Args:
            query_text: Typically the task description string.
            task_type:  Detected task category (e.g. "heat") used to restrict
                        the task-specific candidate pool.

        Returns:
            (general_skills, task_skills): general in library order, task top-K.
        """
        if self._embeddings is None or len(self._skills) == 0:
            return [], []

        # Task-level skills are fixed per episode → same query every step. Hit
        # the cache (if present) to skip re-embedding and re-ranking.
        key = (query_text, task_type)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # Embed the query
        q_emb = get_text_embedding(
            query_text, self._model, self._tokenizer, self._device
        )  # [1, hidden] or [hidden]
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)

        # Cosine similarities [1, N] → [N]
        sims = cosine_similarity_matrix(q_emb, self._embeddings).squeeze(0)

        def topk(idxs, k):
            if not idxs:
                return []
            idx_t = torch.tensor(idxs, device=self._device)
            scores = sims[idx_t]
            top = scores.topk(min(k, len(idxs))).indices
            return [self._skills[idxs[i]] for i in top.tolist()]

        # General: always included in full, kept in library order for stability
        gen_skills = [s for s in self._skills if s.category == "general"]

        # Task-specific: restrict to the detected category (empty/unknown → fallback)
        task_idxs = (
            [i for i, s in enumerate(self._skills) if s.category == task_type]
            if task_type is not None else []
        )
        if not task_idxs:
            task_idxs = [
                i for i, s in enumerate(self._skills)
                if s.category not in ("general", "mistake")
            ]
        task_skills = topk(task_idxs, self.top_k_task)
        self._cache_put(key, gen_skills, task_skills)
        return gen_skills, task_skills

    def _cache_get(self, key: Tuple[str, Optional[str]]):
        if key in self._retrieve_cache:
            self._retrieve_cache.move_to_end(key)
            gen, task = self._retrieve_cache[key]
            return list(gen), list(task)
        return None

    def _cache_put(
        self,
        key:          Tuple[str, Optional[str]],
        gen_skills:   List[Skill],
        task_skills:  List[Skill],
    ) -> None:
        # Store tuples so callers mutating the returned lists cannot poison the cache.
        self._retrieve_cache[key] = (tuple(gen_skills), tuple(task_skills))
        self._retrieve_cache.move_to_end(key)
        while len(self._retrieve_cache) > self._retrieve_cache_cap:
            self._retrieve_cache.popitem(last=False)

    def format_for_prompt(
        self,
        general_skills: List[Skill],
        task_skills:    List[Skill],
    ) -> str:
        """Render retrieved skills into a markdown block for prompt injection."""
        lines = []
        if general_skills:
            lines.append("### General Principles")
            for s in general_skills:
                lines.append(f"- **{s.title}**: {s.principle}")
        if task_skills:
            lines.append("### Task-Specific Skills")
            for s in task_skills:
                lines.append(f"- **{s.title}**: {s.principle}")
                if s.when_to_apply:
                    lines.append(f"  *(Apply when: {s.when_to_apply})*")
        return "\n".join(lines)

    # ── Skill for Encoder (single best match) ─────────────────────────────────

    @torch.no_grad()
    def retrieve_for_encoder(
        self,
        query_text: str,
        task_type:  Optional[str] = None,
    ) -> Skill:
        """
        Return the single best task-specific skill (within task_type) for
        feeding into the Encoder. Falls back to the first skill if the
        library is empty.
        """
        gen, task = self.retrieve(query_text, task_type=task_type)
        all_retrieved = task + gen   # task-specific first
        if all_retrieved:
            return all_retrieved[0]
        return self._skills[0] if self._skills else Skill(
            {"skill_id": "empty", "title": "Explore", "principle": "Explore systematically."}
        )

    # ── Slow Module: add / prune ──────────────────────────────────────────────

    def add_skill(self, skill_dict: Dict, category: str = "general") -> bool:
        """
        Add a new skill to the library.

        Returns True if added, False if a duplicate title was found.
        Rebuilds the embedding cache after a successful add.
        Enforces max_skills by removing the oldest surplus skill.
        """
        title = skill_dict.get("title", "").strip()
        if any(s.title.lower() == title.lower() for s in self._skills):
            return False   # duplicate — skip

        new_skill = Skill(skill_dict, category=category)
        self._skills.append(new_skill)

        # 容量上限统一由 prune() 按 MIG 末位淘汰处理（库超过 max_skills 时
        # 定期触发），这里不做即时删除——否则库永远停在 max_skills，prune 的
        # MIG 排序淘汰逻辑永远不会执行。
        self._invalidate_cache()
        return True

    def remove_skill(self, skill_id: str) -> bool:
        """
        Remove a skill by its skill_id.
        Returns True if removed, False if not found.
        """
        for i, s in enumerate(self._skills):
            if s.skill_id == skill_id:
                self._skills.pop(i)
                self._invalidate_cache()
                return True
        return False

    @torch.no_grad()
    def evaluate_mig(
        self,
        skill: Skill,
        encoder,
        prior_net,
        reward_predictor,
        state_texts: List[str],      # 原始文本 — recent usage history（exp4：
                                       # 不再是预先算好的向量，见 trainer.py
                                       # 的 _resolve_usage_history）
        advantages: torch.Tensor,    # [N]  — corresponding GRPO advantages
        beta: float = 0.001,
    ) -> float:
        """
        Compute the MIG (Mutual Information Gain) for one skill.
        MIG = Fidelity − β × Rate

        Fidelity = −MSE(pred_advantage, actual_advantage)  (higher = more useful)
        Rate     = mean KL(posterior ‖ prior)               (lower = cheaper to store)

        Args:
            skill:           The skill being evaluated.
            encoder:         T5StateConditionalEncoder instance (eval mode)。
                             吃原始文本，内部返回 (mu, log_var, pooled_state)。
            prior_net:       PriorNetwork instance (eval mode)，输入 pooled_state。
            reward_predictor: RewardPredictor instance (eval mode)，输入 pooled_state。
            state_texts:      Recent state texts from steps where skill was used.
            advantages:      Corresponding GRPO advantages.
            beta:            Compression weight (same as training beta).

        Returns:
            Scalar MIG value.
        """
        import torch.nn.functional as F
        from models.encoder import sample_z
        from training.losses import compute_rate_loss

        if len(state_texts) == 0:
            return 0.0

        skill_texts = [skill.grounding_text] * len(state_texts)
        mu, log_var, pooled_state = encoder(state_texts, skill_texts)
        z_tilde = sample_z(mu, log_var)

        # Fidelity: how well z predicts the advantage
        pred_adv = reward_predictor(z_tilde, pooled_state)
        fidelity = -F.mse_loss(pred_adv, advantages).item()

        # Rate: KL divergence vs prior
        prior_mu, prior_logvar = prior_net(pooled_state)
        rate = compute_rate_loss(mu, log_var, prior_mu, prior_logvar).item()

        return fidelity - beta * rate

    def prune(
        self,
        encoder,
        prior_net,
        reward_predictor,
        usage_history: Dict[str, Tuple[List[str], torch.Tensor]],
        beta: float = 0.001,
        min_uses: int = 5,
    ) -> List[str]:
        """
        Slow Module: 库超过 max_skills 时才触发末位淘汰（按 MIG 升序）。

        未使用或使用不足 min_uses 的技能无法可靠评估 MIG，记为 +inf（优先
        保留），避免训练早期模型未收敛时误删 seed 技能。

        Args:
            usage_history: Dict mapping skill_id → (state_texts, advantages)
                           （exp4：state_texts 是原始文本列表，不再是预先
                           算好的向量，见 training/trainer.py 的
                           _resolve_usage_history）。
            min_uses:      Skip skills that haven't been used enough to estimate.

        Returns:
            List of pruned skill_ids.
        """
        if len(self._skills) <= self.max_skills:
            return []  # 库没满，不触发剪枝

        # 评估每个技能的 MIG；无数据/数据不足的技能 MIG=+inf（最后才删）
        scored: List[Tuple[float, int, Skill]] = []
        for idx, skill in enumerate(self._skills):
            entries = usage_history.get(skill.skill_id)
            if entries is None:
                scored.append((float("inf"), idx, skill))
                continue
            state_texts, advs = entries
            if len(state_texts) < min_uses:
                scored.append((float("inf"), idx, skill))
                continue
            mig = self.evaluate_mig(
                skill, encoder, prior_net, reward_predictor,
                state_texts, advs, beta=beta,
            )
            scored.append((mig, idx, skill))

        # MIG 升序：最没用的排最前优先删。第二个 key 用原始序号，避免 inf 并列
        # 时 Python 回退到比较 Skill 对象（未定义 <）而抛 TypeError。
        scored.sort(key=lambda x: (x[0], x[1]))

        n_remove = len(self._skills) - self.max_skills
        remove_ids = set()
        for mig, _, skill in scored[:n_remove]:
            if mig == float("inf"):
                print(f"[SkillLibrary] Pruned unused skill {skill.skill_id!r}: {skill.title!r}")
            else:
                print(f"[SkillLibrary] Pruned skill {skill.skill_id!r} "
                      f"(MIG={mig:.4f}): {skill.title!r}")
            remove_ids.add(skill.skill_id)

        # 批量删除，只重建一次 embedding cache
        self._skills = [s for s in self._skills if s.skill_id not in remove_ids]
        self._invalidate_cache()
        return list(remove_ids)

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self):
        return iter(self._skills)
