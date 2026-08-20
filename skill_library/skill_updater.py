"""
skill_library/skill_updater.py

Slow Module — SkillUpdater: use the local Qwen2.5-7B-Instruct model to summarise
successful trajectories into new Skill candidates.

Called by InfoskillTrainer every T global steps when episode_reward ≥ threshold.
"""

import json
import re
import uuid
from typing import List, Dict, Optional


# ── Prompt template ───────────────────────────────────────────────────────────

_SKILL_GEN_PROMPT = """You are an expert at distilling agent experience into reusable, policy-facing skills.

Below are {n} successful trajectories for the task type "{task_type}".
Each trajectory is a sequence of (observation → action) pairs that led to success.

{trajectories}

Based on these successes, write ONE new reusable skill in JSON format:
{{
  "title":         "<short imperative phrase, ≤8 words>",
  "principle":     "<one-to-two sentence policy: the core decision rule / action ordering that made these trajectories work>",
  "when_to_apply": "<one sentence describing the trigger condition>"
}}

Constraints:
- Write "principle" as an actionable rule the agent can follow, NOT a retrospective summary of what happened.
- Output ONLY the JSON object, nothing else.
"""


class SkillUpdater:
    """
    Generates new Skill candidates from successful episode trajectories.

    Uses the same local Qwen model as the policy (passed in, already loaded).
    Generation is triggered lazily by InfoskillTrainer; this class is stateless
    beyond its reference to the model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_new_tokens: int = 256,
        temperature:    float = 0.3,
    ) -> None:
        self._model         = model
        self._tokenizer     = tokenizer
        self._device        = device
        self._max_new_tokens = max_new_tokens
        self._temperature   = temperature

    def generate_skill(
        self,
        trajectories: List[Dict],
        task_type:    str = "pick_and_place",
    ) -> Optional[Dict]:
        """
        Summarise a batch of successful trajectories into one new Skill dict.

        Args:
            trajectories: List of trajectory dicts, each with key 'steps':
                          [{'obs': str, 'action': str}, ...]
            task_type:    Task category string for prompt context.

        Returns:
            Skill dict {skill_id, title, principle, when_to_apply} or None on parse failure.
        """
        traj_text = self._format_trajectories(trajectories)
        prompt = _SKILL_GEN_PROMPT.format(
            n=len(trajectories),
            task_type=task_type,
            trajectories=traj_text,
        )

        # Build chat-formatted input
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        # Generate with the local model (no soft prefix — this is Slow Module)
        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return self._parse_skill(raw)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_trajectories(trajectories: List[Dict]) -> str:
        """Convert list of trajectory dicts to a readable string."""
        parts = []
        for i, traj in enumerate(trajectories, 1):
            steps = traj.get("steps", [])
            step_lines = "\n".join(
                f"  Obs: {s['obs']}\n  Action: {s['action']}"
                for s in steps
            )
            parts.append(f"--- Trajectory {i} (task: {traj.get('task', '')}) ---\n{step_lines}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_skill(raw: str) -> Optional[Dict]:
        """Extract JSON object from model output."""
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Find the outermost JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            d = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        # Validate required fields
        if not d.get("title") or not d.get("principle"):
            return None

        d["skill_id"]     = "dyn_" + str(uuid.uuid4())[:8]
        d.setdefault("when_to_apply", "")
        return d
