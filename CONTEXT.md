# InfoSkill

InfoSkill 是一个面向具身智能体的 Skill 压缩与演化框架。核心思想：用一个 VAE 风格的编码器将"当前状态 + 技能文本"压缩成少量连续向量（Soft Prefix），注入冻结 LLM 的输入端，同时用信息论指标动态维护技能库。

## Language

**Fast Module**:
VAE 风格的在线编码–投影链路，每步都运行。包含 StateConditionalEncoder、PriorNetwork、Projector、RewardPredictor、GroundingDecoder 五个子模块。每步输出 Soft Prefix 注入 LLM。
_Avoid_: 在线模块、encoder pipeline

**Slow Module**:
每 T 步触发一次的技能库维护例程。依据 Fidelity 和 Rate 指标对库中技能做增删，并用本地 Qwen 从成功轨迹生成新技能候选。
_Avoid_: 离线模块、library update

**Skill**:
技能库中的一条条目，结构为 `{skill_id, title, principle, when_to_apply}`。`title + principle` 拼接后是 GroundingDecoder 的重建目标，也是 embedding 检索的基准文本。
_Avoid_: 经验、策略片段

**Skill Library (B)**:
所有 Skill 的动态集合，以 JSON 持久化。按 `general_skills` / `task_specific_skills` / `common_mistakes` 三类组织，初始化自 SkillRL 的 `claude_style_skills.json`（55 条）。
_Avoid_: 技能池、memory

**Episode**:
一次完整的任务尝试：从 `env.reset()` 到 `done=True` 或达到 `max_steps`（默认 50）。Episode 是奖励计算和 GRPO 分组的基本单位。
_Avoid_: 回合、trajectory（trajectory 特指记录了完整 (obs, action, reward) 序列的数据结构）

**Group (G)**:
GRPO 中同一任务下并行运行的 G 条 Episode（默认 G=8）。同一 Group 内的所有 Episode 共享同一个初始任务描述，用于组内奖励归一化。
_Avoid_: batch（batch 特指 model forward 时的 tensor 维度）

**GRPO Advantage**:
组内归一化后的奖励信号：`A_i = (R_i - mean(R)) / (std(R) + ε)`，其中 R 为 Episode 总奖励。是 policy_loss 的监督信号，也是 fidelity_loss 的预测目标。
_Avoid_: 优势函数（避免和标准 GAE 混淆）

**z_tilde**:
Encoder 用重参数化采样得到的压缩潜码，形状 `[batch, latent_dim]`（默认 latent_dim=64）。是 Fast Module 内部所有子模块的核心中间变量。
_Avoid_: latent vector、隐变量（这些太泛）

**Soft Prefix**:
Projector 将 z_tilde 投影成的 m 个连续向量，形状 `[batch, m, hidden_size]`（默认 m=8，hidden_size=3584）。拼接到 LLM `inputs_embeds` 最前面作为软提示，不经过 tokenizer。
_Avoid_: 软提示（soft prompt 已泛化，Soft Prefix 强调位置在序列头部）

**Fidelity**:
技能对拿高分的贡献程度，由 RewardPredictor 的预测精度衡量：`−MSE(pred_advantage, actual_advantage)`，越大越好。Slow Module 用它决定技能是否有保留价值。
_Avoid_: 保真度（在代码注释里可以用，但设计文档里统一用 Fidelity）

**Rate**:
压缩技能文本所需的信息量，等于后验分布与先验分布之间的 KL 散度：`KL(q(z|s,z_text) ‖ p(z|s))`，越小越好（压缩越高效）。
_Avoid_: 压缩率

**MIG (Mutual Information Gain)**:
技能的留存判据：`MIG = Fidelity − β × Rate`。MIG > 0 则保留，MIG ≤ 0 则剪枝。是 Algorithm 2（Slow Module 决策逻辑）的核心公式。
_Avoid_: 信息增益（information gain 有其他含义）

**BODY**:
ALFWorld 原有的低层执行控制器（TextWorld 引擎），接收高层文本命令并在模拟器中执行，完全保留不修改。对应 `AlfredTWEnv` 下的 TextWorld `env.step(action_str)` 调用链。
_Avoid_: 底层控制器、executor

**BRAIN**:
原 ALFWorld seq2seq 策略网络的替代品，即 Qwen2.5-7B-Instruct（纯文本，主干冻结 + LoRA 微调）。通过接收 Soft Prefix + tokenized 文本 obs 生成动作文本。ALFWorld 使用 `AlfredTWEnv` 文字模式，环境本身不产生图像（符号状态经模板 NLG 直接转文字），因此 BRAIN 不需要视觉能力。
_Avoid_: LLM agent、policy model（在设计层用 BRAIN，在代码层用 llm/model）

**Active Mask**:
长度为 G 的布尔张量，标记哪些 Episode 仍在运行。每步 batch forward 前用它跳过已完成的 Episode 的 `env.step`，loss 计算时置零其贡献。
_Avoid_: done_mask（done 是环境返回值，Active Mask 是训练侧对齐用的）

**Trajectory Buffer**:
单次 Group 采集过程中积累的 `(state_text, skill_text, z_tilde, log_prob, reward, done)` 序列，每步 append，Episode 结束后用于计算本 Group 的 GRPO Advantage 并送入 loss。
_Avoid_: replay buffer（那是 off-policy 的术语）
