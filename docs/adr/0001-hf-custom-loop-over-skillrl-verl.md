# 用 HuggingFace 自定义训练循环替代 SkillRL/verl 框架

InfoSkill 的核心机制是把 VAE 编码后的 Soft Prefix（连续向量）拼接到 LLM 的 `inputs_embeds` 最前面。SkillRL 的 rollout 引擎是 vLLM，而 vLLM 的 `generate()` 接口只接受 `input_ids`，不支持 `inputs_embeds`，导致 Soft Prefix 在采样阶段无法注入，论文核心机制在 rollout 时实际失效。

因此选择用 HuggingFace `transformers` 原生 `model.generate(inputs_embeds=...)` + PEFT/LoRA 搭建独立的轻量训练循环，完全绕开 vLLM/verl/Ray 的分布式层。SkillRL 仅作为 ALFWorld 环境适配的参考（`AlfworldWorker`、`alfworld_projection`、prompt 模板等），不作为训练框架依赖。

## Considered Options

- **SkillRL/verl + vLLM rollout**：分布式支持好，但 vLLM 不兼容 `inputs_embeds`，Soft Prefix 在 rollout 阶段无法生效，论文机制残缺。
- **修改 vLLM 使其支持 `inputs_embeds`**：工程量极大，需 fork vLLM 并维护 patch，风险高。
- **HuggingFace 自定义循环（选定）**：`generate(inputs_embeds=...)` 原生支持，保持论文完整性；7B 单卡/双卡规模下无需分布式，复杂度合理。

## Consequences

- G 条 Episode 并排跑，每 step 做一次 batch forward（批量大小固定为 G，用 Active Mask 处理提前完成的 Episode）；wall-clock 速度慢于 vLLM，但在单实验机上可接受。
- 未来若需扩展到多机多卡，可在 HF + FSDP 路线上扩展，仍不需要引入 vLLM。
