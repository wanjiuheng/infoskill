# 方案二：AlfredThorEnv + MaskRCNN 感知鲁棒性补充实验

## 背景

方案一（`option1-text-alfredtw` 分支）使用 `AlfredTWEnv`（TextWorld 模板文字环境）+ 纯文本 LLM（Qwen2.5-7B-Instruct），环境设定与 SkillRL 完全一致，用于产出可直接对比的主实验表格。

但 `AlfredTWEnv` 下的观测文字是从 ground-truth 符号状态模板生成的，不含任何真实感知噪声，无法回答"InfoSkill 的技能压缩机制在真实感知噪声下是否依然有效"这个问题。方案二正是为了补上这一维度。

## 决策

使用 `AlfredThorEnv`（ai2thor 物理渲染环境）+ 原版 Mask-RCNN 检测器（`mrcnn_astar` controller）+ 纯文本 LLM（与方案一同一模型家族：Qwen2.5-7B-Instruct / Qwen2.5-3B-Instruct / Qwen3-1.7B-Instruct）作为**独立的感知鲁棒性补充实验**。

- 不进入与 SkillRL 直接对比的主实验表格（SkillRL 全部实验跑在 `AlfredTWEnv`，两者任务难度不同，成功率不可直接同表比较）。
- BUTLER::BRAIN 继续用纯文本 LLM，不引入 Qwen2.5-VL，保证 7B/3B/1.7B 等不同规模的 backbone 消融实验里，VISION 模块（检测器）始终保持不变，只有 BRAIN 在变，变量干净。
- BUTLER::VISION 复用 `alfworld-master` 原版 Mask-RCNN 检测器，不用任何 LLM 承担这个角色。

## 架构设计

### 环境层

- `env.type: AlfredThorEnv`（ai2thor 引擎真实渲染 RGB，替代 `AlfredTWEnv` 的模板文字）
- `controller.type: mrcnn_astar`（原版 BUTLER 架构：Mask-RCNN 目标检测 + A* 导航，对应 `alfworld-master/alfworld/agents/controller/mrcnn_astar.py` 及其父类 `MaskRCNNAgent`）
- `mask_rcnn.pretrained_model_path`：需通过 `alfworld-download` 拉取 `$ALFWORLD_DATA/detectors/mrcnn.pth`（原始 ALFWorld 论文训练的检测器权重，非本项目训练产物）

### 感知与决策分工（保持 BUTLER 三段式）

- **BUTLER::VISION** = 原版 Mask-RCNN（`mrcnn_astar` controller 内部调用 `get_maskrcnn_prediction()`，从渲染帧中检测物体、生成文字反馈），完全复用 `alfworld-master` 代码，不引入任何 LLM。
- **BUTLER::BRAIN** = Qwen2.5-7B-Instruct（与方案一相同），接收 `controller.step()` 返回的文字反馈——这段文字由检测结果生成，而非模板 ground-truth，因此天然带有检测噪声（漏检/错检）——拼接 Soft Prefix 后生成动作文本。从 LLM 视角看，输入输出接口和方案一完全一致，只是文字背后的来源变了。
- **BUTLER::BODY** = `ThorEnv`（ai2thor 底层动作执行，`alfworld-master/alfworld/env/thor_env.py`），完全不变。

关键点：`mrcnn_astar` controller 对外仍然吐出**纯文本反馈**（`self.feedback`），因此 InfoSkill 现有的 Fast/Slow Module、Encoder、Soft Prefix 注入机制完全不需要改动——唯一要新增的是一个新的环境封装类，把 `AlfredThorEnv` 接到同样的 `BaseEnvWrapper` 接口上。

### 后续实现阶段需要新增的文件

- `envs/alfworld_thor_config.yaml`：基于 `alfworld-master/configs/eval_config.yaml` 裁剪，设置 `env.type: AlfredThorEnv`、`controller.type: mrcnn_astar`、`mask_rcnn.pretrained_model_path` 指向 `$ALFWORLD_DATA/detectors/mrcnn.pth`。
- `envs/alfworld_thor_env.py`：新增 `AlfredThorTextEnv`，对外接口与现有 `AlfworldTextEnv` 一致（`reset()`/`step()` 返回文字 obs），内部改为调用 `get_environment("AlfredThorEnv")`，`batch_size=1` 跑单实例（沿用现有"G 个 env 并排跑"的并行方式）。
- `configs/alfworld_thor.yaml`：复制 `configs/alfworld.yaml`，仅替换 `paths.alfworld_config` 指向新 env 配置文件，其余训练超参数（loss 权重、rollout、GRPO 等）保持一致，确保除感知模态外变量单一。

### 已知风险 / 待验证项

1. **ai2thor 无头渲染**：确认实验环境为 H100/H20 Linux GPU 服务器，用 `xvfb-run` 起虚拟显示是 ai2thor 官方支持的标准无头渲染方案，风险较低；仍需在实际服务器上跑一次最小化验证（`Xvfb` 是否已装、ai2thor 版本与驱动是否兼容）。之前担心的 Windows 本机兼容性问题不适用，无需处理。
2. **检测器权重下载**：需要 `alfworld-download` 拉取 `detectors/mrcnn.pth`。
3. **与 SkillRL 的可比性**：若想让方案二也有一个公平 baseline，需要用户自行在 `AlfredThorEnv` 下重跑一遍 SkillRL（不能直接抄它论文里 `AlfredTWEnv` 模式的数字）。
4. **Rollout 速度**：真实物理仿真 + 实时 Mask-RCNN 推理远慢于模板文字生成，G 个并行 env 的 wall-clock 开销会明显上升，可能需要视情况缩小 `group_size` 或 `num_episodes`。

## 结论

方案二是一条独立的补充实验分支，用来回答"InfoSkill 的技能压缩机制在真实感知噪声下是否依然带来提升"，不影响方案一（`AlfredTWEnv` + 纯文本，对标 SkillRL 主表格）的地位。当前仅完成分支创建与方案记录，具体代码实现（`alfworld_thor_env.py` 等）留待后续开发。
