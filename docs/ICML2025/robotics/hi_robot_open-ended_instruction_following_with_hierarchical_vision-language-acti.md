---
title: >-
  [论文解读] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
description: >-
  [ICML 2025][机器人][层次化机器人控制] 提出 Hi Robot，一个层次化 VLM 系统：高层 VLM 将复杂用户指令/反馈推理为原子命令，低层 VLA (π0) 执行动作，结合合成数据生成方案，在三类机器人平台上实现了远超 GPT-4o 和扁平 VLA 的开放式指令跟随能力。
tags:
  - ICML 2025
  - 机器人
  - 层次化机器人控制
  - 视觉-语言-动作模型 (VLA)
  - 合成数据
  - 开放式指令跟随
  - 人机交互
---

# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

**会议**: ICML 2025  
**arXiv**: [2502.19417](https://arxiv.org/abs/2502.19417)  
**代码**: [项目主页](https://www.pi.website/research/hirobot)  
**领域**: 机器人  
**关键词**: 层次化机器人控制, 视觉-语言-动作模型 (VLA), 合成数据, 开放式指令跟随, 人机交互

## 一句话总结

提出 Hi Robot，一个层次化 VLM 系统：高层 VLM 将复杂用户指令/反馈推理为原子命令，低层 VLA (π0) 执行动作，结合合成数据生成方案，在三类机器人平台上实现了远超 GPT-4o 和扁平 VLA 的开放式指令跟随能力。

## 研究背景与动机

当前机器人指令跟随系统面临一个根本性挑战：**简单原子指令 vs. 复杂开放式交互**。现有 VLA 模型（如 RT-2、π0）虽然能执行 "拿起杯子" 这样的简单命令，但无法应对真实场景中的复杂需求，例如：

- **复合意图指令**："帮我做一个素三明治，不要番茄，另外如果有牛肉，再给我朋友做一个"
- **情境反馈**："那个不是垃圾"、"别管其余的了"
- **动态纠正**："你得再低一点，不然一直抓不到"

作者类比 Kahneman 的双系统理论：
- **System 1（快思考）** = 低层策略，执行原子操作（抓取、放置）
- **System 2（慢思考）** = 高层推理，解析复杂指令、整合反馈、规划下一步

先前工作主要聚焦 System 1 级别的行为（简单指令执行），或使用 LLM/VLM 配合预定义技能（物理灵巧性受限）。Hi Robot 的核心动机是**同时实现高层推理的灵活性和低层控制的灵巧性**，且能处理开放式用户交互。

## 方法详解

### 整体框架

Hi Robot 将策略分解为两层 VLM 推理过程：

```
用户复杂指令 ℓ_t + 图像观测 I_t → [高层 VLM] → 原子命令 ℓ̂_t (+ 语音回复 u_t)
                                                      ↓
图像观测 I_t + 原子命令 ℓ̂_t + 机器人状态 q_t → [低层 VLA (π0)] → 动作块 A_t
```

**运行频率分离**：
- 低层策略：高频输出动作块（~10 Hz，配合 action chunking 可达 50 Hz）
- 高层策略：低频推理（每 1 秒重新推理一次，或收到用户新输入时立即触发）

**关键接口**：高层和低层通过**自然语言**连接——高层输出的原子命令本质上是低层 VLA 训练时见过的简短语言指令（如 "pick up one piece of lettuce"），这构成了一个灵活且可解释的中间表征。

### 关键设计

#### 1. 层次化推理 (Hierarchical Inference)

高层策略 $p_{\text{hi}}(\hat{\ell}_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \ell_t)$ 接收多相机图像和开放式指令，输出原子命令。低层策略 $p_{\text{lo}}(\mathbf{A}_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$ 使用该命令生成动作。

对于简单熟悉任务，可直接 $\hat{\ell}_t = \ell_t$；层次化结构的优势在于：
- 指令过于复杂，低层策略无法直接解析
- 指令在机器人训练数据语境中不常见
- 涉及与用户的动态交互

#### 2. 用户交互集成 (User Interaction)

用户可以在任务执行过程中随时介入（文本或语音转文字），系统立即触发高层重新推理。高层策略还可以输出语音回复 $u_t$（如确认、澄清），通过 TTS 播放给用户后从命令中移除再传给低层。

关键在于高层策略的响应是**情境化**的：它不仅看到指令 $\ell_t$，还观测当前图像，因此能正确理解 "那个不是垃圾" 这样需要视觉理解的反馈——纯语言系统做不到这一点。

#### 3. 合成数据生成 (Synthetic Data Generation)

这是本文最独特的贡献之一。核心问题：机器人演示数据只有简单原子标注（如 "pick up lettuce"），但高层策略需要学会处理复杂、开放式的指令。

**逆向生成策略**：给定 (观测图像 $\mathbf{I}_t$, 技能标签 $\hat{\ell}_t$)，用大型 VLM $p_{\text{gen}}$ 反向生成"可能导致该技能的复杂用户指令 $\ell_t$"：

$$p_{\text{gen}}(\ell_t, u_t | \mathbf{I}_t^1, \dots, \mathbf{I}_t^n, \hat{\ell}_0, \dots, \hat{\ell}_t, \mathcal{P})$$

其中 $\mathcal{P}$ 是精心设计的提示模板。例如：
- 技能标签 "pick up the lettuce" → 生成指令 "帮我加点蔬菜好吗？"
- 技能标签 "put cup in bin" → 生成指令 "只清理纸杯，塑料杯留着"

**数据多样性保证**：
- **场景分类**：负面任务（"不要做X"）、情境纠正（"那个不是Y"）、特定约束（"我对花生过敏"）
- **回复分类**：简单确认、澄清询问、错误处理
- **上下文条件化**：生成时考虑当前时间步之前的技能序列 $\hat{\ell}_0, \dots, \hat{\ell}_{t-1}$，确保多步任务中指令的连贯性

#### 4. 模型架构

| 组件 | 基座模型 | 参数量 | 特殊设计 |
|------|---------|--------|---------|
| 高层策略 | PaliGemma-3B | 3B | 标准 VLM，输出语言 |
| 低层策略 (π0) | PaliGemma-3B | 3B | 额外 flow matching action expert，输出连续动作 |

两层策略使用相同的 VLM 基座模型，唯一区别是低层额外有一个 flow matching 模块输出连续动作。框架具有模块化特性：低层可替换为其他语言条件策略。

### 损失函数 / 训练策略

**高层策略训练**：
- 数据：$\mathcal{D}_{\text{syn}} \cup \mathcal{D}_{\text{labeled}}$（合成数据 + 人工标注数据）
- 损失：标准交叉熵损失（next-token prediction）
- 全参数微调 PaliGemma-3B

**低层策略训练**：
- 数据：$\mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{demo}}$（人工标注技能 + 遥操作演示）
- 损失：Flow-matching 目标函数（连续动作预测）

**训练超参数**：
- 优化器：AdamW（β₁=0.9, β₂=0.95, 无 weight decay）
- 梯度裁剪：最大范数 1.0
- EMA 权重：衰减因子 0.999
- 学习率：warmup 1000 步后恒定 1×10⁻⁵
- 批大小：512
- 高层策略训练仅需 **~2 小时 (8×H100)**，非常高效

## 实验关键数据

### 主实验

评估在三个任务域、三种机器人平台上进行（每种方法每个任务 20 次试验）：

| 任务 | 机器人 | 指标 | Hi Robot | GPT-4o 高层 | 扁平 VLA | 人类专家高层 |
|------|--------|------|----------|------------|----------|------------|
| 桌面清理 (Table Bussing) | 单臂 UR5e | IA / TP | **最优** | 低（物体误识别） | 低（忽略约束） | 最优（Oracle） |
| 三明治制作 (Sandwich Making) | 双臂 ARX | IA / TP | **最优** | 低（丢失上下文） | 低（默认行为） | 最优（Oracle） |
| 杂货购物 (Grocery Shopping) | 移动双臂 ARX | IA / TP | **最优** | 低（指令不一致） | 低（无反馈能力） | 最优（Oracle） |

**核心发现**：Hi Robot 在所有任务上的指令准确率 (IA) 平均比 GPT-4o 高层高出 **40%以上**，接近人类专家指导的水平。

### 消融实验

| 配置 | 关键指标 (IA / TP) | 说明 |
|------|-------------------|------|
| Hi Robot（完整） | **最优** | 合成数据 + 层次化 |
| Hi Robot 无合成数据 | 显著下降 | 忽略澄清（"这不是垃圾"），加入禁止食材 |
| 扁平 VLA + 合成数据 | 低于层次化 | 有合成数据但无高层推理，回归默认清扫所有物品 |
| 扁平 VLA（原始 π0） | 最低 | 无法处理复杂指令和实时反馈 |

### 关键发现

1. **合成数据至关重要**：没有合成数据时，高层策略虽能与图像观测对齐，但完全忽略用户约束（如饮食限制、选择性清理），合成数据带来的复合语言覆盖是泛化关键
2. **层次化结构优于扁平架构**：即使在相同数据条件下，层次化设计也优于扁平策略——在每个高层步骤重新检查指令，保证多步任务的连贯性
3. **GPT-4o 缺乏物理接地 (grounding)**：GPT-4o 频繁发出无意义命令（如 "pick up bermuda triangle"），将所有物体标为 "plate"，说明大模型虽强但缺乏对机器人能力的理解
4. **人类专家实验表明瓶颈在推理而非执行**：给定正确的原子命令，低层策略几乎完美执行

**推理延迟**：

| 组件 | RTX 4090 | H100 |
|------|----------|------|
| 低层（单步） | 73 ms（板载）/ 86 ms（WiFi） | — |
| 高层（prefill） | 47 ms | 17.3 ms |
| 高层（decode/步） | 13.2 ms | 5.7 ms |

系统在消费级硬件上实现了 ~10 Hz 控制，action chunking 后可达 50 Hz。

## 亮点与洞察

1. **逆向合成数据**是一个优雅且可扩展的方案——不需要收集复杂交互数据，从已有原子标注反向生成复杂指令，成本极低
2. **语言作为中间表征**使系统高度模块化和可解释——可以直接观察高层输出的命令来调试系统
3. **System 1/System 2 类比**提供了清晰的设计哲学——两层都用 VLM，但分工明确
4. 高层训练仅需 2 小时（8×H100），说明合成数据生成 + VLM 微调的效率优势
5. 框架天然支持**多模态人机交互**：语音输入 → Whisper ASR → 高层推理 → 语音回复 + 动作

## 局限与展望

1. **缺乏记忆机制**：高层策略无法处理需要长上下文推理的指令，没有跨时间步的记忆
2. **高层-低层解耦训练**：两层模型不了解彼此能力，高层可能生成低层无法执行的命令
3. **合成数据依赖 prompt engineering**：每个任务域需要精心设计的生成提示
4. **低层偏差**：训练偏向抓取近处物体，可能暂时忽略指令（如靠近奶酪时抓取，尽管用户说不要奶酪）
5. **错误累积与 OOD 恢复**：掉落物品后的恢复能力有限
6. **每个任务分别训练高层策略**：尚未实现统一的多任务高层模型

**未来方向**：将两层合并为单一模型，仅在推理时区分 System 1/System 2；异步多层级推理；让高层感知低层执行成功率的闭环机制。

## 相关工作与启发

- **π0 (Black et al., 2024)**：本文低层策略的基础，PaliGemma + flow matching 的 VLA
- **YAY Robot (Shi et al., 2024)**：先前的层次化方法，但仅限于单一指令和训练数据中见过的纠正类型
- **RACER (Dai et al., 2024)**：需要物理仿真器构建恢复行为，Hi Robot 仅用真实演示
- **SayCan (Brohan et al., 2023)**：LLM + 预定义技能，缺乏视觉理解和物理灵巧性
- **RT-2 (Brohan et al., 2023)**：VLA 模型但仅处理简单命令

本文的**逆向合成数据生成**思路可广泛迁移：任何有低层标注但缺乏高层复杂指令的场景都可采用。

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 创新性 | 8 | 合成数据逆向生成方案新颖，层次化 VLA 设计优雅 |
| 技术质量 | 8 | 系统设计合理，三平台验证充分 |
| 实验设计 | 8 | 多任务、多平台、对比和消融全面，但缺少量化表格 |
| 写作质量 | 9 | System 1/2 类比清晰，论述逻辑严密 |
| 实用价值 | 9 | 直接面向真实场景，消费级硬件可运行 |
| **总分** | **8.4** | 来自 Physical Intelligence + Stanford 的高质量工作 |

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](../../NeurIPS2025/robotics/coopera_continual_open_ended_human_robot_assistance.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](../../NeurIPS2025/robotics/cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](../../CVPR2025/robotics/momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2025/robotics/sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

<!-- RELATED:END -->
