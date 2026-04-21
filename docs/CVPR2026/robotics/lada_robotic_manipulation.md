---
title: >-
  [论文解读] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)
description: >-
  [CVPR 2026][机器人][动作解耦] 提出 LaDA 框架，用自然语言作为语义桥梁将连续 7-DoF 动作解耦为平移/旋转/夹爪三个可解释原语，通过软标签对比学习在共享嵌入空间中对齐跨任务动作表示，仅 0.6B 参数在 LIBERO 上达 93.6% 成功率，超越 1.3B~8.5B 参数的所有基线。
tags:
  - CVPR 2026
  - 机器人
  - 动作解耦
  - 语言语义桥梁
  - 软标签对比学习
  - VLA
  - 跨任务泛化
---

# Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)

**会议**: CVPR 2026  
**arXiv**: [2603.12967](https://arxiv.org/abs/2603.12967)  
**代码**: 无  
**领域**: 机器人操作  
**关键词**: 动作解耦, 语言语义桥梁, 软标签对比学习, VLA, 跨任务泛化

## 一句话总结

提出 LaDA 框架，用自然语言作为语义桥梁将连续 7-DoF 动作解耦为平移/旋转/夹爪三个可解释原语，通过软标签对比学习在共享嵌入空间中对齐跨任务动作表示，仅 0.6B 参数在 LIBERO 上达 93.6% 成功率，超越 1.3B~8.5B 参数的所有基线。

## 研究背景与动机

**领域现状**：视觉-语言-动作（VLA）模型近年推动了机器人操作进展，但高级语义理解与低级动作控制之间的异构性仍是根本挑战。

**现有痛点**：三类范式各有短板——(1) 端到端 VLA（如 OpenVLA、RT-2）将感知和控制紧耦合，动作不可解释且无法复用共享运动结构；(2) 隐式动作学习（如 LAPA、UniSkill）在紧凑隐空间编码动作，但隐空间由观测差异定义，缺乏显式语义标签，跨任务迁移受限；(3) 语言条件策略（如 CLIP-RT、PPL）引入语言但依赖粗粒度离散原语（"向前移动""关闭夹爪"），缺少平移幅度、旋转角度等精细运动参数。

**核心矛盾**："倒水"和"放瓶子"共享大量底层运动原语（到达、抓取、旋转），但现有模型无法利用这些共享结构，导致冗余学习和跨任务泛化差。根本原因在于缺少一个连接符号意图和连续执行的语义接地层。

**本文目标** 构建一个既有语义接地又可跨任务迁移的动作表示，实现细粒度运动语义的共享和对齐。

**切入角度**：语言天然提供了连接人类意图、视觉感知和机器人控制的通用接口——它具有组合性和语义规律性，可编码运动概念并在统一空间中比较、迁移和泛化。

**核心 idea**：用语言锚定的细粒度动作原语作为连续控制和高级语义之间的中间层，通过软标签对比学习实现跨任务动作的语义对齐。

## 方法详解

### 整体框架

输入视觉观测 $V_t$、语言指令 $L_t$ 和 7-DoF 动作 $\mathbf{a}_t$。整个框架分四步：(1) 动作解耦——将 $\mathbf{a}_t$ 投影为三个语言锚定原语（Translation、Rotation、Gripper）；(2) 构建软标签相似度矩阵 $S$，编码原语级语义亲缘度；(3) 双路径软标签对比学习（Action-Action 对齐 + Action-Primitive 对齐）训练统一嵌入空间；(4) 自适应权重平衡对比损失与模仿损失。预训练完成后附加轻量 MLP action head 微调做连续 7-DoF 动作预测。

### 关键设计

1. **语言锚定动作解耦（Language-Grounded Action Decomposition）**:

    - 功能：定义投影 $\Pi: \mathbf{a}_t \mapsto \mathbf{p}_t$，将连续 7-DoF 动作分解为三类可解释原语——平移 "Move [dist] meters along [dir]"、旋转 "Rotate [mag] degrees around [axis]"、夹爪 "Open/Close"
    - 核心思路：每个原语被离散化为语言对齐的符号类别（symbolic bins），将连续控制轨迹转化为可解释的语义类别。例如平移方向离散为"前/后/左/右/上/下"、旋转轴离散为 x/y/z 轴、夹爪为开/关二元状态
    - 设计动机：不同任务间可共享相同的运动原语（如多个任务都涉及"沿 z 轴旋转 90°"），显式语义标签让这些共享结构可被对比学习利用，而非像隐式动作学习那样被埋没在不可解释的 latent code 中

2. **语义引导软标签对比学习（Semantic-Guided Soft-Label Contrastive Learning）**:

    - 功能：在统一嵌入空间中按原语级语义亲缘度对齐多模态表示
    - 核心思路：构建软相似度矩阵 $S = \frac{w_t M_t + w_r M_r + w_g M_g}{w_t + w_r + w_g}$，其中 $M_t, M_r, M_g$ 是平移/旋转/夹爪维度的二元匹配矩阵。用 CLIP 视觉+文本编码器提取嵌入，FiLM 条件化后 MLP 投影得统一嵌入 $A_i$。双路径 soft-label InfoNCE：(i) Action-Action 按 $S_{ij}$ 加权拉近语义相似动作嵌入；(ii) Action-Primitive 将每个动作锚定到其原语文本描述编码 $P_j$。总损失 $\mathcal{L}_{CL} = \mathcal{L}_a + \lambda \mathcal{L}_m$
    - 设计动机：与传统二元正负对不同，软标签允许"部分相似"的动作对有梯度化的相似度（如两个动作平移相同但旋转不同，仍有部分匹配），捕捉更细粒度的运动对应关系

3. **自适应损失权重（Adaptive Loss Weighting）**:

    - 功能：动态平衡模仿损失 $\mathcal{L}_{IL}$（预测离散化原语类别）与对比损失 $\mathcal{L}_{CL}$
    - 核心思路：用移动平均归一化权重 $w_{IL} = \frac{\text{MA}(\mathcal{L}_{IL})}{\text{MA}(\mathcal{L}_{IL}) + \text{MA}(\mathcal{L}_{CL})}$，最终 $\mathcal{L}_{total} = w_{CL} \mathcal{L}_{CL} + w_{IL} \mathcal{L}_{IL}$
    - 设计动机：模仿损失提供粗粒度行为监督，对比损失提供细粒度语义对齐，二者收敛速率和粒度不同，固定权重易导致一方主导。灵感来自课程学习

### 损失函数 / 训练策略

- **预训练**：在 OXE 数据集（约 2250 万帧，22 种机器人）上用 $\mathcal{L}_{total}$ 训练，自动为每个连续动作生成结构化语言描述作为辅助监督
- **微调**：轻量 MLP action head + $\ell_1$ 轨迹回归损失
- **推理**：不需要显式原语标签，直接从 $(V_t, L_t)$ 输出连续动作

## 实验关键数据

### 主实验

| 模型 | 参数量 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | LIBERO-Avg |
|------|--------|---------------|---------------|-------------|-------------|------------|
| OpenVLA | 7.5B | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| FlowVLA | 8.5B | 93.2% | 95.0% | 91.6% | 72.6% | 88.1% |
| CLIP-RT | 1.3B | 95.2% | 99.2% | 94.2% | 83.8% | 93.1% |
| **LaDA** | **0.6B** | **95.2%** | **99.2%** | **93.6%** | **86.4%** | **93.6%** |

| 模型 | MimicGen 9 任务平均 | 代表任务 StackThree_D1 |
|------|-------------------|----------------------|
| OpenVLA | 38% | 20% |
| Phoenix | 58% | 20% |
| CLIP-RT* | 51% | 52% |
| **LaDA** | **67%** | **71%** |

### 消融实验

| 配置 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| w/o SCL（去软标签对比） | 79.2% | 82.8% | 76.6% | 63.4% | 75.5% |
| w/o AW（去自适应权重） | 93.6% | 94.4% | 87.2% | 74.4% | 87.4% |
| **LaDA（完整）** | **95.2%** | **99.2%** | **93.6%** | **86.4%** | **93.6%** |

### 关键发现

- 移除 SCL 后 LIBERO 平均骤降 18.1 个点（93.6→75.5%），其中 Long 从 86.4% 降至 63.4%，说明长序列最依赖跨任务语义共享
- 移除自适应权重后平均降 6.2 个点，Long 降 12 个点，证明优化平衡对长序列尤为关键
- 泛化测试：跨任务设定中 CLIP-RT* 成功率为 0%，LaDA 达 12.3%，证明语言锚定原语使未见指令的运动语义复用成为可能
- MimicGen 上 LaDA 在多任务训练中增益明显（CLIP-RT 几无增益），说明语义结构有效促进运动模式跨任务共享

## 亮点与洞察

- "语言作为语义桥梁"的理念直击 VLA 痛点——不做端到端黑盒映射，而是在动作层建立显式语义接口层，让动作可比较、可迁移。这比隐式动作学习和粗粒度语言条件都更优雅
- 软标签对比学习是方法论创新——传统对比学习用二元正负对，LaDA 用连续亲缘度矩阵做 soft InfoNCE，允许"平移相同但旋转不同"这种部分匹配有适当梯度信号。这个思路可迁移到目标检测/分割等需要细粒度语义对齐的领域
- 参数效率惊人：0.6B 参数超越 7B+ 模型，说明精心设计的结构化归纳偏置（动作解耦+对比对齐）可以大幅减少模型对规模的依赖

## 局限与展望

- 三类原语（平移/旋转/夹爪）覆盖标准工业机械臂的 7-DoF，但对灵巧手操作（如人形手指关节 20+ DoF）可能不够，需要更多运动分量的原语设计
- 语言模板手工设计（"Move X meters along Y"），自动化原语发现可能更灵活
- 实机实验仅 pick-and-place 单一任务，复杂真实场景验证不足
- 软相似度矩阵权重 $(w_t, w_r, w_g)$ 是超参数，不同任务域可能需重新调整

## 相关工作与启发

- **vs CLIP-RT**：同用语言条件控制，但 CLIP-RT 将动作建模为离散语言 token 分类，缺少运动参数的连续语义对齐；LaDA 用软标签对比学习做连续亲缘度匹配，参数量减半但性能持平略优
- **vs LAPA**：隐式动作学习将动作编码为不可解释的 latent code，跨任务迁移需隐式学习；LaDA 让动作空间变得可解释且可通过语义直接对齐迁移
- **vs Phoenix**：Phoenix 依赖运动级自反思纠正，MimicGen 上 58%；LaDA 无自纠正直接达 67%，说明更好的表示比更复杂的推理策略更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ 语言锚定动作解耦 + 软标签对比学习是全新方法论组合
- 实验充分度: ⭐⭐⭐⭐ LIBERO/MimicGen 双基准 + 消融 + 泛化 + 实机，但实机偏简单
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，三类范式对比有说服力
- 价值: ⭐⭐⭐⭐ 0.6B 超越 7B+ 有强实践意义，软标签对比学习可广泛迁移

<!-- RELATED:START -->

## 相关论文

- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2025/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures.md)

<!-- RELATED:END -->
