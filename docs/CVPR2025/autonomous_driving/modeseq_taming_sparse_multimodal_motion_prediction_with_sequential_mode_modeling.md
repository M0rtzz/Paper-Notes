---
title: >-
  [论文解读] ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling
description: >-
  [CVPR 2025][自动驾驶][多模态运动预测] 提出 ModeSeq——一种将轨迹模式建模为序列的全新范式，通过逐步解码多模态轨迹（而非一次性并行解码）来显式捕捉模式间关联，并配合 Early-Match-Take-All (EMTA) 训练策略，在不依赖密集模式预测或启发式后处理的前提下，显著提升了稀疏多模态运动预测的轨迹多样性和置信度校准。
tags:
  - CVPR 2025
  - 自动驾驶
  - 多模态运动预测
  - 序列模式建模
  - 稀疏预测
  - Winner-Take-All
  - 轨迹多样性
---

# ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling

**会议**: CVPR 2025  
**arXiv**: [2411.11911](https://arxiv.org/abs/2411.11911)  
**代码**: 无  
**领域**: 自动驾驶 / 运动预测  
**关键词**: 多模态运动预测, 序列模式建模, 稀疏预测, Winner-Take-All, 轨迹多样性

## 一句话总结
提出 ModeSeq——一种将轨迹模式建模为序列的全新范式，通过逐步解码多模态轨迹（而非一次性并行解码）来显式捕捉模式间关联，并配合 Early-Match-Take-All (EMTA) 训练策略，在不依赖密集模式预测或启发式后处理的前提下，显著提升了稀疏多模态运动预测的轨迹多样性和置信度校准。

## 研究背景与动机

**领域现状**：自动驾驶中的运动预测需要为每个交通参与者预测多条可能的未来轨迹及其置信度。由于真实世界只能观测到一条实际轨迹（缺乏多模态 ground truth），主流方法采用 Winner-Take-All (WTA) 策略训练——只监督与 GT 最接近的预测模式。

**现有痛点**：WTA 训练容易导致模式坍缩（mode collapse），多条预测轨迹高度重叠，置信度评分也难以区分。为缓解这一问题，一些方法生成大量候选轨迹（密集模式预测），再通过 NMS 等后处理选出代表性模式。但后处理的超参数难以调优，也无法适配不同场景，且密集生成+筛选会损害轨迹精度。

**核心矛盾**：现有方法采用"并行模式建模"——所有轨迹一次性解码，模式之间完全独立。这导致模型无法利用已解码模式的信息来推断下一个应该是什么模式，轨迹多样性只能依赖参数差异或 anchor 差异，本质上不可控。

**本文目标** 在不依赖密集预测和后处理的情况下，直接端到端地生成少量但多样、高质量、置信度校准良好的代表性轨迹。

**切入角度**：既然并行解码忽略了模式间关系，那就将模式建模为一个序列——每一步解码一条轨迹，都以之前已解码的模式为条件。这样模型被迫"看看已经预测了什么"，从而避免重复并提升覆盖率。

**核心 idea**：用序列化方式逐步解码轨迹模式，将无序的多模态预测问题转化为有序的条件生成问题。

## 方法详解

### 整体框架
ModeSeq 采用编码器-解码器架构。编码器（基于 QCNet）处理地图和历史轨迹生成场景嵌入。解码器包含多层 ModeSeq Layer，每层以循环方式逐步输出 $K$ 个模式嵌入。每个模式嵌入通过 MLP 头预测一条轨迹和对应置信度。多层之间通过模式重排（Mode Rearrangement）协调顺序，整体用 EMTA 策略训练。

### 关键设计

1. **单层 ModeSeq（Memory Transformer + Context Transformer）**:

    - 功能：以循环方式逐步解码 $K$ 个模式嵌入，每一步都以前面已解码的模式为条件
    - 核心思路：在第 $k$ 步解码时，先用 Memory Transformer 将当前模式嵌入 $\mathbf{m}_k^{(\ell-1)}$ 作为 query，对已解码的模式序列 $\Omega_{k-1}^{(\ell)}$ 做交叉注意力，获得"关于前面模式的感知"。然后用 Context Transformer 进一步将场景嵌入 $\Psi$（地图、历史轨迹、邻近agent）融入，生成最终模式嵌入 $\mathbf{m}_k^{(\ell)}$，加入序列供后续步骤使用。Context Transformer 分解为 mode-time、mode-map、mode-agent 三个子注意力模块以降低复杂度。
    - 设计动机：与 DETR 类解码器在模式间只做自注意力（弱关联）不同，序列化解码形成因果条件依赖链，天然强化了模式间的排斥性。而且参数跨步共享，推理时可以动态调整模式数量（增减解码步数即可），这是并行方法做不到的。

2. **多层迭代精化 + 模式重排 (Mode Rearrangement)**:

    - 功能：通过多轮解码逐步优化轨迹质量和模式顺序
    - 核心思路：堆叠 6 层 ModeSeq Layer。每层解码完 $K$ 个模式后，按置信度降序重排模式嵌入序列，再输入下一层。这样高置信度的模式在早期被解码，低置信度的模式在后期被解码，形成单调递减的置信度顺序。每层的输出都附加训练损失。
    - 设计动机：单层解码时前几步可能产生质量差的模式，扰乱后续模式的学习。重排确保高质量模式排在前面接受优先精化，与 EMTA 训练策略形成协同。

3. **Early-Match-Take-All (EMTA) 训练策略**:

    - 功能：取代 WTA 策略，鼓励模型尽早解码出匹配 GT 的轨迹，从而释放后续位置给其他可能的模式
    - 核心思路：在 $K$ 条预测中找到所有匹配 GT 的轨迹（匹配标准基于位移阈值），将**最早出现的匹配**（而非最佳匹配）标记为正样本，其余（包括后续匹配项）标记为负样本。如果没有任何匹配，则退回 WTA 策略选误差最小的。轨迹回归用 Laplace NLL 损失，置信度用 Binary Focal Loss。
    - 设计动机：传统 WTA 只优化最佳匹配，导致同一区域出现多个高置信度模式。EMTA 通过将早期匹配标为正、后续重复匹配标为负，迫使模型将重复模式腾出来覆盖其他未被覆盖的未来，从而提升覆盖率和置信度区分度。

### 损失函数 / 训练策略
每层都施加 EMTA 损失（轨迹回归 Laplace NLL + 置信度 Binary Focal Loss）。训练时在 WOMD 上用 AdamW 优化 30 个 epoch，初始学习率 $5 \times 10^{-4}$，余弦退火。

## 实验关键数据

### 主实验

| 数据集 | 方法 | Soft mAP6↑ | mAP6↑ | MR6↓ | minADE6↓ | minFDE6↓ |
|---|---|---|---|---|---|---|
| WOMD Val | QCNet | 0.4508 | 0.4452 | 0.1254 | 0.5122 | 1.0225 |
| WOMD Val | **ModeSeq** | **0.4562** | **0.4507** | **0.1206** | 0.5237 | 1.0681 |
| Argoverse2 | QCNet | - | - | 0.16 | 0.65 | 1.29 |
| Argoverse2 | **ModeSeq** | - | - | **0.14** | **0.63** | **1.26** |

ModeSeq 在模式覆盖 (MR) 和置信度评分 (mAP) 上全面优于 QCNet，轨迹精度仅有微小退步。在 2024 Waymo Open Motion Prediction Challenge 中，ModeSeq 在无 LiDAR 方案中排名第一。

### 消融实验

| 解码器 | 训练策略 | Soft mAP6↑ | MR6↓ | minADE6↓ |
|---|---|---|---|---|
| DETR w/ Refinement | WTA | 0.4096 | 0.1536 | 0.5660 |
| ModeSeq | WTA | 0.4138 | 0.1502 | 0.5563 |
| ModeSeq | EMTA | **0.4231** | **0.1457** | 0.5700 |

### 关键发现
- 仅用序列模式建模（不改训练策略），mAP 已提升 0.4%+，MR 降低 0.3%+，说明模式间条件依赖本身有价值
- EMTA 在序列模式建模上进一步大幅提升 mAP（+0.9%）和 MR（-0.45%），轨迹精度仅微降 0.014m (minADE)
- 模式重排对 EMTA 至关重要：有重排时 Soft mAP 0.4231 vs 无重排 0.4112
- ModeSeq 天然具备模式外推能力：训练时解码 6 个模式，推理时解码 24 个模式仍能产生合理、多样的轨迹
- 3-mode ModeSeq 的 mAP6 甚至超过 6-mode QCNet，说明序列建模能用更少模式覆盖更多行为

## 亮点与洞察
- **"并行到序列"的范式转变**：将无序的多模态预测问题转化为有序的序列生成问题，是一个非常优雅的思路。类似于 NLP 中从 set prediction 到 autoregressive generation 的演进，为运动预测带来了新的设计维度。
- **EMTA 的设计哲学**：不追求"最佳匹配"而是"最早匹配"，这个看似反直觉的设计巧妙地利用了序列的时序结构来强制多样性。
- **模式外推能力**：参数共享使得训练时的模式数量和推理时可以不同，这种"弹性预测"能力在不确定性高的场景中非常实用，其他固定 anchor 的方法做不到。

## 局限与展望
- 推理延迟约为 QCNet 的两倍（6-mode: 128ms vs 69ms），因为序列解码无法完全并行化
- minADE/minFDE 略有退步（约 0.01m/0.05m），说明为了多样性牺牲了少量精度
- 序列中模式的顺序对结果有影响，但最优排序策略（按置信度降序）是否在所有场景都最优未充分验证
- 未与 diffusion-based 的运动预测方法（如 MotionDiffuser）做对比

## 相关工作与启发
- **vs QCNet**: 同一编码器下，ModeSeq 通过序列化解码在覆盖率和置信度上超越，证明问题在解码端而非编码端
- **vs MTR/MTR++**: MTR 系列依赖密集 anchor 和后处理来实现多样性，ModeSeq 无需任何 anchor 或后处理即端到端完成
- **vs FJMP/M2I**: 这些方法在 agent 维度做序列化（先预测影响者再预测受影响者），本文首次在 mode 维度做序列化，提供了一个全新的分解角度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 序列模式建模是一个全新范式，EMTA 和模式重排的设计紧密配合
- 实验充分度: ⭐⭐⭐⭐⭐ WOMD 和 Argoverse2 双基准验证，消融非常彻底
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，动机推导流畅，图示精心设计
- 价值: ⭐⭐⭐⭐⭐ 对运动预测领域影响深远，兼具理论优雅和实用价值

<!-- RELATED:START -->

## 相关论文

- [Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](../../ICCV2025/autonomous_driving/emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [Fully Sparse 3D Occupancy Prediction](../../ECCV2024/autonomous_driving/fully_sparse_3d_occupancy_prediction.md)
- [FoSS: Modeling Long-Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier–State Space Integration](../../CVPR2026/autonomous_driving/foss_modeling_long_range_dependencies_and_multimodal_uncertainty_in_trajectory_p.md)

<!-- RELATED:END -->
