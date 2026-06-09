---
title: >-
  [论文解读] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions
description: >-
  [ECCV 2024][自动驾驶][trajectory prediction] 提出 VisionTrap，将环视相机图像和文本描述引入轨迹预测任务，通过 BEV 视觉语义编码器和文本驱动的去偏对比学习引导模型学习视觉语义线索（如行人姿态、转向灯等）…
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "trajectory prediction"
  - "visual semantics"
  - "textual guidance"
  - "对比学习"
  - "nuScenes-Text"
---

# VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions

**会议**: ECCV 2024  
**arXiv**: [2407.12345](https://arxiv.org/abs/2407.12345)  
**代码**: [https://moonseokha.github.io/VisionTrap](https://moonseokha.github.io/VisionTrap) (有项目页)  
**领域**: 自动驾驶 / 轨迹预测  
**关键词**: trajectory prediction, visual semantics, textual guidance, contrastive learning, nuScenes-Text

## 一句话总结
提出 VisionTrap，将环视相机图像和文本描述引入轨迹预测任务，通过 BEV 视觉语义编码器和文本驱动的去偏对比学习引导模型学习视觉语义线索（如行人姿态、转向灯等），在保持 53ms 实时推理的同时显著提升预测精度并发布 nuScenes-Text 数据集。

## 研究背景与动机
轨迹预测是自动驾驶的核心任务。现有方法主要依赖 agent 的历史轨迹和高精地图（HD Map）作为输入，但 HD Map 是静态的、无法反映临时变化（如施工区域），更无法提供行人注视方向、手势、车辆转向灯等对行为预测至关重要的视觉线索。

虽然少数工作尝试引入视觉数据，但它们要么只用前视相机，要么对整张图像进行全局处理而缺乏对关键区域的针对性提取，导致模型只关注显著特征、效果有限。此外，现有意图分类方法将行为划分为有限类别（如静止、变道、右转），存在标注歧义且限制了模型的表达能力。

**核心 idea**：利用环视相机的视觉语义信息增强轨迹预测，并用 VLM+LLM 生成的文本描述作为训练时的监督信号，引导模型从图像中学习更丰富的行为上下文，同时保持实时推理（推理时不需要文本输入）。

## 方法详解

### 整体框架
VisionTrap 由四个主要模块组成：
1. **Per-agent State Encoder**：编码每个 agent 的历史状态
2. **Visual Semantic Encoder**：将环视图像编码为 BEV 特征并与 agent 状态融合
3. **Text-driven Guidance Module**：训练时用文本描述监督视觉语义学习（推理时不使用）
4. **Trajectory Decoder**：预测所有 agent 的未来轨迹

输入：agent 历史轨迹 + 类型 + 环视相机图像 + HD map；输出：所有 agent 的多模态未来轨迹分布。

### 关键设计

1. **Per-agent State Encoder（状态编码器）**:

    - 功能：编码每个 agent 的时空状态和交互关系
    - 核心思路：用相对位移（而非绝对位置）表示历史轨迹，结合 agent 类型和时间位置编码
    - 关键公式：$s_i^t = f_{\text{geometric}}(p_i^t - p_i^{t-1}) + f_{\text{type}}(a_i) + f_{\text{PE}}(e^t)$
    - 三级编码：(1) MLP 编码几何位移，(2) Temporal Transformer 编码时序信息，(3) Cross-attention 编码 agent 间交互关系
    - 交互编码在 ego-centric 坐标系下一次完成，避免逐 agent 重复计算

2. **Visual Semantic Encoder（视觉语义编码器）**:

    - 功能：将环视图像转为 BEV 特征，并与 agent 状态嵌入融合
    - 核心思路：用 BEVDepth 架构将多视图图像编码为 BEV 图像特征 $B_I \in \mathbb{R}^{h \times w \times d_{\text{bev}}}$，与光栅化地图特征 $B_{\text{map}}$ 拼接得到复合 BEV 场景特征 $B = [B_I; B_{\text{map}}]$
    - 场景-agent 交互：使用 **Deformable Cross-Attention** 从 BEV 特征中提取与每个 agent 相关的区域信息。引入 Recurrent Trajectory Prediction 模块，用辅助轨迹预测器的预测位置作为 deformable attention 的参考点
    - 关键公式：$z_i^{\text{scene}} = z_i^{\text{interact}} + \sum_{h=1}^{H} W_h \left[\sum_{o=1}^{O} \alpha_{hio} W'_h \mathbf{B}_{(u_i^{\text{aux}} + \triangle u_{hio}^{\text{aux}})}\right]$
    - 设计动机：相比 ConvNet 全局感知，deformable attention 能选择性聚焦 agent 需要关注的区域，且注意力点数远少于场景元素数，降低计算量

3. **Text-driven Guidance Module（文本引导模块）**:

    - 功能：训练时用文本描述作为监督，引导 agent 状态嵌入学习细粒度的视觉语义
    - 核心思路：通过跨模态对比学习将 agent 状态嵌入 $z_i^{\text{scene}}$ 与其文本描述嵌入 $\mathcal{T}_i$ 对齐
    - 去偏设计：驾驶场景中多个 agent 可能有相似行为，简单对比学习会产生 false negatives。解决方案：(1) 用 BERT 提取 sentence-level 嵌入 $\mathcal{T}_i$，(2) 过滤余弦相似度 > $\theta_{th}=0.8$ 的样本（潜在 false negatives），(3) 取 top-k 最不相似的作为负样本保证数量稳定
    - 关键公式：$\mathcal{L}_{\text{cl}} = -\log\frac{e^{\text{sim}_{\text{cos}}(z_i^{\text{scene}}, \mathcal{T}_i)/\tau}}{\sum_{j=1}^{k} e^{\text{sim}_{\text{cos}}(z_i^{\text{scene}}, \mathcal{T}_j)/\tau}}$
    - 使用非对称对比损失（仅从 agent 到文本方向）
    - 推理时完全不需要文本输入，零额外延迟

4. **Transformation Module + Trajectory Decoder**:

    - 功能：标准化 agent 朝向后进行多模态轨迹预测
    - 核心思路：ego-centric 方法的缺点是相似行为的 agent 特征未标准化；在预测前用旋转矩阵 $\mathcal{R}$ 标准化每个 agent 的朝向
    - Trajectory Decoder 用 GMM 建模未来轨迹分布：$p(u) = \sum_{m=1}^{M}\rho_m \prod_{t=1}^{T_f} \mathcal{N}(u_t - \mu_m^t, \Sigma_m^t)$

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{traj}} + \lambda_{\text{traj}}^{\text{aux}} \mathcal{L}_{\text{traj}}^{\text{aux}} + \lambda_{\text{cl}} \mathcal{L}_{\text{cl}}$
- $\mathcal{L}_{\text{traj}}$：GMM 负对数似然损失
- $\mathcal{L}_{\text{traj}}^{\text{aux}}$：辅助轨迹预测损失（用于 Recurrent Trajectory Prediction 模块）
- $\mathcal{L}_{\text{cl}}$：去偏非对称对比学习损失
- **nuScenes-Text 数据集生成**：(1) 用 DRAMA 数据集微调 BLIP-2，(2) 对 nuScenes 中每个 agent 裁剪图像并生成描述，(3) 用 GPT 结合 GT 类型和规则提取的行为特征进行文本精炼。共生成 1,216,206 条文本描述，平均 13 词/条，人工评估 94.8% 对齐准确率

## 实验关键数据

### 主实验
nuScenes 预测数据集上的对比（单/多 agent 预测）：

| 方法 | 预测模式 | 推理时间(ms) | ADE10↓ | MR10↓ | FDE1↓ |
|------|---------|-------------|--------|-------|-------|
| PGP | 单 agent | 215 | 1.00 | 0.37 | 7.17 |
| LAformer | 单 agent | 115 | 0.93 | 0.33 | - |
| Trajectron++ | 多 agent | 38 | 1.51 | 0.57 | 9.52 |
| VisionTrap baseline | 多 agent | 13 | 1.48 | 0.56 | 10.75 |
| + Map Encoder | 多 agent | 21 | 1.40 | 0.53 | 10.41 |
| + Visual Semantic Encoder | 多 agent | 53 | 1.23 | 0.36 | 9.32 |
| **+ Text-driven Guidance (Full)** | 多 agent | **53** | **1.17** | **0.32** | **8.72** |

完整模型相比 baseline 提升 27.56%，在多 agent 模式下达到与单 agent SOTA 方法可比的精度，同时推理仅 53ms（实时）。

### 消融实验

**nuScenes 全数据集（所有 agent）**：

| 方法 | ADE10↓ | FDE10↓ | MR10↓ |
|------|--------|--------|-------|
| VisionTrap baseline | 0.425 | 0.641 | 0.081 |
| + Map Encoder | 0.407 | 0.601 | 0.075 |
| + Visual Semantic Encoder | 0.382 | 0.551 | 0.056 |
| + Text-driven Guidance | **0.368** | **0.535** | **0.051** |

**Text-driven Guidance Module 内部消融**：

| 配置 | ADE6↓ | FDE6↓ | MR6↓ |
|------|-------|-------|------|
| A. CLIP 对称对比损失 | 0.51 | 0.79 | 0.10 |
| B. 对称损失变体 | 0.50 | 0.76 | 0.10 |
| C. 无负样本精炼 | 0.49 | 0.72 | 0.09 |
| D. 无 top-k 约束 | 0.46 | 0.67 | 0.08 |
| **E. 完整模型** | **0.44** | **0.66** | **0.07** |

### 关键发现
- Visual Semantic Encoder 是最大贡献模块：ADE10 从 1.40 降至 1.23（改善 12.1%），MR10 从 0.53 降至 0.36
- Text-driven Guidance 在此基础上进一步提升 4.9%，证明文本描述有效引导模型关注细粒度视觉语义
- 去偏对比学习策略很关键：去除负样本精炼或 top-k 约束都会降低性能
- 定性分析表明模型能利用：红绿灯状态预测行人是否过马路、行人注视方向、车辆转向灯等视觉线索
- UMAP 可视化显示：加入视觉+文本语义后，相似行为 agent 的嵌入聚类更紧密

## 亮点与洞察
- **文本引导训练、推理免费**：文本描述仅在训练时用于对比学习监督，推理时零额外开销。这种"训练时丰富，推理时精简"的设计范式很实用
- **去偏对比学习**：通过文本语义相似度过滤 false negatives，解决了驾驶场景中多 agent 行为相似导致的负样本歧义问题
- **nuScenes-Text 数据集**：VLM 微调 + LLM 精炼的半自动标注 pipeline，94.8% 准确率，可迁移到其他数据集的文本标注
- **Scene-centric 实时推理**：采用 ego-centric 方案一次处理所有 agent，推理 53ms 远快于 agent-centric 方法

## 局限与展望
- 文本数据集生成依赖 VLM+LLM 的级联 pipeline，标注质量受两个模型性能影响
- 只在 nuScenes 上验证，未测试 Waymo 等其他数据集的泛化性
- 当前不支持长时序预测（仅 2-6 秒），更长时间的预测可能更需要视觉语义
- Transformation Module 的旋转标准化仅考虑朝向，未处理速度等其他状态量的分布差异

## 相关工作与启发
- **vs TPNet/IntentNet**：这些方法使用前视图像或 agent 区域图像，VisionTrap 使用环视 BEV 特征，信息更全面
- **vs Trajectron++**：同为 scene-centric 多 agent 预测，但 VisionTrap 引入视觉+文本语义，ADE10 从 1.51 降至 1.17
- **vs CLIP-based 方法**：VisionTrap 不使用通用 CLIP，而是设计了专门针对轨迹预测的对比学习框架，考虑了 false negative 去偏

## 评分
- 新颖性: ⭐⭐⭐⭐ 将文本描述作为训练时监督引导视觉语义学习的思路新颖，去偏策略设计合理
- 实验充分度: ⭐⭐⭐⭐ 多层次消融、定性可视化、UMAP 分析、nuScenes-Text 数据集人工评估
- 写作质量: ⭐⭐⭐⭐ 图文配合好，定性分析直观展示了视觉语义的贡献
- 价值: ⭐⭐⭐⭐ 证明了视觉语义在轨迹预测中的价值，nuScenes-Text 数据集有公开价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[ECCV 2024\] Adaptive Human Trajectory Prediction via Latent Corridors](adaptive_human_trajectory_prediction_via_latent_corridors.md)
- [\[ECCV 2024\] Progressive Pretext Task Learning for Human Trajectory Prediction](progressive_pretext_task_learning_for_human_trajectory_prediction.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)

</div>

<!-- RELATED:END -->
