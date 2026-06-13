---
title: >-
  [论文解读] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
description: >-
  [NeurIPS 2025][模型压缩][scene flow] 提出 DeltaFlow (ΔFlow)，通过体素帧间差分（Δ scheme）提取运动线索，实现特征尺寸不随帧数增长的多帧场景流估计，在 Argoverse 2/Waymo/nuScenes 上达到 SOTA 且比次优多帧方法快 2 倍。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "scene flow"
  - "multi-frame"
  - "delta scheme"
  - "自动驾驶"
  - "computational efficiency"
---

# DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2508.17054](https://arxiv.org/abs/2508.17054)  
**代码**: [https://github.com/Kin-Zhang/DeltaFlow](https://github.com/Kin-Zhang/DeltaFlow)  
**领域**: 模型压缩  
**关键词**: scene flow, multi-frame, delta scheme, autonomous driving, computational efficiency

## 一句话总结
提出 DeltaFlow (ΔFlow)，通过体素帧间差分（Δ scheme）提取运动线索，实现特征尺寸不随帧数增长的多帧场景流估计，在 Argoverse 2/Waymo/nuScenes 上达到 SOTA 且比次优多帧方法快 2 倍。

## 研究背景与动机

**领域现状**：场景流估计预测连续点云帧之间每个点的 3D 运动。主流方法将点云体素化后，通过拼接多帧特征（沿通道维度）或构建 4D 时空体素来融合时序信息。

**现有痛点**：(1) 拼接法随帧数增加导致通道维度线性膨胀，增大内存和计算开销；(2) 4D 方法引入额外时间维度同样导致输入线性增长；(3) 类别不平衡（行人/骑行者远少于车辆）和实例级运动不一致也制约性能。

**核心矛盾**：想利用更多历史帧提升精度，但现有方案的计算成本随帧数线性或超线性增长，无法扩展到长时序。特征拼接/堆叠都在"积累"时序信息，但场景流本质上关注的是"变化"。

**本文目标**：(1) 高效利用多帧信息而不增加计算开销；(2) 解决类别不平衡和实例运动一致性问题。

**切入角度**：场景流本质是估计"什么在变化"——直接计算帧间体素特征的差值（Δ scheme）天然聚焦于变化部分，且差值特征尺寸恒定、不随帧数增长。

**核心 idea**：用加权帧间差分代替特征拼接/4D 堆叠来编码时序运动线索，配合类别平衡损失和实例一致性损失提升动态物体的估计质量。

## 方法详解

### 整体框架
输入 $N+1$ 帧点云 $\{\mathcal{P}_{t-N}, ..., \mathcal{P}_{t-1}, \mathcal{P}_t\}$（已做自车运动补偿）。经 PointPillars 提取点特征后体素化为稀疏体素特征 $\mathscr{D}$，通过 Δ scheme 计算差分特征 $\mathscr{D}_{\text{delta}}$，送入 MinkowskiNet 3D backbone + DeFlow decoder 估计残差场景流 $\Delta\hat{\mathcal{F}}$。最终场景流 = 自车运动 + 残差流。

### 关键设计

1. **Temporal Δ Scheme**:

    - 功能：从多帧体素特征中提取运动信号，输出特征尺寸恒定不随帧数变化。
    - 核心公式：$\mathscr{D}_{\text{delta}} = \sum_{n=1}^{N} \lambda^{n-1}(\mathscr{D}_t - \mathscr{D}_{t-n}) / N$
    - 其中 $\lambda \in (0,1]$ 是时间衰减因子，使近帧权重更高。
    - 设计动机：(1) 帧间差分聚焦于场景中"变化的部分"（运动物体），天然过滤静态背景，与场景流估计的核心目标一致；(2) 差分结果始终维持 $V \times C$ 尺寸，backbone 的参数和计算量完全不受帧数影响——从 2 帧到 15 帧推理时间几乎恒定；(3) 加权求和积累移动物体的"运动轨迹"，类似于人类从运动模糊中感知运动方向。

2. **稀疏体素表示**:

    - 功能：将点特征聚合到非空体素中，只处理有数据的体素。
    - 公式：$\mathscr{D}[v_i] = \frac{\sum_{p \in \mathcal{P}^{v_i}} \mathbf{f}_p}{|\mathcal{P}^{v_i}|}$
    - 设计动机：保留完整 3D 结构（不像 BEV 压缩掉高度信息），同时通过稀疏存储大幅降低内存消耗。

3. **Category-Balanced Loss**:

    - 功能：按物体类别和速度分桶，对不同类别赋予不同权重，解决行人/骑行者等小类占比极低的问题。
    - 公式：$\mathcal{L}_C = \sum_{c \in \mathcal{C}} w_c \sum_{b \in \mathcal{B}} \gamma_b \frac{1}{|\mathcal{P}_{c,b}|} \sum_{p \in \mathcal{P}_{c,b}} \|\Delta\hat{\mathcal{F}}(p) - \Delta\mathcal{F}_{\text{gt}}(p)\|_2$
    - 设计动机：之前的 DeFlow loss 只区分静态/动态，不区分物体类别，导致安全关键的小目标（行人）被大量车辆点主导。

4. **Instance Consistency Loss**:

    - 功能：约束同一刚体实例内所有点共享一致的场景流。
    - 公式：$\mathcal{L}_I = \frac{1}{|\mathcal{I}'|} \sum_{I \in \mathcal{I}'} \omega_{c_I} \hat{e}_I \exp(\hat{e}_I)$，其中 $\hat{e}_I$ 是实例内平均误差，$\exp(\hat{e}_I)$ 进一步惩罚高误差实例。
    - 设计动机：场景流标注是逐点的，模型可能给同一车辆的不同点预测不同流向。实例级一致性约束确保刚体运动的物理合理性。

### 损失函数
总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{deflow}} + \mathcal{L}_C + \mathcal{L}_I$

## 实验关键数据

### 主实验（Argoverse 2 测试集，公开榜单）

| 方法 | 帧数 | 运行时间/序列 | Mean Bucket-Norm ↓ | Mean EPE (cm) ↓ |
|------|------|-------------|-------------------|----------------|
| **ΔFlow (Ours)** | **5** | **8s** | **0.113** | **2.11** |
| Flow4D | 5 | 15s | 0.145 | 2.24 |
| EulerFlow | all | 24h | 0.130 | 4.23 |
| SSF | 2 | 5.2s | 0.181 | 2.73 |
| DeFlow | 2 | 7.2s | 0.276 | 3.43 |

ΔFlow 比 Flow4D 降低 22% 的 Bucket-Normalized EPE，速度快 2 倍。

| 数据集 | 方法 | Mean EPE ↓ |
|--------|------|-----------|
| Waymo | ΔFlow | **1.64** |
| Waymo | Flow4D | 2.03 |
| nuScenes | ΔFlow | 最优（降 39%） |
| nuScenes | Flow4D | 基线 |

### 消融实验（运行时间 vs 帧数）

| 帧数 | 2 | 5 | 10 | 15 |
|------|---|---|----|----|
| ΔFlow 时间 | 7.6s | 8s | ~8.5s | ~9s |
| Flow4D 时间 | 12.8s | 15s | - | - |

ΔFlow 从 2 帧到 15 帧运行时间几乎不变（核心优势），而 Flow4D 随帧数线性增长。

### 关键发现
- Δ scheme 在 2 帧设置下就追平了 Flow4D 的 5 帧性能，说明差分表示比拼接/堆叠更高效地利用时序信息
- 类别平衡损失对行人/VRU 类别提升显著（PED bucket-norm 从 0.216 降到 0.149）
- 跨数据集泛化能力强：在 Argoverse 2 上训练后直接迁移到 nuScenes 仍有竞争力
- 实例一致性损失使同一物体内的流预测更平滑，减少物理不合理的预测

## 亮点与洞察
- **Δ scheme 极其简洁**：只是帧间差值后加权平均，但效果远超复杂的 4D 卷积方案。体现了"选择正确的表示比堆计算量更重要"的哲学。
- **恒定计算成本**是杀手特性：backbone 参数和 FLOPs 完全不受帧数影响，使得利用任意长历史帧成为可能，这在实时自动驾驶中极有价值。
- 运动模糊类比很直觉：差分叠加类似于长曝光产生运动模糊，让模型从"模糊痕迹"中推断运动方向和速度。

## 局限与展望
- Δ scheme 假设体素对齐——自车运动补偿的误差可能导致差分特征中引入噪声
- 类别权重 $w_c$ 和速度系数 $\gamma_b$ 是手动设定的，未探索自适应方案
- 只用了 MinkowskiNet 作为 3D backbone，未验证与其他稀疏卷积架构（如 SpConv、TorchSparse）的兼容性
- 对非刚体运动（如行人肢体摆动）的处理能力未详细分析

## 相关工作与启发
- **vs Flow4D**: Flow4D 构建 4D 时空体素用 3D 空间+1D 时间卷积处理，计算随帧数线性增长；ΔFlow 用差分将时序压缩为恒定尺寸，效率优势明显
- **vs DeFlow**: DeFlow 只用 2 帧且拼接通道，ΔFlow 在此基础上引入多帧差分和更丰富的损失函数
- **vs EulerFlow**: EulerFlow 是离线方法（24h/序列），精度高但完全无法实时；ΔFlow 兼顾精度和效率

## 评分
- 新颖性: ⭐⭐⭐⭐ Δ scheme 想法简单但非常有效，类别平衡和实例一致性损失也有实际价值
- 实验充分度: ⭐⭐⭐⭐⭐ 三个大规模驾驶数据集、公开榜单验证、详细的效率分析和跨域泛化实验
- 写作质量: ⭐⭐⭐⭐ 图示清晰（特别是三种多帧策略的对比图），方法描述简洁易懂
- 价值: ⭐⭐⭐⭐⭐ 解决了多帧场景流估计的核心可扩展性问题，开源代码和预训练权重，对自动驾驶社区价值很大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](../../ICCV2025/model_compression/multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [\[CVPR 2025\] AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](../../CVPR2025/model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)

</div>

<!-- RELATED:END -->
