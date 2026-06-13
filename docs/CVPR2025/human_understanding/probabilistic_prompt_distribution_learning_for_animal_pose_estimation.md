---
title: >-
  [论文解读] Probabilistic Prompt Distribution Learning for Animal Pose Estimation
description: >-
  [CVPR 2025][人体理解][动物姿态估计] 提出 PPAP（Probabilistic Prompt for Animal Pose），一种基于概率提示分布学习的多物种动物姿态估计方法，通过为每个关键点构建多个可学习属性提示并建模为高斯分布，结合多样性损失和跨模态融合策略，在有监督和零样本设置下均达到 SOTA。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "动物姿态估计"
  - "概率提示学习"
  - "多模态融合"
  - "跨物种泛化"
  - "CLIP"
---

# Probabilistic Prompt Distribution Learning for Animal Pose Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.16120](https://arxiv.org/abs/2503.16120)  
**代码**: [GitHub](https://github.com/Raojiyong/PPAP)  
**领域**: 人体/动物理解  
**关键词**: 动物姿态估计, 概率提示学习, 多模态融合, 跨物种泛化, CLIP

## 一句话总结

提出 PPAP（Probabilistic Prompt for Animal Pose），一种基于概率提示分布学习的多物种动物姿态估计方法，通过为每个关键点构建多个可学习属性提示并建模为高斯分布，结合多样性损失和跨模态融合策略，在有监督和零样本设置下均达到 SOTA。

## 研究背景与动机

- 多物种动物姿态估计（APE）面临物种间巨大的视觉多样性和不确定性挑战
- 直接将人体姿态估计方法应用于动物存在显著的域偏移
- 类别无关姿态估计（CAPE）方法需要额外的支持集和类别先验知识，实用性受限
- 纯视觉APE方法在跨物种场景下仅依赖视觉线索难以处理长尾分布
- 现有多模态APE方法（如CLAMP、X-Pose）使用固定文本模板（确定性提示），文本描述不够丰富
- 单一文本描述无法涵盖关键点的所有细微特征（颜色、位置、形状等）
- 野外场景的复杂性和多物种特性引入了不确定的统计偏移
- 概率提示学习相比确定性提示更适应跨物种挑战，但现有方法在输入空间建模分布效果有限

## 方法详解

### 整体框架

PPAP 基于 CLAMP 框架构建，保留 CLIP 的文本编码器和图像编码器。为每个关键点创建 $N_p$ 个可学习属性提示模板，经文本编码器编码后，通过 text decoder 获取均值、visual-text decoder 获取方差，建模为独立高斯分布。采样后的概率提示表示通过三种跨模态融合策略（启发式/集成/注意力）与视觉特征在空间层面对齐，生成关键点热力图。

### 关键设计

**设计一：多样化提示构建 + 多样性损失**
- **功能**：为每个关键点提供多视角、多属性的丰富文本描述
- **核心思路**：为第 $i$ 个关键点创建 $N_p$ 个属性模板 $p_i^t = \{a_1^t, \ldots, a_L^t | k_i\}$，其中 $\{a_l^t\}$ 为可学习属性token。采用广义关键点放置（GKP）策略，允许关键点名称在模板中随机位置放置。设计多样性损失 $\mathcal{L}_{div} = \frac{1}{K}\sum_{i=1}^{K}\|\tilde{P}_i\tilde{P}_i^T - \mathbb{I}\|_2^2$ 保持属性表示正交
- **设计动机**：单一提示无法捕获关键点的全部语义信息，多个不同属性提示从颜色、空间位置等多角度补充；多样性损失防止学到的属性出现退化为相同表示

**设计二：概率提示分布建模**
- **功能**：通过高斯分布建模提示的不确定性，增强对未见类别的泛化
- **核心思路**：每个属性提示建模为独立高斯 $\mathcal{G}(z_i^t|p_i^t) \sim \mathcal{N}(\mu_i^t, \sigma_i^t\mathbf{I})$。均值由 text decoder（自注意力+MLP）计算，方差由 visual-text decoder（交叉注意力+MLP）利用视觉特征估计。通过重参数化技巧 $\hat{z}_i^t = \mu(p_i^t) + \epsilon \cdot \sigma(p_i^t)$ 采样，KL散度正则项防止方差坍塌
- **设计动机**：确定性提示表示固定，无法适应动物数据的大方差分布；概率建模允许模拟不同物种间的统计变化，合成新的特征统计信息增强鲁棒性

**设计三：三种跨模态融合策略**
- **功能**：将概率提示表示与视觉特征在空间层面对齐
- **核心思路**：(1) 启发式选择：从 $N_s$ 个采样得分图中选择与目标最相似的；(2) 集成选择：拼接所有得分图后卷积融合 $S = \text{Conv}(\text{Concat}(S'))$；(3) 注意力选择：引入可学习query，通过注意力模块从采样提示中学习最优融合
- **设计动机**：不同融合策略适合不同场景，注意力选择在自由度和信息利用之间取得最优平衡

### 损失函数

总损失 $\mathcal{L}_{total} = \mathcal{L}_{pred} + \mathcal{L}_{spatial} + \gamma \cdot \mathcal{L}_{feature} + \beta \cdot \mathcal{L}_{prompt}$，其中 $\mathcal{L}_{pred}$ 为热力图预测MSE损失，$\mathcal{L}_{spatial}$ 为空间适配MSE损失，$\mathcal{L}_{feature}$ 为对比特征对齐损失，$\mathcal{L}_{prompt} = \mathcal{L}_{div} + \text{KL}(\mathcal{G}\|\mathcal{N}(\mathbf{0},\mathbf{I}))$。

## 实验关键数据

### 主实验：AP-10K 数据集（AP指标）

| 方法 | Backbone | AP | AP.50 | AP.75 | AR |
|------|----------|-----|-------|-------|-----|
| HRNet | HRNet-W48 | 74.4 | 95.9 | 80.7 | - |
| ViTPose++ | ViT-Base | 74.5 | 94.9 | 82.2 | 70.0 |
| X-Pose-V | Swin-Large | 79.0 | 95.7 | 86.8 | - |
| CLAMP | ViT-Base | 74.7 | 95.3 | 81.2 | 77.4 |
| **PPAP(Ours)** | **ViT-Base** | **77.2** | **96.0** | **84.0** | **79.7** |

### 消融实验：各组件贡献（AP-10K, AP指标）

| 设置 | AP | 说明 |
|------|-----|------|
| Baseline (CLAMP) | 74.7 | 单提示+确定性 |
| +多属性提示 | 75.6 | +0.9 |
| +概率建模 | 76.4 | +1.7 |
| +注意力融合 | 77.0 | +2.3 |
| +多样性损失 | 77.2 | +2.5 (Full) |

### 关键发现
- PPAP在AP-10K上以ViT-Base骨干达到77.2 AP，超越相同backbone的CLAMP 2.5个点
- 概率建模相比确定性提示贡献最大（+1.7 AP）
- 在AnimalKingdom零样本设置（P3）下表现优异，对未见物种泛化能力强
- 注意力融合策略一致优于启发式和集成策略
- 方差由visual-text decoder（交叉注意力）估计优于仅从文本估计

## 亮点与洞察

1. **概率提示的通用性**：将概率分布引入提示学习，自然地建模跨物种的数据变异
2. **多样性损失的简洁性**：通过正交约束保持属性多样性，设计简洁有效
3. **GKP策略**：允许关键点名称在模板中随机放置，比ProDA的固定位置策略更灵活
4. **视觉引导的方差估计**：用视觉特征调节文本分布的方差，实现视觉-文本的深度交互

## 局限与展望

- 仍依赖CLIP的预训练知识，对于CLIP见过较少的动物种类可能效果有限
- 概率采样引入额外计算开销，推理时需要多次采样
- 当前仅验证2D关键点估计，3D动物姿态估计尚未探索
- 未来可探索将概率提示学习扩展到其他跨域视觉任务

## 相关工作与启发

- 与ProDA在输出嵌入空间建模单一分布不同，PPAP为每个提示独立建模高斯分布
- 与PPL构建混合高斯分布不同，PPAP保持各属性独立正交
- 概率提示方法对其他需要处理大域间差异的视觉任务有启发

## 评分

⭐⭐⭐⭐ — 概率提示学习框架设计合理，实验充分覆盖有监督和零样本场景；方法新颖性在提示学习领域有较好贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Structure-Aware Correspondence Learning for Relative Pose Estimation](structure-aware_correspondence_learning_for_relative_pose_estimation.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)

</div>

<!-- RELATED:END -->
