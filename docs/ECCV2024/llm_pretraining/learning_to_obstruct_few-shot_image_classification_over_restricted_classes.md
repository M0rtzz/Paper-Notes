---
title: >-
  [论文解读] Learning to Obstruct Few-Shot Image Classification over Restricted Classes
description: >-
  [ECCV 2024][少样本学习] 提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。
tags:
  - ECCV 2024
  - 少样本学习
  - 模型安全
  - 元学习
  - 预训练模型保护
  - 类别限制
---

# Learning to Obstruct Few-Shot Image Classification over Restricted Classes

**会议**: ECCV 2024  
**arXiv**: [2409.19210](https://arxiv.org/abs/2409.19210)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: 少样本学习, 模型安全, 元学习, 预训练模型保护, 类别限制

## 一句话总结

提出 Learning to Obstruct (LTO) 算法，通过类似 MAML 的元学习方式修改预训练 backbone 参数，使其成为特定受限类别的"差初始化"，从而阻碍少样本分类方法在受限类上的微调效果，同时保持其他类别的正常性能。

## 研究背景与动机

开源预训练模型极大降低了构建计算机视觉系统的门槛，但这也带来安全隐患：恶意用户可以利用少样本微调快速开发有害应用（如隐私侵犯场景下的人脸分类）。

**核心问题**：能否开发一种预训练模型，使其对特定"受限类别"难以微调，同时对其他类别保持正常的微调能力？

这是一个全新的问题设定，与已有工作的区别：
- **机器遗忘 (Machine Unlearning)**：目标是"移除"已学到的类别知识；LTO 的目标是"阻止学习"新的受限类别
- **数据投毒 (Data Poisoning)**：修改训练数据来破坏模型；LTO 修改模型权重本身
- **MAML**：学习"好的初始化"使模型快速适应新任务；LTO 反其道行之，学习"差的初始化"

## 方法详解

### 整体框架

给定预训练 backbone $\theta^p$、受限类集合 $\mathcal{R}$ 和少样本分类算法 $\bm{F}$，LTO 算法 $\bm{A}$ 修改预训练权重为 $\bm{A}(\theta^p)$，使得 FSC 方法应用后在 $\mathcal{R}$ 上表现差，在其他类 $\mathcal{R}'$ 上保持正常。

### 关键设计

1. **LTO 优化目标 — 学习差初始化**：

受 MAML 启发，LTO 将数据拆分为 $\mathcal{D}_{obs}$（评估阻碍质量）和 $\mathcal{D}_{fsc}$（用于 FSC 训练），优化问题为：

$$\min_\theta \mathbb{E}_{\mathcal{T}^{(t)}} \left[\mathcal{L}_{\mathcal{R}'}([\tilde{\theta}, \tilde{\phi}], \mathcal{D}_{obs}^{(t)}) - \mathcal{L}_{\mathcal{R}}([\tilde{\theta}, \tilde{\phi}], \mathcal{D}_{obs}^{(t)})\right]$$

$$\text{s.t.} \quad \tilde{\theta}, \tilde{\phi} = F([\theta, \phi], \mathcal{D}_{fsc}^{(t)})$$

- $\mathcal{L}_{\mathcal{R}'}$：其他类上的损失（最小化→保持性能）
- $-\mathcal{L}_{\mathcal{R}}$：受限类上的损失取负（最大化→破坏性能）
- 关键约束：$\tilde{\theta}$ 是 FSC 学习器 $F$ 更新后的参数，即先让模型"尝试学习"再评估阻碍效果

2. **双层优化求解**：

类似 MAML 的双层优化：
- **内层 (Inner loop)**：用 FSC 学习器 $F$ 在 $\mathcal{D}_{fsc}$ 上更新参数 $\theta \to \tilde{\theta}$
- **外层 (Outer loop)**：在 $\mathcal{D}_{obs}$ 上计算梯度 $\Delta\theta^{(t)} = \nabla_\theta[\mathcal{L}_{\mathcal{R}'} - \mathcal{L}_{\mathcal{R}}]$，通过一阶近似反向传播
- 每 epoch 恢复参数到 epoch 初始状态，确保批次内各任务使用相同起点

3. **受限类选择策略**：将类别按语义分为超类（如"鸟类"、"电子设备"），选择整个超类作为 $\mathcal{R}$，模拟现实中限制某一类应用的场景。

4. **扩展到属性学习**：对 CelebA 属性分类，将每个属性视为独立的二分类任务，$\mathcal{R}$ 为受限属性集合，分别对各属性构建 CLIP 二分类器并应用 LTO。

5. **CLIP-based FSC 适配**：对基于 CLIP 的方法（CoOp, TipAdapter），由于 GPU 内存限制，采用重采样策略——每隔几步从全部 prompt 和类别中随机采样子集计算梯度，提供无偏估计。

### 损失函数 / 训练策略

LTO 外层优化目标 = $\mathcal{L}_{\mathcal{R}'} - \mathcal{L}_{\mathcal{R}}$，使用 mini-batch 梯度下降。200 步阻碍学习，batch size 20，内层 FSC 学习 20 步。

两个对比基线：
- **Only$\mathcal{R}$**：直接最大化受限类损失（不考虑其他类）
- **NoF**：不经过 FSC 学习器直接优化（不考虑微调过程）

## 实验关键数据

### 主实验

ImageNet 上经典 FSC 的 DropRatio@2% (Δ@2，越高阻碍效果越好)：

| FSC 方法 | 设置 | Only$\mathcal{R}$ | NoF | **LTO (Ours)** |
|----------|------|----------|-----|---------|
| ProtoNet | 1-shot | 1.10 | 3.77 | **4.42** |
| ProtoNet | 5-shot | 1.10 | 2.00 | **2.40** |
| MetaOptNet | 1-shot | 1.95 | 8.65 | **8.85** |
| MetaOptNet | 5-shot | 1.94 | 10.11 | **13.40** |

CLIP-based FSC 在 CIFAR100/ImageNet 上的 Δ@2（平均）：

| FSC 方法 | CIFAR100 Only$\mathcal{R}$ | CIFAR100 Ours | ImageNet Only$\mathcal{R}$ | ImageNet Ours |
|----------|----------|------|----------|------|
| CE | 1.48 | **11.99** | 2.80 | **7.65** |
| CoOp | 1.80 | **6.73** | 1.19 | **4.58** |
| TipAdapter | 2.08 | **10.16** | 2.16 | **5.86** |

### 消融实验

数据效率分析（ImageNet superclass id=1, 增加 FSC 训练数据）：

| FSC 方法 | 1× (5-shot) | 2× | 3× | 4× |
|----------|------------|-----|-----|-----|
| CE | 9.93 | 6.34 | 2.76 | 2.82 |
| CoOp | 6.15 | 2.46 | 2.26 | 2.32 |
| TipAdapter | 5.92 | 3.09 | 3.06 | 3.76 |

跨方法迁移性（LTO 用 $\bm{F}$ 训练，用 $\bm{F}'$ 评估的 Δ@2）：

| $\bm{F}$ \ $\bm{F}'$ | CE | CoOp | TipAdapter | 平均 |
|------|------|------|-----------|------|
| CE | 9.93 | 4.71 | 7.33 | 7.32 |
| CoOp | 4.79 | 6.15 | 4.34 | 5.09 |
| TipAdapter | 4.16 | 7.75 | 5.92 | 5.94 |

### 关键发现

- LTO 在所有 FSC 方法上均能有效阻碍受限类学习，Δ@2 远大于 1（受限类下降远大于其他类下降）
- 考虑 FSC 学习器 $F$ 的内层优化至关重要：NoF 基线（不模拟微调过程）效果明显弱于完整 LTO
- 增加 FSC 训练数据或训练时长只能部分恢复被阻碍的性能，说明阻碍效果具有一定鲁棒性
- LTO 具有跨方法迁移性：用一个 FSC 方法训练的阻碍对其他方法也有效
- 与机器遗忘方法 SSD 的对比表明：SSD 的阻碍效果远弱于 LTO（CIFAR100 avg Δ@2: SSD 1.48 vs LTO 11.99），因为 SSD 只是遗忘而非阻止重新学习
- 属性学习上，某些属性更易阻碍（Pale_Skin: 31.64%），某些更难（Gray_Hair: 0.08%），属性间的语义关联会导致连带阻碍

## 亮点与洞察

1. **问题定义新颖且有实际意义**：在开源模型安全领域提出了一个全新视角——不是防止模型泄露，而是从源头让模型对特定任务"不好用"
2. **方法与 MAML 对偶**：MAML 学好初始化，LTO 学差初始化，概念简洁优雅
3. **跨方法迁移性令人惊喜**：用一种 FSC 方法训练的阻碍，对其他未见过的 FSC 方法同样有效，说明 LTO 确实修改了 backbone 中与受限类相关的特征表示
4. **评估指标设计合理**：DropRatio@β 同时考虑了受限类的下降幅度和其他类的保持程度

## 局限与展望

1. LTO 假设阻碍者知道 FSC 算法，虽然实验表明有跨方法迁移性，但对完全未知的微调策略效果未知
2. 增加足够多的数据（4×）可以部分抵消阻碍效果，说明对数据充足的场景保护力有限
3. 仅在图像分类上验证，对其他下游任务（如检测、分割）的适用性未知
4. 属性学习中存在语义关联导致的非预期阻碍（如阻碍"金发"连带影响"棕发"），精确的单属性阻碍仍是挑战
5. 阻碍 200 步需要重复内层 FSC 训练 20 steps × 20 batch，计算开销不小

## 相关工作与启发

- **MAML [Finn et al.]**：本文的方法论基础，LTO 是其"反向版本"
- **ProtoNet / MetaOptNet**：经典 FSC 方法，作为阻碍目标
- **CoOp / TipAdapter**：CLIP-based FSC 方法，验证了 LTO 对基础模型的有效性
- **SSD [Foster et al.]**：机器遗忘方法，实验对比表明"遗忘"不等于"阻止重新学习"
- **数据投毒**：修改数据而非模型，与 LTO 互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 全新问题定义，方法与 MAML 对偶的设计非常巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 四种 FSC 方法、三个数据集、两种任务类型，加上数据效率/时间效率/跨方法迁移等深入分析
- 写作质量: ⭐⭐⭐⭐ — 问题引入清晰，公式推导完整；表格较密集但信息量大
- 综合价值: ⭐⭐⭐⭐ — 开创了"阻碍学习"这一新方向，对 AI 安全领域有重要启发意义

<!-- RELATED:START -->

## 相关论文

- [Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)
- [DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)
- [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](../../CVPR2025/llm_pretraining/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)

<!-- RELATED:END -->
