---
title: >-
  [论文解读] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis
description: >-
  [CVPR 2025][可解释性][Transformer] 提出 Prompt-CAM，通过为预训练 ViT 注入类别特定的可学习 prompt token，利用最后一层的多头注意力图来识别和定位区分细粒度类别的关键特征（traits），实现了近乎"免费"的可解释细粒度分析。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "Transformer"
  - "细粒度分析"
  - "提示学习"
  - "注意力图"
---

# Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis

**会议**: CVPR 2025  
**arXiv**: [2501.09333](https://arxiv.org/abs/2501.09333)  
**代码**: [https://github.com/Imageomics/Prompt_CAM](https://github.com/Imageomics/Prompt_CAM)  
**领域**: 可解释性  
**关键词**: 可解释性, Vision Transformer, 细粒度分析, Prompt学习, 注意力图

## 一句话总结

提出 Prompt-CAM，通过为预训练 ViT 注入类别特定的可学习 prompt token，利用最后一层的多头注意力图来识别和定位区分细粒度类别的关键特征（traits），实现了近乎"免费"的可解释细粒度分析。

## 研究背景与动机

预训练 ViT（如 DINO）已展示出提取局部化、判别性特征的强大能力，但现有的可视化方法无法有效利用这些特征进行细粒度分析：

- **Grad-CAM 等 saliency map**：在 ViT 上产生模糊、粗糙的热力图，只能高亮整个物体而非区分性特征
- **[CLS] token 的注意力图**：虽能关注局部区域（如头、翅膀、尾巴），但这些区域不是"类别特异的"——所有鸟类都关注相同的身体部位
- **ProtoPNet、INTR 等可解释方法**：需要设计专门模型和复杂训练过程，难以利用最新的预训练 ViT

核心洞察：如果能让每个类别拥有自己的"查询"token，那么每个类别的 token 可以通过注意力机制关注对自己类别"有特殊意义"的图像区域——这就是 traits。通过对比不同类别 token 的注意力图，就能精确定位区分类别的关键特征。

## 方法详解

### 整体框架

Prompt-CAM 基于 Visual Prompt Tuning (VPT) 的框架，核心修改是将预测头从 [CLS] token 输出改为注入的 prompt 输出。给定具有 $C$ 个类别的分类任务，注入 $C$ 个可学习 token，冻结整个 ViT 骨干网络，仅训练这些 token 和一个共享的评分向量 $\boldsymbol{w}$。推理时，各类别 prompt 对图像 patch 的多头注意力图直接揭示该类别的 traits 及其位置。

### 关键设计

1. **类别特定 Prompt 注入（Prompt-CAM-Deep）**:
    - 功能：在 ViT 中引入类别特定的可学习 token，使其注意力图具有类别判别性
    - 核心思路：对 $N$ 层 ViT，在最后一层 $L_N$ 输入 $C$ 个类别特定 prompt $\boldsymbol{P}_{N-1}$，在前 $N-1$ 层各输入 $C$ 个类别无关 prompt $\boldsymbol{P}_i$。最后一层输出 $\boldsymbol{Z}_N$ 通过与共享向量 $\boldsymbol{w}$ 的内积得到各类别得分：$s[c] = \boldsymbol{w}^\top \boldsymbol{z}_N^c$
    - 设计动机：Deep 变体的两个优势：(i) 类别特定 prompt 仅关注最后一层的高级特征 $\boldsymbol{E}_{N-1}$（早期层特征噪声大不适合细粒度判别），(ii) 将"类别特定定位"和"模型适配"解耦为不同层的 prompt，避免一组 prompt 承担双重任务

2. **共享评分向量设计**:
    - 功能：约束模型必须通过注意力图（而非特征通道）来区分类别
    - 核心思路：传统分类器用类别特定权重 $\boldsymbol{w}_c$ 做预测，Prompt-CAM 用共享的 $\boldsymbol{w}$ 做"二值判断"——即判断"类别 $c$ 的 traits 是否出现在图像中"。公式对比：传统 $\hat{y} = \arg\max_c \sum_j \alpha^\star[j] \cdot (\boldsymbol{w}_c^\top \boldsymbol{v}^j)$ vs Prompt-CAM $\hat{y} = \arg\max_c \sum_j \alpha^c[j] \cdot (\boldsymbol{w}^\top \boldsymbol{v}^j)$
    - 设计动机：传统模型可以通过在 value 特征中编码全局判别信息来"抄近路"，即使注意力图无意义也能正确分类。共享 $\boldsymbol{w}$ 消除了这条捷径——如果所有 patch 的 value 相同（无空间信息），所有类别得分也必然相同。模型被迫：(i) 生成保持空间分辨率的局部特征，(ii) 为不同类别生成不同的注意力权重

3. **Trait识别与定位（贪心掩码算法）**:
    - 功能：自动识别每个类别最具判别力的 traits
    - 核心思路：对正确分类的图像，贪心地逐步模糊（替换为均匀注意力 $\frac{1}{M}\boldsymbol{1}$）最不重要的注意力头，直到分类错误。具体每步中，对每个未模糊头 $r'$，临时替换 $\boldsymbol{\alpha}_{N-1}^{c,r'}$ 为均匀向量并重算 $s[c]$，选择"模糊后 $s[c]$ 下降最小"的头为最不重要头。剩余的头就是最关键的 traits
    - 设计动机：$R$ 个注意力头可能关注 $R$ 个不同区域，但并非所有都同等重要。贪心掩码可以自动筛选出"少而精"的判别性 traits

### 损失函数 / 训练策略

仅使用标准交叉熵损失：$-\log \frac{\exp(s[y])}{\sum_c \exp(s[c])}$

- 冻结整个 ViT 骨干，仅学习 prompt token $\boldsymbol{P}$ 和共享向量 $\boldsymbol{w}$
- 使用 SGD 优化器
- 默认骨干：DINO ViT-B，也验证了 DINOv2 和 BioCLIP

## 实验关键数据

### 主实验（忠实度评估，CUB-200-2011）

| 方法 | Insertion ↑ | Deletion ↓ | 说明 |
|--------|------|------|------|
| Grad-CAM | 0.52 | 0.17 | 后验解释方法 |
| Layer-CAM | 0.54 | 0.13 | |
| Eigen-CAM | 0.56 | 0.22 | |
| Attention roll-out | 0.55 | 0.27 | |
| **Prompt-CAM** | **0.61** | **0.09** | 最高插入+最低删除 |

### 分类精度 vs 可解释性权衡（DINO骨干）

| 方法 | CUB | Bird-525 | Dog | Pet |
|------|------|---------|------|------|
| Linear Probing | 78.6 | 99.2 | 82.4 | 92.4 |
| Prompt-CAM | 73.2 | 98.8 | 81.1 | 91.3 |

### 人类评估（Trait 识别质量）

| 方法 | Trait 识别率 | 说明 |
|------|---------|------|
| **Prompt-CAM** | **60.49%** | 显著最优 |
| TesNet | 39.14% | |
| ProtoConcepts | 30.39% | |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Prompt-CAM-Shallow (仅第一层) | 精度略低 | 早期层特征噪声大 |
| Prompt-CAM-Deep (最后一层) | 更优 | 仅关注高级特征 |
| 不同骨干 (DINO/DINOv2/BioCLIP) | 一致的trait定位 | 方法通用性强 |
| 移除红翅黑鸟翅上红斑 | 分类变为Boat-tailed Grackle | 反事实验证忠实度 |

### 关键发现

- **Prompt-CAM 在 13 个跨领域数据集上均有效**：动物（鸟、鱼、昆虫）、植物（花、药叶）、物体（汽车、食物），展现了极强的通用性
- **精度下降可解释**：被 Prompt-CAM 误分类但 Linear Probing 正确的图像中，通常是 trait 被遮挡或姿态异常——说明 Prompt-CAM 确实依赖 trait 而非全局信息
- **多头注意力自然解耦不同 traits**：如 Scott Oriole 的不同注意力头分别关注黄腹、黑头和黑胸，无需额外约束
- **可扩展到层级分类学**：在鱼类数据集上，不同分类层级（科→属→种）的 Prompt-CAM 关注从粗粒度到细粒度的不同特征

## 亮点与洞察

- **极致简洁的设计**：基于 VPT 仅需修改几行代码（改预测头位置），无需新损失函数、无需新模块、无需修改骨干——"几乎免费的午餐"
- **共享 $\boldsymbol{w}$ 的理论分析精彩**：Eq. 6 vs Eq. 7 的对比清楚说明了为什么传统分类器的注意力图无法提供可靠的可解释性——因为类别判别信息可以"逃逸"到 value 特征中
- **跨物种 trait 比较**：可通过可视化 A 类图像上 B 类 prompt 的注意力来发现两个物种的共有特征（Figure 1c），这是独特的功能

## 局限与展望

- 分类精度有所牺牲（CUB 上下降约 5%），是可解释性和精度的固有权衡
- 类别数 $C$ 即为 prompt 数目，$C$ 很大时参数和计算量线性增长
- 当前仅利用最后一层注意力图，中间层的信息被丢弃
- Trait 的语义标注仍需人工解读，未与自然语言描述自动关联

## 相关工作与启发

- 与 INTR（encoder-decoder 架构 + full fine-tuning）相比，Prompt-CAM 更简单、更快、注意力图更清晰，且可自由选择任意 ViT 骨干
- 与 ProtoPNet 系列（原型网络）相比，无需维护原型库和复杂的训练流程
- 启发：VPT 的 prompt 输出被"浪费"了（原始 VPT 丢弃它们），而 Prompt-CAM 发现它们包含丰富的类别特定信息——这种"废物利用"的思路值得在其他 prompt 方法中探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 将VPT的prompt输出重新利用实现可解释性，巧妙且简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 13个数据集跨3大领域，人类评估+反事实分析+层级分类学，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析（Eq.6 vs Eq.7）、可视化案例和实验设计都非常出色
- 价值: ⭐⭐⭐⭐⭐ 为ViT可解释性提供了简单实用的通用工具，对生态学等领域有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](lswag_zero_shot_nas.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](../../ACL2026/interpretability/fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)

</div>

<!-- RELATED:END -->
