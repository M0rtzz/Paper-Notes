---
title: >-
  [论文解读] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval
description: >-
  [CVPR 2026][自监督学习][UCDIR] 提出TPSNet，将CLIP学习的域提示（domain prompt）作为文本先验提供精细语义监督，同时引入相位谱特征作为相位先验来桥接域分布差异并保持语义完整性，通过文本-相位双先验的协同实现无监督跨域图像检索的显著提升。
tags:
  - CVPR 2026
  - 自监督学习
  - UCDIR
  - 提示学习
  - phase spectrum
  - text-phase dual priors
  - cross-domain alignment
---

# Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval

**会议**: CVPR 2026  
**arXiv**: [2603.12711](https://arxiv.org/abs/2603.12711)  
**代码**: 无  
**领域**: 跨域检索 / 自监督学习  
**关键词**: UCDIR, domain prompt, phase spectrum, text-phase dual priors, cross-domain alignment  

## 一句话总结

提出TPSNet，将CLIP学习的域提示（domain prompt）作为文本先验提供精细语义监督，同时引入相位谱特征作为相位先验来桥接域分布差异并保持语义完整性，通过文本-相位双先验的协同实现无监督跨域图像检索的显著提升。

## 研究背景与动机

**无监督跨域图像检索（UCDIR）**旨在无标注数据条件下，在异构图像域之间（如真实图片和草图）检索语义相同的图像。核心困难是：无标注+域分布差异两重挑战叠加。

**现有方法的两个痛点**：（1）**伪标签噪声**——通过K-means聚类生成伪标签作为监督信号，但离散伪标签常不准确，导致域内表征学习和跨域对齐都受噪声干扰，类原型也不可靠；（2）**跨域对齐导致语义退化**——对抗训练、统计分布对齐等策略在消除域差异时会不可避免地损害语义信息，因为域特定特征和语义特征是纠缠的。

**TPSNet的两条解决路径**：（1）用CLIP学习的域提示作为文本先验，比离散伪标签提供更丰富精确的语义监督；（2）用傅里叶变换分离出的相位谱作为相位先验——相位谱编码了结构和语义信息且对域偏移鲁棒——桥接域差异的同时保持语义完整。两条路径协同工作。

## 方法详解

### 整体框架

TPSNet分两模块：Domain Prompt Generation (DPG) 用CLIP对比学习优化每个域的C个类别提示 → Text-Phase Dual Priors Network (TPDP) 用学到的域提示做文本先验指导语义特征提取 + 用相位谱特征做相位先验桥接域差异，通过cross-attention融合双先验得到最终的域不变语义表征。

### 关键设计

1. **Domain Prompt Generation（域提示生成）**:

    - 功能：为每个域学习C个类别特定的可学习text prompt，作为后续的语义监督信号
    - 核心思路：K-means聚类产生伪标签后，初始化C个可学习prompt模板（"An image of a [X]¹...[X]^M"）。用冻结CLIP做图文对比学习 $\mathcal{L}_{prompt} = \mathcal{L}_{i2t} + \mathcal{L}_{t2i}$，仅优化[X] token。对比学习中基于cosine similarity重新配对图文，部分修正不准确的伪标签
    - 设计动机：CLIP的文本表示比离散伪标签提供更丰富的语义先验，通过对比学习优化后domain prompt编码了精确的类别语义信息

2. **Phase-Prior域不变特征提取**:

    - 功能：利用相位谱的域不变性来桥接域差异
    - 核心思路：灰度图做FFT得 $F(u,v)=|A(u,v)|e^{j\phi(u,v)}$。保留相位、用常数R替换幅度：$F'(u,v) = Re^{j\phi(u,v)}$，IFFT重建相位图像。轻量CNN提取相位特征 $I^{phase}$，与RGB特征通过LayerNorm+SelfAttention融合为 $I^f$
    - 设计动机：相位谱编码结构和边缘，对域偏移比幅度谱更鲁棒。丢弃幅度天然消除了部分域特定因素（如风格、颜色分布）

3. **Text-Phase双先验协同融合**:

    - 功能：通过cross-attention让文本先验和相位先验协同指导域不变表征学习
    - 核心思路：域提示文本特征 $T'$ 做Query，融合视觉特征 $I^f$ 做Key/Value，$I' = \text{CrossAttention}(T'; I^f)$。用prototype交叉熵 $\mathcal{L}_{pce}$ 和图文对比 $\mathcal{L}_{i2tce}$（带标签平滑）联合训练。Prototype动量更新 $\mathcal{P} \leftarrow m\mathcal{P} + (1-m)I'$
    - 设计动机：文本先验提供语义引导方向，相位先验消除域偏移，cross-attention让两者在特征空间中互补增强

### 损失函数 / 训练策略

$\mathcal{L} = \alpha \mathcal{L}_{pce} + \beta \mathcal{L}_{i2tce}$，其中 $\mathcal{L}_{i2tce}$ 使用标签平滑 $\sigma_j = (1-\epsilon)y_i + \epsilon/C$ 缓解伪标签噪声。Stage 1仅优化prompt token；Stage 2训练TPDP全模块。

## 实验关键数据

### 主实验

**Office-Home（65类4域, 12个跨域场景）和DomainNet（7类6域）**：

| 方法 | Office-Home 平均P@1 | Office-Home 平均P@15 |
|------|-------------------|---------------------|
| DD | ~45 | ~35 |
| ProtoOT | ~50 | ~47 |
| ShieldIR | ~53 | ~50 |
| **TPSNet** | **显著SOTA** | **显著SOTA** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅伪标签（无域提示） | 基线 | 噪声large |
| +文本先验（域提示） | 显著↑ | 语义监督更精确 |
| +相位先验 | 进一步↑ | 域不变特征有帮助 |
| **双先验协同** | **最优** | 互补增强效果最佳 |

### 关键发现

- 文本先验单独就能显著提升——说明CLIP的语义信号比聚类伪标签丰富得多
- 相位先验在跨域差异大的场景中（如Art↔Clipart）提升更明显——验证了相位谱的域不变性假设
- 标签平滑对缓解伪标签噪声有效

## 亮点与洞察

- 文本先验+相位先验的双路径设计很有启发性——前者从语义空间、后者从频率空间分别提供互补的域不变信号。这种"多视角域不变性"比单一对齐策略更鲁棒。
- 用常数幅度+原始相位重建图像的操作虽然简单，但效果显著——相位确实编码了跨域一致的结构语义信息。

## 局限与展望

- 依赖K-means聚类初始化domain prompt，聚类质量对后续所有步骤影响较大
- 相位谱仅从灰度图提取，丢失了颜色信息中可能的域不变成分
- 数据集的域差异相对有限，更极端域偏移效果待验证

## 相关工作与启发

- **vs DD/CODA**: 直接用伪标签做域内对比和跨域对齐，TPSNet用domain prompt替代伪标签提供更好的语义监督
- **vs FDA/FUDA**: 在频域替换低频做域适应，TPSNet更进一步分离幅度/相位，利用相位的天然域不变性

## 评分

- 新颖性: ⭐⭐⭐⭐ 文本先验+相位先验的双路径设计在UCDIR中是新颖组合
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark、12个跨域场景、消融充分
- 写作质量: ⭐⭐⭐ 结构清晰但图表较复杂
- 价值: ⭐⭐⭐ UCDIR是有意义问题，提升显著
