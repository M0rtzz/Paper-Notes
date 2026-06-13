---
title: >-
  [论文解读] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research
description: >-
  [NeurIPS 2025][计算生物][DNA条形码] BarcodeMamba+ 是面向真菌 ITS DNA 条形码分类的 SSM 基础模型，通过预训练+微调范式充分利用海量未标注序列，并结合层次标签平滑、逆平方根加权损失和多头输出三项增强，在三个测试集所有分类层级上大幅超越 BLAST、CNN 和 Transformer 基线，种级准确率最高达 88.9%。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "DNA条形码"
  - "真菌分类"
  - "状态空间模型"
  - "基础模型"
  - "层次分类"
---

# BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research

**会议**: NeurIPS 2025  
**arXiv**: [2512.15931](https://arxiv.org/abs/2512.15931)  
**代码**: [GitHub](https://github.com/bioscan-ml/BarcodeMamba)  
**领域**: 生物信息学 / 基因组学  
**关键词**: DNA条形码, 真菌分类, 状态空间模型, 基础模型, 层次分类

## 一句话总结

BarcodeMamba+ 是面向真菌 ITS DNA 条形码分类的 SSM 基础模型，通过预训练+微调范式充分利用海量未标注序列，并结合层次标签平滑、逆平方根加权损失和多头输出三项增强，在三个测试集所有分类层级上大幅超越 BLAST、CNN 和 Transformer 基线，种级准确率最高达 88.9%。

## 研究背景与动机

DNA 条形码是实现大规模自动化生物多样性监测的基石，但真菌分类面临极端挑战：

- **标注极度稀疏**：收集的真菌样本中高达 93% 在种级别缺少标注
- **长尾分布严重**：5.23M 训练序列涵盖 14.7K 物种，但分布极度不均
- **传统方法瓶颈**：BLAST 推理慢（208.6 ms/样本）且泛化差；CNN/Transformer 的全监督训练在标注稀疏条件下效果有限
- **基础模型机遇**：海量未标注 DNA 序列可通过预训练学到泛化表示，再用少量标注微调

核心思路：将 Mamba（高效 SSM 架构）引入 DNA 条形码分类，结合预训练+微调范式，配合层次分类增强策略解决真菌分类的数据稀疏和长尾问题。

## 方法详解

### 整体框架

采用两阶段训练范式：

1. **预训练阶段**：在 UNITE+INSD 数据集的 5.23M ITS 序列上进行 next-token prediction 自监督预训练，不使用分类标签
2. **微调阶段**：添加分类头，在有标注数据上微调，融合三项层次分类增强

使用 BPE 分词器处理 DNA 序列（而非字符级或 k-mer 分词），因为 BPE 在真菌 ITS 数据上被验证为最优选择。

### 关键设计

1. **Mamba SSM 架构**

    - 基于状态空间模型，具有线性时间复杂度，适合处理大规模生物序列
    - 基础版 12.1M 参数（与 CNN 基线相当），大版 49.2M 参数
    - 相比 Transformer（BarcodeBERT 44.6M），参数量更小且推理更平衡

2. **层次标签平滑（Hierarchical Label Smoothing）**

    - 利用分类学层次结构（界/门/纲/目/科/属/种 七级）的信息
    - 在 softmax 目标中根据分类学距离分配平滑概率
    - 使分类学上相近的类别获得部分概率，增强泛化

3. **逆平方根加权损失（Weighted Loss）**

    - 对稀有类别赋予更高权重，应对长尾分布
    - 防止模型被高频类别主导

### 损失函数 / 训练策略

- 预训练使用 next-token prediction（语言模型风格）
- 微调使用加权交叉熵 + 层次标签平滑
- 多头输出：每个分类层级（门/纲/目/科/属/种）各有独立分类头
- BPE 分词优于字符级和 k-mer 分词

## 实验关键数据

### 主实验（表格）

在三个测试集上的种级准确率（Accuracy %）：

| 模型 | Yeast | Filamentous | MycoAI | 参数量 | 推理时间 |
|------|-------|-------------|--------|--------|---------|
| BLAST | 75.4 | 33.4 | 55.0 | N/A | 208.6ms |
| MycoAI-CNN | 60.0 | 28.2 | 57.1 | 11.6M | 11.8ms |
| MycoAI-BERT | 33.5 | 16.6 | 39.3 | 18.4M | 4.5ms |
| CNN Encoder | 67.6 | 31.4 | 72.6 | 12.1M | 5.8ms |
| BarcodeBERT | 59.1 | 27.7 | 58.9 | 44.6M | 8.8ms |
| **BarcodeMamba+** | **80.6** | **46.5** | **81.7** | 12.1M | 8.0ms |
| **BarcodeMamba+ (large)** | **83.6** | **50.4** | **88.9** | 49.2M | 14.7ms |

### 消融实验

**预训练 vs 全监督**（BPE 分词，MycoAI 测试集种级准确率）：

| 训练方式 | 准确率 |
|---------|-------|
| 全监督（无预训练） | 78.6% |
| 预训练+微调 | **81.7%** |

预训练在 k-mer 分词下提升更显著（77.0% → 81.1%），证明预训练在标注稀疏场景下的优势。

**分词方法对比**（预训练+微调，MycoAI 种级）：

| 分词方式 | 准确率 |
|---------|-------|
| Char | 79.0% |
| k-mer | 81.1% |
| **BPE** | **81.7%** |

### 关键发现

- BarcodeMamba+ 在所有分类层级和所有测试集上全面领先，种级准确率在 MycoAI 上达 81.7%，比次优的 CNN Encoder（72.6%）高出 9.1 个百分点
- 在分布偏移最大的 Filamentous 测试集上优势最明显（46.5% vs 31.4%，领先 15 个百分点）
- 扩大模型到 49.2M 参数后，MycoAI 种级准确率从 81.7% 跃升至 88.9%，确认架构可扩展性
- 推理速度为 8ms/样本，比 BLAST（208.6ms）快 25 倍以上

## 亮点与洞察

- **预训练+微调范式**在标注极度稀疏（93%无种级标注）的基因组领域展现出巨大优势，这是传统全监督无法比拟的
- SSM 架构在 DNA 序列上的线性复杂度优势使其特别适合大规模生物多样性监测
- 层次分类的三项增强（标签平滑+加权损失+多头输出）各自贡献显著性能提升，且彼此互补
- BPE 分词在 DNA 序列上优于 k-mer 和字符级分词，这与 NLP 领域的经验一致

## 局限与展望

- 仅在真菌 ITS 区域验证，向其他生物类群（如昆虫 COI）的迁移效果待验证
- 未与蛋白质语言模型（如 ESM）或最新 DNA 基础模型进行对比
- 测试集 2（Filamentous）的绝对准确率仍不到 50%，说明极端分布偏移下仍有改进空间
- 大模型版本的推理时间几乎翻倍（14.7ms），对实时部署场景可能有影响

## 相关工作与启发

- 与 BarcodeBERT（Transformer 基础模型）相比，Mamba 在更小参数量下实现更好性能
- MycoAI 引入的多头输出和层次增强策略被本文系统集成到 SSM 架构中
- 该工作验证了基础模型范式在生物多样性监测中的实际工具价值

## 评分

- 新颖性: ⭐⭐⭐ Mamba 用于 DNA 分类是合理的架构选择但非突破性创新
- 实验充分度: ⭐⭐⭐⭐ 三个测试集 + 分词/训练范式/增强策略的完整消融
- 写作质量: ⭐⭐⭐⭐ 清晰且实验设计严谨
- 价值: ⭐⭐⭐⭐ 对生物多样性研究有实际工具价值，开源代码可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](../../ICML2025/computational_biology/elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)
- [\[ICML 2025\] LDMol: A Text-to-Molecule Diffusion Model with Structurally Informative Latent Space Surpasses AR Models](../../ICML2025/computational_biology/ldmol_a_text-to-molecule_diffusion_model_with_structurally_informative_latent_sp.md)

</div>

<!-- RELATED:END -->
