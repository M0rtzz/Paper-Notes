---
title: >-
  [论文解读] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging
description: >-
  [AAAI2026][医学图像][DNA foundation model] 提出 MergeDNA，通过可微分的 Token Merging 机制实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder Transformer 和自适应 masked token modeling 预训练，在多个基因组 benchmark 上取得 SOTA。
tags:
  - AAAI2026
  - 医学图像
  - DNA foundation model
  - token merging
  - dynamic tokenization
  - genome modeling
  - masked language modeling
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging

**会议**: AAAI2026  
**arXiv**: [2511.14806](https://arxiv.org/abs/2511.14806)  
**代码**: 待确认  
**领域**: medical_imaging  
**关键词**: DNA foundation model, token merging, dynamic tokenization, genome modeling, masked language modeling

## 一句话总结
提出 MergeDNA，通过可微分的 Token Merging 机制实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder Transformer 和自适应 masked token modeling 预训练，在多个基因组 benchmark 上取得 SOTA。

## 背景与动机
- DNA 序列的信息密度分布极不均匀：仅约 2% 是编码序列（CDS），大量为非编码区域
- DNA 没有天然的"词"边界：有意义的单元可能是 3 bases（codon）、6-10 bases（转录因子结合位点）或更长
- DNA 序列极长（数万到数百万 bases），需要同时捕获短程 motif 和长程依赖
- 现有方法在 tokenization（固定 k-mer / BPE）、长序列建模（SSM / Transformer）、预训练目标三个维度上各自优化，缺乏统一框架

## 核心问题
如何设计一个端到端可学习的基因组建模框架，同时解决：(1) 上下文敏感的动态 tokenization，(2) 自适应地关注信息密集区域的预训练目标？

## 方法详解

### 整体框架
MergeDNA 采用层次化 autoencoder 架构，包含四个模块：
1. **Local Encoder**（可学习 tokenizer）：堆叠多层 local-window self-attention + 可微分 token merging，将相邻 bases 合并为变长 tokens，输出 $Z_L \in \mathbb{R}^{L \times D}$ 和 source matrix $\mathcal{S} \in \{0,1\}^{L \times N}$
2. **Latent Encoder**：全注意力 Transformer，捕获全局长程依赖，输出 $Z'_L = \mathcal{E}_\psi(Z_L)$
3. **Latent Decoder**：对称结构，将 $Z'_L$ 映射回 token 空间
4. **Local Decoder**：通过 token unmerging（$\bar{Z}_N = \mathcal{S}^\top \hat{Z}_L$）恢复到原始长度，再用 local attention 重建

### 关键设计
- **Local-window Token Merging**：每层用 lightweight grouping embedding 计算相似度，选 top-$r_l$ 对 merge，soft merging 保证可微分
- **Merged Token Reconstruction (MTR)**：端到端重建损失 $\mathcal{L}_{MTR} = -\frac{1}{N}\sum_{i=1}^{N}\log P(\hat{X}_i | X_i; \theta)$，训练时对压缩率做高斯采样（$L \in [0.4N, 0.6N]$）
- **Adaptive Masked Token Modeling (AMTM)**：利用 Latent Encoder 的 global token merging 结果识别重要 token，按重要性采样 mask $K$ 个 token 进行预测，masking 概率与 merge group 大小成反比
- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{MTR}(\theta) + \lambda \mathcal{L}_{MTR}(\theta \setminus \{\phi\}) + \mathcal{L}_{AMTM}(\theta)$，$\lambda = 0.25$

## 实验关键数据

| 方法 | Params | Enhancers (3) | Species (2) | Regulatory (3) | Avg (8 tasks) |
|------|--------|--------------|-------------|----------------|---------------|
| NT-500M | 500M | 84.56 | 96.64 | 89.05 | 89.26 |
| GENERator | 1.3B | 84.87 | 96.95 | 90.30 | 90.71 |
| **MergeDNA** | **380M** | **85.11** | 96.84 | **90.66** | **90.87** |

- NT Benchmark (18 tasks): MergeDNA 平均 MCC 78.39%，超越 MxDNA (78.14%) 和其他所有 baseline
- 在 Splice Site 任务上尤其突出（Donor: 98.93%，Acceptor: 98.67%）
- 380M 参数量下超越 1.3B 的 GENERator

## 亮点
1. **统一框架**：首次将动态 tokenization、长序列建模和自适应预训练目标整合在一个端到端可学习框架中
2. **信息感知**：tokenizer 能自动对信息密集区域分配更细粒度的 token，repetitive 区域则合并
3. **参数高效**：380M 参数超越 1.3B 的 GENERator，压缩比到 $L \approx N/2$ 后还能保持性能
4. **跨模态迁移**：在 RNA 和 protein 下游任务上也表现出良好的泛化能力

## 局限与展望
- 预训练序列长度仅 4096，对于真实基因组级别（数百万 bases）仍显不足
- Token merging 的 local window 固定为 16，可能限制了对更长 motif 的发现
- 缺少与 Evo2 等超大规模模型的直接对比
- 下游任务主要是分类，缺少生成任务（如序列设计）的验证

## 与相关工作的对比
- vs **DNABERT-2** (BPE tokenizer)：MergeDNA 的动态 tokenizer 更灵活，平均提升 3.5+%
- vs **VQDNA** (VQ tokenizer)：同为可学习 tokenizer，但 MergeDNA 是连续的 token merging 而非离散的 VQ，性能更优
- vs **HyenaDNA/Caduceus** (SSM)：MergeDNA 用层次化 Transformer 替代 SSM，在效率和性能间取得更好平衡
- vs **MxDNA** (Dynamic Context)：架构理念相近，但 MergeDNA 有更强的预训练目标设计

## 启发与关联
- Token merging 从 ViT（ToMe）迁移到 DNA 的思路有启发性，可考虑扩展到其他长序列模态（音频、时序信号）
- 自适应 masking 策略（按信息密度调整 mask 概率）可作为通用预训练技巧
- 压缩率采样策略（训练时随机化压缩比）可提升模型鲁棒性，值得借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ — Token merging 在 DNA 上的首次系统应用，框架设计完整
- 实验充分度: ⭐⭐⭐⭐ — 三大 benchmark + 多组消融，但缺少超大规模对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ — 为 DNA foundation model 的 tokenization 提供了新范式

<!-- RELATED:START -->

## 相关论文

- [Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](../../ICML2025/medical_imaging/context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)
- [Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)
- [Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)

<!-- RELATED:END -->
