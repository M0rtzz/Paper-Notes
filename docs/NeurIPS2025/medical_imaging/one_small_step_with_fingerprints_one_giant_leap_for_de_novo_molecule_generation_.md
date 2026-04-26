---
title: >-
  [论文解读] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra
description: >-
  [NeurIPS 2025 (AI4Mat Workshop)][医学图像][质谱] 通过将 MIST 作为质谱-指纹编码器、MolForge 作为指纹-结构解码器，并采用先验调整阈值策略，在 MassSpecGym 基准上实现了从质谱从头生成分子结构的十倍性能提升（top-1 准确率从 2.3% 提升至 31%）。
tags:
  - NeurIPS 2025 (AI4Mat Workshop)
  - 医学图像
  - 质谱
  - 分子指纹
  - 从头分子生成
  - MolForge
  - MIST
---

# One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra

**会议**: NeurIPS 2025 (AI4Mat Workshop)  
**arXiv**: [2508.04180](https://arxiv.org/abs/2508.04180)  
**代码**: 有（使用开源 MIST + MolForge）  
**领域**: 计算化学 / 分子生成  
**关键词**: 质谱, 分子指纹, 从头分子生成, MolForge, MIST

## 一句话总结

通过将 MIST 作为质谱-指纹编码器、MolForge 作为指纹-结构解码器，并采用先验调整阈值策略，在 MassSpecGym 基准上实现了从质谱从头生成分子结构的十倍性能提升（top-1 准确率从 2.3% 提升至 31%）。

## 研究背景与动机

质谱（Mass Spectrometry）是分析化学中用于小分子结构表征的基础技术，但从质谱解析未知化合物的结构仍然是一个巨大瓶颈，需要大量领域知识和人工工作。自动化的"从头分子生成"（de novo molecule generation）旨在从质谱直接推断分子结构。

现有的两阶段流水线方法（质谱→分子指纹→分子结构）存在关键问题：编码器对每一位的不确定性会在解码器中产生复合误差，导致性能低下。先前最优方法 DiffMS 使用扩散模型作为解码器，但 top-1 准确率仅 2.3%。

本文的核心动机是：选择正确的解码器可以有效缓解编码器的弱点。MolForge 只使用指纹中的 on-bits（值为 1 的位），这意味着它专注于已存在的子结构，降低了假阳性位带来的噪声影响。

## 方法详解

### 整体框架

采用经典的两阶段流水线：
1. **编码阶段**：使用预训练的 MIST 模型将质谱编码为 4096 位的分子指纹（概率值）
2. **阈值处理**：对指纹概率进行阈值化，将连续概率转换为二值
3. **解码阶段**：使用 MolForge（基于 Transformer 的自回归模型）将指纹的 on-bit 索引解码为 SMILES 字符串

### 关键设计

**MIST 编码器**：采用 DiffMS 论文中预训练的 MIST 模型，在 MassSpecGym 上训练。每个质谱峰通过化学公式标注，然后经过化学公式 Transformer 编码，最终输出预测指纹。

**MolForge 解码器**：
- 自回归 Transformer 模型，输入为指纹 on-bit 的索引，输出为 SMILES token 序列
- 编码器-解码器架构：编码器处理 on-bit 索引映射，解码器生成 SMILES token
- 使用 beam search 策略生成 top-k 候选分子

**阈值策略**：
- 固定阈值 t=0.5：直接将概率 ≥0.5 的位设为 1
- 先验调整阈值 t=0.172：计算训练集中 on-bit 的比例（1.09%），选择使测试集阈值化后 on-bit 比例匹配训练集先验的阈值，无数据泄漏

### 训练策略

- MIST 使用 DiffMS 提供的预训练权重，无需额外训练
- MolForge 在约 280 万化合物的组合数据集上训练（DSSTox + HMDB + COCONUT + MOSES），使用 Morgan 指纹（半径 2，4096 位）
- 训练超参：学习率 5e-4，batch size 128，训练 6 个 epoch，Nvidia A40 GPU 约 3 天
- 推理时 beam size = 10

## 实验关键数据

### 主实验：从头分子生成性能

| 模型 | Top-1 Acc ↑ | Top-1 MCES ↓ | Top-1 Tanimoto ↑ | Top-10 Acc ↑ | Top-10 MCES ↓ | Top-10 Tanimoto ↑ |
|------|------------|--------------|------------------|-------------|---------------|-------------------|
| MADGEN | 1.31% | 27.47 | 0.20 | 1.54% | 16.84 | 0.26 |
| DiffMS | 2.30% | 18.45 | 0.28 | 4.25% | 14.73 | 0.39 |
| MIST+MolForge (t=0.5) | 28.27% | 14.72 | 0.64 | 36.11% | 10.69 | 0.70 |
| MIST+MolForge (t=0.172)* | **30.97%** | **12.38** | **0.68** | **40.04%** | **8.63** | **0.74** |

### MolForge 解码性能消融

| 训练数据集 | 输入指纹 | Top-1 Acc ↑ | Top-1 Tanimoto ↑ | Top-10 Acc ↑ | Top-10 Tanimoto ↑ |
|-----------|---------|------------|------------------|-------------|-------------------|
| 组合 DiffMS (~3M) | MIST 预测 | 30.97% | 0.68 | 40.04% | 0.74 |
| 组合 DiffMS (~3M) | Ground truth | 46.00% | 0.89 | 59.28% | 0.93 |
| MassSpecGym (~17k) | Ground truth | 0.00% | 0.18 | 0.00% | 0.19 |

### 关键发现

1. **解码器选择至关重要**：仅更换解码器为 MolForge，准确率提升约 10 倍（2.3% → 31%）
2. **数据规模效应显著**：MolForge 在 17k 数据上完全失败（0% 准确率），在 3M 数据上达到 31% top-1 准确率，符合数据缩放定律
3. **阈值策略有效**：先验调整阈值（0.172）比固定阈值（0.5）提升约 3 个百分点
4. **编码器是瓶颈**：使用 ground truth 指纹时 top-10 准确率达 59%，说明指纹预测质量是主要限制因素
5. **MolForge 的纠错能力**：36% 的测试样本中，MolForge 生成的结构比输入指纹更接近 ground truth
6. **峰标注重要**：移除质谱峰的化学公式标注后，Tanimoto 相似度从 0.731 降至 0.627

## 亮点与洞察

- **简单但有效的方法**：没有设计新模型，而是通过巧妙组合现有模型（MIST + MolForge）获得了数量级的提升，体现了"选对工具比造新工具更重要"
- **on-bit 索引输入设计**：MolForge 只接收 on-bit 索引而非完整指纹向量，天然适合处理稀疏、有噪声的预测指纹
- **先验调整阈值无数据泄漏**：巧妙利用训练集统计信息调整阈值，无需访问测试集标签
- **揭示了明确的改进方向**：指纹预测（编码器）是主要瓶颈，为后续研究指明了方向

## 局限与展望

- 编码器（MIST）是整个流水线的瓶颈，指纹预测质量直接限制最终性能
- 依赖质谱峰的化学公式标注，自动标注程序的质量会影响结果
- 先验调整阈值假设训练集和测试集的 on-bit 分布相似，可能在分布偏移严重时失效
- 仅在 MassSpecGym 数据集上评估，泛化性有待验证
- MolForge 训练耗时约 3 天，扩展到更大数据集的可行性未探讨

## 相关工作与启发

- **DiffMS**：先前 SOTA，使用扩散模型解码指纹到分子图，但性能有限（2.3%）
- **CSI:FingerID**：使用碎片树生成指纹，但指纹类型与 Morgan 指纹不同，可扩展性差
- **MSNovelist**：使用 RNN+LSTM 解码指纹，MolForge 的 Transformer 架构在数据扩展性上更优
- **启发**：在复杂流水线中，瓶颈分析和组件优化往往比端到端重新设计更有效

## 评分

- **新颖性**: ⭐⭐⭐ — 方法本身无架构创新，但组合方式和阈值策略有洞察
- **有效性**: ⭐⭐⭐⭐⭐ — 10 倍提升，结果非常令人信服
- **可复现性**: ⭐⭐⭐⭐⭐ — 所有组件开源，实验细节完整
- **影响力**: ⭐⭐⭐⭐ — 为质谱分子生成建立了强基线，明确了未来方向

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[CVPR 2025\] Towards All-in-One Medical Image Re-Identification](../../CVPR2025/medical_imaging/towards_all-in-one_medical_image_re-identification.md)
- [\[CVPR 2025\] EchoONE: Segmenting Multiple Echocardiography Planes in One Model](../../CVPR2025/medical_imaging/echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](../../ICCV2025/medical_imaging/pvchat_personalized_video_chat_with_one-shot_learning.md)

<!-- RELATED:END -->
