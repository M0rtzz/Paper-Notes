---
title: >-
  [论文解读] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook
description: >-
  [ICLR 2026][图像tokenizer] 提出SemHiTok——通过语义引导层次codebook(SGHC)统一理解和生成的tokenizer：预训练语义codebook上建像素子codebook，结构和训练解耦(分阶段优化)避免联合训练的语义-像素冲突，LLaVA设定下离散tokenizer中理解和重建都SOTA。
tags:
  - ICLR 2026
  - 图像tokenizer
  - 层次codebook
  - 语义引导
  - 理解+生成统一
  - SGHC
  - MLLM
---

# SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook

**会议**: ICLR 2026  
**arXiv**: [2503.06764](https://arxiv.org/abs/2503.06764)  
**领域**: 统一多模态/图像tokenization  
**关键词**: 图像tokenizer, 层次codebook, 语义引导, 理解+生成统一, SGHC, MLLM

## 一句话总结
提出SemHiTok——通过语义引导层次codebook(SGHC)统一理解和生成的tokenizer：预训练语义codebook上建像素子codebook，结构和训练解耦(分阶段优化)避免联合训练的语义-像素冲突，LLaVA设定下离散tokenizer中理解和重建都SOTA。

## 研究背景与动机

**领域现状**：统一MLLM需同时支持理解(高层语义)和生成(低层像素)的tokenizer。

**现有痛点**：
   - (1) CLIP族→语义好丢像素; VQGAN族→保像素缺语义
   - (2) 联合训练(VILA-U混合loss→子最优; SDE encoder解耦但codebook混合)
   - (3) 双编码器(Janus)→token翻倍或词汇爆炸→不高效
   - (4) TokenFlow shared mapping但联合训练仍影响性能

**切入角度**：观察到同语义code的patches有相似像素→在每个语义code下建子codebook→结构+训练都解耦。

## 方法详解

### 整体框架
语义分支(VQKD对齐SigLIP)→固定C_sem；像素分支(ViT)→学C_pix。语义+像素token沿channel拼接→统一离散表示。

### 关键设计

1. **语义Codebook**：SigLIP→EMA VQ量化→cosine+L1蒸馏→训练后固定不再修改

2. **SGHC**：C_pix={C_pix^1,...,C_pix^K}(K语义码×m子码)→patch i先由C_sem得index k→选第k子codebook量化像素

3. **分阶段训练**：Stage 1训练语义(VQKD)固定→Stage 2训练像素(L1+perceptual+GAN)→两阶段不冲突

4. **统一MLLM集成**：展平h=i*m+j；Dual-MLP adapter分别投影语义/像素→拼接送LLM

### 训练策略
- SigLIP frozen; K语义码, m=8子码→总196,608; Qwen2.5-7B-Instruct base

## 实验关键数据

### 重建(Table 1, ImageNet-50k)

| 方法 | 类型 | Codebook | rFID↓ |
|------|------|----------|-------|
| LlamaGen | Only Recon | 16,384 | 2.19 |
| IBQ | Only Recon | 262,144 | 1.00 |
| VILA-U | Unified | 16,384 | 1.80 |
| TokenFlow | Unified | 32,768 | 1.37 |
| **SemHiTok** | Unified | 196,608 | **1.16** |
| **SemHiTok-384** | Unified | 196,608 | **0.66** |

### 理解(Table 2, LLaVA-v1.5)

| 模型 | 分辨率 | POPE | MME-P | SEED | GQA |
|------|--------|------|-------|------|-----|
| SigLIP(连续) | 256 | 83.8 | 1481 | 65.3 | 61.9 |
| VILA-U | 256 | 81.6 | 1312 | 56.9 | 55.3 |
| **SemHiTok** | 256 | **82.5** | **1356** | **62.9** | **60.3** |
| **SemHiTok-384** | 384 | **86.3** | **1466** | **64.1** | **62.3** |

### 关键发现
- 离散tokenizer中理解SOTA→接近甚至部分超越连续SigLIP
- rFID 1.16/0.66→统一tokenizer中重建SOTA级
- POPE 82.5 vs VILA-U 81.6(+0.9); SEED 62.9 vs 56.9(+6.0)
- 总codebook K*m=196K与LLM文本词汇量级相当(Qwen2 ~150K)→无膨胀

## 亮点与洞察
- **SGHC设计**：同语义→相似像素的观察→子codebook细化→简洁优雅
- **分阶段训练**：完全避免语义-像素冲突→更好trade-off→关键创新
- **无token膨胀**：展平后可控(196K)→兼容现有LLM词汇→无缝集成
- **非冲突扩展**：像素训练不影响已冻结语义codebook→理解能力不退化

## 局限性
- 主要验证256/384分辨率→更高分辨率扩展性未测
- 子codebook大小m=8固定→自适应m未探索
- 仅验证Qwen2.5-7B和Vicuna-7B→更大LLM待测
- 生成质量评估(MJHQ/GenEval)篇幅有限
- 语义codebook大小K的选择对性能的影响未充分消融
- SGHC的像素子空间可能在某些语义code下数据不足→导致子codebook欠拟合
- 训练策略中各loss权重(lambda1/2/3)的敏感性分析有限

### 统一MLLM实验补充
- 在理解和生成任务上均超越先前统一离散MLLM
- Und&Gen Discrete类别中多数benchmark SOTA
- 与部分连续tokenizer(Only Und.)性能可比

## 相关工作与启发
- VILA-U联合loss→子最优→SemHiTok分阶段解决
- TokenFlow shared mapping→但联合训练冲突→SemHiTok完全解耦
- VQKD语义codebook方法→SemHiTok在此上扩展像素层
- 启发：层次结构(语义→像素)可能是统一视觉tokenizer最佳范式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SGHC+分阶段训练首创
- 技术深度: ⭐⭐⭐⭐ 简洁有效，动机清晰
- 实验充分度: ⭐⭐⭐⭐ 重建+理解+生成覆盖
- 实用性: ⭐⭐⭐⭐⭐ 直接集成现有MLLM→统一理解+生成
- 综合: ⭐⭐⭐⭐⭐ 统一视觉tokenizer的优雅方案

<!-- RELATED:START -->

## 相关论文

- [Unsupervised Morphological Tree Tokenizer](../../ACL2025/llm_pretraining/unsupervised_morphological_tree_tokenizer.md)
- [Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [A Unified Framework for Heterogeneous Semi-supervised Learning](../../CVPR2025/llm_pretraining/a_unified_framework_for_heterogeneous_semi-supervised_learning.md)
- [ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)

<!-- RELATED:END -->
