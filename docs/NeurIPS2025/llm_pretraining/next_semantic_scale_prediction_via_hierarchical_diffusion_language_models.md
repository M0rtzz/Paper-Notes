---
title: >-
  [论文解读] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models
description: >-
  [NeurIPS 2025][预训练][离散扩散模型] 提出 HDLM（Hierarchical Diffusion Language Model），通过在 clean token 和 mask token 之间引入具有粗粒度语义的聚类 token 中间层级，实现"下一语义尺度预测"的离散扩散语言建模…
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "离散扩散模型"
  - "层级词汇表"
  - "语义尺度预测"
  - "语言建模"
  - "CTMC"
---

# Next Semantic Scale Prediction via Hierarchical Diffusion Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.08632](https://arxiv.org/abs/2510.08632)  
**代码**: [https://github.com/zhouc20/HDLM](https://github.com/zhouc20/HDLM)  
**领域**: Image Generation / Diffusion Language Model  
**关键词**: 离散扩散模型, 层级词汇表, 语义尺度预测, 语言建模, CTMC

## 一句话总结

提出 HDLM（Hierarchical Diffusion Language Model），通过在 clean token 和 mask token 之间引入具有粗粒度语义的聚类 token 中间层级，实现"下一语义尺度预测"的离散扩散语言建模，推导闭式 ELBO，在 OpenWebText 上困惑度一致优于 MDLM/GIDD，随机扰动后生成困惑度降低 62%。

## 研究背景与动机

自回归语言模型是当前 SOTA，但"下一 token 预测"范式无法修正已生成 token。离散扩散模型因渐进去噪和修正能力受关注，主要有两类：

**Masked 离散扩散**（MDLM）：所有被 mask 的 token 共享同一 embedding，缺乏丰富语义；已解码 token 无法自纠正

**Uniform 离散扩散**（SEDD）：均匀扰动为随机 token，语义不一致，性能弱于 masked 扩散

GIDD 统一框架结合 masked 和 uniform 噪声，但噪声 token 仍缺乏丰富语义，自纠正能力仅来自 uniform 噪声（实际损害性能）。

**核心矛盾**：masked 扩散缺乏自纠正且中间状态语义贫乏；uniform 扩散语义不一致且性能差。

**核心 idea**：受视觉 VAR "下一尺度预测"启发，在语言 token 中引入语义层级——word token 和 mask token 之间插入 cluster token（由预训练 embedding 聚类），前向过程逐级映射到更抽象的祖先，逆向过程逐级预测更精细的语义。

## 方法详解

### 整体框架

HDLM 基于 CTMC 框架。层级词汇表：word tokens - cluster tokens - mask token。前向过程 x - c - m；逆向：模型从抽象逐步恢复精细。

### 关键设计

1. **层级词汇表构建**:

    - 功能：在标准词汇和 mask 之间建立语义中间层
    - 核心思路：用预训练模型 embedding 做 K-means 聚类，建立满射映射。最优聚类数约为词汇量的平方根
    - 设计动机：cluster token 是"带高层语义的部分 mask"，比纯 mask 信息量更丰富，比随机 token 语义更一致，不确定性为自纠正提供空间

2. **层级 CTMC 过程**:

    - 功能：定义层级扩散的前向-逆向过程
    - 核心思路：边际分布为 Cat(z_t; alpha_t x + beta_{t,c} c(x) + beta_{t,m} m)。转移率矩阵为分块上三角结构（word to cluster to mask），mask 为吸收态
    - 设计动机：分块结构确保 token 只向更高层级转移，逆向通过贝叶斯后验实现层级解码

3. **闭式 ELBO（定理 3）**:

    - 功能：推导训练目标
    - 核心思路：ELBO 分解为两个 CE 损失：cluster token 做 cluster 内 word 分类，mask token 做 cluster 级分类。两个权重期望恒等于 1（定理 4）
    - 设计动机：自然形成由易到难的课程学习。MDLM 是 n=1 的特例

4. **随机扰动（xi < 1）**:

    - 功能：训练时以概率 1-xi 将 token 扰动到错误 cluster
    - 设计动机：训练模型从不准确上下文恢复正确 token，缓解 train-test gap。xi=0.8 时 Gen PPL 降 62%

### 实用技巧

- **Force transition 解码**：限制 cluster token 只解码为 cluster 内 word token
- **灵活权重截断**：裁剪极端权重稳定优化
- **Hard 训练模式**：将 cluster 级 CE 替换为 token 级 CE，性能略降，验证渐进去噪优势

## 实验关键数据

### 主实验

OpenWebText（DiT 架构，GPT-2 tokenizer，131B tokens）：

| 模型 | 参数量 | Valid PPL | Gen PPL |
|------|--------|-----------|----------|
| MDLM-small | 170M | 27.39 | 163.7 |
| GIDD+-small | 170M | 25.82 | 170.2 |
| **HDLM-small-64** | 170M | **23.36** | **144.2** |
| **HDLM-small-128** | 170M | **23.25** | 148.0 |
| **HDLM-base-128** | 425M | **19.22** | **139.9** |
| GPT-2 | 117M | 23.40 | - |

### 消融实验

聚类数实验：

| n | Valid PPL | Gen PPL | 说明 |
|---|-----------|----------|------|
| 1 (=MDLM) | 25.72 | 163.9 | 退化为 MDLM |
| 64 | **23.36** | **144.2** | 最佳区间 |
| 128 | 23.25 | 148.0 | Valid PPL 最佳 |
| 256 | 23.65 | 150.4 | 过多反而下降 |

随机扰动（HDLM-64）：

| xi | Valid PPL | Gen PPL | 说明 |
|----|-----------|----------|------|
| 1.0 | 23.36 | 144.2 | 标准 |
| 0.9 | 23.54 | **69.76** | Gen PPL 降 51% |
| 0.8 | 25.93 | **54.15** | Gen PPL 降 **62%** |

### 关键发现

- HDLM 一致优于 MDLM/GIDD，small 模型 Valid PPL 从 27.39/25.82 降至 23.25
- base 模型 Valid PPL 19.22 可匹配自回归模型
- 最优聚类数约 sqrt(|V|)，将生成分为两个等复杂度阶段
- 随机扰动是杀手锏：Gen PPL 从 144.2 暴降至 54.15
- MDLM 是 HDLM 的特例（n=1），理论和实验均验证
- Force transition 在 xi<1 时仍有效，上下文鲁棒性比就地纠错更重要

## 亮点与洞察

- "下一语义尺度预测"开创性地将视觉 VAR 多尺度思想迁移到语言领域
- 理论扎实：闭式 ELBO、权重不变性、MDLM 特例关系
- 随机扰动效果惊人（Gen PPL 降 62%），简单有效
- ELBO 的课程学习解释很自然：cluster token 做簇内分类、mask token 做簇级分类

## 局限与展望

- 实验规模较小（170M/425M），7B+ 下表现待验证
- 仅 1 层中间层级，多层级实验留作未来工作
- 静态聚类策略，可学习映射可能更好
- 仅在语言建模验证，下游任务效果未知

## 相关工作与启发

- 与 MDLM、GIDD 形成演进链：masked - masked+uniform - hierarchical
- 随机扰动缓解 train-test gap 与连续扩散中 noise perturbation 异曲同工
- 层级词汇表+课程学习可迁移到其他离散生成任务
- LLaDA、Dream 等大规模扩散语言模型表明 HDLM 有很大 scaling 潜力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](memory_mosaics_at_scale.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](../../ACL2026/llm_pretraining/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)

</div>

<!-- RELATED:END -->
