---
title: >-
  [论文解读] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs
description: >-
  [ACL 2025][跨语言迁移] 通过大规模分析 1000+ 语言对（35 种语言、1190 个方向）发现 LLM **中间层**具有最强跨语言语义对齐潜力，提出在任务微调中交替优化中间层对比对齐损失，在槽填充（F1 +1.5）、机器翻译（COMET +1.1）和 JSON 生成三大任务上显著提升跨语言迁移，且对未见语言和不同域数据均有效；分别训练的对齐与任务 LoRA 模块可通过权重平均合并使用。
tags:
  - ACL 2025
  - 跨语言迁移
  - middle-layer alignment
  - 对比学习
  - 表示对齐
  - low-resource languages
---

# Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs

**会议**: ACL 2025  
**arXiv**: [2502.14830](https://arxiv.org/abs/2502.14830)  
**代码**: https://github.com/dannigt/mid-align  
**领域**: LLM对齐 / 多语言  
**关键词**: 跨语言迁移, middle-layer alignment, 对比学习, 表示对齐, low-resource languages

## 一句话总结

通过大规模分析 1000+ 语言对（35 种语言、1190 个方向）发现 LLM **中间层**具有最强跨语言语义对齐潜力，提出在任务微调中交替优化中间层对比对齐损失，在槽填充（F1 +1.5）、机器翻译（COMET +1.1）和 JSON 生成三大任务上显著提升跨语言迁移，且对未见语言和不同域数据均有效；分别训练的对齐与任务 LoRA 模块可通过权重平均合并使用。

---

## 研究背景与动机

- **领域现状**：Decoder-only LLM 通过 SFT 在特定语言的特定任务上表现出色，但将这种能力扩展到多语言（尤其是低资源语言）仍然困难——微调数据很少覆盖 LLM 支持的所有语言，跨语言迁移至关重要。
- **核心矛盾**：先前跨语言对齐方法主要针对 encoder-only 或 encoder-decoder 模型（可在 encoder 输出端对齐），decoder-only LLM 没有明确的输入/输出表示边界，**在哪一层、如何对齐**是一个开放问题。
- **现有方案不足**：(1) 任务微调（包括多语言微调）保持但**不增强**跨语言语义对齐（Figure 3 实验验证），说明纯 SFT 不够；(2) 已有工作仅关注分类任务的迁移，生成任务（可变长度输出）更具挑战；(3) 很多方法需要每种目标语言的单语数据做 LM 适配，成本高。
- **本文切入角度**：对 Llama 3-8B 和 Qwen 2.5-7B 在 FLoRes-200 数据集上做大规模跨语言检索分析（35 语言、1190 方向），**数据驱动地**发现中间层（~16 层）的翻译检索准确率最高，且与下游跨语言迁移性能呈强正相关（p<0.01），由此提出在中间层施加显式对比对齐损失。
- **核心 idea**：在 LLM 中间层加入对比式跨语言对齐目标，与任务损失交替优化，增强跨语言迁移。

---

## 方法详解

### 整体框架

训练过程分为两个交替执行的目标：(1) **任务目标**——标准因果语言建模的交叉熵损失；(2) **对齐目标**——在中间层对平行翻译句对施加对比损失。每个 training step 只优化其中一个目标，避免手动平衡权重和梯度冲突。使用 LoRA (rank=8) 参数高效微调，基于 Llama 3-8B-Instruct 和 Qwen 2.5-7B-Instruct。

### 关键设计

1. **跨语言对齐诊断分析（Translation Retrieval Probing）**
    - **目标**：量化 LLM 各层的跨语言语义对齐程度，找到最优对齐层
    - **做法**：在 FLoRes-200 上提取 35 种语言 × 每层的 hidden states → mean pooling 得到句向量 → 用 ratio-based margin similarity 做翻译检索，覆盖全部 N(N-1)=1190 个语言方向
    - **核心发现**：中间层（Llama 第 16 层、Qwen 类似位置）检索准确率最高，底层和顶层较弱；低资源语言对齐程度不到整体平均的一半；中间层检索准确率与下游迁移 F1 呈显著正相关（p<0.01）
    - **设计意义**：为后续对齐层的选择提供了可靠的实证基础

2. **中间层对比对齐损失（Mid-Layer Contrastive Alignment）**
    - **目标**：在中间层显式拉近平行翻译对的表示、推远非翻译对
    - **做法**：对 batch 内 n 对平行句，提取第 i 层（中间层）mean-pooled hidden states，用 InfoNCE 对比损失：$\mathcal{L}_{\text{align}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_s^i, \mathbf{h}_t^i))}{\sum_{v \in \mathcal{B}} \exp(\text{sim}(\mathbf{h}_s^i, \mathbf{h}_v^i))}$，sim 为 cosine similarity，可选温度参数 τ
    - **对齐数据**：使用 Tatoeba 或任务数据的平行句对，低资源语言仅需几百句即可；对齐数据按语言重采样至近似均匀分布
    - **设计意义**：与任务损失交替优化，不修改模型架构，训练开销约为标准 SFT 的两倍但收益显著

3. **模块化 LoRA 合并（Post-hoc Module Merging）**
    - **目标**：使已有任务模型获得跨语言能力而无需重训
    - **做法**：分别训练任务 LoRA 适配器和对齐 LoRA 适配器，通过加权平均合并（权重在 dev set 上调优）
    - **效果**：合并后的性能接近联合训练（槽填充 F1 +1.1 vs 联合 +1.5，翻译 COMET +0.6 vs 联合 +1.1），且增益在各语言间分布更均匀
    - **设计意义**：对齐能力与任务能力解耦，新语言适配或能力增强无需获取原始任务训练数据

### 训练细节

| 配置项 | 设置 |
|--------|------|
| 基座模型 | Llama 3-8B-Instruct / Qwen 2.5-7B-Instruct |
| 参数高效微调 | LoRA rank=8，覆盖所有 attention 和线性投影层 |
| 有效 batch size | 128（任务 & 对齐均是） |
| 对比学习 mini-batch | 32 对平行句 |
| 对齐层位置 | 中间层（Llama 第 16 层 / 32 层总共） |
| 对齐数据量 | 低资源语言仅需几百句平行数据 |
| 对齐数据采样 | 多语言重采样至近似均匀分布 |

---

## 实验关键数据

### 主实验结果

| 任务 & 指标 | 模型 | SFT 基线 | + 中间层对齐 | 提升 |
|-------------|------|----------|-------------|------|
| 槽填充 监督 (5 语言) F1 | Llama 3 | 76.6 | **77.0** | +0.4 |
| 槽填充 迁移 (15 语言) F1 | Llama 3 | 60.2 | **61.7** | +1.5 |
| 槽填充 对齐语言 F1 | Llama 3 | 51.7 | **55.5** | +3.8 |
| 机器翻译 迁移→En BLEU | Llama 3 | 31.8 | **32.3** | +0.5 |
| 机器翻译 En→迁移 COMET | Llama 3 | 79.6 | **80.7** | +1.1 |
| 检索准确率 (20 语言平均) | Llama 3 | 39.4% | **73.2%** | +33.8 |
| 槽填充 迁移 F1 | Qwen 2.5 | 53.5 | **55.3** | +1.8 |

### 消融与分析

| 分析维度 | 核心发现 |
|----------|----------|
| **对齐层位置** | 中间层 (16) 最优且增益跨语言最均匀；底层 (8) 严重损害性能；顶层 (32) 可行但增益跨语言不均衡 (SD↑) |
| **对齐语言资源水平** | 低资源组增益最大 (+3.8 F1)，高资源组增益最小 (+0.7 F1)——初始对齐弱的语言受益最多 |
| **未见语言泛化** | 非对齐语言平均仍有 +0.4 F1 提升，说明方法增强的是通用迁移能力 |
| **大规模对齐** | 19 语言→En 对齐 (+1.9 F1) > 5 语言→En (+1.5)；多向对齐不额外提升，En 对齐隐含多向效果 |
| **域泛化** | Tatoeba / IWSLT 域数据对齐仍有效（检索准确率 71.9% / 68.5% vs oracle 77.7%） |
| **模块合并** | 分别训练后合并 ≈ 联合训练效果（槽填充 +1.1 vs +1.5，翻译 +0.6 vs +1.1） |
| **长序列任务** | JSON 生成中对齐语言 +1.0 F1，但监督集（含中文）下降 1.0，句级对齐与长序列存在冲突 |
| **非拉丁文字** | 非拉丁文字语言增益仅 +0.5 F1（vs 整体 +1.5），受限于分词质量影响 mean pooling |

---

## 亮点与洞察

- **大规模实证驱动**：1190 个语言方向的检索分析为"中间层最优"提供了强有力的统计支撑，而非凭经验猜测
- **交替优化策略简洁高效**：不修改模型架构、不手动调损失权重，实现了跨语言对齐与任务学习的解耦
- **数据需求极低**：低资源语言仅需几百句平行数据即可获得显著迁移提升，实用性强
- **模块化设计**：对齐和任务 LoRA 可独立训练再合并，面向工程部署友好——新语言到来时只需训练轻量对齐模块
- **中间层对齐的"辐射效应"**：在第 16 层施加对比损失后，前面多层的对齐也随之增强（Figure 4），顶层/底层对齐则无此效果

## 局限性与未来方向

- 实验限于 7-8B 模型，更大模型的最优对齐层位置可能不同
- 非拉丁文字语言增益有限，根因在分词质量 → 需要探索更优的 pooling 机制（attention pooling 初步实验未成功）
- 句级 mean pooling 对齐与长序列任务存在冲突（JSON 生成中中文下降 2.2 F1）
- 交替优化使训练计算量翻倍；可通过模块合并方案缓解
- 多层同时对齐效果因任务而异，最优多层策略仍需探索

---

## 总体评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 新颖性 | 7 | 对比对齐本身不新，核心贡献在于"中间层最优"的系统性发现及将其应用于 decoder-only LLM |
| 实用性 | 8 | 数据需求低（几百句平行数据）、即插即用的模块合并设计，工程落地门槛低 |
| 实验充分度 | 9 | 3 个任务 × 2 个模型 × 多种消融（层位置/语言资源/域泛化/模块合并），分析全面 |
| 写作质量 | 8 | 动机清晰、实验编排合理、分析系统化，图表丰富 |

## 相关工作与启发
- **vs mBERT/XLM-R 的跨语言方法**：之前方法针对 encoder-only 模型的分类任务，本文首次系统研究 decoder-only LLM 的生成任务跨语言迁移
- **vs 简单翻译数据增强**：翻译所有训练数据成本高且可能引入翻译错误，中间层对齐更高效
- 对 LLM 多语言能力从何而来、存储在哪一层有重要洞察

## 评分
- 新颖性: ⭐⭐⭐⭐ 中间层是最佳跨语言对齐位置的发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 1000+ 语言对 + 3 类任务 + 模块化验证
- 写作质量: ⭐⭐⭐⭐ 分析系统，结论清晰
- 价值: ⭐⭐⭐⭐ 对多语言 LLM 部署有实际意义

<!-- RELATED:START -->

## 相关论文

- [Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)
- [Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)

<!-- RELATED:END -->
