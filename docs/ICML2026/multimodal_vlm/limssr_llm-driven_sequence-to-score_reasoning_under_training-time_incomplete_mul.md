---
title: >-
  [论文解读] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations
description: >-
  [ICML 2026][多模态VLM][多模态] 作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉…
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "多模态"
  - "LLM Reasoning"
  - "Action Quality Assessment"
  - "Mask-Aware Fusion"
  - "Token-level Regularization"
---

# LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations

**会议**: ICML 2026  
**arXiv**: [2605.00434](https://arxiv.org/abs/2605.00434)  
**代码**: https://github.com/XuHuangbiao/LIMSSR  
**领域**: 多模态 VLM / 不完整多模态学习 / 动作质量评估  
**关键词**: Incomplete Multimodal Learning、LLM Reasoning、Action Quality Assessment、Mask-Aware Fusion、Token-level Regularization

## 一句话总结
作者把"训练阶段就缺模态"的多模态动作质量评估重新建模成"基于 LLM 的条件序列到分数推理"问题，用 prompt + 特殊 token 让 LLM 在没有完整数据监督的情况下补全缺失语义，再配合掩码感知的双路融合抑制幻觉，在三个 AQA 数据集上全面超越依赖完整训练数据的 SOTA。

## 研究背景与动机

**领域现状**：现实场景里多模态数据常常缺模态——传感器故障、隐私脱敏、采集成本都会让 video/audio/flow 等数据残缺。学术界研究的 Incomplete Multimodal Learning（IML）主要分两条线：(a) reconstruction-based（ActionMAE、IMDer、GAIN、DMVG）直接重建缺失模态特征；(b) distillation/prior-based（CorrKD、MoMKE、MCMoE）用完整模态作为教师做蒸馏或先验。

**现有痛点**：这两类方法都隐含一个"上帝视角"假设——训练时必须有完整模态作为重建目标或蒸馏教师。但真实数据采集就缺，比如某些受试者从未录过音频。一旦训练数据本身就缺，重建无 GT、蒸馏无教师，整个 IML 框架就坍塌。

**核心矛盾**：当训练阶段就缺模态时，缺失的语义如何"凭空"想象出来？传统重建-蒸馏路线需要"完整-不完整"配对，但配对本身不存在；而单纯填零会让模型把"缺失"当噪声学，导致主任务掉点；需要一种不依赖配对监督就能"推断"缺失语义的机制。

**本文目标**：(i) 形式化"训练时不完整观测"这个更现实的设定；(ii) 提出一个不依赖完整训练数据就能推断缺失语义的框架；(iii) 在长视频 Action Quality Assessment（AQA）这个高度依赖多模态的任务上验证。

**切入角度**：作者注意到 LLM 不仅是序列模型，还自带海量世界知识和推理能力——给定可观测模态 + 缺失结构的描述，LLM 应该能像"完形填空"一样推出缺失部分的语义表示，而无需 pixel-level 重建。

**核心 idea**：把不完整多模态学习重新表述为"条件序列推理"——用 prompt 描述任务和缺失状态，用 missing token 占位、fusion token 收集，让 LLM 在不可见缺失模态的条件下推断 latent semantic，再用 mask-aware gating 校准推理的不确定性。

## 方法详解

### 整体框架
对样本 $(\mathbf{X} \odot \boldsymbol{m}, \boldsymbol{m}, y)$（$\boldsymbol{m}\in\{0,1\}^M$ 是缺失掩码），LIMSSR 走三步：(1) Context Construction $\Phi_{in}$ 把指令 prompt、可见模态特征 $\tilde{\mathbf{X}}^m$、missing token 占位序列、fusion token 拼成统一 embedding $\mathbf{Z}_{in}$；(2) LLM 推理 $\mathbf{H}_{out} = \mathrm{LLM}(\mathbf{Z}_{in})$ 同时完成缺失语义推断和多模态融合；(3) Mask-Aware Dual-Path Aggregation $\Psi_{agg}$ 把高层语义路径和底层跨模态路径用掩码加权融合，输出动作质量分 $\hat{y}$。模态侧用冻结的 VST/AST/I3D 提 video/audio/flow 特征，经 2 层 conv 投影到 LLM 输入空间。

### 关键设计

1. **Prompt-Guided Context-Aware Modality Imputation (PCMI)**:

    - 功能：把缺失模态从"零向量"提升为"待推断的潜变量"，让 LLM 把缺失的位置当成可填空的 token 来推理。
    - 核心思路：每个模态 $m$ 都用一对边界 token `<m_start>, <m_end>` 包起来。对可见模态，里面放 $\tilde{\mathbf{X}}^m$；对缺失模态，里面放 $T$ 个重复的可学习 `<missing_m>` 嵌入。再设计一段任务 prompt 显式描述可见与缺失模态："Given the available {avail} features... The {miss} modality is missing. Based on the available modalities, please infer and reconstruct the useful latent representations for the missing {miss} modalities at the designated positions"。LLM 输出后，从 missing token 位置抽取隐藏态 $\mathbf{H}_{miss}^m = \mathrm{LLM}(\mathbf{Z}_{in})|_{\text{positions of }\mathbf{E}_{miss}^m}$ 作为推断的缺失表示。
    - 设计动机：传统零填充让缺失信号在 attention 中被"埋没"；MissRAG/TAMML 等用 RAG 或文本桥接，需要额外检索库或预训练对齐。PCMI 把缺失结构直接编入序列，使得 LLM 的 next-token 推理机制天然适配——"猜下一个 token"和"推断 missing latent"在数学形式上是同一件事。

2. **LLM-Driven Multidimensional Representation Fusion (LMRF)**:

    - 功能：在不破坏 LLM 输出空间的前提下，把跨模态信息蒸馏到 $K$ 个 fusion slot 中，得到一个紧凑的任务相关表示。
    - 核心思路：在 prompt 末尾追加 $K$ 个特殊 token `<emb_dim_1>, ..., <emb_dim_K>` 作为"信息槽"，再在 prompt 中显式让 LLM "integrate and enhance all multimodal features for action quality assessment. Output the fused multi-dimensional feature representations at the designated feature dimension positions"。LLM 最后一层在这些位置的输出 $\mathbf{H}_{fusion} = \{\boldsymbol{h}_1, \dots, \boldsymbol{h}_K\}$ 被认为分别承载不同评价维度（如难度、执行、艺术性）。再用可学习角色权重 $\boldsymbol{w}_{role}$ 计算 $\boldsymbol{z}_{main} = \sum_k \mathrm{Softmax}(\boldsymbol{w}_{role})_k \cdot \boldsymbol{h}_k$ 作为主融合向量。
    - 设计动机：直接对 LLM 输出做 mean-pooling 会破坏长序列生成能力，作者借鉴 BERT 的 `[CLS]` 思路但推广到多维度，让 LLM 自己学着"把不同方面的信息塞进不同 slot"，比 pooling 更结构化、比 attention head 更对人类解释友好。

3. **Mask-Aware Dual-Path Aggregation (MDA)**:

    - 功能：用 LLM 推理路径处理高层语义、用 cross-modal attention 路径处理底层特征，并根据缺失掩码动态校准两条路的可信度，避免严重缺失下 LLM 幻觉。
    - 核心思路：Path 1（不确定性校准推理）—— 计算门控 $\boldsymbol{g} = \sigma(\mathrm{MLP}_{gate}([\boldsymbol{z}_{main}, \boldsymbol{m}]))$ 和残差 $\boldsymbol{\delta} = \mathrm{MLP}_{res}([\boldsymbol{z}_{main}, \boldsymbol{m}])$，得到精修表示 $\tilde{\boldsymbol{z}}_{main} = \boldsymbol{z}_{main} + \boldsymbol{g}\odot \boldsymbol{\delta}$。Path 2（跨模态模式恢复）—— 把每个模态对应位置的 LLM 隐藏态做 temporal pooling 得 $\boldsymbol{h}_v, \boldsymbol{h}_a, \boldsymbol{h}_f$，stack 后做 self-attention 得 $\mathbf{Z}_{attn}$；再用 $\alpha_{m_j} = \boldsymbol{m}_j \cdot 1 + (1-\boldsymbol{m}_j)\cdot \gamma_{m_j}$ 按可用性加权（$\gamma_m = \sigma(\lambda_m)$ 是模态级可学习置信度），最终 $\boldsymbol{z}_{aux} = \sum_m \alpha_m (\boldsymbol{z}_{attn}^m \odot \mathcal{G}(\mathbf{H}_{stack})^m)$。两路融合输出最终分数。
    - 设计动机：单纯靠 LLM 推理在缺失严重时容易幻觉；单纯靠统计聚合又缺乏高层语义。掩码感知地把两路"按需混合"，相当于给模型一个"我现在到底信不信我自己"的元认知能力，对极端缺失情况尤其重要。

### 损失函数 / 训练策略
除了主任务回归 loss，作者引入：(1) Consistency Learning 约束两路输出一致，逼推理路径与统计路径相互校验；(2) Token-Level Metric Regularization 让不同 fusion token 学习不同特征维度（避免 collapse），具体是用 token 之间的相似度矩阵 + 正则项最大化非对角元素的距离；(3) LLM 主干部分可选 LoRA 微调，避免全量训练。

## 实验关键数据

### 主实验（FS1000，7-class，Spearman ↑ / MSE ↓，T-Miss 表示训练时也缺模态）

| 方法 | T-Miss | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|------|--------|-------|-------|------|------|---------|---------|
| ActionMAE | ✗ | 0.775/24.66 | 0.766/64.13 | 0.761/50.64 | 0.458/41.66 | 0.651/38.18 | 0.809/17.96 |
| GCNet | ✗ | 0.730/25.56 | 0.740/23.86 | 0.696/26.67 | 0.442/39.40 | 0.610/28.62 | 0.764/21.82 |
| MoMKE | ✗ | 0.798/18.86 | 0.805/23.88 | 0.785/37.96 | 0.499/27.53 | 0.668/26.08 | 0.819/16.85 |
| MCMoE | ✗ | 0.845/12.66 | 0.882/11.85 | 0.845/13.64 | 0.615/16.72 | 0.782/15.37 | 0.881/11.53 |
| **LIMSSR** | **✓** | **0.854/12.51** | **0.891/10.54** | **0.853/12.50** | **0.687/15.51** | **0.789/14.08** | **0.891/10.44** |

| Δ vs SOTA | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|-----------|-------|-------|------|------|---------|---------|
| ΔSpearman | ↑1.1% | ↑1.0% | ↑0.9% | ↑11.7% | ↑0.9% | ↑1.1% |
| ΔMSE | ↓1.2% | ↓11.1% | ↓8.4% | ↓7.2% | ↓8.4% | ↓9.5% |

注意：LIMSSR 是表里唯一在 T-Miss ✓ 下训练的模型，却在几乎所有缺失组合上都超过了所有 T-Miss ✗（即拿到完整训练数据）的方法。这是这篇论文最有力的"质性差异"证据。

### 消融实验

| 配置 | Average Spearman | 说明 |
|------|------------------|------|
| Full LIMSSR | 0.789 | 完整框架 |
| w/o PCMI（直接零填充缺失模态）| 显著下降 | 缺失语义无法被 LLM 推断 |
| w/o LMRF（用 mean pooling 代替 fusion token）| 下降 | 多维度信息坍缩 |
| w/o MDA Path 1（只走 cross-modal aggregation）| 下降 | 缺高层语义校准 |
| w/o MDA Path 2（只走 LLM 推理）| 下降 | 严重缺失下幻觉 |
| w/o Consistency Loss | 下降 | 两路缺乏相互校验 |
| w/o Token-Level Regularization | 下降 | fusion token 出现冗余 |

### 关键发现
- **训练时缺模态反而能赢过训练时不缺的对手**：这是论文最反直觉的结果——只有音频的极端缺失情况下，LIMSSR Spearman 比 SOTA 高 11.7%、MSE 低 7.2%，说明 LLM 的世界知识在补全缺失语义上确实有"质性"优势。
- **Path 1 + Path 2 互补**：单独任何一路都掉点；MDA 的掩码自适应融合是抗幻觉的关键。
- **Fusion token 数 $K$ 的甜点**：$K=3$ 最匹配 AQA 的"难度/执行/艺术性"三维结构，再多就开始过拟合。
- **音频模态最难补**：所有方法在 {a}-only 设置下都最差，因为音频对动作质量本身相关性最低，但 LIMSSR 仍然远超基线，说明 LLM 推断能力在低信息模态上的相对增益最大。

## 亮点与洞察
- **任务重构是最大贡献**：把 IML 从"重建/蒸馏"重构为"条件序列推理"，等于把一个监督受限问题变成了一个 LLM 擅长的 next-token 问题；这种把领域任务"reformulate as LM task"的思路可以迁移到很多 multimodal 残缺场景。
- **特殊 token 设计很优雅**：missing token 占位 + boundary token 分块 + fusion token 收尾，把 LLM 当成一个可编程的"语义计算器"，无需修改 LLM 架构就能完成定制功能。
- **掩码感知的双路自适应**：把"我对自己有多自信"也编码进网络，配合可学习的模态级置信度 $\gamma_m$，体现了对推理不确定性的工程化处理。
- **训练时缺数据反胜训练时完整数据**：这一发现给 IML 社区一个全新视角——LLM 的先验本身可能比 paired data 还宝贵，未来或许该重新审视"我们是不是被 paired data 的范式绑架了"。

## 局限与展望
- 主要在 AQA 任务上验证，迁移到情感识别、医学诊断等其他 IML 场景的有效性需要更多实验。
- LLM 推理引入显著计算成本，对实时性要求高的应用（如直播评分）可能不实用。
- 缺少对 LLM scale（7B/13B/70B）的系统性实验；只有 LLM 的世界知识与任务相关时才能发挥作用，对低资源语言或冷门动作类型未必有效。
- "幻觉"的定义和测量没有量化指标，只是用 MDA 间接缓解。
- 没有给出在文本-视觉-音频之外更多模态（如生理信号、深度图）上的扩展实验。

## 相关工作与启发
- **vs ActionMAE / IMDer / DMVG（reconstruction）**：这些方法依赖完整训练对，本文打破了这一约束。
- **vs MoMKE / MCMoE / CorrKD（distillation/prior）**：它们仍需要完整模态作为教师，LIMSSR 用 LLM 先验取而代之，本质上是"用通用世界知识替代领域内 paired supervision"。
- **vs MissRAG / TAMML（基于 LLM 的 IML）**：MissRAG 需要预先构建模态原型池，TAMML 把所有模态文本化损失精细信息；LIMSSR 直接让 LLM 在原 embedding 空间推理，更通用且无外部依赖。
- **vs Hedgehog / LoLCATs（LLM 端解决其他任务）**：思路上一致——挖掘 LLM 的非语言能力来解决领域问题；但 LIMSSR 关注的是"缺信息推断"而非"长序列建模"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 提出"训练时不完整观测"新设定，并用 LLM 序列推理重塑 IML 范式，问题和方法都很新。
- 实验充分度: ⭐⭐⭐⭐ 三个公开 AQA benchmark + 多种缺失组合 + 与 10+ 基线对比；但只在 AQA 上验证，跨任务推广性证据不足。
- 写作质量: ⭐⭐⭐⭐ 故事讲得清楚，图 1 三种范式对比一目了然；公式较多但都有具体语义。
- 价值: ⭐⭐⭐⭐ 给 IML 社区一个新范式，也给 LLM-as-tool 应用一个有说服力的非语言案例。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)
- [\[ICML 2026\] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)
- [\[ICML 2026\] Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](mitigating_perceptual_judgment_bias_in_multimodal_llm-as-a-judge_via_perceptual_.md)

</div>

<!-- RELATED:END -->
