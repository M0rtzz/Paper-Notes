---
title: >-
  [论文解读] FastDraft: How to Train Your Draft
description: >-
  [ACL 2025][speculative decoding] 提出 FastDraft，一套高效的 draft 模型预训练与对齐流程，可在24小时内用单节点8卡训练出约50M参数的 draft 模型，配合 Speculative Decoding 实现最高3倍内存带宽加速和2倍实际推理加速。
tags:
  - ACL 2025
  - speculative decoding
  - Draft Model
  - 知识蒸馏
  - LLM Inference
  - Edge Deployment
---

# FastDraft: How to Train Your Draft

**会议**: ACL 2025  
**arXiv**: [2411.11055](https://arxiv.org/abs/2411.11055)  
**代码**: 无（Intel Labs 内部实现）  
**领域**: NLP / LLM推理加速  
**关键词**: speculative decoding, Draft Model, knowledge distillation, LLM Inference, Edge Deployment

## 一句话总结

提出 FastDraft，一套高效的 draft 模型预训练与对齐流程，可在24小时内用单节点8卡训练出约50M参数的 draft 模型，配合 Speculative Decoding 实现最高3倍内存带宽加速和2倍实际推理加速。

## 研究背景与动机

Speculative Decoding (SD) 是一种无损的 LLM 推理加速技术：先用小模型（draft）快速生成候选 token 序列，再用大模型（target）并行验证。这种方法能在不牺牲生成质量的前提下获得2-3倍的加速。

然而，SD 的实际部署面临一个关键瓶颈：**缺乏高质量的 draft 模型**。原因在于 draft 必须与 target 共享相同的词表（vocabulary），而大多数流行的开源 LLM（如 Phi-3、Llama-3.1）并没有现成的、词表兼容的小模型可用。此外，与训练通用 LLM 不同，draft 模型的目标不是生成高质量回答，而是生成**容易被 target 接受的 token 序列**——这一训练目标此前缺乏系统性研究。

FastDraft 的核心动机就是填补这一空白：提出一套端到端的、资源友好的 draft 模型训练方法，使得任何给定的 target LLM 都能快速获得配套的 draft 模型。

## 方法详解

### 整体框架

FastDraft 采用三阶段流水线：

1. **预训练（Pre-Training, PT）**：在自然语言数据上从零训练 draft 模型
2. **继续预训练（Continued Pre-Training, CP）**：混合代码和自然语言数据进行继续训练
3. **微调对齐（Fine-Tuning, FT）**：使用 target 模型生成的合成数据进行序列级知识蒸馏

### 关键设计

1. **Draft 架构选择**：draft 唯一的硬性约束是必须输出与 target 相同词表上的概率分布。作者探索了两种规模：50M 和 120M 参数（分别比 Phi-3-mini 小约76倍和32倍）。架构直接沿用 target 的 Transformer 设计，仅缩小隐藏维度和层数。

2. **预训练数据规模实验**：通过在 {0.1, 0.5, 1, 2, 5, 10} BT（billion tokens）上系统消融，发现 5BT 是一个优秀的平衡点——更多数据在接受率上收益递减，甚至在某些任务上下降。这一发现打破了"越多数据越好"的直觉，表明 draft 模型的训练有其特殊规律。

3. **继续预训练策略**：直接在代码数据上训练无法保持自然语言能力。FastDraft 采用混合 CP：以文本预训练模型为起点，用 5BT 代码 + 2.5BT 文本混合数据继续训练。这比从零混合训练或反向 CP（代码初始化→文本 CP）都更优。

4. **序列级知识蒸馏（Sequence-level KD）**：使用 target 模型（Phi-3-mini / Llama-3.1-8B）对多个指令数据集生成回答，包括 Alpaca、OIG、Evol-Instruct 等。生成时使用多种温度（0.6, 0.8, 1.0）和贪心采样以增加多样性。还采用 Magpie 方法直接从 target 生成指令再补充回答。

5. **硬件感知架构设计**：在固定参数预算下，分析了不同深度和宽度对延迟的影响。发现增加层数对延迟影响最大，而增加隐藏维度影响较小。加速曲线呈倒U型——最优 draft 架构既不能过深也不能过浅。

### 损失函数 / 训练策略

微调阶段对比了两种知识蒸馏策略：

- **序列级 KD**：直接在 target 生成的序列上用交叉熵损失训练 draft，不使用 target 的 logits
- **Token 级 KD**：使用 target 的稀疏 logits（仅保留最显著的几个），计算 KL 散度或 TVD

实验发现：**序列级 KD 已经足够有效**，token 级 KD 的额外收益不明确。这是一个务实的结论——token 级 KD 需要预计算和存储 logits，开销大得多。

具体对比的损失函数组合包括：
- $\mathcal{L}_{CE}$（交叉熵）
- $\frac{1}{2}\mathcal{L}_{CE} + \frac{1}{2}\mathcal{L}_{KL}$
- $\frac{1}{2}\mathcal{L}_{CE} + \frac{1}{2}\mathcal{L}_{TVD}$
- $\mathcal{L}_{KL}$
- $\mathcal{L}_{TVD}$

## 实验关键数据

### 主实验（表格）

| 模型 | 阶段 | CNN-DM | TinyStories | Dolly | HumanEval |
|------|------|--------|-------------|-------|-----------|
| Phi3-mini 50M | PT | 0.311 | 0.277 | 0.245 | 0.229 |
| Phi3-mini 50M | PT→CP | 0.304 | 0.287 | 0.226 | 0.561 |
| Phi3-mini 50M | PT→CP→FT | **0.369** | **0.306** | **0.370** | **0.562** |
| Llama3.1 150M | PT | 0.280 | 0.227 | 0.247 | 0.248 |
| Llama3.1 150M | PT→CP | 0.280 | 0.235 | 0.273 | 0.606 |
| Llama3.1 150M | PT→CP→FT | **0.307** | **0.266** | **0.334** | **0.649** |

*接受率结果（γ=3，multinomial 采样 T=0.6）*

### 消融实验（表格）

| Draft | 数据量 | PPL | CNN-DM AR | TinyStories AR | Dolly AR |
|-------|--------|-----|-----------|----------------|----------|
| 50M | 2BT | 297.4 | 0.323 | 0.264 | 0.241 |
| 50M | 5BT | 256.6 | 0.311 | 0.277 | 0.245 |
| 50M | 10BT | 240.9 | 0.312 | 0.283 | 0.234 |
| 120M | 2BT | 199.6 | 0.362 | 0.297 | 0.284 |
| 120M | 5BT | 167.7 | 0.366 | 0.327 | 0.281 |
| 120M | 10BT | 147.4 | 0.351 | 0.331 | 0.251 |

*困惑度随数据量下降，但接受率不一定——Dolly 和 CNN-DM 在数据增多后反而下降*

### 关键发现

1. **使用 target 生成数据 vs 原始数据**微调，前者在所有任务上一致更好（表2），Dolly 任务提升约6个百分点
2. **代码继续预训练**对 HumanEval 带来巨大提升（~33个百分点），同时混合自然语言数据可保持甚至提升自然语言任务性能
3. **50M draft 在 Intel Core Ultra 上实测加速**：自然语言任务平均1.5x，代码补全任务平均2x
4. **MBSU 指标**：自然语言~2x，代码~3x
5. **完整训练仅需24小时**，单节点8张 Intel Gaudi 2 加速器

## 亮点与洞察

- **"够用就好"的数据哲学**：5BT 预训练数据 + 序列级 KD 就足以训练出优秀的 draft，不需要动辄万亿 token 的数据或复杂的蒸馏损失
- **draft 训练与 LLM 训练规律不同**：更多数据不一定提升接受率，因为 draft 的目标是"像 target"而非"更聪明"
- **极小的 draft 即可有效**：50M 参数（比 target 小76倍）就能达到有意义的加速，说明 Speculative Decoding 对 draft 质量的要求并不苛刻
- **端到端训练成本极低**：24小时内完成全部流程，这使得为任何新 LLM 快速适配 draft 成为现实

## 局限与展望

1. **仅在英语上验证**，不同语言的句法结构差异可能影响效果
2. **仅考虑单序列投机**，没有探索多序列/树状投机等更高级策略
3. **Draft 和 target 架构相同**，未探索异构架构组合的潜力
4. **评估任务较单一**：主要是摘要、文本补全和代码，缺乏对话、翻译等更多样的场景

## 相关工作与启发

- MEDUSA、EAGLE 等方法使用插件式预测头或目标模型隐表示，灵活性受限
- DistillSpec（Zhou et al., 2023）研究 TVD 损失与接受率的关系
- TVD++（Goel et al., 2024）在600B token 预训练的 draft 上微调
- FastDraft 强调独立 draft 的优势：更灵活、可与1000+兼容模型配对、无需额外 target 推理

## 评分

- **新颖性**: ⭐⭐⭐ 方法组件并不全新，但系统性地研究 draft 训练各维度（数据量、CP、KD 策略、硬件感知架构）是有价值的贡献
- **实验充分度**: ⭐⭐⭐⭐ 消融实验全面且扎实，覆盖数据量、CP 策略、KD 损失、架构宽深、硬件部署等维度
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富，实验设计有条理
- **价值**: ⭐⭐⭐⭐ 为工业界和边缘部署场景提供了一套切实可行且低成本的 draft 训练方案

<!-- RELATED:START -->

## 相关论文

- [Your Model is Overconfident, and Other Lies We Tell Ourselves](your_model_is_overconfident_and_other_lies_we_tell_ourselves.md)
- [How to Mitigate Overfitting in Weak-to-Strong Generalization?](how_to_mitigate_overfitting_in_weak-to-strong_generalization.md)
- [Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [From Lists to Emojis: How Format Bias Affects Model Alignment](from_lists_to_emojis_how_format_bias_affects_model_alignment.md)
- [The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)

<!-- RELATED:END -->
