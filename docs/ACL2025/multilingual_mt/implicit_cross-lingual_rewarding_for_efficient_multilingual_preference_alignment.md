---
title: >-
  [论文解读] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment
description: >-
  [ACL 2025 (Findings)][多语言/翻译][多语言偏好对齐] 本文提出利用已对齐的英文 DPO 模型中的隐式奖励信号，通过跨语言指令-响应配对标注偏好关系，结合迭代 DPO 训练实现高效的多语言偏好对齐，在 X-AlpacaEval 上平均 Win Rate 提升 12.72%。 领域现状：DPO（Direc…
tags:
  - "ACL 2025 (Findings)"
  - "多语言/翻译"
  - "多语言偏好对齐"
  - "隐式奖励模型"
  - "跨语言迁移"
  - "DPO"
  - "迭代训练"
---

# Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment

**会议**: ACL 2025 (Findings)  
**arXiv**: [2503.04647](https://arxiv.org/abs/2503.04647)  
**代码**: [GitHub](https://github.com/ZNLP/Implicit-Cross-Lingual-Rewarding)  
**领域**: 对齐RLHF / 多语言  
**关键词**: 多语言偏好对齐, 隐式奖励模型, 跨语言迁移, DPO, 迭代训练

## 一句话总结

本文提出利用已对齐的英文 DPO 模型中的隐式奖励信号，通过跨语言指令-响应配对标注偏好关系，结合迭代 DPO 训练实现高效的多语言偏好对齐，在 X-AlpacaEval 上平均 Win Rate 提升 12.72%。

## 研究背景与动机

**领域现状**：DPO（Direct Preference Optimization）已经成为将 LLM 与人类偏好对齐的主流方法，在英文 LLM 对齐方面取得了显著进展，但多语言偏好对齐的进展受限于数据稀缺。

**现有痛点**：多语言偏好数据获取成本极高——需要对每种语言单独收集人类偏好标注，这在低资源语言中几乎不可行。现有的一些方案尝试通过翻译英文偏好数据到其他语言来解决，但翻译过程会引入噪声和偏差，导致偏好信号失真。

**核心矛盾**：一方面英文模型已经通过 DPO 学到了良好的偏好知识，另一方面将这些偏好知识迁移到其他语言时缺乏可靠的桥梁——直接翻译会扭曲奖励信号，而为每种语言重新收集数据成本过高。

**本文目标**：设计一种不依赖于翻译的跨语言偏好迁移方法，直接利用英文模型中已学到的偏好知识来指导多语言对齐。

**切入角度**：作者观察到 DPO 对齐后的模型和其参考模型之间的 logits 差异本身就编码了隐式奖励函数。这个隐式奖励模型可以直接对跨语言的响应进行评分——用英文指令来评估多语言响应，避免了翻译引入的失真。

**核心 idea**：从英文 DPO 模型中提取隐式奖励信号，构建跨语言指令-响应对，让隐式奖励模型在英文指令下评估多语言响应的偏好关系，再用这些标注数据进行迭代 DPO 训练。

## 方法详解

### 整体框架

方法分为三个阶段形成迭代循环：（1）从当前多语言模型生成多语言响应；（2）构建跨语言指令-响应对并用隐式奖励模型标注偏好；（3）在标注数据上进行 DPO 微调。框架以 Llama-3-Base-8B-SFT-DPO 作为起点，该模型已在英文偏好数据上完成 DPO 训练。

### 关键设计

1. **隐式奖励模型（Implicit Reward Model）**:

    - 功能：从已对齐的英文 DPO 模型和其参考模型中提取偏好评分信号
    - 核心思路：基于 DPO 的理论推导，DPO 训练后的策略模型 $\pi_\theta$ 和参考模型 $\pi_{\text{ref}}$ 之间的 log 概率差 $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 就是隐式奖励。这个奖励不需要额外训练奖励模型，直接从两个模型的 logits 计算得到
    - 设计动机：避免了训练单独奖励模型的开销，同时保留了英文偏好数据中学到的完整奖励信号

2. **跨语言偏好标注（Cross-Lingual Preference Annotation）**:

    - 功能：将英文偏好知识迁移到目标语言的偏好对
    - 核心思路：对于多语言提示 $x_l$（$l$ 为目标语言），找到对应的英文指令 $x_{\text{en}}$，用当前模型生成多对目标语言响应 $(y_l^1, y_l^2)$，然后用隐式奖励模型在英文指令下评估这些响应的偏好关系——即 $r(x_{\text{en}}, y_l^1) > r(x_{\text{en}}, y_l^2)$ 则 $y_l^1$ 被选为偏好响应
    - 设计动机：用英文指令评估多语言响应，这样偏好评分的锚点始终是英文（模型最擅长的语言），避免了目标语言指令可能带来的评估噪声

3. **迭代偏好迁移训练（Iterative Preference Transfer Training）**:

    - 功能：通过多轮迭代逐步提升多语言对齐质量
    - 核心思路：每轮用当前最优模型生成新的多语言响应，重新标注偏好对，再进行 DPO + NLL 联合训练。NLL 损失 $\mathcal{L}_{\text{NLL}} = -\log \pi_\theta(y_w | x)$ 作用于偏好响应，防止模型在 DPO 训练中偏离正确的输出分布
    - 设计动机：单轮训练受限于初始模型的响应质量，迭代训练可以逐步提升响应质量和偏好标注准确度，形成正反馈循环

### 损失函数 / 训练策略

总损失函数为 DPO 损失和 NLL 损失的加权组合：$\mathcal{L} = \mathcal{L}_{\text{DPO}} + \alpha \mathcal{L}_{\text{NLL}}$。其中 DPO 损失是标准的偏好优化损失，NLL 损失在偏好响应上计算负对数似然。训练过程进行两轮迭代（M0 → M1），每轮的参考模型更新为上一轮训练得到的模型。

## 实验关键数据

### 主实验

在 X-AlpacaEval 上评估，涵盖高资源（es, ru, de, fr）和低资源（bn, sw, th）语言：

| 模型 | 平均 Win Rate | 平均 LC Win Rate | 迭代次数 |
|------|-------------|----------------|---------|
| Llama-3-Base-8B-SFT | 5.36% | 5.78% | - |
| Llama-3-Base-8B-SFT-DPO (英文) | 11.14% | 11.96% | - |
| ICR-M0 (DPO) | 18.62% | 15.40% | 1 |
| ICR-M1 (DPO) | 23.86% | 17.93% | 2 |
| ICR-M0 (KTO) | 17.29% | 14.85% | 1 |
| ICR-M1 (KTO) | 21.44% | 16.72% | 2 |

两轮迭代后平均 Win Rate 提升 12.72%，LC Win Rate 提升 5.97%。

### 消融实验

| 配置 | 平均 Win Rate | 说明 |
|------|-------------|------|
| ICR-M1 完整模型 | 23.86% | 两轮 DPO 迭代 |
| 仅一轮 M0 | 18.62% | 不迭代，少 5.24% |
| 不加 NLL 损失 | ~20.5% | 掉约 3% |
| 每语言 1000 样本 | ~19.8% | 数据量减少性能下降 |
| 每语言 5000 样本 | ~22.1% | 接近完整性能 |
| 使用 t-1 参考模型 | ~21.5% | 参考模型策略影响性能 |

### 关键发现

- 迭代训练显著提升效果——M1 相比 M0 在所有语言上都有明显提升，说明迭代生成-标注-训练循环确实有效
- NLL 损失的加入对防止 DPO 训练中的退化至关重要
- 在低资源语言（bn, sw, th）上的提升幅度甚至超过高资源语言，表明隐式奖励迁移对低资源语言更为有效
- DPO 和 KTO 两种偏好优化方法都能从隐式跨语言奖励中获益，但 DPO 略优

## 亮点与洞察

- **隐式奖励的跨语言可迁移性**是本文最核心的洞察：DPO 模型中编码的偏好知识可以跨语言使用，不需要翻译数据也不需要多语言奖励模型。这说明偏好评估在某种程度上是语言无关的
- **免训练奖励模型**的设计非常精巧——利用 DPO 训练的副产品（策略模型和参考模型的 logits 差）作为奖励信号，零额外成本获取奖励评分能力
- 迭代训练的思路可以迁移到其他 "知识从强到弱迁移" 的场景：例如从大模型到小模型、从通用模型到领域模型的偏好迁移

## 局限与展望

- 方法依赖于英文指令和多语言响应的配对，如果目标语言的语言特性与英文差异极大（如日语的敬语系统），英文偏好评分可能无法准确反映目标语言的偏好
- 仅在 Llama-3-8B 上验证，对于更大规模模型或非 Llama 架构的泛化性未知
- 评估主要依赖 LLM-as-a-judge（GPT-4 评估），缺乏人类评估验证
- 可以探索将方法扩展到更多语言（如中文、阿拉伯语），以及在文化敏感话题上的偏好对齐

## 相关工作与启发

- **vs SimPO/ORPO**: 这些方法也不需要显式奖励模型，但仍需要目标语言的偏好数据；本文进一步去除了多语言偏好数据的需求
- **vs 翻译+对齐方法**: 直接翻译英文偏好数据到目标语言会引入翻译偏差，本文通过在英文指令空间中评估避免了这个问题
- **vs 多语言 RLHF**: 传统方法需要为每种语言训练奖励模型或收集偏好数据，本文的方法大幅降低了成本

## 评分

- 新颖性: ⭐⭐⭐⭐ 隐式奖励的跨语言迁移视角新颖，但迭代 DPO 框架较为常见
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种语言和消融维度，但缺少人类评估
- 写作质量: ⭐⭐⭐⭐ 思路清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为多语言对齐提供了低成本的实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment](modular_sentence_encoders.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)

</div>

<!-- RELATED:END -->
