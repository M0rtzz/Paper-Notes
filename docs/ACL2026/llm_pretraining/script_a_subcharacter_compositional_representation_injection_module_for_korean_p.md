---
title: >-
  [论文解读] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models
description: >-
  [ACL 2026][预训练][子字符组合表示] 本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。
tags:
  - "ACL 2026"
  - "预训练"
  - "子字符组合表示"
  - "韩语预训练模型"
  - "Hangul结构建模"
  - "形态音韵变化"
  - "即插即用模块"
---

# SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models

**会议**: ACL 2026 Findings  
**arXiv**: [2604.12377](https://arxiv.org/abs/2604.12377)  
**代码**: [GitHub](https://github.com/SungHo3268/SCRIPT)  
**领域**: LLM 预训练 / 韩语NLP  
**关键词**: 子字符组合表示, 韩语预训练模型, Hangul结构建模, 形态音韵变化, 即插即用模块

## 一句话总结

本文提出 SCRIPT，一个模型无关的即插即用模块，通过双通道策略将韩文 Hangul 的子字符（Jamo）组合知识注入现有子词级 PLM 的嵌入层，无需重新预训练即可在韩语 NLU/NLG 任务上获得一致提升，并使嵌入空间更好地捕捉语法规律和语义变化。

## 研究背景与动机

**领域现状**：当前主流韩语 PLM（包括 HyperCLOVA X、EXAONE 等先进 LLM）几乎都依赖子词分词（subword tokenization）。子词建模擅长从大规模语料中捕捉词汇语义，但其分词粒度不足以反映 Hangul 的内部组合结构。

**现有痛点**：(1) 韩语是形态丰富的粘着语，大量形态音韵变化发生在子字符（Jamo）级别——语料分析显示 92.75% 的形态修改是子字符级的；(2) 现有子词 PLM 对细粒度形态语法变化（如时态屈折、音韵同化）不敏感；(3) 少数基于 Jamo 的语言模型（如 KOMBO）虽然对形态变化鲁棒，但因语义表示弱和计算成本高，在下游任务上表现不佳。

**核心矛盾**：子词级建模擅长语义但忽略结构，子字符级建模捕捉结构但牺牲语义——两者优势互补但难以兼得。

**本文目标**：设计一个轻量级模块，将子字符组合知识注入现有子词级 PLM，在不修改架构、不额外预训练的情况下获得两者的优势。

**切入角度**：利用 Hangul 造字的三大基本原则（组合规则、空间排列、顺序规则）来指导子字符到子词的层次化压缩，而非使用通用的注意力或线性池化。

**核心 idea**：在 PLM 的嵌入层附加双通道模块——一路压缩子字符序列为结构感知的子词表示，另一路保留原始预训练嵌入，通过交叉注意力融合两者。

## 方法详解

### 整体框架

SCRIPT 附加在 PLM 的嵌入层。给定韩文输入，使用两条并行分词路径：(1) 子词分词器生成 PLM 原始子词序列；(2) 子字符分词器生成 Jamo 细粒度序列供 SCRIPT 处理。SCRIPT 将 Jamo 序列压缩为子词级表示，与 PLM 原始嵌入融合后送入 Transformer 层。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IN["韩文输入"] --> SUB["子词分词器<br/>→ PLM 原始嵌入 e_S"]
    IN --> JAMO["子字符分词器<br/>→ Jamo 三元组序列"]
    JAMO --> S1["子字符到字符的层次化压缩<br/>GRU 顺序编码 → 空间拼接 → 卷积 → 平均池化"]
    S1 --> CHAR["字符表示 h_C"]
    CHAR --> S2["字符到子词的压缩<br/>GRU + 子词边界池化 → h_S"]
    SUB --> FUSE["双通道融合<br/>交叉注意力 Q=e_S，KV=h_S"]
    S2 --> FUSE
    FUSE --> OUT["融合嵌入 e_F → Transformer 层"]
```

### 关键设计

**1. 子字符到字符的层次化压缩（Stage 1）：让神经网络照着 Hangul 的造字规则去拼字**

Jamo 三元组（初声、中声、终声）到字符的压缩如果用通用的注意力池化或线性池化，会把 Hangul 自带的组合结构先验冲掉——消融实验里这类通用方法比造字原则方法低 1.8–4.6%p。SCRIPT 因此把 Hangul 三大原则直接写进结构里：先用 GRU 按顺序规则（原则 3：初声→中声→终声）编码 Jamo 的书写次序；再按空间排列规则（原则 2）把初声+中声的融合表示与终声纵向拼接成 $\mathbf{h}_R = [\mathbf{h}_{I+V}; \mathbf{h}_F] \in \mathbb{R}^{2 \times N/3 \times D}$；随后用卷积层捕捉这些子字符之间的相对位置，最后平均池化得到字符表示 $\mathbf{h}_C$。这条流水线本质上是把"韩文字怎么拼出来"的语言学规则当成归纳偏置，让压缩过程保留住组合结构而不只是把符号揉成一团。

**2. 字符到子词的压缩（Stage 2）：在子词边界上对齐 PLM 的粒度**

字符表示要和 PLM 原有的子词序列对齐，但直接对字符表示做平均或求和会让训练变得不稳定。SCRIPT 的做法是再套一层 GRU 编码字符在子词内部的组合顺序，然后在每个子词边界处取最后一个字符表示作为该子词的代表：

$$\mathbf{h}_S = \text{Pooling}(\text{GRU}(\mathbf{h}_C)) \in \mathbb{R}^{N' \times D}.$$

先用 GRU 把组合顺序信息沉淀进序列、再选边界表示，比起一股脑求和能更稳地把字符级信息收束到与 PLM 一致的子词粒度上，为后续融合提供干净的对接点。

**3. 双通道融合（Fusion）：让 PLM 嵌入主动"挑"需要的结构信息**

SCRIPT 产出的是结构感知表示，PLM 原始嵌入则是语义丰富表示，二者要合并又不能互相稀释。SCRIPT 用一层交叉注意力，以 PLM 原始嵌入 $\mathbf{e}_S$ 作 Query、SCRIPT 输出 $\mathbf{h}_S$ 作 Key/Value：$\mathbf{e}_F = \text{CrossAttn}(Q=\mathbf{e}_S, KV=\mathbf{h}_S)$。消融显示这比简单求和或拼接更好，原因在于交叉注意力让 PLM 嵌入能逐位置、动态地决定吸收多少结构信息，而不是被一刀切地叠加，从而真正实现"语义为主、结构按需补充"的融合。

### 损失函数 / 训练策略

SCRIPT 仅在微调阶段附加训练，使用任务相关的标准损失函数。无需额外预训练，即插即用。基座模型包括 BERT_base、KoGPT2_base、KoGPT3-1.2B、EXAONE-2.4B。

## 实验关键数据

### 主实验

**韩语 NLU 任务（9项基准）**

| 模型 | KorNLI | KorSTS | NSMC | PAWS-X | KoBEST Avg. |
|------|--------|--------|------|--------|-------------|
| BERT_base | 75.85 | 76.72 | 88.96 | 72.38 | 64.22 |
| BERT_base + SCRIPT | 76.49 | 77.68 | 88.96 | 73.68 | 65.14 |
| KoGPT3-1.2B | 80.11 | 76.14 | 90.51 | 77.40 | 81.62 |
| KoGPT3-1.2B + SCRIPT | 80.39 | 79.60 | 90.53 | 79.95 | 82.17 |
| EXAONE-2.4B | 83.99 | 85.08 | 90.04 | 85.24 | 89.57 |
| EXAONE-2.4B + SCRIPT | **85.77** | **85.27** | **90.89** | **85.90** | **90.40** |

**韩语 NLG 任务（KoCommonGen）**

| 模型 | BLEU-4 | ROUGE-2 | ROUGE-L | METEOR |
|------|--------|---------|---------|--------|
| KoGPT2_base | 10.33 | 44.24 | 54.50 | 40.05 |
| KoGPT2_base + SCRIPT | 15.57 | 47.42 | 60.00 | 42.53 |
| EXAONE-2.4B | 28.41 | 62.25 | 64.84 | 54.84 |
| EXAONE-2.4B + SCRIPT | **31.80** | **71.03** | **72.16** | **61.27** |

### 消融实验

**SCRIPT 架构消融（KoBEST 平均，基于 KoGPT2_base）**

| 配置 | 平均准确率 |
|------|----------|
| SCRIPT (Jamo + Principles + CrossAttn) | **73.85** |
| Stroke 替代 Jamo | 72.36 |
| 用 Linear 压缩替代 Principles | 69.30 |
| 用 Attention 压缩替代 Principles | 72.01 |
| Summation 替代 CrossAttn | 72.28 |
| Subword 粒度（非 Jamo） | 67.92 |

### 关键发现

- SCRIPT 在所有基座模型和所有任务类型上一致有效，证明了模型无关性
- 韩语语法纠错任务（Kor-Learner）提升最大（>3.2%p），因为该任务涉及大量子字符级的助词/语尾变化
- NLG 任务比 NLU 任务提升更显著——可能因为 Hangul 的顺序组合结构天然适配自回归解码
- 嵌入空间分析显示 SCRIPT 使形态相关词对的余弦相似度从 0.71 提升到 0.80（+11%）

## 亮点与洞察

- 92.75% 形态变化发生在子字符级的语料统计分析为方法设计提供了强有力的语言学动机
- 即插即用设计极具实用价值——无需重新预训练，对任意韩语 PLM 都可即时增强
- 造字原则驱动的架构设计（而非通用注意力）体现了领域知识的重要性

## 局限与展望

- 目前仅验证到 2.4B 参数规模，更大模型（7B+）上的效果未知
- 设计紧密绑定韩语 Hangul 特性，对其他语言的适用性有限
- 交叉注意力增加了嵌入层的推理成本，尤其在长序列上
- 未与最新的 7B+ 韩语 LLM 进行对比

## 相关工作与启发

- **vs KOMBO**: KOMBO 也建模 Jamo 组合结构，但需要从头预训练且仅支持编码器架构；SCRIPT 是即插即用的
- **vs 多粒度表示**: 之前工作在 Jamo 和子词间切换，而非结构化融合；SCRIPT 通过双通道策略实现真正的融合
- **vs char2subword**: 类似模块只做简单字符到子词的映射，不利用 Hangul 造字原则

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 Hangul 造字原则编码为神经网络架构，语言学与深度学习的精巧结合
- 实验充分度: ⭐⭐⭐⭐⭐ 4种基座模型×9项NLU+多项NLG任务+详细消融+语言学分析
- 写作质量: ⭐⭐⭐⭐ 动机链清晰，语料统计分析有说服力
- 价值: ⭐⭐⭐⭐ 对韩语NLP社区有直接实用价值，即插即用设计降低了使用门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](../../ACL2025/llm_pretraining/leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](compact_example-based_explanations_for_language_models.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)

</div>

<!-- RELATED:END -->
