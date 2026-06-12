---
title: >-
  [论文解读] Unified Multimodal Understanding via Byte-Pair Visual Encoding
description: >-
  [ICCV2025][多模态VLM][BPE Visual Tokenization] 将 NLP 中的 Byte-Pair Encoding (BPE) 策略应用于视觉 token 化，提出优先级引导的编码方案（融合频率和空间一致性）、课程式数据混合和渐进式参数解冻三阶段训练策略…
tags:
  - "ICCV2025"
  - "多模态VLM"
  - "BPE Visual Tokenization"
  - "Discrete Visual Tokens"
  - "多模态"
  - "Vocabulary Construction"
  - "Curriculum Training"
---

# Unified Multimodal Understanding via Byte-Pair Visual Encoding

**会议**: ICCV2025  
**arXiv**: [2506.23639](https://arxiv.org/abs/2506.23639)  
**代码**: [https://beingbeyond.github.io/Being-VL-0.5](https://beingbeyond.github.io/Being-VL-0.5)  
**领域**: 多模态VLM  
**关键词**: BPE Visual Tokenization, Discrete Visual Tokens, Multimodal LLM, Vocabulary Construction, Curriculum Training

## 一句话总结
将 NLP 中的 Byte-Pair Encoding (BPE) 策略应用于视觉 token 化，提出优先级引导的编码方案（融合频率和空间一致性）、课程式数据混合和渐进式参数解冻三阶段训练策略，构建的 Being-VL-0.5（8B）在离散 token 路线上接近连续 embedding 方法的主流水平。

## 研究背景与动机

**领域现状**：多模态大模型（MLLM）主要有两条路线：(a) **连续 embedding**方法（LLaVA、Qwen-VL 等）使用预训练视觉编码器（如 CLIP）将图像映射为连续向量，通过投影层对齐到语言模型空间；(b) **离散 token** 方法（Chameleon、Unified-IO-2）使用向量量化（VQ-GAN/VQVAE）将图像离散化为 token 序列，与文本 token 统一处理。

**现有痛点**：
   - 连续方法存在**模态鸠沟**（视觉编码器输出的高维连续特征与语言模型期望的离散 token 不匹配）和**信息瓶颈**（压缩过程丢失低频视觉细节，导致幻觉问题）
   - 离散方法虽然天然统一了多模态表示，但简单的 VQ 量化不考虑视觉内容的语义结构、关键视觉概念在 token 空间分布不均匀、且当前性能显著落后于连续方法

**核心矛盾**：NLP 中 BPE token 化已被证明能显著提升 Transformer 的学习效率，因为它将频繁共现的字符合并为语义丰富的 token。先前工作 (Being-VL-0) 已将 BPE 应用于视觉数据的理论框架，但从理论到实践的落地面临三个挑战：(1) 如何设计超越简单频率的编码策略；(2) 如何构建匹配 BPE 层次性的数据混合策略；(3) 如何设计多阶段训练流程。

**本文目标**：将 BPE 视觉 token 化从理论概念推进到实用化的多模态大模型，缩小离散 token 方法与连续 embedding 方法之间的性能差距。

**切入角度**：BPE 视觉 token 天然具有层次结构（底层 token 对应简单图像块，高层 token 编码越来越复杂的视觉模式），因此训练策略也应该是层次化/课程式的。

**核心 idea**：优先级引导的 BPE 词表构建（频率 + 空间一致性）+ 课程式数据混合 + 渐进式参数解冻 = 实用化离散视觉 token MLLM。

## 方法详解

### 整体框架
整体分为两个阶段。**词表构建阶段**：输入训练图像 → VQ-GAN 量化为离散 index 网格 → 优先级引导的 BPE 迭代合并相邻 token 对，扩展词表。**模型训练阶段**：输入图像 → VQ-GAN 量化 → BPE 编码（用构建好的词表）→ 视觉 token 与文本 token 拼接成统一序列 → 送入扩展后的 LLM 进行自回归建模。输出为文本 token 序列。

### 关键设计

1. **优先级引导的 BPE 词表构建（Priority-Guided Encoding）**:

    - 功能：构建既考虑出现频率又考虑空间关系的 BPE 视觉词表
    - 核心思路：定义优先级函数 $P(a,b) = F(a,b) + \alpha \cdot S(a,b)$，其中 $F(a,b)$ 是 token 对 $(a,b)$ 的共现频率，$S(a,b) = \frac{1}{N_{a,b}} \sum_{i=1}^{N_{a,b}} d(u_i(a,b), \bar{u}(a,b))$ 是空间一致性分数，衡量该 token 对在不同图像中的相对位置是否一致（用高斯核 $d(u_1,u_2) = \exp(-\|u_1 - u_2\|^2 / 2\sigma^2)$ 度量）。每轮迭代选择优先级最高的 token 对合并为新 token，直到词表达到目标大小
    - 设计动机：纯频率 BPE 在文本中有效的原因是文本是一维序列，但视觉数据是二维的，空间关系至关重要。一个 token 对即使频繁共现，如果其空间关系不一致（如在不同图像中位于不同相对位置），合并后的 token 缺乏稳定的语义含义

2. **模型扩展（Model Expanding）**:

    - 功能：将预训练的文本 LLM 扩展为支持视觉 token 的多模态模型
    - 核心思路：将 embedding 层从 $|V_{\text{text}}|$ 扩展到 $|V_{\text{text}}| + |D|$（默认 8K VQ + 8K BPE = 16K 新 token），新增 embedding 使用 He 初始化。同步扩展 output head 的词表
    - 设计动机：直接扩展词表而非使用额外的投影层，保持了统一 token 表示的简洁性

3. **多阶段渐进式训练（Multi-Stage Training）**:

    - 功能：通过三阶段训练逐步释放模型容量
    - 核心思路：
        - **Stage 1 (Embedding Alignment)**：仅训练新增的视觉 token embedding，冻结所有 LLM 参数。数据以基础 image-caption 对为主
        - **Stage 2 (Selective Fine-tuning)**：解冻前 25% 的 Transformer 层，数据逐渐加入感知任务（详细视觉属性描述）
        - **Stage 3 (Full Fine-tuning)**：解冻全部参数，数据侧重复杂推理和指令跟随任务
    - 设计动机：BPE token 具有层次性——底层 token 对应简单图像块，高层 token 编码复杂视觉模式。训练也需要匹配这个层次：先让 token 学会基础语义映射，再逐步挑战复杂推理任务。这与直接全参数微调相比，避免了语言能力的灾难性遗忘

### 损失函数 / 训练策略
标准自回归交叉熵损失：$\mathcal{L}(\theta) = -\mathbb{E}_{(X,I,Y) \sim \mathcal{D}} [\sum_{i=1}^{|Y|} \log p_\theta(y_i | y_{<i}, X, T(Q(I)))]$。数据分为 Foundation、Perception、Reasoning、Instruction 四类，按课程式策略在三个阶段中调整混合比例。

## 实验关键数据

### 主实验

| 模型 | Token类型 | VQAv2 | MMBench | MME-P | SciQA | POPE | VizWiz |
|------|----------|-------|---------|-------|-------|------|--------|
| Being-VL-0.5 (ours) | 离散 | 80.2 | 71.8 | 1525.8 | 70.3 | 84.3 | 57.4 |
| Being-VL-0.5+ (16K) | 离散 | 80.6 | 72.1 | 1536.3 | 69.0 | 86.0 | 57.8 |
| Being-VL-0 (前作) | 离散 | 60.6 | 44.0 | 1316.2 | 64.3 | 81.3 | 48.2 |
| w/o BPE | 离散 | 54.3 | 38.2 | 1301.2 | 57.8 | 76.1 | 45.0 |
| LLaVA-1.5 | 连续 | 78.5 | 64.3 | 1510.7 | 66.8 | 85.9 | 50.0 |
| VILA-1.5 | 连续 | 80.9 | 72.3 | - | - | 84.4 | 58.7 |

### 消融实验

| 配置 | Perception Avg | Reasoning Avg | 说明 |
|------|---------------|---------------|------|
| 完整方案 (Curriculum + Progressive) | **80.3** | **71.1** | 最优 |
| 仅 Progressive 解冻 | 74.9 | 65.1 | 无课程数据，掉 ~6% |
| 仅 Curriculum 数据 | 76.8 | 67.5 | 无渐进解冻，掉 ~4% |
| 单阶段训练 | 71.2 | 62.3 | 基线，掉 ~9% |

### 关键发现
- **BPE 是核心贡献**：去掉 BPE 词表（w/o BPE），模型在所有 benchmark 上大幅崩塌（VQAv2 从 80.2 降至 54.3），证明 BPE 视觉 token 化的关键性
- **词表大小的权衡**：8K BPE 词表在效率和性能间达到最佳平衡，16K 词表有更高的 scaling potential 但当前数据量下存在未激活 token（embedding 可视化中出现白色条纹）
- **课程数据比渐进解冻更重要**：消融实验中，仅去掉课程策略掉 6%，仅去掉渐进解冻掉 4%，说明 BPE token 的学习更依赖合理的数据排序
- **离散方法逼近连续方法**：Being-VL-0.5 在 VQAv2 (80.2 vs 80.9) 和 MMBench (71.8 vs 72.3) 上已接近 VILA-1.5 等连续方法

## 亮点与洞察
- **将 NLP 的成功经验迁移到视觉**：BPE 在文本 tokenization 中的巨大成功已被证明对 Transformer 学习至关重要。本文系统性地将这一思路扩展到二维视觉数据，弥补了此前从理论到实践的 gap
- **空间一致性是视觉 BPE 的 key insight**：文本 BPE 只需考虑频率，但视觉数据的二维空间结构要求 token 对在不同图像中保持一致的空间关系。这个 insight 使视觉 BPE 不再是文本 BPE 的简单照搬
- **Embedding 可视化揭示统一表示空间**：Figure 3 的 embedding 权重分布图清晰地展示了 BPE token 如何弥合视觉与文本 token 之间的表示差距，这是理解离散 token 方法工作原理的重要窗口

## 局限与展望
- **仅 8B 模型规模**：受限于计算资源，未在更大模型上验证。scaling 分析暗示更大词表 + 更多数据可能进一步提升
- **仅做理解任务**：离散 token 天然支持生成任务（可以像生成文本 token 一样生成视觉 token），但本文未涉及图像生成
- **VQ-GAN 是瓶颈**：词表构建依赖 VQ-GAN 的量化质量，如果 VQ-GAN codebook 质量不高，后续 BPE 也难以补救
- **16K 词表未充分利用**：embedding 可视化显示大量 BPE token 未被激活，说明当前数据量不足以支撑更大词表

## 相关工作与启发
- **vs Being-VL-0**：前作提出了视觉 BPE 的理论框架但使用简单频率编码。本文引入空间一致性、课程训练和渐进解冻，在 MMBench 上从 44.0 提升到 71.8（+27.8）
- **vs Chameleon**：Chameleon 使用简单 VQ token 无 BPE，VQAv2 只有 56.2 vs 本文 80.2，差距巨大。BPE 的结构化 token 化是核心差异
- **vs LLaVA-1.5**：连续 embedding 方法，VQAv2 78.5 vs 80.2，本文已略微超越。这证明离散 token 路线在充分优化后可以与连续方法竞争

## 评分
- 新颖性: ⭐⭐⭐⭐ 视觉 BPE 方向本身有前作铺垫，但优先级编码和训练策略是新贡献
- 实验充分度: ⭐⭐⭐⭐ 多个 benchmark + 详细消融 + 可视化分析，但缺少生成任务验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论和实践的衔接自然
- 价值: ⭐⭐⭐⭐ 证明了离散 token 路线的可行性，为统一多模态表示提供了实践路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](../../NeurIPS2025/multimodal_vlm/unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](../../CVPR2026/multimodal_vlm/customized_visual_storytelling_with_unified_multimodal_llms.md)

</div>

<!-- RELATED:END -->
