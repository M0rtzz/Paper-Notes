---
title: >-
  [论文解读] Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats
description: >-
  [NeurIPS 2025][多模态][hallucination] 提出 AllPath，一个基于 Transformer 因果架构的多路径幻觉干预框架，首次发现 LVLM 的幻觉不来自单一因果路径而是 image-to-input-text、image-to-output-text、text-to-text 三条路径的交互，并且模型会根据问答对齐格式自适应选择不同路径；通过为每条路径设计轻量级关键 head 识别方法并自适应干预，在 POPE、MCQ-POPE、CHAIR、MME 四个不同格式 benchmark 上一致降低幻觉。
tags:
  - NeurIPS 2025
  - 多模态
  - hallucination
  - 注意力机制
  - causal path
  - multi-path framework
  - training-free
---

# Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats

**会议**: NeurIPS 2025  
**arXiv**: [2511.17254](https://arxiv.org/abs/2511.17254)  
**代码**: [https://github.com/SooLab/AllPath](https://github.com/SooLab/AllPath)  
**领域**: 多模态VLM / 幻觉缓解 / 注意力头干预  
**关键词**: hallucination, attention head intervention, causal path, multi-path framework, training-free

## 一句话总结
提出 AllPath，一个基于 Transformer 因果架构的多路径幻觉干预框架，首次发现 LVLM 的幻觉不来自单一因果路径而是 image-to-input-text、image-to-output-text、text-to-text 三条路径的交互，并且模型会根据问答对齐格式自适应选择不同路径；通过为每条路径设计轻量级关键 head 识别方法并自适应干预，在 POPE、MCQ-POPE、CHAIR、MME 四个不同格式 benchmark 上一致降低幻觉。

## 研究背景与动机

**领域现状**：LVLM 幻觉缓解方法主要分两类——对比解码（VCD、ICD）通过校准输出分布来减少语言偏置，干预方法（PAI、AD-HH）通过直接操纵注意力权重/头来增强视觉定基或抑制文本主导行为。

**现有痛点**：每种方法通常只干预单一因果路径——PAI 只干预 image→output-text 路径，AD-HH 只干预 input-text→output-text 路径——导致它们各自只在部分 benchmark 上表现好（PAI 擅长 CHAIR 但 POPE 一般，VCD 反之）。

**核心矛盾**：幻觉不是来自单一路径，而是多路径交互的结果；更关键的是，**LVLM 对不同问答格式（二选一/多选/开放描述）使用不同的因果路径**，单路径干预无法覆盖所有场景。

**切入角度**：从 Transformer 因果架构出发，系统性地分析所有可能的信息传播路径，并为每种路径设计针对性的关键头识别和干预方法。

**核心 idea**：多路径框架 + 自适应路径选择干预，仅需单次前向传播即可完成所有 head 评分。

## 方法详解

### 整体框架
AllPath 分三步：(1) 用两种轻量方法识别 text-to-text (T2T) 和 image-to-text (I2T) 关键注意力头；(2) 分析这些头在不同路径和问答格式下的角色；(3) 根据问题类型自适应选择干预路径，对正面 head 放大、负面 head 抑制。

### 关键设计

1. **Text-to-Text Head 识别（LPI Score）**：

    - 功能：量化每个注意力头对幻觉 token / 非幻觉 token 的推动程度
    - 核心思路：定义 Log Probability Increase (LPI) score $\text{logProb}_{\uparrow}^{(l,n)}(\mathcal{B}_t) = \log\sum_{b\in\mathcal{B}_t}\mathbb{P}(b|h_t^{(l-1)}+H_t^{(l,n)}) - \log\sum_{b\in\mathcal{B}_t}\mathbb{P}(b|h_t^{(l-1)})$，分别对幻觉集 $\mathcal{B}_t^-$ 和非幻觉集 $\mathcal{B}_t^+$ 计算平均 LPI，T2T Score = $S_{\text{T2T}}^{(l,n),+} - S_{\text{T2T}}^{(l,n),-}$，越小表示该 head 越倾向促进幻觉
    - 设计动机：相比 zero-out 策略（需要对每个 head 做完整前向）和训练方法（需标注数据），LPI 只需一次前向即可获得所有 head 的分数

2. **Image-to-Text Head 识别（I2T Score）**：

    - 功能：识别哪些注意力头在视觉 token 上有语义对齐的注意力模式
    - 核心思路：选取目标 token $\mathcal{T}_{\text{I2T}}$（首次出现的物体词），分为图中存在的 $\mathcal{T}_{\text{I2T}}^+$ 和不存在的 $\mathcal{T}_{\text{I2T}}^-$。对 $\mathcal{T}_{\text{I2T}}^+$，计算 head 在对应区域 $M_r$ 上的注意力总和；对 $\mathcal{T}_{\text{I2T}}^-$，计算在整个图像上的注意力总和。$S_{\text{I2T}}^{(l,n)} = S_{\text{I2T}}^{(l,n),+} - S_{\text{I2T}}^{(l,n),-}$
    - 设计动机：好的 I2T head 应在物体存在时集中注意力于对应区域，物体不存在时注意力分散

3. **关键发现 — T2T Head 与对齐格式的强相关**：

    - 同一对齐格式（如 Yes/No）的不同 benchmark 之间 T2T head 高度相关（$\rho=0.82$），但不同格式（Yes/No vs MCQ）之间相关性急剧下降（$\rho=0.12$）
    - 这意味着 T2T head 主要负责指令遵循/格式对齐，而非视觉理解

4. **关键发现 — Image-to-input-text 与 image-to-output-text 路径不同**：

    - 虽然两种 I2T head 都比平均 head 更关注视觉内容，但它们之间几乎不相关
    - 这意味着只干预输出路径是不够的，输入路径的 head 也需要干预

5. **自适应干预策略**：

    - 功能：根据问题类型选择干预的路径组合，对选定 head 用不同缩放因子
    - 核心思路：选择 top-$\xi$ 和 bottom-$\xi$ 的 T2T head（$Z_{\text{T2T}}^+$, $Z_{\text{T2T}}^-$），top-$\zeta$ 的 I2T head（$Z_{\text{I2T}}^+$）。修改 MHA 输出：$\tilde{H}_{\leq t}^{(l)} = \sum_n \lambda^{(l,n)} H_{\leq t}^{(l,n)}$，其中 $\lambda^{(l,n)} = \gamma^+$ if $(l,n)\in Z^+$，$\gamma^-$ if $(l,n)\in Z^-$，否则 1
    - 默认设置：$\gamma^+=2.0$, $\gamma^-=0.0$（直接关闭负面 head）

### 损失函数 / 训练策略
- **无需训练**：整个方法是 training-free 的，仅在推理时做一次前向传播识别 head + 缩放干预
- 短回答任务：$\xi=20, \zeta=10$；开放式任务：$\xi=40, \zeta=50$

## 实验关键数据

### 主实验 — POPE & MCQ-POPE & CHAIR

| 方法 | POPE-Random Acc | POPE-Advers. Acc | MCQ-POPE-Random Acc | CHAIR $C_S$↓ | CHAIR $C_I$↓ |
|------|----------------|-----------------|---------------------|-------------|-------------|
| Vanilla | 85.1 | 80.9 | 72.8 | 52.2 | 14.6 |
| VCD | 86.3 | 81.4 | 78.2 | 58.2 | 16.1 |
| PAI | 86.4 | 82.5 | 78.0 | 28.8 | 7.9 |
| AD-HH | 85.0 | 80.9 | 78.5 | 33.2 | 7.5 |
| **AllPath** | **87.2** | **82.8** | **80.5** | **26.6** | **7.2** |

### MME 幻觉子集

| 方法 | Existence↑ | Count↑ | Position↑ | Color↑ | Total↑ |
|------|-----------|--------|-----------|--------|--------|
| Vanilla | 180.0 | 113.9 | 116.7 | 129.4 | 540.0 |
| VCD | 177.8 | 122.8 | 122.2 | 141.7 | 564.4 |
| PAI | 185.0 | 122.8 | 114.4 | 144.4 | 566.7 |
| **AllPath** | **188.3** | **126.1** | **132.2** | **153.3** | **600.0** |

### 消融实验 — head 数量与缩放因子

| $\xi$ (T2T heads) | $\zeta$ (I2T heads) | POPE-Rand Acc | POPE-Advers. Acc |
|---|---|---|---|
| 0 | 10 | 86.2 | 81.6 |
| 20 | 0 | 86.4 | 82.5 |
| **20** | **10** | **87.2** | **82.8** |
| 30 | 10 | 88.3 | 82.9 |
| 20 | 15 | 88.2 | 83.0 |

### 关键发现
- VCD 在 POPE 上好但 CHAIR 上差（$C_S$ 从 52.2 恶化到 58.2），因为只干预文本路径；PAI 在 CHAIR 上好但 POPE/MCQ-POPE 改进有限——验证了单路径干预的局限性
- AllPath 是唯一在所有 benchmark（三种不同对齐格式）上都一致提升的方法
- 完全去掉 output-text 到 image 的注意力后，POPE 仅下降约 2%（说明 text-to-text 路径对 Yes/No 格式重要），但 CHAIR 下降 10%（说明 image-to-text 路径对开放描述重要）
- MME 直接复用 POPE 的 head 设置仍能一致提升，验证了方法的泛化性

## 亮点与洞察
- **单次前向即完成所有 head 评分**的高效性设计是核心技术亮点，相比 zero-out 策略（每 head 一次前向）效率提升数量级
- **T2T head 与对齐格式强相关**的发现具有理论价值——解释了为什么不同方法在不同 benchmark 上表现不一致，为未来干预方法设计提供了结构化指导
- **image-to-input-text 与 image-to-output-text 路径独立**的发现打破了"只需关注输出"的直觉假设
- 方法的 plug-and-play 特性使其可直接应用于任何基于 Transformer 的 LVLM

## 局限与展望
- 主要在 LLaVA-v1.5-7B 上验证，虽附录有 Qwen-VL 和 Qwen2.5-VL 结果但不够充分
- 缩放因子 $\gamma^+, \gamma^-$ 和 head 数 $\xi, \zeta$ 对不同任务格式需要手动调整
- T2T head 和 I2T head 的选择依赖于标注的幻觉/非幻觉 token，需要一个小的标注集
- 对于完全新的问答格式（如 chain-of-thought），是否需要重新识别 head 尚未探讨
- 多模态大模型中是否存在更深层级的路径交互（如二阶效应）值得进一步研究

## 相关工作与启发
- **vs VCD**：VCD 通过对比蒸馏视觉输入来校准输出分布，本质上只干预 image-to-output-text 路径，CHAIR 上甚至恶化；AllPath 多路径干预更全面
- **vs PAI**：PAI 增强图像注意力（image-to-text 路径），CHAIR 表现好但 POPE/MCQ-POPE 改进有限；AllPath 同时覆盖 T2T 路径
- **vs AD-HH**：AD-HH 抑制"懒惰" text-dominant head，但只关注 text-to-text 路径；AllPath 证明需要同时干预多条路径

## 评分
- 新颖性: ⭐⭐⭐⭐ 多路径框架概念清晰，T2T/I2T head 识别方法简洁优雅，"格式决定路径"的发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 四个不同格式 benchmark + 相关性分析 + 消融 + 多模型验证，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，分析->发现->方法->验证的叙事结构合理
- 价值: ⭐⭐⭐⭐ Training-free 且适用于多种幻觉场景，对理解 LVLM 内部机制有启发

<!-- RELATED:START -->

## 相关论文

- [GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](../../ACL2025/multimodal_vlm/agent_rewardbench.md)
- [UniTok: A Unified Tokenizer for Visual Generation and Understanding](unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)

<!-- RELATED:END -->
