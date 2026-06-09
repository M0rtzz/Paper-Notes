---
title: >-
  [论文解读] SIGN: Schema-Induced Games for Naming
description: >-
  [AAAI 2026][模型压缩][命名博弈] SIGN 提出在LLM多智能体命名博弈中引入轻量级消息Schema（如 `@say {name: Ck}`），发现结构化先验可将群体约定一致性提升5.8×，收敛所需Token减少一个数量级，为高效多智能体协调提供了简单可控的"调节旋钮"。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "命名博弈"
  - "LLM多智能体"
  - "约定形成"
  - "Schema引导"
  - "多智能体协调"
---

# SIGN: Schema-Induced Games for Naming

**会议**: AAAI 2026  
**arXiv**: [2510.21855](https://arxiv.org/abs/2510.21855)  
**代码**: [github.com/ryanzhangofficial/llm-naming-game-steering](https://github.com/ryanzhangofficial/llm-naming-game-steering)  
**领域**: 模型压缩  
**关键词**: 命名博弈, LLM多智能体, 约定形成, Schema引导, 多智能体协调

## 一句话总结
SIGN 提出在LLM多智能体命名博弈中引入轻量级消息Schema（如 `@say {name: Ck}`），发现结构化先验可将群体约定一致性提升5.8×，收敛所需Token减少一个数量级，为高效多智能体协调提供了简单可控的"调节旋钮"。

## 研究背景与动机

### 领域现状
大语言模型正从单体应用走向多智能体协作场景（协同编程、分布式规划等）。多智能体系统中，智能体之间需要形成一致的沟通约定（convention）。已有研究表明LLM群体可以通过交互自发产生共享约定，类似人类语言演化。

### 现有痛点

**约定不一致**：LLM智能体在无约束的自然语言沟通中容易产生不一致的约定，导致协调失败

**收敛缓慢**：无结构约束下，大量Token被浪费在冗余表达中，群体约定收敛极慢

**可扩展性差**：随着系统规模增大，无结构通信的效率问题更加突出

### 核心矛盾
LLM可以被指示使用JSON Schema等结构化格式来提升推理和协作效率，但这种结构化先验能否**引导约定形成本身**（而非仅提升单次交互质量）尚不清楚。

### 本文切入角度
通过经典的命名博弈（Naming Game）框架，系统研究轻量级Schema先验对LLM群体约定形成的影响。

## 方法详解

### 整体框架
命名博弈定义在 $N$ 个智能体和固定词典 $\mathcal{L} = \{C_1, \ldots, C_M\}$ 上。每轮随机配对两个智能体，各自生成消息，通过解码器映射到词典中的名称。每个智能体维护大小为 $K$ 的记忆窗口，记录最近 $K$ 次与伙伴的交互。

### 关键设计

#### 1. **三种实验条件**
- **NL（Natural Language）**：智能体生成无约束的自然语言输出，解码器尝试提取有效Token
- **NL-SW（Natural Language Sliding Window）**：在NL基础上加入记忆窗口 $K$，最近交互影响后续提案
- **Schema**：要求回复匹配 `@say {name: Ck}` 格式，正则表达式解析；不合规输出重试一次，仍无效则默认随机选择
- **设计动机**：Schema提供了一个显式的、易解析的词典条目句柄，使回复对监听者透明，开销极小

#### 2. **Lose-Shift机制**
- 当 $y_i \neq y_j$（两个智能体不一致）时，以概率 $\alpha$ 采纳伙伴的选择
- $\alpha$ 是关键参数，控制约定传播速度
- 实验探索 $\alpha \in \{0.5, 0.75, 0.9\}$

#### 3. **非合规处理策略**
- 第一次不合规：发送简短提醒后重试
- 第二次仍不合规：对自由文本进行解码，若不可解码则标记为None
- 确保实验在边界情况下也能正常运行

### 实验设置
- **模型**：Phi-3 Mini 4K Instruct（主实验）、LLaMA 3.2 3B Instruct（附录验证）
- **解码参数**：max\_tokens=32, temperature=0.7, top-p=0.9, repeat\_penalty=1.1
- **规模**：$N \in \{12, 24\}$ 个智能体，词典大小 $M=12$，$T=300$ 轮
- **评价指标**：群体一致率（agreement）、收敛所需Token数

## 实验关键数据

### 主实验

| N | K | NL | NL-SW | Schema | 提升 |
|---|---|-----|-------|--------|------|
| 12 | 0 | 0.111 ± 0.048 | — | — | — |
| 24 | 0 | 0.125 ± 0.042 | — | — | — |
| 12 | 5 | — | 0.278 ± 0.127 | **0.611 ± 0.293** | 2.2× |
| 24 | 5 | — | 0.292 ± 0.042 | **0.556 ± 0.064** | 1.9× |
| 12 | 10 | — | 0.333 ± 0.144 | **0.639 ± 0.096** | 1.9× |
| 24 | 10 | — | 0.295 ± 0.039 | **0.588 ± 0.085** | 2.0× |

### 消融实验

| 配置 | 50%一致率所需Token | 60%一致率所需Token | 70%一致率 | 说明 |
|------|-------------------|-------------------|----------|------|
| NL | 未达到 | 未达到 | 未达到 | 根本无法收敛到高一致率 |
| NL-SW | ~10×Schema | ~100×Schema | 未达到 | 收敛极慢 |
| Schema | 基准 | 基准 | **唯一达到** | 一个数量级Token优势 |

### 跨模型验证

| 模型配置 | Schema一致率 | NL/NL-SW一致率 |
|---------|-------------|---------------|
| Phi-3 only | 0.6-0.65 | <0.3 |
| LLaMA only | 0.75-0.8 | 0.65-0.7 |
| Phi+LLaMA混合 | 明显优势 | 低于Schema |

### 关键发现
1. Schema条件下群体一致率可达0.6-0.65，NL-SW仅约0.3，NL低于0.2，最高达**5.8×提升**
2. Schema收敛到50%一致率所需Token比NL/NL-SW**少一个数量级**，60%时近两个数量级
3. 增大 $\alpha$ 对NL-SW和Schema均略微降低一致率，$\alpha=0.5$ 产生最一致的结果
4. 一致性提升主要来源于Schema引导，而非群体规模或记忆窗口
5. LLaMA整体一致性高于Phi-3，但两者上Schema均提供额外优势
6. 混合模型群体中Schema依然有效，对异构系统也适用

## 亮点与洞察
1. **极简设计哲学**：仅一个轻量级标签`@say {name: Ck}`即可显著改善多智能体协调，体现"最少结构化"的力量
2. **从"提升单次交互"到"引导约定形成"**：将结构化格式的价值从微观扩展到宏观社会动力学
3. **模型无关的控制旋钮**：Schema是一个通用的、不依赖特定模型的协调机制
4. **与语言演化的联系**：LLM群体的约定形成过程类似人类语言社区的演化，Schema类似于语法规范

## 局限与展望
1. 词典大小固定为12，未探索更大词典下的效果
2. 群体规模仅测试到24，百级甚至千级规模的可扩展性未验证
3. 仅测试了命名博弈这一简单任务，更复杂的协作场景（如联合规划、多轮谈判）效果未知
4. Schema的一致性是否会限制更广泛任务中的表达能力和创造力，需进一步探讨
5. 未考虑动态Schema或自适应Schema设计

## 相关工作与启发
- 命名博弈理论（Baronchelli et al. 2008）提供了研究约定形成的经典框架
- LLM结构化输出（JSON Schema等）在单次推理中已证实有效，本文将其推广到群体动力学
- 对LLM多智能体系统设计的启示：即使是最简单的通信协议也能显著提升效率

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将结构化格式与约定形成结合是新颖视角
- 实验充分度: ⭐⭐⭐ — 任务较简单，规模较小，但跨模型验证充分
- 写作质量: ⭐⭐⭐⭐ — 简洁清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ — 为多智能体LLM系统设计提供了可操作的洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](../../ICCV2025/model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)
- [\[AAAI 2026\] From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations](from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)
- [\[AAAI 2026\] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)](satisficing_and_optimal_generalised_planning_via_goal_regression_extended_versio.md)
- [\[AAAI 2026\] Renormalization Group Guided Tensor Network Structure Search](renormalization_group_guided_tensor_network_structure_search.md)

</div>

<!-- RELATED:END -->
