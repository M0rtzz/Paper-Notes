---
title: >-
  [论文解读] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding
description: >-
  [CVPR 2025][多模态][幻觉缓解] 本文通过大量实验揭示了 LVLM 幻觉成因的混合性——不同样本和不同生成步骤面临不同类型的幻觉挑战，据此提出 Octopus 框架，利用可学习的 decision token 和 transformer block 在每个生成步自适应选择最合适的对比解码（CD）策略，通过 DPO 优化，在四个基准上全面超越现有 CD 方法。
tags:
  - CVPR 2025
  - 多模态
  - 幻觉缓解
  - 对比解码
  - 动态策略选择
  - 大视觉语言模型
  - DPO
---

# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding

**会议**: CVPR 2025  
**arXiv**: [2503.00361](https://arxiv.org/abs/2503.00361)  
**代码**: https://github.com/LijunZhang01/Octopus  
**领域**: 多模态VLM  
**关键词**: 幻觉消除, 对比解码, 动态策略选择, LVLM, DPO

## 一句话总结

本文揭示了多模态大模型幻觉的混合特性——不同样本甚至同一回答中的不同 token 面临不同类型的幻觉挑战（语言先验、视觉信息丢失、注意力偏差），据此提出 Octopus 框架，通过可学习的"眼睛"模块自适应识别幻觉类型，动态选择最适合的对比解码策略（"触手"），在四个基准上实现了 SOTA。

## 研究背景与动机

**领域现状**：大型视觉-语言模型（LVLM）在视觉理解和多模态推理上表现出色，但严重受到幻觉问题的困扰——生成看似合理但与实际图像内容不符的描述（虚构物体、错误属性、不存在的关系）。幻觉消除方法主要分两条路线：(1) 重训练方法——构建高质量数据微调模型；(2) 对比解码（CD）方法——对比原始输入和扰动输入的输出分布，无需训练。

**现有痛点**：现有 CD 方法（VCD、M3ID、AVISC）都采用"一刀切"策略——对所有样本和所有生成步骤使用同一种扰动方式。VCD 用高斯噪声遮蔽图像来克服语言先验，M3ID 屏蔽文本来减少视觉信息损失，AVISC 用盲标记来缓解注意力偏差。但每种策略只针对特定类型的幻觉有效。

**核心矛盾**：幻觉的成因是混合的——作者通过实验证明了两个关键事实：(1) 在样本层面，约 60% 的样本只能被某一种特定 CD 策略纠正，三种策略同时有效的样本仅 ~10%；(2) 在 token 层面，同一个描述中不同的幻觉词（如"sitting"、"lying"、"person"）分别由不同的幻觉原因导致。

**本文目标** 设计一个能自适应地为每个样本和每个生成步骤选择最合适的 CD 策略的框架。

**切入角度**：将多种现有 CD 方法视为"诊断工具"——不是要替代它们，而是要学会在正确的时刻使用正确的工具。这类似章鱼用不同触手完成不同任务。

**核心 idea**：用可学习的决策模块动态选择每个生成步骤最适合的对比解码策略，实现个性化的幻觉消除。

## 方法详解

### 整体框架

Octopus 由两部分组成：(1) "眼睛"——一个 Transformer 块 + 可学习决策 token，负责在每个生成步骤识别幻觉类型；(2) "触手"——多个候选 CD 策略（VCD、M3ID、AVISC + null action），根据"眼睛"的决策执行相应的对比解码操作。训练通过 DPO 完成，无需人工标注决策标签。输入是 LVLM 的隐状态序列，输出是每步的策略选择动作序列。

### 关键设计

1. **决策 Token（Octopus 之眼）**:

    - 功能：在每个生成步骤自适应识别当前面临的幻觉类型
    - 核心思路：构建一个 Transformer 块 $\mathcal{O}_\phi$，将 LVLM 的隐状态序列 $H_t = \{h_i\}_{i=1}^t$（包含视觉、文本和已生成的 token 信息）与可学习的决策 token $eye \in \mathbb{R}^d$ 拼接后输入。通过 self-attention 机制，$eye$ 自适应聚合各模态隐状态的信息，输出 $h_{eye}^t$。再经 MLP 映射到 $k$ 维动作向量，取 argmax 得到每步的策略选择：$a_t = \text{argmax}(\text{Softmax}(\text{MLP}(h_{eye}^t)))$
    - 设计动机：LVLM 每步生成受视觉、文本和上下文共同影响，决策 token 利用 self-attention 综合考虑所有信息源来判断当前最可能的幻觉类型

2. **多触手 CD 策略池**:

    - 功能：提供四种候选操作应对不同幻觉成因
    - 核心思路：四个动作空间包括 VCD（高斯噪声遮蔽图像对比，克服语言先验）、M3ID（屏蔽文本独立输入图像对比，减少视觉信息丢失）、AVISC（用盲 token 替代视觉输入对比，缓解注意力偏差）、以及空操作（当前步骤无需对比解码）。根据决策 token 的 one-hot 输出 $a_t$ 选择对应策略执行
    - 设计动机：实验已证明这三种策略分别针对不同幻觉成因有效，加上空操作应对无幻觉情况，四种选择覆盖了主要幻觉场景

3. **基于 DPO 的训练机制**:

    - 功能：在无人工标注的情况下学习最优决策策略
    - 核心思路：对每个样本随机生成 10 个不同的动作序列（每步随机选四种操作之一），用 CHAIR 指标评估生成质量，将表现好的序列标为正样本 $\mathcal{A}^+$、差的标为负样本 $\mathcal{A}^-$。使用无参考模型的 DPO 目标优化：$\max_{\mathcal{O}_\phi} \mathbb{E} \log \sigma(\beta \log \mathcal{O}_\phi(\mathcal{A}^+|x) - \beta \log \mathcal{O}_\phi(\mathcal{A}^-|x))$。仅更新 Octopus 参数 $\phi$，LVLM 权重冻结
    - 设计动机：token 级别的动作序列空间极大，无法穷举或人工标注。DPO 将问题转化为偏好学习，通过正负样本对比自动发现最优策略组合

### 损失函数 / 训练策略

训练数据：生成任务在 MSCOCO 上构建 10,000 条偏好数据；判别任务构建 7,000 条。使用 Adam 优化器，4 张 3090 GPU，batch size=4。DPO 中 $\beta=1$。

## 实验关键数据

### 主实验（生成任务）

| LVLM | 方法 | AMBER-CHAIR↓ | Object-HalBench CHAIRs↓ | MMHalBench Score↑ |
|------|------|-------------|------------------------|-------------------|
| LLaVA-1.5-7B | Base | 8.0 | 25.0 | 1.59 |
| LLaVA-1.5-7B | +VCD | 6.7 | 23.6 | 1.96 |
| LLaVA-1.5-7B | +M3ID | 6.0 | 23.2 | 2.14 |
| LLaVA-1.5-7B | +AVISC | 6.3 | 22.1 | 2.19 |
| LLaVA-1.5-7B | **+Octopus** | **4.8** | **20.8** | **2.61** |

### 判别任务

| 方法 | AMBER Acc↑ | AMBER F1↑ | POPE ALL Acc↑ | POPE ALL F1↑ |
|------|-----------|----------|-------------|-------------|
| LLaVA-1.5-7B Base | 67.00 | 71.10 | 82.04 | 80.42 |
| +VCD | 67.30 | 71.10 | 82.96 | 81.81 |
| +AVISC | 70.70 | 75.45 | 83.39 | 81.01 |
| **+Octopus** | **76.70** | **82.70** | **85.79** | **83.44** |

### 关键发现

- Octopus 在 AMBER 的 CHAIR 指标上比 Base 模型降低了约 40%（8.0→4.8），比最好的单一 CD 方法 M3ID 又进一步降低了 20%
- 判别任务上提升更加显著——AMBER 准确率从 67.0% 提升到 76.7%（+9.7%），F1 从 71.1% 提升到 82.7%（+11.6%）
- 即使与需要重训练整个模型的方法（HACL、POVID、HA-DPO 等）相比，Octopus 作为后处理方法仍然更优
- Octopus 在 InstructBLIP 上也有显著效果，证明框架的通用性

## 亮点与洞察

- **幻觉的混合性质的系统论证**：通过样本级和 token 级两个维度的定量实验，首次严格证明了幻觉成因是混合的且动态变化的。这个发现本身就是重要贡献，改变了社区对幻觉问题的认知
- **DPO 优化动作序列**：将 token 级别的 CD 策略选择问题转化为序列偏好优化问题，巧妙绕过了巨大的组合搜索空间。这个"先随机采样再用偏好学习"的范式可以迁移到其他需要序列级别决策的场景
- **即插即用的框架设计**：Octopus 不修改 LVLM 权重，新的 CD 方法可以随时添加为新"触手"，无需重新训练整个框架，扩展性极强

## 局限与展望

- 当前仅集成三种 CD 策略 + 空操作，策略池较小。随着新的幻觉消除技术出现，如何高效地增加"触手"而不急剧增加训练成本是个问题
- DPO 训练需要为每个样本生成 10 个随机序列来构建偏好数据，计算开销较大
- argmax 操作使得梯度不能直接反传，训练效率和收敛性可能受影响
- 仅在 LLaVA-1.5-7B 和 InstructBLIP 上验证，尚未在更大规模模型（如 13B/72B）或更新的模型上测试
- 依赖 CHAIR 指标评估幻觉，该指标本身有局限性（只能检测物体级幻觉）

## 相关工作与启发

- **vs VCD**: VCD 专注于克服语言先验，对注意力偏差导致的幻觉无效；Octopus 可以在需要时调用 VCD，同时在其他情况下切换到更合适的策略
- **vs AVISC**: AVISC 通过增强视觉信息缓解注意力偏差，但对语言先验导致的幻觉帮助不大；Octopus 将其作为"触手"之一灵活调用
- **vs 重训练方法（HACL/HA-DPO）**: 重训练方法需要修改模型权重且训练成本高，Octopus 作为后处理方案更轻量且效果更好
- 这种"将多个专家方法作为工具，学习何时使用哪个"的思路，在 MoE、工具使用等场景中也有广泛应用前景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题视角新颖（幻觉混合性质），解决方案自然优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖生成和判别任务、多个基准，但缺少大模型上的验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机推导层层递进，图表精美，实验设计严谨
- 价值: ⭐⭐⭐⭐ 框架思路具有广泛的可扩展性，对幻觉研究有启发
# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding

**会议**: CVPR 2025  
**arXiv**: [2503.00361](https://arxiv.org/abs/2503.00361)  
**代码**: https://github.com/LijunZhang01/Octopus  
**领域**: 多模态VLM  
**关键词**: 幻觉缓解, 对比解码, 动态策略选择, 大视觉语言模型, DPO

## 一句话总结

本文通过大量实验揭示了 LVLM 幻觉成因的混合性——不同样本和不同生成步骤面临不同类型的幻觉挑战，据此提出 Octopus 框架，利用可学习的 decision token 和 transformer block 在每个生成步自适应选择最合适的对比解码（CD）策略，通过 DPO 优化，在四个基准上全面超越现有 CD 方法。

## 研究背景与动机

**领域现状**：大视觉语言模型（LVLMs）如 LLaVA、InstructBLIP 在视觉理解和多模态推理上表现优异，但普遍存在幻觉问题——生成虚构的物体、错误的属性和不存在的关系。对比解码（Contrastive Decoding, CD）作为一种无需重训练的后处理方法成为缓解幻觉的重要方向。

**现有痛点**：
1. **单一策略的局限**：现有 CD 方法（VCD、M3ID、AVISC）各自针对特定类型的幻觉设计——VCD 对抗语言先验、M3ID 缓解视觉信息丢失、AVISC 减少注意力偏差。但它们都采用"一刀切"的方式，对所有样本和所有生成步使用相同的干扰策略。
2. **幻觉成因的复杂性被忽视**：没有工作系统研究过不同样本和不同 token 是否面临相同类型的幻觉。

**核心矛盾**：幻觉的成因是混合的（语言先验 + 视觉信息丢失 + 注意力偏差），但现有方法只能"头痛医头"，用单一策略应对所有情况，必然导致次优结果。

**本文切入角度**：先通过诊断实验证明幻觉的混合性，然后设计一个自适应框架，让模型在每个生成步自动选择最合适的 CD 策略。

**核心 idea**：像章鱼（Octopus）一样，用"眼睛"（decision token）识别幻觉类型，用多条"触手"（多种 CD 策略）分别应对不同的幻觉挑战。

## 方法详解

### 整体框架

Octopus 框架包含两个核心组件：(1) **决策模块**（"眼睛"）——一个 transformer block + 可学习 decision token，负责在每个生成步判断当前 token 面临哪种幻觉类型；(2) **执行模块**（"触手"）——多种现成的 CD 策略（VCD、M3ID、AVISC + null 动作），根据决策结果执行对应的对比操作。通过 DPO 优化决策模块参数，LVLM 参数保持冻结。

### 关键设计

1. **样本级幻觉诊断实验**:
    - 功能：证明单一 CD 策略无法覆盖所有幻觉样本
    - 核心思路：在 AMBER、Object-HalBench、MMHalBench 三个数据集上，分别使用 VCD、M3ID、AVISC 三种 CD 方法对 LLaVA-1.5-7B 的每个样本进行干预，统计每种方法有效纠正的样本比例。结果显示~60% 的样本只能被某一种特定 CD 策略纠正，三种策略同时有效的重叠区域仅约 10%
    - 设计动机：为动态策略选择提供实证依据——如果一种策略就能解决所有问题，就没必要做动态选择

2. **Token 级幻觉诊断实验**:
    - 功能：证明同一个样本中不同 token 的幻觉成因也不同
    - 核心思路：在 AMBER 数据集上，对每个描述中前 3 个幻觉名词使用枚举法测试不同 CD 策略组合。量化结果表明组合多种策略（如 strategy-1+3，strategy-1+2+3）显著优于单一策略。定性分析中，通过注意力图发现同一句话中 "sitting" 受注意力偏差影响（关注了视觉盲 token），"lying" 是因为对视觉信息关注不足，"person" 则完全依赖语言 token——三个词对应三种不同的幻觉成因
    - 设计动机：进一步将动态策略选择的粒度从样本级细化到 token 级

3. **Octopus 决策与执行架构**:
    - 功能：在每个生成步自适应选择最合适的 CD 策略
    - 核心思路：构建一个轻量 transformer block $\mathcal{O}_\phi$，将 LVLM 的隐状态序列 $H_t = \{h_i\}_{i=1}^t$（包含视觉、文本和已生成 token 的信息）与一个可学习的 decision token $eye \in \mathbb{R}^d$ 拼接，加上位置编码后送入 transformer block：$[h_{eye}^t; H_t'] = \mathcal{O}_\phi(\text{concat}[eye; H_t] + E_{pos})$。通过自注意力机制，$h_{eye}^t$ 聚合来自全序列的信息。然后经 MLP 映射为动作向量 $h_{act}^t \in \mathbb{R}^k$（$k=4$，对应三种 CD 策略 + null 动作），取 argmax 得到当前步的策略选择 $a_t$。最终生成一个完整的工作流 $\mathcal{A} = \{a_t\}_{t=1}^N$
    - 设计动机：利用 transformer 的自注意力机制让 decision token 综合考虑视觉输入、文本指令和已生成内容来做出全局性的策略决策

### 损失函数 / 训练策略

**DPO 优化**：由于 argmax 操作不可微分，且缺乏显式的决策标签，采用 Direct Preference Optimization 进行训练。

- **数据构建**：对每个样本随机生成 10 个不同的动作序列（每步随机选择 4 种动作之一），根据 CHAIR 指标将序列分为正样本（减少幻觉的工作流 $\mathcal{A}^+$）和负样本（增加幻觉的工作流 $\mathcal{A}^-$）
- **优化目标（去除参考模型的 DPO）**：$\max_{\mathcal{O}_\phi} \mathbb{E} \log \sigma(\beta \log \mathcal{O}_\phi(\mathcal{A}^+ | x) - \beta \log \mathcal{O}_\phi(\mathcal{A}^- | x))$，其中 $x = (v, q)$ 是视觉-文本输入，$\beta = 1$
- **关键特性**：仅优化 Octopus 的参数 $\phi$，LVLM 权重完全冻结，保证部署灵活性

**训练数据**：生成任务使用 MSCOCO 的 10,000 个偏好数据对，判别任务使用 7,000 个幻觉数据。训练在 4×3090 GPU 上完成，batch size 为 4。

## 实验关键数据

### 主实验（生成任务，LLaVA-1.5-7B）

| 数据集 | 指标 | LLaVA Base | +VCD | +M3ID | +AVISC | +Octopus | 提升 vs 最佳CD |
|--------|------|-----------|------|-------|--------|----------|---------------|
| AMBER | CHAIR↓ | 8.0 | 6.7 | 6.0 | 6.3 | **4.8** | -1.2 |
| AMBER | Cover↑ | 44.5 | 46.5 | 48.9 | 46.6 | **49.2** | +0.3 |
| AMBER | HalRate↓ | 31.0 | 27.8 | 26.0 | 25.6 | **23.4** | -2.2 |
| Object-HalBench | CHAIRs↓ | 25.0 | 23.6 | 23.2 | 22.1 | **20.8** | -1.3 |
| Object-HalBench | CHAIRi↓ | 9.2 | 8.4 | 7.3 | 7.8 | **6.6** | -0.7 |
| MMHalBench | Score↑ | 1.59 | 1.96 | 2.14 | 2.19 | **2.61** | +0.42 |

### 消融实验（AMBER 数据集）

| 配置 | CHAIR↓ | Cover↑ | Hal↓ | Cog↓ |
|------|--------|--------|------|------|
| LLaVA Base（无 CD） | 8.0 | 44.5 | 31.0 | 2.2 |
| 随机选择三种 CD 策略 | 6.9 | 46.2 | 26.1 | 2.2 |
| Octopus (Str1+Str2) | 5.5 | 48.7 | 25.8 | 1.5 |
| Octopus (Str1+Str3) | 5.7 | 48.2 | 25.3 | 1.5 |
| Octopus (Str2+Str3) | 5.5 | 48.4 | 26.2 | 1.6 |
| **Octopus (全部三种+null)** | **4.8** | **49.2** | **23.4** | **1.2** |

### 判别任务（LLaVA-1.5-7B）

| 数据集 | 指标 | LLaVA Base | +VCD | +Octopus | 提升 vs Base |
|--------|------|-----------|------|----------|-------------|
| AMBER | Acc | 67.00 | 67.30 | **76.70** | +9.70 |
| AMBER | F1 | 71.10 | 71.10 | **82.70** | +11.60 |
| POPE (ALL) | Acc | 82.04 | 82.96 | **85.79** | +3.75 |
| POPE (ALL) | F1 | 80.42 | 81.81 | **83.44** | +3.02 |

### 关键发现

- Octopus 在 AMBER 数据集上将 CHAIR 指标从 8.0 降至 4.8，相比 Base 减少约 40% 的幻觉
- 相比需要重训练整个模型的方法（如 HA-DPO、HALVA），Octopus 仍大幅领先，且无需修改 LVLM 权重
- 消融实验证明：(1) 即使随机选择 CD 策略也有帮助，但远不如 Octopus 的自适应选择；(2) 增加更多"触手"（CD 策略）可以持续提升性能，框架具有良好的可扩展性
- 不同 RL 优化方法（DPO、Monte-Carlo、PPO）都能获得满意结果，说明框架对优化算法不敏感
- 不同的评判标准（CHAIR、Cover、平均分）都可作为正负样本划分依据，框架具有跨领域适应性

## 亮点与洞察

1. **诊断先于治疗的研究范式**：先通过系统的样本级和 token 级诊断实验揭示幻觉的混合成因，再据此设计解决方案，这种"先理解问题再解决问题"的思路值得借鉴
2. **优雅的"元策略"设计**：不是发明新的 CD 方法，而是设计一个"策略选择器"来组合现有 CD 方法，这种元学习的思想使框架具有天然的可扩展性——未来任何新 CD 方法都可以直接作为新的"触手"接入
3. **LVLM 权重完全冻结**：仅训练轻量的 decision module，不修改部署模型的任何参数，实用性极强
4. **DPO 的巧妙应用**：将策略选择问题转化为偏好学习问题，通过随机采样 + CHAIR 评估自动构造正负样本对，避免了人工标注

## 局限与展望

- 目前仅集成了三种 CD 策略（VCD、M3ID、AVISC），随着新 CD 方法的出现，候选策略空间可以进一步扩大
- DPO 训练数据通过随机采样构造，质量可能不够理想，可探索更高效的数据构造方式
- 多次前向传播（每种 CD 策略需要额外的 distorted input 推理）带来推理延迟，实际部署时需要权衡效率
- token 级动态选择的计算开销较大——每个 token 都需要运行 decision module + 对应的 CD 前向传播
- 框架的有效性依赖于候选 CD 策略的多样性和互补性，如果候选策略高度同质化则收益有限

## 相关工作与启发

- **vs VCD/M3ID/AVISC**：这三种 CD 方法是 Octopus 的"触手"，各自只能覆盖~60% 的幻觉样本。Octopus 的核心贡献是学会在不同情况下选择最合适的策略
- **vs 重训练方法（HACL, POVID, HA-DPO）**：这些方法需要构造高质量数据并重训练 LVLM 参数，成本高且不适用于已部署模型。Octopus 作为即插即用方案，甚至性能优于这些重量级方法
- **vs OPERA/LCD/ICD**：这些也是后处理方法，但仍采用单一策略。Octopus 的动态组合思路是本质性的提升
- **启发**：这种"元策略"的思想可以推广到其他领域——在任何存在多种互补解决方案的场景中，学一个策略选择器可能比设计一个更好的单一策略更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 将幻觉诊断和动态策略选择结合的思路很新颖，但核心 CD 策略仍是现有方法
- 实验充分度: ⭐⭐⭐⭐⭐ 诊断实验充分有说服力，主实验覆盖生成和判别两种任务，消融全面
- 写作质量: ⭐⭐⭐⭐ "章鱼"的类比贯穿全文，结构清晰，诊断实验的呈现方式很直观
- 价值: ⭐⭐⭐⭐ 作为通用框架具有良好的扩展性和实用性，对幻觉缓解研究有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucination in Multimodal LLMs](../../ACL2025/multimodal_vlm/mixture_of_decoding_an_attention-inspired_adaptive_decoding_strategy_to_mitigate.md)
- [Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/multimodal_vlm/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](../../ACL2025/multimodal_vlm/symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)
- [Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)

<!-- RELATED:END -->
