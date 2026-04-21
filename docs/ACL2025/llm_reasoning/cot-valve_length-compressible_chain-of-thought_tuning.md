---
title: >-
  [论文解读] CoT-Valve: Length-Compressible Chain-of-Thought Tuning
description: >-
  [ACL 2025][LLM推理][Chain-of-Thought压缩] 本文提出CoT-Valve，一种通过在参数空间中识别"长度控制方向"（以LoRA实现）来弹性控制推理链长度的方法，仅训练一次即可生成从长到短不同长度的推理路径，在QwQ-32B-Preview上将GSM8K推理链从741压缩至225 tokens且准确率仅降0.15%（95.07%→94.92%）。
tags:
  - ACL 2025
  - LLM推理
  - Chain-of-Thought压缩
  - 推理长度控制
  - LoRA
  - 参数空间方向
  - 测试时计算效率
---

# CoT-Valve: Length-Compressible Chain-of-Thought Tuning

**会议**: ACL 2025  
**arXiv**: [2502.09601](https://arxiv.org/abs/2502.09601)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: Chain-of-Thought压缩, 推理长度控制, LoRA, 参数空间方向, 测试时计算效率

## 一句话总结
本文提出CoT-Valve，一种通过在参数空间中识别"长度控制方向"（以LoRA实现）来弹性控制推理链长度的方法，仅训练一次即可生成从长到短不同长度的推理路径，在QwQ-32B-Preview上将GSM8K推理链从741压缩至225 tokens且准确率仅降0.15%（95.07%→94.92%）。

## 研究背景与动机
Chain-of-Thought推理显著增强了模型的推理能力，但代价是推理链过长导致推理成本高昂。核心观察是：**推理模型对简单任务分配了过多的token，而对复杂任务可能token不足**。例如QwQ在GSM8K（简单数学）上平均用741 tokens，但在AIME（竞赛数学）上用6827 tokens。

现有的推理链压缩方法存在以下问题：
- **直接移除中间步骤+训练**会降低性能
- **Prompt-based控制**效果有限——即使要求"20字以内"，模型可能输出350+ tokens，且无法生成真正短的推理
- **蒸馏到System 1**在省略中间步骤时未观察到改进
- **SimPO优化、RL剪枝**等方法需要额外训练且控制粒度有限

本文的核心idea：**在参数空间中存在一个"方向"Δθ，沿这个方向走大步就生成短链，走小步就生成长链**。通过LoRA实现这个方向的控制，作为可调"阀门"（Valve），仅需一次训练即可生成任意长度的推理链。

## 方法详解

### 整体框架
CoT-Valve分为两个阶段：
1. **Stage 1：确定长度控制方向Δθ**：通过蒸馏或后训练获得LoRA参数，这个参数差即为Δθ
2. **Stage 2：增强控制精度**：利用Δθ构建MixChain数据集（同一问题的不同长度推理链），再用两种增强方法（CoT-Valve++精确控制 或 CoT-Valve+P渐进压缩）细化

推理时通过调节LoRA的缩放因子α来控制链长——α=0不加LoRA（原始长链），α=1完全加载（短链），α>1外推得到更短的链。

### 关键设计

1. **参数空间中的长度方向**

    - 训练目标：找到参数更新 $\Delta\theta$ 使模型生成更短的推理链但仍得到正确答案
    - $\Delta\theta$ 被解释为一个"任务向量"——任务是"控制CoT长度"
    - 用LoRA实现：低rank的外部分支，调节其强度α即可控制推理链长度
    - 关键性质：**可内插和外推**——α∈(0,1)平滑过渡长短链，α>1可进一步缩短到训练未见的长度

2. **MixChain数据集构建**

    - 利用已训练的CoT-Valve在不同α值下生成同一问题的多种长度推理链
    - 两种构建场景：
        - **Cold-start（MixChain-C）**：有标注数据集（如GSM8K）时，先用标注训练基础模型，再用不同α生成
        - **Zero-shot（MixChain-Z）**：无标注时，利用基础LLM和对应推理模型之间的参数差作为Δθ（如LLaMA-3.1-8B vs DeepSeek-R1-Distill-Llama-8B）
    - 过滤掉答案错误的推理链

3. **CoT-Valve++：精确控制**

    - 在MixChain上训练时，引入归一化因子β来表示推理链长度：$\beta = 1 - \frac{m - m_{min}}{m_{max} - m_{min}}$
    - 训练目标要求在所有β值下都能生成对应长度的正确推理：$\max_{\Delta\theta'} \mathbb{E} p(a|t_{<m}, q; \theta + \beta\Delta\theta')$
    - 解决了原始CoT-Valve中"只在α=1训练但在所有α推理"的训练-推理不一致问题

4. **CoT-Valve+P：渐进压缩**

    - 类似模型压缩中的迭代剪枝思路
    - 每个epoch使用MixChain中更短的推理链训练，逐步压缩而非直接跳到最短
    - 5个epoch依次使用Solution 4→3→2→1→0（ground truth），最终准确率从直接训练的92.19%提升至94.92%

### 损失函数 / 训练策略
- 使用标准语言建模损失，核心区别在于训练数据和LoRA缩放因子的设计
- 大多数实验使用LoRA微调，LIMO实验使用全参数微调
- 效率指标ACU（Accuracy per Computation Unit）：ACU = Accuracy / (#Params × #Tokens)

## 实验关键数据

### 主实验（QwQ-32B-Preview on GSM8K）

| 方法 | 准确率 | Token数 | ACU↑ |
|------|--------|---------|------|
| QwQ-32B-Preview原始 | 95.07% | 741 | 0.40 |
| Prompt控制(Han) | 93.6% | 355 | 0.82 |
| Overthink-SimPO | 94.8% | 326 | 0.91 |
| O1-Pruner(RL) | 96.5% | 534 | 0.56 |
| **CoT-Valve++ MixChain-C** | **94.4%** | **276** | **1.07** |
| **CoT-Valve+P MixChain-Z** | **94.9%** | **225** | **1.32** |

### AIME24（QwQ-32B-Preview）

| 方法 | 得分 | Token数 | ACU↑ |
|------|------|---------|------|
| QwQ-32B-Preview原始 | 14/30 | 6827 | 0.021 |
| Overthink | 13/30 | 5154 | 0.026 |
| **CoT-Valve+P** | **13/30** | **4630** | **0.029** |

### 小模型蒸馏（LLaMA-3.2-1B）

| 方法 | 准确率 | Token数 | ACU↑ |
|------|--------|---------|------|
| SFT - QwQ蒸馏 | 52.7% | 759 | 6.94 |
| CoT-Valve - QwQ蒸馏 | 55.5% | 267 | 20.79 |
| **CoT-Valve - MixChain Solution 1** | **58.9%** | **275** | **21.39** |

### 消融实验（渐进压缩 vs 直接训练）

| 方法 | 准确率 | Token数 |
|------|--------|---------|
| 直接用最短链训练5 epochs | 92.19% | 250 |
| 渐进压缩(4→3→2→1→0) | **94.92%** | 225 |

### 训练数据长度影响（LLaMA-3.2-1B）

| 训练链长度 | 准确率 | Token数 |
|-----------|--------|---------|
| Ground-Truth (116 tokens) | 43.8% | 139 |
| Solution 1 (280 tokens) | **57.0%** | 288 |
| Solution 4 (497 tokens) | 52.5% | 558 |

### 关键发现
- **短推理链有时优于长链**：在GSM8K上，CoT-Valve生成的短链（267 tokens）比原始QwQ长链（741 tokens）准确率反而更高（55.5% vs 52.7%，LLaMA-3.2-1B）
- **不是所有推理链都适合训练**：过短或过长的链都不理想，中等长度（Solution 1, ~280 tokens）效果最好，尤其对小模型
- **渐进压缩显著优于直接压缩**：准确率从92.19%提升至94.92%
- **CoT-Valve的可外推性**：α>1可以生成比训练集更短的链
- **CoT-Valve实现了Prompt无法达到的短链**：Prompt最短只能到355 tokens，CoT-Valve可以到133.8 tokens
- **Long-Short-Long策略有效**：先训练长链再压缩（Short-Long-Short）比直接训练短链效果更好

## 亮点与洞察
- **"参数空间中的长度方向"概念非常优雅**：将推理链长度控制转化为参数空间中的向量算术，与task arithmetic/model merging等工作形成一致的理论框架
- **LoRA作为"阀门"的直觉类比**：旋转阀门调节流量大小，调节α控制推理链长度，让方法非常直观
- **ACU指标的提出**：综合考虑准确率、参数量和token数的效率指标，对比推理模型更公平
- **MixChain数据集的自生成机制**：不需要额外采样，利用CoT-Valve本身生成不同长度的链，自举性强
- **"不是所有正确链都适合训练"的发现**对蒸馏研究有重要启示

## 局限与展望
- 目前仅在数学推理（GSM8K、AIME）上验证，代码、科学推理等领域未覆盖
- 当前按整体链控制长度，未实现对链内不同部分的差异化压缩（简单部分多压缩、复杂部分保留）
- α值的最优选择仍需根据任务和数据集手动调整
- AIME上压缩后性能有明显下降（14/30→13/30），复杂任务的推理链压缩还有挑战
- **研究idea**：可以结合reward model做自适应链长控制——基于问题难度自动选择α值（简单问题α大生成短链，复杂问题α小保留长链），实现真正的"按需推理"

## 相关工作与启发
- Overthinking（Chen et al., 2024）：识别QwQ的"过度思考"问题，用SimPO优化，但压缩比不如CoT-Valve
- O1-Pruner（Luo et al., 2025）：用RL缩短推理，准确率更高但token数也更多（534 vs 225）
- Kimi K1.5：提出训练无关的长短CoT模型融合，理念与CoT-Valve互补
- Task Arithmetic（Ilharco et al., 2022）：CoT-Valve的理论基础，证明参数空间中的方向可编码任务
- 本文的核心贡献是证明了**推理链长度可以被编码为参数空间中的一个可控方向**，这一发现为推理效率优化开辟了新的技术路线

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 参数空间长度方向的概念非常新颖，CoT-Valve的设计优雅简洁
- 实验充分度: ⭐⭐⭐⭐ 多模型（QwQ、R1-Distill、LLaMA、Qwen）、多场景（长→短、短→长、短→长→短）、丰富消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰直观，实验安排有逻辑，但公式符号偶有混乱
- 价值: ⭐⭐⭐⭐⭐ 解决了推理模型推理成本高这一核心痛点，ACU提升显著（0.40→1.32），实用价值极高

<!-- RELATED:START -->

## 相关论文

- [TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)
- [Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](../../NeurIPS2025/llm_reasoning/transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)
- [CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](../../NeurIPS2025/llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)

<!-- RELATED:END -->
