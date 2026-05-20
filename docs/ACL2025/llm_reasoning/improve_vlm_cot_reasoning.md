---
title: >-
  [论文解读] Improve Vision Language Model Chain-of-thought Reasoning
description: >-
  [ACL 2025][LLM推理][Chain-of-thought] 通过(1)从GPT-4o蒸馏193K多任务CoT推理数据进行SFT，(2)利用模型自生成的推理链构建正负样本对进行DPO强化学习，显著提升VLM的链式推理能力，CoT预测平均+11.7%，同时直接回答也提升+7.3%。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "Chain-of-thought"
  - "CoT推理"
  - "知识蒸馏"
  - "DPO"
  - "强化学习"
  - "VLM"
---

# Improve Vision Language Model Chain-of-thought Reasoning

**会议**: ACL 2025  
**arXiv**: [2410.16198](https://arxiv.org/abs/2410.16198)  
**代码**: [https://github.com/RifleZhang/LLaVA-Reasoner-DPO](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)  
**领域**: LLM推理  
**关键词**: Chain-of-thought, CoT推理, 知识蒸馏, DPO, 强化学习, VLM  

## 一句话总结
通过(1)从GPT-4o蒸馏193K多任务CoT推理数据进行SFT，(2)利用模型自生成的推理链构建正负样本对进行DPO强化学习，显著提升VLM的链式推理能力，CoT预测平均+11.7%，同时直接回答也提升+7.3%。

## 研究背景与动机

**领域现状**：链式推理（CoT）对提升VLM的可解释性和可信度至关重要。然而现有VLM训练数据以短答案为主（如"14"），缺少详细的推理过程。

**现有痛点**：(1) **训练数据中推理步骤缺失**——标注者倾向于直接给出简短答案，写出完整推理过程更耗时；(2) **短答案训练无法隐式学到CoT**——作者实验发现，在ChartQA上训练26K直接回答，直接预测准确率提升2.9，但CoT仅提升0.6；(3) **缺乏高质量CoT训练数据**——现有VQA数据集几乎不包含推理步骤。

**核心矛盾**：VLM需要CoT推理能力，但(a)高质量CoT数据极度匮乏，(b)仅在短答案上训练的模型泛化不到需要详细推理的任务，(c)CoT推理的质量需要进一步校准。

**本文目标** 解决VLM CoT推理数据的匮乏问题并提升推理质量。

**核心 idea**：先蒸馏CoT数据做SFT教会模型推理，再用DPO利用模型自生成的推理正负样本对来校准推理质量。

## 方法详解

### 整体框架
三阶段流程（如Figure 2所示）：
- **Stage A**：从GPT-4o蒸馏CoT推理数据（ShareGPT-4o-Reasoning，193K样本）
- **Stage B**：用CoT和直接回答数据混合训练VLM（SFT），得到LLaVA-Reasoner-SFT
- **Stage C**：构建推理正负样本对，用DPO进一步校准推理质量，得到LLaVA-Reasoner-DPO

### 关键设计

1. **CoT数据蒸馏（ShareGPT-4o-Reasoning）**：
    - 覆盖9个数据集、4大推理技能：
        - 常识推理：A-OKVQA（16.9K）
        - 图表理解：ChartQA（26.0K）
        - 文档/文本理解：DocVQA（37.3K）、InfoVQA（22.4K）、TextVQA（29.7K）
        - 数学/科学：MathVision（11.0K）、G-LLaVA（30.3K）、SQA（6.1K）、AI2D（11.9K）
    - 蒸馏方式：输入（图像, 问题, 参考短答案）给GPT-4o，让其生成推理过程
    - 质量过滤：GPT-4o预测答案与标注不一致的样本被过滤（还发现了标注错误）
    - CoT响应平均约100 tokens，短答案通常<5 tokens

2. **SFT数据混合策略**：
    - 两种prompt模板：直接预测（"Answer with a short answer"）和CoT预测（"Generate a reason first and then output a short answer"）
    - 最优组合④：CoT + Direct + Format对齐数据（450条）+ LLaVA指令数据（2K）
    - 关键设计：CoT答案格式化为"推理过程... ### Answer: 最终答案"，便于自动提取

3. **DPO推理校准**：
    - 用SFT模型对每个问题生成32个候选CoT推理（temperature 1.0/1.2）
    - 比较每个推理的最终预测与标注答案：正确→正样本$y_w$，错误→负样本$y_l$
    - 只选择准确率在0.25-0.85之间的问题（太简单/太难的不适合DPO）
    - 每个问题最多3对，总计64.8K偏好数据对
    - DPO目标函数：$\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w|x,\mathcal{V})}{\pi_{\text{ref}}(y_w|x,\mathcal{V})} - \beta\log\frac{\pi_\theta(y_l|x,\mathcal{V})}{\pi_{\text{ref}}(y_l|x,\mathcal{V})})]$
    - **截断trick**：将响应截断至90 tokens效果最好

### 损失函数
- SFT阶段：标准因果语言模型损失
- DPO阶段：上述DPO损失函数，$\beta = 0.1$，学习率5e-7

## 实验

### 主实验——SFT数据组合消融（Table 2）

| 训练数据 | 推理方式 | A-OK | ChartQA | DocVQA | InfoVQA | TextVQA | AI2D | SQA | MathVista | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Format only ① | direct | 85.8 | 70.2 | 75.7 | 37.7 | 68.2 | 71.5 | 75.4 | 39.3 | 65.5 |
| Format only ① | CoT | 84.3 | 71.2 | 67.0 | 34.9 | 62.2 | 67.4 | 74.4 | 40.3 | 62.7 |
| Direct only ② | direct | 86.4 | 73.7 | 78.0 | 45.4 | 71.9 | 78.8 | 91.5 | 43.2 | 71.1 |
| Direct only ② | CoT | 85.7 | 71.8 | 68.8 | 38.6 | 63.6 | 72.5 | 85.4 | 38.6 | 65.6 |
| CoT only ③ | direct | 84.9 | 71.8 | 81.2 | 45.7 | 72.1 | 75.3 | 85.0 | 41.9 | 69.7 |
| CoT only ③ | CoT | 85.1 | 82.2 | 81.2 | 49.7 | 69.9 | 77.0 | 91.3 | 49.2 | 73.2 |
| **Both ④ (SFT)** | direct | 85.4 | 76.1 | 82.9 | 50.6 | 73.1 | 79.4 | 90.4 | 44.3 | **72.8** |
| **Both ④ (SFT)** | CoT | 86.2 | 83.0 | 81.8 | 51.6 | 71.1 | 78.5 | 92.7 | 50.6 | **74.4** |

### DPO实验（Table 6）

| 方法 | 推理方式 | A-OK | ChartQA | DocVQA | InfoVQA | TextVQA | AI2D | SQA | MathVista | 平均 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SFT ④ | CoT | 86.2 | 83.0 | 81.8 | 51.6 | 71.1 | 78.5 | 92.7 | 50.6 | 74.4 |
| + RLAIF-V ⑤ | CoT | 86.7 | 83.0 | 82.4 | 50.8 | 71.4 | 79.1 | 92.9 | 50.8 | 74.6 |
| **+ DPO-ours ⑥** | CoT | **87.0** | **84.2** | **82.7** | **52.7** | **71.5** | **79.5** | 92.6 | **52.1** | **75.3** |

### 与GPT-4o和SOTA对比（Table 5）

| 模型 | A-OK | ChartQA | DocVQA | SQA | MathVista | 平均(best) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 90.1 | 84.7 | 90.8 | 87.2 | 63.4 | 77.9 |
| Cambrian-7B | 83.1 | 73.3 | 77.8 | 80.4 | 49.0 | 64.5 |
| **LLaVA-Reasoner-SFT** | 86.2 | 83.0 | 82.9 | 92.7 | 50.6 | **68.8** |

### 关键发现
1. **短答案训练不能教会CoT推理**：Direct only训练直接预测+5.6，但CoT仅+2.9，且在计算密集任务上CoT甚至下降
2. **CoT训练也能提升直接预测**：CoT only训练的直接预测比Direct only训练还好（文档理解类任务尤为明显）
3. **混合训练最优**：同时用CoT和Direct数据训练，两种预测方式都达到最佳
4. **DPO有效但需要推理数据对**：用RLAIF-V通用DPO数据提升微小（+0.2），用自构建的推理数据对提升显著（+1.1）
5. **DPO模型可作为验证器**：用DPO reward score进行best-of-N重排和加权投票，进一步提升性能
6. **DPO学到了token级奖励**：credit assignment可视化显示DPO模型对推理中的首个错误/幻觉特别敏感

## 亮点
- **数据集贡献**：释放193K多任务CoT推理数据集ShareGPT-4o-Reasoning，社区可直接使用
- **关键发现**：实证证明短答案训练无法隐式学习CoT推理，需要显式CoT数据
- **方法通用**：SFT+DPO两阶段框架适用于任何VLM，不限于特定架构
- **DPO双用途**：既作为生成器改善推理质量，又作为验证器用于重排
- **实验极其充分**：主文+6个附录，覆盖数据消融、baseline对比、RFT vs DPO、prompt优化等

## 局限性
- 基座模型仅用LLaVA-NeXT-8B，未在更大模型上验证
- CoT蒸馏依赖GPT-4o，成本高，蒸馏质量受限于GPT-4o能力
- 部分任务（TextVQA、DocVQA、AI2D）CoT并不优于直接预测，可能因为简单事实提取不需要推理
- DPO数据仅用3个数据集构建偏好对，更多数据集的scaling留待未来
- 评估主要在VQA基准上，未覆盖开放式视觉推理场景

## 相关工作
- **VLM推理**：MAVIS、Visual CoT等关注特定领域（数学、定位）的推理训练
- **VLM/LLM对齐**：DPO、PPO用于减少幻觉和提升factuality；Step-DPO用于数学CoT推理
- **CoT数据**：现有VQA数据集几乎不含推理步骤，本文是首个大规模多任务VLM CoT蒸馏工作
- **本文定位**：首次系统研究VLM CoT推理的SFT+RL训练策略，填补VLM推理训练的空白

## 评分
- **创新性**: ⭐⭐⭐⭐ — 问题提出清晰（短答案不教CoT），方法虽不全新但组合有效
- **实用性**: ⭐⭐⭐⭐⭐ — 193K CoT数据集+训练流程直接可用，代码开源
- **技术深度**: ⭐⭐⭐⭐ — SFT数据混合消融极其细致，DPO分析深入（credit assignment）
- **实验充分度**: ⭐⭐⭐⭐⭐ — 主文+6个附录，超级充分
- **总体推荐**: ⭐⭐⭐⭐⭐ — VLM推理训练的重要参考工作，数据和代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](../../CVPR2025/llm_reasoning/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)
- [\[ACL 2025\] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](unveiling_the_key_factors_for_distilling_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Critic-CoT: Boosting the Reasoning Abilities of Large Language Model via Chain-of-Thoughts Critic](critic-cot_boosting_the_reasoning_abilities_of_large_language_model_via_chain-of.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](../../CVPR2025/llm_reasoning/argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)

</div>

<!-- RELATED:END -->
