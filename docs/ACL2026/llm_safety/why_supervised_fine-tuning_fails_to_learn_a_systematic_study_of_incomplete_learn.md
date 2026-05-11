---
title: >-
  [论文解读] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models
description: >-
  [ACL 2026][不完全学习] 本文首次系统研究了 SFT 中的"不完全学习现象"（ILP）——即模型收敛后仍无法正确复现部分训练数据，识别了五种反复出现的原因（知识缺失、知识冲突、数据内部矛盾、左侧遗忘、不充分优化），并提出诊断框架和针对性缓解策略。
tags:
  - ACL 2026
  - 不完全学习
  - SFT诊断
  - 知识冲突
  - 遗忘
  - 微调失败模式
---

# Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.10079](https://arxiv.org/abs/2604.10079)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 不完全学习, SFT诊断, 知识冲突, 遗忘, 微调失败模式

## 一句话总结

本文首次系统研究了 SFT 中的"不完全学习现象"（ILP）——即模型收敛后仍无法正确复现部分训练数据，识别了五种反复出现的原因（知识缺失、知识冲突、数据内部矛盾、左侧遗忘、不充分优化），并提出诊断框架和针对性缓解策略。

## 研究背景与动机

**领域现状**：SFT 是 LLM 适应下游任务的标准方法，被广泛认为是可靠且高效的专业化机制。

**现有痛点**：(1) 即使训练损失完全收敛，模型仍频繁无法正确回答部分训练样本——这不是过拟合或泛化问题，而是在训练集本身上的失败；(2) 未学会的样本往往不是随机的，而是对应罕见案例、组合模式或知识密集型实例；(3) 聚合指标的提升可能掩盖持续存在的未学习子集。

**核心矛盾**：SFT 数据集（尤其在法律、医疗等专业领域）构建成本高昂，但 15.3%±2.1% 的样本在训练后仍未被学会——这直接降低了数据的利用率。

**本文目标**：不是提出新的微调算法，而是系统刻画、诊断和验证 SFT 中不完全学习的来源。

**切入角度**：将未学习的样本视为诊断信号而非噪声——通过分析为什么这些特定样本未被学会来理解 SFT 的局限性。

**核心 idea**：五种 ILP 来源各需不同的缓解策略——没有"一刀切"的解决方案，需要细粒度的样本级诊断。

## 方法详解

### 整体框架

三阶段框架：(1) SFT 微调至收敛；(2) 将训练数据转为选择题格式，用 pass@N 和 BoN 采样检测未学习样本；(3) 通过五种诊断测试归因未学习原因并施加针对性干预。

### 关键设计

1. **未学习样本检测（BoN-5 采样）**:

    - 功能：可靠识别训练后持续失败的样本
    - 核心思路：将 SFT 样本转为选择题，进行 N 次独立推理，计算 pass@N 比例。pass@5 < 0.2 且跨种子稳定的样本被标记为未学习。选 top-K=1000 最严重案例做深入分析
    - 设计动机：区分随机解码噪声和真正的学习失败——如果多次采样都失败，说明模型确实没有内化这个知识

2. **五种 ILP 来源的诊断与干预**:

    - 功能：将未学习样本归因到具体原因并验证因果关系
    - 核心思路：
        - **预训练知识缺失**：用 OpenIE 提取事实三元组，BoN 探测基础模型→知识增强+持续预训练
        - **知识冲突**：检测基础模型高置信度错误答案（与 SFT 标签矛盾）→外部知识纠正+CPT
        - **SFT 数据内部矛盾**：语义相似样本间标签不一致→GPT 评估+分桶训练避免 mini-batch 冲突
        - **左侧遗忘**：数据集顺序处理中早期样本被后期覆盖→随机打乱+动态重采样
        - **不充分优化**：罕见或复杂模式训练信号不足→增加训练或重新加权
    - 设计动机：缓解策略不是通用解决方案而是因果干预——如果策略 X 有效则验证了原因 Y

3. **知识状态探测（JSD 诊断）**:

    - 功能：区分知识缺失和知识冲突
    - 核心思路：计算基础模型和微调模型预测分布的 Jensen-Shannon 散度。高 JSD + 基础模型错误 = 知识冲突；低 JSD + 微调模型仍错 = 知识缺失
    - 设计动机：仅看最终准确率无法区分"不知道"和"知道但错误"——需要分布级信号

### 损失函数 / 训练策略

标准 SFT 交叉熵损失。在 Qwen、LLaMA、OLMo2 上评估。CPT 使用混合语料 $\mathcal{C}_{\text{mix}} = 0.8\mathcal{C}_{\text{general}} + 0.2\mathcal{C}_{\text{aug}}$。

## 实验关键数据

### 主实验

**ILP 普遍性（10 个基准 SFT 数据集平均）**

| 指标 | 数值 |
|------|------|
| 平均未学习比例 | 15.3% ± 2.1% |
| 跨模型一致性 | Qwen/LLaMA/OLMo2 均观察到 |
| 跨领域一致性 | 医疗/法律/金融均存在 |

### 消融实验

**CPT 干预效果（知识缺失+冲突类）**

| 领域 | SFT only Acc | +CPT Acc | 提升 |
|------|-------------|---------|------|
| 医疗 (MedQA) | baseline | 显著提升 | 验证知识缺失假说 |
| 法律 (LegalBench) | baseline | 显著提升 | 验证知识冲突假说 |
| 金融 (FinanceBench) | baseline | 显著提升 | — |

### 关键发现

- ILP 普遍存在且异质——没有单一干预能解决所有失败
- 知识缺失和知识冲突是最常见的两大原因，CPT 对这两类有效
- 左侧遗忘在多任务 SFT 中尤为严重——简单打乱数据顺序即可缓解大部分
- SFT 数据内部矛盾（标注不一致）导致的 ILP 通过分桶训练可部分解决
- 聚合指标的提升可以掩盖未学习子集的持续存在——需要样本级监控

## 亮点与洞察

- "ILP"的概念化本身是重要贡献——将一个广泛存在但未被系统研究的现象形式化
- 五种来源的分类学对 SFT 实践者有直接指导价值——可以对照检查自己的数据和模型
- 诊断优先于治疗的哲学——先理解为什么失败，再设计针对性修复

## 局限与展望

- 选择题格式的样本级评估可能引入评估偏差
- 缓解策略（CPT、分桶训练等）的计算成本未详细报告
- 五种来源的分类可能不完备——可能存在其他未识别的 ILP 原因
- 未分析 RLHF/DPO 等后续训练阶段是否会加剧或缓解 ILP

## 相关工作与启发

- **vs 灾难性遗忘**: 后者关注丢失已学能力，ILP 关注未能获取新知识——方向相反
- **vs 数据质量研究**: 后者通常关注提升整体性能，ILP 关注为什么特定样本学不会
- **vs 课程学习 (Bengio et al., 2009)**: 课程学习按复杂度排序训练，ILP 诊断表明排序本身不够——需要识别并处理五种不同的失败模式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究 SFT 不完全学习，概念和分类学都有原创性
- 实验充分度: ⭐⭐⭐⭐ 多模型 × 多领域 + 因果干预验证 + 10 个基准
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，诊断框架逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 对 SFT 实践和理论都有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](jailbreaking_large_language_models_with_morality_attacks.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](../../AAAI2026/llm_safety/anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](../../ICML2025/llm_safety/learning_safety_constraints_for_large_language_models.md)

</div>

<!-- RELATED:END -->
