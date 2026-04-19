---
title: >-
  [论文解读] De-mark: Watermark Removal in Large Language Models
description: >-
  [ICML2025][AI安全][水印去除] 提出De-mark框架，通过随机选择探测(random selection probing)策略估计n-gram水印强度并重建红绿列表，无需知道哈希函数即可去除水印，并提供去除后LM分布与原始分布之间的理论差距保证。
tags:
  - ICML2025
  - AI安全
  - 水印去除
  - LLM水印
  - n-gram水印
  - 红绿列表
  - 水印窃取
---

# De-mark: Watermark Removal in Large Language Models

**会议**: ICML2025  
**arXiv**: [2410.13808](https://arxiv.org/abs/2410.13808)  
**代码**: [GitHub - De-mark](https://github.com/RayRuiboChen/De-mark)  
**领域**: ai_safety  
**关键词**: 水印去除, LLM水印, n-gram水印, 红绿列表, 水印窃取

## 一句话总结
提出De-mark框架，通过随机选择探测(random selection probing)策略估计n-gram水印强度并重建红绿列表，无需知道哈希函数即可去除水印，并提供去除后LM分布与原始分布之间的理论差距保证。

## 研究背景与动机

### 水印的脆弱性
n-gram水印(Kirchenbauer et al.)通过偏向绿色token来嵌入可检测信号，但其鲁棒性被显著高估。如果对手能逆向工程水印规则，就能有效去除水印。

### 已有水印窃取方法的局限
Jovanovic et al.的先驱工作需要知道底层哈希函数，且依赖释义工具去除水印（无法保持原始LM分布）。

### De-mark的改进
完全不需要已知哈希函数，提供分布级别的理论保证，还可进行水印利用（让其他模型生成水印内容）。

## 方法详解

### 随机选择探测(Random Selection Probing)
向水印LM发送精心设计的查询序列，通过统计token频率重建每个n-gram上下文的红绿列表分割和水印强度delta。

### 水印强度估计
提出无偏估计器，可准确评估水印偏移量delta，理论保证估计的一致性。

### 水印去除
基于估计的红绿列表和delta，在推理时修正token概率分布，将绿色token的logit偏移还原。

### 理论保证
证明去除后的LM分布与原始无水印分布之间的KL散度有界。

### 水印利用
反过来应用估计的水印规则到另一个LM，使其生成带水印的内容——展示了水印方案的脆弱性。

## 实验关键数据

### 水印去除效果（Llama3/Mistral）

| 模型 | 去除前z-score | 去除后z-score | 文本质量保持 |
|------|-------------|-------------|-----------|
| Llama3-8B | 4.2 | 0.3 | 高 |
| Mistral-7B | 3.8 | 0.4 | 高 |

### 与已有方法对比

| 方法 | 需要哈希函数 | 保持分布 | 效果 |
|------|-----------|---------|------|
| Jovanovic et al. | 是 | 否 | 中等 |
| **De-mark** | **否** | **是(理论保证)** | **高** |

### ChatGPT案例研究
在商业级LLM上也成功去除水印，验证了方法的实际威胁。

### 关键发现
1. 随机选择探测比频率分析更高效
2. 理论分布保证使去除后文本自然
3. 水印利用功能揭示严重安全威胁
4. 方法对不同n-gram长度(1-4)都有效

## 亮点与洞察

1. 从攻击者角度揭示水印方案的根本脆弱性。
2. 不依赖任何先验知识（哈希函数/水印参数）。
3. 理论保证（KL散度有界）使方法可信。
4. 水印利用能力使问题更严峻——攻击者可伪造水印。
5. ChatGPT实验说明商业系统也不安全。

## 局限与展望

1. 需要较多查询次数来估计红绿列表。
2. 对language-aware或语义水印的适用性未验证。
3. 水印利用的伦理风险需要社区讨论。
4. 防御对策（如何使水印更鲁棒）未深入探讨。
5. 多水印叠加场景未考虑。

## 相关工作与启发

- 与distortion-free水印的关系：De-mark针对有偏移类水印。
- 与语义水印的对比：语义水印可能更难去除。
- 启发：水印设计应考虑对抗性探测的鲁棒性。

## 评分
- 新颖性: 4.5/5 — 不依赖先验知识的水印去除+理论保证
- 实验充分度: 4.5/5 — 多模型+商业LLM验证
- 写作质量: 4.5/5
- 价值: 5.0/5 — 对水印安全有重要警示

## 补充技术细节

### 随机选择探测的工作原理
向LM发送精心设计的前缀序列，观察哪些token被系统性偏好。通过多次探测积累统计信号，重建每个n-gram上下文的红绿分割。

### KL散度的理论保证
去除水印后的分布与原始LM分布之间的KL散度随delta估计误差和查询次数的增加而收缩。

### 对防御的启示
建议水印设计应考虑对随机探测的鲁棒性，例如使用随机化哈希或多级水印。

<!-- RELATED:START -->

## 相关论文

- [Improved Unbiased Watermark for Large Language Models](../../ACL2025/ai_safety/improved_unbiased_watermark_for_large_language.md)
- [Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)
- [Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models](ferret_federated_full-parameter_tuning_at_scale_for_large_language_models.md)
- [Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)
- [Ensemble Watermarks for Large Language Models](../../ACL2025/ai_safety/ensemble_watermarks_llm.md)

<!-- RELATED:END -->
