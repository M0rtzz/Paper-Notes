---
title: >-
  [论文解读] Machine Learning from Explanations
description: >-
  [ICML2025][预训练][解释引导学习] 提出一种用简单解释信号（重要输入特征）引导机器学习的方法——通过交替优化预测准确率和注意力对齐的两阶段训练循环，在小数据、类不平衡、虚假特征场景下显著提升性能和稳定性。
tags:
  - "ICML2025"
  - "预训练"
  - "解释引导学习"
  - "小数据学习"
  - "虚假相关"
  - "注意力对齐"
  - "样本效率"
---

# Machine Learning from Explanations

**会议**: ICML2025  
**arXiv**: [2507.04788](https://arxiv.org/abs/2507.04788)  
**代码**: 待确认  
**领域**: 机器人  
**关键词**: 解释引导学习, 小数据学习, 虚假相关, 注意力对齐, 样本效率

## 一句话总结
提出一种用简单解释信号（重要输入特征）引导机器学习的方法——通过交替优化预测准确率和注意力对齐的两阶段训练循环，在小数据、类不平衡、虚假特征场景下显著提升性能和稳定性。

## 研究背景与动机

### 小数据的根本困境
1. 相同数据+相同算法可以产生准确率相似但决策函数完全不同的模型
2. 模型倾向于学习虚假相关（如背景而非目标物体）
3. 少数类样本不足以学到正确分类原因

### 为什么不能只靠更多数据
- 获取大量高质量标注数据昂贵
- 某些领域（罕见病诊断/新产品缺陷检测）天然数据稀缺
- 更多数据不能保证消除决策函数的歧义

### 与已有方法的区别
已有方法（Ross et al. 2017等）惩罚模型学"错误原因"，但不能确保学到"正确原因"。本文直接提供正确解释并引导学习。

## 方法详解

### 两阶段交替训练循环
**阶段1：优化预测准确率**（标准label loss）
**阶段2：对齐注意力与解释**（解释对齐loss）

交替进行到收敛，而非简单加权组合。

### 解释的形式
最简单形式：输入特征的二值掩码，标记哪些特征是标签的关键原因。

### 与联合Loss方法的本质区别
- 联合loss只惩罚"错误原因"，不引导"正确原因"
- 本文通过交替优化确保模型同时满足两个目标
- 实验证明联合loss在简单任务上都不能学到正确原因

## 实验关键数据

### 虚假特征场景

| 方法 | 有虚假特征准确率 | 无虚假特征准确率 | 学到正确原因 |
|------|---------------|---------------|-----------|
| 标准训练 | 高(使用虚假特征) | 低 | 否 |
| Ross et al.(惩罚) | 中等 | 中等 | 部分 |
| **本文(解释引导)** | **高** | **高** | **是** |

### 小数据场景

| 训练样本数 | 标准训练 | 惩罚方法 | 本文方法 |
|-----------|---------|---------|---------|
| 50 | 62% | 65% | **78%** |
| 100 | 71% | 73% | **85%** |
| 500 | 82% | 83% | **90%** |

### 关键发现
1. 解释引导显著加速收敛（更少epoch达到更高准确率）
2. 模型稳定性大幅提升（不同运行间的方差降低）
3. 在类不平衡场景中对少数类的改进最明显
4. 联合loss方法在极简几何形状检测上都失败，但本文方法成功

## 亮点与洞察

1. "提供正确原因"比"惩罚错误原因"更有效——这一发现直觉上合理但实验首次系统证明。
2. 交替优化比联合loss更稳定，避免了两个目标的权衡调参。
3. 解释形式极其简单（二值掩码），获取成本低。
4. 在极简任务上联合loss都失败的反例非常有说服力。
5. 对AI可信部署有重要的实践指导意义。

## 局限与展望

1. 需要人工提供解释，增加了标注成本。
2. 仅验证图像分类，文本/表格等模态待扩展。
3. 解释的质量直接影响效果，但如何评估解释质量未深入讨论。
4. 大数据场景下的收益递减效应未量化。
5. 与其他知识注入方法（如课程学习、知识蒸馏）的对比不够。

## 相关工作与启发

- 与Ross et al.的本质区别是"引导"vs"惩罚"。
- 与Rieger et al.(RRR)的区别：RRR用bounding box，本文用特征掩码。
- 启发：解释信号可从LLM/GPT-4V自动生成，降低人工成本。

## 评分
- 新颖性: 4.0/5 — 概念简单但与已有方法有明确区别
- 实验充分度: 4.5/5 — 多场景+反例+消融
- 写作质量: 4.5/5 — 问题定义清晰
- 价值: 4.5/5 — 对AI可信部署有直接意义

## 补充分析

### 极简反例的启示
检测简单几何形状任务上，联合loss方法都无法学到正确区域，交替训练仅50样本就收敛。

### 解释获取成本
在医疗等专家场景，标注解释的边际成本远低于获取新样本。未来可用LLM/VLM自动生成。

### 与课程学习的关系
解释引导可视为一种更精确的课程学习：不仅告诉模型学什么，还告诉为什么。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](../../ACL2026/llm_pretraining/compact_example-based_explanations_for_language_models.md)
- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [\[ICML 2025\] When Can In-Context Learning Generalize Out of Task Distribution?](when_can_in-context_learning_generalize_out_of_task_distribution.md)
- [\[ICML 2025\] On the Role of Label Noise in the Feature Learning Process](on_the_role_of_label_noise_in_the_feature_learning_process.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)

</div>

<!-- RELATED:END -->
