---
title: >-
  [论文解读] Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback
description: >-
   > 提出 Dolphin，一个闭环自动科研框架，包含"想法生成→实验验证→结果反馈"三阶段循环，通过任务属性引导的论文排序和异常回溯引导的调试流程，在 3D 分类等任务上自动提出并验证了接近人类设计 SOTA 的方法。
tags:

---

# Dolphin: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback

| 信息 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2501.03916](https://arxiv.org/abs/2501.03916) |
| 代码 | [GitHub](https://github.com/Alpha-Innovator/Dolphin) |
| 领域 | others (自动科研 × LLM Agent × 代码生成) |
| 关键词 | auto-research, closed-loop, idea generation, experimental verification, feedback |

## 一句话总结

> 提出 Dolphin，一个闭环自动科研框架，包含"想法生成→实验验证→结果反馈"三阶段循环，通过任务属性引导的论文排序和异常回溯引导的调试流程，在 3D 分类等任务上自动提出并验证了接近人类设计 SOTA 的方法。

## 研究背景与动机

- **科研范式变革**：AI 辅助科研正从"完全人类驱动"向"自动科研"演进，经历四个阶段：完全人驱 → AI 辅助 → 半自动 → 全自动。
- **现有工作的关键挑战**：
  1. **想法评估不准确**：大多工作 (Si et al., 2024; Li et al., 2024) 依赖人或 LLM 评估想法质量，但仅关注新颖性而非实验有效性。AI-Scientist (Lu et al., 2024) 虽做了实验验证但使用简单自建数据集，缺乏与同领域方法的有意义对比。
  2. **缺乏反馈机制**：人类研究者根据实验结果迭代改进想法，但现有工作要么在孤立的想法生成阶段内反馈，要么完全缺乏反馈。
- **核心动机**：构建一个真正的"闭环"系统——实验结果反馈到下一轮想法生成，模拟人类研究者的迭代研究过程。

## 方法详解

### 整体框架

Dolphin 由三个核心过程组成闭环（图 2）：

**1. 想法生成过程 (Ideas Generation Process)**

#### 论文检索与排序
- 使用 Semantic Scholar API 检索 50 篇相关论文
- **任务属性引导排序 (Task-Attribute-Guided Paper Ranking)**：
    - LLM 先提取输入任务的属性（模型输入、输出等特征）
    - 按两个标准对论文打分（1-10）：主题相关性 & 任务属性匹配度
    - 过滤掉 8 分以下的论文
    - 效果：对 3D 分类任务，3D 检测等不相关论文显著减少

#### 想法生成与过滤
- 基于已排序高相关论文，LLM 生成 N 个想法（每个含标题、实验计划、摘要）
- **独立性检查**：用句子嵌入计算想法间余弦相似度，阈值 0.8 过滤冗余
- 维护想法库 **B**（存储已检查过的想法嵌入）
- **新颖性检查**：通过 Semantic Scholar 搜索判断想法是否新颖

**2. 实验验证过程 (Experimental Verification Process)**

- LLM 生成详细实验计划并修改参考代码
- 核心创新：**异常回溯引导的调试过程 (Exception-Traceback-Guided Debugging)**
    - 直接喂 traceback 给 LLM 的执行成功率低（4/15），因 LLM 无法理解复杂的嵌套关系
    - 解决方案：从 traceback 中提取信息 → 引导 LLM 生成与错误相关的局部代码结构 → 基于代码结构和 traceback 进行修复
    - 仅关注自定义代码，排除库函数调用
    - 最多 5 次调试迭代
    - 效果：成功率从 33.3% 提升到 50.0%

**3. 结果反馈过程 (Results Feedback Process)**

- 将实验结果分为三类：提升、维持、下降
- 将维持/下降想法的嵌入加入想法库 B 以避免冗余验证
- 将提升性能的想法加入下一轮的想法生成 prompt
- 闭环效果：Loop 1 改进率 2/7 → Loop 3 改进率 4/8

## 实验

### 实验设置

- **LLM Agent**：GPT-4o-2024-08-06（想法生成）；DeepSeek-v2.5 via Ollama（代码执行）
- **任务**：
    - 3D 点云分类：ModelNet40 + PointNet 基线
    - 2D 图像分类：CIFAR-100 + WRN-28-10 基线
    - 情感分类：SST-2 + BERT-base 基线
- 每个任务做 2 轮循环（共 40 个想法）

### 主实验结果

| 任务 | 基线 | 平均提升 | 最大提升 | 人类设计 SOTA | 有效想法数 |
|------|------|---------|---------|-------------|-----------|
| ModelNet40 OA | 91.0 (PointNet) | 92.0 (+1.0) | **93.9 (+2.9)** | 93.8 (GPSFormer) | 5/40 |
| ModelNet40 mAcc | 87.6 (PointNet) | 88.7 (+1.1) | 91.1 (+3.5) | 91.8 (GPSFormer) | 5/40 |
| CIFAR-100 | 81.2 (WRN) | 81.8 (+0.6) | 82.0 (+0.8) | 82.2 (ResNeXt) | 6/40 |
| SST-2 | 91.0 (BERT-base) | 91.8 (+0.8) | 92.5 (+1.5) | 93.1 (BERT-large) | 6/40 |

**亮点结果**：在 ModelNet40 上自动生成的 PointNet-CSR 达到 93.9% OA，接近人类设计的 SOTA GPSFormer (93.8%)！

### MLE-bench 结果

| 任务 | 代码来源 | 之前分数 | Dolphin 分数 |
|------|---------|---------|------------|
| 社交侮辱检测 | AIDE | 81.0 | 84.7 |
| 表格预测 | Kaggle | 95.3 | 96.2 |
| 毒性评论分类 | Kaggle | 94.7 | 97.2 |

- Dolphin 可以灵活集成 AIDE、Agent Laboratory 等框架
- 能进行技术/代码版本更新

### 消融实验

#### 想法生成过程分析

| 方法 | 新颖想法数 | 平均成本/想法 |
|------|-----------|-------------|
| 朴素生成（无检索） | 8/20 | $0.106 |
| 朴素检索 + 生成 | 13/20 | $0.187 |
| 任务属性过滤（本文） | **19/20** | $0.184 |

- 任务属性过滤将新颖想法比例从 40% 提升到 95%

#### 调试过程分析

| 局部代码结构 | Traceback 信息提取 | 成功执行率 (L1/L2/L3) |
|-------------|-------------------|---------------------|
| ✗ | ✗ | 4/15, 5/13, 5/14 |
| ✓ | ✗ | 3/15, 5/13, 6/14 |
| ✓ | ✓ | **7/15, 6/13, 8/14** |

- 仅加局部代码结构不够（可能包含无关库信息），需要从 traceback 提取信息来聚焦自定义代码

#### 闭环反馈分析

| 循环 | Loop 1 | Loop 2 | Loop 3 |
|------|--------|--------|--------|
| 改进率 | 2/7 | 3/6 | 4/8 |

- 随迭代推进，想法质量持续提升，验证了闭环反馈的价值

### 案例分析：PointNet-CSR vs DGCNN

| 维度 | DGCNN（人类设计） | PointNet-CSR（Dolphin） |
|------|------------------|----------------------|
| 想法层次 | 架构级 | 模块级 |
| 参数 | 有可学习参数 | 无可学习参数 |
| 结构 | 重复块 | 单模块 |
| mAcc / OA | 90.2% / 92.9% | **91.1% / 93.9%** |
| 训练速度 | ~20.86s/epoch | **~6.12s/epoch (>3x faster)** |

PointNet-CSR 通过更简洁的架构实现了更好且更快的性能。

## 亮点与洞察

1. **真正的闭环**：实验结果 → 反馈 → 下一轮想法生成，是目前自动科研框架中少有的完整闭环设计。
2. **在公开 benchmark 上验证**：不同于 AI-Scientist 使用自建数据集，Dolphin 在 ModelNet40、CIFAR-100、SST-2 等标准 benchmark 上验证想法有效性。
3. **接近人类 SOTA**：3D 分类上自动生成的方法达到 93.9%，接近甚至匹配人类精心设计的方法。
4. **成本低廉**：每个想法的平均成本仅 ~$0.2。
5. **异常回溯引导调试**：解决了 LLM 理解复杂嵌套代码的成功率问题，从一般性的"喂 error log"提升为结构化的"局部代码分析"。

## 局限性

1. **知识泄漏**：LLM 的历史知识可能导致"重新发现"已有方法而非真正创新。
2. **仅用摘要和标题**：想法生成仅基于论文的标题和摘要，无法深入理解技术细节和论文间逻辑。
3. **代码能力限制**：LLM 无法理解复杂的项目级代码，限制了对复杂任务的验证能力。
4. **有效想法比例不高**：40 个想法中仅 5-6 个有效（12.5-15%），大量计算资源用于无效验证。
5. **任务范围有限**：验证的基线模型相对简单（PointNet, WRN, BERT-base），未在更复杂的现代架构上尝试。
6. **反馈信号简单**：仅用"提升/维持/下降"分类，缺乏对失败原因的深入分析。

## 相关工作

- **开放式科研**：AI-Scientist (Lu et al., 2024) 提出端到端框架但缺乏反馈和真实 benchmark 验证；Chain of Ideas (Li et al., 2024) 基于论文链生成想法但无实验验证；NOVA (Hu et al., 2024) 通过迭代优化提高新颖性。
- **受限科研**：AutoML-GPT (Zhang et al., 2023b) 用 LLM 做超参调优；AgentHPO (Liu et al., 2024) 迭代优化超参。
- **代码生成**：AIDE (Schmidt, 2024)、Agent Laboratory (Schmidgall et al., 2025) 分别在 ML 竞赛和实验室场景中自动生成代码。

## 评分 ⭐⭐⭐⭐

闭环设计理念正确且效果已验证（循环间改进率持续提升），在标准 benchmark 上接近人类 SOTA 的结果令人印象深刻。任务属性引导排序和异常回溯调试都是实用的技术贡献。主要限制在于有效想法比例不高、基线模型相对简单、以及 LLM 知识泄漏的隐患。但整体而言，这是向全自动科研迈出的扎实一步。

<!-- RELATED:START -->

## 相关论文

- [Learning Dynamics of RNNs in Closed-Loop Environments](../../NeurIPS2025/others/learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring](feat_a_preference_feedback_dataset_through_a_cost-effective_auto-generation_and_.md)
- [Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](../../ICML2025/others/regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

<!-- RELATED:END -->
