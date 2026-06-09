---
title: >-
  [论文解读] Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions
description: >-
  [ACL 2025][目标检测][锚定偏差] 首次从失败案例角度对GPT-2系列在MCQ中的"锚定偏差"（始终选A）进行机械分析，通过Logit Lens定位到MLP中存储"A"偏好的特定值向量，用极简干预（更新值向量）将MCQ准确率平均提升70%+。
tags:
  - "ACL 2025"
  - "目标检测"
  - "锚定偏差"
  - "GPT-2"
  - "机械可解释性"
  - "Logit Lens"
  - "MLP值向量"
---

# Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions

**会议**: ACL 2025  
**arXiv**: [2405.03205](https://arxiv.org/abs/2405.03205)  
**代码**: [GitHub](https://github.com/ruizheliUOA/Anchored_Bias_GPT2)  
**领域**: 目标检测  
**关键词**: 锚定偏差, GPT-2, 机械可解释性, Logit Lens, MLP值向量

## 一句话总结

首次从失败案例角度对GPT-2系列在MCQ中的"锚定偏差"（始终选A）进行机械分析，通过Logit Lens定位到MLP中存储"A"偏好的特定值向量，用极简干预（更新值向量）将MCQ准确率平均提升70%+。

## 研究背景与动机

**领域现状**: LLM在MCQ中存在位置偏差，正确答案位置影响预测准确率。**现有痛点**: 已有研究通过prompt工程缓解偏差但未深入分析内部机制；机械可解释性工作（Lieberum et al.）仅关注成功案例。**核心矛盾**: GPT-2系列存在极端锚定偏差——始终倾向输出"A"——具有跨数据集的高度规律性但内部机制未知。**本文目标**: 从失败案例角度定位GPT-2内部导致锚定偏差的具体模块并修复。**切入角度**: 将MLP视为键-值记忆，用Logit Lens逐层追踪偏差源。**核心idea**: MLP值向量在预训练中存储了对"A"的偏好，在MCQ格式下表现为锚定偏差，可通过直接编辑消除。

## 方法详解

### 整体框架

三阶段：(1) 5个MCQ数据集上定量确认锚定偏差普遍性；(2) Logit Lens定位MLP层+维度和注意力头中的偏差源；(3) 值向量更新和注意力权重交换干预。

### 关键设计

1. **MLP偏差定位**:
    - 功能：定位存储"A"偏好的特定MLP层和维度
    - 核心思路：计算logit差异 $\text{logit}_T^\ell[\text{A}](\mathbf{m}_T^\ell) - \text{logit}_T^\ell[\text{B/C/D/E}](\mathbf{m}_T^\ell)$找到关键层，用MLP Contribution $|\mathbf{k}_T^{\ell,n}| \|\mathbf{v}_T^{\ell,n}\|$定位维度，对值向量做unembedding验证top-10 token
    - 设计动机：MLP作为key-value memory，值向量的top tokens如果是"A"相关词则直接证明是偏差存储位置

2. **MLP值向量更新**:
    - 功能：直接修改值向量消除"A"偏好
    - 核心思路：$\mathbf{v}^{\ell,n} = \mathbf{v}^{\ell,n} - \lambda_1 W_U[\text{A}] + \lambda_2 W_U[\text{B/C/D/E}]$，$\lambda_1=1, \lambda_2=8$
    - 设计动机：直接在知识存储层面"重写"偏差信息，无需重训模型

3. **注意力权重交换**:
    - 功能：交换A和正确答案位置的加权注意力值
    - 核心思路：$\mathbf{r}_{T,p(\text{A})}^{\ell,h} \leftrightarrow \mathbf{r}_{T,p(\text{B/C/D/E})}^{\ell,h}$
    - 设计动机：注意力头对A位置有额外关注，交换可进一步消除

### 损失函数 / 训练策略

不涉及训练。$\lambda_2$通过消融实验确定，从2到8准确率持续提升。

## 实验关键数据

### 主实验

MLP值向量更新后的MCQ准确率（%）：

| GPT-2 | 修改向量 | IOI(2选) | LD(3选) | Greater(4选) | ARC(4选) | CSQA(5选) |
|-------|---------|:---:|:---:|:---:|:---:|:---:|
| Small | v9,1853 | 100 | 100 | 100 | 100 | 100 |
| Large | v34,1541 | 100 | 100 | 100 | 96.7 | 99.7 |
| XL | v44,4967 | 98.2 | 100 | 100 | 90.7 | 94.8 |

### 消融实验

锚定偏差在GPT-2全系列的发生率（%）：

| 数据集 | Small | Medium | Large | XL |
|-------|:---:|:---:|:---:|:---:|
| IOI(2选) | 45.5 | 97.4 | **100** | 85.8 |
| ARC(4选) | 54.6 | 91.6 | 97.6 | 69.9 |
| CSQA(5选) | 34.8 | 81.5 | 99.6 | 97.7 |

### 关键发现

1. **MLP是主要偏差源**: 特定层的值向量直接存储了对"A"的偏好知识
2. **仅修改1-2个值向量即可消除偏差**: Small修改v9,1853后5个数据集全部100%
3. **注意力头起辅助作用**: 在IOI上有效（Medium达92.47%），其他数据集效果有限
4. **干预对一般能力影响可控**: 原始IOI任务仍保持85.8%准确率

## 亮点与洞察

- **从失败案例分析**与成功案例分析互补——理解弱点比理解长处更实用
- **Logit Lens精准定位到特定维度**——极高可操作性
- **最小干预高回报**: 改1个值向量从0%到100%
- **完整偏差电路追踪**: GPT-2全系列的MLP+注意力头可视化图

## 局限与展望

- 仅GPT-2系列（124M-1.5B），更大模型机制可能不同
- 注意力交换需要事先知道正确答案位置
- 值向量更新对一般能力有一定损害
- 不同架构（MoE/Mamba）的偏差存储位置可能完全不同

## 相关工作与启发

- **vs Lieberum et al. 2023**: 分析成功案例的"正确字母头"——本文分析失败案例，互补
- **vs PriDe（Zheng et al. 2024）**: 推理时去偏——本文直接修改参数
- **启发**: MLP值向量不仅存储事实知识也存储偏差——模型编辑的双刃剑

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从失败案例角度机械分析MCQ偏差
- 实验充分度: ⭐⭐⭐⭐ GPT-2全系列×5+数据集×多种干预
- 写作质量: ⭐⭐⭐⭐⭐ 分析层层深入
- 价值: ⭐⭐⭐⭐ 对LLM可解释性和偏差缓解有方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multiple Object Tracking as ID Prediction](../../CVPR2025/object_detection/multiple_object_tracking_as_id_prediction.md)
- [\[ECCV 2024\] Rectify the Regression Bias in Long-Tailed Object Detection](../../ECCV2024/object_detection/rectify_the_regression_bias_in_long-tailed_object_detection.md)
- [\[AAAI 2026\] Towards Multiple Missing Values-Resistant Unsupervised Graph Anomaly Detection](../../AAAI2026/object_detection/towards_multiple_missing_values-resistant_unsupervised_graph_anomaly_detection.md)
- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](../../CVPR2026/object_detection/ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)
- [\[AAAI 2026\] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking](../../AAAI2026/object_detection/when_trackers_date_fish_a_benchmark_and_framework_for_underwater_multiple_fish_t.md)

</div>

<!-- RELATED:END -->
