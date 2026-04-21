---
title: >-
  [论文解读] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection
description: >-
  [CVPR 2025][OOD检测] 提出H2ST方法，用层次化的两样本检验架构实现增量学习中的OOD检测——每个任务对应一个特征级别的源-目标二分类器层，通过Clopper-Pearson置信区间假设检验自动判定ID/OOD（无需手动阈值），同时提供任务ID预测能力，在7个基准上优于MSP/Energy/ODIN且计算效率提升$(T+1)/2$倍。
tags:
  - CVPR 2025
  - OOD检测
  - 增量学习
  - 两样本检验
  - 层次化架构
  - 无阈值
---

# H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection

**会议**: CVPR 2025  
**arXiv**: [2503.14832](https://arxiv.org/abs/2503.14832)  
**代码**: https://github.com/YuhangLiuu/H2ST  
**领域**: 其他  
**关键词**: OOD检测、增量学习、两样本检验、层次化架构、无阈值

## 一句话总结
提出H2ST方法，用层次化的两样本检验架构实现增量学习中的OOD检测——每个任务对应一个特征级别的源-目标二分类器层，通过Clopper-Pearson置信区间假设检验自动判定ID/OOD（无需手动阈值），同时提供任务ID预测能力，在7个基准上优于MSP/Energy/ODIN且计算效率提升$(T+1)/2$倍。

## 研究背景与动机

**领域现状**：任务增量学习（TIL）假设数据是闭集分布内的，但开放世界中模型必须识别OOD样本。现有OOD检测方法（MSP、Energy、ODIN）依赖阈值选择、严重依赖模型性能、且无法提供任务级别的ID。

**现有痛点**：（1）阈值敏感——不同部署环境需要不同阈值，手动调参不可行；（2）依赖模型输出——softmax/energy score受模型过拟合影响大；（3）仅做二元OOD判定，不能识别具体是哪个任务的OOD。

**核心矛盾**：增量学习中任务持续增加，OOD检测的复杂度和任务ID识别的需求同步增长，但现有方法没有设计增量式的OOD检测机制。

**本文目标** 设计一个无阈值、可增量扩展、且能同时检测OOD和预测任务ID的统计检测框架。

**切入角度**：用两样本检验（比较测试样本特征分布与训练数据分布是否一致）替代基于score的阈值判定，用层次化架构实现增量扩展和早退出机制。

**核心 idea**：用层次化串联的两样本检验层做特征级OOD检测，每层对应一个任务，通过Clopper-Pearson置信区间自动判定ID/OOD（无需阈值），早退出机制同时给出任务ID。

## 方法详解

### 整体框架
$T$个任务对应$T$个两样本检验层串联。测试样本从第1层开始，如果被接受为ID则输出任务1的预测并退出；否则传递到第2层继续检验...直到某层接受或全部拒绝（判定为OOD）。每层由一个源-目标二分类器和Clopper-Pearson假设检验组成。

### 关键设计

1. **特征级两样本检验**:

    - 功能：在特征空间而非输出空间判断分布一致性
    - 核心思路：对每个任务训练一个二分类器$h(\psi(x))$区分"来自该任务的训练数据"和"其他数据"。用滑动窗口的准确率$\hat{\mu}_{w,\tau,j}$估计在线测试性能，通过Clopper-Pearson置信区间判断是否显著高于50%（随机猜测）。如果置信区间下界>0.5则接受为该任务的ID
    - 设计动机：特征级比输出级更鲁棒——softmax输出容易因过拟合/校准不良而不可靠，而中间特征$\psi(x)$更稳定

2. **层次化早退出架构**:

    - 功能：同时实现OOD检测和任务ID预测
    - 核心思路：$T$个检验层按任务顺序排列，样本从第1层开始逐层检验。一旦某层判定为ID就输出该任务的预测并退出。如果所有层都拒绝则判定为OOD。期望检测次数$(T+1)/2$（vs 标准C2ST需要$T$次）
    - 设计动机：传统OOD方法需要对所有$T$个任务都做一次检测（$O(T)$），层次化架构利用序列早退出将平均复杂度减半

3. **Clopper-Pearson校准统计检验**:

    - 功能：无阈值自动判定ID/OOD
    - 核心思路：用Clopper-Pearson精确二项置信区间替代固定阈值。给定窗口内$w$个样本的检测准确率$\hat{\mu}$，计算置信水平$1-\alpha$下的下界$\underline{\mu}$。如果$\underline{\mu} > 0.5$则判定为ID（检测器准确率显著高于随机）
    - 设计动机：固定阈值在不同数据/模型上需要重新调节；置信区间检验自动适应且有统计保证

### 损失函数 / 训练策略
每个检验层训练一个简单的二分类器（区分源/目标特征），使用标准交叉熵。可与Experience Replay等TIL方法兼容。窗口大小$w$和显著性水平$\alpha$为主要超参数。

## 实验关键数据

### 主实验（7个基准，F1和Task Accuracy）
- 在MNIST、SVHN、CIFAR-10/100、Mini-ImageNet、CoRe50、Stream-51上评估
- H2ST在F1和Task Accuracy上优于MSP、Energy、ODIN、MaxLogit
- 计算效率：H2ST期望$(T+1)/2$次检测 vs C2ST的$T$次

### 消融实验

| 配置 | 说明 |
|------|------|
| 层次化 vs 非层次化 | 层次化保持检测性能+降低计算量 |
| 特征级 vs 输出级 | 特征级更鲁棒，不依赖softmax校准 |
| 无阈值(CP区间) vs 固定阈值 | CP区间消除调参需求 |
| 窗口大小$w$影响 | 太小抖动大，太大响应慢 |

### 关键发现
- 无阈值设计在多数据集上表现稳定，免去了为每个场景调阈值的麻烦
- 特征级检测不受模型过拟合影响，在模型性能较差时优势更明显
- 层次化早退出将平均检测复杂度减半，且提供了传统方法不具备的任务ID能力

## 亮点与洞察
- **无阈值OOD检测**：用统计假设检验替代手动阈值，这在安全关键应用（医疗、自动驾驶）中极其重要
- **检测+识别双功能**：不仅判定OOD，还识别属于哪个任务的ID——二合一能力
- **增量可扩展**：新任务到来时只需添加一个新的检验层，不影响已有层

## 局限与展望
- 层次化顺序可能影响性能——如果某任务的OOD样本与另一个任务的ID样本相似，层序不当可能误判
- 滑动窗口机制要求测试样本流式到达，不适合单张图片OOD检测
- $\alpha$和$w$的选择仍需一定领域知识

## 相关工作与启发
- **vs MSP/Energy/ODIN**: 这些方法依赖阈值和模型输出；H2ST在特征空间做统计检验，更鲁棒且无阈值
- **vs C2ST**: 标准两样本检验需要对所有$T$个任务做检测；H2ST的层次化将复杂度减半
- **vs OSDN**: OSDN用Weibull分布建模logit；H2ST基于非参数假设检验，假设更弱更通用

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化两样本检验+无阈值设计的组合创新
- 实验充分度: ⭐⭐⭐⭐ 7个数据集、多基线、统计保证
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，统计方法有理论基础
- 价值: ⭐⭐⭐⭐ 对增量学习中的安全部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](../../NeurIPS2025/others/double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)
- [Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/others/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)

<!-- RELATED:END -->
