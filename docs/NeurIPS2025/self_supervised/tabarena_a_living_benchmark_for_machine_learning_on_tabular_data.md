---
title: >-
  [论文解读] TabArena: A Living Benchmark for Machine Learning on Tabular Data
description: >-
  [NeurIPS 2025][自监督学习][表格数据基准] 提出 TabArena，首个持续维护的"活跃"表格数据基准系统，从 1053 个数据集中精选 51 个、纳入 16 个模型，通过大规模实验（约 2500 万次模型训练）发现：后验集成下深度学习模型已追平甚至超越 GBDT…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "表格数据基准"
  - "活跃基准"
  - "梯度提升树"
  - "深度学习"
  - "表格基础模型"
---

# TabArena: A Living Benchmark for Machine Learning on Tabular Data

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.16791](https://arxiv.org/abs/2506.16791)  
**代码**: [有 (tabarena.ai)](https://tabarena.ai)  
**领域**: 表格数据 / 基准测试 / AutoML  
**关键词**: 表格数据基准, 活跃基准, 梯度提升树, 深度学习, 表格基础模型

## 一句话总结
提出 TabArena，首个持续维护的"活跃"表格数据基准系统，从 1053 个数据集中精选 51 个、纳入 16 个模型，通过大规模实验（约 2500 万次模型训练）发现：后验集成下深度学习模型已追平甚至超越 GBDT，表格基础模型在小数据上表现突出，跨模型集成可进一步推进 SOTA。

## 研究背景与动机
**领域现状**: 表格数据机器学习基准数量不断增长，但现有基准多为静态的——发布后即使发现缺陷、模型更新或出现新方法，也不会更新设计。

**现有痛点**:
   - 数据集质量参差不齐：许多基准中的数据集过时、含数据泄露、非真实表格任务、或许可证有问题
   - 评估协议不统一：不同基准使用不同的验证策略（holdout vs. 交叉验证）、超参数搜索预算、集成策略，导致结论不可比
   - 缺乏后验集成评估：大多数基准不评估模型集成后的峰值性能，低估了模型的真正能力
   - 后续基准复制前人的缺陷，且不比较真正的 SOTA

**核心矛盾**: 社区迫切需要可靠的基准来评估深度学习 vs. GBDT 等关键问题，但现有静态基准无法提供持续可信的答案。

**本文目标**: 建立首个持续维护、版本化、社区驱动的"活跃基准"系统，使表格 ML 基准测试变得可靠且可持续。

**切入角度**: 从数据筛选、模型实现、评估设计三个维度制定严格协议，并组建跨机构维护团队。

**核心 idea**: 用软件工程思维做基准——版本化、持续维护、社区贡献，而非发布即弃。

## 方法详解

### 整体框架
TabArena 是一个包含三大核心协议的活跃基准系统：
1. **模型与超参数优化协议**: 规范模型实现、搜索空间、集成策略
2. **数据集协议**: 严格的人工筛选标准，从 1053 个候选中精选 51 个
3. **评估设计协议**: 统一的交叉验证、重复策略、Elo 评分排行榜

### 关键设计
1. **数据集人工精选 (51/1053)**:

    - 10 项筛选标准：唯一性、IID 性、真实表格域、真实分布（非生成）、真实预测任务、规模限制（500–250K 样本）、无不可逆预处理/数据泄露、合规许可证、可公开下载、无伦理问题
    - 只有去重和规模过滤可自动化，其余需人工逐数据集审查
    - 公开每个数据集的审查记录，邀请社区质疑和贡献

2. **模型实现标准化 (16 个模型)**:

    - 所有模型基于 AutoGluon 的 AbstractModel 框架实现（兼容 scikit-learn API）
    - 包括 5 个树模型（RF、ExtraTrees、XGBoost、LightGBM、CatBoost）、6 个神经网络（FastaiMLP、TorchMLP、RealMLP、TabM、ModernNCA、EBM）、3 个基础模型（TabPFNv2、TabICL、TabDPT）和 2 个基线（Linear、KNN）
    - 与原作者对话确认搜索空间，每个模型评估 1 个默认 + 200 个随机超参数配置

3. **交叉验证与后验集成**:

    - 默认 8 折内层交叉验证 + 交叉验证集成
    - 后验集成（Weighted Post-hoc Ensembling）: 对不同超参数配置产生的模型做加权集成
    - 基础模型不使用交叉验证集成，而是在训练+验证集上 refit

4. **Elo 评分系统**:

    - 基于成对比较的 Elo 评分（类似 ChatBot Arena），1000 Elo 校准为默认 RandomForest
    - 400 分差 ≈ 91% 胜率；每个数据集贡献相等
    - 200 轮 bootstrap 获得 95% 置信区间
    - 分类用 ROC AUC / log-loss，回归用 RMSE

5. **重复策略**: ≤2500 样本的数据集用 10 次重复 3 折交叉验证；其他数据集 3 次重复

### 损失函数 / 训练策略
- 每个超参数配置限时 1 小时
- CPU: AWS M6i.2xlarge (8 核 Intel Xeon)；GPU: NVIDIA L40S 48GB VRAM
- 总计算量约 15 年墙钟时间，约 2500 万次模型训练

## 实验关键数据

### 主实验（TabArena-v0.1 排行榜，后验集成）

| 排名 | 模型 | 类型 | Elo（集成后） |
|---|---|---|---|
| 1 | TabM | 神经网络 | 最高 |
| 2 | LightGBM | 树模型 | 第二 |
| 3 | RealMLP | 神经网络 | 第三 |
| 4 | CatBoost | 树模型 | 第四（调参下第一） |
| 5 | XGBoost | 树模型 | 第五 |
| 参考 | AutoGluon (4h) | 系统 | 约第二梯队 |

- CatBoost 在常规调参（无集成）下排名第一，但后验集成后 TabM、LightGBM、RealMLP 反超
- 基础模型中 TabPFNv2 在兼容数据集（≤10K 样本）上大幅领先，甚至超越 AutoGluon

### 消融实验

| 评估维度 | 关键发现 |
|---|---|
| Holdout vs. 交叉验证 | Holdout 验证严重低估所有模型性能，并偏向已使用集成的模型 |
| 后验集成效果 | Top 3 模型（TabM、LightGBM、RealMLP）不做集成时均不如 CatBoost |
| 跨模型集成 | 使用所有模型的集成 pipeline 超越所有单模型和 AutoGluon |
| 集成权重分布 | 排行榜名次最高的模型不一定集成权重最大（验证集过拟合效应） |
| 推理效率 Pareto 前沿 | EBM 和 CatBoost 推理最快；RealMLP 需 ~100× 推理时间换取更高性能 |
| 基础模型小数据 | TabPFNv2 在 ≤10K 样本数据集上即使不调参也非常强 |

### 关键发现
- **GBDT vs. 深度学习是伪命题**: 两类模型在集成中互补，跨模型集成显著优于单一模型族
- **后验集成是释放深度学习潜力的关键**: 没有集成，DL 模型普遍不如 GBDT
- **基础模型适合小数据**: TabPFNv2 的 in-context learning 在小数据场景下表现出色
- **验证策略至关重要**: Holdout 验证会系统性地错误评估模型排名
- **适合基准的高质量数据集令人意外地少**: 1053 个候选中仅 51 个通过所有筛选标准

## 亮点与洞察
- **活跃基准理念**: 将基准视为"软件"而非"论文"，引入版本控制、维护协议、社区贡献流程，是基准研究的范式转变
- **公正评估峰值性能**: 后验集成的引入使得不同模型可以在公平条件下展示最佳性能，而非被训练策略差异所干扰
- **实用性极强**: 所有模型都在 AutoGluon 框架中实现，可直接用于实际应用；预计算结果公开共享，新模型可低成本对比
- **数据筛选的透明度**: 公开每个数据集的审查笔记，是基准研究难得的透明度标杆
- **Elo 评分体系**: 从 LLM 排行榜借鉴 Elo 评分，避免了传统平均排名对极端数据集的敏感性

## 局限与展望
- 当前仅覆盖 IID、小中规模（500–250K 样本）的分类和回归任务，未涵盖时序、分布偏移、聚类、异常检测等场景
- 200 个随机超参数配置的固定预算限制了对更高级 HPO 策略（如 Bayesian optimization）的研究
- 每配置 1 小时时间限制依赖硬件，跨用户结果的可比性在边界情况下受影响
- 严格的数据集筛选标准导致数据集数量仅 51 个，统计效力有限
- 未考虑特征工程的影响，而特征工程可能改变模型排名
- 公开基准存在刷榜（过拟合数据集、foundation model 数据污染）风险

## 相关工作与启发
- **对标 ChatBot Arena / LiveBench 等 LLM 基准**: 借鉴活跃排行榜和 Elo 评分的思路，但针对表格数据有本质不同的模型和评估设计
- **vs. 既有表格基准 (OpenML-CC18, TabZilla, AutoML Benchmark 等)**: TabArena 是首个整合后验集成评估、严格人工数据筛选、持续维护协议的基准
- **AutoGluon 作为参考 pipeline**: 代表实践者容易达到的性能水平，为模型评估提供现实对照
- **TabRepo 的预计算结果思想**: TabArena 扩展了结果共享的粒度（含预测值、元数据），使得后续研究可以零成本对比

## 评分
- ⭐⭐⭐⭐⭐ (5/5)
- 理由: 这不仅是一个基准论文，更是表格 ML 基准研究的范式创新。从数据筛选到评估设计的每个环节都极其严谨，实验规模空前（2500 万次训练），结论改变了社区对 GBDT vs. DL 的认知，且活跃维护的理念具有长远影响力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
