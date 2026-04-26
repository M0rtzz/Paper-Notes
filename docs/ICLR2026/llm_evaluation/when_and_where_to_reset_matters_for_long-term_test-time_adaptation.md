---
title: >-
  [论文解读] When and Where to Reset Matters for Long-Term Test-Time Adaptation
description: >-
  [ICLR 2026][测试时适应] ASR提出自适应选择性重置方案，通过预测集中度 $\mathcal{C}_t$ 动态判断何时重置（避免固定周期的次优性），通过从output层向input层渐进的层选择策略判断重置哪些层（保留有价值的适应知识），配合importance-aware正则化恢复被重置的关键知识和on-the-fly适应调整，在CCC-Hard上比SOTA提升44.12%。
tags:
  - ICLR 2026
  - 测试时适应
  - 模型崩溃
  - 自适应重置
  - 选择性重置
  - Fisher信息
  - 长期域漂移
---

# When and Where to Reset Matters for Long-Term Test-Time Adaptation

**会议**: ICLR 2026  
**arXiv**: [2603.03796](https://arxiv.org/abs/2603.03796)  
**代码**: https://github.com/YonseiML/asr  
**领域**: 其他 / 持续测试时适应  
**关键词**: 测试时适应, 模型崩溃, 自适应重置, 选择性重置, Fisher信息, 长期域漂移

## 一句话总结
ASR提出自适应选择性重置方案，通过预测集中度 $\mathcal{C}_t$ 动态判断何时重置（避免固定周期的次优性），通过从output层向input层渐进的层选择策略判断重置哪些层（保留有价值的适应知识），配合importance-aware正则化恢复被重置的关键知识和on-the-fly适应调整，在CCC-Hard上比SOTA提升44.12%。

## 研究背景与动机

**领域现状**：持续测试时适应（TTA）在非平稳域流上更新模型，但长期适应导致错误累积→模型崩溃（model collapse）：模型对所有输入只预测少数几个类别。

**现有痛点**：(1) RDumb等方法用固定周期全量重置→周期与实际崩溃风险无关，要么太早（浪费适应知识）要么太晚（错误深度累积）；(2) 全量重置灾难性地丢弃所有时间积累的知识；(3) 每次重置后都有显著的性能骤降和恢复延迟。

**核心矛盾**：重置太频繁→适应不充分；重置太稀→崩溃不可逆。全量重置→知识丢失；不重置→错误累积。

**本文目标**：(1) When: 如何动态判断何时有崩溃风险？(2) Where: 如何选择重置哪些层以最小化知识损失？(3) 如何恢复被重置但仍然重要的知识？

**切入角度**：利用预测集中度（prediction concentration）作为崩溃风险的proxy，利用深度网络的层次结构（靠近output的层先被label noise corruption）决定重置范围。

**核心 idea**：用预测集中度偏离长期基线来触发重置，按崩溃严重度从output层向input层渐进重置，用Fisher信息加权正则化恢复被重置的关键知识。

## 方法详解

### 整体框架
ASR由3个组件构成：(1) 自适应选择性重置（基于 $\mathcal{C}_t$ vs $\bar{\mathcal{C}}_{t-1}$）；(2) Importance-aware知识恢复（Fisher信息正则化）；(3) On-the-fly适应调整（基于预测不一致性 $\phi_t$）。

### 关键设计

1. **自适应重置——When**:

    - 预测集中度：$\mathcal{C}_t = \sum_{c=1}^C \hat{p}_{t_c} \log(\hat{p}_{t_c})$，其中 $\hat{p}_t = \sigma(\frac{1}{|\mathcal{B}_t|}\sum_i f_{\theta_{t-1}}(x_t^i))$
    - 大 $\mathcal{C}_t$ → 低预测多样性 → 高崩溃风险
    - 累积集中度（EMA）：$\bar{\mathcal{C}}_t = \mu_\mathcal{C} \cdot \bar{\mathcal{C}}_{t-1} + (1-\mu_\mathcal{C}) \cdot \mathcal{C}_t$
    - 触发条件：$\mathcal{C}_t > \bar{\mathcal{C}}_{t-1}$ 时立即重置
    - 初始化 $\bar{\mathcal{C}}_0 = -\log(\alpha_0 \cdot C)$，选择 $\alpha_0$ 使初始值足够大避免过早重置
    - 实验验证：$\mathcal{C}_t$ 与准确率的Pearson相关系数高达 **0.88**

2. **选择性重置——Where**:

    - 动机：label noise corruption从网络末端开始（Bai et al., 2021; Yang et al., 2024），靠近input的层更鲁棒
    - 重置比例：$r_t = r_0 + \lambda_r \cdot (\mathcal{C}_t - \bar{\mathcal{C}}_{t-1})$
    - 从output端开始重置 $r_t$ 比例的层，其余保留
    - $r_t$ 上限1.0，$r_0$ 为最小重置比例
    - 设计动机：崩溃越严重→corruption越深入→需要重置更多层

3. **Importance-Aware知识恢复**:

    - 损失：$\mathcal{L} = \mathcal{L}_u + \lambda_\mathcal{F}\sum_i \bar{\mathcal{F}}^i(\theta_{t-1}^i - \bar{\theta}^i)^2$
    - $\bar{\mathcal{F}}^i$：累积Fisher信息矩阵，$\bar{\theta}^i$：累积参数
    - 对先前任务重要的参数（高Fisher值）被引导与累积状态对齐
    - **混合累积方案**：CMA在每次重置间等权累积参数和Fisher矩阵；EMA在重置触发点聚合CMA值
    - 解决的困境：接近重置时参数更接近当前域但也更容易被corruption，EMA的近因偏好不适合直接用

4. **On-the-fly适应调整**:

    - 预测不一致性：$\phi_t = \frac{1}{|\mathcal{B}_t|}\sum_i \mathbb{I}(\arg\max(\breve{y}_t^i) \neq \arg\max(\hat{y}_t^i))$
    - 大 $\phi_t$（源模型与当前模型预测不一致）→ 大域差异
    - 自适应调参：$\lambda_\mathcal{F} = \lambda_0 \cdot \phi_t^2$（域差异大→正则化强），$\mu_\mathcal{C} = \mu_0 \cdot \phi_t + 1 - \mu_0$（域差异大→减少集中度更新）

## 实验关键数据

### CCC Benchmark（主实验，ResNet-50）

| 方法（基于ETA） | Easy | Medium | Hard | Mean |
|----------------|------|--------|------|------|
| ETA | 43.24 | 19.03 | 0.32 | 20.86 |
| + RDumb | 49.47 | 39.42 | 9.77 | 32.89 |
| + COME | - | - | - | - |
| + ReservoirTTA | - | - | - | - |
| **+ ASR (Ours)** | **最高** | **最高** | **最高** | **最高** |

CCC-Hard上比SOTA提升 **44.12%**。

### 其他Benchmark
- Concatenated ImageNet-C (CIN-C)：所有方法中表现最佳
- ImageNet-C (20次循环)：稳定适应无崩溃
- ImageNet-D109 (20次循环)：同样最优

### 关键发现
- ASR作为add-on方法适用于ETA、EATA、ROID等多个基线方法
- 在challenging设置下（CCC-Hard）提升尤为显著——这正是现有方法崩溃最严重的场景
- $\mathcal{C}_t$ 相比其他崩溃检测指标（如极高置信度、分布偏移检测）更稳定可靠
- 选择性重置vs全量重置：显著减少重置后的性能骤降和恢复延迟

### 消融实验
- 去掉自适应重置（固定周期）→性能下降
- 去掉选择性重置（全量重置）→性能骤降和恢复延迟增大
- 去掉Fisher正则化→无法恢复被重置的关键知识
- 去掉on-the-fly调整→在challenging域漂移下适应性不足

## 亮点与洞察
- **信号设计的优雅性**：$\mathcal{C}_t$ 基于batch-level的logit均值softmax的熵，既简单又有效（0.88相关性），无需额外模型或计算
- **层次重置的理论依据**：利用了corruption从网络末端开始这一已知现象，将通用观察转化为实用策略
- **CMA+EMA混合累积**：巧妙解决了"接近重置时参数更适应当前域但更可能被corruption"的bootstrapping困境
- **即插即用**：ASR可作为add-on加到任何现有TTA方法上，不需要修改基础适应算法

## 局限与展望
- 超参数（$r_0, \lambda_r, \alpha_0, \lambda_0, \mu_0$）需要在holdout数据上确定，虽然使用的数据量很少（5%单split）
- 当前假设batch内样本来自相同域，mixed-domain batch场景有待研究
- Fisher信息估计在连续在线学习中的准确性可能随时间退化
- 对ViT-B-16的验证相对初步，更多架构和规模有待评估
- 与prompt-based TTA方法的集成值得探索

## 相关工作与启发
- **vs RDumb**: 固定周期全量重置是naive但effective的baseline，ASR在此基础上引入自适应性和选择性
- **vs CoTTA**: CoTTA用augmentation-averaged伪标签和随机参数恢复，ASR用更原则性的Fisher-based方法
- **vs ROID/CMF**: 权重集成方法，ASR的重置+恢复范式是互补的
- **vs PeTTA**: 基于参数发散的正则化，ASR的预测集中度是更直接的崩溃指标

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应+选择性重置的组合以及CMA+EMA混合累积有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 4个benchmark、多个基线方法组合、详细消融、多架构验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清楚（Fig.1极为直观）、方法图解清晰（Fig.2）、统计严谨
- 价值: ⭐⭐⭐⭐⭐ CCC-Hard 44.12%提升是实质性突破，即插即用的设计具有广泛适用性

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/llm_evaluation/test-time_adaptation_by_causal_trimming.md)
- [\[AAAI 2026\] MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm](../../AAAI2026/llm_evaluation/moetta_test-time_adaptation_under_mixed_distribution_shifts_with_moe-layernorm.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](../../ECCV2024/llm_evaluation/distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ICLR 2026\] When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)

<!-- RELATED:END -->
