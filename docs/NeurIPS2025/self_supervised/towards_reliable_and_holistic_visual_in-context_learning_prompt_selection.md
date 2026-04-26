---
title: >-
  [论文解读] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection
description: >-
  [NeurIPS 2025][自监督学习][visual in-context learning] 提出RH-Partial2Global方法，首次用Spearman秩相关检验证明VICL中"相似性优先假设"虽统计显著但相关强度极弱($\bar{\rho} \approx 0.03\text{-}0.05$)，通过Jackknife共形预测构建可靠候选集+覆盖设计实现全面均匀的成对偏好采样，在分割/检测/着色三个视觉任务上一致超越SOTA。
tags:
  - NeurIPS 2025
  - 自监督学习
  - visual in-context learning
  - conformal prediction
  - covering design
  - 提示学习
  - global ranking
---

# Towards Reliable and Holistic Visual In-Context Learning Prompt Selection

**会议**: NeurIPS 2025  
**arXiv**: [2509.25989](https://arxiv.org/abs/2509.25989)  
**代码**: [github.com/Wu-Wenxiao/RH-Partial2Global](https://github.com/Wu-Wenxiao/RH-Partial2Global)  
**领域**: 视觉上下文学习 / 提示选择  
**关键词**: visual in-context learning, conformal prediction, covering design, prompt selection, global ranking

## 一句话总结
提出RH-Partial2Global方法，首次用Spearman秩相关检验证明VICL中"相似性优先假设"虽统计显著但相关强度极弱($\bar{\rho} \approx 0.03\text{-}0.05$)，通过Jackknife共形预测构建可靠候选集+覆盖设计实现全面均匀的成对偏好采样，在分割/检测/着色三个视觉任务上一致超越SOTA。

## 研究背景与动机
**领域现状**：视觉上下文学习(VICL)通过给视觉基础模型提供少量in-context示例来适配新任务，核心挑战是如何为每个查询选择最优的上下文示例。现有方法（VPR、Partial2Global）将此formulate为全局排序问题。

**现有痛点**：(1) **相似性优先假设缺乏充分论证**——VPR和Partial2Global都依赖"与查询视觉相似度越高的图像越是好的上下文示例"，但该假设从未被严格验证；(2) **Partial2Global的随机采样策略有缺陷**——随机打乱候选集生成子序列进行局部排序，无法保证所有成对关系被覆盖（$K=50, k=5$需要至少130个子序列才能覆盖所有对，而原方法仅用50个），且可能产生冗余比较。

**核心矛盾**：如何在不完全依赖相似性假设的前提下，构建既可靠又全面的in-context示例选择流程？

**切入角度**：用共形预测理论过滤不可靠候选（R），用覆盖设计理论保证成对比较的全面性（H）。

## 方法详解

### 整体框架
RH-Partial2Global在Partial2Global基础上引入两个正交增强：(1) Jackknife共形预测筛选可靠候选集$\mathcal{Y}_\alpha$，与相似性候选集$\mathcal{Y}_q$取交集得到精炼集$\mathcal{Y}_q^* = \mathcal{Y}_\alpha \cap \mathcal{Y}_q$；(2) 用$(K', k, 2)$覆盖设计替代随机打乱引导局部排序的子序列采样，保证所有成对偏好被至少一个子序列覆盖。两个模块均不引入额外模型训练。

### 关键设计
1. **Jackknife共形预测引导的候选选择**:
    - 功能：从训练集中筛选出"质量与相似性一致"的可靠样本，过滤掉相似但质量差的候选
    - 核心思路：对训练集中每个样本$x_i^{trn}$，计算其作为prompt应用于所有其他样本的质量分数集$\mathcal{Q}(x_i) = \{\mathfrak{q}(\mathcal{F}(x_j, x_i), x_i)\}$和相似性分数集$\mathcal{S}(x_i) = \{\mathfrak{s}(x_j, x_i)\}$，得到一致性分数$\ell(x_i) = f(\mathcal{Q}(x_i), \mathcal{S}(x_i))$（$f$为负KL散度）。计算$(1-\alpha)$分位数$q_{1-\alpha}$作为阈值，可靠集$\mathcal{Y}_\alpha = \{x_i | \ell(x_i) > q_{1-\alpha}\}$。对查询$x_q$：$\mathcal{Y}_q^* = \mathcal{Y}_\alpha \cap \text{top-K}(\mathfrak{s}(x_q, \cdot))$
    - 设计动机：Spearman检验发现相似性与质量的相关系数极低（$\bar{\rho} \approx 0.03\text{-}0.05$），说明仅靠相似性选择候选不够可靠。共形预测提供分布无关的覆盖保证，Jackknife方式充分利用训练数据

2. **覆盖设计引导的全面采样**:
    - 功能：用组合数学中的覆盖设计替代随机打乱，保证所有候选对被至少一个局部排序子序列覆盖
    - 核心思路：$(K, k, 2)$覆盖设计要求在$K$元素集合中，所有2元素子集都至少出现在一个$k$元素块中。Schonheim下界$C(K,k,t) \geq \lceil\frac{K}{k}\lceil\frac{K-1}{k-1}...\rceil\rceil$给出最少块数（$C(50,5,2) \geq 130$）。使用预计算的最优覆盖设计$C^*(K',k,2)$引导采样，生成随机打乱的候选集后按覆盖设计结构提取$k$长度子序列
    - 设计动机：Partial2Global用50个随机子序列无法覆盖$C\binom{50}{2} = 1225$个成对关系，且重复比较导致偏好权重不均匀。覆盖设计保证穷尽性+最小化子序列数

3. **相似性优先假设的统计验证**:
    - 功能：首次对VICL中的基础假设进行严格统计检验
    - 核心思路：在Pascal-5i训练集上，对每个查询样本计算所有候选的(IoU分数, 视觉相似性)两个序列，进行Spearman秩相关检验。结果：78-88%样本拒绝零假设（$p < 0.05$），说明统计显著存在单调关系；但$\bar{\rho} \approx 0.03\text{-}0.05$极低，说明关系强度很弱
    - 设计动机：Partial2Global质疑过该假设但未提供统计依据。本文的量化分析为引入共形预测提供了理论动机

### 损失函数 / 训练策略
元学习训练阶段与Partial2Global完全相同——训练transformer-based list-wise ranker $\phi_k$（长度5和10），用DINOv2提取特征，AdamW优化器lr=$5\times10^{-5}$，batch=64。推理阶段：$\alpha=0.85$（85%置信度），一致性函数用负KL散度，相似性用CLIP视觉编码器。RH-Partial2Global仅修改推理阶段的候选选择和采样策略，不需要额外训练。

## 实验关键数据

### 主实验：跨视觉任务对比

| 方法 | 分割Avg(mIoU)↑ | 检测(mIoU)↑ | 着色(MSE)↓ |
|---|---|---|---|
| MAE-VQGAN (NeurIPS'22) | 27.56 | 25.45 | 0.67 |
| SupPR (NeurIPS'23) | 35.56 | 28.22 | 0.63 |
| Partial2Global (NeurIPS'24) | 38.40 | 30.66 | 0.58 |
| **RH-Partial2Global** | **39.02** | **30.94** | **0.56** |
| Partial2Global+voting | 42.69 | 32.52 | — |
| **RH-Partial2Global+voting** | **43.08** | **33.28** | — |

### 消融实验：各组件贡献（分割任务，4 folds平均）

| 配置 | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Avg |
|---|---|---|---|---|---|
| Partial2Global baseline | 38.81 | 41.54 | 37.25 | 36.01 | 38.40 |
| + 共形预测(R) | 39.05 | 41.80 | 37.72 | 36.35 | 38.73 |
| + 覆盖设计(H) | 39.10 | 41.95 | 37.85 | 36.42 | 38.83 |
| + R + H (完整) | **39.25** | **42.15** | **38.06** | **36.60** | **39.02** |

### 关键发现
- 相似性优先假设统计显著（78-88%样本$p<0.05$），但相关强度极弱（$\bar{\rho} \approx 0.03\text{-}0.05$）
- 共形预测过滤约15%候选但高质量示例性能上界几乎不变（top-5 IoU仅下降0.26）
- RH-Partial2Global在所有3个任务、4个folds上**一致**改进，且无额外模型训练
- 可视化显示RH选择的示例在姿态、场景等细粒度属性上与查询更对齐

## 亮点与洞察
- 首次对VICL的基础假设（相似性优先）提供严格统计证据证明其不够稳健
- 共形预测为可靠候选选择提供了理论保证（分布无关的覆盖概率），且与具体任务无关
- 覆盖设计是解决"如何系统性采样成对关系"的优雅数学工具，将组合优化引入排序聚合
- 两个增强模块均为推理阶段modification，不增加训练成本，即插即用

## 局限与展望
- 改进幅度一致但较小（平均~0.6%），在小fold上受限于校准集大小
- $\alpha=0.85$为所有任务统一设定，自适应$\alpha$选择可能进一步提升
- 覆盖设计的预计算对超大规模候选集（$K > 100$）可能有计算开销
- 仅在MAE-VQGAN作为VICL backbone上验证，对其他VFM的普适性未知
- 一致性函数（负KL散度）的选择缺乏系统比较

## 相关工作与启发
- **vs Partial2Global (NeurIPS'24)**：直接改进，核心贡献在于挑战基础假设+修复采样缺陷。分割平均mIoU 38.40→39.02，检测30.66→30.94
- **vs VPR (NeurIPS'23)**：VPR完全依赖相似性，本文证明相似性$\bar{\rho}<0.06$不够可靠
- **vs prompt-SelF**：prompt-SelF用投票集成（正交方向），RH-Partial2Global+voting = 43.08 > prompt-SelF = 41.02
- **启发**：共形预测在ML中的应用是一个趋势（不确定性量化、LLM benchmarking等），本文将其用于in-context example selection是新颖的实例

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题分析（假设检验）新颖，共形预测+覆盖设计的组合独特
- 实验充分度: ⭐⭐⭐⭐ 3任务、4 folds、消融完整，但改进幅度有限
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义严谨，数学推导清晰，动机-方法逻辑线强
- 价值: ⭐⭐⭐⭐ 对VICL提示选择有方法论贡献，但实际影响受限于改进幅度

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[NeurIPS 2025\] Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge](foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[CVPR 2025\] Spectral State Space Model for Rotation-Invariant Visual Representation Learning](../../CVPR2025/self_supervised/spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](../../ECCV2024/self_supervised/promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)

<!-- RELATED:END -->
