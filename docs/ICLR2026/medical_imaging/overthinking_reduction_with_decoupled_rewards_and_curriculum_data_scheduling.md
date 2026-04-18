---
title: >-
  [论文解读] Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling
description: >-
  [ICLR 2026][医学图像][overthinking] 从理论上揭示了现有长度惩罚方法的两个根本缺陷——错误惩罚高熵探索token和错误奖励冗余token，提出 DeCS 框架，通过解耦token级奖励和课程批次调度，在7个基准上将推理token减少50%以上同时保持甚至提升模型性能。
tags:
  - ICLR 2026
  - 医学图像
  - overthinking
  - 解耦奖励
  - 课程学习
  - RLVR
  - NRP
---

# Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling

**会议**: ICLR 2026  
**arXiv**: [2509.25827](https://arxiv.org/abs/2509.25827)  
**代码**: [github.com/pixas/DECS](https://github.com/pixas/DECS)  
**领域**: LLM推理效率 / 强化学习  
**关键词**: overthinking, 解耦奖励, 课程学习, RLVR, NRP

## 一句话总结

从理论上揭示了现有长度惩罚方法的两个根本缺陷——错误惩罚高熵探索token和错误奖励冗余token，提出 DeCS 框架，通过解耦token级奖励和课程批次调度，在7个基准上将推理token减少50%以上同时保持甚至提升模型性能。

## 研究背景与动机

**领域现状**：大推理模型（LRM）通过RLVR展现强大推理能力，但存在严重的"overthinking"问题——模型在得到正确答案后仍生成大量冗余推理步骤，推理效率低下。

**现有痛点**：现有方法通过在正确性奖励中加入长度惩罚$r'(\boldsymbol{o}_i) = r(\boldsymbol{o}_i) - \gamma |\boldsymbol{o}_i|$来鼓励简洁推理，但效率提升往往以性能下降为代价，无法达到最优效率-性能权衡。

**核心矛盾**：轨迹级长度奖励与token级策略优化之间的根本性不对齐——(1) 负advantage反向传播到所有token，错误地抑制了正确的高熵探索token（如"wait"、"however"）；(2) 较短轨迹中NRP后的冗余token仍获得正advantage，被错误地强化。

**本文目标** 如何精准区分和惩罚冗余token，同时保护对推理有贡献的必要token，实现真正无损的推理压缩。

**切入角度**：定义"必要推理前缀"（NRP）作为判断标准，将奖励在NRP边界处解耦，对NRP前后的token施加不同的奖励信号。

**核心 idea**：训练轻量判别器识别NRP边界，对NRP内token给最大奖励、NRP后冗余token给递减惩罚，配合课程调度控制简单样本比例以保护高熵探索能力。

## 方法详解

### 整体框架

(1) 微调轻量语言模型$\mathcal{M}_{\text{judge}}$检测每个正确轨迹的NRP边界；(2) 设计解耦token级奖励，NRP内token获得$r_+$，NRP后冗余token获得与位置成反比的低奖励；(3) 课程调度策略根据当前批次的NRP比例自适应调节简单样本占比$\kappa_m$。

### 关键设计

1. **NRP检测与解耦奖励 (Decoupled Reward)**:
    - 做什么：精确识别每个正确轨迹中"到何处为止就足够得出正确答案"的边界，并据此分配差异化token级奖励
    - 核心思路：轻量模型$\mathcal{M}_{\text{judge}}$将推理过程分割为多个chunk $\{s_1, \ldots, s_{|S|}\}$，对每个chunk判断是否已包含正确答案$j_{s_c} \sim \mathcal{M}_{\text{judge}}(\cdot | q, s_c, y^*)$。NRP定义为首次包含正确答案的chunk及其之前所有chunk。token级奖励为：$r_{i,j} = r_+ \cdot \mathbf{1}_{\text{correct}}$（$j \leq K_{o_i}^*$）或 $r_{i,j} = (r_0 - (r_+ - r_0)L_i/L_{\max}) \cdot \mathbf{1}_{\text{correct}}$（$j > K_{o_i}^*$且属于思考token）
    - 设计动机：Theorem 2 证明了序列级长度奖励下，NRP后第一个冗余token的梯度信号$\mathcal{J}(A; j=K^*+1) > 0$——即模型被鼓励继续生成而非停止。解耦奖励确保NRP后任何前导冗余token都获得负advantage，从而利用自回归性质压制整段冗余

2. **课程批次调度 (Curriculum Prompt Schedule)**:
    - 做什么：自适应控制训练批次中简单样本（所有rollout都正确的prompt）的比例
    - 核心思路：$\kappa_m = \text{clip}(\kappa_{m-1} + \beta(\mathcal{R}_m - \mathcal{R}_{m-1}), 0, \kappa_m^0)$，其中$\mathcal{R}_m$是当前批次中正确序列的NRP比例。当NRP比例增加（冗余减少），允许更多简单样本参与训练；Theorem 1 给出条件 $\kappa \sigma_L < C$ 以维持高熵token的生成概率
    - 设计动机：简单样本是效率优化的主要来源（因为所有rollout都正确时长度成为唯一区分信号），但简单样本比例过大会使高熵token的logit下降主导整个批次梯度，导致性能退化。课程调度实现了探索与压缩的动态平衡

3. **理论分析框架**:
    - 做什么：为方法设计提供理论支撑
    - 核心思路：Lemma 1 建立policy gradient下logit变化与advantage的线性关系；Lemma 2 证明长度惩罚使高熵token的期望logit变化严格为负；Theorem 1 给出批次学习中维持高熵token的充要条件；Theorem 2 证明序列级长度奖励无法在NRP边界处停止生成
    - 设计动机：现有方法的失败不是偶然的实验现象，而是有理论根基的——这指导了解耦奖励和课程调度的精确设计

### 损失函数 / 训练策略

基于GRPO的PPO代理损失（Eq. 3），token级advantage $A_{i,j}^{\text{DeCS}} = (r_{i,j} - \text{mean})/\text{std}$。$r_+=1.1$, $r_0=1.0$, $\beta=0.2$。训练集为DeepScaleR（40k数学题），每prompt 16个rollout。base模型为DS-1.5B和DS-7B，使用veRL框架。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DeCS(1.5B) | Base(1.5B) | 最佳基线 | 说明 |
|--------|------|------|------|------|------|
| 7基准平均 | Pass@1 | 47.78 | 45.21 | 45.83(ThinkPrune) | +2.57, 效率与性能双提升 |
| 7基准平均 | #Token | 4000 | 9340 | 3975(ThinkPrune) | 减少57.17% |
| 7基准平均 | AES | 0.74 | 0.00 | 0.62(ThinkPrune) | AES最优 |
| AIME2024(1.5B) | Pass@1 | 31.25 | 27.99 | 29.87(TLMRE) | +3.26提升 |
| AIME2024(1.5B) | #Token | 5550 | 12202 | 5306(ThinkPrune) | 减少54.5% |
| 7基准平均(7B) | Pass@1 | 62.48 | 61.57 | 62.17(ThinkPrune) | +0.91 |
| 7基准平均(7B) | #Token | 3968 | 7857 | 4940(ThinkPrune) | 减少49.5% |

### 消融实验

| 配置 | Pass@1 | #Token | 说明 |
|------|------|------|------|
| 仅DR（解耦奖励） | 提升但残留~25%冗余 | 受限 | 缺调度导致高熵token被过度抑制 |
| 仅CS（课程调度） | 性能下降 | 减少有限 | 缺解耦奖励无法精准惩罚冗余 |
| DR+CS (DeCS完整) | 最优 | 最大减少 | 二者互补 |
| Qwen3-4B骨干 | 69.72(+1.32) | 4115(减少54.8%) | AES 0.61, 骨干通用性好 |

### 关键发现

- DeCS在减少>50% token的同时保持甚至提升Pass@1，且Pass@K曲线与base模型几乎完全重合，证明探索能力未受损
- NRP检测器虽训练于数学语料，在域外任务（GPQA-D减少56.33%、LCB减少33.52%）上同样有效
- token分析显示DeCS主要减少了"自校正/验证"和"结论"类token，"探索/替代"类token频率几乎不变

## 亮点与洞察

- 理论分析是核心贡献：两个定理精确刻画了序列级长度奖励的两个失败模式，不仅解释了现有方法为何次优，还直接指导了解耦奖励的设计。这种"先理论证明失败，再对症设计方案"的研究范式值得学习。
- NRP的概念定义简洁而深刻——"首次得出正确答案的最短前缀"，将模糊的"overthinking"概念精确化为可操作的token级标签。

## 局限与展望

- NRP检测器的质量直接影响方法效果，检测错误可能导致必要推理被惩罚
- 当前chunk分割依赖预定义分隔符（如换行），更精细的语义分割可能带来进一步提升
- 实验仅覆盖数学/编程/科学推理，对自然语言推理等软任务的泛化性未验证

## 相关工作与启发

- **vs ThinkPrune**: ThinkPrune虽减少同量token，但部分减少来自必要推理token（PNRP得分低），导致性能下降；DeCS精准减少非NRP部分
- **vs LC-R1**: LC-R1残留~10%冗余，DeCS通过解耦奖励进一步压缩
- **vs GRPO+长度惩罚**: 理论证明GRPO+长度惩罚必然导致高熵token退化（Lemma 2），DeCS通过NRP保护这些token

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论分析深刻且指导方法设计，NRP概念和解耦奖励方案原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 7基准+2模型规模+骨干泛化+5个研究问题分析，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，分析透彻，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 解决推理型LLM的核心效率问题，50%+压缩无损性能的实用价值极高
