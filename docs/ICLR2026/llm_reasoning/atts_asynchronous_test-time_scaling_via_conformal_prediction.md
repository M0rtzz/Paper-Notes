---
title: >-
  [论文解读] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction
description: >-
  [ICLR 2026][LLM推理][测试时缩放] 提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销…
tags:
  - "ICLR 2026"
  - "LLM推理"
  - "测试时缩放"
  - "推测解码"
  - "共形预测"
  - "asynchronous inference"
  - "rejection sampling"
---

# ATTS: Asynchronous Test-Time Scaling via Conformal Prediction

**会议**: ICLR 2026  
**arXiv**: [2509.15148](https://arxiv.org/abs/2509.15148)  
**代码**: [https://github.com/menik1126/Asynchronous-Test-Time-Scaling](https://github.com/menik1126/Asynchronous-Test-Time-Scaling)  
**领域**: LLM推理  
**关键词**: 测试时缩放, 推测解码, 共形预测, asynchronous inference, rejection sampling

## 一句话总结
提出 ATTS，一个基于 conformal prediction 的异步 test-time scaling 框架，通过将 rejection sampling 重构为假设检验过程来消除同步开销，在 MATH/AIME 等数学推理任务上实现最高 56.7x 加速和 4.14x 吞吐量提升，且无精度损失；1.5B/70B 的 draft/target 组合可达到 o3-mini (high) 的 AIME 水平。

## 研究背景与动机
**领域现状**：Test-time scaling（推理时增加计算预算）通过顺序缩放（更长推理链）和并行缩放（更多采样）显著提升 LLM 推理能力。Speculative decoding 是加速推理的自然选择（小模型生成、大模型验证）。

**现有痛点**：当 speculative decoding 遇到 test-time scaling 时面临两个瓶颈：(1) **内存瓶颈**——高并发采样时 KV cache 爆炸，GPU 内存溢出；(2) **同步开销**——rejection sampling 需要对所有候选进行全局排名或 softmax 归一化，随采样轮数指数增长的同步等待时间成为主要瓶颈。

**核心矛盾**：高效的 test-time scaling 需要同时沿并行和顺序维度缩放，但全局同步排名和归一化操作使得异步执行不可行——所有候选必须等待其他候选完成才能进行排名。

**本文目标** 如何在保持统计保证的前提下，消除 test-time scaling 中 rejection sampling 的同步瓶颈？

**切入角度**：引入 conformal prediction 构建 prediction set，用 p-value 替代归一化 softmax 分数做序数分类，使每个候选可以独立判断接受/拒绝，无需等待全局排名。

**核心 idea**：用 conformal prediction 的 p-value 替代全局排名实现异步 rejection sampling，消除 test-time scaling 的同步瓶颈。

## 方法详解

### 整体框架
ATTS 要解决的问题是：把 speculative decoding 搬进 test-time scaling 后，rejection sampling 的全局排名/归一化让所有候选必须互相等待，同步开销随采样轮数爆炸。它的破局点是把"哪些候选该接受"从一次全局排名，改成每个候选各自独立做的一次假设检验。

整条 pipeline 分三步走：Draft 模型先并行生成 $m$ 个候选推理链；接着对每个候选独立算一个 conformal p-value，与阈值 $\alpha$ 一比就能立刻决定接受还是拒绝，不必等其他候选算完；被接受的候选再交给 Target 模型继续往下生成。多跑几轮就是顺序缩放，每轮多放几个候选就是并行缩放——两个维度都能放大，而放大时不再积累同步等待。

### 关键设计

**1. 异步算术强度：把"同步开销"摆到台面上量化。**

要论证异步有必要，先得说清同步到底有多贵。传统的算术强度只权衡计算量和内存访问，但在 test-time scaling 里真正卡脖子的是同步等待。ATTS 因此定义异步算术强度 $r = T_c / (T_m + T_s) \approx T_c / T_s$，其中 $T_c$ 是计算时间、$T_m$ 是内存访问时间、$T_s$ 是同步开销；之所以能近似掉 $T_m$，正是因为这里 $T_s \gg T_m$。随着采样数增多，$r$ 持续下降，定量地说明瓶颈不在算力也不在显存带宽，而在"等别人算完"这件事上——这就为后面的异步改造提供了量化依据和优化目标。

**2. 基于 conformal prediction 的序数分类：用 p-value 替掉全局排名。**

同步的根源是 rejection sampling 要对所有候选做 softmax 归一化或全局排名，这两件事都得"凑齐所有候选"才能算。ATTS 把它重写成一个假设检验：对每个候选 $k$ 先算一个**非归一化**的 conformity score $s_\xi^k = -\ell(X_\xi, \hat{Y}_\xi^k)$（$\ell$ 为损失，score 越高代表候选越可信），再算出它的 conformal p-value $p_\xi^k$，与阈值 $\alpha$ 比较即可判定是否落入 prediction set。关键在于 p-value 不依赖全局归一化——它只把当前候选的 score 和 calibration set 里的历史 scores 比一比就能得到，于是每个候选都能独立、异步地评估，彻底拆掉了"必须等齐"的约束。这套构造还自带统计保证，能给出边际覆盖与条件覆盖两种形式，即 $\mathbb{P}(y \in C_\alpha(Y)) \geq 1 - \alpha$，保证高质量候选不会被误丢。

**3. 在线校准与 budget 控制：没有验证集也能动态维护 calibration set。**

conformal p-value 要靠一份 calibration set 做参照，但 test-time scaling 现场并没有预留的 held-out 数据。ATTS 用一个 memory bank 在线积累历史采样的 scores，随着测试推进持续更新这份 calibration set。更妙的是，阈值 $\alpha$ 直接控制 rejection rate，从而让 prediction set 的大小恰好等于预先设定的 budget $B$——这相当于给并发采样上了一个硬闸门，把 KV cache 占用钉在预算内，避免高并发时 GPU OOM。

### 损失函数 / 训练策略
无需训练（training-free, lossless）。ATTS 完全工作在推理时，不修改模型权重，对 draft / target 模型组合也无侵入。

## 实验关键数据

### 主实验（跨不同 Draft-Target 模型家族）

| Dataset | Draft Model | Target Model | Accuracy | Mar Speedup | Con Speedup |
|---------|------------|-------------|----------|-------------|-------------|
| MATH100 | Qwen2.5-7B-Inst | QwQ-32B | 96.0% (=TM) | **7.19x** | 5.35x |
| AIME24 | Qwen2.5-7B-Inst | QwQ-32B | 46.7% | **5.71x** | 10.10x |
| AIME25 | Qwen2.5-7B-Inst | QwQ-32B | 40.0% | **14.50x** | 12.82x |
| AMC23 | Qwen2.5-7B-Inst | QwQ-32B | 76.0% | **10.42x** | 8.20x |

### 大规模缩放结果

| 配置 | 说明 |
|------|------|
| 最高 56.7x 加速 | test-time scaling 场景下 |
| 4.14x 吞吐量提升 | 同时顺序+并行缩放 |
| 1.5B/70B draft/target | 达到 o3-mini (high) 的 AIME 水平 |
| Rejection rate 控制准确 | 与预设 $\alpha$ 高度一致 |

### 关键发现
- **跨家族 draft-target 组合有效**：即使 draft 和 target 来自不同模型家族（Qwen → QwQ, Llama → QwQ），ATTS 仍能提供有效的加速
- 红色标记的结果表示"无损加速"——加速后精度等于或超过 target model baseline
- 异步方案在采样数较多时优势明显——同步开销是指数增长的，而异步是常数
- 条件覆盖（per-instance 保证）通常比边际覆盖更保守但更可靠，对不同场景需权衡

## 亮点与洞察
- **conformal prediction 在 LLM 推理加速中的创新应用**：将统计学的 conformal prediction 引入 speculative decoding，用假设检验替代全局排名，是一个优雅的理论-工程结合
- **异步算术强度指标**：提供了量化 test-time scaling 瓶颈的新工具，可用于指导系统设计
- **"无损加速"的工程实用性**：training-free、model-agnostic、有统计保证，可直接部署

## 局限与展望
- 在线校准需要积累足够的历史 scores，冷启动阶段可能不够准确
- 精度在某些 draft-target 组合下不如 target model baseline（尤其是弱 draft model），说明 draft 质量仍很重要
- 仅在数学推理任务上验证，对开放式生成任务的适用性未知
- 需要同时部署 draft 和 target model，对 GPU 资源有额外需求

## 相关工作与启发
- **vs 标准 Speculative Decoding**: ATTS 将 speculative decoding 从 token-level 扩展到 chain-level（整条推理链），并解决了 test-time scaling 场景特有的同步瓶颈
- **vs BoN (Best-of-N)**: BoN 需要 N 条完整推理链全部生成完才能选择，ATTS 可以异步逐步筛选，大幅降低延迟
- **vs TPT 早停方法**: 早停可能剪掉正确推理路径，ATTS 通过 conformal 保证不丢失高质量候选

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ conformal prediction + 异步 test-time scaling 的结合非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多 benchmark、多 draft-target 组合、加速和精度双指标
- 写作质量: ⭐⭐⭐⭐ 理论推导扎实，系统分析清晰
- 价值: ⭐⭐⭐⭐⭐ 为 test-time scaling 的高效部署提供了实用框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)

</div>

<!-- RELATED:END -->
