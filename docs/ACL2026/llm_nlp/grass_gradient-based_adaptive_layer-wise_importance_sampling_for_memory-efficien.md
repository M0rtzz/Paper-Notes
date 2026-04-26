---
title: >-
  [论文解读] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning
description: >-
  [ACL 2026][LLM/NLP][层级采样] 提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。
tags:
  - ACL 2026
  - LLM/NLP
  - 层级采样
  - 梯度重要性
  - 内存高效微调
  - 优化器状态卸载
  - 自适应训练
---

# GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning

**会议**: ACL 2026  
**arXiv**: [2604.07808](https://arxiv.org/abs/2604.07808)  
**领域**: LLM效率 / 参数高效微调  
**关键词**: 层级采样, 梯度重要性, 内存高效微调, 优化器状态卸载, 自适应训练

## 一句话总结

提出 GRASS 框架，使用均值梯度范数（MGN）作为任务感知和训练阶段感知的层重要性指标，自适应地采样和更新模型层子集进行微调，配合层级优化器状态卸载机制，在平均准确率提升最高 4.38 分的同时减少最高 19.97% 的内存使用。

## 研究背景与动机

**领域现状**：LLM 的全参数微调（FFT）在下游任务适配中效果最好，但随着模型规模增长，GPU 显存需求成为瓶颈。参数高效微调（PEFT）方法如 LoRA 通过只更新少量参数来降低内存，是目前最流行的折中方案。

**现有痛点**：LoRA 等低秩方法虽然高效，但低秩参数化限制了模型表达能力，性能不可避免地低于 FFT。层级微调方法（如 LISA）提供了另一条路——每次只激活部分层进行全参数更新，避免低秩约束。但 LISA 采用静态均匀采样策略选择层，隐式假设各层重要性恒定，这与实际情况不符。例如 LISA 在 GSM8K 上比 FFT 低 4.4%，SingleEq 低 8.9%。

**核心矛盾**：层级微调面临层重要性的动态性问题——不同任务需要更新不同的层，同一任务不同训练阶段重点层也在变化，而静态选择策略无法捕捉这种动态性。

**本文目标**：设计一种能自适应感知任务和训练阶段的层采样策略，在保持层级微调内存优势的同时逼近甚至超越 FFT 性能。

**切入角度**：梯度直接编码了损失对参数更新的敏感度——一阶 Taylor 近似下，梯度范数大的层更新后对训练目标影响更大。因此梯度统计量是实时层重要性的天然指标。

**核心 idea**：用均值梯度范数（MGN）动态量化各层对损失下降的贡献，通过 softmax 转化为采样概率并周期性更新，自适应选择最重要的层进行微调。

## 方法详解

### 整体框架

GRASS 分为两个阶段：(1) 探测阶段（前 Tp 步）——标准前向/后向传播但不更新参数，收集各层初始 MGN；(2) 自适应微调阶段——交替进行层采样（根据 MGN 概率采样 gamma 层更新）和概率刷新（每 Tu 步重算 MGN 并更新采样概率），同时用层级优化器状态卸载减少显存。

### 关键设计

1. **均值梯度范数（MGN）层重要性度量**:

    - 功能：提供任务感知和训练阶段感知的层重要性评估
    - 核心思路：对每一层 l，在连续 T 步上聚合归一化梯度幅值：$m_l(T) = \frac{1}{T}\sum_{t=1}^T \sqrt{\frac{1}{N_p^{(l)}} \|g_t^{(l)}\|_2^2}$。除以参数数量使不同大小的层可比。实验验证：TinyLlama 在算术推理和常识推理上各层的归一化 MGN 分布差异显著，第 20 层在常识推理中重要性高但在算术推理中不突出
    - 设计动机：LISA 用均匀采样、OWS 用权重范数、IST 用响应抑制+强化学习，都是静态或启发式的。梯度是最直接反映当前优化需求的信号

2. **自适应层采样概率更新**:

    - 功能：将动态 MGN 信号转化为持续优化的层选择策略
    - 核心思路：每隔 Tu 步，MGN 通过带温度的 softmax 转化为概率：$p^{(l)} = \frac{\exp(m_l/\tau)}{\sum_i \exp(m_i/\tau)}$，据此采样 gamma 层。冻结层保留上轮 MGN，采样层用指数移动平均更新：$m_l(T) = \alpha m_l(T_u) + (1-\alpha)m_l(T-T_u)$
    - 设计动机：如果只用初始 MGN 固定策略（静态 GRASS），训练推进后重要性分布变化会导致策略次优

3. **层级优化器状态卸载（Overlapped Offloading）**:

    - 功能：进一步降低 GPU 显存而不牺牲训练吞吐量
    - 核心思路：GPU 只保留当前更新层的优化器状态，其余存 CPU。关键创新是计算-通信重叠：更新第 i 层时异步预取第 i+1 层状态（HtoD），同时回写第 i-1 层状态（DtoH），传输与计算完全重叠
    - 设计动机：层级微调的所有可训练层都需保存优化器状态，全留 GPU 爆显存，全放 CPU 有延迟。重叠卸载取得最优平衡，内存增长从 1.63GB 降至 0.14GB

### 损失函数 / 训练策略

GRASS 不改变原始训练损失，仅改变哪些层参与梯度计算和参数更新。冻结层参与前向但不产生梯度。探测阶段跳过参数更新和优化器状态管理，开销可控。

## 实验关键数据

### 主实验

算术推理任务上的准确率对比（六个 benchmark 平均）：

| 模型 | 方法 | MultiArith | GSM8K | SingleEq | 平均 |
|------|------|-----------|-------|----------|------|
| TinyLlama | FFT | 64.17 | 15.16 | 42.92 | 33.48 |
| TinyLlama | LoRA r=128 | 61.17 | 15.16 | 38.19 | 29.84 |
| TinyLlama | LISA | 65.00 | 17.74 | 43.11 | 33.63 |
| TinyLlama | **GRASS** | **68.00** | 17.13 | 42.52 | **34.22** |
| Gemma-2B | FFT | 86.67 | 42.53 | 80.12 | 60.16 |
| Gemma-2B | LISA | 90.17 | 40.18 | 75.00 | 56.46 |
| Gemma-2B | **GRASS** | **93.50** | **43.06** | 78.35 | **60.65** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| GRASS (完整) | 34.22 (TinyLlama avg) | 完整自适应框架 |
| 静态 GRASS | 部分任务下降 | 只用初始 MGN 不更新概率 |
| w/o Offloading | +1.49GB 显存 | 优化器状态全留 GPU |
| FFT vs GRASS 显存 | 51.3GB vs 19.1GB | LLaMA2-7B 减少 62.8% |

### 关键发现
- GRASS 在 TinyLlama 和 Gemma-2B 上甚至超越 FFT，说明自适应层选择可能起到隐式正则化效果
- 相比 LoRA r=128，GRASS 在 TinyLlama 上提升 4.38 分（34.22 vs 29.84）
- LISA 性能在不同任务间波动大，GRASS 表现更稳定
- 长序列（1792 tokens）时 LoRA/DoRA 超出 24GB 显存限制，GRASS 仍在 23.25GB 以内
- 常识推理任务上 GRASS 同样全面优于其他 PEFT 方法，显示跨任务泛化能力

## 亮点与洞察
- **梯度范数作为层重要性信号**：相比权重范数等静态指标，梯度范数直接反映当前训练目标对各层的需求，理论直觉清晰且实验有效。可迁移到混合精度训练、知识蒸馏中的层选择等场景
- **超越 FFT 的"意外"发现**：选择性更新可能带来正则化效果，与 dropout 和模型剪枝的理论有呼应，暗示并非所有层在所有时刻都需要更新
- **计算-通信重叠的工程价值**：优化器状态的层级卸载+重叠传输将内存增长从 1.63GB 压到 0.14GB，展示了算法设计与系统优化的协同效果

## 局限与展望
- 实验仅在 1B-7B 规模模型上验证，7B 上 GRASS 已不如 FFT，更大模型效果未知
- 超参数较多（gamma, Tp, Tu, Ts, tau, alpha），调参成本可能抵消部分便利性
- 仅在单 GPU 上实验，多卡分布式训练场景下的适配未讨论
- 未与最新的 GaLore、量化微调等内存高效方法对比

## 相关工作与启发
- **vs LISA**: LISA 使用均匀静态采样，在某些任务上严重退化，GRASS 通过自适应采样全面改善
- **vs LoRA/DoRA**: LoRA 受低秩约束限制表达能力，GRASS 保持全秩更新同时通过层选择降低内存
- **vs LIFT**: LIFT 用固定前到后更新顺序，缺乏层重要性判断，GRASS 的梯度驱动选择更有针对性

## 评分
- 新颖性: ⭐⭐⭐⭐ 梯度范数作为层采样权重的 idea 直觉清晰，自适应更新+卸载组合有效
- 实验充分度: ⭐⭐⭐⭐ 三个模型规模x两大类任务，消融充分，但缺少更大模型对比
- 写作质量: ⭐⭐⭐⭐ 行文清晰，动机和方法的逻辑链完整
- 价值: ⭐⭐⭐⭐ 为层级微调提供了实用且通用的自适应框架，对内存受限场景有实际意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](../../ACL2025/llm_nlp/a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](../../ACL2025/llm_nlp/gorp_continual_gradient_projection.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](../../ACL2025/llm_nlp/efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](../../ACL2025/llm_nlp/gapo_multi_objective_alignment.md)

<!-- RELATED:END -->
