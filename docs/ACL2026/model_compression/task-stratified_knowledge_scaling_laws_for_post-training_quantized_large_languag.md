---
title: >-
  [论文解读] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs
description: >-
  [ACL 2026][模型压缩][后训练量化] 本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。
tags:
  - ACL 2026
  - 模型压缩
  - 后训练量化
  - 缩放定律
  - 知识分层
  - 记忆应用推理
  - 细粒度量化因素
---

# Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs

**会议**: ACL 2026  
**arXiv**: [2508.18609](https://arxiv.org/abs/2508.18609)  
**代码**: 无  
**领域**: 模型压缩 / 量化  
**关键词**: 后训练量化, 缩放定律, 知识分层, 记忆应用推理, 细粒度量化因素

## 一句话总结

本文建立了首个面向后训练量化（PTQ）的任务分层知识缩放定律，将 LLM 能力分为记忆/应用/推理三层，统一建模模型大小、位宽、组大小和校准集大小四个因素，在 293 种 PTQ 配置上验证，揭示推理对精度敏感、应用随规模提升、记忆对校准敏感的差异化规律。

## 研究背景与动机

**领域现状**：PTQ 已成为 LLM 压缩的主流策略（~70% 量化相关研究聚焦 PTQ）。现有缩放定律（如 Chinchilla）主要描述全精度模型的行为，少数量化缩放定律仅考虑模型大小和位宽。

**现有痛点**：(1) 忽略了组大小和校准集大小等细粒度 PTQ 参数的系统性影响；(2) 将所有任务的性能混在一起，无法捕捉量化对记忆、应用和推理能力的差异化影响。

**核心矛盾**：现有缩放定律无法指导"在低位量化下如何配置组大小和校准集大小以保持特定能力"这类实际问题。

**本文目标**：建立统一的四因素幂律框架，为三层知识能力分别拟合缩放定律。

**切入角度**：基于 Bloom's Taxonomy 将 LLM 能力分为记忆（精确事实回忆）、应用（灵活知识运用）和推理（多步逻辑），用 14 个基准测试覆盖三层。

**核心 idea**：$-\ln(\text{Acc}_{\text{adj}}) = A \cdot N^{\alpha} \cdot (\log_2 B)^{\beta} \cdot (\log_2 C_b)^{\gamma} \cdot G^{\delta}$，其中指数 $\alpha, \beta, \gamma, \delta$ 是任务层级特定的，量化了不同能力对各因素的敏感度。

## 方法详解

### 整体框架

对 Qwen3 系列（0.6B-14B）+ Llama-3 系列做系统性 PTQ 配置扫描（位宽 3/4/8、组大小 32/64/128/1024、校准集大小 8/32/128/1024），共 293 种配置。用 GPTQ 作为统一量化方法，在 14 个基准上评估后用 OLS 回归拟合对数变换的幂律。

### 关键设计

1. **四因素统一幂律框架**:

    - 功能：统一建模模型大小 $N$、位宽 $B$、校准集 $C_b$ 和组大小 $G$ 对量化性能的联合影响
    - 核心思路：对 $B$ 和 $C_b$ 取对数以建模边际收益递减效应，用 $-\ln(\text{Acc}_{\text{adj}})$ 变换到无界"损失"空间，通过 OLS 对对数变换后的方程拟合。基线调整 $\text{Acc}_{\text{adj}} = \frac{\text{Acc} - \text{Acc}_{\text{random}}}{1 - \text{Acc}_{\text{random}}}$ 消除不同任务随机基线的差异
    - 设计动机：指数可解释为弹性系数——衡量性能对各因素相对变化的敏感度。对数变换恢复了有界 accuracy 所需的单调凸性

2. **任务分层知识体系**:

    - 功能：分离量化对不同认知层级能力的差异化影响
    - 核心思路：L1 记忆（TriviaQA/NQ/LAMA 等精确事实回忆）、L2 应用（MMLU/Hellaswag 等灵活知识运用）、L3 推理（GSM8K/ARC-C 等多步逻辑），分别拟合独立的缩放定律
    - 设计动机：如果只拟合聚合性能，会掩盖关键差异——推理可能已经崩溃但应用仍然良好

3. **低位场景细粒度因素的关键作用**:

    - 功能：证明在 2-3 位量化下，组大小和校准集不再是可选参数而是防止崩溃的必要条件
    - 核心思路：消融实验显示 $f(N,B)$ 的 $R^2 = 0.91$，加入 $G$ 后跳至 0.95，说明组大小解释了约 4% 的额外方差——而这 4% 恰好集中在低位区域
    - 设计动机：实践者在低位量化时往往使用默认组大小和校准集，可能导致不必要的性能崩溃

### 损失函数 / 训练策略

不涉及训练。GPTQ 用 Hessian 矩阵逐层最小化量化重建误差。

## 实验关键数据

### 主实验

**各层知识能力的缩放指数对比**

| 能力层 | α(N) | β(B) | γ(Cb) | δ(G) | Adj R² |
|--------|------|------|-------|------|--------|
| 通用 | -0.359 | -1.067 | -0.032 | 0.073 | 0.9475 |
| L1 记忆 | -0.315 | -0.964 | **-0.040** | 0.064 | 0.9350 |
| L2 应用 | **-0.400** | -1.100 | -0.030 | 0.075 | 0.9500 |
| L3 推理 | -0.320 | **-1.200** | -0.025 | **0.085** | 0.9300 |

### 关键发现

- 推理精度瓶颈：$\beta_{\text{KR}} = -1.200$（最大绝对值），说明推理对位宽最敏感
- 应用规模响应：$\alpha_{\text{KA}} = -0.400$（最大绝对值），说明应用能力随模型规模显著提升
- 记忆校准敏感：$\gamma_{\text{KM}} = -0.040$（最大绝对值），说明精确事实回忆对校准数据量最敏感
- 四因素模型比二因素基线（N,B）提升 3.5% 的 Adj R²，且 Qwen3-32B 上的外推验证成功

## 亮点与洞察

- 任务分层缩放定律的思路非常有实用价值——指导实践者在资源约束下做出知情的量化配置决策
- 细粒度因素在低位场景的关键性是重要发现——默认配置在 3-bit 下可能导致能力崩溃
- 跨架构（Qwen3→Llama-3）的一致性证明了定律的普适性

## 局限与展望

- 仅使用 GPTQ 一种量化方法
- 2-bit 数据因性能崩溃被排除在拟合外
- 未考虑 QAT 或混合精度场景
- 未覆盖生成任务的评估

## 相关工作与启发

- **vs Chinchilla Laws**: 针对全精度模型，本文扩展到量化场景并增加组大小/校准集因素
- **vs QiD Laws**: 仅建模聚合退化，本文分层建模三种知识能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个任务分层PTQ缩放定律，四因素统一框架
- 实验充分度: ⭐⭐⭐⭐⭐ 293种配置+14基准+跨架构验证+外推测试
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，图表有说服力
- 价值: ⭐⭐⭐⭐⭐ 对LLM量化实践有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ICLR 2026\] A universal compression theory for lottery ticket hypothesis and neural scaling laws](../../ICLR2026/model_compression/a_universal_compression_theory_for_lottery_ticket_hypothesis_and_neural_scaling_.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](../../AAAI2026/model_compression/stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)

<!-- RELATED:END -->
