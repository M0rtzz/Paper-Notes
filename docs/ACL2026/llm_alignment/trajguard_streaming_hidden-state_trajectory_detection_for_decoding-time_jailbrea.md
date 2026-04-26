---
title: >-
  [论文解读] TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense
description: >-
  [ACL 2026][LLM对齐][越狱防御] 本文提出 TrajGuard，一种无需训练的解码时越狱防御框架，通过滑动窗口聚合关键层隐藏状态轨迹实时量化风险，仅在风险持续超过阈值时触发轻量级语义裁判，在 12 种越狱攻击上实现 95% 平均防御率，检测延迟仅 5.2ms/token，误报率低于 1.5%。
tags:
  - ACL 2026
  - LLM对齐
  - 越狱防御
  - 隐藏状态轨迹
  - 解码时检测
  - 实时安全
  - 无训练防御
---

# TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense

**会议**: ACL 2026  
**arXiv**: [2604.07727](https://arxiv.org/abs/2604.07727)  
**代码**: 无  
**领域**: LLM对齐 / AI安全  
**关键词**: 越狱防御, 隐藏状态轨迹, 解码时检测, 实时安全, 无训练防御

## 一句话总结

本文提出 TrajGuard，一种无需训练的解码时越狱防御框架，通过滑动窗口聚合关键层隐藏状态轨迹实时量化风险，仅在风险持续超过阈值时触发轻量级语义裁判，在 12 种越狱攻击上实现 95% 平均防御率，检测延迟仅 5.2ms/token，误报率低于 1.5%。

## 研究背景与动机

**领域现状**：LLM 已深度集成到现实服务中，其安全性至关重要。尽管经过严格的安全对齐训练（RLHF等），精心构造的越狱攻击仍能绕过安全护栏，在经过 RLHF 对齐的模型上实现高攻击成功率。

**现有痛点**：现有防御主要依赖静态检测——要么在输入端过滤提示（如 Llama Guard），要么在输出端检查完整回复。输入端过滤无法检测语义伪装的越狱提示，输出端过滤虽然更有效但需要生成完整回复后才能审查，引入不可忽略的端到端延迟。一些利用模型内部激活的方法仍然操作于静态的提示表示上，且依赖高维几何分数，可解释性差。

**核心矛盾**：越狱风险不是在某个时刻瞬间触发的，而是在解码过程中通过上下文的恶意意图逐步积累形成的。现有方法将安全检测视为离散的二分类任务，忽略了解码过程中语义的动态演化——这是当前防御范式的关键盲区。

**本文目标**：利用解码过程中隐藏状态的动态轨迹来实现实时越狱检测，不依赖额外训练的安全模型。

**切入角度**：作者通过实证分析发现了一个关键的"伪装-暴露"模式：越狱提示在潜空间中与良性提示纠缠（语义伪装），但一旦模型开始生成具体的有害步骤，隐藏状态就会持续漂移向恶意区域。这种漂移在早期解码片段中就已出现。

**核心 idea**：将解码过程中隐藏状态的时序轨迹作为越狱检测信号，通过"流式几何监控 + 按需语义裁判"的粗到细架构，实现低开销、实时的越狱拦截。

## 方法详解

### 整体框架

TrajGuard 采用粗到细的层次架构，包含两个协同组件：(1) SGS（流式几何监控）持续监控隐藏状态轨迹作为第一道防线，使用轻量级向量计算筛查潜在风险段；(2) PAIR-Judge（提示-回答推理裁判）仅在 SGS 检测到持续异常时被触发，提供准确的语义裁决。对于几乎所有良性交互，TrajGuard 仅依赖 SGS 模块运行在低开销的"仅监控"模式。

### 关键设计

1. **流式几何监控 (SGS)**:

    - 功能：从嘈杂的隐藏状态流中提取稳定的风险信号，实时判断解码路径是否偏离良性行为
    - 核心思路：首先用 MVD（均值向量差异）指标选择 Top-K（K=8）关键层；在选定层上建模良性/恶意模式的高斯分布。解码时计算每个 token 隐藏状态到良性和恶意质心的 Mahalanobis 距离之差 $r_{l,t} = d^{\mathcal{B}}_{l,t} - d^{\mathcal{M}}_{l,t}$；通过三阶段聚合：层内滑动窗口（w=8）截断均值 → 跨层平均 → EWMA 时序平滑，得到稳定的流式风险分数 $p_t$；仅当风险分数连续 k=3 步超过阈值 γ 时才触发警报
    - 设计动机：单步 token 的风险判断噪声大，真正的越狱表现为持续驻留在高风险区域。滞后触发机制有效抑制了瞬态几何噪声，确保仅持续的恶意意图才触发昂贵的裁判过程

2. **提示-回答推理裁判 (PAIR-Judge)**:

    - 功能：对 SGS 标记的异常进行语义级别的安全裁决，将高维内部信号转化为可解释的安全决策
    - 核心思路：当 SGS 触发警报时暂停生成，将当前上下文（提示 x + 已生成前缀 $y_{\leq t}$）包装进安全系统提示，送入安全对齐的 LLM 进行 SAFE/UNSAFE 二元判决 $d = \mathcal{M}_{judge}(\mathcal{P}(x, y_{\leq t}))$。如果判定 UNSAFE 则立即终止生成
    - 设计动机：几何接近恶意区域不等同于语义上的恶意。需要语义层面的验证来避免误判，同时保持可解释性

3. **闭环状态重置 (State Reset)**:

    - 功能：当 PAIR-Judge 判定 SAFE 时，清除 SGS 积累的"假阳性"风险动量
    - 核心思路：如果语义裁判认为当前内容安全，则将 SGS 的风险分数 $S_t$ 强制重置为初始安全值，防止系统因历史几何偏差在后续解码中反复触发警报
    - 设计动机：没有状态重置，一次误触发可能导致后续连锁误报，严重影响正常使用

### 损失函数 / 训练策略

TrajGuard 是完全无训练的框架。只需要一个预处理步骤：使用 8,000 条良性指令和 10,000 条恶意指令来估计隐藏空间中的安全/不安全区域分布（质心和协方差矩阵），采用收缩正则化 $\widehat{\Sigma}_{\star,l} = \Sigma_{\star,l} + \lambda I$ 增强高维空间中的数值稳定性。

## 实验关键数据

### 主实验

| 模型 | 防御方法 | 12种攻击平均ASR↓ | 最佳单攻击ASR |
|------|---------|-----------------|-------------|
| Llama-2-7B | No Defense | 0.52 | - |
| Llama-2-7B | Llama Guard 3 | 0.20 | GCG: 0.02 |
| Llama-2-7B | Qwen3Guard | 0.07 | GCG: 0.00 |
| Llama-2-7B | **TrajGuard** | **0.02** | 多数攻击: 0.00 |
| Llama-3.1-8B | No Defense | 0.57 | - |
| Llama-3.1-8B | **TrajGuard** | **0.04** | - |
| Mistral-7B | No Defense | 0.75 | - |
| Mistral-7B | **TrajGuard** | **0.05** | - |

| 指标 | TrajGuard 表现 |
|------|---------------|
| 平均防御率 | 95% |
| 检测延迟 | 5.2 ms/token |
| 误报率 (XSTest) | < 1.5% |
| Alpaca 正常任务保持率 | 高（详见论文） |

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| Full TrajGuard | AVG ASR ≈ 0.02-0.05 | 完整模型 |
| w/o PAIR-Judge | 误报率上升 | 仅靠几何监控会将安全但敏感的内容误判 |
| w/o State Reset | 连锁误报 | 误触发后后续解码持续报警 |
| w/o 持续性触发 | 噪声增加 | 单步判断容易被瞬态波动影响 |
| 不同窗口大小 w | w=8 最优 | 太小噪声大，太大延迟高 |

### 关键发现

- **隐藏状态轨迹比输入提示提供更强更稳定的越狱信号**：越狱提示在潜空间中与良性提示纠缠（t=0时重叠），但解码开始后隐藏状态持续向恶意区域漂移
- **不同模型的"漂移延迟"差异显著**：Llama-2-7B 在 37 步后才开始恶化，而 Vicuna-7B 几乎立即下降，反映了不同模型安全对齐的鲁棒性差异
- **TrajGuard 在多数攻击上将 ASR 降至接近 0**，尤其在 GCG、AutoDAN、PAIR 等主流攻击上表现突出
- Cipher 类攻击是唯一仍有一定成功率的攻击类型（ASR 0.10-0.25），可能因为加密输入在隐藏空间中的表示模式与常规越狱不同

## 亮点与洞察

- **"伪装-暴露"观察非常精妙**：越狱提示的语义伪装在输入阶段有效，但模型一旦开始生成具体有害步骤，内部表示就不可避免地向恶意区域漂移。这个观察将越狱检测从静态分类问题转化为动态轨迹监控问题
- **粗到细的层次设计实用性强**：绝大部分时间仅运行轻量级的几何监控（5.2ms/token），只有疑似风险时才调用昂贵的语义裁判，实现了精度和效率的极佳平衡
- **完全无训练**的特性使其可即插即用到任何开源 LLM，无需额外安全数据或微调成本
- **闭环状态重置机制**可以迁移到其他异常检测系统中，解决"一次误报导致连锁反应"的通用问题

## 局限与展望

- 需要预先构建良性/恶意区域的分布估计，依赖 8K+10K 标注数据的质量和覆盖范围
- 对 Cipher 类加密攻击防御效果相对较弱，隐藏状态可能未充分暴露加密输入的恶意意图
- 仅在 7B-8B 规模的开源模型上验证，对更大规模或闭源模型的适用性未知
- PAIR-Judge 使用目标模型本身作为裁判，在模型安全对齐较弱时裁判质量可能下降

## 相关工作与启发

- **vs Llama Guard 3**：静态输入/输出过滤器，无法利用解码过程中的动态信息。TrajGuard 在几乎所有攻击上大幅优于它
- **vs SafeDecoding (Xu et al., 2024)**：需要训练安全专家模型来重新加权解码概率，TrajGuard 无需训练，直接利用基础模型的隐藏状态
- **vs ShieldHead (Xuan et al., 2025)**：附加 token 级安全头需要额外训练，且仍是逐 token 的静态判断，不建模时序轨迹
- **vs Goal Prioritization (Zhang et al., 2024)**：在部分模型上表现不佳（Mistral-7B 上 AVG ASR 0.44），说明提示工程方法对攻击的鲁棒性不足

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将解码时隐藏状态轨迹用于越狱检测，"伪装-暴露"观察新颖且有说服力
- 实验充分度: ⭐⭐⭐⭐⭐ 12种攻击、4个模型、多个基线、完整消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机推导自然，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 无训练、低延迟、高防御率的实时防御方案，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](../../ACL2025/llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](../../ACL2025/llm_alignment/hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)

<!-- RELATED:END -->
