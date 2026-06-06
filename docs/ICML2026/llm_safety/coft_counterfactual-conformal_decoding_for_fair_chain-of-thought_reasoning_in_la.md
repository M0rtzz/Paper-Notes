---
title: >-
  [论文解读] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models
description: >-
  [ICML 2026][LLM安全][反事实公平性] COFT 通过在解码时构造反事实掩码分支并与原始分支进行 logit 融合，再用双分支分裂共形预测过滤 token，以无训练、免梯度的方式在冻结 LLM 上实现了逐步 token 级别的反事实公平性保证…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "反事实公平性"
  - "共形预测"
  - "链式推理去偏"
  - "解码时干预"
  - "无训练去偏"
---

# COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models

**会议**: ICML 2026  
**arXiv**: [2605.30641](https://arxiv.org/abs/2605.30641)  
**代码**: 无  
**领域**: LLM安全/公平性  
**关键词**: 反事实公平性, 共形预测, 链式推理去偏, 解码时干预, 无训练去偏  

## 一句话总结

COFT 通过在解码时构造反事实掩码分支并与原始分支进行 logit 融合，再用双分支分裂共形预测过滤 token，以无训练、免梯度的方式在冻结 LLM 上实现了逐步 token 级别的反事实公平性保证，将偏见指标降低 30–55%（中位 38%）且几乎不损失任务性能。

## 研究背景与动机

**领域现状**：大语言模型（LLM）在 CoT（链式推理）生成过程中会逐 token 暴露并放大训练数据中的社会偏见——即使最终答案看似中立，推理轨迹中也可能包含有害的刻板联想。

**现有痛点**：已有的去偏方案各有局限。数据清洗和微调需要重新训练且可能损害通用能力；辅助分类器引导的方法（如 DExperts、GeDi）依赖外部模型并继承其盲区；表示空间去偏（如 INLP）做全局线性投影，无法适应特定 prompt 语义，且可能误删合法内容。

**核心矛盾**：上述方法都缺少两个关键属性的同时满足：(1) **逐步统计保证**——在每一步解码时，无法保证所选 token 在敏感属性替换后仍然稳定；(2) **局部反事实对等**——公平性目标通常只在聚合层面定义，而非逐 token 操作。

**本文目标**：设计一个解码时框架，同时实现三个属性——逐 token 反事实不变性、无梯度/模型无关（适用于冻结权重）、可审计的逐步边际保证。

**切入角度**：将每个 prompt 同时构造为原始（factual）和掩码（masked）两个分支，通过对比两者的 logit 分布差异来定位并消除敏感属性驱动的偏差。再利用共形预测（Conformal Prediction）的分布无关保证来过滤不安全 token。

**核心 idea**：用反事实掩码 + logit 凸插值融合 + 双分支共形过滤，三阶段联合实现无训练的逐 token 反事实公平性解码。

## 方法详解

### 整体框架

COFT 是一个纯推理时框架，作用于冻结的因果语言模型。对于给定 prompt $p$，COFT 在每一步解码时执行三个阶段：(1) 将 prompt 中的敏感片段替换为中性哨兵 token 生成掩码 prompt $\tilde{p} = M(p)$；(2) 分别对原始和掩码 prompt 做前向推理，对两组 logit 做凸插值融合以衰减属性驱动的偏差；(3) 用离线校准的双分支共形预测阈值过滤 token，仅从两个分支都高概率支持的候选集中采样。整个流程只需多做一次缓存的前向传播，不需要训练、梯度或外部分类器。

### 关键设计

1. **反事实掩码（Counterfactual Masking）**:

    - 功能：生成与原始 prompt 在结构上完全对齐的"去敏感化"版本，作为反事实分支
    - 核心思路：定义确定性掩码算子 $M$，将 prompt 中每个敏感片段 $s \in S$（如性别、种族标识词）替换为中性哨兵 token `[MASK]`。关键设计是**保持 token 数量不变**：若敏感片段被 tokenizer 切分为 $k$ 个 token，则替换为 $k$ 个哨兵副本，确保两个分支在绝对位置上严格对齐，使 $z_t^F \leftrightarrow z_t^{CF}$ 的逐位配对比较有效
    - 设计动机：删除敏感片段会破坏语法和注意力几何；替换为另一身份会注入新属性；唯有掩码既保留结构又切断与敏感属性的直接词汇关联

2. **反事实 Logit 融合（Counterfactual Logit Fusion）**:

    - 功能：在 logit 空间中衰减由敏感属性驱动的 token 概率偏差
    - 核心思路：定义逐 token 属性敏感度 $\Delta_t = z_t^F - z_t^{CF}$，通过凸插值生成融合 logit $\hat{z}_t = (1-\lambda) z_t^F + \lambda z_t^{CF}$，其中 $\lambda \in [0,1]$ 控制去偏强度。等价于 $\hat{\pi}_t(v) \propto (\pi_t^F(v))^{1-\lambda} (\pi_t^{CF}(v))^{\lambda}$，即两个分支概率分布的归一化几何混合。$\lambda$ 通过验证集 Pareto 曲线的拐点选取（通常 $\lambda \approx 0.6$）
    - 设计动机：融合先于过滤执行，可提前移除虚假放大方向，使后续共形过滤在已对齐的高概率区域上操作，减少误拒和过度保守阈值

3. **双分支分裂共形过滤（Dual-Branch Split-Conformal Filtering）**:

    - 功能：为每步解码构造一个经统计认证的候选 token 集，提供分布无关的边际覆盖保证
    - 核心思路：定义双分支非一致性得分 $s_t(v) = 1 - \min\{\hat{\pi}_t(v), \pi_t^{CF}(v)\}$——仅当 token $v$ 在融合分布和掩码分布中都有足够高概率时，该得分才小。离线在校准集上计算所有真实 next-token 的得分，取 $(1-\alpha)$ 分位数 $q_t$ 作为阈值。在线解码时，候选集 $C_t = \{v : \min\{\hat{\pi}_t(v), \pi_t^{CF}(v)\} \geq \tau_t\}$（$\tau_t = 1 - q_t$），然后从 $\hat{\pi}_t$ 限制在 $C_t$ 上的条件分布采样；若 $C_t = \emptyset$ 则回退到 $\arg\max$
    - 设计动机：单分支共形预测无法保证反事实稳定性，双分支要求 token 同时被两个世界支持，直接操作化了反事实对等

## 实验关键数据

### 主实验：偏见度量

| 数据集 | 指标 | Vanilla | SDD | DExperts | DT-CD | COFT | 改善(vs DT-CD) |
|--------|------|---------|-----|----------|-------|------|----------------|
| StereoSet (LLaMA-13B) | Bias↓ | 0.41 | 0.36 | 0.33 | 0.31 | **0.26** | -16% |
| CrowS-Pairs (LLaMA-13B) | Acc↑ | 58.7 | 60.1 | 61.0 | 61.3 | **63.5** | +2.2 |
| BBQ (LLaMA-13B) | Bias Rate↓ | 0.27 | 0.22 | 0.20 | 0.19 | **0.14** | -26% |
| BOLD (LLaMA-13B) | Toxicity↓ | 0.123 | 0.105 | 0.099 | 0.094 | **0.079** | -16% |
| Utrecht (LLaMA-13B) | DP Gap↓ | 0.184 | 0.153 | 0.149 | 0.141 | **0.118** | -16% |
| COMPAS (LLaMA-13B) | Bias Gap↓ | 0.161 | 0.147 | 0.141 | 0.136 | **0.119** | -12% |
| BBQ (Mistral-7B-Inst) | Bias Rate↓ | 0.24 | 0.20 | 0.18 | 0.17 | **0.12** | -29% |
| Utrecht (Mistral-7B-Inst) | DP Gap↓ | 0.173 | 0.146 | 0.141 | 0.136 | **0.112** | -18% |

### 消融实验

| 配置 | BiasAvg↓ | UtilityAvg↑ | 说明 |
|------|----------|-------------|------|
| COFT (完整) | **0.129** | 68.0 | 三阶段全开 |
| w/o 融合 (仅CP) | 0.171 | 68.2 | 去掉 logit 融合后偏见指标升高 32% |
| 单分支CP (仅factual) | 0.158 | 68.1 | 无法保证反事实稳定性 |
| 仅融合 (无CP) | 0.149 | 67.9 | 缺少统计认证，残留偏见 |

### 关键发现

- **Logit 融合贡献最大**：单独去掉融合后 BiasAvg 从 0.129 升至 0.171（+33%），是三个组件中贡献最大的，因为它在 logit 源头机械性地衰减了属性驱动的 log-odds 偏差
- **双分支 vs 单分支 CP**：双分支 CP 比单分支额外减少 18% 偏见（0.158→0.129），验证了要求 token 同时在两个世界有高概率支持的必要性
- **任务性能几乎无损**：COFT 在 GSM8K、StrategyQA、ARC-easy、PIQA 上与 Vanilla 差距 ≤ 0.2 点，PPL 和 MAUVE 也几乎无差异
- **效率开销可控**：额外约 10.2% 的吞吐量开销（相当于一次缓存前向传播），峰值显存仅增加 ≤ 0.8 GB
- **$\lambda$ 和 $\alpha$ 的敏感性**：$\lambda$ 在 0.4–0.8 范围内偏见-效用 Pareto 曲线较平稳，默认取拐点 $\lambda \approx 0.6$；$\alpha = 0.10$ 是共形过滤的最佳风险水平

## 亮点与洞察

- **三阶段解耦设计**：掩码→融合→过滤的流水线使每个组件可独立分析和替换，融合先压缩 logit 差异空间再交给共形过滤，二者协同效果远超单独使用，这种"先去噪再认证"的范式可迁移到任何需要在解码时施加约束的场景
- **共形预测在公平性中的创新应用**：将分布无关的统计保证从传统的"置信集"场景拓展到"反事实稳定性认证"，通过双分支得分设计将公平性约束转化为标准的分位数校准问题，方法论上具有通用性
- **完全免训练的实用优势**：只需一次额外缓存前向传播（≤11% 开销），适用于任何冻结 LLM 检查点，无需权重访问、辅助分类器或微调，对于 API-only 部署场景极具实用价值

## 局限与展望

- **敏感片段检测依赖外部工具**：COFT 控制的是解码时对已识别敏感片段的使用，但本身不是万能的隐式偏见检测器；未被识别的代理词（proxy terms）可能逃逸
- **保证为边际覆盖而非条件覆盖**：共形预测提供的是边际（marginal）而非输入条件（conditional）保证，在分布严重偏移时可能失效
- **序列级别保证需要额外处理**：当前逐步保证不直接延伸到整条推理链，需要用联合上界或 rollout score 校准来获得端到端控制
- **$\lambda$ 选取需验证集**：需要一个干净的验证分裂来做 Pareto 拐点选取，在新领域部署时可能需要重新调优

## 相关工作与启发

- **反事实公平性**（Kusner et al. 2017）提供了核心理论框架，COFT 将其操作化为逐 token 局部对等；**共形预测**（Vovk et al. 2005）提供分布无关保证工具，COFT 创新地将其适配到自回归解码的双分支场景
- 与 DExperts/GeDi 等推理时方法相比，COFT 不需要外部分类器，且提供统计保证；与 INLP 等表示去偏方法相比，COFT 是 prompt 级别自适应的而非全局固定投影
- 启发：可将类似的"反事实掩码+共形过滤"范式推广到其他可信 AI 目标（如隐私保护、事实性约束），通过定义不同的掩码算子和非一致性得分来实现不同的安全属性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](../../AAAI2026/llm_safety/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](../../ACL2026/llm_safety/cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](../../ACL2026/llm_safety/reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)

</div>

<!-- RELATED:END -->
