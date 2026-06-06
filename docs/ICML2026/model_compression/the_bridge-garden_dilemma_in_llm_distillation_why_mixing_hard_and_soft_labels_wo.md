---
title: >-
  [论文解读] The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works
description: >-
  [ICML 2026][模型压缩][知识蒸馏] 作者发现 LLM 蒸馏里"软标签 + 硬标签线性混合"几乎总是打过纯软标签，并把原因从直觉上的"硬标签信息更少但优化更易"修正为"硬标签压低了暴露偏差"，进一步用 Bridge-Garden 分解把生成序列拆成"必须精确的桥"和"可灵活替换的花园"两类位置…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "知识蒸馏"
  - "暴露偏差"
  - "硬软标签混合"
  - "Bridge-Garden 分解"
  - "风险引导"
---

# The Bridge-Garden Dilemma in LLM Distillation: Why Mixing Hard and Soft Labels Works

**会议**: ICML 2026  
**arXiv**: [2605.26246](https://arxiv.org/abs/2605.26246)  
**代码**: https://github.com/ghwang-s/bridge_garden_hybrid_kd_release  
**领域**: 模型压缩 / 知识蒸馏 / LLM 训练  
**关键词**: 知识蒸馏, 暴露偏差, 硬软标签混合, Bridge-Garden 分解, 风险引导

## 一句话总结
作者发现 LLM 蒸馏里"软标签 + 硬标签线性混合"几乎总是打过纯软标签，并把原因从直觉上的"硬标签信息更少但优化更易"修正为"硬标签压低了暴露偏差"，进一步用 Bridge-Garden 分解把生成序列拆成"必须精确的桥"和"可灵活替换的花园"两类位置，从而把 mix coefficient 与上下文风险绑定起来，提出 4 种自适应混合策略并以 9.7× 训练成本优势在 7 对 teacher-student 上超越主流 on-policy / divergence-based KD。

## 研究背景与动机
**领域现状**：LLM 蒸馏主流走两条路。一条是软标签蒸馏（Hinton-style，KL/JS/反向 KL 各种 divergence），让学生匹配教师在整个词表上的下一 token 分布，被认为信息量"严格更大"。另一条是硬标签蒸馏（Kim & Rush，Alpaca/Vicuna 风格的 SFT），只用教师采样出的 token 做交叉熵，常见于黑盒教师或大词表场景。

**现有痛点**：直觉上软标签压倒硬标签，但社区里早就有人在配方里写 `loss = λ * KL + (1-λ) * CE`，并报告这样的混合损失比纯 KL 强。问题是没人讲清楚"混合到底好在哪里"——是因为优化更容易？还是更稳？还是因为某种 regularization？没人讲清楚就意味着没法系统设计混合策略。

**核心矛盾**：作者通过 Qwen2.5-7B→3B 实验观察到一个反直觉现象：混合损失的训练阶段对教师的 imitation error（forward KL on teacher prefix）反而比纯软标签更高，但 benchmark 精度却更高。这说明"训练拟合更好 → 推理更好"的朴素链路在自回归生成中根本不成立。

**本文目标**：(1) 解释混合 KD 收益的真正来源；(2) 把这个解释做成可计算的位置级风险量；(3) 据此设计能跑赢现有 SOTA 的自适应混合算法，并把训练成本压下去。

**切入角度**：作者从经典的 exposure bias（Bengio 2015）出发，把推理总误差分解成 training-fit error + 偏移项，发现硬标签恰恰是在压偏移项；进一步用算法稳定性视角（Bousquet-Elisseeff）定义"单步覆盖策略"的灵敏度 $\kappa(a\mid s)$，把序列拆成 Bridge / Garden。

**核心 idea**：硬标签是"在桥上把所有概率压到唯一安全 token 上避免崩塌"，软标签是"在花园里保持教师的多样性"；混合损失之所以好，是因为它能同时拿到这两份福利，因此最优的混合系数 $\lambda$ 必须随上下文风险动态调整。

## 方法详解

### 整体框架
论文的"方法"分两个层面：一层是理论分解（解释为什么 hybrid 必赢），一层是 4 种实操算法（解释 $\lambda$ 怎么定）。理论层先把推理 imitation error 写成训练 error + exposure bias 残差，再对 exposure bias 做"单步覆盖策略 + 风险灵敏度 $\kappa$ + Bridge/Garden 二分"的上界拆解；算法层把这套理论落成 4 种 hybrid 损失：confidence 加权、entropy 加权、curriculum 调度，以及一个把 risk 嵌入 hard 项的 Risk-Guided Hybrid。整个 pipeline 不引入新模块，只改训练损失的形态，因此与任意 divergence 选择正交。

### 关键设计

1. **Exposure-bias 分解与硬标签的真正贡献**:

    - 功能：把"混合好在哪"从经验观察上升为一句可证的等式。
    - 核心思路：记教师前缀分布 $d_T$、学生前缀分布 $d_\theta$，则学生推理总误差 $\mathcal{L}_{d_\theta}(\pi_\theta) = \mathcal{L}_{d_T}(\pi_\theta) + (\mathcal{L}_{d_\theta}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))$，第二项正是 exposure bias。实验 Fig.2(b,c) 显示加入硬标签会让训练项 $\mathcal{L}_{d_T}$ 上升，但 exposure bias 下降幅度更大，净效果是推理误差下降——彻底否决"硬标签 = 优化更易"的旧解释。
    - 设计动机：把"加 hard 有用"从 trick 提升为可计量的因果机制。一旦认定真正在受惠的是 exposure bias，后续就有动力去找一个能区分"哪些位置 exposure bias 敏感"的局部量，引出 Bridge/Garden 分解。

2. **Bridge-Garden 分解与风险灵敏度 $\kappa$**:

    - 功能：把序列里的每一步标上一个"偏离一格会有多惨"的局部风险标签。
    - 核心思路：构造单步覆盖策略 $\pi^{(s,a)}$（在 prefix $s$ 强制输出 $a$，其余处与教师相同），定义 $\kappa(a\mid s) = (\mathcal{L}_{d^{(s,a)}}(\pi_\theta) - \mathcal{L}_{d_T}(\pi_\theta))/d_T(s)$，即"一次性把当前 token 改成 $a$ 给整段序列带来的损失增量"。聚合 $\kappa(s) = \sum_{a\in\mathcal{V}}\kappa(a\mid s)$ 后，把 $\kappa(s)$ 大的位置叫 Bridge（一动就崩，如数学推导里的 + / −）、小的位置叫 Garden（如开放对话里替换近义词）。论文证明 exposure bias 上界可写成 Bridge 段 $F_\mathcal{B}$ 与 Garden 段 $F_\mathcal{G}$ 之和，且 $F_\mathcal{B}$ 由高风险 token 上的 $\lvert\Delta\pi_\theta(a\mid s)\rvert$ 主导，$F_\mathcal{G}$ 退化为普通分布距离。
    - 设计动机：这一步把"硬标签好在哪 / 软标签好在哪"形式化——Thm.4.4 证明 $\pi_{\text{hard}}$ 在 $F_\mathcal{B}$ 上更小但 $F_\mathcal{G}$ 上更大，$\pi_{\text{soft}}$ 反之，因此存在严格优于二者的 hybrid 解。Bridge/Garden 不只是个比喻，它是写在 $\kappa$ 上的硬阈值。

3. **四种 Bridge-Garden 自适应混合损失**:

    - 功能：把理论里的"$\kappa$ 大就多用 hard"翻译成训练循环里能跑的损失。
    - 核心思路：基础形式仍是 $\ell_{\text{hyb}}(s;\theta) = \lambda\,\ell_{\text{soft}}(s;\theta) + (1-\lambda)\,\ell_{\text{hard}}(s;\theta)$，但 $\lambda$ 不再是常数。(a) Confidence 加权：用教师 top-1 概率近似 $\kappa$，$\lambda_{\text{conf}}(s) = 1 - \max_a \pi_T(a\mid s)$，教师越自信越当作 Bridge；(b) Entropy 加权：$\lambda_{\text{ent}}(s) = H_T(s)/\log|\mathcal{V}|$，把全局不确定性映成 Garden 权重；(c) Curriculum 调度：$\lambda(t) = \min(t/T,1)\cdot\lambda_{\max}$，先压 Bridges 后放 Gardens；(d) Risk-Guided Hybrid：把硬损失改写成 $\ell'_{\text{hard}} = \ell_{\text{hard}} + (\alpha/4)\Delta_\theta(s,a^\*)^2$，其中 $\Delta_\theta(s,a^\*) = \log\sum_{a'} \exp(f_\theta(a'\mid[s,a^\*]) - f_\theta(a^\*\mid s))$，相当于把 $\kappa$ 解释成 reward，让学生在 Bridges 处自动锐化。
    - 设计动机：四个变种从不同角度近似 $\kappa$，互为消融。Confidence/Entropy 是 token 级局部信号，几乎零开销；Curriculum 是时间级，绕开 token 级估计噪声；Risk-Guided 把理论里"override 策略"映成 reward 项，只需 1 个 log-sum-exp，与标准 soft KD 同阶成本，但首次把 Bridge/Garden 的 $\kappa$ 写进 loss。

### 损失函数 / 训练策略
全部基于教师 prefix（off-policy）训练，主 divergence 默认 forward KL，但实验中也跑了 reverse KL/JS/Total Variation 等 7 种 divergence，证明 hybrid 收益与 divergence 选择正交。Risk-Guided 用 $\alpha = 0.1$，全程不调；其余无新超参。和 GKD/MiniLLM 等 on-policy 方法相比，本文完全不在学生分布上采样，所以训练成本从 on-policy 的多次 rollout 降到一次教师 forward，作者实测端到端便宜 9.7×。

## 实验关键数据

### 主实验
7 对 teacher-student 覆盖 Qwen / Llama / Gemma / DeepSeek-Coder，benchmark 含 BBH / MMLU / ARC-C / TheoremQA（reasoning）与 GSM8K / MATH / Gaokao23（math）与 HumanEval / MBPP（code）。

| 配置 | BBH | MMLU | ARC-C | ThmQA | Avg |
|------|-----|------|-------|-------|-----|
| Student (No Distill) | 22.34 | 64.61 | 78.40 | 12.22 | 44.39 |
| Hard KD (Kim & Rush) | 41.52 | 65.76 | 78.75 | 23.75 | 52.45 |
| Soft KD (Hinton) | 41.65 | 64.45 | 78.33 | 23.02 | 51.87 |
| Static Hybrid (本文) | 42.61 | 66.89 | 79.30 | 25.45 | 53.56 |
| Confidence-weighted | 44.07 | 67.50 | 80.77 | 22.78 | 53.78 |
| Entropy-weighted | 46.83 | 67.06 | 79.81 | 23.65 | 54.34 |
| Curriculum Schedule | 44.39 | 67.32 | 80.51 | 22.13 | 53.59 |
| **Risk-Guided Hybrid** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

（Qwen2.5-7B → 3B，标准差略去，详见原文 Table 1。）

### 消融实验
和近期 7 种 divergence-based SOTA 对比，在 Qwen2.5-7B → 0.5B / 3B 两个 capacity gap 下都领先；下表抽取 Qwen2.5-7B → 3B。

| Divergence | BBH | MMLU | ARC-C | ThmQA | Avg |
|------------|-----|------|-------|-------|-----|
| Reverse KL (Gu 2024) | 44.07 | 65.67 | 77.68 | 24.20 | 52.90 |
| Total Variation (Wen 2023) | 40.50 | 64.52 | 78.11 | 22.83 | 51.49 |
| JS (Agarwal 2024) | 45.50 | 64.68 | 78.85 | 22.27 | 52.83 |
| Adaptive KL (Wu 2025) | 44.71 | 64.69 | 79.23 | 22.25 | 52.72 |
| Skew FKL (Ko 2024) | 41.39 | 64.67 | 77.75 | 23.77 | 51.89 |
| Skew RKL (Ko 2024) | 41.22 | 63.95 | 76.91 | 23.67 | 51.44 |
| $\alpha$-$\beta$ div (Wang 2025) | 45.12 | 64.95 | 79.81 | 22.94 | 53.21 |
| **HybKD (Ours)** | **46.53** | **69.05** | **81.23** | 23.82 | **55.16** |

### 关键发现
- Fig.2(b,c) 是论文最有冲击的图：训练阶段 hybrid 的 forward KL 比 soft 更高（拟合更差），但推理阶段 exposure bias 显著更低，正好对应理论分解 $\mathcal{L}_{d_\theta} = \mathcal{L}_{d_T} + \text{exposure}$。
- Risk-Guided Hybrid 在 7 个 teacher-student 对里几乎全部领先，且与各种 divergence 正交（Fig.3 把它叠到 RKL / JS / $\alpha$-$\beta$ 上都涨点），说明 Bridge-Garden 信号是普适的，不依赖具体 divergence。
- Qwen2.5-Math-7B → 1.5B 在 GSM8K 从 65.75 提到 71.65、MATH 从 50.44 提到 51.37、Gaokao23 从 41.04 提到 44.10；DeepSeek-Coder-6.7B → 1.3B 在 HumanEval+ 36.59、MBPP 63.12 双双登顶，说明 Bridge 概念在"必须精确"的推理 / 代码任务上收益最大。
- 训练成本是 on-policy KD（GKD 等）的 1/9.7，因为整个框架完全在教师 prefix 上跑，无需学生自采样，工业部署友好。
- App.F 在合成域里能精确算 token 级 $\kappa$，直接验证了 Bridge/Garden 分解；这是少有的在 LLM 蒸馏里能做"理论与实验闭环"的工作。

## 亮点与洞察
- "训练 fit 变差但推理变好"这种反直觉现象的解释非常干净——把推理误差等式分两段，硬标签压的是被多年来被忽视的 exposure bias 那段，瞬间把 hybrid 从 trick 变成 first-class citizen。
- Bridge/Garden 的 $\kappa$ 既能解释为"算法稳定性意义下的扰动灵敏度"，又能解释为"reward magnitude"，把 KD、稳定性理论、RL 三个圈子的语言对齐到一起，思想迁移到 SFT / DPO 同样适用（例如 DPO 损失也可以按 $\kappa$ 加权）。
- Risk-Guided Hybrid 把 $\kappa$ 写成 $\log\sum\exp$ 单步项，几乎免费，工程上极易加挂；这种"理论 motivate 出极轻量 loss tweak"是工业团队最爱的范式。
- 9.7× 训练成本优势不靠工程优化而是范式选择（off-policy + 教师 prefix），对一个本来就在做大模型的实验室来说意义巨大——意味着用同样预算可以做接近 10 倍 ablation。

## 局限与展望
- 理论上的 Bridge/Garden 二分依赖 $\kappa(s)$，但 $\kappa$ 在 LLM 上无法精确算，4 种近似都只是 proxy；confidence / entropy 在某些任务（如指令跟随）可能与真实风险脱节，是后续可挖的方向。
- 实验最大学生只到 3B，最大教师 8B；当 capacity gap 极大（如 70B → 0.5B）或学生 / 教师架构差异显著（如 Mamba ← Transformer）时，Bridge/Garden 是否还保持是开放问题，附录 F 只做了部分扫描。
- 推理 / 代码任务上收益最大，开放生成（如 dialog summarization）收益相对小；这与"Bridge 浓度低于推理任务"一致，但论文未在开放生成上做更深入的 Bridge 比例统计。
- 所有训练仍是教师 prefix 上的 off-policy，没回答"on-policy 信号 + Bridge/Garden 加权"是否还能再叠一层增益；与 GKD 等的正交化是顺手可做的下一步。

## 相关工作与启发
- **vs Hinton et al. (2015) soft KD**：保留 KL 形式，但用 hybrid 取代纯 KL；首次系统解释 hybrid 为何普遍占优。
- **vs Kim & Rush (2016) hard KD**：保留硬标签思想，但把它从"教师采样的 noise reduction"重新定位为"exposure bias 抑制器"。
- **vs GKD / MiniLLM (Gu et al. 2024; Agarwal et al. 2024) on-policy KD**：他们靠在学生分布上 rollout 去匹配 exposure bias，本文用 off-policy + Bridge/Garden 加权达到相同目标，成本下降 9.7×。
- **vs $\alpha$-$\beta$ divergence、Skew KL、Adaptive KL**：那些工作改 divergence 形态，本文改"loss mix coefficient"，二者正交并可叠加（Fig.3 验证）。
- **vs Cundy & Ermon (2023) reward-shaping for LM**：Risk-Guided Hybrid 把他们的"reward consistency"思想搬到 KD，把硬标签当作 reward，软标签当作 entropy regularizer。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)
- [\[ICML 2026\] DSL-Topic: Improving Topic Modeling by Distilling Soft Labels from Language Models](dsl-topic_improving_topic_modeling_by_distilling_soft_labelsfrom_language_models.md)
- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[ICML 2026\] Demystifying When Pruning Works via Representation Hierarchies](demystifying_when_pruning_works_via_representation_hierarchies.md)
- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](../../ACL2026/model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)

</div>

<!-- RELATED:END -->
