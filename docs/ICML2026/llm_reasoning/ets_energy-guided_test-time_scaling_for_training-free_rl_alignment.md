---
title: >-
  [论文解读] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment
description: >-
  [ICML 2026][LLM推理][KL 正则 RL 闭式解] ETS 直接从 KL 正则化 RLHF 目标的**闭式最优解**采样，把它写成「参考策略 × 指数 reward 的条件期望（能量项）」，再用 Monte Carlo + 自归一化重要性采样在测试时近似这个能量项…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "KL 正则 RL 闭式解"
  - "能量重加权"
  - "Monte Carlo"
  - "重要性采样"
  - "ARM/DLM 通用"
---

# ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment

**会议**: ICML 2026  
**arXiv**: [2601.21484](https://arxiv.org/abs/2601.21484)  
**代码**: https://github.com/sheriyuo/ETS (有)  
**领域**: LLM 推理 / 测试时扩展 / 训练免对齐  
**关键词**: KL 正则 RL 闭式解、能量重加权、Monte Carlo、重要性采样、ARM/DLM 通用

## 一句话总结
ETS 直接从 KL 正则化 RLHF 目标的**闭式最优解**采样，把它写成「参考策略 × 指数 reward 的条件期望（能量项）」，再用 Monte Carlo + 自归一化重要性采样在测试时近似这个能量项，从而**不训练**就达到甚至超过经过 RL 后训练的策略，并通过 lightweight proposal + Fast-dLLM 把延迟控制在可用范围。

## 研究背景与动机

**领域现状**：RLHF / DPO / GRPO 已成 LLM 后训练标配，把模型对齐到「reward 高 + 不偏离参考策略 $p_{\text{ref}}$」。理论上这个 KL 正则目标早有 Rafailov 等给出的闭式解 $p(\boldsymbol{x}_0\mid\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y})\exp(r/\lambda)$，但现有 RL pipeline 仍用梯度迭代去逼近。

**现有痛点**：训练版 RL 需要昂贵 reward model + 大量人类偏好、训练不稳定、超参敏感、reward 一改就得重训；且 Power Sampling / Quest 之类 MH 采样虽免训练却串行慢。

**核心矛盾**：「最优分布已知闭式」与「实际仍靠迭代训练逼近」之间存在巨大鸿沟 — 如果能在测试时**直接采样**那个闭式分布，所有训练问题就都消失了。

**本文目标**：(1) 给出统一 MLM 框架（含 ARM 和扩散语言模型 DLM）下闭式解的**反向 Markov 转移核**形式；(2) 设计 Monte Carlo 估计 + 加速器使其可用；(3) 给出收敛速率与误差累计的理论保证。

**切入角度**：把生成过程视作从 $\boldsymbol{x}_T\to\boldsymbol{x}_0$ 的反向 Markov 链（ARM 是固定左→右、DLM 是动态 unmask），在该框架下推导最优反向转移核会自然分解为「参考转移 × 能量项」。

**核心 idea**：在每个 guidance step 用候选采样 + 能量重加权 + 多项式抽样实现「沿反向链一步步走向最优分布」，避免任何参数更新。

## 方法详解

### 整体框架
ETS 是一个 inference-time 的 search 算法。给定 query $\boldsymbol{y}$、初始 mask 序列 $\boldsymbol{x}_T$、guidance 步数 $I$ 与候选数 $M$，从 $i=I$ 反推到 $i=1$：(1) 用 $p_{\text{ref}}$ 从 $\boldsymbol{x}_{t_i}$ 采样 $M$ 个候选 $\boldsymbol{x}_{t_{i-1}}(m)$；(2) 对每个候选估计能量 $\widehat{\mathcal E}$；(3) 自归一化得到权重 $w_m\propto \widehat{\mathcal E}$；(4) 按多项式抽样选出一个候选作为下一步状态。最终 $\boldsymbol{x}_0$ 就是从近似最优分布 $p(\boldsymbol{x}_0\mid\boldsymbol{y})$ 中抽出的样本。

注意当 $I=1,\lambda\to 0$ 时算法退化为 Best-of-N，所以 ETS 严格泛化了 BoN，并通过 $I$ 提供「分多步对齐」的更细旋钮。

### 关键设计

1. **能量重加权反向转移核 (Proposition 2)**:

    - 功能：把 KL 正则 RLHF 闭式解从最终 token 序列分布转化为反向 Markov 链的逐步可采样形式。
    - 核心思路：对任意 $s<t$ 推出 $p(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\propto p_{\text{ref}}(\boldsymbol{x}_s\mid\boldsymbol{x}_t,\boldsymbol{y})\cdot \mathbb E_{p_{\text{ref}}(\boldsymbol{x}_0\mid\boldsymbol{y},\boldsymbol{x}_s)}\!\big[\exp(r/\lambda)\big]$。后一项即「能量」 $\mathcal{E}(\boldsymbol{y},\boldsymbol{x}_s)$ — 衡量从当前 partial state $\boldsymbol{x}_s$ 出发未来能拿到多高奖励。
    - 设计动机：把全局最优 $p(\boldsymbol{x}_0\mid\boldsymbol{y})$ 这个不可直接采样的对象，分解成「参考模型可直接采」+「条件期望可 Monte Carlo」两块，每块都可操作；同时统一覆盖 ARM 与 DLM。

2. **能量项的 Monte Carlo 估计 + 自归一化 IS (Algorithm 1)**:

    - 功能：对每个候选 $\boldsymbol{x}_{t_{i-1}}(m)$ 估出 $\widehat{\mathcal E}$，再在 $M$ 个候选间做相对加权。
    - 核心思路：从 $\boldsymbol{x}_s$ 出发用 $p_{\text{ref}}$ rollout 出 $K$ 条完整 $\boldsymbol{x}_0(k)$，能量估计为 $\widehat{\mathcal E}(\boldsymbol{y},\boldsymbol{x}_s)=\frac{1}{K}\sum_k \exp(r(\boldsymbol{y},\boldsymbol{x}_0(k))/\lambda)$。全局归一化常数不可解，但 batch 内自归一化恰好给出「相对最优概率」，从中按 multinomial 抽样等价于从最优分布的有限候选受限版中抽样。Proposition 3 证明总变差距离上界 $\widetilde{\mathcal O}(I/\sqrt M + I\epsilon)$，$\epsilon$ 为能量估计误差。
    - 设计动机：直接近似 partition function 需要全序列空间求和不可行；自归一化把绝对概率问题转化为相对采样问题，是从能量基模型 / 扩散指导继承的稳定 trick。

3. **重要性采样加速 (Algorithm 2, ETS-IS)**:

    - 功能：用更便宜的 proposal 模型 $p_{\text{small}}$ 替换 $p_{\text{ref}}$ 完成 rollout，大幅压低 Monte Carlo 估计延迟。
    - 核心思路：基于 $\mathcal E(\boldsymbol{y},\boldsymbol{x}_s)=\mathbb E_{p_{\text{small}}}[\frac{p_{\text{ref}}}{p_{\text{small}}}\exp(r/\lambda)]$，得到无偏 IS 估计；对 ARM 用同 tokenizer 的 Qwen3 小模型，对 DLM 没小模型可用就用 Fast-dLLM (KV cache + 并行解码) 作为 $p_{\text{small}}$。Theorem 1 证明：选 $K$ 足够大时 IS 版本保持 $\widetilde{\mathcal O}(I/\sqrt M + I/\sqrt K)$ 同阶收敛。
    - 设计动机：能量估计是延迟瓶颈，单纯靠 $p_{\text{ref}}$ 跑 $M\times K$ 条 rollout 极贵；IS 给「便宜采样 + 正确无偏」打通了一条工程可行的捷径。

### 损失函数 / 训练策略
**完全不训练**。reward 不靠 reward model，而用 self-consistency proxy：对每个候选采 $K$ 条 completion，对最终答案做 majority vote，匹配多数票即 reward=1，否则 0。这种 proxy 在文中实验里给出最接近 ground-truth 的奖励分布，是同类 uncertainty 度量中表现最好的。

## 实验关键数据

### 主实验
在 MATH500 / GSM8K / HumanEval / GPQA-Diamond 上 pass@1（单次最终回答）评测；ARM 用 Qwen3-1.7B/8B（non-thinking），DLM 用 LLaDA-8B-Instruct。baseline 包括 Base、Beam Search、Best-of-N、Power Sampling、以及 Verl 训出来的 RL 与 LLaDA-1.5。

| 模型 | 数据集 | Base | Best-of-N | Power Sampling | RL 训练版 | **ETS / ETS-IS** |
|---|---|---|---|---|---|---|
| Qwen3-8B (ARM) | MATH500 | baseline | 提升 | 提升但慢 | 强 baseline | **超过 RL 版** |
| Qwen3-8B (ARM) | GPQA-Diamond | baseline | 中 | 中 | 强 | **最优** |
| LLaDA-8B (DLM) | HumanEval | baseline | 中 | 中 | LLaDA-1.5 | **超过 LLaDA-1.5** |
| Qwen3-1.7B (ARM) | GSM8K | baseline | 中 | 慢 | 强 | **最优**（不需 IS） |

（具体数值随设置变化，但总趋势：ETS 在所有四个 benchmark 上都稳定优于 TTS baselines，且常常胜过专门 RL 后训练的同尺寸模型。）

### 消融实验

| 配置 | 关键效果 | 说明 |
|---|---|---|
| Full ETS ($I>1$) | 最优 | guidance 多步分摊误差 |
| $I=1,\lambda\to 0$ | 退化为 Best-of-N | 证明 ETS 严格泛化 BoN |
| 去掉 IS（纯 $p_{\text{ref}}$） | 同精度但延迟 ↑↑ | IS 是延迟救星 |
| reward 改为 logits 置信度 / entropy | 精度下降 | self-consistency reward 最接近 oracle |
| 增大 $M$ | 精度↑、延迟↑ | 符合 $1/\sqrt M$ 收敛 |

### 关键发现
- 「训练免对齐」首次在主流推理 benchmark 上做到了**与 RL 后训练同档甚至更好**，说明现有 RL 训练浪费了大量计算去做闭式解能直接采样的事。
- $I=1$ 不一定最差也不一定最好 — 误差并非线性累加，guidance 步数与 $\lambda$ 联合决定最优工作点（Remark 2）。
- 用对齐良好的 Qwen3 小模型做 IS proposal 时，效率/精度 trade-off 最优；speculative decoding（EAGLE-3）因不兼容 batch 反而吃亏。

## 亮点与洞察
- **方法论亮点**：把「RLHF 的闭式解」这件已知但被忽视的事实，扩展成可在 ARM/DLM 通用的反向链转移核，并配上完整误差分析 — 这是把 score-based / diffusion guidance 的思路平滑迁移到离散 MLM 的范本。
- **理论闭环**：Proposition 2（转移核） → Proposition 3（误差） → Theorem 1（含 IS 加速误差），层层闭合，且和扩散模型中的误差累加结果（$\propto I$）类比清晰。
- **可迁移 trick**：「自归一化 + lightweight proposal IS」可直接搬到其他 inference-time alignment 任务（对话偏好、agent reward shaping、tool selection 重排），且天然兼容 batch 并行。

## 局限与展望
- proxy reward 用 self-consistency，本质要求「多数答案 = 正确答案」，在创造性 / 开放问答 / 多解任务中会失效。
- 误差上界假设各步 guidance error $\epsilon$ 一致，但实际不同 $\boldsymbol{x}_t$ 误差差异大；更精细的随状态变化的上界是开放问题。
- 对 DLM 的加速依赖 Fast-dLLM 工程实现，未来若有真正对齐良好的小 DLM 可显著进一步提速。
- 与 speculative decoding / 量化等加速路径的真正打通仍未完成。

## 相关工作与启发
- **vs Power Sampling / Quest**：都瞄准从 RL 最优分布采样，但 MH 算法天然串行；ETS 借助 batched MC + IS，天然并行，速度高一档。
- **vs Dang 2025 / Uehara 2024**：他们在连续时间扩散模型下推类似公式，本文适用于离散 MLM 且统一 ARM/DLM。
- **vs Best-of-N / Beam Search**：BoN 是 $I=1$ 特例；Beam Search 是确定性最大化，未必匹配最优概率分布。ETS 兼具理论保证与实证收益。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把闭式 RL 最优解直接「采」出来，免训练匹敌 RL，方法范式上是新的
- 实验充分度: ⭐⭐⭐⭐ 覆盖 ARM/DLM × 数学/代码/科学共 4 benchmark + 多个加速消融
- 写作质量: ⭐⭐⭐⭐ 推导工整，从闭式解一路到 IS 加速逻辑顺畅；记号略多但可读
- 价值: ⭐⭐⭐⭐⭐ 给出了「测试时对齐」的可行实现，工程上能省下整套 RLHF 流程，应用前景大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)

</div>

<!-- RELATED:END -->
