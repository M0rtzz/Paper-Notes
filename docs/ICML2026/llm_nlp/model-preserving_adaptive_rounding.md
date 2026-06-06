---
title: >-
  [论文解读] YAQA: 端到端 KL 最小化的 LLM 自适应权重量化
description: >-
  [ICML 2026][LLM/NLP][量化] YAQA 把 LLM 权重量化的代理目标从「逐层激活误差」换成「端到端模型输出 KL 散度」，用 Kronecker 分解的 Hessian 草图给出第一个端到端误差界，相对 GPTQ/LDLQ 把 KL 再降约 30%，甚至比量化感知训练（QAT）更准…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "量化"
  - "自适应舍入"
  - "端到端 KL"
  - "Hessian 草图"
  - "Kronecker 分解"
---

# YAQA: 端到端 KL 最小化的 LLM 自适应权重量化

**会议**: ICML 2026  
**arXiv**: [2505.22988](https://arxiv.org/abs/2505.22988)  
**代码**: 暂未公布  
**领域**: 模型压缩 / LLM 量化  
**关键词**: 量化, 自适应舍入, 端到端 KL, Hessian 草图, Kronecker 分解

## 一句话总结
YAQA 把 LLM 权重量化的代理目标从「逐层激活误差」换成「端到端模型输出 KL 散度」，用 Kronecker 分解的 Hessian 草图给出第一个端到端误差界，相对 GPTQ/LDLQ 把 KL 再降约 30%，甚至比量化感知训练（QAT）更准，且推理速度不变。

## 研究背景与动机

**领域现状**：LLM 量化分两条路线——QAT 通过修改训练流程学习低精度表示，质量好但成本巨大；PTQ 通过事后舍入把全精度权重映射到一组离散码本，代表方法是 GPTQ/LDLQ，因便宜而流行。GPTQ 用「当前层激活误差」的 Hessian $H_1 = \mathbb{E}[x^\top x]$ 当作端到端误差的代理。

**现有痛点**：$H_1$ 只看当前层的输入分布，完全忽略后续层会如何放大或抵消这层的舍入误差。结果是「逐层最优」不等于「全模型最优」，KL 散度往往被无谓地拉高。GuidedQuant/SqueezeLLM 改用经验 Fisher 的块对角近似，但它来自交叉熵任务损失，而非真正的 KL Hessian，且块结构是临时拼凑，没有理论保证——经验上加大块数效果反而不一致。

**核心矛盾**：要直接对 $\nabla^2 L(W^*) \in \mathbb{R}^{mn \times mn}$（KL 关于一层权重的真 Hessian）做自适应舍入，规模就爆炸；要保留 tractable 结构，又得有可证的逼近质量。已有结构化近似要么没有界，要么近似得不好。

**本文目标**：找一个结构化 Hessian 草图，既能在 $O(m+n)$ 步内完成 LDLQ 风格的迭代舍入，又能用「与真 Hessian 的余弦相似度」严格控制端到端 KL。

**切入角度**：作者引入「结构性幂零度」（SND）这个组合量来刻画 LDLQ 收敛步数。证明对 Kronecker 积 $L_O \otimes L_I$ 有 $\mathrm{snd}(L_O \otimes L_I) = \mathrm{snd}(L_O) + \mathrm{snd}(L_I) \le m+n-1$，这就把「可 tractable 计算」直接落到 Kronecker 分解上。

**核心 idea**：用 Kronecker 分解 $\tilde{H} = H_O \otimes H_I$ 作为 $\nabla^2 L(W^*)$ 的近似，通过在真 Fisher 上做幂迭代得到「近最优」的 $H_O, H_I$；舍入算法在 LDLQ 上加一个对称的输出端反馈分量，整体 $\approx 2\times$ LDLQ 时间却把 KL 显著拉低。

## 方法详解

### 整体框架

YAQA 由两块组成：(1) **舍入算法**——把 LDLQ 推广到 Kronecker 分解的 Hessian 草图，得到 $W = Q(W^* + L_O'^{\top} \Delta L_I' + L_O'^{\top} \Delta + \Delta L_I')$，其中 $L_O', L_I'$ 是 $H_O, H_I$ 的 LDL 三角因子去单位阵，$\Delta = W^* - W$；(2) **Hessian 草图**——用幂迭代构造 $H_O, H_I$，使 $\tilde{H}$ 与真 Hessian 在 Frobenius 内积下尽可能对齐。整套流程对每个线性层独立做一次，不改变推理结构，因此量化模型的推理速度由码本（如 E8P）决定，与 YAQA 无关。

### 关键设计

1. **结构性幂零度 SND + Kronecker Hessian 让 LDLQ 端到端化**:

    - 功能：把「能 tractable 跑 LDLQ」的 Hessian 结构刻画出来，并选出一个最优的。
    - 核心思路：定义 $\mathrm{snd}(L)$ 为与 $L - I$ 同支撑的二元幂零矩阵的幂零度，证明 LDLQ 迭代在 $\le \mathrm{snd}(L)$ 步收敛。对 Kronecker 积 $L_O \otimes L_I$，$\mathrm{snd}$ 是两侧之和，于是 $\tilde{H} = H_O \otimes H_I$ 既允许「输入端 + 输出端」对称反馈，又只需 $O(m+n)$ 步小矩阵乘法。GuidedQuant 在同框架下等价于在块对角近似上跑 LDLQ，没有输出端反馈，难怪超过 4 块就饱和。
    - 设计动机：以前的 PTQ 算法只能在「逐层 $H_1$（无输出反馈）」或「QAT（昂贵但全局）」之间选；Kronecker 是第一个把「端到端」和「tractable」同时拿下的结构。

2. **端到端 KL 误差界 + 余弦相似度作为优化目标**:

    - 功能：把模型 KL 上界写成 Hessian 草图与真 Hessian 的几何关系，告诉算法该怎么挑 $H_O, H_I$。
    - 核心思路：定理 3.4 证 $\mathrm{vec}(\Delta) H \, \mathrm{vec}(\Delta)^\top \le \|H\|_F (\|\Delta\|_F^2 \sqrt{2 - 2c} + \text{(incoherence/trace term)})$，其中 $c = \langle H, H_O \otimes H_I \rangle / (\|H\|_F \|H_O\|_F \|H_I\|_F)$ 是余弦相似度。意思是：草图越「方向上」贴合真 Hessian，端到端 KL 上界越紧；同时 $H_O, H_I$ 要低 incoherence + 低秩。
    - 设计动机：这是任何量化算法第一次拿到端到端误差界，把「Hessian 草图选择」从经验玄学升级为可优化目标，余弦相似度直接告诉我们该用幂迭代去逼近。

3. **两种可扩展的幂迭代 Hessian 草图**:

    - 功能：在 LLM 规模上把 $H_O, H_I$ 算出来——不能直接 $mn \times mn$ 操作真 Hessian。
    - 核心思路：Sketch A 假设序列内 token 独立，把 $H \approx \mathbb{E}[x^\top x \otimes (\nabla_y \ell)^\top (\nabla_y \ell)]$，从 $(H_I)_0 = H_1, (H_O)_0 = I$ 出发幂迭代 3 步左右就收敛，10B 模型约 20 GPU-hour；Sketch B 直接在真 Fisher 上跑一轮幂迭代（从 $I, I$ 出发），需要按 sequence 计算梯度但只跑一遍数据，10B 模型约 30 GPU-hour。两者都用 modified backward pass 做分布式幂迭代，跟 Shampoo 的预条件类似但用真 Fisher 而非经验 Fisher。
    - 设计动机：直接 Monte-Carlo 估真 Hessian 方差太大；Sketch A 用 bias 换 variance，适合数据少；Sketch B 用一轮幂迭代换更高的方差容忍度，数据多时质量更好。这给出了「便宜」和「最优」两档选择。

### 损失函数 / 训练策略

YAQA 是 PTQ，无显式损失。优化目标隐式地是 $\mathrm{tr}(\Delta^\top H_O \Delta H_I)$，即 Kronecker 草图下的代理损失。舍入算法是定点迭代（Equation 5/6），用 scalar 或 vector quantizer 都行；与 QuIP# 的 randomized Hadamard transform 互补——前者让 $W$ 近高斯并降 incoherence，后者负责精确算 Hessian。

## 实验关键数据

### 主实验：LLM 量化质量

| 模型 / 设定 | 方法 | KL ↓ (vs 全精度) | 下游基准 (acc%) ↑ |
|------------|------|-------|-----------------|
| Llama 3.1 8B Inst, W2 | LDLQ (GPTQ) | 基线 | 基线 |
| Llama 3.1 8B Inst, W2 | GuidedQuant | 略优于 LDLQ | 略优 |
| Llama 3.1 8B Inst, W2 | **YAQA Sketch A** | $\approx -30\%$ vs LDLQ | 显著领先 |
| Llama 3.1 8B Inst, W2 | **YAQA Sketch B** | 最低 | 最高 |
| Llama 3.1 8B Inst, W2 | QAT | 高于 YAQA | 低于 YAQA |

（数字按论文摘要与图表汇总；Sketch B 在多个 chat/reasoning 任务上都建立了新的 PTQ SOTA。）

### 消融实验

| 设定 | KL ↓ | 说明 |
|------|------|------|
| LDLQ ($H_O = I, H_I = H_1$) | 基线 | YAQA 退化情形 |
| Sketch A，1 步幂迭代 | 中等 | 初始就用 $H_1$ 起步 |
| Sketch A，3 步幂迭代 | 优秀 | 经验收敛步数 |
| Sketch B，2K 序列 | 优秀 | 1 GPU-hour 也能 SOTA |
| Sketch B，64K 序列 | 最佳 | 30 GPU-hour |
| GuidedQuant，>4 块 | 不再改进 | 缺输出端反馈 |

### 关键发现

- 经验上 $H_O$ 近似低秩（图 1），刚好对应理论里「low rank 时 YAQA 界严格优于 LDLQ」的条件，解释了为什么 Kronecker 草图能赢。
- Sketch B 只跑一轮幂迭代就比 Sketch A 好，说明真 Fisher 的方差与 sequence 级估计能压住——「真的不是」需要严格收敛幂迭代。
- YAQA 用极少数据（2K 序列、1 GPU-hour）就能拿到 SOTA，对 PTQ 的实用性是重要 selling point。
- KL 比 QAT 还低这一结果反直觉但符合理论：QAT 是首阶下降，可能局部最优；YAQA 是「在 Hessian 球内一次性最优舍入」，避开了 QAT 的优化困难。

## 亮点与洞察

- **第一个端到端 KL 上界**：把「Hessian 草图选择」变成「最大化余弦相似度 + 控制 incoherence/rank」的明确数学问题，避免了过往经验式拼凑结构。
- **SND 框架统一了已有方法**：GPTQ、LDLQ、GuidedQuant 都能装进同一个 SND/Kronecker 视角，立刻看清楚谁有输出端反馈、谁没有，理论指导工程选型。
- **Kronecker + 幂迭代恰好兼顾 tractable 与最优**：低 SND 决定速度，余弦相似度决定质量，幂迭代是 Frobenius 范数下最优 Kronecker 逼近的经典工具，三者天然契合。
- **真 Fisher vs 经验 Fisher 的差异**：先前 GuidedQuant/SqueezeLLM 用经验 Fisher（任务损失），YAQA 指出对 KL 目标必须用真 Fisher，用 Monte-Carlo 采 logits，否则方向就偏。

## 局限与展望

- 仅讨论 weight-only PTQ，对 activation 量化和 KV-cache 量化的迁移没展开。
- Sketch B 的 30 GPU-hour 对 70B+ 模型仍偏重；如何用更激进的稀疏化或低秩近似把成本再降一档值得探索。
- 余弦相似度上界中还有 incoherence/trace 项，rank 完全不可控，未来若能控制 $H_O, H_I$ 的有效秩，理论界会更紧。
- 对非线性层（注意力 softmax 等）的全局 Hessian 行为还没单独分析，端到端 KL 在跨层耦合下的更细颗粒结构未深挖。

## 相关工作与启发

- **vs GPTQ / LDLQ**：等价于 YAQA 在 $H_O = I, H_I = H_1$ 的退化情形，被严格包含；理论上 $H_O$ 低秩时 YAQA 上界更紧。
- **vs GuidedQuant / SqueezeLLM**：同样想超越 $H_1$，但用经验 Fisher + 块对角近似，缺输出端反馈、无端到端界；本文用真 Fisher + Kronecker，理论+实验双胜。
- **vs QAT / DiscQuant / PV-Tuning**：QAT 路线要长时间训练，本文证明只用一次幂迭代就能比 QAT 还准，对 PTQ 路线信心很大。
- **vs Shampoo / KFAC**：Hessian 草图思想类似（Kronecker + 幂迭代），但目的是预条件优化器；YAQA 用同款工具做 PTQ 的舍入方向。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 第一个把端到端 KL 上界落到量化算法里，并用 SND/Kronecker 给出可证的算法-理论闭环。
- 实验充分度: ⭐⭐⭐⭐⭐ 横扫多个 Llama/Gemma 规模 + 多 bit 配置，跟 LDLQ、GuidedQuant、QAT 都正面交锋，并附 ablation 量化数据需求。
- 写作质量: ⭐⭐⭐⭐ 理论部分推导清晰，但 SND/Kronecker 论证密集，初读门槛偏高；附录补丁较多。
- 价值: ⭐⭐⭐⭐⭐ 对实际 LLM 部署直接有用：几乎零推理成本下把质量推到甚至超过 QAT，是 PTQ 路线的明显进步。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Universal Reasoner: 冻结 LLM 的可组合即插即用推理器](universal_reasoner_a_single_composable_plug-and-play_reasoner_for_frozen_llms.md)
- [\[ICML 2026\] "I've Seen How This Goes"：用渐进条件惊奇度刻画 LLM 与人类写作的多样性](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ACL 2026\] 当梯度相撞：多目标提示优化对 LLM 评判员的失效模式](../../ACL2026/llm_nlp/when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)
- [\[ICML 2026\] Rethinking LLM Ensembling from the Perspective of Mixture Models](rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)

</div>

<!-- RELATED:END -->
