---
title: >-
  [论文解读] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter
description: >-
  [ICML2026][模型压缩][LoRA合并] 提出 Compress-then-Merge (CtM) 管线，在合并多个 LoRA 之前先学习共享 $r$ 维子空间并将各 adapter 投影为 $r \times r$ 坐标矩阵，再在低维空间中执行合并，从而在构造层面保证输出为 rank-$r$ LoRA，避免了传统 Merge-then-Compress 方法的截断 SVD 性能损失。
tags:
  - "ICML2026"
  - "模型压缩"
  - "LoRA合并"
  - "低秩约束"
  - "共享子空间"
  - "Tucker分解"
  - "参数高效微调"
---

# Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter

**会议**: ICML2026  
**arXiv**: [2606.03723](https://arxiv.org/abs/2606.03723)  
**代码**: https://github.com/ZhengbaoHe/compress-then-merge  
**领域**: 模型压缩  
**关键词**: LoRA合并, 低秩约束, 共享子空间, Tucker分解, 参数高效微调  

## 一句话总结
提出 Compress-then-Merge (CtM) 管线，在合并多个 LoRA 之前先学习共享 $r$ 维子空间并将各 adapter 投影为 $r \times r$ 坐标矩阵，再在低维空间中执行合并，从而在构造层面保证输出为 rank-$r$ LoRA，避免了传统 Merge-then-Compress 方法的截断 SVD 性能损失。

## 研究背景与动机

**领域现状**：LoRA 已成为大模型参数高效微调的标准选择，HuggingFace 等平台上积累了大量针对不同任务的 LoRA adapter。如何将多个 LoRA 合并为一个多任务 adapter 是一个日益重要的实际需求。

**现有痛点**：当前主流方案是 Merge-then-Compress (MtC)——先在全参数空间合并多个 LoRA 得到高秩更新 $\Delta W_{\text{merged}}$，再通过截断 SVD 压缩到目标秩 $r$。但这种"先合并后压缩"的策略把秩约束当作事后处理，导致两个问题：(1) 不同任务 LoRA 的范数差异巨大，截断 SVD 按 Frobenius 范数最优但偏向大范数任务，小范数任务被系统性压制；(2) LoRA 训练中可能产生"入侵维度"——高能量但与任务精度弱相关的方向，会抢占有限的秩预算。

**核心矛盾**：合并后的 rank-$r$ 结果必须位于某个 $r$ 维子空间中，MtC 将子空间选择作为合并结果频谱的副产物，而非主动设计。一旦确定子空间，正交于它的所有分量不可恢复——因此子空间选择应是核心设计考量。

**切入角度**：近期研究表明，多任务模型的权重变化倾向于集中在低维光谱子空间中。作者据此提出：**先学共享子空间再合并**，将子空间选择从被动副产物转变为主动可控的机制。

**核心 idea**：反转 MtC 管线——先将各 LoRA 压缩投影到学习得到的共享 $r$ 维子空间，再在低维坐标空间中执行合并规则，输出天然满足 rank-$r$ 约束。

## 方法详解

### 整体框架
给定 $T$ 个同构 LoRA adapter $\{(A^{(t)}, B^{(t)})\}_{t=1}^{T}$（相同基座模型、相同注入层、相同输入秩 $r_{\text{in}}$），CtM 把传统 MtC 的"先合并后压缩"反转成"先压缩后合并"：先学一对共享的 $r$ 维正交基 $(U, V)$，把每个 adapter 投影成紧凑的 $r \times r$ 坐标矩阵，然后在这个低维坐标空间里跑标准合并规则、再提升回原始参数空间。由于子空间在合并之前就固定下来，输出天然落在 rank-$r$ 子空间内，根本不需要事后的截断 SVD。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["输入：T 个同构 LoRA adapter<br/>各任务更新 ΔW⁽ᵗ⁾ = B⁽ᵗ⁾A⁽ᵗ⁾"] --> S1
    subgraph S1["重缩放感知的共享子空间学习（设计 1）"]
        direction TB
        B["重缩放代理目标 ΔW_target<br/>先按范数归一化消偏差，再用 λ 软注回尺度"]
        C["Tucker-2 分解求共享正交基 (U, V)<br/>HOSVD 初始化 + HOOI 迭代"]
        D["Core-Space 无损加速（设计 3）<br/>薄 SVD 投影到 (Tr)×(Tr) 核心空间求解"]
        B --> C
        D -.加速 Tucker 求解.-> C
    end
    S1 -->|输出共享基 (U, V)| S2
    subgraph S2["低维坐标空间合并（设计 2）"]
        direction TB
        E["投影原始 LoRA 得坐标<br/>O⁽ᵗ⁾ = Uᵀ ΔW⁽ᵗ⁾ V（r×r）"]
        F["在 r×r 空间套用现成规则合并<br/>TIES / DARE 等"]
        G["提升回参数空间<br/>ΔW_LoRA = U · O_merged · Vᵀ"]
        E --> F --> G
    end
    S2 --> H["输出：单个 rank-r LoRA<br/>构造层面保证秩约束，免截断 SVD"]
```

### 关键设计

**1. 重缩放感知的共享子空间学习：怎么选一个不偏袒大范数任务的子空间**

CtM 的命门在于这对正交基 $U \in \mathbb{R}^{n \times r}$、$V \in \mathbb{R}^{m \times r}$ ——一旦确定，正交于它的所有分量都不可恢复，所以必须让它能均衡地重建所有任务的更新。难点是不同任务 LoRA 的范数差异巨大：直接在原始 $\Delta W^{(t)}$ 上学子空间会被大范数任务主导，小范数任务被挤出秩预算；可如果完全归一化又把尺度信息丢光了。作者的解法是构造一个重缩放代理目标 $\Delta W_{\text{target}}^{(t)} = \lambda^{(t)} \cdot \Delta W^{(t)} / \|\Delta W^{(t)}\|_F$，先归一化消除范数偏差，再用 $\lambda^{(t)} = \beta \|\Delta W^{(t)}\|_F + (1-\beta) \|\Delta W\|_{F,\text{Avg}}$ 软性地把尺度信号注回来——$\beta \in [0,1]$ 就是在"完全归一化"和"保留原始尺度"之间滑动的旋钮。把这些代理目标沿任务维堆叠成三阶张量 $\mathcal{X} \in \mathbb{R}^{n \times m \times T}$ 后，学子空间就变成了一个标准的 Tucker-2 分解问题，用 HOSVD 初始化、HOOI 迭代求解，得到的 $(U, V)$ 反映的是跨任务的共同结构而非某个任务的私货。

**2. 低维坐标空间合并：在 $r \times r$ 的紧凑空间里跑现成的合并规则**

子空间定好后，合并就退化成一件简单的事。注意这里用的是**原始**（非重缩放）LoRA：对每个任务算坐标 $O^{(t)} = U^\top \Delta W^{(t)} V$，把真实更新投影进共享基；然后直接套用任意现成规则（TIES、DARE 等）合并这些坐标 $O_{\text{merged}} = \mathcal{M}(\{O^{(t)}\})$；最后提升回参数空间 $\Delta W_{\text{LoRA}} = U \cdot O_{\text{merged}} \cdot V^\top$。代理目标只在上一步学基时用过，坐标和合并全程基于真实 LoRA，所以各任务的真实贡献没有被重缩放污染。输出秩由构造保证 $\leq r$，这正是绕开截断 SVD 的关键。额外的好处是：在 $r \times r$ 这样的紧凑空间里跨任务冲突更集中，TIES 这类符号冲突消解方法反而更容易处理。

**3. Core-Space 无损加速：把全参数空间的张量分解搬进一个小核心空间**

直接在全参数空间做 HOSVD/HOOI 对 LLaMA3-8B 这种大模型代价高得离谱，约 $\mathcal{O}(n^3 T)$。作者注意到所有 LoRA 更新本就活在一个低维核心里：对拼接的因子 $B_{\text{cat}}$、$A_{\text{cat}}$ 做薄 SVD 得到 $U_B$、$V_A$，把每个 LoRA 投影成核心表示 $M^{(t)} = U_B^\top \Delta W^{(t)} V_A$，在 $(Tr_{\text{in}}) \times (Tr_{\text{in}})$ 的核心空间里求解 Tucker 问题，再通过 $U = U_B U_{\text{core}}$ 提升回原空间。论文证明这个投影对所有 LoRA 更新无损（Theorem 4.1）、且核心空间的最优解等价于原空间（Theorem 4.3），所以是纯加速、零精度损失。维度从 $n$ 降到 $Tr$，加速比约 $n^2 / (Tr)^2$，实测在 LLaMA3-8B 上把子空间学习从 691s 压到 21s。

## 实验关键数据

### 主实验——视觉任务（CLIP ViT-B/32, 8 数据集）

| 合并规则 | 方法 | Cars | DTD | EuroSAT | GTSRB | MNIST | RESISC | SUN397 | SVHN | Avg |
|---------|------|------|-----|---------|-------|-------|--------|--------|------|-----|
| TIES | CoreSpace (MtC) | 83.54 | 73.81 | 53.09 | 39.97 | 65.18 | 68.05 | 95.25 | 43.73 | 65.33 |
| TIES | **CtM (Ours)** | 82.87 | 75.27 | **64.46** | 38.50 | **78.22** | 70.67 | 97.07 | **50.93** | **69.75** |
| DARE-TIES | CoreSpace (MtC) | 84.01 | 73.45 | 53.95 | 40.90 | 64.55 | 69.13 | 96.17 | 45.10 | 65.91 |
| DARE-TIES | **CtM (Ours)** | 83.04 | 75.00 | **64.95** | 38.28 | **79.98** | 69.90 | 96.53 | **57.73** | **70.68** |

### 主实验——语言任务（LLaMA3-8B, 6 NLI 数据集）

| 合并规则 | 方法 | SNLI | MNLI | SICK | QNLI | RTE | SCITAIL | Avg |
|---------|------|------|------|------|------|-----|---------|-----|
| TIES | CoreSpace (MtC) | 91.65 | 93.15 | 93.46 | 83.46 | 99.19 | 97.71 | 93.10 |
| TIES | **CtM (Ours)** | 91.08 | 90.26 | **96.11** | **89.71** | 100.81 | 96.93 | **94.15** |
| DARE-TIES | CoreSpace (MtC) | 91.90 | 91.82 | 95.39 | 80.79 | 100.00 | 97.37 | 92.88 |
| DARE-TIES | **CtM (Ours)** | 92.96 | **94.12** | **97.44** | **94.24** | 97.58 | **97.42** | **95.63** |

### 消融实验

| 配置 | TIES Avg | DARE-TIES Avg | 说明 |
|------|---------|--------------|------|
| Best MtC baseline | 65.33 | 65.91 | CoreSpace + 截断 SVD |
| CtM + SVD 子空间 | 67.42 | 69.04 | 用简单 SVD 替代学习子空间 |
| CtM + 学习子空间 | **69.75** | **70.68** | 完整方法 |
| $\beta=1$（无重缩放） | ~67 | ~68 | 性能一致下降 |

### 关键发现
- **截断损失不可忽视**：MtC 在视觉任务上截断前 Avg$_\text{full}$ = 75.87（Iso-C CoreSpace），截断后骤降至 65.71，损失超 10 个点；CtM 通过在构造层面保证秩约束避免了此损失
- **子空间均衡性优势**：CtM 学到的子空间在能量保持（mean 87.94 vs 69.70）和功能保持（mean 96.18 vs 86.54）上均大幅领先 MtC，且方差极低（std 2.71 vs 21.52），说明 CtM 将秩预算更均匀地分配给各任务
- **Core-Space 加速**：LLaMA3-8B 上从 691s 降至 21s（33× 加速），精度完全一致，Chordal 距离仅 0.031

## 亮点与洞察
- **反转管线的设计哲学**巧妙地将"子空间选择"从被动副产物提升为主动设计对象。这种思维方式在有约束输出的场景中具有通用价值——凡是"先自由操作再约束"可能不如"先约束再操作"的问题都值得考虑
- **重缩放机制**用 $\beta$ 插值在完全归一化和保留原始尺度之间取得平衡，简单有效且对超参不敏感（$\beta \in [0, 0.75]$ 均表现稳定），是处理异质尺度的实用技巧
- **Tucker 分解视角**将子空间学习转化为标准张量分解问题，既有成熟的算法（HOSVD + HOOI），又能与 Core-Space 框架结合实现无损加速，理论和工程上都优雅

## 局限与展望
- 当多个任务的 LoRA 更新高度正交时，任何固定秩方法都不可避免地丢失信息，CtM 也无法突破这一基本瓶颈
- 仍需调节超参（目标秩 $r$、重缩放系数 $\beta$、基础合并规则），开发自适应策略是未来方向
- 当前主要聚焦于同构 LoRA（相同秩和注入模块），异构场景虽有扩展讨论但验证尚不充分
- 可探索为 CtM 坐标空间定制的合并规则（而非直接复用 TIES/DARE），进一步释放低维空间的优势

## 相关工作与启发
- **模型合并**：Task Arithmetic、TIES（符号冲突消解）、DARE（稀疏化）、Iso-C（全参数子空间对齐）——CtM 与这些方法正交，可作为"底座"合并规则的增强框架
- **LoRA 合并**：KnOTS 和 CoreSpace 构建无损基实现坐标对齐但仍依赖截断 SVD；LoRA-LEGO 和 RobustMerge 直接输出低秩但不显式学习共享子空间
- **启发**：CtM 的"先约束后操作"范式可迁移到其他需要结构化输出的合并场景，如合并不同 PEFT 方法（Adapter、Prefix）或多模态 adapter 的场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](../../ACL2026/model_compression/rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)
- [\[ICML 2026\] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)
- [\[ICML 2026\] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters](finer_parameter_steps_for_low-rank_peft_a_controlled_study_with_cp_tensor_adapte.md)
- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)

</div>

<!-- RELATED:END -->
