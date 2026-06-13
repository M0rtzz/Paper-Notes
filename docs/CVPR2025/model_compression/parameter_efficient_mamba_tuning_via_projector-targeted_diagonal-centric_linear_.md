---
title: >-
  [论文解读] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation
description: >-
  [CVPR 2025][模型压缩][Mamba架构] 本文揭示了 Mamba 架构中 Projector（投影层）而非 SSM 才是迁移学习的关键组件，并提出 ProDiaL 方法——通过对角中心线性变换矩阵间接微调冻结的 Projector 权重，仅训练不到 1% 的参数即可在视觉和语言 Mamba 模型上实现超越 LoRA/DoRA 的下游任务性能。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "Mamba架构"
  - "参数高效微调"
  - "投影层"
  - "对角线变换"
  - "状态空间模型"
---

# Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation

**会议**: CVPR 2025  
**arXiv**: [2411.15224](https://arxiv.org/abs/2411.15224)  
**代码**: 无  
**领域**: 模型压缩/参数高效微调  
**关键词**: Mamba架构, 参数高效微调, 投影层, 对角线变换, 状态空间模型

## 一句话总结

本文揭示了 Mamba 架构中 Projector（投影层）而非 SSM 才是迁移学习的关键组件，并提出 ProDiaL 方法——通过对角中心线性变换矩阵间接微调冻结的 Projector 权重，仅训练不到 1% 的参数即可在视觉和语言 Mamba 模型上实现超越 LoRA/DoRA 的下游任务性能。

## 研究背景与动机

**领域现状**：Mamba 架构凭借选择性 SSM 机制和硬件感知操作，在保持线性计算复杂度的同时实现了上下文感知推理，已在 LLM 和视觉模型中广泛应用。随着 Mamba 模型规模增大，参数高效微调 (PEFT) 变得重要。

**现有痛点**：现有 PEFT 方法（LoRA、Adapter、Prompt Tuning 等）几乎全部针对 Transformer 的注意力模块设计。将这些方法直接应用于 Mamba 时，自然会将 SSM 视为类似注意力模块的核心组件进行微调。唯一探索 Mamba PEFT 的工作（Halloran et al.）也将 LoRA 应用于 SSM 的参数 $W_x$（控制 $B$, $C$, $\Delta$）、Projector 和 Embedding，但未分析各组件的实际贡献。

**核心矛盾**：SSM 虽然是 Mamba 的理论核心，但在迁移学习中其作用可能被高估——Projector 占据了模型约 65% 的参数量且直接控制信息的输入/输出映射，可能才是下游任务适应的关键。然而直接全量微调 Projector 参数过多，需要针对性的 PEFT 方案。

**本文目标**：(1) 系统分析 Mamba 各组件对下游任务的贡献；(2) 设计专门针对 Projector 的高效微调方法。

**切入角度**：通过线性变换视角分析预训练与微调 Projector 权重的关系，发现变换矩阵 $T = W^{-1}W'$ 近似于单位矩阵——对角元素接近 1，非对角元素接近 0，且训练梯度主要集中在对角线上。

**核心 idea**：冻结预训练 Projector $W$，通过训练一个以块对角矩阵 $D_b$ 为主、低秩矩阵 $\epsilon$ 为辅的变换来间接更新权重：$W' = sWD_b + \epsilon$，仅需不到 1% 的参数。

## 方法详解

### 整体框架

ProDiaL 针对 Mamba 块中的 Input-Projector 和/或 Output-Projector 进行参数高效微调。冻结预训练权重 $W$，附加可学习的块对角矩阵 $D_b$、缩放参数 $s$ 和低秩矩阵 $\epsilon = B_\epsilon A_\epsilon$。前向传播时计算 $W' = sWD_b + \epsilon$，训练结束后将变换后的权重直接存储，无额外推理开销。

### 关键设计

1. **Projector 主导发现**:

    - 功能：确定 Mamba 架构中 PEFT 应该针对的核心组件
    - 核心思路：系统地对 Mamba 块中的各组件（SSM 的 $W_x$、Input-Projector、Output-Projector、Embedding）进行选择性微调实验。在 Vision Mamba 上，仅微调 Out-Proj (1.789M 参数) 达到 72.77% 准确率，而微调 SSM (1.441M) 仅 70.04%。在 Mamba LLM 上，仅微调 Out-Proj (28.3M) 达 40.89%，而 SSM (5.38M) 仅 37.52%。更重要的是，Both-Proj 用 LoRA (2.36M) 达 38.33%，仍远超 SSM 的 37.52%——排除了参数量差异的影响
    - 设计动机：打破"SSM 是 Mamba 核心所以 PEFT 应聚焦 SSM"的直觉。Projector 控制信息的输入/输出变换，与下游任务的特征适配更直接相关

2. **对角中心线性变换**:

    - 功能：以极少参数间接更新 Projector 权重
    - 核心思路：将微调权重与预训练权重的关系建模为 $W' = WT$，通过伪逆计算确定性变换矩阵 $T_{det} = W^{-1}W'$。可视化发现 $T_{det}$ 近似单位矩阵，对角线值接近 1，非对角值接近 0。训练 $T$ 时 $L_1$ 范数分析证实梯度集中在对角线上。因此将 $T$ 分解为对角矩阵 $D$ 和非对角矩阵 $b$：$W' = WD + Wb = WD + \epsilon$。用块对角矩阵替代纯对角矩阵（增加表达力，允许微小旋转）；用 LoRA 低秩分解 $\epsilon = B_\epsilon A_\epsilon$
    - 设计动机：LoRA 将权重更新建模为 $W' = W + \Delta W$（加法视角），ProDiaL 从乘法视角出发 $W' = WT$，更符合 Projector 微调的实际特性。对角中心的先验减少了需要学习的参数空间

3. **ProDiaL 完整公式化**:

    - 功能：将上述分析转化为可训练的 PEFT 模块
    - 核心思路：$W' = sWD_b + \epsilon$，其中 $D_b = [\mathbb{I} - \text{relu}(\mathbb{I} * D_a)] + (1 - \mathbb{I}) * D_a$，$D_a = \text{diag}(x_1, ..., x_n)$（$x_i \in \mathbb{R}^{(d_{in}/r_b) \times (d_{in}/r_b)}$ 为小矩阵），$\epsilon = B_\epsilon A_\epsilon$。通过块大小 $r_b$ 和秩 $r_\epsilon$ 灵活控制参数量。$s \in \mathbb{R}^{d_{out}}$ 为逐输出维度的可学习缩放。$D_b$ 通过学习与单位矩阵的差值来初始化近恒等变换
    - 设计动机：块对角结构比纯对角矩阵多一些自由度（可编码局部旋转/混合），同时参数量仍远小于全矩阵。relu 约束确保对角元素非负，$\mathbb{I} - \text{relu}(\mathbb{I} * D_a)$ 使初始值从 1 出发衰减

### 损失函数 / 训练策略

下游分类任务使用标准交叉熵损失。Vision Mamba 在 ImageNet 预训练后迁移到 StanfordCars、Caltech、Flowers 等数据集；Mamba LLM (130M) 在 PILE 预训练后迁移到 HellaSwag、Winogrande、ARC-E、ARC-C 推理任务。训练结束后 $D_b$ 和 $\epsilon$ 合并进 $W$，不增加推理开销。

## 实验关键数据

### 主实验

Mamba LLM (130M) 下游推理任务准确率 (%)：

| 方法 | 参数量 | HellaSwag | Winogrande | ARC-E | ARC-C | Avg. |
|------|--------|-----------|------------|-------|-------|------|
| Full-FT | 130M | 38.23 | 53.12 | 53.54 | 28.84 | 43.43 |
| Strong (SSM+Proj+Emb) | 3.80M | 38.66 | 53.04 | 54.17 | 28.67 | 43.64 |
| Both-Proj LoRA | 2.36M | 38.33 | 53.12 | 53.87 | 29.52 | 43.71 |
| **Both-Proj ProDiaL** | **2.42M** | **38.92** | **53.28** | **55.18** | **28.84** | **44.06** |

Vision Mamba (Vim-tiny) 下游分类准确率 (%)：

| 方法 | 参数量 | StanfordCars | Caltech | Flowers | Avg. |
|------|--------|-------------|---------|---------|------|
| Full-FT | 7.00M | 90.06 | 92.86 | 92.05 | 91.66 |
| Both-Proj LoRA | 0.63M | 85.06 | 96.01 | 87.32 | 89.46 |
| Both-Proj DoRA | 0.69M | 85.18 | 96.09 | 86.60 | 89.29 |
| **Both-Proj ProDiaL** | **0.67M** | **85.38** | **96.24** | **88.00** | **89.87** |

### 消融实验

| 配置 | Vision Avg. | LLM Avg. | 说明 |
|------|-----------|---------|------|
| SSM 微调 | 70.04 | 37.52 | SSM 不是 PEFT 关键 |
| Both-Proj 微调 | 92.22 (5.35M) | 44.16 (84.9M) | Proj 是关键组件 |
| In-Proj only | 91.96 | 44.44 | 单个 Proj 也有效 |
| Out-Proj only | 91.98 | 44.51 | Out-Proj 略优 |
| ProDiaL w/o $\epsilon$ | 降低 | 降低 | 非对角项有贡献 |
| ProDiaL w/o $D_b$ | 降低 | 降低 | 对角项是核心 |

### 关键发现

- Projector 是 Mamba PEFT 的核心：用 LoRA 仅微调 Projector (2.36M) 即超过微调 SSM+Proj+Emb (3.80M) 的 Strong 方法
- Embedding 参数在迁移学习中有害（加入后性能下降），与 Transformer 中 embedding 的作用不同
- 对角线先验经过严格验证：变换矩阵 $T_{det}$ 的可视化和梯度 $L_1$ 分析均证实对角元素主导
- ProDiaL 在不同模型规模（Mamba-130M/370M/1.4B、Vim-tiny/small）上表现一致
- 训练后参数可合并回权重矩阵，无推理开销增加

## 亮点与洞察

- "Projector 而非 SSM 是 Mamba PEFT 关键"这一发现令人意外且有说服力——通过排除参数量因素的对照实验（LoRA 2.36M vs SSM 5.38M 仍然 Proj 优）严格证明了这一点
- 从乘法视角（线性变换 $W'=WT$）而非加法视角（$W'=W+\Delta W$）分析权重变化，提供了新的 PEFT 设计思路。发现 $T$ 近似单位矩阵这一经验规律可能推广到其他架构
- ProDiaL 训练后合并的特性保持了与 LoRA 相同的部署优势，同时通过对角先验提供了更好的归纳偏置

## 局限与展望

- 实验仅在 Mamba-1/2 架构上验证，未涉及 Mamba 的其他变体（如 Jamba、Zamba 等混合架构）
- Vision Mamba (Vim-tiny) 和 Mamba LLM (130M) 的规模相对较小，是否在十亿级模型上仍保持优势有待验证
- 块对角大小 $r_b$ 和低秩 $r_\epsilon$ 需要针对不同任务调优
- 未来方向：(1) 将 ProDiaL 推广到 Mamba-Transformer 混合架构；(2) 探索 Projector 主导迁移学习的理论解释；(3) 在扩散模型等生成任务中验证

## 相关工作与启发

- **vs LoRA**：LoRA 用加法 $W + BA$ 建模权重变化，ProDiaL 用乘法 $WD_b + \epsilon$ 建模，对角先验更契合 Mamba Projector 的微调特性。在相同参数量下 ProDiaL 一致性优于 LoRA
- **vs DoRA**：DoRA 将权重分解为方向和大小两个维度，ProDiaL 将变换分解为对角（缩放）和非对角（旋转/混合）两个维度，后者对 Mamba Projector 更自然
- **vs Strong (Halloran et al.)**：Strong 联合微调 SSM $W_x$、Projector 和 Embedding，本文证明仅需 Projector 即可，且去掉 Embedding 反而更好

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统分析 Mamba 各组件的 PEFT 贡献，发现 Projector 主导性并设计对角先验方法
- **实验充分度**: ⭐⭐⭐⭐ — 视觉+语言双验证、多模型规模、多消融维度，但模型规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 从发现到方法的逻辑链条清晰，可视化分析直观有说服力
- **价值**: ⭐⭐⭐⭐ — 为 Mamba 架构的 PEFT 提供了方法论和实践指导，Projector 主导的发现有独立价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[CVPR 2025\] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)](faster_parameter-efficient_tuning_with_token_redundancy_reduction.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)

</div>

<!-- RELATED:END -->
