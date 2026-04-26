---
title: >-
  [论文解读] CVPT: Cross Visual Prompt Tuning
description: >-
  [ICCV2025][多模态][提示学习] 针对 Visual Prompt Tuning (VPT) 中 prompt token 参与 self-attention 导致的计算冗余和注意力破坏问题，提出 CVPT，通过 cross-attention 解耦 prompt 与 image token 的交互，并利用权重共享机制初始化 cross-attention，在 25 个数据集上显著超越 VPT，性能媲美主流 adapter 方法。
tags:
  - ICCV2025
  - 多模态
  - 提示学习
  - 注意力机制
  - PEFT
  - parameter-efficient fine-tuning
  - Weight Sharing
---

# CVPT: Cross Visual Prompt Tuning

**会议**: ICCV2025  
**arXiv**: [2408.14961](https://arxiv.org/abs/2408.14961)  
**代码**: https://github.com/Lingyun0419/CVPT  
**领域**: 多模态VLM / 参数高效微调  
**关键词**: Visual Prompt Tuning, Cross-Attention, PEFT, parameter-efficient fine-tuning, Weight Sharing

## 一句话总结
针对 Visual Prompt Tuning (VPT) 中 prompt token 参与 self-attention 导致的计算冗余和注意力破坏问题，提出 CVPT，通过 cross-attention 解耦 prompt 与 image token 的交互，并利用权重共享机制初始化 cross-attention，在 25 个数据集上显著超越 VPT，性能媲美主流 adapter 方法。

## 研究背景与动机

**领域现状**：大规模预训练模型（如 ViT）全量微调成本巨大，参数高效微调（PEFT）成为主流。PEFT 中有两大流派：adapter 方法和 prompt 方法。在视觉领域，adapter 方法（如 AdaptFormer、LoRA）通常优于 prompt 方法（如 VPT），导致社区普遍认为"prompt 方法不如 adapter"。

**现有痛点**：VPT 将 prompt token 直接拼接到 image token 序列中一起送入 Transformer block 的 self-attention，带来两个严重问题：
   - **计算复杂度**：self-attention 的复杂度从 $n^2$ 增长到 $(n+m)^2$，随 prompt 数量增加，开销急剧上升
   - **注意力破坏**：prompt token 在 softmax 归一化中"抢占"了 embedded token 之间的注意力权重。当 prompt 数为 196 时，prompt 占据了超过 80% 的注意力权重，严重破坏原始特征表示

**核心矛盾**：VPT 需要大量 prompt 来适应下游任务，但放入更多 prompt 反而破坏性能。这个矛盾的根源在于 prompt 的"部署方式"——它与 image token 共享同一个 self-attention，二者耦合在一起。

**本文目标**：如何在保留 prompt 灵活性的同时，消除其对 self-attention 的干扰，使 prompt 方法能够在使用大量 prompt 时仍保持高性能和高效率。

**切入角度**：作者观察到 prompt token 并非原始序列的固有组成部分，不携带语义信息，只是作为间接调优因子。因此，将 prompt 从 self-attention 中解耦是根本解法。

**核心 idea**：用 cross-attention 替代 self-attention 中的 prompt 拼接，让 embedded token 作为 query、prompt 作为 key-value，解耦二者交互。

## 方法详解

### 整体框架
CVPT 的 pipeline 与 VPT 类似：冻结预训练 ViT 主干参数，只训练 prompt token 和最终分类头。关键区别在于 prompt token 不再拼接到 image token 序列中，而是通过一个独立的 cross-attention 模块与 embedded token 交互。具体流程为：输入 → Patch Embedding → 经过多层 Transformer Block（每个 block 中依次执行 frozen self-attention → cross-attention with prompt → frozen MLP）→ 分类头输出。

### 关键设计

1. **Cross-Attention 解耦模块**:

    - 功能：在每个 Transformer block 的 self-attention 之后、MLP 之前，插入一个 cross-attention 层
    - 核心思路：embedded token 作为 query（$Q = X_1 W^Q$），prompt token 作为 key 和 value（$K = V = X_2 W^K$），计算 $\text{CrossAttention}(X_1, X_2) = \text{Softmax}(\frac{Q \cdot K}{\sqrt{d_k}}) V$，结果以残差方式加回 embedded token
    - 设计动机：这样做有三个好处：(1) self-attention 中只有 image token，保留完整的特征表示能力；(2) cross-attention 的计算复杂度为 $O(n \cdot m)$，是线性的而非二次的；(3) 输出维度与 embedded token 一致，可以直接做残差连接，不会给后续 MLP 增加额外计算

2. **权重共享机制（Weight Sharing）**:

    - 功能：用预训练 self-attention 的权重初始化 cross-attention，并在训练过程中冻结 cross-attention 的参数
    - 核心思路：cross-attention 和 self-attention 结构相同（都是 QKV 投影 + softmax），因此可以直接复用 self-attention 的预训练权重作为初始化
    - 设计动机：(1) 避免引入大量可学习参数（frozen CA 只需 0.09M 参数 vs 可学习 CA 需 28.4M）；(2) self-attention 的预训练权重编码了丰富的视觉知识，提供了强有力的归纳偏置；(3) 实验证明 frozen CA + weight sharing 的性能与 learnable CA 相当

3. **最优部署位置**:

    - 功能：探索 cross-attention 在 Transformer block 中的最佳插入位置
    - 核心思路：测试了 5 个候选位置，发现在 SA 之后（位置 3）效果最好，准确率 74.0%
    - 设计动机：SA 模块产生丰富的上下文特征，在其后立即进行 cross-attention 可以更有效地利用这些特征进行 prompt 适配

### 损失函数 / 训练策略
沿用 VPT 的训练策略，使用 AdamW 优化器，仅优化 prompt token 和分类头。prompt 数量从 {1, 5, 10, 20, 50, 100, 200} 中选择最优值。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CVPT | VPT-Deep | LoRA | DMLoRA | 提升(vs VPT) |
|--------|------|------|----------|------|--------|-------------|
| VTAB-1K (19ds avg) | Top-1 Acc | **77.2** | 72.0 | 74.5 | 77.0 | +5.2% |
| VTAB Natural (7ds) | Avg Acc | 83.3 | 71.6 | - | - | +11.7% |
| VTAB Structured (8ds) | Avg Acc | 61.7 | 55.0 | - | - | +6.7% |
| FGVC (5ds avg) | Top-1 Acc | **90.5** | 89.1 | 89.5 | 90.7 | +1.4% |
| ADE20K (P=200) | mIoU-SS | **45.66** | 42.11 | - | - | +3.55% |
| ADE20K (P=200) | mIoU-Ms | **47.92** | 44.06 | - | - | +3.86% |

### 消融实验

| 配置 | VTAB Avg Acc | FGVC Avg Acc | Params (M) | 说明 |
|------|-------------|-------------|------------|------|
| Weight Sharing + Frozen CA | **74.0** | **89.3** | 0.09 | 完整模型，最高效 |
| Weight Sharing + Learnable CA | 74.6 | 89.5 | 28.4 | 参数量大 300 倍，性能仅微升 |
| Random Init + Learnable CA | 74.0 | 89.5 | 28.4 | 权重共享无明显优势当 CA 可学习时 |
| Random Init + Frozen CA | 63.7 | 86.0 | 0.09 | 无 weight sharing 时 frozen CA 崩塌 |
| Linear Probing | 57.6 | 79.3 | 0 | 基线 |

### 关键发现
- **Prompt 数量敏感性**：VPT 在 prompt 数 >50 时性能急剧下降（200 prompts 时从 73.0 跌至 64.0），而 CVPT 性能随 prompt 数稳步上升（200 prompts 达到 74.8），验证了解耦设计的有效性
- **OOD 数据集优势明显**：在与预训练分布差异大的 Structured 数据集上，CVPT 提升最显著，说明更多 prompt 有助于适配分布外任务
- **效率优势**：使用 200 prompts 时，CVPT 的 FLOPs 和显存开销远低于 VPT，且不随 prompt 数量爆炸式增长

## 亮点与洞察
- **从根本上解决 VPT 的架构缺陷**：不是在 VPT 框架内修补（如调整 prompt 位置、改变 prompt 初始化），而是重新思考 prompt 与 image token 的交互方式。这种"解耦思维"可以广泛应用于其他涉及辅助 token 注入的场景
- **权重共享是 zero-cost lunch**：frozen CA + weight sharing 达到了 learnable CA 的性能，但参数量只有 0.09M（对比 28.4M），是一个极其高效的设计。这个 trick 可以迁移到任何需要插入新模块但又希望避免大量参数的场景
- **推翻了"prompt 不如 adapter"的社区共识**：CVPT 在 VTAB-1K 上超过了所有 adapter 方法，证明 prompt 方法的瓶颈在于部署方式而非方法本身

## 局限与展望
- **Prompt 初始化策略未探索**：作者承认没有提出新的 prompt 初始化方法，目前沿用 VPT 的随机初始化。更好的初始化可能进一步提升性能
- **仅在 ViT-B/16 和 ViT-L 上验证**：未测试更大规模模型（如 ViT-H、ViT-G），跨模型泛化性有待确认
- **cross-attention 位置固定**：所有层使用相同的 cross-attention 布局，未探索层级自适应的部署策略
- **未与最新 PEFT 方法对比**：如 QLoRA、DoRA 等更新的方法未纳入对比

## 相关工作与启发
- **vs VPT**：VPT 将 prompt 拼接入 self-attention，CVPT 用独立 cross-attention 解耦。CVPT 在 VTAB-1K 上超出 5.2%，且不受 prompt 数量限制
- **vs Adapter (AdaptFormer)**：Adapter 在 Transformer block 内插入小型可学习模块。CVPT 用不同思路（prompt + cross-attention）达到了相当甚至更优的效果，证明两条路线并非有优劣之分
- **vs E²VPT**：E²VPT 仅将 prompt 用于 key-value 矩阵，仍在 self-attention 内部操作。CVPT 彻底将 prompt 移出 self-attention

## 评分
- 新颖性: ⭐⭐⭐⭐ 从"解耦 prompt"角度切入，cross-attention 本身不新但应用在 VPT 改进上是新颖的
- 实验充分度: ⭐⭐⭐⭐⭐ 25 个数据集、3 类任务、详尽的消融和效率分析
- 写作质量: ⭐⭐⭐⭐ 分析清晰，问题-方法-实验逻辑链完整
- 价值: ⭐⭐⭐⭐ 推翻了 prompt vs adapter 的固有认知，对 PEFT 社区有重要启示

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](../../CVPR2025/multimodal_vlm/dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)

<!-- RELATED:END -->
