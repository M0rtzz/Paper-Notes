---
title: >-
  [论文解读] EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models
description: >-
  [CVPR 2026][多模态][提示学习] EvoPrompt 通过轨迹感知的 prompt 进化策略（统一 embedding 投影 + 方向-幅度解耦训练 + 特征几何正则化）解决 VLM prompt learning 中的灾难性遗忘和模态偏差问题，在 few-shot/跨数据集/域泛化任务上全面 SOTA 且保持 zero-shot 能力。
tags:
  - CVPR 2026
  - 多模态
  - 提示学习
  - 灾难性遗忘
  - 低秩分解
  - 特征去相关
  - VLM适应
---

# EvoPrompt: Evolving Prompt Adaptation for Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2603.09493](https://arxiv.org/abs/2603.09493)  
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: prompt learning, 灾难性遗忘, 低秩分解, 特征去相关, VLM适应

## 一句话总结
EvoPrompt 通过轨迹感知的 prompt 进化策略（统一 embedding 投影 + 方向-幅度解耦训练 + 特征几何正则化）解决 VLM prompt learning 中的灾难性遗忘和模态偏差问题，在 few-shot/跨数据集/域泛化任务上全面 SOTA 且保持 zero-shot 能力。

## 研究背景与动机
**领域现状**：大规模视觉语言模型（CLIP、ALIGN 等）通过对比预训练获得强大的 zero-shot 泛化能力。为了高效适配下游任务，prompt learning（如 CoOp、CoCoOp、MaPLe）通过在冻结 backbone 上插入可学习 prompt token 来实现参数高效微调。

**现有痛点**：
   - **层间隔离**：MaPLe 等方法在每层独立插入 prompt，各层 prompt 相互孤立，无法传递跨层的语义信息流，破坏了 Transformer 的分层语义递进结构。
   - **模态偏差**：现有方案（如 MaPLe）存在文本中心偏差（text-centric bias），未能充分利用视觉-语言的互补交互。
   - **灾难性遗忘**：few-shot 适配时，可学习 prompt 迅速偏离预训练的语义锚点，过拟合少量下游数据，导致 zero-shot 泛化能力严重退化。

**核心矛盾**：任务特定适应 vs. 预训练知识保持之间的 trade-off——现有方法要么 base 类准确率高但 novel 类崩溃，要么保守适应但 base 类提升有限。

**本文目标**：(a) 建立跨层跨模态的 prompt 生成机制；(b) 控制训练过程中 prompt 的演化轨迹，防止知识遗忘；(c) 防止低数据场景下的特征表示坍缩。

**切入角度**：作者观察到 prompt 在训练过程中自然经历"通用语义锚点→任务特定特征"的渐进演化。如果能显式引导这条轨迹——保留早期学到的语义方向，只调整幅度——就能实现"忘不了"的适应。

**核心 idea**：将 prompt 投影器的低秩更新解耦为方向和幅度，冻结历史方向只训练幅度，配合特征几何正则化实现轨迹受控的 prompt 进化。

## 方法详解

### 整体框架
EvoPrompt 基于冻结的 CLIP（ViT-B/16）构建。输入图像和文本分别进入视觉编码器 $F$ 和文本编码器 $G$，两者保持完全冻结。从第 $J$ 层开始到第 $L$ 层，每层都注入通过 Modality-Shared Prompt Projector (MPP) 从统一 embedding 空间投影生成的 prompt。训练过程采用 Evolutionary Trajectory-Aware Learning (ETL) 策略，配合 Feature Geometric Regularization (FGR) 和知识恒常损失 (KCL)。输出为视觉特征 $f^v$ 和文本特征 $f^t$ 的余弦相似度，用于分类。

### 关键设计

1. **Modality-Shared Prompt Projector (MPP)**

    - 功能：从一个统一的可学习 embedding 空间为每层每模态生成 prompt，替代传统的逐层独立 prompt。
    - 核心思路：初始化一个共享的 learnable embedding $E \in \mathbb{R}^{K \times d_r}$（$K=5, d_r=512$），通过解耦投影器将其变换为每一层、每一模态的 prompt。对模态 $m \in \{v, t\}$，第 $i$ 层的 prompt 为 $P_i^m = \text{Proj}_i^m(E)$。投影器权重分解为共享基底加低秩适配器：$W_i^m = W_{\text{shared}}^m + A_i \cdot B_i$，其中 $W_{\text{shared}}^m \in \mathbb{R}^{d_r \times d_m}$ 跨层共享，$A_i \in \mathbb{R}^{d_r \times r}$, $B_i \in \mathbb{R}^{r \times d_m}$ 为层特定低秩矩阵。
    - 设计动机：共享 $W_{\text{shared}}$ 捕获跨层的基础语义知识（如通用视觉/文本模式），低秩 $A_i B_i$ 编码层特定的适应（如浅层纹理 vs. 深层语义）。参数量从 $O((L-J+1) \cdot d_r \cdot d_m)$ 降至 $O(d_r \cdot d_m + (L-J+1) \cdot r \cdot (d_r + d_m))$，比 MaPLe 少 4.6 倍。

2. **Evolutionary Trajectory-Aware Learning (ETL)**

    - 功能：通过方向-幅度解耦和渐进式知识积累控制 prompt 的训练轨迹，防止灾难性遗忘。
    - 核心思路：在训练 epoch $t$ 时，将层 $i$ 的低秩更新分解为幅度系数 $\alpha_i^t$ 和归一化方向矩阵：$\Delta W_i^t = \alpha_i^t \cdot \overline{A_i^t B_i^t}$（$\overline{\cdot}$ 表示 Frobenius 归一化）。到 epoch $T$ 时，总权重为历史方向的加权和：
    $W_i^T = W_{\text{shared}} + \sum_{t=1}^{T-1} \alpha_i^t \cdot \overline{A_i^t B_i^t} + \alpha_i^T \cdot \overline{A_i^T B_i^T}$
   关键：冻结所有历史方向 $\{\overline{A_i^t B_i^t}\}_{t=1}^{T-1}$，只训练幅度 $\{\alpha_i^t\}_{t=1}^T$ 和当前新方向 $\overline{A_i^T B_i^T}$。
    - 设计动机：先前研究表明低秩适应中方向比幅度更关键（DoRA）。早期训练建立的方向编码了鲁棒的语义结构，冻结方向相当于保护了"认知骨架"，只让幅度系数做自由调整来适应任务。同时新 epoch 引入的新方向允许学习增量知识。
    - **Adaptive Rank Reduction**：训练过程中分阶段降低低秩矩阵的秩 $r_1 > r_\mu > r_\nu$（在 epoch $\mu$ 和 $\nu$ 处降秩）。后期 epoch 的边际贡献递减，用更低的秩既是结构化正则化（防止过拟合），又减少了累积参数和计算开销。

3. **Feature Geometric Regularization (FGR)**

    - 功能：防止特征空间维度冗余和表示坍缩，增强特征的正交性和去相关性。
    - 核心思路：基于 Soft-HGR（Hirschfeld-Gebelein-Rényi）最大相关性框架。InfoNCE 对比损失可以看作最大化跨模态对齐项（Soft-HGR 目标函数的第一项），但忽略了模态内协方差结构（第二项）。FGR 显式最小化模态内协方差矩阵乘积的迹：
    $\mathcal{L}_{fgr}(\mathcal{F}^v, \mathcal{F}^t) = \frac{1}{2} \text{tr}(\text{cov}(\mathcal{F}^v) \cdot \text{cov}(\mathcal{F}^t))$
    - 设计动机：对比学习只关注实例级对齐，容易导致特征维度高度冗余。FGR 鼓励学到的表示中各维度去相关，有效利用特征空间的每个维度，这在低数据场景下尤其关键。

### 损失函数 / 训练策略
总训练目标为三项加权和：
$$\mathcal{L}_{total} = \mathcal{L}_{InfoNCE} + \gamma \cdot \mathcal{L}_{fgr} + \eta \cdot \mathcal{L}_{kcl}$$
其中知识恒常损失 $\mathcal{L}_{kcl} = \frac{1}{2}[(1 - \cos(f^v, f_0^v)) + (1 - \cos(f^t, f_0^t))]$ 约束 prompted 特征不偏离原始冻结 CLIP 的特征分布。最优超参 $\gamma=25$, $\eta=0.5$。

训练配置：ViT-B/16 backbone，16-shot per class，prompt 从第 6 层插入到第 12 层，token 长度 $l=5$，embedding 向量数 $K=5$，单卡 A800，3 个随机种子取均值。

## 实验关键数据

### 主实验：Base-to-Novel Generalization（11 个数据集平均）

| 方法 | Base | Novel | HM |
|------|------|-------|------|
| CLIP (zero-shot) | 69.34 | 74.22 | 71.70 |
| CoOp | 82.69 | 63.22 | 71.66 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| PromptSRC | 84.26 | 76.10 | 79.97 |
| MMA | 83.20 | 76.80 | 79.87 |
| **EvoPrompt** | **84.28** | **77.76** | **80.73** |

EvoPrompt HM 提升 +0.76%，Novel 类提升 +0.96%（对比之前最佳）。在 FGVCAircraft 上 Novel 类提升 1.27%，在 EuroSAT 上 HM 达 86.54%（SOTA MMA 为 83.87%）。

### 消融实验（ImageNet Base-to-Novel）

| 配置 | Base | Novel | HM | 说明 |
|------|------|-------|------|------|
| w/o MPP | 75.32 | 70.15 | 72.64 | 去掉统一投影器，退回逐层独立 prompt，HM 掉 1.65% |
| w/o $W_{\text{shared}}$ | 75.80 | 71.42 | 73.54 | 去掉共享权重，各层独立投影 |
| w/o AB | 76.15 | 70.90 | 73.43 | 去掉低秩适配器，用全秩投影 |
| w/o E.T. | 77.42 | 70.25 | 73.66 | 去掉轨迹感知训练：Base 上升但 Novel 暴跌 |
| w/o $\mathcal{L}_{kcl}$ | 77.24 | 70.55 | 73.74 | 去掉知识恒常损失 |
| w/o $\mathcal{L}_{fgr}$ | 76.70 | 70.52 | 73.48 | 去掉几何正则化 |
| **Full EvoPrompt** | 76.98 | 71.80 | **74.29** | 最佳平衡 |

### 关键发现
- **MPP 是基础**：去掉 MPP 后 HM 从 74.29% 跌至 72.64%，贡献最大。
- **ETL 和 KCL 的作用是"防遗忘"**：去掉后 Base 反而升高（过拟合 base），但 Novel 大幅下降，符合灾难性遗忘特征。
- **FGR 提供关键微调**：去掉后 HM 降至 73.48%，说明特征去相关对低数据泛化很重要。
- **训练效率**：仅 0.764M 可训练参数，推理速度 1282.1 FPS，训练 4.5ms/image。参数量比 MaPLe（3.555M）少 4.6 倍。
- **幅度演化规律**：学习到的 $\alpha$ 在 epoch 2 达峰值后逐渐衰减，说明早期建立核心特征方向、后期做精细微调。

## 亮点与洞察
- **方向-幅度解耦是核心技巧**：将低秩更新分解为 direction + magnitude，冻结历史 direction 只调 magnitude，是一种优雅的"渐进式知识积累"机制。这个思路不仅适用于 prompt learning，可以迁移到任何 LoRA 微调场景——在 continual learning 场景下特别有价值。
- **Soft-HGR 正则化的跨领域迁移**：把信息论中的最大相关性分析用于约束对比学习的特征几何结构，比简单的正交正则化更有原理支撑。这个 $\mathcal{L}_{fgr}$ 可以直接用于其他对比学习框架。
- **过拟合拐点"breakpoint"分析**（Fig.4）：MaPLe 在 breakpoint 后 Novel 类不可逆退化，而 EvoPrompt 在 breakpoint 后 Novel 类稳定。这个诊断方法本身就有参考价值。

## 局限与展望
- **仅在 CLIP ViT-B/16 上验证**：未报告在更大 backbone（ViT-L/14）或其他 VLM（如 SigLIP、EVA-CLIP）上的表现。
- **适应性 rank reduction 的阈值 $\mu, \nu$ 是手工设定的**：更理想的做法是根据验证集性能自适应调整。
- **方向累积的存储开销**：随 epoch 增加，需要存储所有历史方向矩阵。虽然有 rank reduction 缓解，但长期训练时可能成为瓶颈。可以探索方向合并（merging）策略。
- **FGR 的 batch size 敏感性**：协方差矩阵的估计质量依赖于 batch 大小，在极端 few-shot（1-shot）下可能不稳定。

## 相关工作与启发
- **vs MaPLe**: MaPLe 在每层独立插入 prompt 并通过耦合函数链接视觉-文本 prompt。EvoPrompt 用统一 embedding + shared base 替代，参数效率高 4.6 倍且跨层信息流更自然。
- **vs PromptSRC**: PromptSRC 用自一致性正则化（预测一致性约束）防遗忘。EvoPrompt 从优化轨迹层面解决，通过冻结方向从根本上阻止"方向漂移"，更直接。
- **vs DoRA**: DoRA 也做方向-幅度分解但是单次训练，EvoPrompt 把它扩展为 epoch 级别的渐进式积累，更适合 prompt evolution 的时序建模。

## 评分
- 新颖性: ⭐⭐⭐⭐ 方向-幅度解耦+渐进积累的训练策略很有创意，FGR 的引入也有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 4 种评估设置 × 11 个数据集，消融/超参/效率/轨迹分析全覆盖
- 写作质量: ⭐⭐⭐⭐ 整体结构清晰，数学推导完整，但分支组件较多读起来密度较大
- 价值: ⭐⭐⭐⭐ 方法有效且参数高效，核心 trick（方向冻结 + 幅度调整）有广泛迁移价值

<!-- RELATED:START -->

## 相关论文

- [Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)
- [Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)
- [Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)

<!-- RELATED:END -->
