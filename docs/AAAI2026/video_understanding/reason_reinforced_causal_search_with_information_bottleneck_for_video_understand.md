---
title: >-
  [论文解读] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding
description: >-
  [AAAI 2026][视频理解][关键帧选择] 提出因果信息瓶颈（CIB）理论框架，将关键帧选择形式化为同时优化"预测充分性"和"因果必要性"的信息论问题，并基于此设计 ReaSon 强化学习框架，通过三种 CIB 对齐的奖励（答案奖励、循环一致性奖励、反事实奖励）训练选择策略，在限定帧数设置下显著超越已有方法。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "关键帧选择"
  - "因果信息瓶颈"
  - "强化学习"
  - "反事实推理"
  - "视频问答"
---

# ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding

**会议**: AAAI 2026  
**arXiv**: [2511.12530](https://arxiv.org/abs/2511.12530)  
**代码**: [github.com/robin-hlt/AAAI26-ReaSon](https://github.com/robin-hlt/AAAI26-ReaSon)  
**领域**: 视频理解  
**关键词**: 关键帧选择, 因果信息瓶颈, 强化学习, 反事实推理, 视频问答

## 一句话总结

提出因果信息瓶颈（CIB）理论框架，将关键帧选择形式化为同时优化"预测充分性"和"因果必要性"的信息论问题，并基于此设计 ReaSon 强化学习框架，通过三种 CIB 对齐的奖励（答案奖励、循环一致性奖励、反事实奖励）训练选择策略，在限定帧数设置下显著超越已有方法。

## 研究背景与动机

**领域现状**：视觉-语言模型（VLM）在视频理解中取得了显著进展，但受限于输入 token 预算和视频固有的时间冗余。为此，近期研究聚焦于**关键帧选择**策略，从视频中提取关键帧子集以提升计算效率和推理准确性。

**现有痛点**：

**关键帧定义不清**：现有方法（如 VideoAgent、T*、AKEYS）将关键帧定义为"信息丰富且紧凑的帧子集"，通常通过**视觉或语义相关性**作为信息量的代理指标。然而，高视觉相关性≠决定性证据——视觉上相关的帧可能并非推理所必需，而因果决定性的帧（如原因和结果帧）反而被忽略。

**缺乏因果分析**：现有方法隐式遵循信息瓶颈（IB）原则，但 IB 框架仅关注预测充分性，无法捕捉**因果必要性**（即移除某帧是否会显著影响输出）。

**理论基础薄弱**：无论是启发式静态选择还是代理交互搜索，都缺乏统一的理论框架来指导选择过程。

**核心矛盾**：视觉相关≠因果必要。一帧可能与问题高度视觉相关却对推理无贡献（冗余帧），另一帧可能视觉上不突出但对正确推理不可或缺（因果帧）。

**切入角度**：从因果推理视角重新审视关键帧概念，关键帧应同时满足两个条件——（1）**预测充分性**：所选帧子集足以准确推理答案；（2）**因果必要性**：子集中没有帧可以被移除而不影响输出。将经典 IB 扩展为**因果信息瓶颈（CIB）**，并用强化学习训练选择策略。

## 方法详解

### 整体框架

ReaSon 由两个核心模块组成：**预测充分性模块**和**因果必要性模块**。预测充分性模块先通过视觉元素匹配构建候选帧池，再用可学习策略网络选择紧凑子集；因果必要性模块通过反事实干预评估每帧的因果不可或缺性。两个模块产生的奖励信号通过强化学习联合训练选择策略。

### 关键设计

#### 1. **因果信息瓶颈（CIB）**

经典 IB 目标为 $\max \mathcal{I}(F;O) \text{ s.t. } \mathcal{I}(V,Q;F) \leq \beta$，仅关注预测充分性。本文引入**可干预的选择变量** $S$ 替代 $F$ 作为瓶颈变量，将目标扩展为：

$$\max \mathcal{I}(S;O) + \mathcal{I}_c(O;\text{do}(S)) \quad \text{s.t. } \mathcal{I}(V,Q;S) \leq \beta$$

其中第一项 $\mathcal{I}(S;O)$ 确保预测充分性，第二项 $\mathcal{I}_c(O;\text{do}(S))$ 通过因果干预量化所选帧的因果影响力，约束 $\mathcal{I}(V,Q;S) \leq \beta$ 限制信息容量防止冗余选择。

**核心贡献在于**：将选择视为一个可干预的决策，从而可以分析因果必要性——传统 IB 的瓶颈变量 $F$ 无法表示可干预的选择。

#### 2. **预测充分性模块**

**候选帧池构建**：受 T* 启发，先用均匀采样帧通过冻结 VLM 提取问题相关视觉元素 $E_q$，再用开放词汇检测器 YOLO-World 在所有帧中匹配这些元素，构建大小为 $M=32$ 的候选帧池。

**策略网络选择**：用 BLIP 编码帧和问题，三层 LSTM + MLP 组成策略网络 $\pi_\theta(S|f,q)$，为候选池中每帧分配选择概率，通过多项式采样得到 $|s| \leq K=8$ 的关键帧子集。

**答案奖励**：将选中帧和问题送入冻结 VLM 生成答案，与 ground truth 对比：
$$R_{\text{ans}} = \mathbb{I}[\text{VLM}(s,q) = \text{gt}]$$

**循环一致性奖励**：将生成的答案与问题拼接，让 VLM 反推视觉元素 $E_a$（不访问视频帧），与原始 $E_q$ 计算 IoU：
$$R_{\text{cycle}} = \text{IoU}(E_q, E_a)$$

这是一个语义循环验证：从视觉输入→答案推理→反推视觉归因，如果对齐良好说明所选帧真正保留了推理所需的语义线索。

#### 3. **因果必要性模块**

**反事实选择策略**：将原策略概率取反并归一化，构造反事实选择概率：
$$\tilde{\pi}(f_i) = \frac{1 - \pi_\theta(f_i)}{\sum_j (1 - \pi_\theta(f_j))}$$

即原策略倾向于选择的帧在反事实策略中被抑制，反之亦然。

**反事实奖励**：采样原始子集 $s$ 和反事实子集 $s'$，分别送入 VLM 得到 logits 输出 $o$ 和 $o'$，计算 KL 散度：
$$R_{\text{cf}} = D_{\text{KL}}(\text{softmax}(o) \| \text{softmax}(o'))$$

散度越大说明当前帧子集的因果影响力越强——移除这些帧（用反事实帧替代）会导致输出显著变化。

### 损失函数 / 训练策略

**复合奖励**：$R = R_{\text{ans}} + \lambda_1 R_{\text{cycle}} + \lambda_2 R_{\text{cf}}$，$\lambda_1 = \lambda_2 = 0.5$。

**组策略梯度**：每个训练实例采样 $G=4$ 组关键帧+1个反事实子集，计算组内优势 $\hat{A}_i = R_i - \frac{1}{G}\sum_j R_j$（均值中心化而非标准差归一化以减少偏差）。策略更新：
$$\nabla_\theta \mathcal{L} = \frac{1}{G} \sum_{i=1}^G \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(s_i | f, q)$$

在 NExT-QA 训练集上训练，8帧实验用单张 RTX 3090，32帧实验用 A100。

## 实验关键数据

### 主实验

**NExT-QA 验证集 & EgoSchema 子集（8帧）**：

| 方法 | VLM | 帧数 | Temporal | Causal | Descriptive | Avg | EgoSchema |
|------|-----|------|----------|--------|-------------|-----|-----------|
| VideoAgent | GPT-4 | 8.4 | 64.5 | 72.7 | 81.1 | 71.3 | 60.2 |
| AKEYS | GPT-4o | 26.7 | 72.9 | 79.0 | 86.1 | 78.1 | 68.6 |
| T* | LLaVA-OV-7B | 8 | - | - | - | 80.4 | 66.6 |
| **ReaSon** | LLaVA-Video-7B | 8 | **77.3** | **82.1** | **87.4** | **81.4** | 69.0 |
| **ReaSon** | GPT-4o | 8 | 70.6 | 80.2 | 83.6 | 77.6 | **72.2** |

**Video-MME（无字幕）**：

| 方法 | VLM | 帧数 | Short | Medium | Long | Overall |
|------|-----|------|-------|--------|------|---------|
| T* | GPT-4o | 8 | 56.4 | 57.3 | 56.4 | 56.5 |
| **ReaSon** | GPT-4o | 8 | **65.9** | **57.1** | **54.4** | **59.1** |
| T* | GPT-4o | 32 | 69.5 | 63.5 | 59.3 | 64.1 |
| **ReaSon** | GPT-4o | 32 | **76.8** | **64.2** | 58.2 | **66.4** |

### 消融实验

| 配置 | NExT-QA Avg | EgoSchema |
|------|-------------|-----------|
| 仅 $R_{\text{ans}}$ | 80.1 | 66.0 |
| $R_{\text{ans}} + R_{\text{cycle}}$ | 80.5 | 68.2 |
| $R_{\text{ans}} + R_{\text{cycle}} + R_{\text{cf}}$（完整） | **81.4** | **69.0** |

**VLM 泛化性**：

| VLM | NExT-QA (原始→+ReaSon) | EgoSchema (原始→+ReaSon) |
|-----|----------------------|------------------------|
| LLaVA-Video-7B | 80.2→**81.4** | 65.2→**69.0** |
| Qwen2.5-VL-7B | 79.9→**80.4** | 65.8→**68.0** |
| GPT-4o | 72.0→**77.6** | 70.0→**72.2** |

### 关键发现

1. **因果必要性是核心提升来源**：添加反事实奖励 $R_{\text{cf}}$ 后 EgoSchema 提升 0.8%，NExT-QA Descriptive 提升 3.1%，说明因果帧选择显著改善了推理准确性。
2. **短视频提升最显著**：在 Video-MME 短视频上 8 帧设置下 ReaSon 比 T* 提升 9.5%，因为短视频中因果线索更集中。
3. **少帧胜多帧**：ReaSon 用 8 帧（7B 模型）超越了使用 90 帧的 VideoINSTA（GPT-4），验证了"质量优于数量"。
4. **模型无关的即插即用**：在三个不同 VLM 上一致提升，最大增幅出现在 GPT-4o（NExT-QA +5.6%）和 Qwen2.5-VL（EgoSchema +3.8%）。

## 亮点与洞察

1. **理论贡献突出**：将关键帧选择从经验启发式提升到信息论层面，CIB 提供了统一的理论框架来分析和设计选择策略。  
2. **反事实干预设计精妙**：通过概率取反构造反事实选择，既简洁又有效地近似了因果必要性的度量。
3. **循环一致性奖励**：将答案反推回视觉元素作为语义验证，是一种巧妙的自监督一致性检查。
4. **计算效率高**：训练和推理仅需单张 3090（8帧），且策略网络非常轻量（3层 LSTM + MLP），不增加 VLM 推理成本。
5. **强泛化性**：训练在 NExT-QA 上完成，直接在 EgoSchema 和 Video-MME 上泛化表现优异。

## 局限与展望

1. **长视频场景仍有瓶颈**：在 Video-MME 长视频上增益有限（32帧时 Long 类别低于 T*），因为候选帧池大小固定（M=64）可能无法覆盖超长视频的关键信息。
2. **候选帧池依赖视觉元素检测**：使用 YOLO-World 检测视觉元素可能遗漏抽象或非物体线索。
3. **单 Monte Carlo 样本近似**：反事实奖励使用单次采样近似期望，可能引入方差。
4. **训练仅在 NExT-QA 上**：未探索在更大规模或更多样化数据上训练的效果。

## 相关工作与启发

- 将 IB 原理引入帧选择是自然但未被充分探索的方向，本文的 CIB 扩展为后续工作（如长视频中的关键片段选择、多模态信息筛选）提供了理论工具。
- 反事实推理在因果机器学习中广泛使用，本文展示了其在视频理解中的实践价值。
- 组策略梯度方法比 PPO 更轻量，适合帧选择这种离散组合优化问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （CIB 理论框架+反事实帧选择，理论和方法层面双重创新）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个数据集、三个 VLM、完整消融、可视化分析）
- 写作质量: ⭐⭐⭐⭐⭐ （理论推导严谨，符号体系完整，逻辑清晰）
- 价值: ⭐⭐⭐⭐⭐ （为关键帧选择建立理论基础，即插即用设计实用性强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](causality_matters_how_temporal_information_emerges_in_video_language_models.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)

</div>

<!-- RELATED:END -->
