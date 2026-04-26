---
title: >-
  [论文解读] Steering When Necessary: Flexible Steering Large Language Models with Backtracking
description: >-
  [NeurIPS 2025][视频理解][激活引导] 提出 FASB（Flexible Activation Steering with Backtracking）框架，通过跟踪 LLM 生成过程中的内部状态动态判断干预必要性和强度，并引入回溯机制纠正已偏离的 token，在 TruthfulQA 上 True*Info 达 80.56%、6 个多选任务平均准确率 78.8%，显著优于所有基线。
tags:
  - NeurIPS 2025
  - 视频理解
  - 激活引导
  - 表征工程
  - 回溯机制
  - 真实性
  - 动态干预
---

# Steering When Necessary: Flexible Steering Large Language Models with Backtracking

**会议**: NeurIPS 2025  
**arXiv**: [2508.17621](https://arxiv.org/abs/2508.17621)  
**代码**: https://github.com/gjw185/FASB  
**领域**: Video Understanding / LLM Alignment  
**关键词**: 激活引导, 表征工程, 回溯机制, 真实性, 动态干预

## 一句话总结

提出 FASB（Flexible Activation Steering with Backtracking）框架，通过跟踪 LLM 生成过程中的内部状态动态判断干预必要性和强度，并引入回溯机制纠正已偏离的 token，在 TruthfulQA 上 True*Info 达 80.56%、6 个多选任务平均准确率 78.8%，显著优于所有基线。

## 研究背景与动机

LLM 虽然在文本生成上取得巨大成功，但仍会产生有害信息、幻觉等不期望输出。如何在不进行昂贵微调的前提下控制 LLM 行为是核心挑战。

**激活引导（Activation Steering）**是近年兴起的轻量级方法，通过在推理时直接修改 LLM 内部激活来引导输出方向，避免了大规模数据收集和微调成本。

现有激活引导方法的两大痛点：

**无差别干预**：CAA、ORTHO 等方法对所有生成都施加相同强度的干预向量，无法区分已经正确的输出和需要纠正的输出。对正确生成强行干预反而会降低输出质量。

**仅基于问题判断**：CAST 等方法仅根据问题的表征决定是否干预，但在如真实性等复杂场景中，仅从问题很难预判回答是否会偏离。实验证实，在 TruthfulQA 上仅基于问题的分类器预测概率集中在 0.3-0.5 区间，难以做细粒度决策。

**核心矛盾**：(1) 不同的生成内容需要不同程度的干预；(2) 仅在检测到偏离后再干预往往为时已晚——已生成的偏离 token 无法修正。

**核心 idea**：在每个 token 生成后实时跟踪 LLM 内部状态，同时考虑问题和已生成内容来动态判断干预必要性和强度，并引入回溯机制——检测到偏离时回退若干 token 重新生成，将干预前置到偏离发生之前。

## 方法详解

### 整体框架

FASB 分两步：Step 1 用 Probe 方法识别与目标行为相关的注意力头，获取引导向量和分类器；Step 2 在生成过程中用分类器实时追踪状态，一旦检测到偏离就触发回溯和自适应激活引导。

### 关键设计

1. **注意力头锚定与引导向量提取（Probe 方法）**:

    - 功能：找到 LLM 中与目标行为（如真实性）最相关的注意力头，并构建引导向量
    - 核心思路：将问题和答案拼接，提取每个注意力头在最后一个 token 位置的激活，训练逻辑回归探针 $p_\theta(x) = \text{sigmoid}(\langle \theta^{\ell,h}, x^{\ell,h} \rangle)$。选择验证集上准确率最高的 top-k 个头，直接将探针参数 $\theta^{\ell,h}$ 作为引导向量
    - 设计动机：探针能有效区分正负样本的注意力头最适合做干预目标，且探针参数本身就是区分正负激活方向的最佳向量。只在少量关键头上干预，最小化对模型整体行为的干扰

2. **实时状态追踪**:

    - 功能：在每个 token 生成后评估当前生成是否偏离期望行为
    - 核心思路：生成第 $j$ 个 token 后，用 top-k 个头的分类器分别对第 $j$ 个 token 的激活做预测，取平均偏离概率 $p(x_{i,j}) = \frac{1}{k} \sum_{(\ell,h) \in \text{top-k}} (1 - p_{\theta^{\ell,h}}(x_{i,j}^{\ell,h}))$
    - 设计动机：利用生成过程中已有的隐状态（零额外计算），同时考虑问题和已生成内容，比仅看问题的判断更准确

3. **回溯机制（Backtracking）**:

    - 功能：当检测到偏离时，回退 $s$ 个 token 并在激活引导下重新生成
    - 核心思路：若第 $j$ 个 token 后偏离概率超过阈值 $\beta$，保留前 $j-s$ 个 token，对后续 token 施加引导向量重新生成。干预强度自适应：$r = \mathbb{I}(p > \beta) \cdot p \cdot \alpha$
    - 设计动机：关键洞察——检测到偏离时再干预只能影响后续内容，已生成的偏离 token 无法挽回。回溯机制以极低成本（仅多生成 $s$ 个 token）实现了对已偏离内容的纠正

### 激活引导的实现

干预仅作用于选定的 top-k 注意力头，多头注意力输出变为：
$h_{i,j-s+1}^\ell = \text{concat}(x_{i,j-s+1}^{\ell,1} + r\theta^{\ell,1}c^{\ell,1}, \cdots, x_{i,j-s+1}^{\ell,H} + r\theta^{\ell,H}c^{\ell,H}) W^{\ell,O}$

其中 $c^{\ell,H}$ 为二值标量，仅对选中的头为 1，确保干预的精准性。

## 实验关键数据

### 主实验

TruthfulQA 开放生成任务（LLaMA2-7B-CHAT）：

| 方法 | True (%) | Info (%) | True*Info (%) | MC1 (%) | MC2 (%) |
|------|---------|---------|--------------|---------|---------|
| Baseline | 66.83 | 99.51 | 66.50 | 33.41 | 51.07 |
| ITI | 94.49 | 80.55 | 76.11 | 38.31 | 57.15 |
| CAA | 71.60 | 83.84 | 60.03 | 34.03 | 52.76 |
| SADI-HEAD | 77.72 | 98.53 | 76.58 | 35.90 | 54.65 |
| **Probe (FASB)** | 93.88 | 85.81 | **80.56** | **48.71** | **66.58** |

6 个多选任务平均准确率：

| 方法 | COPA | StoryCloze | NLI | MMLU | SST2 | Winogrande | 平均 |
|------|------|-----------|-----|------|------|-----------|------|
| Baseline | 64.4 | 60.2 | 63.5 | 60.2 | 92.2 | 50.2 | 65.1 |
| ITI | 66.6 | 59.7 | 64.3 | 60.2 | 92.3 | 51.5 | 65.8 |
| **Probe (FASB)** | **90.0** | **93.5** | **80.0** | 62.4 | **92.8** | **54.1** | **78.8** |

### 消融实验

| 配置 | True*Info | MC1 | MC2 | 说明 |
|------|----------|-----|-----|------|
| Probe（完整） | **80.56** | **48.71** | **66.58** | 全部组件 |
| 固定强度（All fixed） | 76.11 | 38.31 | 57.15 | 所有样本相同干预强度 |
| 不自适应（w/o Adaptive） | 82.08 | 42.96 | 62.06 | 仅判断是否干预，不调强度 |
| 无回溯（w/o Backtracking） | 62.11 | 35.01 | 53.55 | 检测到偏离才干预，已太晚 |
| 仅看问题（After Question） | 72.55 | 41.86 | 59.88 | 仅用问题隐状态判断 |

### 关键发现

- **回溯机制至关重要**：无回溯时 True*Info 仅 62.11%，低于基线，说明检测到偏离后再干预确实为时已晚
- **仅看问题不够**：After Question 的 72.55% 远低于考虑生成内容的 80.56%，验证了需要实时追踪
- **自适应强度有效**：固定强度 76.11% → 不自适应 82.08% → 完全自适应 80.56%（True*Info 最优点）
- **跨模型泛化**：在 6 个 LLM（LLaMA2、LLaMA3.1、Qwen2.5）上一致有效，Qwen2.5-7B 上 MC1 提升 24.61
- **跨域泛化**：TruthfulQA 训练的分类器在 Natural Questions（MC2: 49.54→59.25）和 TriviaQA（61.22→67.55）上同样有效

## 亮点与洞察

- 回溯机制是本文最核心的贡献——将"被动检测后干预"转变为"主动回退纠正"，以极低额外成本（仅 $s$ 个 token 重生成）解决了激活引导的固有缺陷
- 状态追踪复用生成过程中已有的隐状态，零额外计算开销
- Probe 方法的探针参数直接复用为引导向量和分类器权重，设计极其简洁统一

## 局限与展望

- 回溯步数 $s$ 和阈值 $\beta$ 需要超参搜索，增加了调参成本
- 当前仅在真实性场景验证，安全性、公平性等场景的效果待探索
- 回溯机制在长序列生成中可能被多次触发，累积开销需评估
- 探针在推理时需缓存 top-k 头的激活，增加了少量内存开销

## 相关工作与启发

- ITI 是先驱工作但无差别干预，FASB 的"探针参数即引导向量"思路更简洁
- 回溯思想与 beam search 回溯有异曲同工之妙，但作用于内部表征层面而非 token 层面
- 对"LLM 内部状态可被轻量级探针实时追踪"的验证对可解释性研究有启示

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](reinforcement_learning_with_backtracking_feedback.md)

<!-- RELATED:END -->
