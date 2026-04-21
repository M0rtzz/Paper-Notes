---
title: >-
  [论文解读] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations
description: >-
  [NeurIPS 2025][多模态][VLM 幻觉] 提出 Topic-level Preference Rewriting（TPR），通过 topic 级别的细粒度语义控制系统性优化偏好数据中的 reward gap 配置，结合课程学习策略逐步提高负样本难度，在多个幻觉基准上实现约 93% 的幻觉减少。
tags:
  - NeurIPS 2025
  - 多模态
  - VLM 幻觉
  - DPO
  - 偏好学习
  - Topic-level Rewriting
  - 课程学习
---

# Systematic Reward Gap Optimization for Mitigating VLM Hallucinations

**会议**: NeurIPS 2025  
**arXiv**: [2411.17265](https://arxiv.org/abs/2411.17265)  
**代码**: https://tpr-dpo.github.io (有)  
**领域**: 多模态大模型 / VLM 幻觉缓解  
**关键词**: VLM 幻觉, DPO, 偏好学习, Topic-level Rewriting, 课程学习

## 一句话总结

提出 Topic-level Preference Rewriting（TPR），通过 topic 级别的细粒度语义控制系统性优化偏好数据中的 reward gap 配置，结合课程学习策略逐步提高负样本难度，在多个幻觉基准上实现约 93% 的幻觉减少。

## 研究背景与动机

VLM（如 GPT-4V、LLaVA）在多模态任务上表现出色，但普遍存在**视觉幻觉**问题——生成与图像不一致的内容。现有基于 DPO 的幻觉缓解方法在偏好数据构建上存在系统性缺陷：

**排序方法**（如 RLAIF-V、AMP）：从模型输出中直接选取 $y_w$ 和 $y_l$，无法纠正底层幻觉，信息量不足，reward gap 可能过小

**重写方法**（如 HA-DPO、HSA-DPO）：依赖外部"黑盒"模型（GPT-4V）重写，难以精确控制修改的类型和幅度，且可能引入偏离模型内在失败模式的幻觉

核心矛盾：DPO 的有效性取决于偏好对中**真实 reward gap** 的质量和幅度——即 $r(y_w;x) - r(y_l;x)$，但现有方法缺少对 reward gap 的**系统性、精细化控制**机制。

TPR 的切入角度：在 **topic 级别**进行操作，利用模型自身的重采样候选（避免外部偏差），并通过选择性替换精确控制每个语义主题上 $y_w$ 和 $y_l$ 的差异，从而系统性地塑造最优的 reward gap 配置。

## 方法详解

### 整体框架

TPR 的 pipeline 分三个核心步骤：
1. **Topic-level Alternatives Generation**（§3.2）：生成每个语义 topic 的多样候选
2. **Selective Topic Replacement**（§3.3）：策略性替换构建偏好对
3. **Curriculum Learning Strategy**（§3.4）：逐步调整负样本难度

### 关键设计

1. **Topic-level 候选生成**：

    - **分解**：用参考模型 $\pi_{ref}$ 将多个候选回复分解为细粒度语义单元 $\{u_{m,n}\}$
    - **Topic 聚类**：基于文本一致性（模型判断两个单元是否描述同一 topic）和视觉相关性（CLIP 特征验证两个单元是否指向图像相同区域）进行聚类
    - **Intra-topic 自重采样**：将每个语义单元转化为 wh- 问题，多次查询 $\pi_{ref}$ 获取同 topic 的多样候选。优势：比完整回复重采样更高效（只需单个 topic 正确），且提供 topic 级别的细粒度控制

2. **选择性 Topic 替换**：

    - **Intra-topic 排名**：将语义单元转化为 yes-no 问题，用 $\pi_{label}$ 打分 $S(u) = p_Y - p_N$，高分表示事实正确、低分表示幻觉
    - **选择性替换**：随机选模板回复 $y_k$，对其每个 topic 的语义单元，用排名池中的候选按策略替换。Greedy 策略：替换为最高/最低分候选（最大化 reward gap）
    - **In-context 重写**：用 $\pi_{ref}$ 将替换后的语义内容无缝整合到模板中，保持语言流畅性

3. **课程学习策略（TPR-CL）**：

    - **Warm-Up 阶段**（60%）：采用 greedy 策略，$y_l$ 中使用最低分候选，提供强初始学习信号
    - **Hard-Mining 阶段**（40%）：$y_l$ 中使用分数逐渐升高的错误候选（更接近决策边界的"硬负样本"），强迫模型学会区分细微幻觉
    - 这种逐步缩小 reward gap 的策略类似于硬负例挖掘，让模型从"区分明显错误"过渡到"区分微妙错误"

### 损失函数 / 训练策略

使用标准 DPO 损失：$\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$。构建 20,000 个偏好对，AdamW 优化器，学习率 $5 \times 10^{-7}$，cosine 衰减。

## 实验关键数据

### 主实验（幻觉基准）

| 方法 | ObjHal CHs↓ | ObjHal CHi↓ | MMHal Score↑ | MMHal Hall.↓ | AMBER Acc↑ | POPE F1↑ |
|------|------|------|----------|------|------|------|
| LLaVA-1.5-7B | 53.6 | 25.2 | 2.36 | 51.0 | 73.5 | 77.6 |
| RLAIF-V-7B | 8.5 | 4.3 | 3.06 | 29.2 | 76.8 | 84.5 |
| HSA-DPO-13B | 5.3 | 3.2 | 2.61 | 48.0 | - | - |
| **TPR-CL-7B** | **3.4** | **1.8** | **3.06** | **30.2** | **82.7** | **87.8** |

TPR-CL 在 ObjHal 上相比基线 LLaVA-1.5 减少了 ~93% 的幻觉（CHs: 53.6→3.4），在 MMHal 上减少 ~41%。

### 消融实验

| 配置 | ObjHal CHs↓ | AMBER Acc↑ | 说明 |
|------|---------|------|------|
| 无多回复采样 | 6.8 | 79.1 | 多回复增加 topic 多样性 |
| 无 intra-topic 重采样 | 5.2 | 80.3 | 自重采样丰富候选池 |
| 仅替换 preferred | 5.8 | 80.0 | 双向替换效果更好 |
| 无 in-context 重写 | 5.6 | 79.5 | 重写保持流畅性 |
| Greedy（TPR） | 4.0 | 82.3 | 贪心策略已有效 |
| **Curriculum（TPR-CL）** | **3.4** | **82.7** | 课程学习进一步提升 |

### 关键发现
- Topic 级别的操控比整体回复级别更精细、高效，是 TPR 成功的核心
- 课程学习策略（从易到难）一致性优于贪心策略，验证了渐进式 reward gap 优化的有效性
- TPR 具有出色的数据效率：仅用 20K 偏好对就达到 SOTA，远优于需要人工标注的方法
- 幻觉缓解不损害通用能力（LLaVA-Bench、MMStar 指标持平或提升）

## 亮点与洞察
- **Reward gap 配置优化**这一视角切中了 DPO 方法的核心痛点——不仅要正确排序偏好，还要精心设计差异的幅度和维度
- 使用模型自身重采样避免外部模型引入偏差，是一个优雅的设计
- Topic 级别的解耦操作（不同 topic 弱相关）为精细控制提供了理论基础

## 局限与展望
- 依赖 LLaVA-NeXT-34B 作为 labeler 模型打分，成本不低
- Topic 聚类的质量依赖 VLM 的 topic 判断能力，可能对复杂场景不准
- 课程学习的阶段划分（60/40）和难度调度是手动设计的，可以探索自适应策略
- 仅在 LLaVA-1.5-7B 上验证，需在更大模型上确认泛化性

## 相关工作与启发
- **vs RLAIF-V**: RLAIF-V 用分治打分排序模型输出，但不修改内容；TPR 主动重写控制 reward gap
- **vs HA-DPO/HSA-DPO**: 这些方法依赖 GPT-4V 重写，TPR 用模型自重采样避免外部偏差
- **vs AMP**: AMP 用不同规模模型对比构建偏好，粒度较粗；TPR 在 topic 级别精确操控

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Reward gap 配置优化视角新颖，topic 级别操控和课程学习的结合巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 多个幻觉基准、通用能力基准、详细消融、数据效率分析
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，motivation 到方法到实验环环相扣
- 价值: ⭐⭐⭐⭐⭐ 93% 幻觉减少 + 高数据效率，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Mitigating Object Hallucinations via Sentence-Level Early Intervention](../../ICCV2025/multimodal_vlm/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [DASH: Detection and Assessment of Systematic Hallucinations of VLMs](../../ICCV2025/multimodal_vlm/dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)

<!-- RELATED:END -->
