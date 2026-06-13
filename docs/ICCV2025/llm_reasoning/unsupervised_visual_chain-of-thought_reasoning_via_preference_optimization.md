---
title: >-
  [论文解读] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization
description: >-
  [ICCV 2025][LLM推理][Visual CoT] 提出UV-CoT框架，通过自动生成偏好数据和改进的Score-DPO损失函数，在不需要人工标注bounding box的情况下实现图像级链式思维（Visual CoT）推理，在6个基准上超越有监督的Visual-CoT方法。 CoT推理显著提升了MLLM的可解释性…
tags:
  - "ICCV 2025"
  - "LLM推理"
  - "Visual CoT"
  - "偏好优化"
  - "无监督学习"
  - "多模态推理"
  - "Bounding Box"
---

# Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization

**会议**: ICCV 2025  
**arXiv**: [2504.18397](https://arxiv.org/abs/2504.18397)  
**代码**: [https://github.com/kesenzhao/UV-CoT](https://github.com/kesenzhao/UV-CoT)  
**领域**: LLM Reasoning / Multimodal  
**关键词**: Visual CoT, 偏好优化, 无监督学习, 多模态推理, Bounding Box

## 一句话总结

提出UV-CoT框架，通过自动生成偏好数据和改进的Score-DPO损失函数，在不需要人工标注bounding box的情况下实现图像级链式思维（Visual CoT）推理，在6个基准上超越有监督的Visual-CoT方法。

## 研究背景与动机

CoT推理显著提升了MLLM的可解释性和问题求解能力，但现有方法集中于文本CoT，无法动态调整对输入图像不同空间区域的关注。唯一的Visual CoT工作（Visual-CoT）虽然引入了图像级推理，但存在两个关键缺陷：（1）依赖大规模人工标注bounding box数据，成本高且难以扩展；（2）基于SFT仅从正样本学习，泛化能力受限。

UV-CoT的核心动机是：能否不用任何人工标注，让模型自主学会"先看哪里、再推理"的能力？关键洞察在于——直接让MLLM生成精确坐标很困难，但让它在多个候选区域之间做排序（ranking）则简单得多。这将困难的坐标回归问题转化为了更可处理的偏好比较问题。

## 方法详解

### 整体框架

UV-CoT在推理时模拟人类感知过程：给定原图和问题，先通过CoT提示引导模型生成关键区域的bounding box坐标，然后通过视觉采样器裁剪该区域，最后综合原图和裁剪图的视觉token生成更精确的答案。训练包含两个核心阶段：偏好数据自动生成和基于Score-DPO的偏好优化。

### 关键设计

1. **自动偏好数据生成管线（Algorithm 1）**: 

    - **Response Generation**: 给定图像-问题对 $x$，目标模型 $f_{\text{tar}}$（LLaVA-1.5-7B）通过模板提示和随机解码，生成 $n$ 个不同的候选bounding box及对应回答 $\{y_t^i\}_{i=1}^n$。
    - **Response Evaluation**: 评估器模型 $f_{\text{eval}}$（OmniLMM-12B）为每个回答打分。关键创新在于引入累积评估：$s^i = s_{\text{cur}}^i + \gamma s_{\text{nxt}}^i$，其中 $s_{\text{nxt}}^i$ 衡量当前区域对后续推理步骤的影响，$\gamma$ 为超参数。
    - **Pair Construction**: 从 $n$ 个候选中随机选取 $k$ 个偏好对（preferred vs dis-preferred），每对包含完整推理链和对应分数 $\{y_w, s_w, y_l, s_l\}$。
    - **Response Selection**: 保留最高分回答作为下一步推理的上下文，形成"最优链"。

2. **Score-DPO（sDPO）损失函数**: 标准DPO仅排序偏好数据而不量化偏好强度。UV-CoT改进为引入分数间距的sDPO：
    $\mathcal{L}_{\text{sDPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - (g(s_w) - g(s_l))\right)\right]$
   其中 $g(\cdot)$ 为单调递增函数，将偏好分数映射到DPO目标的logit空间。基于Gumbel分布推导可知，$\Delta_r = g(s_w) - g(s_l)$ 量化偏好对之间的差异程度，使模型不仅区分偏好顺序，还优化偏好差异的幅度。

3. **迭代学习策略（Algorithm 2）**: 为避免标准DPO中静态偏好数据与模型演进分布不匹配的问题，将训练查询集 $\mathcal{X}$ 均分为 $m$ 个子集，迭代 $m$ 次。每次迭代用当前模型 $f_{\text{tar}}^i$ 在子集 $\mathcal{X}_i$ 上生成新偏好数据 $\mathcal{D}_i$，然后训练得到 $f_{\text{tar}}^{i+1}$。这确保偏好数据始终与模型当前能力匹配。实际用4次迭代，共249K偏好数据对（少于Visual-CoT的376K标注数据）。

### 损失函数 / 训练策略

使用AdamW优化器，每次迭代训练4个epoch，学习率 $5\times10^{-7}$，$\beta=0.1$，batch size为8。数据生成80小时，训练60小时，均在8×A100 40GB上完成。目标模型为LLaVA-1.5-7B，评估器为OmniLMM-12B。

## 实验关键数据

### 主实验

| 模型 | DocVQA | TextVQA | GQA | VSR | 平均 |
|------|--------|---------|-----|-----|------|
| LLaVA-1.5-7B | 0.198 | 0.507 | 0.480 | 0.504 | 0.393 |
| OmniLMM-12B(评估器) | 0.254 | 0.578 | 0.509 | 0.523 | 0.443 |
| Visual-CoT-7B(100%标注) | 0.294 | 0.673 | 0.546 | 0.532 | 0.482 |
| **UV-CoT(0%标注)** | 0.265 | **0.686** | 0.536 | **0.548** | 0.473 |
| **UV-CoT(10%标注)** | 0.283 | **0.711** | **0.568** | **0.553** | **0.494** |

| 零样本数据集 | Visual-CoT | UV-CoT | UV-CoT* | 说明 |
|-----------|----------|--------|---------|------|
| DUDE | 0.206 | 0.241 | 0.253 | 文档理解 |
| Visual7w | 0.397 | 0.432 | 0.455 | 通用VQA |
| V*Bench OCR | 0.593 | **0.677** | - | 高分辨率OCR |
| V*Bench Avg | 0.347 | **0.402** | - | 高分辨率推理 |

### 消融实验

| 配置 | 平均精度 | 说明 |
|------|---------|------|
| UV-CoT(10%标注) | **0.494** | 完整模型 |
| 无UV-CoT推理 | 0.417 | 移除CoT直接回答，-7.7% |
| UV-CoT用GT BBox | 0.618 | 上限参考，+12.4% |
| 用标准DPO | 0.475 | -1.9%，无法定量偏好强度 |
| 无迭代学习 | 0.459 | -3.5%，静态数据分布不匹配 |
| 无$\gamma$（不考虑下一步影响） | 0.406 | -8.8%，MLLM难以直接评估BBox质量 |

### 关键发现

- UV-CoT超越了其评估器OmniLMM-12B平均5.1%，说明这不是简单的模型蒸馏
- 仅用10%标注数据的UV-CoT就超过了使用100%标注的Visual-CoT（0.494 vs 0.482）
- 在V*Bench高分辨率图像推理上，Visual CoT方法的优势最为显著（相比非CoT基线提升>50%OCR性能），而UV-CoT进一步超过Visual-CoT 5.5%
- $\gamma$参数的消融表明"考虑区域对后续推理的影响"至关重要（-8.8%），MLLM无法直接可靠地评估bounding box质量

## 亮点与洞察

- 将Visual CoT问题巧妙转化为偏好排序问题，规避了MLLM坐标生成不精确的瓶颈
- sDPO损失的数学推导基于Gumbel分布，理论上比标准DPO更适合刻画连续偏好差异
- 迭代学习+自动数据生成构成了"自我改进循环"，体现了online learning的精神
- 评估时考虑当前+下一步影响的设计（$\gamma$参数）类似于强化学习中的时序差分思想

## 局限与展望

- 当前每步仅生成一个bounding box，多步推理的链长度和分支数受限
- 数据生成需要80小时+训练60小时，效率可进一步提升
- 在DocVQA和InfographicsVQA上与GT BBox的差距仍较大，精确定位仍有提升空间
- 仅在7B规模模型上验证，对更大模型是否同样有效尚不确定

## 相关工作与启发

- 与RLHF/DPO的文本CoT优化工作互补——UV-CoT专注于图像级决策
- 自动偏好数据生成管线可推广到其他需要空间推理的视觉任务
- sDPO的margin设计思想可应用到其他需要量化偏好强度的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 无监督Visual CoT的首次成功尝试，sDPO理论推导扎实
- 实验充分度: ⭐⭐⭐⭐ 10个数据集覆盖多任务，消融完善，零样本验证说服力强
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰，图例信息丰富
- 价值: ⭐⭐⭐⭐⭐ 无需标注即可超越有监督方法，实用性和可扩展性优秀

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Self-Consistency Preference Optimization](../../ICML2025/llm_reasoning/self-consistency_preference_optimization.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ACL 2025\] RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought](../../ACL2025/llm_reasoning/rsvp_reasoning_segmentation_via_visual_prompting_and_multi-modal_chain-of-though.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ACL 2025\] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](../../ACL2025/llm_reasoning/self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)

</div>

<!-- RELATED:END -->
