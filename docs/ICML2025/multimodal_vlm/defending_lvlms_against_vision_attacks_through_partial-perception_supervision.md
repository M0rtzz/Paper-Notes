---
title: >-
  [论文解读] Defending LVLMs Against Vision Attacks through Partial-Perception Supervision
description: >-
  [ICML2025][多模态][LVLM安全] 提出 DPS（Defense through Partial-Perception Supervision），利用裁剪图像的响应作为"弱监督"来引导全图模型在推理时自我修正，实现无需训练的黑盒 LVLM 视觉攻击防御，平均攻击成功率降低 76.3%。
tags:
  - ICML2025
  - 多模态
  - LVLM安全
  - 视觉对抗攻击
  - 部分感知监督
  - 弱到强学习
  - 黑盒防御
  - 越狱防御
---

# Defending LVLMs Against Vision Attacks through Partial-Perception Supervision

**会议**: ICML2025  
**arXiv**: [2412.12722](https://arxiv.org/abs/2412.12722)  
**代码**: [GitHub](https://github.com/tools-only/DPS)  
**领域**: LVLM安全 / 对抗防御  
**关键词**: LVLM安全, 视觉对抗攻击, 部分感知监督, 弱到强学习, 黑盒防御, 越狱防御

## 一句话总结

提出 DPS（Defense through Partial-Perception Supervision），利用裁剪图像的响应作为"弱监督"来引导全图模型在推理时自我修正，实现无需训练的黑盒 LVLM 视觉攻击防御，平均攻击成功率降低 76.3%。

## 研究背景与动机

- **问题**: LVLM（如 GPT-4o、Gemini）易受视觉对抗攻击（对抗噪声、排版攻击）误导，输出错误或有害内容
- **现有方案不足**: SmoothVLM 等方法通过裁剪图像 + 多数投票防御，但裁剪破坏语义，导致正常图像上回答质量显著下降
- **核心矛盾**: 裁剪可破坏攻击信号，但也破坏正常图像语义——如何兼得？
- **关键观察**:
    - **敏感性**: 视觉攻击对裁剪等图像修改非常敏感，裁剪后攻击语义被破坏
    - **置信度差异**: 模型处理干净图像时置信度高、不受干扰项影响；处理被攻击图像时置信度低、易被 prompt 中的干扰信息左右
- **灵感来源**: 弱到强学习（weak-to-strong learning）——弱模型（看部分图像的模型）可以有效监督和引导强模型（看全图的模型）

## 方法详解

### DPS 框架（两步推理）

**Step 1: 部分感知初始响应（Part-Perc Model）**

对裁剪后的图像生成客观描述：

> "Please provide an objective, detailed description of the image, avoiding subjective conjecture and associations. Then answer the question: (Original Question)."

采用三种裁剪策略生成 3 份部分图像：

- **中心裁剪 (CC)**: 从图像中心提取 1/2 大小的区域
- **随机裁剪 (RC)**: 从随机位置提取 1/4~1/2 大小区域
- **自适应裁剪 (AC)**: 利用 LVLM 提取图像中的主要物体区域

**Step 2: 部分感知监督（Full-Perc Model）**

将 Part-Perc 模型的响应作为监督信息，引导 Full-Perc 模型重新分析全图：

> "Here is the information provided by the local observation agents: (Supervisory message). Re-analyze the given image, and provide your final answer to the question: (Original Question)."

**核心机制**:

- 干净图像 → 模型置信度高 → Part-Perc 的（可能不准确的）描述不影响最终输出
- 被攻击图像 → 模型置信度低 → Part-Perc 的（去除攻击信号的）描述促使模型自我修正

### 安全增强：LS-DPS

针对越狱攻击，在 Step 2 的 prompt 中加入安全提醒：

> "Consider whether you might be led into discussing harmful, malicious, or unethical topics."

并外接一个 LLM 安全检查器（LLM-Secured DPS），对最终输出进行过滤：

$$\text{ASR}(\mathcal{D}_k) = \frac{1}{|\mathcal{D}_k|} \sum_{(x_i, q_i, t_i) \in \mathcal{D}_k} \mathbb{I}(\mathcal{F}(x_i, q_i), t_i)$$

其中 $\mathcal{F}$ 为 LVLM，$\mathbb{I}$ 为攻击是否成功的指示函数，$t_i$ 为攻击目标/安全标准。

## 实验关键数据

### 误导防御（ASR ↓，越低越好）

| 模型 | 方法 | RTA-100 | Self-Gen | MultiTrust | 平均 |
|------|------|---------|----------|------------|------|
| Qwen-VL-Plus | SmoothVLM | 0.92 | 0.83 | 1.00 | 0.91 |
| Qwen-VL-Plus | **DPS** | **0.24** | **0.30** | **0.40** | **0.31** |
| GPT-4o-Mini | SmoothVLM | 0.68 | 0.85 | - | 0.76 |
| GPT-4o-Mini | **DPS** | **0.35** | **0.43** | - | **0.39** |
| Gemini-1.5-Flash | SmoothVLM | 0.85 | 1.00 | 0.80 | 0.88 |
| Gemini-1.5-Flash | **DPS** | **0.58** | **0.49** | **0.11** | **0.39** |

DPS 在误导防御上ASR 平均降至 0.31~0.39，是最佳 baseline 的 **2~2.5 倍防御效果**。

### 越狱防御（ASR ↓）

| 模型 | 方法 | MM-Safety | HADES | VisualAtt | 平均 |
|------|------|-----------|-------|-----------|------|
| Qwen-VL-Plus | Protector | 0.07 | 0.22 | 0.18 | 0.16 |
| Qwen-VL-Plus | **LS-DPS** | **0.02** | **0.10** | **0.02** | **0.05** |
| GPT-4o-Mini | ECSO | 0.24 | 0.05 | 0.15 | 0.15 |
| GPT-4o-Mini | **LS-DPS** | **0.03** | **0.04** | **0.04** | **0.04** |
| Gemini-1.5-Flash | ECSO | 0.14 | 0.11 | 0.13 | 0.13 |
| Gemini-1.5-Flash | **LS-DPS** | **0.06** | **0.03** | **0.06** | **0.05** |

LS-DPS 在越狱防御上 ASR 降至 0.04~0.05，远超所有 baseline。

### 标准性能（MM-Vet）

DPS 对标准性能影响极小，与 vanilla 接近；而 SmoothVLM 有明显性能下降。

### 消融实验（裁剪策略对比，Qwen-VL-Plus）

| 策略 | 误导任务 (Avg) | 安全任务 (Avg) |
|------|---------------|---------------|
| CC（中心裁剪） | ~0.40 | ~0.08 |
| RC（随机裁剪） | ~0.48 | ~0.09 |
| AC（自适应裁剪） | ~0.37 | ~0.07 |
| DPS（三者融合） | **~0.31** | **~0.05** |

多种裁剪策略融合显著提升防御能力。

## 亮点与洞察

1. **新颖的弱到强防御范式**: 将"看部分图/看全图"类比为"弱模型/强模型"，用弱监督引导强模型自我修正，而非简单投票
2. **完全黑盒 + 免训练**: 不需要模型内部权重访问，不需要额外训练，仅通过 prompt 层面操作即可防御
3. **利用置信度差异**: 发现模型对干净图像高置信、对攻击图像低置信的特性，是方法能同时保持性能和防御效果的理论基础
4. **兼顾误导攻击和越狱攻击**: 通过 prompt 微调和 LLM safety checker 的即插即用扩展，一个框架覆盖两类主流攻击场景
5. **标准性能无损**: 与 SmoothVLM 的显著性能下降相比，DPS 几乎不影响正常任务表现

## 局限与展望

1. **计算开销**: 需要多次裁剪 + 多次模型推理（至少 4 次 LVLM 调用），相比单次推理开销显著增加
2. **裁剪失效场景**: 若攻击信息分布在全图（非局部），裁剪可能无法消除攻击信号，部分感知监督将失效
3. **依赖置信度差异假设**: 方法基于"攻击使模型置信度下降"的观察，若攻击方式使模型在被攻击时仍保持高置信度，方法可能失效
4. **仅在 API 模型上验证**: 主要实验在 Qwen-VL-Plus、GPT-4o-Mini、Gemini-1.5-Flash 上，开源模型仅补充了 Qwen2.5-VL-32B
5. **交互策略有优化空间**: 当前使用简单的两步 prompt，更复杂的交互机制（如多轮辩论）可能进一步提升效果

## 评分

- 新颖性: ⭐⭐⭐⭐ — 弱到强监督范式应用于视觉攻击防御是全新视角
- 实验充分度: ⭐⭐⭐⭐ — 三个模型 × 六个数据集 × 七个baseline，覆盖误导和越狱两类攻击
- 写作质量: ⭐⭐⭐⭐ — 动机分析（Section 3）层层递进，逻辑清晰
- 价值: ⭐⭐⭐⭐ — 实用性强（黑盒免训练），但计算开销限制了部署场景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](../../CVPR2025/multimodal_vlm/steering_away_from_harm_an_adaptive_approach_to_defending_vision_language_model_.md)
- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [\[ICML 2025\] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration](towards_rationale-answer_alignment_of_lvlms_via_self-rationale_calibration.md)
- [\[ICML 2025\] CoMemo: LVLMs Need Image Context with Image Memory](comemo_lvlms_need_image_context_with_image_memory.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/multimodal_vlm/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)

<!-- RELATED:END -->
