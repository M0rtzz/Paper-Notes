---
title: >-
  [论文解读] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning
description: >-
  [ECCV 2024][图像生成] 提出第一人称视角动作帧生成新问题，通过视觉指令微调 VLLM 生成丰富动作描述并将其嵌入作为扩散模型的额外条件，实现高质量的自我中心动作图像合成。
tags:
  - ECCV 2024
  - 图像生成
---

# LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

**会议**: ECCV 2024  
**arXiv**: [2312.03849](https://arxiv.org/abs/2312.03849)  
**领域**: 图像生成

## 一句话总结

提出第一人称视角动作帧生成新问题，通过视觉指令微调 VLLM 生成丰富动作描述并将其嵌入作为扩散模型的额外条件，实现高质量的自我中心动作图像合成。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：在技能传递场景中，LLM 生成的文字指导不够直观，而人脑处理图像远快于文字。本文提出 **自我中心动作帧生成** 新问题：给定第一人称图像和动作文本查询，合成展示该动作执行过程的图像。

面临两大挑战：
1. 现有自我中心数据集动作标注过于简单（仅动词+名词），缺乏详细描述
2. 现有扩散模型预训练于第三人称图像，对第一人称视角的动作状态转换能力有限（存在域差距）

## 方法详解

### 整体框架

LEGO 模型由两阶段组成：
1. **视觉指令微调阶段**：微调 VLLM（基于 LLaVA）生成丰富的动作描述
2. **动作帧生成阶段**：利用潜在扩散模型（LDM），以 VLLM 嵌入为额外条件合成动作帧

### 关键设计

**Prompt Enhancement**：利用 GPT-3.5 进行 in-context learning，生成详细动作描述作为 VLLM 视觉指令微调的训练数据。微调后的 VLLM 可在推理时大规模生成丰富动作描述，无需边界框输入。

**VLLM 嵌入注入**：将微调后 VLLM 的图像嵌入 $\mathcal{H}_i$ 和文本嵌入 $\mathcal{H}_t$ 分别通过投影层映射到 LDM 特征空间，与 CLIP 文本编码拼接作为 U-Net 的条件输入：

$$\mathcal{C} = [\psi(\mathcal{R}), \sigma(\mathcal{H}_i), \pi(\mu(\mathcal{H}_t))] \in \mathbb{R}^{(2N+M) \times D}$$

文本嵌入还加了自注意力层 $\pi$ 以获取整体语义。条件通过交叉注意力注入 U-Net 各层。

### 损失函数

- VLLM 微调：交叉熵损失，训练 3 个 epoch
- LDM 训练：L2 噪声预测损失，训练 20000 步
- 图像分辨率 256×256，采用 classifier-free guidance

## 实验关键数据

### 主实验

图像到图像指标对比（Ego4D 数据集）：

| 方法 | EgoVLP ↑ | EgoVLP⁺ ↑ | CLIP ↑ | FID ↓ | PSNR ↑ | LPIPS ↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| ProxEdit | 44.51 | 72.68 | 68.17 | 33.01 | 11.88 | 40.90 |
| SDEdit | 50.07 | 72.90 | 73.35 | 33.35 | 11.81 | 41.60 |
| InstructPix2Pix | 62.19 | 78.84 | 78.75 | 24.73 | 12.16 | 37.16 |
| **LEGO** | **65.65** | **80.44** | **80.61** | **23.83** | **12.29** | **36.43** |

LEGO 在全部 6 个指标上均超过最强基线 IP2P。用户研究中 LEGO 胜率 52%，远超 IP2P 的 8%。

### 消融实验

不同条件组合对模型性能的影响（Ego4D, 用户研究 Win Rate）：

| 条件设置 | 用户研究 | EgoVLP ↑ | EgoVLP⁺ ↑ | CLIP ↑ |
|----------|:-:|:-:|:-:|:-:|
| 原始动作标签 | 5.33 | 62.19 | 78.84 | 78.75 |
| 丰富描述 | 13.00 | 62.91 | 79.09 | 79.18 |
| 描述+图像嵌入 | 26.00 | 65.35 | 80.13 | 80.57 |
| 描述+文本嵌入 | 21.33 | 63.29 | 79.40 | 79.21 |
| **描述+联合嵌入** | **34.34** | **65.65** | **80.44** | **80.61** |

微调后 VLLM 嵌入比未微调的嵌入带来更大提升（EgoVLP +1.08%），验证了视觉指令微调对缩小域差距的必要性。VLLM 生成的描述与帧的对齐率达 87%（未微调仅 27%）。

### 关键发现

- 图像嵌入比文本嵌入贡献更大，说明 VLLM 图像嵌入包含自编码器无法捕获的高层语义
- 丰富描述可以适度提升性能，但 VLLM 嵌入的提升更显著
- 模型可对同一输入帧生成不同动作的帧（泛化能力好）

## 亮点与洞察

- 首创自我中心动作帧生成问题，有 AR/VR 应用潜力
- VLLM 嵌入作为扩散模型条件的新颖设计，有效缩小域差距
- 端到端的数据增强管线（GPT-3.5 → 指令微调 → 大规模描述生成）可复用
- 同一帧不同动作的泛化实验展示了模型对动作语义的理解能力

## 补充分析

**数据处理的工程细节**值得关注：
- 使用美学评分筛选输入帧和目标帧，避免模糊图像
- 基于 CLIP 相似度过滤相似度过低（<0.81）或过高（>0.97）的样本对
- 最终 Ego4D: 85521/9931 训练/测试，Epic-Kitchens: 61841/8893 训练/测试

**指令微调的对齐率差异显著**：微调后 VLLM 的描述与帧对齐率为 87%（Ego4D）/ 84%（Epic-Kitchens），而未微调仅 27%/30%，且 92% 不对齐案例存在幻觉——说明通用 VLLM 在自我中心领域确实存在严重域差距，微调不可或缺。

## 局限与展望

- 生成分辨率仅 256×256，高分辨率场景适用性有限
- 依赖数据效果评估，自动指标在自我中心领域仍存在域差距
- 高度动态场景（频繁头部运动）需要数据过滤，限制了训练样本量
- GPT-3.5 数据策管的质量依赖手动编写的 few-shot 示例

## 评分

⭐⭐⭐⭐ 问题定义新颖，方法设计合理，实验详尽，但实际应用场景有限

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [\[ECCV 2024\] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)
- [\[ECCV 2024\] WebRPG: Automatic Web Rendering Parameters Generation for Visual Presentation](webrpg_automatic_web_rendering_parameters_generation_for_visual_presentation.md)
- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)

<!-- RELATED:END -->
