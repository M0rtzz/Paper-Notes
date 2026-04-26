---
title: >-
  [论文解读] HPSv3: Towards Wide-Spectrum Human Preference Score
description: >-
  [ICCV 2025][图像生成][人类偏好评分] HPSv3 构建了首个宽谱人类偏好数据集 HPDv3（1.08M 图文对、1.17M 标注对），采用 VLM 骨干（Qwen2-VL）+ 不确定性感知排序损失训练偏好模型，并提出 CoHP 链式思维迭代生成方法，显著提升图像生成评估的准确性和覆盖范围。
tags:
  - ICCV 2025
  - 图像生成
  - 人类偏好评分
  - 图像质量评估
  - 不确定性感知排序
  - VLM
  - 图像生成评价指标
---

# HPSv3: Towards Wide-Spectrum Human Preference Score

**会议**: ICCV 2025

**arXiv**: [2508.03789](https://arxiv.org/abs/2508.03789)

**领域**: 图像生成/人类偏好评估

**关键词**: 人类偏好评分, 图像质量评估, 不确定性感知排序, VLM, 图像生成评价指标

## 一句话总结

HPSv3 构建了首个宽谱人类偏好数据集 HPDv3（1.08M 图文对、1.17M 标注对），采用 VLM 骨干（Qwen2-VL）+ 不确定性感知排序损失训练偏好模型，并提出 CoHP 链式思维迭代生成方法，显著提升图像生成评估的准确性和覆盖范围。

## 研究背景与动机

- **数据覆盖不足**：现有人类偏好数据集（HPDv2、Pick-a-Pic、ImageReward）主要包含 Stable Diffusion 系列模型输出，无法评估更先进的扩散 Transformer（FLUX）和自回归模型（Infinity）。缺乏高质量真实照片作为质量上界参考
- **特征提取不够强**：HPSv2、PickScore 等使用 CLIP 作为骨干，BLIP 用于 ImageReward，但这些编码器在多模态特征提取能力上不如最新的 VLM
- **训练策略粗糙**：直接使用 KL 散度或简单排序损失**未考虑标注中的不确定性和不一致性**，对困难样本容易引入偏差
- **缺乏高质量真实图片对比**：此前数据集缺少真实摄影照片与 AI 生成图片的对比，无法建立完整的质量谱

## 方法详解

### 整体框架

HPSv3 包含三个部分：(1) HPDv3 宽谱人类偏好数据集构建；(2) 基于 VLM 的不确定性感知偏好模型；(3) CoHP 链式偏好迭代图像生成。

### 关键设计

**1. HPDv3 数据集构建**

数据来源三部分：

- **扩展 HPDv2**：保留原有 103,700 文本提示，使用 10+ 最新模型（FLUX.1-dev、Infinity、Hunyuan、Kolors、SD3 等）重新生成图像
- **基于真实照片描述生成**：从互联网收集高质量摄影照片 -> 分类为 12 个类别 -> 按分布对齐 JourneyDB prompt 分布 -> 美学过滤取 top 10% -> VLM 生成描述 -> 各模型生成对应图像。最终获得 57,759 张高质量真实图
- **Midjourney 数据**：收集 331,955 张用户生成图 + Discord 平台上的用户真实偏好选择

数据规模：**1.08M 图文对 + 1.17M 标注对比**，覆盖 GAN、扩散、自回归 + 高/低质量真实图。

标注质量控制：
- 标注员需通过 600 对验证集（20 名专业艺术家标注，80% 收敛率），至少正确评估 16/20 对
- 每对图像由 9-19 名标注员评估，**平均收敛率 76.5%**（HPDv2 仅 59.9%）
- 超过 95% 置信度的对用于训练

**2. HPSv3 偏好模型**

骨干选择：使用 **Qwen2-VL** 作为视觉语言模型提取图文联合特征，替代 CLIP/BLIP。

不确定性感知排序损失：传统方法预测确定性分数 r，偏好概率为 sigmoid(r1 - r2)。HPSv3 将分数建模为高斯分布 r ~ N(mu, sigma)，引入预测不确定性。MLP 最后两层分别预测 mu 和 sigma。最终偏好概率通过对高斯分布积分得到。这让模型能区分"确定性高的偏好"和"标注有分歧的困难样本"，避免对后者过度自信。

**3. CoHP：链式偏好图像生成**

两阶段迭代生成流程：

- **Model-wise Preference**：给定 prompt，M 个候选模型各生成 N 轮图像，HPSv3 评分选出最佳模型
- **Sample-wise Preference**：选定模型生成 B 张图 -> HPSv3 评分选最佳 -> 将最佳图与噪声混合作为下一轮输入条件 -> 迭代 S 轮 -> 选全局最高分图像

### 损失函数

HPSv3 训练使用不确定性感知排序的负对数似然损失，MLP 最后两层分别预测均值和标准差。

## 实验关键数据

### HPDv3 Benchmark 生成模型排名（HPSv3 评分）

| 模型 | 综合评分 |
|------|---------|
| Kolors | 10.55 |
| FLUX-dev | 10.43 |
| Playground-v2.5 | 10.27 |
| Infinity | 10.26 |
| CogView4 | 9.61 |
| PixArt-Sigma | 9.37 |
| Gemini 2.0 Flash | 9.21 |
| SDXL | 8.20 |
| Hunyuan | 8.19 |
| SD3 | 5.31 |
| SD v2.0 | -0.24 |

### 数据集对比

| 数据集 | 图片数 | 对比数 | 覆盖模型类型 | 含真实图 | 收敛率 |
|--------|--------|--------|------------|---------|--------|
| HPDv2 | 458K | 798K | GAN+Diff+AR | 无(HQI) | 59.9% |
| Pick-a-Pic | 638K | 584K | Diff | 无 | - |
| MHP | 608K | 918K | GAN+Diff+AR | 无 | - |
| **HPDv3** | **1.08M** | **1.17M** | **全部** | **有** | **76.5%** |

### 关键发现

- HPSv3 使用 VLM 骨干显著优于 CLIP/BLIP 骨干
- 不确定性感知排序对困难样本（标注分歧大）表现更鲁棒
- CoHP 迭代生成在无需额外训练数据的情况下提升图像质量
- Kolors 和 FLUX-dev 在 HPSv3 综合评分中位居前列
- 分类别评估显示不同模型在不同类别有各自优势（如 FLUX 在建筑、交通工具类别更强）

## 亮点与洞察

1. **"宽谱"理念**：首次系统性地在同一评估框架下纳入 GAN、扩散模型、自回归模型和高质量真实照片，建立了从最差到最优的完整质量谱
2. **VLM 替代 CLIP**：使用 Qwen2-VL 作为特征提取器是自然但有效的升级，充分利用了 VLM 更强的多模态理解能力
3. **不确定性建模**：将偏好分数从点估计扩展为高斯分布，对人类标注天然存在的主观性和不一致性，这种建模更加合理
4. **CoHP 免训练提升**：通过 HPSv3 作为 reward model 指导迭代采样，核心是 best-of-N 策略 + 图像到图像的迭代精炼
5. **数据集的工程价值**：HPDv3 本身作为高质量标注的大规模偏好数据集，对社区有重要的基础设施价值
6. **严格的标注质量控制**：标注员准入测试 + 多人交叉标注 + 95% 置信度过滤，远超前作

## 局限性

- VLM 骨干（Qwen2-VL）参数量大，推理速度和部署成本显著高于 CLIP-based 方法
- CoHP 需要多轮生成和评分，推理效率较低
- 主观偏好的文化差异和个体差异未被显式建模
- Midjourney 数据的用户偏好标签质量可能低于专业标注
- 数据集以英文 prompt 为主，多语言泛化能力未经验证

## 相关工作

- **HPSv2** [Wu et al., 2023]：前作，CLIP 骨干 + HPDv2 数据集
- **PickScore** [Kirstain et al., 2023]：CLIP 微调的偏好模型
- **ImageReward** [Xu et al., 2023]：BLIP 骨干的 reward 模型
- **MPS** [Zhang et al., 2024]：CVPR 2024，考虑多样性的人类偏好评分
- **Qwen2-VL** [Wang et al., 2024]：视觉语言模型，HPSv3 的骨干

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | 4/5 |
| 有效性 | 4/5 |
| 实用性 | 5/5 |
| 清晰度 | 4/5 |
| 综合 | 4/5 |

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[ECCV 2024\] Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis](../../ECCV2024/image_generation/stable_preference_redefining_training_paradigm_of_human_preference_model_for_tex.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] CompleteMe: Reference-based Human Image Completion](completeme_reference-based_human_image_completion.md)

<!-- RELATED:END -->
