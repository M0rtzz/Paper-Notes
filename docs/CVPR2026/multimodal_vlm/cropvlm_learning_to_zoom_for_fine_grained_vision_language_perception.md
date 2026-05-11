---
title: >-
  [论文解读] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception
description: >-
  [CVPR 2026][多模态][视觉裁剪] 提出CropVLM——一个256M参数的轻量裁剪网络，通过GRPO强化学习训练（无需人工标注边界框），动态选择图像最有信息量的区域供VLM聚焦，可与开源和商用VLM即插即用地提升细粒度视觉理解性能。
tags:
  - CVPR 2026
  - 多模态
  - 视觉裁剪
  - 多模态VLM
  - GRPO
  - 细粒度感知
  - 即插即用
---

# CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception

**会议**: CVPR 2026  
**arXiv**: [2511.19820](https://arxiv.org/abs/2511.19820)  
**代码**: [GitHub](https://github.com/miguelscarv/cropvlm)  
**领域**: 多模态VLM  
**关键词**: 视觉裁剪, 强化学习, GRPO, 细粒度感知, 即插即用

## 一句话总结

提出CropVLM——一个256M参数的轻量裁剪网络，通过GRPO强化学习训练（无需人工标注边界框），动态选择图像最有信息量的区域供VLM聚焦，可与开源和商用VLM即插即用地提升细粒度视觉理解性能。

## 研究背景与动机

VLM在需要细粒度视觉感知的任务（文档分析、场景文字识别等）中受限于输入分辨率——LLaVA-1.5的336×336分辨率无法分辨小文字。均匀提高分辨率计算代价巨大且不必要（研究表明大多数请求只需少量image token即可回答）。

现有方法的局限：
- 架构修改（如Matryoshka、S2）需要大量重训练，有灾难性遗忘风险
- 不适用于商用模型（权重不可访问）
- ViCrop等无训练方法依赖注意力图/梯度，分布外泛化差
- UV-CoT使用DPO训练，需要合成偏好对，数据效率低

CropVLM的独特定位：轻量外挂模块，GRPO训练无需人工bbox，兼容开源/商用VLM。

## 方法详解

### 整体框架

输入图像 + 问题 → CropVLM（SmolVLM 256M）生成边界框坐标 → 裁剪原图相应区域 → 原图 + 裁剪图一起送入目标VLM → 生成答案。

### 关键设计

1. **基于GRPO的裁剪训练**:
    - 功能：无需GT边界框，直接优化裁剪对下游VLM性能的贡献
    - 核心思路：对每个图像-问题对生成G=6个候选边界框，每个裁剪后与原图一起送入奖励VLM评估质量，通过组内标准化得到相对优势函数
    - 设计动机：GT边界框标注昂贵且往往不是最优的（人标注不一定最有助于模型回答）

2. **双奖励设计**:
    - 功能：提供学习信号指导裁剪质量
    - 核心思路：准确率奖励（VLM用原图+裁剪图回答后与GT对比）和对数似然奖励（VLM对正确答案的log-likelihood，无需生成，单次前向传播更快）
    - 设计动机：似然奖励更细粒度（几乎消除组内奖励相同的情况），使更多样本有效参与权重更新

3. **SFT种子初始化**:
    - 功能：赋予模型生成有效边界框格式的基本能力
    - 核心思路：用Qwen 2.5-VL 7B生成合成边界框数据集进行SFT，小面积bbox按百分位扩展
    - 设计动机：SmolVLM原始不支持bbox格式输出，需先建立基本能力再RL优化

### 损失函数 / 训练策略

- 两阶段：SFT（学习bbox格式）→ GRPO（优化裁剪质量）
- 所有训练在单张A100 GPU上完成，SFT约3小时，GRPO约24小时（2048px版本）
- 使用LoRA（rank 128, alpha 256）微调SmolVLM

## 实验关键数据

### 主实验（搭配不同VLM）

| 目标VLM | 无CropVLM | +CropVLM(2048) | 平均提升 |
|---------|-----------|----------------|----------|
| LLaVA 1.5 (336px) | 36.69 | 42.71 | +6.02 |
| Qwen 2.5 VL (448px) | 56.42 | 67.14 | +10.72 |
| GPT 4.1 nano (512px) | 41.27 | 47.41 | +6.14 |

### 对比其他裁剪方法

| 方法 | TextVQA | DocVQA | V* | HR-8k | 平均 |
|------|---------|--------|-----|-------|------|
| ViCrop (Qwen) | 74.15 | 72.27 | 53.40 | 46.00 | 59.67 |
| UV-CoT (Qwen) | 74.56 | 76.60 | 56.54 | 47.25 | 60.64 |
| CropVLM (Qwen) | 75.72 | 84.41 | 59.69 | 60.75 | 67.14 |

### 消融实验

| 配置 | 1024px平均 | 说明 |
|------|-----------|------|
| 基线SmolVLM | 44.55 | 无裁剪 |
| + SFT | 46.55 | 合成bbox训练 |
| + GRPO (准确率) | 49.75 | RL优化 |
| + GRPO (似然) | 50.89 | 似然奖励更优 |

### 关键发现

- CropVLM(1024px)搭配SmolVLM的性能超过基线SmolVLM(2048px)——低分辨率+智能裁剪优于高分辨率暴力处理
- 在分布外基准（V*、HR-Bench）上也有显著提升，说明裁剪策略泛化性好
- GPT 4.1 nano搭配CropVLM后拒绝回答的问题从31/191降至2/191
- 似然奖励一致优于准确率奖励

## 亮点与洞察

- 即插即用设计：无需修改目标VLM权重，甚至可用于商用API模型
- 极低成本：256M参数裁剪网络+单GPU训练，但提升显著
- GRPO训练的优雅之处：不需要GT bbox，不需要额外评估器模型，直接用下游性能作为奖励
- 证明了"裁剪"这个简单操作在VLM细粒度理解中的巨大价值

## 局限与展望

- 仅支持单次裁剪，多区域或多步推理未探索
- SmolVLM的数字输出词汇表受限（只有0-9数字），生成bbox坐标较慢
- 训练资源保守（单GPU、小group size），可能是性能下界
- 裁剪网络输入分辨率固定，未探索自适应分辨率策略

## 相关工作与启发

- **vs ViCrop**: 无训练方法依赖注意力图/梯度，分布外性能差；CropVLM学到的策略更鲁棒
- **vs UV-CoT**: DPO训练需249k偏好对+7B模型；CropVLM仅需62k数据+256M模型，更高效
- **vs DeepEyes/Mini-o3**: 多轮推理开销大；CropVLM单次裁剪即可，推理效率高

## 评分

- 新颖性: ⭐⭐⭐⭐ GRPO裁剪训练+即插即用设计在该领域新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多VLM、多基准、多方法对比、开销分析全面
- 写作质量: ⭐⭐⭐⭐ 方法简洁清晰，实验呈现规范
- 价值: ⭐⭐⭐⭐ 实用性极强的即插即用方案，低成本高回报

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models](oddgridbench_exposing_the_lack_of_fine-grained_visual_discrepancy_sensitivity_in.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)

</div>

<!-- RELATED:END -->
