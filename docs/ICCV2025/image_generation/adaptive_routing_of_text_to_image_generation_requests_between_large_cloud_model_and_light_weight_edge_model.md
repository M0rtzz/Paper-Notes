---
title: >-
  [论文解读] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model
description: >-
  [ICCV 2025][图像生成][text-to-image routing] 提出RouteT2I框架，通过多维质量评估指标和双门控token选择MoE路由模型，动态将文本到图像生成请求分配到边缘轻量模型或云端大模型，在50%路由率下实现云端全用83.97%的质量提升。 大型文本到图像模型（如Stable Diffusi…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "text-to-image routing"
  - "edge-cloud collaboration"
  - "mixture-of-experts"
  - "Pareto relative superiority"
  - "dual-gate MoE"
---

# Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 文本到图像生成 / 模型路由  
**关键词**: text-to-image routing, edge-cloud collaboration, mixture-of-experts, Pareto relative superiority, dual-gate MoE

## 一句话总结

提出RouteT2I框架，通过多维质量评估指标和双门控token选择MoE路由模型，动态将文本到图像生成请求分配到边缘轻量模型或云端大模型，在50%路由率下实现云端全用83.97%的质量提升。

## 研究背景与动机

大型文本到图像模型（如Stable Diffusion 3.5的80亿参数）生成质量出色但依赖昂贵的云服务器，而轻量级边缘模型成本低但复杂提示下质量不足。关键洞察在于：并非所有用户提示都需要大模型处理——简单提示下轻量模型可产生媲美甚至优于大模型的结果。然而，现有路由方法主要针对LLM设计，直接迁移到T2I场景面临两大挑战：(1) 图像质量缺乏统一评估标准，具有多维性和主观性；(2) T2I的输出空间远大于输入文本空间，从文本预测图像质量更加困难。

## 方法详解

### 整体框架

RouteT2I包含三个核心组件：(1) 多维质量评估体系——通过正负文本对和CLIP相似度衡量生成图像在定义、细节、清晰度等10个维度的质量；(2) 路由模型——基于双门控token选择MoE的Transformer网络，从用户提示预测边缘与云端生成图像的质量差异（Pareto相对优势度PRS）；(3) 路由策略——基于PRS阈值在成本约束下将提示分配到边缘或云端。

### 关键设计

1. **多维对比质量度量与PRS**: 定义10个图像质量维度（清晰度、细节、色彩、一致性、物体完整性等），每个维度通过正/负文本提示对与CLIP相似度的sigmoid差值衡量。引入帕累托相对优势度（PRS）量化边缘/云端图像的多维质量差距，PRS > 0.5表示边缘更优，< 0.5表示云端更优。通过质量距离归一化和温度参数调节分布。

2. **双门控Token选择MoE**: 将用户提示视为token序列，每个expert对应一个质量维度。Token选择门控通过token-expert亲和力矩阵选择对各维度最相关的Top-K token。正/负双门控分别评估token对质量的正面和负面影响，通过对比两个门控输出判断token的主导影响方向。将原始线性层分解为正/负投影矩阵和共享评分矩阵，减少参数量。

3. **多头质量预测与路由策略**: 模型包含多个预测头，分别输出各质量维度的预测值。路由策略基于PRS阈值α（上界为0.5）确定路由决策：PRS低于阈值的提示路由到云端，其余留在边缘，通过调整阈值控制路由率以满足成本约束。

### 损失函数 / 训练策略

使用Adam优化器，学习率2e-5，batch size 16，在NVIDIA 4090D上训练约10个epoch。监督信号为预计算的真实PRS值（由边缘和云端模型分别生成图像后计算得到）。

## 实验关键数据

### 主实验

| 方法 | 定义 | 细节 | 完整性 | ΔP(%) |
|------|------|------|--------|-------|
| 仅边缘模型 | 0.6251 | 0.6685 | 0.4690 | - |
| 仅云端模型 | 0.6337 | 0.6847 | 0.4972 | - |
| 随机路由(50%) | 0.6294 | 0.6766 | 0.4831 | 40.00 |
| RouteT2I(50%) | 0.6350 | 0.6786 | 0.4865 | **83.97** |
| ZOOTER | 0.6350 | 0.6796 | 0.4854 | 77.95 |

在50%路由率下，RouteT2I的相对性能提升ΔP达到83.97%，显著优于所有基线。

### 消融实验

- **归一化Win率提升**: 40%路由率下RouteT2I达30.60%，领先最佳基线3.83%
- **成本节省**: 达到50%质量提升时，RouteT2I比随机路由减少70.24%的云端请求
- **Token选择门控**: 去除后性能明显下降，验证了关注关键token的重要性
- **双门控设计**: 相比单门控，双门控能更好地区分token的正/负面影响

### 关键发现

- 提示中的名词数量（实体数）与云端模型的质量优势正相关
- 简单提示下边缘模型可能优于云端，盲目使用大模型既浪费成本又可能降低质量
- 多维质量评估比单一指标更鲁棒，能更准确地捕捉质量差异

## 亮点与洞察

- 首次研究T2I生成请求的边缘-云端路由问题，应用场景明确且实际
- PRS多维质量比较方式巧妙，避免了图像质量主观评估的困难
- 双门控MoE设计优雅，将MoE中expert与质量维度对齐，token选择模拟了生成过程中的cross-attention

## 局限与展望

- 路由模型训练需要先用两个模型分别生成大量图像计算PRS，前置成本较高
- 仅在开源Stable Diffusion系列上验证，未涉及闭源API模型（如DALL·E、Midjourney）
- 质量评估依赖CLIP的正负文本对相似度，CLIP本身的偏见可能影响结果
- 未考虑用户个性化偏好对路由的影响

## 相关工作与启发

- RouteLLM、ZOOTER等LLM路由方法是重要参考，但因T2I输出空间的特殊性需要重新设计
- 多维质量评估思路可扩展到视频生成、3D生成等领域的路由
- MoE token选择机制可能对其他需要分析提示关键词的任务有启发

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次研究T2I路由，问题定义清晰
- 技术深度: ⭐⭐⭐⭐ — PRS定义和双门控MoE设计扎实
- 实验充分性: ⭐⭐⭐⭐ — 18个模型对、多角度评估、消融完整
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，公式推导连贯
- 实用价值: ⭐⭐⭐⭐ — 直接应用于商业T2I服务的成本优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)
- [\[ICCV 2025\] Discovering Divergent Representations between Text-to-Image Models](discovering_divergent_representations_between_text-to-image_models.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICCV 2025\] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation](multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)

</div>

<!-- RELATED:END -->
