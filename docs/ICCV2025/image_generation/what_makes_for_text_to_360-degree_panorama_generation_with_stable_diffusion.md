---
title: >-
  [论文解读] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?
description: >-
  [图像生成] 通过系统分析LoRA微调中$W_{\{q,k,v,o\}}$各组件的行为，揭示了$W_v$和$W_o$负责学习全景球面结构而$W_q$和$W_k$保留透视域共享知识的机制，并据此提出高效的单分支全景生成框架UniPano。
tags:
  - 图像生成
---

# What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2505.22129](https://arxiv.org/abs/2505.22129)
- **代码**: [GitHub](https://github.com/jinhong-ni/UniPano)
- **领域**: 扩散模型 · 全景图生成
- **关键词**: 全景图生成, LoRA微调, 注意力机制分析, Stable Diffusion, 球面失真

## 一句话总结
通过系统分析LoRA微调中$W_{\{q,k,v,o\}}$各组件的行为，揭示了$W_v$和$W_o$负责学习全景球面结构而$W_q$和$W_k$保留透视域共享知识的机制，并据此提出高效的单分支全景生成框架UniPano。

## 研究背景与动机

全景图像（360度）具有等矩形投影导致的球面失真和2:1宽高比，与标准透视图像存在根本性差异。由于全景数据稀缺（如Matterport3D仅10,800张），直接训练生成模型不可行。

现有方法分为两类：

**多视角拼接方法**：生成多张透视图像再拼接为全景图，但过程复杂

**端到端微调方法**：如PanFusion使用双分支框架，但内存和训练开销巨大（60.12GB峰值显存）

一个有趣的现象是：简单的LoRA微调预训练扩散模型就能产生合理的全景图。**这背后的机制是什么？** 本文的核心动机就是回答这个问题：预训练透视扩散模型中的哪些组件在全景适应中起关键作用？

## 方法详解

### 整体框架

本文的分析在LoRA微调范式下展开，系统研究交叉注意力模块中四个可训练组件$W_{\{q,k,v,o\}}$的角色。基于分析结论，提出UniPano：一种高效的单分支全景生成框架。

### 关键发现1：孤立训练实验

分别仅训练$W_q$、$W_k$、$W_v$、$W_o$中的一个LoRA，其余冻结：

- **$W_q$和$W_k$独立训练失败**：无法捕获球面失真结构
- **$W_v$和$W_o$独立训练成功**：能学习全景的等矩形投影特征

定量结果显示$W_v$和$W_o$的FAED和FID远优于$W_q$和$W_k$。

### 关键发现2：联合训练后分解实验

先联合训练$W_{\{q,k,v,o\}}$的LoRA，然后推理时选择性禁用部分LoRA：

- 禁用$W_v$和$W_o$ LoRA → 模型恢复生成透视图像的能力
- 禁用$W_q$和$W_k$ LoRA → 模型仍能生成高质量全景图像

**结论**：$W_q$和$W_k$学习的是跨域共享的透视知识（不含球面变形信息），而$W_v$和$W_o$专门负责将透视知识适配为全景域的球面结构。

### UniPano设计

基于上述洞察，UniPano的核心设计为：

1. **冻结$W_q$和$W_k$**：因为它们与全景特定信息无关
2. **增强$W_o$的表达能力**：使用MoE（Mixture of Experts）替换$W_o$的LoRA

$$\text{MoE}(x) = \sum_{i=1}^{E} g_i(x) \cdot \text{LoRA}_i(x)$$

其中$g_i$为路由网络学习的权重，每个expert是一个独立的LoRA模块。

### 损失函数

训练使用标准扩散模型去噪损失：

$$\mathcal{L} = \mathbb{E}_{z_t, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

同时加入辅助路由负载均衡损失以稳定MoE训练。

## 实验

### 主实验：与SOTA方法比较（512×1024全景生成）

| 方法 | 峰值显存(GB) | 训练时间(hrs) | FAED↓ | FID↓(全景) | FID↓(20视角) | FID↓(8视角) |
|------|-------------|-------------|-------|-----------|-------------|------------|
| SD+LoRA | 31.69 | 2.26 | 7.19 | 51.69 | 19.32 | 20.68 |
| PanFusion | 60.12 | 6.61 | 6.04 | 46.47 | 17.04 | 19.88 |
| **UniPano** | **32.59** | **3.43** | **5.90** | **46.47** | **17.09** | **17.74** |

UniPano在FAED和8视角FID上取得最优，同时显存仅为PanFusion的54%、训练时间仅52%。

### 消融实验：$W_o$增强策略对比

| 策略 | FAED↓ | FID↓(全景) | FID↓(20视角) | FID↓(8视角) |
|------|-------|-----------|-------------|------------|
| Pano Only基线 | 7.90 | 50.40 | 20.10 | 20.56 |
| LoRA(r=8) | 8.34 | 48.58 | 16.94 | 19.42 |
| Deformable Attn | 9.78 | 46.41 | 16.56 | 18.77 |
| Local Window Attn | 7.65 | 50.39 | 18.24 | 19.41 |
| **MoE** | **7.21** | **48.83** | 19.50 | 20.05 |

MoE在FAED指标上最优，说明其在捕获全景球面结构方面最有效。

### 关键发现

1. 注意力机制中Query/Key学习的是跨域通用知识，Value/Output才是领域适配的关键
2. 这一结论为其他领域适配任务提供了通用指导：增强$W_v$和$W_o$的容量比全量LoRA更高效
3. UniPano可扩展到更高分辨率（1024×2048端到端生成）

## 亮点与洞察

1. **机制性发现**：首次系统解释了预训练透视扩散模型如何通过LoRA微调适配全景域
2. **设计原则清晰**：$W_q/W_k$保留通用知识 + $W_v/W_o$学习域特定知识的二分法思想简洁有力
3. **效率优势明显**：仅增加2.8%的显存开销，性能匹配甚至超越显存翻倍的双分支方法

## 局限性

- MoE选择是启发式的，文中承认可能非最优设计
- 分析仅限于交叉注意力模块中的LoRA微调，未涉及自注意力
- 评估仅在Matterport3D数据集上进行，场景多样性有限
- 生成分辨率距实际应用仍有差距

## 相关工作

- **扩散模型**: Stable Diffusion, DDPM, LoRA微调
- **全景图生成**: PanFusion, StitchDiffusion, CubeDiff, PanoFree
- **LoRA分析**: 本文为首个系统分析LoRA各组件在域适配中角色的工作

## 评分
- 新颖性：★★★★☆ — 机制分析视角新颖
- 技术深度：★★★★☆ — 实验设计系统、结论有说服力
- 实用性：★★★★☆ — 框架简洁高效，易于推广

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)

<!-- RELATED:END -->
