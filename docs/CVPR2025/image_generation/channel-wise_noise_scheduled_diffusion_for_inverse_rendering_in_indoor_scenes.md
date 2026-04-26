---
title: >-
  [论文解读] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes
description: >-
  [CVPR 2025][图像生成][逆渲染] 提出通道级噪声调度方法，让单一扩散模型架构通过不同噪声调度实现精度优先（SDM, T=4）和多样性优先（PDM, T=1000）两种逆渲染模式，同时引入 ILR 隐式光照表征支持逐像素环境图推理和真实物体插入。
tags:
  - CVPR 2025
  - 图像生成
  - 逆渲染
  - 通道级噪声调度
  - 扩散模型
  - 逐像素光照
  - 室内场景
---

# Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes

**会议**: CVPR 2025  
**arXiv**: [2503.09993](https://arxiv.org/abs/2503.09993)  
**代码**: 无（未提及）  
**领域**: 扩散模型 / 3D视觉  
**关键词**: 逆渲染, 通道级噪声调度, 扩散模型, 逐像素光照, 室内场景

## 一句话总结
提出通道级噪声调度方法，让单一扩散模型架构通过不同噪声调度实现精度优先（SDM, T=4）和多样性优先（PDM, T=1000）两种逆渲染模式，同时引入 ILR 隐式光照表征支持逐像素环境图推理和真实物体插入。

## 研究背景与动机

**领域现状**：单图逆渲染将 RGB 图像分解为几何（法线、深度）、材质（漫反射、粗糙度）和光照。确定性方法（Li et al.、IRISformer、Zhu et al.）用大规模合成数据训练可以给出准确解，但不考虑解空间的多义性。基于 LDM 的方法（RGB↔X、Kocsis et al.）利用预训练先验可生成多样解，但忽视了多样性与精度之间的 trade-off，且各模态独立预测不处理依赖关系。

**现有痛点**：逆渲染本质上是病态问题——相同的 radiance 可以由不同的材质-光照-几何组合产生。需要同时具备"给出一个精确解"和"展示多种可能解"两种能力，但这两个目标是矛盾的。此外，现有 LDM 方法依赖预训练的三通道自编码器，无法处理高维的逐像素环境图，限制了物体插入等应用。

**核心矛盾**：扩散模型的总步数 T 控制了精度-多样性 trade-off（大 T → 高多样性但低精度，小 T → 高精度但低多样性），且多模态同时生成时不同模态间存在依赖关系（几何→材质→光照），统一的噪声调度无法利用这种依赖。

**本文目标** 如何用扩散模型同时实现逆渲染的精度和多样性，同时处理高维逐像素光照。

**切入角度**：几何的不确定性最低（最容易确定），光照最高。通过对不同模态通道应用不同噪声调度（几何先生成、光照后生成），可以利用模态间依赖提升质量。通过 T 的选择来分别优化精度和多样性。

**核心 idea**：对几何/材质/光照三类通道施加不同速率的噪声调度以利用模态依赖，并用大 T（PDM）和小 T（SDM）分别优化多样性和精度。

## 方法详解

### 整体框架
输入单张 RGB 图像，输出法线图、深度图、漫反射图、粗糙度图和逐像素环境图。首先在低分辨率（60×80）像素空间用条件扩散模型生成所有模态（非 latent 空间，因为 LDM 自编码器不支持环境图），然后通过 RGB 引导的超分辨率模型（SRM）上采样到 240×320。两个变体：PDM（T=1000, 渐进生成，多样性优先）和 SDM（T=4, 可切换生成，精度优先）。

### 关键设计

1. **通道级噪声调度（Channel-wise Noise Scheduling）**

    - 功能：为不同模态组（几何/材质/光照）设置不同的噪声衰减速率，利用模态间依赖关系
    - 核心思路：将输出 $\mathbf{z}_0$ 分为三组——几何（N, D）、材质（A, R）、光照（$\mathbf{f}$）。用参数化的余弦调度器 $\bar{\alpha}_t = \cos(((b-s)\cdot t+s)\cdot 2\pi)^{2\tau}$ 控制每组的噪声，其中 $\tau$ 越大噪声衰减越慢（生成越晚）。PDM 设 $\tau=(0.9, 1.2, 1.5)$ 分别对应几何/材质/光照，即几何最先清晰、光照最后清晰。SDM（T=4）用二值噪声调度表：step 0 几何+材质清晰、step 1 几何+光照清晰、step 2 材质+光照清晰、step 3 全噪声起始，实现"每步预测一个模态，已生成的模态作为下一步条件"
    - 设计动机：消融实验验证了几何→材质→光照的生成顺序最优（反序最差），因为几何约束材质，材质约束光照

2. **隐式光照表征（ILR）改进**

    - 功能：将高维逐像素环境图（3×128×H×W）压缩为紧凑的 96 维特征向量
    - 核心思路：设计 MLP 自编码器，编码器将环境图经 log1p 变换+批归一化后编码为 $\mathbf{f} \in \mathbb{R}^{96}$。三个解码器分别恢复环境图、shading 和 specular radiance，解码时用 expm1 逆变换回 HDR 值。相比原始 MAIR++ 的 ILR（从材质/几何推断而非从环境图编码），本文的 ILR 更准确地保留真值光照信息
    - 设计动机：直接处理 HDR 环境图的扩散模型不可行（数值范围太大），LDM 的三通道自编码器无法处理。ILR 提供了一个紧凑的中间表示

3. **精度-多样性双模型设计（PDM + SDM）**

    - 功能：用同一架构但不同 T 值实现两种互补目标
    - 核心思路：消融实验发现 T 从 1000→1 时，MSE 单调下降（精度提高）但方差=0（多样性丧失）。因此设计 PDM（T=1000, 渐进噪声调度, 采样 10 次取均值/最优）用于显示多个可能解，SDM（T=4, 可切换噪声调度, 确定性推理）用于给出单一最优解。两者在不同场景下互补：SDM 在歧义少的场景更优，PDM 在复杂歧义场景更优
    - 设计动机：逆渲染需要同时满足"精确分解"（如物体插入需精确光照）和"显示不确定性"（如几种材质-光照组合都可能）两种需求

### 损失函数 / 训练策略
扩散模型训练使用 velocity prediction loss。ILR 自编码器用环境图重建损失+渲染损失训练。SRM 用标准超分损失。在 OpenRooms FF 数据集上训练，CLIP 图像编码特征作为 cross-attention 条件。

## 实验关键数据

### 主实验（OpenRooms FF 合成数据, MSE×10⁻²）

| 方法 | Normal↓ | Depth↓ | Albedo↓ | Lighting↓ | Re-render↓ |
|------|---------|--------|---------|-----------|------------|
| Li et al. | 3.782 | 0.135 | 0.869 | 15.49 | 0.491 |
| Zhu et al. | 1.738 | 0.062 | 0.528 | *31.15 | *3.565 |
| **SDM** | **1.502** | **0.039** | **0.349** | **13.35** | **0.156** |
| PDM(Mean) | 1.831 | 0.047 | 0.394 | 14.08 | 0.150 |

### 消融实验

| 生成顺序 (τ_geo, τ_mat, τ_light) | Normal MSE↓ | Albedo MSE↓ | Lighting MSE↓ |
|----------------------------------|-------------|-------------|---------------|
| **(0.9, 1.2, 1.5) 几何→材质→光照** | **最优** | **最优** | **最优** |
| (1.5, 1.2, 0.9) 光照→材质→几何 | 最差 | 最差 | 最差 |

用户研究（物体插入）: SDM 被选中率 **32.2%**，接近真值 34.8%，远超基线 15-17%

### 关键发现
- 通道级噪声调度显著优于统一调度，生成顺序验证了直觉：几何→材质→光照
- SDM (T=4) 在所有单视图方法中精度最高，albedo MSE 甚至超越多视图方法 MAIR
- ILR 改进后的 re-rendering MSE (0.156) 比 Li et al. (0.491) 好 3 倍+，说明 HDR 光照表征非常关键
- T 值的精度-多样性 trade-off：T=1000 方差 0.183 但 MSE 2.236，T=1 方差 0 但 MSE 1.817

## 亮点与洞察
- **通道级噪声调度**是多模态扩散的通用技巧：当多个输出模态存在自然依赖顺序时，可以用不同调度实现"先生成稳定的、再生成不确定的"，减少误差累积
- **精度-多样性用 T 分离**的观察虽然简单但很实用：同一架构两种 T 值即可得到互补的两个模型
- **ILR 的 log-space MLP 自编码器**有效解决了 HDR 环境图的编码难题

## 局限与展望
- 需要维护两个独立模型（PDM + SDM），尚无统一方案
- 未复用预训练 LDM 先验（从零训练在合成数据上），泛化到真实场景的能力受限于训练数据
- 在低分辨率（60×80）操作后超分辨率，可能丢失细节
- 训练数据仅限于 OpenRooms FF 合成数据集

## 相关工作与启发
- **vs Li et al. / Zhu et al.**: 确定性方法，不考虑解的多义性。SDM 在精度上全面超越，同时 PDM 提供了多样性
- **vs RGB↔X**: LDM-based，各模态独立预测且不支持环境图。本文考虑模态依赖+支持逐像素光照
- **vs MAIR++**: 多视图方法，使用 ILR 但不从环境图直接编码。本文的 ILR 更精确

## 评分
- 新颖性: ⭐⭐⭐⭐ 通道级噪声调度和 SDM 设计有创新，但整体框架较工程化
- 实验充分度: ⭐⭐⭐⭐ 合成+真实+用户研究+消融完整，但真实场景评估较少
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融设计有说服力
- 价值: ⭐⭐⭐⭐ 通道级调度是可推广的技术，物体插入应用有实际价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion](uni-renderer_unifying_rendering_and_inverse_rendering_via_dual_stream_diffusion.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [\[CVPR 2025\] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] Improving Editability in Image Generation with Layer-wise Memory](improving_editability_in_image_generation_with_layer-wise_memory.md)

<!-- RELATED:END -->
