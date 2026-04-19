---
title: >-
  [论文解读] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers
description: >-
  [3D视觉] JointDiT 基于 Flux 扩散 Transformer 构建 RGB-Depth 联合分布模型，通过自适应调度权重和非平衡时间步采样策略，使单一模型通过控制各模态的时间步即可灵活执行联合生成、深度估计和深度条件图像生成三种任务。
tags:
  - 3D视觉
---

# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2505.00482](https://arxiv.org/abs/2505.00482)
- **代码**: [项目页面](https://byungki-k.github.io/JointDiT/)
- **领域**: 3D视觉
- **关键词**: 扩散Transformer, RGB-Depth联合生成, 深度估计, 联合分布建模, Flow Matching

## 一句话总结
JointDiT 基于 Flux 扩散 Transformer 构建 RGB-Depth 联合分布模型，通过自适应调度权重和非平衡时间步采样策略，使单一模型通过控制各模态的时间步即可灵活执行联合生成、深度估计和深度条件图像生成三种任务。

## 研究背景与动机

扩散模型在图像生成和条件生成（深度估计、深度引导生成等）上取得巨大进展。近期研究探索了 RGB 与深度的**联合分布建模**，发现其不仅能联合生成，还可作为条件生成的统一替代方案。但存在两个核心问题：

**生成质量有限**：现有联合模型（LDM3D, JointNet）基于较弱的 Stable Diffusion 架构，生成的图像保真度和深度精度都不理想

**时间步分离训练挑战**：要实现"一个模型多种任务"需要对两种模态使用独立的噪声水平训练，但如何有效训练尚未被充分探索

关键洞察：Flux 等先进扩散 Transformer 拥有卓越的图像先验和全局感受野（Transformer 架构），而 Transformer 在深度估计任务中也已被证明有效（DPT, Depth Anything）。

## 方法详解

### 整体框架

JointDiT 在 Flux 的 RGB 分支旁构建并行的 Depth 分支，通过联合连接模块（Joint Connection Module）交换特征，实现联合分布建模。冻结预训练 backbone，仅训练 LoRA 和联合连接模块。

### 联合条件 Flow Matching（JCFM）

扩展 flow matching 框架学习联合向量场 $v_{t_x,t_y}(x,y|x_1,y_1)$，两个模态使用独立时间步 $t_x, t_y$：

$$\mathcal{L}_{\text{JCFM}}(\theta) = \mathbb{E}_{t_x,t_y}\left[\|v_{t_x,t_y,\theta}(x,y) - v_{t_x,t_y}(x,y|x_1,y_1)\|\right]$$

任务通过控制初始时间步切换：
- **联合生成**：$t_x=0, t_y=0$
- **深度估计**：$t_x=1, t_y=0$（图像干净，深度从噪声开始）
- **深度条件生成**：$t_x=0, t_y=1$

### 关键设计一：自适应调度权重（Adaptive Scheduling Weights）

在联合交叉注意力中，根据两个模态的相对噪声水平动态调整信息传递权重：

$$w_x(t_x, t_y) = \text{sigmoid}\left(\alpha\left(\frac{t_y}{t_x+t_y} - \frac{1}{2}\right)\right)$$

$$w_y(t_x, t_y) = \text{sigmoid}\left(\alpha\left(\frac{t_x}{t_x+t_y} - \frac{1}{2}\right)\right)$$

其中 $\alpha=3$。直觉是：噪声更大的分支应更多地参考较干净分支的结构信息。

### 关键设计二：非平衡时间步采样（Unbalanced Timestep Sampling）

为充分覆盖联合生成和条件生成的时间步组合空间：
- 50% 概率：$t_x, t_y$ 分别从两个不同分布 $f(t), g(t)$ 独立采样
- 50% 概率：$t_x = t_y$ 从 $f(t)$ 采样

确保模型在各种 $(t_x, t_y)$ 组合上都获得充足训练。

### 损失函数

最终输出结合自注意力和联合交叉注意力：

$$\mathbf{G}_x = \text{Attn}(\mathbf{S}_x) + w_x \cdot \text{JointAttn}(\mathbf{S}_x, \mathbf{S}_y)$$

## 实验

### 主实验：深度估计零样本泛化

| 类型 | 方法 | NYUv2 AbsRel↓ | KITTI AbsRel↓ | ETH3D AbsRel↓ |
|------|------|-------------|-------------|-------------|
| 判别式 | Depth-Anything-V2 | 4.4 | 7.5 | 13.2 |
| 扩散式 | Marigold | 5.5 | 9.6 | 6.5 |
| 扩散式 | GeoWizard | 5.2 | 10.1 | 6.4 |
| **联合式** | **JointDiT** | **4.9** | **9.4** | **5.6** |

JointDiT 作为联合模型在深度估计上可与专用深度估计模型媲美。

### 消融实验：关键技术贡献

| 自适应调度权重 | 非平衡采样 | 联合生成 FID↓ | 深度估计 AbsRel↓ |
|-------------|---------|-----------|-------------|
| ✗ | ✗ | 较高 | 较高 |
| ✓ | ✗ | 改善 | 改善 |
| ✗ | ✓ | 改善 | 改善 |
| ✓ | ✓ | **最低** | **最低** |

两种技术均有显著贡献且互补。

### 关键发现
- JointDiT 的 3D 提升结果远优于 LDM3D 和 JointNet，生成几何精确的 3D 点云
- RGB 和 Depth 分支在生成过程中表现出互补行为：深度分支捕获结构信息，RGB 分支聚焦纹理和外观
- 在挑战性领域（卡通、像素艺术）中，JointDiT 的深度估计优于专用方法，得益于联合建模的互补优势

## 亮点与洞察

1. **联合分布作为条件生成的替代方案**：单一模型通过时间步控制即可覆盖多种任务
2. **利用先进扩散 Transformer**：Flux 的图像先验 + Transformer 的全局感受野是联合建模成功的关键
3. **轻量级适配**：仅训练 LoRA 和联合连接模块，保留预训练知识
4. **互补行为发现**：RGB 和 Depth 分支在生成过程中自然分工

## 局限性
- 训练数据仅 50k 对，可能限制了泛化能力
- 依赖 Depth-Anything-V2 生成的伪深度标签训练，可能继承其偏差
- 联合生成的 FID 虽然大幅改善，绝对值仍有提升空间
- 仅支持 $512 \times 512$ 分辨率

## 相关工作
- LDM3D / JointNet：基于 SD 的 RGB-Depth 联合生成
- Marigold / GeoWizard：基于扩散模型的深度估计
- Flux：先进的 Flow Matching 扩散 Transformer
- ControlNet：深度条件图像生成

## 评分
- **创新性**: ⭐⭐⭐⭐ — 自适应调度权重和非平衡采样策略新颖
- **实用性**: ⭐⭐⭐⭐ — 单模型多任务，应用灵活
- **实验完整度**: ⭐⭐⭐⭐ — 联合生成/深度估计/条件生成全面评测
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，技术细节完整

<!-- RELATED:START -->

## 相关论文

- [HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)

<!-- RELATED:END -->
