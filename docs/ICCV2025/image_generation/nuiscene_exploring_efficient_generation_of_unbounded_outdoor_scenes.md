---
title: >-
  [论文解读] NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes
description: >-
  [ICCV 2025][图像生成][无界场景生成] NuiScene 提出使用向量集（vector set）编码场景块的高效方法，配合显式 outpainting 扩散模型实现快速无界户外场景生成，并策划了 NuiScene43 高质量户外场景数据集。 大规模户外场景生成对开放世界游戏、电影 CGI、VR 模拟至关重要…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "无界场景生成"
  - "向量集表示"
  - "outpainting"
  - "扩散模型"
  - "户外场景"
  - "3D生成"
---

# NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes

**会议**: ICCV 2025  
**arXiv**: [2503.16375](https://arxiv.org/abs/2503.16375)  
**领域**: 3D生成·场景生成  
**关键词**: 无界场景生成, 向量集表示, outpainting, 扩散模型, 户外场景, 3D生成  

## 一句话总结

NuiScene 提出使用向量集（vector set）编码场景块的高效方法，配合显式 outpainting 扩散模型实现快速无界户外场景生成，并策划了 NuiScene43 高质量户外场景数据集。

## 研究背景与动机

大规模户外场景生成对开放世界游戏、电影 CGI、VR 模拟至关重要。与室内场景生成相比，户外场景面临独特挑战：

**高度差异巨大**：从矮小房屋到摩天大楼，场景块高度变化剧烈。先前方法使用 triplane 等空间结构化潜变量，需要统一尺寸，导致高建筑被压缩丢失细节或内存爆炸

**推理速度慢**：现有方法依赖 RePaint 的重采样式 inpainting 进行 outpainting，需要额外扩散步

**缺乏高质量户外数据**：语义驾驶数据集网格质量差，Objaverse 场景缺乏统一尺度

## 方法详解

### 1. 数据策划

从 Objaverse 筛选 43 个高质量场景，通过以下步骤处理：
- 使用 DuoduoCLIP 嵌入过滤场景
- 标注相对尺度建立统一缩放
- 清理地面几何并统一地面厚度
- 场景分割为 $(50, h_{vox}, 50)$ 大小的块

### 2. 向量集 VAE

**编码器**：对每个场景块均匀采样 $N_p$ 个点云 $\mathbf{p} \in \mathbb{R}^{N_p \times 3}$，通过交叉注意力聚合为紧凑表示：

$$\mathbf{z}^{\mathbf{p}} = \mathcal{E}(\mathbf{p}) \in \mathbb{R}^{V \times c}$$

其中 $V=16$ 为向量数，$c=64$ 为通道数。相比 triplane 的 $V=3 \times 4^2=48$，向量集仅需 16 个 token，压缩率更高。

**防止后验坍塌**：从同一块采样两个点云 $\mathbf{p}, \mathbf{q}$ 并约束嵌入一致：$\mathcal{L}_{emb} = (\mathbf{z}^{\mathbf{p}} - \mathbf{z}^{\mathbf{q}})^2$

**高度预测**：学习高度嵌入 $\mathbf{e}_h$ 查询潜变量以预测块高度，推理时用于裁剪不必要的体素查询。

**解码器**：向量集通过交叉注意力预测占据率 $\hat{o}_r = \text{FC}(\text{CA}(\mathbf{f}_{out}, \text{PE}(\mathbf{r})))$

**总损失**：$\mathcal{L} = \lambda_{kl}\mathcal{L}_{kl} + \lambda_{emb}\mathcal{L}_{emb} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_{height}\mathcal{L}_{height}$

### 3. 显式 Outpainting 扩散模型

不同于 RePaint 的重采样方式，显式训练扩散模型在 $2 \times 2$ 网格上生成四个块，通过条件掩码和已生成块嵌入进行条件化。

四种条件配置对应栅格扫描中的所有场景：
- $\{0,0,0,0\}$：无条件生成
- $\{1,0,1,0\}$：左列已知
- $\{1,1,0,0\}$：上行已知
- $\{1,1,1,0\}$：仅右下角待生成

训练目标：$\mathbb{E}[\|\boldsymbol{\epsilon} - \epsilon_\theta((\mathbf{X}_t \oplus \mathbf{C}), t)\|_2^2]$

## 实验

### VAE 重建质量

| 方法 | 输出分辨率/S | IoU↑ | CD↓ | F-Score↑ |
|------|------------|------|-----|----------|
| triplane | 3×32²/6 | 0.734 | 0.168 | 0.508 |
| triplane | 3×64²/6 | 0.940 | 0.064 | 0.831 |
| **vecset** | **-** | **0.989** | **0.055** | **0.864** |

向量集在所有指标上全面超越 triplane，IoU 达 0.989。

### 扩散生成质量与效率

| 方法 | FPD↓ | KPD↓ | token 数 | 训练时间 | 显存 |
|------|------|------|---------|---------|------|
| triplane | 1.406 | 2.589 | 192 | 27.6h | 24.4GB |
| **vecset** | **0.571** | **0.951** | **64** | **11.1h** | **10.4GB** |

向量集扩散模型训练快 2.5 倍，显存占用仅 42%，同时 FPD 降低 59%。

### Outpainting 速度对比

| 方法 | 生成 21×21 块时间 (s) |
|------|---------------------|
| RePaint (r=5) | 1022.20 |
| **显式 outpainting** | **215.92** |

显式 outpainting 快约 4.7 倍，且无需重采样即可保持连贯性。

## 亮点与洞察

1. **向量集 vs triplane**：向量集用更少的 token 实现更好的压缩，且天然适配不同高度的场景块
2. **显式 outpainting**：通过训练四种条件配置，避免了 RePaint 的重采样开销
3. **跨场景融合**：在多场景联合训练后，模型能生成混合城堡与摩天楼的场景，展现了泛化能力

## 局限性

- 数据集小（43 个场景），限制了泛化性
- 缺乏全局上下文，无法进行大范围规划（如道路网络布局）
- 块间连接偶尔出现断裂或噪声伪影
- 无条件/标签控制能力

## 相关工作

- 室内无界生成：BlockFusion（triplane）、LT3SD（密集特征网格）
- 户外场景：SemCity（语义驾驶）、CityDreamer、SceneDreamer
- 3D 场景表示：3DShape2VecSet、NeRF、高斯溅射

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[ICML 2026\] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](../../ICML2026/image_generation/scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](../../CVPR2025/image_generation/channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)

</div>

<!-- RELATED:END -->
