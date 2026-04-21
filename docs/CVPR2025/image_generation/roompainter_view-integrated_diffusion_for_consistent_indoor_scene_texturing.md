---
title: >-
  [论文解读] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing
description: >-
  [CVPR 2025][图像生成][室内场景纹理] 提出 RoomPainter，通过零样本的多视角集成采样(MVIS)和相关视角注意力机制，将 2D 扩散模型适配为 3D 一致的室内场景纹理合成工具，采用两阶段策略确保全局和局部一致性。
tags:
  - CVPR 2025
  - 图像生成
  - 室内场景纹理
  - 扩散模型
  - 多视角一致性
  - 零样本纹理合成
  - 遮挡修复
---

# RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing

**会议**: CVPR 2025  
**arXiv**: [2412.16778](https://arxiv.org/abs/2412.16778)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 室内场景纹理, 扩散模型, 多视角一致性, 零样本纹理合成, 遮挡修复

## 一句话总结

提出 RoomPainter，通过零样本的多视角集成采样(MVIS)和相关视角注意力机制，将 2D 扩散模型适配为 3D 一致的室内场景纹理合成工具，采用两阶段策略确保全局和局部一致性。

## 研究背景与动机

室内场景纹理合成在 VR、数字媒体和创意艺术中有重要应用。现有方法面临三大挑战：(1) 基于修复的方法（如 Text2Tex）逐视角生成纹理，导致严重的跨视角不一致和明显接缝；(2) 基于优化的方法（如 SceneTex）使用 SDS 损失优化全局纹理，但计算开销巨大且训练不稳定；(3) 物体遮挡导致纹理缺失。

核心问题：如何高效地生成全局一致的室内场景纹理，同时解决遮挡问题？RoomPainter 通过两阶段策略：(1) 全局阶段使用 MVIS 生成整体一致纹理；(2) 细化阶段使用 MVRS 对每个实例进行遮挡修复和纹理细化。

## 方法详解

### 整体框架

基于 SDXL + ControlNet（深度条件），分两阶段运行。第一阶段：环绕房间中心的 $N$ 个相机同时生成纹理，通过动态合并 UV 纹理图实现全局一致。第二阶段：对每个实例单独应用 MVRS，修复遮挡区域并细化纹理。关键的零样本技术为 MVIS + 相关视角注意力。

### 关键设计一：多视角集成采样（MVIS）

**功能**：零样本适配 2D 扩散模型生成多视角一致的纹理图

**核心思路**：给定 $N$ 个相机 $\{C^n\}$，在扩散采样的每个时间步 $t$：(1) 对所有视角运行去噪得到估计 $x_{0,t}^n$；(2) 解码到图像空间得到 $\mathcal{I}_t^n$；(3) 反投影到 UV 空间得到逐视角纹理 $\mathcal{T}_t^n$；(4) 使用动态合并策略融合：

$$\mathcal{T}_t = \frac{\sum_{n=1}^{N}(W_n^{exp(t)} \cdot \mathcal{T}_t^n)}{\sum_{n=1}^{N} W_n + \gamma}$$

其中 $W_n$ 基于法线-视角余弦相似度的反向值，$exp(t)$ 随 $t$ 降低线性增大使融合更锐化；(5) 从 $\mathcal{T}_t$ 重新渲染各视角图像并编码回潜空间，替换原始 $x_{0,t}^n$ 引导下一步采样。

**设计动机**：通过在每个扩散步骤中将所有视角的信息投影到共享 UV 空间并动态融合，实现了"一致的全局纹理指导个体视角采样"的循环。权重基于视角角度确保每个面片使用最优视角的纹理。

### 关键设计二：多视角集成重绘采样（MVRS）

**功能**：修复遮挡区域并细化实例级纹理

**核心思路**：对每个实例解耦后，使用 MVIS 的变体——将已有纹理区域（来自第一阶段）加噪到当前时间步后通过掩码 $P$ 与 MVIS 采样结果混合：已绘制区域保持不变，未绘制区域由 MVIS 生成。本质上是将扩散修复（repaint）与多视角采样结合。

**设计动机**：第一阶段因物体遮挡导致部分区域无纹理。MVRS 在修复遮挡时保持与已有纹理的风格一致性，并通过实例级操作避免全局重生成的计算浪费。

### 关键设计三：相关视角注意力机制

**功能**：增强扩散模型采样过程中的多视角信息共享

**核心思路**：修改 U-Net 的自注意力——将每个视角 $n$ 的 Query $Q_n$ 与其相关视角（如相邻左右视角）的 Key/Value 拼接后计算注意力：

$$\text{softmax}\left(\frac{Q_n \tilde{K}_n^T}{\sqrt{d}} \tilde{V}_n\right)$$

其中 $\tilde{K}_n = [K_1, K_2, ..., K_R]^T$ 为 $R$ 个相关视角的 Key 拼接。

**设计动机**：无训练地在扩散采样中注入跨视角信息，确保相邻视角间的纹理风格和内容一致。在房间级别使用相邻视角，在实例级别使用所有视角。

### 损失函数

无训练方法——零样本推理时约束。基于 DDPM 采样过程，通过纹理图反馈机制引导生成。

## 实验关键数据

### 主实验：室内场景纹理合成

| 方法 | 生成时间(分钟)↓ | CLIP Score↑ | Aesthetic Score↑ |
|------|-----------------|-------------|-----------------|
| Text2Tex-H | 8.50 | 21.58 | 4.34 |
| Text2Tex-C | 70.75 | 21.93 | 4.85 |
| SceneTex | 2614.50 | 21.87 | 4.75 |
| **RoomPainter** | **46.00** | **23.47** | **5.03** |

### 消融实验

| 配置 | CLIP Score↑ | Aesthetic Score↑ |
|------|-------------|-----------------|
| 完整方法 | 23.47 | 5.03 |
| 无相关视角注意力 | 23.32 | 5.01 |
| 无 MVIS | 23.27 | 5.01 |
| 无 MVRS | 22.39 | 4.40 |

### 关键发现

- 比 SceneTex 快 **57 倍**（46 分钟 vs 2614 分钟），同时 CLIP Score 高 +1.6
- MVRS 对最终质量贡献最大——去除后 Aesthetic Score 从 5.03 降至 4.40，证明遮挡修复至关重要
- Text2Tex-H 速度快但一致性差（可见接缝）；Text2Tex-C 质量可接受但逐实例生成耗时 70 分钟
- 相关视角注意力显著改善跨视角一致性（定性对比中消除了颜色和风格不一致）

## 亮点与洞察

1. **零样本 3D 一致性**：无需训练多视角扩散模型，通过 UV 纹理图反馈机制在采样过程中实现跨视角一致
2. **两阶段策略的精妙**：先整体后局部，全局 MVIS 建立风格基调，实例级 MVRS 修复遮挡并增强细节
3. **效率优势明显**：相比优化方法快 50+ 倍，使实际部署成为可能

## 局限与展望

- 需要预先获取高质量的室内场景网格作为输入
- 全局阶段相机位置固定于房间中心，对非规则房间可能不够灵活
- 纹理分辨率受 SDXL 输出分辨率限制
- 未来可探索结合 3D 生成模型直接生成带纹理的场景

## 相关工作与启发

- **Text2Tex / TEXTure**：逐视角修复的纹理生成先驱，但缺乏全局一致性
- **SceneTex**：基于 VSD 优化的场景级纹理方法，质量好但耗时极长
- **SyncMVD**：修改自注意力实现跨视角信息共享的思路与相关视角注意力类似

## 评分

⭐⭐⭐⭐ — 方法设计清晰实用，两阶段策略解决了室内场景纹理的核心问题。57 倍加速且质量提升的结果令人印象深刻。零样本方法的通用性使其可直接应用于各种扩散模型。但对输入网格质量的依赖限制了应用场景。

<!-- RELATED:START -->

## 相关论文

- [LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)
- [MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](../../ECCV2024/image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)

<!-- RELATED:END -->
