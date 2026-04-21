---
title: >-
  [论文解读] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion
description: >-
  [CVPR 2025][3D视觉][图像到3D生成] 提出Ouroboros3D，通过将多视图生成与3D重建整合为递归扩散过程，利用3D感知反馈机制（渲染CCM和颜色图作为去噪条件）和联合训练策略，解决了两阶段方法中的3D一致性不足和域间差距问题，在GSO数据集上取得SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 图像到3D生成
  - 递归扩散
  - 3D感知反馈
  - 多视图一致性
  - 联合训练
---

# Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion

**会议**: CVPR 2025  
**arXiv**: [2406.03184](https://arxiv.org/abs/2406.03184)  
**代码**: [项目页面](https://costwen.github.io/Ouroboros3D/)  
**领域**: 3D视觉 / 图像到3D  
**关键词**: 图像到3D生成, 递归扩散, 3D感知反馈, 多视图一致性, 联合训练

## 一句话总结

提出Ouroboros3D，通过将多视图生成与3D重建整合为递归扩散过程，利用3D感知反馈机制（渲染CCM和颜色图作为去噪条件）和联合训练策略，解决了两阶段方法中的3D一致性不足和域间差距问题，在GSO数据集上取得SOTA。

## 研究背景与动机

1. **领域现状**：单图到3D生成主流方法分为两阶段：先用多视图扩散模型生成多角度图像，再用前馈重建模型恢复3D表示。这种pipeline取得了较好结果，代表方法包括InstantMesh、LGM、CRM等。

2. **现有痛点**：(a) 多视图生成阶段在2D图像空间优化而非3D空间，难以保证几何一致性；(b) 重建模型主要在合成数据上训练，处理生成的多视图图像时存在域间差距；(c) 两个模型独立设计和训练，无法相互受益。

3. **核心矛盾**：多视图扩散模型和3D重建模型作为独立组件各自优化时信息不互通——扩散模型不知道生成的图像能否被正确重建，重建模型对分布外的生成图像适应性差。

4. **本文目标**：将两个阶段统一为一个端到端可训练的递归扩散过程，实现两个模型的相互增强。

5. **切入角度**：将重建模型的渲染结果作为3D感知条件反馈到扩散模型的去噪循环中，同时联合训练两个模型消除域间差距。

6. **核心idea**：递归扩散——每个去噪步先预测clean多视图→送入重建模型→渲染3D-aware maps→作为下一步去噪的条件，反复迭代形成自我优化的闭环。

## 方法详解

### 整体框架

基于Stable Video Diffusion（SVD）作为多视图生成器，Large Gaussian Model（LGM）作为3D重建器。在去噪采样循环中，每步先将预测的 $\tilde{\mathbf{x}}_0^f$ 解码为多视图图像送入LGM重建3D高斯，然后从重建的3D模型渲染颜色图和CCM（Canonical Coordinates Map），编码后注入下一步的去噪网络。

### 关键设计

1. **3D感知反馈机制**:
    - 功能：将显式3D几何信息注入多视图扩散模型的去噪过程
    - 核心思路：在每个去噪步，从LGM重建的3D高斯中渲染两种maps：(a) RGB颜色图（保留纹理信息）；(b) Canonical Coordinates Map（CCM，每个像素对应3D模型上的全局归一化顶点坐标）。用两个轻量卷积编码器（类似T2I-Adapter）将这些maps编码为与U-Net编码器中间特征同尺度的特征，在每个分辨率层加到U-Net编码器中。
    - 设计动机：选择CCM而非深度图/法线图，因为CCM捕捉的是全局顶点坐标（跨视图一致），而深度图是相对于各自视角归一化的。CCM天然编码了跨视图的几何对应关系，为多视图一致性提供更强的约束。

2. **联合训练策略（Joint Training）**:
    - 功能：同时训练多视图扩散模型和3D重建模型，消除两阶段之间的域间差距
    - 核心思路：训练时，重建模型不使用原始GT多视图图像，而是使用扩散过程中恢复的图像 $\tilde{\mathbf{x}}_0$ 作为输入。重建损失包含RGB L2损失和LPIPS感知损失。LGM中引入零初始化的time embedding层感知不同噪声级别。以0.5概率进行self-conditioning（一半时间使用前一步的3D反馈，一半时间不使用），防止模型过度依赖3D信息。
    - 设计动机：独立训练的重建模型只见过"干净"渲染图像，处理"带生成噪声"的多视图图像时性能下降。联合训练让重建模型适应扩散模型的输出分布，同时重建损失反向传播到扩散模型相当于增加了3D一致性的隐式监督。

3. **3D感知递归推理策略**:
    - 功能：推理时通过迭代循环逐步优化多视图图像和3D模型
    - 核心思路：初始条件设为零（无3D反馈），每个后续去噪步用前一步的重建结果更新3D条件。随着去噪推进，信噪比提高→重建质量提升→反馈条件更准确→去噪结果更一致→形成正向螺旋。
    - 设计动机：与仅在推理时结合（如VideoMV的re-sampling策略）相比，联合训练确保了模型在训练时就学会利用3D反馈，且不会因为初期不准确的3D反馈导致偏离输入图像。

### 损失函数 / 训练策略

扩散模型使用标准去噪损失。重建模型使用 $\mathcal{L}_G = \mathcal{L}_{rgb} + \lambda \mathcal{L}_{LPIPS}$，输入为扩散模型恢复的图像 $\tilde{\mathbf{x}}_0$（含噪声级别嵌入）。训练数据为Objaverse过滤后的~80K物体，渲染512×512分辨率的16帧轨道视频。Self-conditioning概率0.5。

## 实验关键数据

### 主实验

GSO数据集（100个物体，零样本评估）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 类型 |
|------|-------|-------|--------|------|
| VideoMV (多视图) | 18.605 | 0.841 | 0.155 | 两阶段 |
| SV3D (多视图) | 21.042 | 0.850 | 0.130 | 两阶段 |
| InstantMesh (3D) | 19.948 | 0.873 | 0.121 | 两阶段 |
| LGM (3D) | 17.716 | 0.832 | 0.189 | 两阶段 |
| **Ouroboros3D (多视图)** | **21.770** | **0.887** | **0.109** | 统一 |
| **Ouroboros3D (3D)** | **21.761** | **0.889** | **0.109** | 统一 |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| 无反馈（基线SVD+LGM） | 20.5 | 0.870 | 0.125 | 标准两阶段 |
| +联合训练（无3D反馈） | 21.0 | 0.878 | 0.118 | 联合训练本身减少域间差距 |
| +RGB反馈 | 21.3 | 0.882 | 0.114 | 颜色引导提供外观信息 |
| +CCM反馈 | 21.5 | 0.885 | 0.111 | 坐标图提供更强几何约束 |
| +RGB+CCM反馈 (Full) | **21.8** | **0.887** | **0.109** | 完整方案 |

### 关键发现

- Ouroboros3D在多视图质量和3D重建质量上同时超越所有两阶段方法，证明了统一框架的优势
- CCM反馈比RGB反馈贡献更大——全局坐标提供了跨视图的显式几何对应
- 联合训练即使不加3D反馈也能提升性能（PSNR +0.5），说明消除域间差距本身就有价值
- 与仅在推理时结合的VideoMV(GS)相比，Ouroboros3D的联合训练策略效果更优

## 亮点与洞察

- **递归扩散**的概念名副其实（Ouroboros=衔尾蛇）——3D重建的输出反馈到生成的输入，形成自我改进的闭环。这种"生成→理解→反馈"的范式有更广泛的适用性。
- **CCM作为3D-aware条件**是一个被忽视但极有价值的选择——比深度图更具全局性（不依赖单一视角），比法线图更具唯一性（全局坐标vs局部方向）。
- **Self-conditioning概率0.5**的训练策略值得注意——允许模型同时学习有/无3D反馈的场景，增加鲁棒性。

## 局限与展望

- 依赖LGM的重建能力，LGM本身对细节的恢复有限
- 递归过程增加了推理时间（每步都需要重建和渲染）
- 仅使用8帧多视图，覆盖的角度有限
- 未来可以扩展到更多帧或视频扩散模型，进一步提升覆盖和一致性

## 相关工作与启发

- **vs InstantMesh/CRM**: 这些两阶段方法独立训练两个模块，Ouroboros3D通过联合训练和3D反馈实现两者相互增强
- **vs IM-3D/VideoMV**: 它们在推理时通过re-sampling引入3D信息但缺乏联合训练，Ouroboros3D的训练时集成更彻底
- **vs DMV3D**: DMV3D将3D重建作为扩散去噪器但从头训练，泛化能力差；Ouroboros3D基于预训练SVD保留了泛化性

## 评分

- 新颖性: ⭐⭐⭐⭐ 递归扩散+3D反馈的统一框架设计优雅，CCM条件的选择有洞察力
- 实验充分度: ⭐⭐⭐⭐ GSO定量对比+消融+定性比较全面
- 写作质量: ⭐⭐⭐⭐ 框架概念图清晰，对比示意图直观
- 价值: ⭐⭐⭐⭐ 为"多视图生成+3D重建"的统一提供了有效范式

<!-- RELATED:START -->

## 相关论文

- [Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)

<!-- RELATED:END -->
