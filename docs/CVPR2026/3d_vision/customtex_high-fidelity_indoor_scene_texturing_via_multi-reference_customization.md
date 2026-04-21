---
title: >-
  [论文解读] CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization
description: >-
  [CVPR 2026][3D视觉][室内场景纹理] 提出CustomTex框架，通过实例级的多参考图像驱动和双蒸馏训练策略（语义级VSD蒸馏+像素级超分蒸馏），实现3D室内场景的高保真、实例可控纹理生成，在语义一致性、纹理清晰度和减少"烘焙阴影"方面全面超越现有方法。
tags:
  - CVPR 2026
  - 3D视觉
  - 室内场景纹理
  - 多参考图像定制
  - 双蒸馏
  - VSD优化
  - 实例级控制
---

# CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization

**会议**: CVPR 2026  
**arXiv**: [2603.19121](https://arxiv.org/abs/2603.19121)  
**代码**: https://chenweilinx.github.io/CustomTex/  
**领域**: 3D视觉  
**关键词**: 室内场景纹理, 多参考图像定制, 双蒸馏, VSD优化, 实例级控制

## 一句话总结
提出CustomTex框架，通过实例级的多参考图像驱动和双蒸馏训练策略（语义级VSD蒸馏+像素级超分蒸馏），实现3D室内场景的高保真、实例可控纹理生成，在语义一致性、纹理清晰度和减少"烘焙阴影"方面全面超越现有方法。

## 研究背景与动机
创建逼真的3D室内场景纹理是VR/AR、建筑可视化和电影制作的基石。**现有方法的痛点**：（1）**文字驱动方法**（SceneTex、TEXture等）语义模糊，无法传达精确视觉特征（如布料纹理、木纹、壁纸图案）；（2）即使用单张参考图做驱动也只能提供全局粗粒度控制；（3）纹理质量不足——模糊、伪影多，且扩散模型会学习训练数据的光照信息产生"烘焙阴影（baked-in shading）"，不适合不同光照渲染。

**核心矛盾**：扩散过程中语义控制和像素质量耦合——InstanceTex虽支持多文本实例级控制，但仍受文本精度和质量限制。**本文切入角度**：用多张参考图像（每个实例一张）替代文本，将"语义生成"和"像素增强"分离为两个独立蒸馏过程，在VSD框架下统一优化。

## 方法详解

### 整体框架
输入未纹理化的3D室内场景mesh（含UV展开）和每个物体实例的参考图像。每次迭代：（1）随机视点渲染RGB图、深度图和实例mask；（2）语义级蒸馏用depth-to-image扩散+Instance Cross-Attention+LoRA计算VSD梯度；（3）像素级蒸馏用预训练SR模型计算SR梯度；（4）两个梯度联合更新隐式纹理场。

### 关键设计
1. **Instance Cross-Attention + InsVSD（语义级蒸馏）**:

    - 功能：确保每个实例的纹理与其参考图像语义一致
    - 核心思路：IP-Adapter提取参考图特征$f^{ref}_i$，用实例mask $m_i$在feature级调制cross-attention：
    $Z' = \frac{1}{N}\sum_{i=1}^N m_i \cdot \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}_i^\top}{\sqrt{d_k}}\right)\mathbf{V}_i$
    - 基于VSD交替优化：冻结LoRA更新纹理$\theta$（VSD梯度$\nabla_\theta\mathcal{L}_{\text{VSD}} = \mathbb{E}[\omega(t)(\epsilon_{\phi_d} - \epsilon_{\phi_{\text{LoRA}}})\frac{\partial\mathcal{T}}{\partial\theta}]$），再冻结$\theta$更新LoRA $\phi$
    - 设计动机：Feature-level mask比noise-level mask更稳定（消融证实），精确对齐每个参考特征到对应实例区域

2. **像素级蒸馏（Pixel-Level Distillation）**:

    - 功能：增强纹理清晰度和高频细节
    - 核心思路：利用预训练SR模型$\phi_{SR}$计算SR梯度：$\nabla_\theta\mathcal{L}_{\text{SR}} = \mathbb{E}[\omega(t)(\epsilon_{\phi_{SR}} - \epsilon_{\phi_{\text{LoRA}}})\frac{\partial\mathcal{T}}{\partial\theta}]$
    - 最终梯度：$\nabla_\theta\mathcal{L} = \nabla_\theta\mathcal{L}_{\text{VSD}} + \lambda_{SR}\nabla_\theta\mathcal{L}_{\text{SR}}$
    - 训练策略：前5000次$\lambda_{SR}=0$仅做语义蒸馏，之后$\lambda_{SR}=1.2$加入像素增强
    - 设计动机：集成到蒸馏过程比后处理SR好得多——UV纹理缺乏自然图像语义结构，SR模型无法直接对UV纹理做超分

3. **多分辨率哈希网格纹理表示**:

    - 功能：隐式表示纹理并支持任意分辨率输出
    - 核心思路：基于Instant-NGP的多分辨率哈希网格，UV坐标→多尺度grid→hash映射→特征拼接→Cross-Attention解码器→RGB
    - 推理效率：4K纹理约2.4秒，12K约22秒
    - 设计动机：比固定分辨率纹理贴图更灵活，优化更高效

### 损失函数 / 训练策略
- VSD梯度（语义）+ SR梯度（像素），交替优化纹理$\theta$和LoRA $\phi$
- 时间退火：前5000次$t\sim U(0.02,0.98)$，之后$t\sim U(0.02,0.5)$
- 30000次迭代，5000球面分布视点，LR纹理0.001/LoRA 0.0001
- 约48小时在单张RTX A800

## 实验关键数据

### 主实验
图像到纹理（10个3D-FRONT场景）：

| 方法 | CLIP-I↑ | CLIP-FID↓ | Q-Align IQA↑ | Q-Align IAA↑ |
|------|---------|-----------|-------------|-------------|
| **CustomTex** | **0.797** | **106.229** | **4.469** | **3.629** |
| SceneTex-IPA | 0.741 | 121.118 | 4.009 | 3.594 |
| Paint3D | 0.694 | 130.138 | 2.896 | 2.401 |
| HY3D-2.1 | 0.682 | 134.680 | 2.187 | 1.838 |

文本到纹理：

| 方法 | CLIP-T↑ | IS↑ | Q-Align IQA↑ |
|------|---------|-----|-------------|
| **CustomTex** | **0.766** | **3.311** | **4.252** |
| SceneTex | 0.639 | 3.009 | 3.824 |
| HY3D-2.1 | 0.734 | 2.381 | 2.774 |

### 消融实验

| 配置 | CLIP-I↑ | CLIP-FID↓ | Q-Align IQA↑ | 说明 |
|------|---------|-----------|-------------|------|
| post-SR | 0.746 | 114.612 | 2.959 | 后处理SR质量差 |
| w/o $\mathcal{L}_{SR}$ | 0.736 | 118.247 | 3.330 | 缺乏高频细节 |
| w/o multi-ref | 0.757 | 109.243 | 4.053 | 实例一致性下降+烘焙阴影 |
| w/o f-mask | 0.743 | 111.205 | 3.689 | 物体边界处光照不稳定 |
| **Full model** | **0.797** | **106.229** | **4.469** | 最优 |

### 关键发现
- **集成SR蒸馏 >> 后处理SR**：post-SR的IQA仅2.959 vs 完整模型4.469
- Feature-level mask比noise-level mask光照更稳定
- Multi-reference输入至关重要：拼接参考图导致无法区分实例
- 实例mask分解全局→局部生成是**减少烘焙阴影的关键**
- 用户研究（60人）中视觉质量和一致性评分均最高

## 亮点与洞察
- **"双蒸馏"解耦范式**：语义蒸馏负责"生成什么"，像素蒸馏负责"生成得多好"
- **Instance Cross-Attention精确对齐**：mask调制注意力实现参考图→实例区域的精准映射
- **减少烘焙阴影的洞察深刻**：实例mask分解全局为局部生成，阻止扩散模型跨图像形成统一光影
- 支持写实和艺术风格（Van Gogh、Cyberpunk）
- 推理高效：4K纹理仅2.4秒

## 局限与展望
- 训练耗时48小时（单GPU）
- 仅生成diffuse albedo纹理，不生成PBR材质（normal/roughness/metallic map）
- 依赖高质量UV展开
- 未来方向：加速训练、扩展到完整PBR材质生成

## 相关工作与启发
- 双蒸馏范式可推广到其他需同时保持语义正确和视觉质量的3D生成任务
- Instance Cross-Attention的设计可用于其他多实例/多区域条件化生成
- "SR集成到蒸馏vs后处理"的结论对SDS/VSD社区有参考价值
- GPT-4v生成参考图的text→image→texture管线提供新交互范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 双蒸馏+Instance Cross-Attention组合方案有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 定量+定性+用户研究+5组消融+闭源方法对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融分析深入，图表丰富
- 价值: ⭐⭐⭐⭐ 建立了实例级场景纹理定制新标杆，实用性强

<!-- RELATED:START -->

## 相关论文

- [Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](catalyst4d_high-fidelity_3d-to-4d_scene_editing_via_dynamic_propagation.md)
- [HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)
- [CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

<!-- RELATED:END -->
