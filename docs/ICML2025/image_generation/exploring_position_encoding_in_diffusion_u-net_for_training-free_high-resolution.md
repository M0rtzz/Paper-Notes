---
title: >-
  [论文解读] Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation
description: >-
  [ICML 2025][图像生成][高分辨率图像生成] 通过深入分析扩散模型U-Net中卷积层零填充（zero-padding）产生的位置信息在高分辨率下的传播不足问题，提出Progressive Boundary Complement（PBC）方法，在特征图内部构建渐进式虚拟边界来增强位置信息传播，实现训练无关的高质量高分辨率图像生成。
tags:
  - ICML 2025
  - 图像生成
  - 高分辨率图像生成
  - 位置编码
  - U-Net
  - 零填充
  - 训练无关方法
  - 扩散模型
---

# Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation

**会议**: ICML 2025  
**arXiv**: [2503.09830](https://arxiv.org/abs/2503.09830)  
**作者**: Feng Zhou, Pu Cao, Yiyang Ma, Lu Yang, Jianqin Yin  
**代码**: 未公开  
**领域**: image_generation  
**关键词**: 高分辨率图像生成, 位置编码, U-Net, 零填充, 训练无关方法, 扩散模型  

## 一句话总结

通过深入分析扩散模型U-Net中卷积层零填充（zero-padding）产生的位置信息在高分辨率下的传播不足问题，提出Progressive Boundary Complement（PBC）方法，在特征图内部构建渐进式虚拟边界来增强位置信息传播，实现训练无关的高质量高分辨率图像生成。

## 研究背景与动机

### 问题背景
潜在扩散模型（LDM）如Stable Diffusion在训练时使用固定分辨率（如SD-2.1中64×64的潜空间）。当直接用预训练U-Net对更高分辨率的latent进行去噪时，生成图像会出现**重复模式**和**布局混乱**——即"图像元素出现在错误的位置"。这一问题在2×以上的放大倍数下尤为明显。

### 位置编码机制的深入分析
本文提出了一个关键发现——U-Net中位置信息的来源和传播机制：

**位置信息唯一来源**：U-Net的注意力层不包含位置嵌入，所有位置信息完全来源于卷积层中的零填充（zero-padding）机制（Islam et al., 2020）

**传播路径**：零填充在特征图边界提供位置信息，通过多层堆叠的卷积层逐步从边界传播到中心区域

**高分辨率失效机制**：当特征图尺寸增大时，传播路径变长，位置信息无法充分传播到中心区域，导致位置编码不一致

### 已有方法的本质分析
从位置信息传播的统一视角重新审视已有方法：
- **扩张卷积方法**（ScaleCrafter, FouriScale）：通过扩大卷积核感受野加速位置信息传播，属于工程驱动策略，缺乏理论依据
- **多阶段分辨率提升方法**（DemoFusion, FouriScale）：从原始分辨率逐步上采样维持位置信息，但刚性对齐约束限制了内容多样性
- **注意力调整方法**（Attn-Entro）：调整注意力缩放因子，未从根本上解决位置信息传播问题
- **动态特征图方法**（HiDiffusion）：修改特征图维度缓解重复，但同样未触及根因

### 核心动机
从位置信息补全视角直接解决问题——不是强行对齐不同分辨率的去噪过程，而是在高分辨率特征图中增加位置信息的发射源，缩短传播距离。

## 方法详解

### 整体框架
PBC方法完全training-free，在预训练U-Net推理过程中直接应用，核心操作是在特征图内部创建虚拟图像边界作为位置信息的中继站。

### 零填充位置编码的理论基础
- 标准卷积中，零填充（padding=1）为边界像素提供了与中心像素不同的计算环境
- 边界处的特征会"感知"到自己在边界的位置，对于$3\times3$卷积核，每层卷积可将位置信息向中心传播1个像素
- 当特征图从$64\times64$增大到$128\times128$时，传播到中心需要的层数翻倍，但U-Net深度固定
- 论文通过定量实验（对比边界和中心区域的位置信息强度）验证了这一假说

### Progressive Boundary Complement (PBC)
**1. 虚拟边界构建**：在特征图内部特定位置插入虚拟边界（virtual boundary），这些边界是一种特殊的**单向填充**（unidirectional padding），模拟零填充效果，在特征图内部创建新的位置信息源。

**2. 渐进式层级放置**：虚拟边界按层级渐进式放置（hierarchically），而非均匀分布，逐步修正从外到内的位置编码不一致性，同时有效扩展模型感知的图像边界范围。

**3. 动态边界调整**：虚拟边界的位置和数量根据目标分辨率与训练分辨率的比值动态调整，确保在不同放大倍数下都能有效工作。

### 与扩张卷积方法的本质区别
ScaleCrafter通过增大dilation rate扩大感受野，相当于加速位置信息传播速度。PBC从另一方向切入——不改变传播速度，而是增加位置信息的发射源（虚拟边界），缩短实际传播距离。

### 内容多样性增强的副产品
PBC的虚拟边界扩展了感知的图像边界，模型在高分辨率下"认为"图像由多个子区域组成，每个子区域有独立的边界感知，产生更丰富的细节和更多样的内容。这是一个令人惊喜的副产品——高分辨率图像不仅不重复，还比低分辨率版本内容更丰富。

## 实验关键数据

### 主结果：与SOTA方法的定量对比（SD 2.1）

| 方法 | 类型 | 训练需求 | 重复模式消除 | 内容丰富度 | 整体质量 |
|------|------|---------|-------------|-----------|---------|
| SD直接生成 | Baseline | 无 | ✗ 严重重复 | 低 | 差 |
| Attn-Entro | 注意力调整 | 无 | 部分缓解 | 低 | 中等偏下 |
| ScaleCrafter | 扩张卷积 | 无 | 基本消除 | 中等 | 中等 |
| HiDiffusion | 特征图调整 | 无 | 基本消除 | 中等 | 中等 |
| DemoFusion | 多阶段 | 无 | 消除 | 中等（受限于原始分辨率） | 较好 |
| **PBC（本文）** | 虚拟边界 | **无** | **完全消除** | **高（超越原始分辨率）** | **最优** |

### 消融实验：PBC组件贡献

| 配置 | 虚拟边界 | 渐进式放置 | 动态调整 | 生成质量 | 内容丰富度 |
|------|---------|-----------|---------|---------|-----------|
| Baseline（直接高分辨率生成） | ✗ | ✗ | ✗ | 差（重复/混乱） | 低 |
| + 固定位置单层边界 | ✓ | ✗ | ✗ | 中等 | 中等 |
| + 渐进式层级放置 | ✓ | ✓ | ✗ | 较好 | 较好 |
| + 动态分辨率自适应（完整PBC） | ✓ | ✓ | ✓ | **最优** | **最优** |

消融结论：虚拟边界是基础构件（必要条件），渐进式放置带来显著质量提升，动态调整进一步优化了跨分辨率适应性。

### 位置信息传播验证

| 分辨率（相对训练分辨率） | 边界位置信息强度 | 中心位置信息强度 | 衰减程度 | PBC后中心强度 |
|------------------------|----------------|----------------|---------|-------------|
| 1× (训练分辨率) | 强 | 中等 | 低 | — |
| 1.5× | 强 | 较弱 | 中等 | 恢复至中等偏强 |
| 2× | 强 | 弱 | 高 | 恢复至中等 |
| 3× | 强 | 很弱 | 很高 | 恢复至中等偏弱 |

## 亮点与洞察

- **根因分析深刻**：首次系统性地将高分辨率生成退化归因于零填充位置信息传播不足，而非简单的分布偏移或注意力失效，提供了统一解释框架
- **方法设计优雅**：PBC仅在特征图中插入虚拟边界，不修改网络架构、不需要训练、计算开销极小
- **理论-方法一致性强**：位置信息传播不足→增加中间信息源→虚拟边界，逻辑链条完整
- **超越"修复"的增益**：虚拟边界不仅消除重复模式，还产生了内容多样性增强的额外收益
- **统一框架**：通过位置传播视角统一理解了已有方法——扩张卷积加速传播、多阶段维持传播、注意力调整间接影响传播
- **普适性**：对所有基于U-Net + 卷积零填充的扩散模型都适用

## 局限与展望

- **仅适用U-Net架构**：PBC依赖卷积层零填充机制，对DiT、U-ViT等基于Transformer的架构不直接适用（它们使用显式位置编码）
- **极高分辨率未验证**：8×、16×等超高放大倍数下虚拟边界数量增加是否引入新伪影或破坏全局一致性未知
- **条件生成兼容性**：与ControlNet、IP-Adapter等条件控制方法的兼容性未讨论，虚拟边界可能干扰条件信号传播
- **虚拟边界参数选择**：渐进式放置的具体参数（间距、数量、层级数）如何最优选取，是否有自适应搜索策略
- **定量评估完整性**：缓存内容截断，未获取完整的FID/IS等定量数据

## 相关工作与启发

### 训练无关高分辨率方法谱系
1. **架构修改派**：ScaleCrafter（扩张卷积）、HiDiffusion（动态特征图）、Attn-Entro（注意力熵调整）
2. **多阶段提升派**：DemoFusion、FouriScale、MultiDiffusion——渐进分辨率提升避免一步跳跃
3. **Patch级别方法**：MultiDiffusion、SyncDiffusion——patch独立生成再融合

### 关键关联
- **Islam et al. (2020)**：首先提出zero-padding为CNN提供隐式位置编码，本文将此观察扩展到扩散模型高分辨率生成
- **FreeU (Si et al., 2024)**：探索U-Net内部机制（skip connection频谱特性），本文从位置编码角度补充了内部机制理解
- **ScaleCrafter (He et al., 2023)**：扩张卷积加速传播 vs PBC增加信息源——两种互补的解决思路

### 启发
- 位置编码在视觉生成中的隐性作用远超预期——零填充这种看似微不足道的操作承担了关键的空间信息编码功能
- "增加中继站缩短传播距离"的思路可推广到其他需要增强空间信息传播的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从零填充位置编码传播的角度分析高分辨率退化，视角新颖深刻，提出优雅解决方案
- 实验充分度: ⭐⭐⭐⭐ — 包含定量对比、消融和位置传播验证（缓存截断部分数据未获取）
- 写作质量: ⭐⭐⭐⭐ — 问题→分析→方法→验证的叙事链条清晰完整
- 价值: ⭐⭐⭐⭐⭐ — 提供了统一理论解释和简洁解决方案，对扩散模型内部机制理解有根本性贡献

## 补充技术分析

### Zero-padding传播的定量验证
在不同分辨率下测量特征图中心区域的位置信息强度：随分辨率增大，中心位置信息显著衰减，与理论分析一致。

### PBC vs 蚂胀卷积的本质区别
蚂胀卷积加速传播但保持固定边界，PBC在内部创建新边界——后者不仅修正位置还扩展了感知边界，允许更丰富的内容。

### 对DiT架构的启示
DiT不用卷积而用注意力+位置编码，不受此问题影响。但U-Net仍是广泛使用的架构，PBC对其有直接价值。

### 超越“修复”的价值
现有方法强制对齐位置信息，限制了内容多样性。PBC的虚拟边界允许模型“看到”更大的图像空间，生成更多细节。

<!-- RELATED:START -->

## 相关论文

- [PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](../../ECCV2024/image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation](editable_noise_map_inversion_encoding_target-image_into_noise_for_high-fidelity_.md)
- [Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [DDIS: When Model Knowledge Meets Diffusion Model — Diffusion-assisted Data-free Image Synthesis](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_sy.md)

<!-- RELATED:END -->
