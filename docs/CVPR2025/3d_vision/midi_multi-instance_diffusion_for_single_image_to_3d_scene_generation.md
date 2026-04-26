---
title: >-
  [论文解读] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation
description: >-
  [CVPR 2025][3D视觉][3D场景生成] MIDI 将预训练的 image-to-3D 单物体生成模型扩展为多实例扩散模型，通过新颖的多实例注意力机制在 3D 生成过程中直接捕获物体间的空间交互关系，从单张图片同时生成多个具有正确空间布局的 3D 实例，在合成和真实数据集上均大幅超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D场景生成
  - 多实例扩散
  - 单图3D重建
  - 空间关系
  - 注意力机制
---

# MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation

**会议**: CVPR 2025  
**arXiv**: [2412.03558](https://arxiv.org/abs/2412.03558)  
**代码**: https://github.com/huanngzh/MIDI-Page (项目页面)  
**领域**: 3D视觉 / 3D场景生成  
**关键词**: 3D场景生成, 多实例扩散, 单图3D重建, 空间关系, 注意力机制

## 一句话总结
MIDI 将预训练的 image-to-3D 单物体生成模型扩展为多实例扩散模型，通过新颖的多实例注意力机制在 3D 生成过程中直接捕获物体间的空间交互关系，从单张图片同时生成多个具有正确空间布局的 3D 实例，在合成和真实数据集上均大幅超越现有方法。

## 研究背景与动机

1. **领域现状**：从单张图片生成 3D 场景是计算机视觉的核心挑战。现有方法分为三类：(a) 前馈重建方法（如 Total3D、InstPIFu）通过神经网络直接回归 3D 几何；(b) 检索方法（如 DiffCAD）从数据库中检索并对齐 3D 模型；(c) 组合生成方法（如 REPARO、Gen3DSR）分割场景→补全物体图像→逐物体 3D 生成→优化空间关系。

2. **现有痛点**：前馈方法受限于场景级训练数据稀缺，泛化性差；检索方法依赖数据库覆盖率，无法精确匹配；组合生成方法流程冗长（分割→补全→生成→优化），错误逐步累积，且逐物体生成时缺乏全局场景上下文导致空间关系不协调。

3. **核心矛盾**：预训练的 image-to-3D 单物体生成模型质量很高，但独立地逐个生成物体无法建模物体间的空间关系。后续的空间关系优化无法反过来改善已生成的 3D 物体。

4. **本文目标** 如何在高质量 3D 物体生成的同时，端到端地建模多个物体之间的空间关系。

5. **切入角度**：不是将 3D 生成模型当作"工具"在多阶段流水线中使用，而是直接在生成模型内部引入多实例交互机制，让多个物体在去噪过程中同时生成并协调空间关系。

6. **核心 idea**：将预训练 3D 物体扩散模型的 self-attention 扩展为 multi-instance attention，让多个实例的 latent token 在去噪时互相查询，从而在生成过程中直接建模空间关系。

## 方法详解

### 整体框架
MIDI 基于预训练的 3D 物体生成模型（VAE + DiT），输入一张场景图片，分割出 $N$ 个物体的图像和 mask。对 $N$ 个物体的 3D latent 表示同时进行去噪——共享 DiT 权重、共享噪声水平，但各自有独立的噪声。关键扩展是在 DiT 中加入 multi-instance attention 层以建模跨实例交互，并通过 cross-attention 注入每个物体的局部图像和全局场景上下文信息。

### 关键设计

1. **多实例注意力机制（Multi-Instance Attention）**:

    - 功能：在 3D 生成的去噪过程中直接建模多个实例之间的空间关系
    - 核心思路：将 DiT 中的部分（$K=5$ 层）object self-attention 扩展为 multi-instance attention。原本每个实例的 token 只能查询自己，现在每个实例的 token 可以查询场景中所有实例的 token：$f^i_{out} = \text{Attention}(f^i, \{f^j\}_{j=1}^N)$。这使得注意力机制能考虑所有实例 token 的集合，捕获物体间的空间依赖关系。
    - 设计动机：$K=0$ 时（无多实例注意力），模型无法捕获正确的空间关系；$K=21$（所有层都转换）时会过拟合并破坏预训练先验导致物体几何扭曲。$K=5$ 在建模交互和保留预训练先验之间取得最佳平衡。

2. **全局-局部图像条件编码**:

    - 功能：为每个实例提供局部物体信息和全局场景上下文
    - 核心思路：对每个实例 $z^i$，将其 RGB 图像 $c_l^i$、mask $m_l^i$ 和全局场景图 $c_g$ 沿通道维度拼接为 7 通道（3+1+3）的复合表示 $y \in \mathbb{R}^{h \times w \times 7}$。用扩展了输入通道的 DINO ViT 编码，通过 cross-attention 注入 DiT。
    - 设计动机：局部物体图像提供物体外观信息，全局场景图提供空间布局和上下文。去掉全局场景图会导致物体放置错误和空间关系缺失（消融验证）。

3. **混合训练策略（Mixed Training）**:

    - 功能：在有限的场景数据上微调的同时保持预训练模型的泛化能力
    - 核心思路：训练时以 30% 的概率将多实例扩散模型退化为标准的单物体生成模型——关闭 multi-instance attention，在 Objaverse 单物体数据上训练。这起到正则化作用，防止模型在较小的场景数据集上过拟合。使用 LoRA 进行参数高效微调。
    - 设计动机：3D-Front 场景数据集仅约 15000 个场景，远小于预训练使用的大规模单物体数据。不加正则化会导致物体几何质量退化。

### 损失函数 / 训练策略
- 采用 rectified flow 架构，对所有实例共享噪声水平 $t$，沿线性轨迹扰动。
- 损失函数：$\mathbb{E}\left[\sum_{i=1}^N \|z_0^i - \epsilon^i - \epsilon_\theta(z_t^i, t, \tau_\theta(y))\|_2^2\right]$
- 使用 LoRA 高效微调预训练模型；DINO 初始化图像编码器并扩展输入通道到 7

## 实验关键数据

### 主实验

**3D-Front（合成数据）**：

| 方法 | CD-S↓ | F-Score-S↑ | CD-O↓ | F-Score-O↑ | IoU-B↑ | Runtime |
|------|-------|------------|-------|------------|--------|---------|
| InstPIFu | 0.138 | 39.99 | 0.165 | 38.11 | 0.299 | 32s |
| DiffCAD | 0.117 | 43.58 | 0.190 | 37.45 | 0.392 | 64s |
| REPARO | 0.129 | 41.68 | 0.160 | 40.85 | 0.339 | 4min |
| **MIDI** | **0.080** | **50.19** | **0.103** | **53.58** | **0.518** | 40s |

**BlendSwap（合成数据）**：

| 方法 | CD-O↓ | F-Score-O↑ | IoU-B↑ |
|------|-------|------------|--------|
| REPARO | 0.151 | 42.84 | 0.410 |
| Gen3DSR | 0.148 | 40.76 | 0.449 |
| **MIDI** | **0.090** | **62.94** | **0.663** |

物体级 F-Score 提升 12+ 点，空间布局 IoU 提升 15+ 点。

### 消融实验

| #K | Scene | Objaverse | CD-S↓ | F-Score-S↑ | CD-O↓ | F-Score-O↑ | IoU-B↑ |
|----|-------|-----------|-------|------------|-------|------------|--------|
| 0 | ✓ | ✓ | 0.145 | 40.94 | 0.096 | 54.16 | 0.327 |
| **5** | **✓** | **✓** | **0.080** | **50.19** | **0.103** | **53.58** | **0.518** |
| 21 | ✓ | ✓ | 0.127 | 44.88 | 0.141 | 48.55 | 0.423 |
| 5 | ✗ | ✓ | 0.134 | 41.49 | 0.102 | 52.91 | 0.459 |
| 5 | ✓ | ✗ | 0.137 | 42.00 | 0.126 | 51.62 | 0.502 |

### 关键发现
- **多实例注意力是核心**：$K=0$（无 multi-instance attention）时 IoU-B 仅 0.327，$K=5$ 时跃升至 0.518，证明空间关系建模的关键作用
- **注意力层数需要平衡**：$K=5$ 最优，$K=21$（所有层都转换）反而下降，因为过多 attention 层在小规模场景数据上过拟合，破坏了预训练的 3D 先验
- **全局场景图不可或缺**：去掉全局场景图后 CD-S 从 0.080 升至 0.134，空间布局严重退化
- **混合训练作为正则化**：不混合 Objaverse 数据会导致物体几何质量下降（CD-O 从 0.103 升至 0.126）
- **运行效率显著**：MIDI 40s vs REPARO 4min vs Gen3DSR 9min，快 6-13 倍

## 亮点与洞察
- **从"工具使用"到"能力内化"的范式转变**：之前的组合生成方法将 3D 生成模型当作多阶段流水线中的工具，MIDI 直接在模型内部建模空间关系。这个思路——把原来在后处理/优化阶段做的事情内化到生成过程中——是一个重要的设计范式。
- **部分层转换而非全转换**：仅将 5/21 层 self-attention 转为 multi-instance attention，既建模了交互又保留了预训练先验。这种"最小侵入性"的微调策略值得学习——对预训练模型的修改越少越好。
- **混合训练防止遗忘**：30% 概率退回单物体模式训练，类似于多任务学习中的"回放"策略，有效防止场景微调导致的物体生成能力退化。

## 局限与展望
- **不处理背景**：地板、墙壁等平面背景不在生成范围内，需要额外方法补充
- **物体数量受限**：当前实验主要处理少量物体（2-5个），大量物体场景的可扩展性未验证
- **低分辨率输入表现差**：论文自述在小尺寸物体图像输入时表现不佳
- **不建模复杂交互**：如人与物体的交互（"熊猫弹吉他"）需要专门的数据集和设计
- **依赖场景分割**：输入图片需要已分割的物体 mask，分割质量直接影响生成结果
- 改进思路：扩展到开放世界场景生成；引入显式 3D 几何先验增强 attention 效率；探索模型隐含的 3D 感知能力

## 相关工作与启发
- **vs REPARO [Lu et al.]**: REPARO 采用典型的多阶段组合生成（分割→补全→逐物体生成→优化），MIDI 省略了物体图像补全和后优化阶段，直接在扩散模型内完成多实例协调，避免了错误累积。MIDI 在所有指标上大幅超越 REPARO（IoU-B: 0.518 vs 0.339）。
- **vs Gen3DSR [Li et al.]**: Gen3DSR 与 REPARO 类似采用多阶段流水线，且需要 9 分钟处理一个场景。MIDI 仅需 40 秒且效果更好。
- **vs DiffCAD [Choi et al.]**: DiffCAD 属于检索方法，使用扩散模型辅助 CAD 检索和对齐。受限于数据库覆盖率且无法生成数据库中不存在的物体形状。MIDI 具有生成能力，不受数据库限制。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多实例扩散的范式确实是新的，将物体间空间关系内化到生成过程中是优雅的解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 在合成、真实、风格化三类数据上验证，消融全面且有说服力
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，与已有方法的对比图（Fig. 2）非常直观
- 价值: ⭐⭐⭐⭐⭐ 开辟了 3D 场景生成的新范式，对后续工作有很强的指导意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation](scenefactor_factored_latent_3d_diffusion_for_controllable_3d_scene_generation.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)

<!-- RELATED:END -->
