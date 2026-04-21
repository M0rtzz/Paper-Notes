---
title: >-
  [论文解读] WordRobe: Text-Guided Generation of Textured 3D Garments
description: >-
  [ECCV 2024][人体理解][Text-to-3D Garment] 提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。
tags:
  - ECCV 2024
  - 人体理解
  - Text-to-3D Garment
  - Latent Space
  - UDF
  - ControlNet Texture
  - CLIP-guided Generation
---

# WordRobe: Text-Guided Generation of Textured 3D Garments

**会议**: ECCV 2024  
**arXiv**: [2403.17541](https://arxiv.org/abs/2403.17541)  
**代码**: 计划开源（有项目主页）  
**领域**: 3D 服装生成 / 文本驱动 3D 内容创建  
**关键词**: Text-to-3D Garment, Latent Space, UDF, ControlNet Texture, CLIP-guided Generation

## 一句话总结

提出 WordRobe，通过 coarse-to-fine 两阶段编码-解码框架学习 3D 服装 UDF 隐空间，利用弱监督 CLIP 映射网络实现文本驱动的 3D 服装生成与编辑，并利用 ControlNet 的 view-composited 属性在单次前向推理中生成视角一致的纹理贴图，速度比 Text2Tex 快 13 倍。

## 研究背景与动机

**领域现状**：3D 服装建模在虚拟试穿、游戏角色、AR/VR 体验中需求巨大。传统方法依赖 CLO 等设计工具手动建模或高端 3D 扫描仪，成本高且难以规模化。

**现有痛点**：

**参数化方法**（BCNet, SMPLicit）：受限于 SMPL 等人体模型模板，只能处理紧身衣物，服装种类有限

**非参数化方法**（ReEF, xCloth）：能建模多种样式但输出带姿态、纹理质量低，无法直接用于图形管线

**DrapeNet**：虽能学习潜空间但不支持纹理、生成不可控、编辑需要显式标签

**通用 text-to-3D**（DreamBooth3D 等）：生成服装表面质量差，NeRF/SDF 无法处理开放曲面（袖口、领口），且多视角优化极慢

**核心矛盾**：需要一种既能生成高质量无姿态（canonical T-pose）3D 服装网格、又能通过文字直觉控制形状与纹理的方法。

**本文切入角度**：分治策略——先学习服装几何的隐空间，再通过 CLIP 映射实现文本控制，最后利用 ControlNet 零样本能力高效生成纹理。

**核心 idea**：学一个好的 3D 服装 UDF 隐空间 + CLIP-to-Latent 弱监督映射 + ControlNet 前后视角合成一次性生成纹理。

## 方法详解

### 整体框架

输入文本 prompt → CLIP text encoder 得到 embedding $\psi$ → Mapping Network 预测服装 latent code $\phi \in \Omega$ → Coarse Decoder + Fine Decoder 两步解码为 UDF → Marching Cubes 提取 mesh → UV 参数化 → 前后深度图拼接送入 ControlNet → 生成 view-composited RGB 图 → 投影到 UV 纹理贴图 → 最终纹理 3D 服装。

### 关键设计

1. **Coarse-to-Fine 3D 服装隐空间**:

    - 功能：将多类 3D 服装编码到 32 维隐空间 $\Omega$，并能解码恢复高质量几何
    - 核心思路：使用 DGCNN 作为编码器 $\xi$ 将服装表面点云编码为 $\phi \in \mathbb{R}^{32}$；采用两个 MLP 解码器分工：$D_{coarse}$ 预测平滑 UDF，$D_{fine}$ 预测残差修正高频细节
    - 关键公式：$\sigma_{fine} = D_{coarse}(\phi) + D_{fine}(\phi) = \sigma_{coarse} + \sigma_{delta}$
    - 设计动机：单一解码器无法同时学好正则化隐空间和高频几何细节（如褶皱、裥褶）。Coarse decoder 负责整体形状 + 隐空间规范化，Fine decoder 负责细节补充
    - 隐空间去纠缠损失：$\mathcal{L}_{latent} = \|\Sigma_b - \mathbf{I}_k\|$，鼓励 latent 各维度独立，让单维度操控对应单一属性变化
    - Coarse 阶段损失：$\mathcal{L}_{coarse} = \lambda_{dist}\mathcal{L}_{dist} + \lambda_{grad}\mathcal{L}_{grad} + \lambda_{latent}\mathcal{L}_{latent}$
    - Fine 阶段损失：$\mathcal{L}_{fine} = \lambda_{dist}\mathcal{L}_{dist} + \lambda_{grad}\mathcal{L}_{grad}$

2. **CLIP-Guided 弱监督映射网络**:

    - 功能：训练 $MLP_{map}$ 将 CLIP embedding 映射到服装 latent code，实现文本控制
    - 核心思路：无需手动文本标注——对每件训练服装随机旋转渲染深度图，送入 ControlNet 生成服装图像，再通过 CLIP image encoder 得到 $\psi_i$；同时通过编码器 $\xi$ 得到 $\phi_i$。以 $(\psi_i, \phi_i)$ 对训练 $MLP_{map}$
    - 训练损失：简单的 L1 loss $\|\text{MLP}_{map}(\psi_i) - \phi_i\|_1$
    - 设计动机：3D 服装数据缺少文本标注。利用 ControlNet 生成逼真服装图像再过 CLIP image encoder，巧妙消除了标注需求
    - 模板 prompt 构造：随机组合 "a garment made of {silk/cotton/wool/leather}, with {vibrant/dull/bright/shiny/matte} colors"

3. **View-Composited ControlNet 纹理合成**:

    - 功能：单次前向推理为 3D 服装生成视角一致的高质量纹理贴图
    - 核心思路：发现 ControlNet 的关键属性——当多视角深度图拼为一张图输入时，生成的 RGB 图在不同视角间保持颜色和光照一致性。利用 front-back 正交投影渲染深度图 $\pi_{depth}$ 拼接为单张 1024×1024 图像，ControlNet 生成 $\pi_{rgb}$，再投影到 UV 纹理贴图
    - 设计动机：Text2Tex 等多视角优化方法慢（~5min/prompt）且会出现视角不一致和斑块伪影。本方法仅需 ~22 秒
    - 正交投影选择：透视投影在切线区域信息丢失更多；front-back 分割是服装的自然选择，减少可见接缝

### 损失函数 / 训练策略

- 编码器 $\xi$ + $D_{coarse}$ 联合训练 20 epochs（$\lambda_{dist}=1.0, \lambda_{grad}=0.3, \lambda_{latent}=0.2$）
- $D_{fine}$ 单独训练 10 epochs
- $MLP_{map}$：10 层 MLP，含 skip connection，用 AdamW 优化
- 训练数据：来自 [20] 的约 20,000 件无姿态服装，19 类，12 类训练 / 7 类测试

## 实验关键数据

### 主实验——服装几何质量

| 方法 | CD↓ | P2S↓ |
|------|-----|------|
| DrapeNet | 1.796 | 0.573 |
| Ours (Single Stage) | 1.631 | 0.494 |
| **Ours (Full)** | **1.078** | **0.329** |

相比 DrapeNet，CD 降低约 40%，P2S 降低约 42%。

### 跨数据集泛化（CLOTH3D）

| 方法 | CD (topwear)↓ | P2S (topwear)↓ |
|------|--------------|----------------|
| DrapeNet (trained on CLOTH3D) | 1.522 | **0.631** |
| Ours (trained on [20]) | **1.491** | 0.635 |

仅在 [20] 数据集训练、在 CLOTH3D 测试仍可媲美甚至超越在 CLOTH3D 上训练的 DrapeNet。

### 纹理合成对比

| 方法 | CLIP Score (ViT-H/14)↑ | 速度 |
|------|----------------------|------|
| Text2Tex | 0.263 ± 0.047 | ~5 min |
| **WordRobe** | **0.304 ± 0.043** | **~22 sec** |

快 13 倍，CLIP Score 更高，视角一致性更好。

### 消融实验

| 配置 | CD↓ | P2S↓ | 说明 |
|------|-----|------|------|
| w/o $\mathcal{L}_{grad}$ | 1.886 | 0.612 | 缺乏梯度正则化 |
| w/o $\mathcal{L}_{latent}$ | 1.094 | 0.331 | 去纠缠损失辅助但非决定性 |
| Full | **1.078** | **0.329** | 最优 |

| 插值评估 | $\Delta_{area}$↓ | $\Delta_{vol}$↓ |
|---------|-----------------|----------------|
| w/o $\mathcal{L}_{latent}$ | 0.028 | 1.275 |
| with $\mathcal{L}_{latent}$ | **0.022** | **1.206** |

### 关键发现
- Coarse-to-fine 两阶段解码显著减少表面噪声和孔洞
- $\mathcal{L}_{grad}$ 对减少高频噪声起关键正则化作用
- $\mathcal{L}_{latent}$ 对 CD/P2S 影响不大但显著提升插值质量
- 用户研究中 63% 用户偏好 WordRobe（vs 27% 偏好 DreamFusion 变体）

## 亮点与洞察

- **弱监督 CLIP 映射方案**：利用 ControlNet 生成→CLIP 编码来获取训练对，完全避免手动文本标注，思路巧妙且通用
- **ControlNet 的 view-composited 属性**：这个经验发现非常实用——将多视角深度图组合到一张图中输入 ControlNet，输出天然保持视角一致性。这是一种不需多视角优化的 texture generation 新范式
- **Canonical T-pose 生成的实用性**：直接对接标准动画/仿真管线（rigging, skinning, cloth simulation），有工业应用价值
- **CLIP arithmetic 用于 latent editing**：利用 CLIP 文本-文本的向量算术来自动定位 latent 中需要修改的维度，无需显式标注

## 局限与展望

- front-back 正交投影在切线区域会丢失纹理信息，需要 inpainting 填补可能产生模糊接缝
- UDF 隐式表示难以建模精细几何细节（口袋、纽扣等）
- 纹理合成存在虚假阴影/光照/边缘幻觉，限制了在新光照环境下的适用性
- 仅处理单件服装，尚不支持分层穿着
- 训练和评估数据均为合成数据（CLOTH3D, [20]），在真实世界服装上的泛化有待验证

## 相关工作与启发

- **vs DrapeNet**: 同样学习服装 UDF 隐空间，但 DrapeNet 无纹理、无文本控制、编辑需显式标签。WordRobe 在所有方面全面升级
- **vs Text2Tex**: Text2Tex 用多视角渐进 inpainting 生成纹理，慢且视角不一致。WordRobe 单次前向、快 13 倍
- **vs DreamBooth3D/DreamFusion**: 通用 text-to-3D 方法用 SDF 无法处理服装的开放曲面，几何质量远不及专用方法

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个文本驱动的无姿态纹理 3D 服装生成框架，coarse-to-fine + 弱监督 CLIP 映射 + view-composited texture 三大创新点
- 实验充分度: ⭐⭐⭐⭐ 定量对比、用户研究、跨数据集泛化、消融分析覆盖全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法表述完整，图示丰富
- 价值: ⭐⭐⭐⭐ 工业应用价值高，view-composited 纹理生成思路可广泛迁移

<!-- RELATED:START -->

## 相关论文

- [TELA: Text to Layer-wise 3D Clothed Human Generation](tela_text_to_layer-wise_3d_clothed_human_generation.md)
- [A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)
- [FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)

<!-- RELATED:END -->
