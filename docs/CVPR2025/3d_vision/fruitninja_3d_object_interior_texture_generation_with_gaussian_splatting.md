---
title: >-
  [论文解读] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] FruitNinja 首次提出为 3DGS 物体生成内部纹理的方法，通过渐进式截面修复 + 体素平滑 + OpaqueAtom GS 策略，实现切割后实时渲染无需额外优化，在语义对齐和纹理一致性上显著优于基线。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 内部纹理生成
  - 截面修复
  - 扩散模型引导
  - 实时渲染
---

# FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2411.12089](https://arxiv.org/abs/2411.12089)  
**代码**: 无（计划发布）  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 内部纹理生成, 截面修复, 扩散模型引导, 实时渲染

## 一句话总结
FruitNinja 首次提出为 3DGS 物体生成内部纹理的方法，通过渐进式截面修复 + 体素平滑 + OpaqueAtom GS 策略，实现切割后实时渲染无需额外优化，在语义对齐和纹理一致性上显著优于基线。

## 研究背景与动机

**领域现状**：3D Gaussian Splatting (3DGS) 已成为高效的新视角合成方法，并被广泛用于 3D 编辑任务（风格化、变形、物体移除、修复等）。交互式 3D 应用中，用户经常需要对物体进行切割、撕裂等几何操作。

**现有痛点**：当前所有 3DGS 编辑方法都只关注物体外表面的编辑。当物体被切开或切割时，暴露出的内部纹理是未训练的，呈现不真实的随机噪声色彩。现有方案要么简单地从表面向内填充相同颜色（PhysGaussian，假设内外纹理一致），要么在每次编辑后逐一进行 2D 修复（VR-GS/Infusion，需要 ~30 秒且多次编辑间不一致）。

**核心矛盾**：获取物体完整内部结构的数据极其困难（需要 CT 扫描或多次破坏性切割），但真实物体的内部纹理通常与外表面截然不同（如西瓜的绿皮红肉）。现有方法要么假设内外一致，要么无法保证多视角一致性。

**本文目标** 在无完整内部数据的情况下，为 3DGS 物体合成逼真的内部纹理，且支持任意角度切割的实时渲染。

**切入角度**：利用常见物体的截面对称性——同一方向不同位置的截面通常呈现相似的纹理模式。只需少量典型截面视角作为参考，配合扩散模型生成引导，就能推广到整个内部空间。

**核心 idea**：用预训练扩散模型通过 SDS 损失渐进式修复少量截面参考视角，配合 OpaqueAtom GS 约束实现精细内部纹理的稳定训练和实时渲染。

## 方法详解

### 整体框架
输入是已重建好的 3DGS 物体模型。首先用 OpaqueAtom GS 策略改造高斯粒子（限制大小+设置全不透明），并在物体内部空区域填充原始粒子。然后针对用户定义的每个切割角度，通过 SDS 优化生成截面参考图。接着用这些参考图联合外表面视角训练 3DGS，同时渐进式细化参考图并定期进行体素平滑，最终得到内外纹理一致的完整 3D 模型。

### 关键设计

1. **OpaqueAtom GS（不透明原子高斯策略）**:

    - 功能：确保内部纹理训练的稳定性和精细度，解决标准 3DGS 的两个关键缺陷
    - 核心思路：两个约束——(a) Atomic Clipping：将每个高斯粒子的尺度上限设为物体尺寸的 1/3000，防止大粒子跨越多个截面区域造成训练冲突；(b) Uniform Opacification：所有高斯粒子设为完全不透明（opacity=1），确保前方粒子完全遮挡后方，避免前后混色
    - 设计动机：标准 3DGS 倾向于优化出大高斯来覆盖更多面积，但大高斯无法精细表达纹理且切割时无法处理。半透明混合在几何编辑后会产生颜色伪影（如绿皮和红肉混色）

2. **渐进式截面修复（Conditioned Cross-section Inpainting）**:

    - 功能：为每个切割角度生成高质量的内部截面参考图
    - 核心思路：两阶段优化——第一阶段对每个截面独立进行 SDS 优化，用深度条件 Stable Diffusion 配合角度特定文本提示（如"watermelon 的水平截面"）生成初始参考图；第二阶段用参考图作为目标，通过 $\mathcal{L}_{recon} = \alpha \mathcal{L}_{MSE} + (1-\alpha)\mathcal{L}_{SSIM}$ 联合训练 3DGS 参数。可选用 DreamBooth 微调扩散模型以适应截面图像这一稀缺领域
    - 设计动机：直接对未训练的内部高斯做 SDS 优化效率很低（初期渲染无特征），两阶段策略先生成可靠的 2D 参考再指导 3D 优化

3. **体素平滑与渐进式纹理细化（Voxel Smoothing & Progressive Refinement）**:

    - 功能：解决不同截面视角间的空间不一致性，并平滑未被任何截面覆盖的区域
    - 核心思路：纹理细化——每轮迭代后重新渲染当前截面，再对参考图施加几步 SDS 更新，直到所有截面的重建损失收敛到阈值以下。体素平滑——构建 512³ 体素网格，每隔 30-40 轮对未训练的高斯用距离加权平均赋色 $C = \sum_i w_i C_i / \sum_i w_i$
    - 设计动机：不同方向的截面可能在交叉区域产生冲突信号（如垂直切面显示种子但水平切面同位置只有果肉），迭代细化让 3DGS 和扩散模型逐步达成一致

### 损失函数 / 训练策略
- SDS 损失用于截面参考图生成：$\mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}[w(t)\|\epsilon - \epsilon_\theta(\mathbf{I}_{label}^p + \sigma_t \epsilon, t, e, d)\|^2]$
- 重建损失用于 3DGS 训练：$\mathcal{L}_{recon} = \alpha \mathcal{L}_{MSE} + (1-\alpha)\mathcal{L}_{SSIM}$
- 每轮随机选取 20 个外表面视角联合训练，保证外观不退化
- 训练 120-200 轮，每个参考视角初始 20 步 SDS + 每轮 3-4 步细化

## 实验关键数据

### 主实验

| 方法 | CLIP Score↑ | FID↓ | KID↓(×10⁻³) |
|------|-------------|------|-------------|
| **FruitNinja** | **33.1** | 209.2 | 323.7 |
| 2D Inpainting (微调) | 32.3 | **176.2** | **224.5** |
| 2D Inpainting | 25.1 | 314.2 | 536.3 |
| PhysGaussian | 24.6 | 520.1 | 816.4 |

FruitNinja 在 CLIP Score（语义对齐）上最高，KID/FID 比 PhysGaussian 好 ~60%，与微调 2D Inpainting 可比但后者需要逐帧 ~30 秒优化。

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| w/o Progressive Refinement | 截面间出现冲突纹理（种子位置不一致），噪声明显 |
| w/o Atomic Clipping | 3D 模型难以收敛，无法生成与参考对齐的真实纹理 |
| w/o Uniform Opacity | 无法准确表达尖锐颜色过渡（绿皮-白肉-红肉界面糊混） |
| Full OpaqueAtomGS | 收敛稳定，纹理过渡清晰 |

| 方法 | CLIP Score↑ | Cosine Similarity↑ |
|------|-------------|---------------------|
| **FruitNinja** | **29.1** | **0.96** |
| 2D Inpainting | 27.8 | 0.87 |
| PhysGaussian | 23.9 | 0.89 |

120 个随机角度切割的一致性测试中，FruitNinja 余弦相似度 0.96，远超 2D Inpainting 的 0.87。

### 关键发现
- 渐进式细化是解决截面间冲突的关键——没有它，不同方向截面在交叉区域会给粒子相矛盾的训练信号
- Atomic Clipping 对训练稳定性至关重要——大高斯粒子跨区域会导致优化不收敛
- OpaqueAtom 的两个组件（原子限制 + 不透明）分别解决不同问题：前者保证精度和稳定性，后者保证颜色过渡的锐利度

## 亮点与洞察
- **截面对称性假设**：利用常见物体的天然对称性（水平/垂直切面相似），只需少量切割角度就能推广到全内部——这个观察简洁有效
- **OpaqueAtom 设计**：对 3DGS 缺陷的分析非常精准——大粒子倾向和半透明混合分别造成不同问题，两个简单约束各解决一个
- **零额外优化的实时渲染**：内部纹理在训练期间就已嵌入 3DGS，推理时任意切割直接渲染，这比逐帧修复快几个数量级

## 局限与展望
- 需要人工指定切割角度和文本提示，自动化程度有限
- 仅验证了 6 种常见物体（水果/蛋糕/面包），对结构更复杂的物体（机械零件、生物组织）效果未知
- DreamBooth 微调需要 1-4 张真实截面图片，仍有数据收集成本
- SDS 优化和体素平滑的迭代训练仍需较长时间（120-200 轮），可探索更高效的引导策略

## 相关工作与启发
- **vs PhysGaussian**: PhysGaussian 简单地将表面颜色复制到内部（假设内外一致），效果模糊不自然。FruitNinja 通过扩散模型生成语义合理的内部纹理
- **vs 2D Inpainting (VR-GS/Infusion)**: 逐帧修复需要 ~30s/帧且多次编辑间不一致。FruitNinja 预先生成好内部纹理，实时可用
- **vs AtomGS**: FruitNinja 借鉴了 AtomGS 的小高斯致密化思想用于内部建模，并增加了不透明约束

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次解决 3DGS 内部纹理生成问题，问题定义新颖且实用
- 实验充分度: ⭐⭐⭐ 仅 6 个物体，数据集较小，缺少与更多方法的对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图例直观
- 价值: ⭐⭐⭐⭐ 对 VR/游戏交互场景有直接价值，开创了新研究方向

<!-- RELATED:START -->

## 相关论文

- [Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

<!-- RELATED:END -->
