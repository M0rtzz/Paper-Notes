---
title: >-
  [论文解读] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][稀疏视角新视角合成] DropGaussian 提出了一种无需额外先验的简单正则化方法，通过在 3DGS 训练中随机丢弃高斯并引入不透明度补偿因子，使被遮挡的远距离高斯获得更大梯度和可见性，并采用渐进式丢弃率策略有效缓解稀疏视角下的过拟合问题，在不增加计算复杂度的情况下达到与先验方法可比的性能。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "稀疏视角新视角合成"
  - "3D高斯泼溅"
  - "过拟合正则化"
  - "Dropout"
  - "无先验方法"
---

# DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2504.00773](https://arxiv.org/abs/2504.00773)  
**代码**: [https://github.com/DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)  
**领域**: 3D视觉  
**关键词**: 稀疏视角新视角合成, 3D高斯泼溅, 过拟合正则化, Dropout, 无先验方法

## 一句话总结
DropGaussian 提出了一种无需额外先验的简单正则化方法，通过在 3DGS 训练中随机丢弃高斯并引入不透明度补偿因子，使被遮挡的远距离高斯获得更大梯度和可见性，并采用渐进式丢弃率策略有效缓解稀疏视角下的过拟合问题，在不增加计算复杂度的情况下达到与先验方法可比的性能。

## 研究背景与动机

1. **领域现状**：3DGS 在新视角合成中展现出实时渲染和高质量图像的优势，但在稀疏视角（如仅 3 个输入视角）下性能严重退化，原因是对训练视角的过拟合。
2. **现有痛点**：现有方法主要通过引入先验信息来缓解过拟合：（a）单目深度估计作为外部监督（DNGaussian），但不同视角深度尺度不一致且误差会传播；（b）2D 生成模型先验（FSGS），但计算成本高且优化不稳定；（c）光流正则化（CoR-GS），需要额外预训练模型。
3. **核心矛盾**：在稀疏视角下，远离相机且被前方高斯遮挡的高斯因可见范围极其有限而获得很少的梯度反馈，其属性（尺度、颜色、不透明度）无法充分更新，最终导致对少数训练视角的过拟合。
4. **本文目标** 不依赖任何外部先验，仅通过对 3DGS 本身的简单修改缓解稀疏视角过拟合。
5. **切入角度**：作者观察到过拟合主要发生在训练后期而非初期，且远距离高斯因低可见性（低 transmittance $T_i$）而缺乏梯度更新。类比神经网络中的 Dropout，随机丢弃高斯可以让被遮挡的高斯"露出来"获得更多梯度。
6. **核心 idea**：通过随机丢弃高斯提高剩余高斯的可见性和梯度反馈，并渐进增加丢弃率以匹配过拟合在训练后期才显著的特性。

## 方法详解

### 整体框架
DropGaussian 在标准 3DGS 框架之上仅做一处修改：训练时随机丢弃一部分高斯，测试时使用全部高斯渲染。整个方法没有引入任何额外模块或外部先验，是一种即插即用的正则化技术。框架分为两部分：（1）带补偿因子的随机高斯丢弃；（2）渐进式丢弃率调度。

### 关键设计

1. **带补偿因子的随机高斯丢弃（DropGaussian with Compensation）**:

    - 功能：随机移除一部分高斯，让剩余高斯（尤其是被遮挡的远距离高斯）获得更大梯度和更高可见性。
    - 核心思路：定义丢弃率 $r$（如 $r=0.1$ 表示丢弃 10%），对每个高斯生成随机 mask $M(i)$，保留的高斯不透明度乘以补偿因子 $\tilde{o}_i = M(i) \cdot o_i$，其中 $M(i) = \frac{1}{1-r}$ 对保留的高斯，$M(i) = 0$ 对丢弃的高斯。这样保证了每个像素的总颜色贡献不变。
    - 设计动机：在稀疏视角下，远距离高斯的 transmittance $T_i$ 很低（被前方高斯遮挡），可见范围有限导致梯度不足。随机丢弃前方高斯后，后方高斯的 $T_i$ 增大，可见性提升，获得更充分的梯度更新。补偿因子确保了训练期间不改变整体颜色贡献的期望值。

2. **渐进式丢弃率调度（Progressive Dropping Schedule）**:

    - 功能：根据训练进度动态调整丢弃率，在训练后期加强正则化。
    - 核心思路：丢弃率随迭代线性增长：$r_t = \gamma \cdot \frac{t}{t_{total}}$，其中 $\gamma$ 为缩放因子（取 0 到 1 之间），$t$ 为当前迭代，$t_{total}$ 为总迭代数。
    - 设计动机：作者通过实验发现过拟合主要发生在训练后期——早期 PSNR 持续上升，后期新视角 PSNR 开始下降。在训练初期过多丢弃高斯可能影响正常的场景结构学习；而在后期加强丢弃可以更有针对性地抑制过拟合。

3. **无先验策略设计（Prior-Free Design Philosophy）**:

    - 功能：仅依靠 3DGS 自身的简单修改实现正则化，无需任何外部模型或额外计算。
    - 核心思路：方法不引入深度估计、光流、扩散模型等任何外部先验，不添加任何辅助损失或新模块。仅通过修改渲染流程中的高斯选择逻辑实现正则化。训练期间丢弃高斯，推理期间使用全部高斯。
    - 设计动机：先验方法存在误差传播（深度不准确）、计算开销大（生成模型）、训练不稳定等问题。DropGaussian 证明了一个关键洞察：稀疏视角的性能退化本质上是过拟合问题，可以通过经典的正则化思路解决，不一定需要额外先验。

### 损失函数 / 训练策略
- 使用标准 3DGS 颜色重建损失：$\mathcal{L}_{color} = \mathcal{L}_1(\hat{I}, I) + \lambda \mathcal{L}_{D-SSIM}(\hat{I}, I)$，$\lambda = 0.2$
- 训练 10000 次迭代，每 100 次迭代执行一次 densification
- densification 梯度阈值设为 $5 \times 10^{-4}$
- 在 NVIDIA RTX 3090Ti 上运行

## 实验关键数据

### 主实验

| 数据集 | 设置 | 指标 | DropGaussian | 3DGS | FSGS | CoR-GS |
|--------|------|------|-------------|------|------|--------|
| LLFF | 3-view | PSNR↑ | **20.76** | 19.22 | 20.43 | 20.45 |
| LLFF | 3-view | SSIM↑ | **0.713** | 0.649 | 0.682 | 0.712 |
| LLFF | 6-view | PSNR↑ | **24.74** | 23.80 | 24.09 | 24.49 |
| Mip-NeRF360 | 12-view | PSNR↑ | **19.74** | 18.52 | 18.80 | 19.52 |
| Mip-NeRF360 | 24-view | PSNR↑ | **24.13** | 22.80 | 23.70 | 23.39 |
| Blender | 8-view | PSNR↑ | **25.42** | 21.56 | 24.64 | 24.43 |

### 消融实验

| 配置 | PSNR (LLFF-3view) | 说明 |
|------|-------------------|------|
| Full Model (progressive) | **20.76** | 渐进式丢弃，最优 |
| Fixed drop rate | ~20.4 | 固定丢弃率，性能略低 |
| w/o compensation factor | ~20.2 | 无补偿因子，颜色偏差 |
| Vanilla 3DGS | 19.22 | 无 DropGaussian |

### 关键发现
- **简单方法媲美复杂先验方法**：DropGaussian 在 LLFF 3-view 上以 20.76 PSNR 超越了使用深度/光流先验的 FSGS（20.43）和 CoR-GS（20.45），证明过拟合是稀疏视角退化的主因。
- **渐进式丢弃优于固定丢弃**：验证了"过拟合主要在后期发生"的观察——渐进增加丢弃率能更好地匹配这一特性。
- **远距离高斯获得更大梯度**：分析表明 DropGaussian 使远距离高斯的梯度分布明显改善，拥有大梯度的远距离高斯数量显著增加。
- **在更多视角下仍有效**：即使在 6-view、9-view、24-view 设置下，DropGaussian 仍然保持竞争力甚至最优，说明正则化效果不仅限于极度稀疏场景。

## 亮点与洞察
- **极致简洁的设计哲学**：整个方法只需修改一行渲染代码（随机丢弃+补偿），不引入任何新参数、新模块、新损失函数或外部模型。这种 Occam's Razor 式的解决方案非常漂亮——证明了理解问题本质比堆叠复杂模块更重要。
- **从 Dropout 到 DropGaussian 的类比**：将神经网络中经典的 Dropout 正则化迁移到 3D 高斯表示，核心洞察是：Dropout 防止神经元共适应，DropGaussian 防止高斯之间的过度遮挡导致的梯度不均衡。
- **可组合性强**：作为即插即用的正则化可与其他 3DGS 方法（FSGS、CoR-GS）组合使用，且都能带来一致的提升。

## 局限与展望
- **超参数敏感性**：缩放因子 $\gamma$ 的最优值依赖数据集，需要手动调整。未来可探索自适应机制动态调整 $\gamma$。
- **缺乏理论保证**：虽然实验验证了方法有效性，但缺乏关于丢弃率与过拟合程度之间关系的理论分析。
- **对密集视角场景无明显增益**：当训练视角充足时，过拟合不再是主要问题，DropGaussian 的提升有限。
- **CLIP-I/DINO 等感知指标未报告**：仅使用 PSNR/SSIM/LPIPS 评估，未涉及更多感知质量指标。

## 相关工作与启发
- **vs DNGaussian**: DNGaussian 使用深度先验进行正则化，DropGaussian 完全无先验。在 LLFF 3-view 上，DropGaussian（20.76）超越 DNGaussian（19.12），表明无先验方法可以比有先验方法更有效。
- **vs FSGS**: FSGS 使用生成模型先验增强覆盖不足区域，计算成本高。DropGaussian 以零额外计算成本达到更优性能（LLFF: 20.76 vs 20.43）。
- **vs CoR-GS**: CoR-GS 使用光流正则化像素对应关系，需要预训练光流模型。DropGaussian 与之性能相当但方法更简洁。
- **与 DropoutGS 的关系**：两者独立提出了类似的 dropout 思路用于稀疏视角 3DGS，但技术细节不同——DropGaussian 使用补偿因子保持颜色一致性，而 DropoutGS 使用 RDR 损失配合边缘引导分裂策略。

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Dropout 概念迁移到 3DGS 是一个简洁有力的洞察，渐进式丢弃率策略有实验支撑
- 实验充分度: ⭐⭐⭐⭐ 三个标准数据集 + 多个视角设置 + 与前馈方法对比 + 梯度分布分析
- 写作质量: ⭐⭐⭐⭐ 方法直观易懂，动机分析到位，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 极简方法达到 SOTA 性能，作为即插即用插件实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)

</div>

<!-- RELATED:END -->
