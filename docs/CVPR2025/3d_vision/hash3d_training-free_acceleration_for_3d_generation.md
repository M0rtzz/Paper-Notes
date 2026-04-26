---
title: >-
  [论文解读] Hash3D: Training-free Acceleration for 3D Generation
description: >-
  [CVPR 2025][3D视觉][3D生成加速] Hash3D 发现 SDS 优化过程中扩散模型对相邻相机位姿和时间步的特征高度冗余，通过自适应网格哈希表缓存和复用中间特征，在无需训练的情况下将多种text-to-3D和image-to-3D方法加速1.3~4倍，同时还提升了多视角一致性。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D生成加速
  - Score Distillation
  - 特征复用
  - 哈希表
  - 扩散模型
---

# Hash3D: Training-free Acceleration for 3D Generation

**会议**: CVPR 2025  
**arXiv**: [2404.06091](https://arxiv.org/abs/2404.06091)  
**代码**: https://github.com/Adamdad/hash3D (有)  
**领域**: 3D视觉  
**关键词**: 3D生成加速, Score Distillation, 特征复用, 哈希表, 扩散模型

## 一句话总结
Hash3D 发现 SDS 优化过程中扩散模型对相邻相机位姿和时间步的特征高度冗余，通过自适应网格哈希表缓存和复用中间特征，在无需训练的情况下将多种text-to-3D和image-to-3D方法加速1.3~4倍，同时还提升了多视角一致性。

## 研究背景与动机

1. **领域现状**：基于2D扩散模型的3D生成（SDS）已成为主流方法，通过在不同视角和去噪时间步上采样score function来蒸馏3D模型。

2. **现有痛点**：SDS需要数千到数万次迭代，每次都需要一次完整的扩散模型前向推理，单个物体生成可能需要数小时，严重限制了实际应用。

3. **核心矛盾**：现有的三类加速方案（训练推理模型、改进3D表示、直接生成稀疏视图）要么需要大量训练资源，要么需要对每种表示做特殊设计，要么受限于视角一致性。问题的根源——扩散模型推理次数过多——没有被直接解决。

4. **本文目标** 能否减少SDS中扩散模型的实际推理次数？

5. **切入角度**：通过实验发现，在相邻相机位姿（±10°以内）和相邻时间步上，扩散模型提取的特征（U-Net最后上采样层输入）的余弦相似度极高（>0.8）。这意味着大量推理的输出是冗余的。

6. **核心 idea**：用网格哈希表缓存扩散模型的中间特征，对相邻视角/时间步的新请求直接复用已有特征，避免重复推理。

## 方法详解

### 整体框架
Hash3D 是一个即插即用的模块，嵌入到任意SDS优化流程中。维护一个哈希表存储扩散模型的中间特征。每次采样新的相机位姿和时间步时，先查表：命中则复用特征跳过大部分推理，未命中则正常推理并更新表。使用自适应网格大小来平衡特征复用的准确性和效率。

### 关键设计

1. **网格哈希与特征缓存**:

    - 功能：高效索引和复用相邻视角/时间步的扩散模型中间特征
    - 核心思路：哈希键由4维构成：方位角 $\theta$、仰角 $\phi$、距离 $\rho$ 和时间步 $t$。网格大小为 $\Delta\theta, \Delta\phi, \Delta\rho, \Delta t$，通过取整运算将连续空间离散化为网格单元。哈希函数为 $\text{idx} = (i + N_1 \cdot j + N_2 \cdot k + N_3 \cdot l) \mod n$，其中 $N_1, N_2, N_3$ 为大素数。每个桶维护一个最大长度为3的队列存储特征。检索时用距离加权平均融合队列中的特征：$\mathbf{v} = \sum W_i \mathbf{v}_i$，权重为 $e^{-\|\mathbf{x}-\mathbf{x}_i\|_2^2}$ 的softmax。
    - 设计动机：网格哈希天然保持了空间-时间的邻近性，相近的位姿被映射到同一桶中。队列设计保证存储的是最新数据，适应不断优化的3D表示。

2. **特征复用位置选择**:

    - 功能：确定在U-Net的哪个位置截断推理并注入缓存特征
    - 核心思路：提取U-Net最后上采样层的输入特征 $\mathbf{v}_{l-1}^{(U)}$。命中时，只需运行U-Net的最后一个上采样层（最浅层）来生成最终预测，跳过前面所有层的计算。哈希概率 $\eta=0.1$ 控制检索/更新的平衡——以90%概率检索特征，10%概率正常推理并更新表。
    - 设计动机：最后上采样层的输入特征融合了高层语义（来自更深层）和低层细节（来自skip连接），复用这一层的特征可以在保持输出质量的同时减少最多的计算量。

3. **自适应网格大小**:

    - 功能：动态选择最优的网格粒度，适应不同物体和不同视角的差异
    - 核心思路：同时维护 $M=3$ 套不同网格大小的哈希表（$\Delta\theta, \Delta\phi, \Delta t \in \{10°, 20°, 30°\}$）。每次更新时计算新特征与已有特征的余弦相似度，用指数移动平均维护每个桶的平均相似度 $s_{\text{idx}^{(m)}} \leftarrow \gamma s + (1-\gamma)\text{cos}(\mathbf{v}_{new}, \mathbf{v}_i)$。检索时选择平均相似度最高的网格大小。
    - 设计动机：实验发现最优窗口大小因物体而异（如鬼魂适合5°但水豚不适合），固定网格大小无法兼顾所有情况。虽然维护多套哈希表看似开销大，但哈希操作本身极轻量，只存引用不存数据副本。

### 损失函数 / 训练策略
Hash3D是training-free的，不引入新的损失函数。它直接嵌入到已有SDS框架的优化循环中，只修改扩散模型的推理方式。

## 实验关键数据

### 主实验

| 方法 | 原始时间 | +Hash3D时间 | 加速比 | PSNR | CLIP-G/14 |
|------|---------|-------------|--------|------|-----------|
| DreamGaussian | 2min | **30s** | 4.0× | 16.36(+0.15) | 0.694 |
| Zero-123(NeRF) | 20min | **7min** | 3.3× | 17.96(+0.19) | 0.665 |
| Zero-123(GS) | 6min | **3min** | 2.0× | 18.62(+0.21) | 0.632 |
| Magic123 | 120min | **90min** | 1.3× | 18.63(-0.09) | 0.715 |
| GaussianDreamer | 15min | **10min** | 1.5× | - | 0.412 |

### 消融实验

| 配置 | 加速比 | CLIP Score | 说明 |
|------|--------|------------|------|
| Hash3D (η=0.1, M=3) | 3.3× | 0.665 | 默认配置 |
| 固定网格 (Δ=10°) | ~2.5× | 略低 | 无自适应，较保守 |
| 固定网格 (Δ=30°) | ~4× | 明显降低 | 窗口太大产生伪影 |
| η=0.5 (50%复用) | ~2× | 0.665 | 复用率低，加速少 |
| η=0.01 (99%复用) | ~4× | 略降 | 过度复用，质量下降 |

### 关键发现
- Hash3D不仅加速，还轻微提升了渲染质量（PSNR/SSIM普遍有小幅提升），原因是特征共享增强了多视角一致性
- 特征相似度在±10°以内高于0.8，这个发现是方法成立的基础
- 与3DGS结合效果最佳：text-to-3D约10分钟，image-to-3D约30秒
- 用户研究（44人）显示Hash3D生成的3D物体视觉质量与原始方法相当甚至略优
- 理论MACs减少约8%（168.78G→154.76G），但实际加速因跳过整层推理而更大

## 亮点与洞察
- **观察驱动的设计**：先发现特征冗余现象，再设计缓存策略——这种"先观察后设计"的思路比盲目优化更优雅
- **加速=提质的意外发现**：特征共享无意中充当了跨视角的一致性正则化，这解释了为什么加速方法反而提高了质量
- **极强的通用性**：完全即插即用，在8种不同3D生成方法上都有效，说明SDS中的特征冗余是一个普遍现象

## 局限与展望
- 对于渲染本身就很快的方法（如DreamGaussian已只需2分钟），加速的绝对时间节省有限
- 自适应网格使用暴力搜索3种预设大小，可以探索更精细的学习策略
- 仅缓存U-Net最后上采样层的输入，如果扩展到多层缓存可能获得更大加速
- 对于DiT架构的扩散模型（如Stable Diffusion 3），需要研究类似的特征冗余是否存在

## 相关工作与启发
- **vs DeepCache/FORA**: 2D扩散模型的特征缓存方法，Hash3D将此思路扩展到3D SDS的多视角场景
- **vs DreamGaussian**: DreamGaussian通过改进3D表示加速，Hash3D从扩散模型推理端加速，二者正交可叠加
- **vs Instant-NGP哈希**: 同样使用多分辨率哈希，但场景不同——NGP哈希空间坐标，Hash3D哈希相机位姿+时间步

## 评分
- 新颖性: ⭐⭐⭐⭐ 将特征缓存与SDS优化结合的思路新颖且直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖8种方法的全面验证，含用户研究和消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验详实
- 价值: ⭐⭐⭐⭐ 高度通用的即插即用加速方案，实践价值高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](../../CVPR2026/3d_vision/fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)
- [\[CVPR 2025\] A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)

<!-- RELATED:END -->
