---
title: >-
  [论文解读] Faster and Better 3D Splatting via Group Training
description: >-
  [ICCV 2025][3D视觉][3D高斯泼溅] 提出 **Group Training** 策略，通过将高斯基元周期性分组为"训练组"和"缓存组"来加速 3DGS 训练，结合**基于透明度的优先采样**（OPS），在4个标准数据集上实现约 **30% 训练加速**的同时**提升渲染质量**和**减少模型体积**，且可即插即用于 3DGS 和 Mip-Splatting 等框架。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯泼溅
  - 训练加速
  - Group Training
  - 基于透明度的采样
  - 新视角合成
---

# Faster and Better 3D Splatting via Group Training

**会议**: ICCV 2025  
**arXiv**: [2412.07608](https://arxiv.org/abs/2412.07608)  
**代码**: [项目页面](https://chengbo-wang.github.io/3DGS-with-Group-Training/)  
**领域**: 3d_vision  
**关键词**: 3D高斯泼溅, 训练加速, Group Training, 基于透明度的采样, 新视角合成

## 一句话总结

提出 **Group Training** 策略，通过将高斯基元周期性分组为"训练组"和"缓存组"来加速 3DGS 训练，结合**基于透明度的优先采样**（OPS），在4个标准数据集上实现约 **30% 训练加速**的同时**提升渲染质量**和**减少模型体积**，且可即插即用于 3DGS 和 Mip-Splatting 等框架。

## 研究背景与动机

### 核心问题

3D 高斯泼溅（3DGS）在新视角合成中展现了卓越的实时高质量渲染能力，但训练效率受限于**指数增长的高斯基元数量**（通常达数百万个），显著增加了训练负担。

### 现有加速方法的不足

最直接的策略是通过**周期性剪枝低透明度高斯**来减少基元数量：
- 阈值 $\epsilon_\alpha$ 设置过保守 → 加速效果微弱
- 阈值过激进 → 渲染质量严重下降
- 参数敏感性极高（见 Figure 2 左），很难找到好的平衡点

### 关键直觉

**缓存**而非直接剪枝——暂时将部分高斯排除出训练但保留它们，通过周期性重采样让缓存高斯轮换参与训练。这样既减少了每次迭代的计算量，又不丢失"重要"基元。

## 方法详解

### 整体框架

Group Training 在训练中周期性地将所有高斯基元分为两组：

$$G_{\text{Under-training}} = \{g_i | g_i \in G, i \in I, I \subseteq \{1,2,...,|G|\}\}$$
$$G_{\text{Cached}} = G \setminus G_{\text{Under-training}}$$

- **训练组（Under-training）**：参与渲染和梯度优化
- **缓存组（Cached）**：暂时不参与任何计算
- 每 500 次迭代重新合并→重新采样
- 训练组比例（UTR）默认为 0.6

### 训练调度

- 稠密化阶段（0~15K iter）：Group Training 正常运行，在 14.5K 时执行**全局稠密化**（合并所有组）
- 优化阶段（15K~30K iter）：继续 Group Training，在 29K 时执行**全局优化**
- Group Training 从第 500 次迭代开始，保护初始高斯的重要性

### 关键设计：采样策略

#### 随机采样（RS）

最简单的方案——均匀随机采样。实验表明 RS 已能加速训练，但可能导致冗余高斯过多。

#### 基于透明度的优先采样（OPS）

OPS 是论文核心创新。采样概率为：

$$p_i = \frac{\alpha_i}{\sum_{i=1}^{N} \alpha_i}$$

即透明度越高的高斯越优先被选入训练组。这由两个数学命题支撑：

**命题 1：高透明度促进有效稠密化**

高斯对位置的梯度为：

$$\frac{\partial L}{\partial x_m} = o_m \sum_{\text{pixel}} \frac{\partial L}{\partial \hat{C}} \frac{\partial \hat{C}}{\partial \alpha_m} \frac{\partial G_m^{2D}}{\partial \Delta x} \frac{\partial \Delta x}{\partial x_m}$$

- 梯度与透明度 $o_m$ 成正比
- $\frac{\partial \hat{C}}{\partial \alpha_m}$ 随 $o_m$ 期望增大而增大
- 因此高透明度高斯更容易满足稠密化阈值 $\tau_{\text{grad}}$，是稠密化的主要贡献者

**命题 2：高透明度加速渲染**

α-blending 的终止条件取决于透射率 $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ 达到饱和阈值：

$$\mathbb{E}[T_N] = (1 - \mathbb{E}[o_i] \cdot \mathbb{E}[G_i^{2D}])^N$$

- 高透明度 → 更快达到 α 饱和 → 更少的混合步骤 N → 更快渲染
- 实验验证：当 $\mu_o$ 增大时，渲染时间减少约 40%

### 设计总结

缓存低透明度高斯（保留高透明度进入训练组）→ 促进有效稠密化 + 减少冗余基元 + 加速渲染，三重收益。

## 实验关键数据

### 主实验：3DGS 重建效率与质量（Table 1 & 2）

| 方法 | 配置 | Mip-NeRF360 PSNR↑ | 时间(min)↓ | 模型大小(MB)↓ |
|------|------|-------------------|-----------|-------------|
| 3DGS* | baseline | 27.445 | 26.7 | 792 |
| + GT w/ RS | 0~30K | 27.537 | 22.6 | 902 |
| + GT w/ OPS | 0~15K | 27.582 | **22.5** | **678** |
| + GT w/ OPS | 0~30K | 27.564 | **19.6** | 679 |

| 方法 | Tanks&Temples PSNR↑ | 时间↓ | Deep Blending PSNR↑ | 时间↓ |
|------|---------------------|------|---------------------|------|
| 3DGS* | 23.697 | 15.0 | 29.586 | 23.9 |
| + GT w/ OPS (0~30K) | **23.853** | **11.0** | **29.746** | **16.9** |

**核心结果**：OPS 在所有场景上实现 ~27-30% 训练加速 + 质量提升 + 模型体积减少 10-40%。

### Mip-Splatting 上的验证（Table 3）

| 方法 | T&T PSNR↑ | 时间↓ | Deep Blending PSNR↑ | 时间↓ |
|------|-----------|------|---------------------|------|
| Mip-Splatting* | 23.749 | 23.0 | 29.358 | 35.1 |
| + GT w/ OPS (0~30K) | **24.156** | **18.2** | **29.788** | **24.0** |

**速度提升**：T&T 上快 21%，Deep Blending 上快 32%，且质量持续提升。

### 消融实验（Table 5, Tanks & Temples）

| 周期重采样 | 全局稠密化 | 全局优化 | PSNR↑ | 模型大小(MB)↓ | 时间(min)↓ |
|-----------|----------|---------|-------|-------------|-----------|
| - | - | - | 23.697 | 434 | 15.0 |
| ✓ | - | - | 23.866 | 292 | 11.8 |
| ✓ | ✓ | - | 23.769 | **231** | **11.0** |
| ✓ | - | ✓ | 23.835 | 485 | 11.8 |
| ✓ | ✓ | ✓ | **23.853** | 384 | **11.0** |

**关键发现**：
- **周期重采样**贡献最大的加速（15.0 → 11.8 min）
- **全局稠密化**进一步减小模型体积（292 → 231 MB）
- **全局优化**在有全局稠密化时提升质量（23.769 → 23.853）

### 反直觉发现

1. **RS 增大模型但仍加速**：Group Training w/ RS 生成更多基元，但训练依然更快——说明训练动态而非模型大小决定效率
2. **OPS 同时更好更小**：更高质量 + 更小模型，说明性能提升来自优化过程而非模型容量

## 亮点与洞察

1. **简洁有效**：核心思想极其简单（分组+轮换），但通过严格数学分析（两个命题的完整证明）给出了为何有效的深层原因
2. **即插即用**：作为训练策略可无缝嫁接到 3DGS 和 Mip-Splatting，无需修改架构
3. **透明度的双重角色**：首次揭示透明度同时控制稠密化有效性和渲染速度——是3DGS中被低估的关键属性
4. **超参数鲁棒性**：与剪枝阈值的极端敏感性相比，Group Training 的缓存比例参数在很大范围内都保持稳定性能（Figure 2 右）

## 局限性

1. **仅验证 30K 迭代**：未探索更长训练或极大规模场景下的表现
2. **GPU 内存开销**：组合并时可能增加峰值内存消耗（需固定 SH 系数来缓解）
3. **OPS 的独立性假设**：证明中假设高斯属性互相独立，实际可能存在相关性
4. **仅测试两个框架**：未验证在 2DGS、InstantSplat 等更新变体上的通用性

## 相关工作与启发

- **与剪枝策略的本质区别**：剪枝永久删除低透明度高斯，Group Training 仅暂时缓存并轮换——保留了重要基元的贡献机会
- **与模型压缩的关系**：OPS 自然产生更紧凑的模型（减少冗余高斯），可与后训练压缩方法互补
- **对 3DGS 训练的启发**：透明度分布是理解和优化 3DGS 行为的关键切入点，未来可探索动态调节 UTR 或自适应采样

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ — 简洁想法+严格理论分析，产出 surprising 的质量-效率双赢  
**实用性**: ⭐⭐⭐⭐⭐ — 即插即用，无超参数敏感性，工程友好  
**实验深度**: ⭐⭐⭐⭐ — 4数据集×2框架，含完整消融和采样策略对比  
**写作质量**: ⭐⭐⭐⭐ — 命题证明清晰，实验组织系统

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs_lm_faster_gaussian_splatting_optimization_with_levenberg_marquardt.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](../../CVPR2025/3d_vision/sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](../../CVPR2026/3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[ICLR 2026\] Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention](../../ICLR2026/3d_vision/efficient-lvsm_faster_cheaper_and_better_large_view_synthesis_model_via_decouple.md)

<!-- RELATED:END -->
