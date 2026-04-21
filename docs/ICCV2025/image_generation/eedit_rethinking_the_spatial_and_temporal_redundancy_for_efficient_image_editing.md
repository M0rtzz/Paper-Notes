---
title: >-
  [论文解读] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing
description: >-
  [图像生成] 提出 EEdit 高效图像编辑框架，通过空间局部性缓存（SLoC）跳过未编辑区域计算、Token 索引预处理（TIP）无损加速缓存操作、以及反演步跳过（ISS）减少反演冗余，在 prompt 引导、拖拽、图像合成等多种编辑任务上实现平均 2.46× 加速且无质量损失。
tags:
  - 图像生成
---

# EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.10270](https://arxiv.org/abs/2503.10270)
- **代码**: [yuriYanZeXuan/EEdit](https://github.com/yuriYanZeXuan/EEdit)
- **领域**: Image Generation / Image Editing
- **关键词**: Diffusion Model, Image Editing, Inversion, Cache Acceleration, Spatial Redundancy, Temporal Redundancy

## 一句话总结

提出 EEdit 高效图像编辑框架，通过空间局部性缓存（SLoC）跳过未编辑区域计算、Token 索引预处理（TIP）无损加速缓存操作、以及反演步跳过（ISS）减少反演冗余，在 prompt 引导、拖拽、图像合成等多种编辑任务上实现平均 2.46× 加速且无质量损失。

## 研究背景与动机

- **反演式编辑的开销**：当前扩散模型编辑遵循两阶段流程——先反演（inversion）将图像映射到噪声空间，再去噪（denoising）生成编辑结果。反演与去噪各占一半计算量，总开销巨大
- **两类冗余的发现**：
    - **空间冗余**：编辑通常只修改图像的小区域，但流程需要计算所有像素。未编辑区域在潜空间中编辑前后余弦相似度极高，计算贡献微乎其微
    - **时间冗余**：反演过程的冗余远大于去噪。实验发现跳过反演步骤几乎不影响编辑质量，而跳过去噪步骤则迅速导致纹理和结构退化
- **现有缓存方案的不足**：
    - 层级缓存粒度过大，忽略 token 级重要性不对称
    - 部分方案需要访问 attention map 或存储 KV 矩阵，与 FlashAttention 不兼容
    - 未利用编辑任务的先验信息（已知编辑区域位置）

## 方法详解

### 整体框架

EEdit 基于 FLUX-Dev（12B 参数 flow matching 模型），提出三个无需训练的加速模块：

### 1. 空间局部性缓存（Spatial Locality Caching, SLoC）

**核心思想**：在去噪和反演过程中，对编辑区域及邻域做完整计算，对其他区域复用上一时步的缓存特征。

**编辑区域分数加成**（Score Bonus）：

$$\mathbf{S_E}(\mathbf{x}) = \begin{cases} 1 + b \cdot r^k, & \mathbf{x} \in \mathcal{N}_k(M_s), k \in 0,1,...,K \\ 1, & \mathbf{x} \notin \bigcup_{k=0}^{K} \mathcal{N}_k(M_s) \end{cases}$$

- $b > 1$: 加分因子，$0 < r < 1$: 衰减比率
- $\mathcal{N}_k(M_s)$: 编辑区域 $M_s$ 的 L1 距离为 $k$ 的邻域
- 编辑区域及邻近 token 获得更高优先级被选中计算

**缓存频率控制**：追踪每个 token 被复用的次数 $\mathcal{M}_{freq}$，复用次数越多越优先被刷新，避免累积误差的同时抑制冗余重计算。

综合分数：$\mathcal{S}_l \leftarrow (\mathcal{R} \odot \mathbf{S_E}) \oplus \mathcal{M}_{freq}$

按 top-R% 选取 token 做完整计算，其余从缓存读取。

### 2. Token 索引预处理（Token Index Preprocessing, TIP）

**关键洞察**：SLoC 的分数更新和 token 选择逻辑可以从在线操作转为离线预计算，且数学上严格等价：

$$\mathcal{I}^{(t)}_{\text{topR\%}}(\text{offline}) = \mathcal{I}^{(t)}_{\text{topR\%}}(\text{online}) \quad \forall t \in [1...T]$$

预计算所有时步的 token 索引并存储，推理时仅需单次读写，省略分数计算、排序、选择等步骤。实测额外提速 15% 以上，且完全无损。

### 3. 反演步跳过（Inversion Step Skipping, ISS）

基于 DDIM 启发，在 rf-inversion 过程中设置跳步间隔 $m$：

$$\mathbf{Z}_t \leftarrow \begin{cases} \mathbf{Z}_{t-1} & \text{if } t \bmod m \neq 1 \text{ and } m \neq T \\ \text{RF-inversion}(\mathbf{Z}_{t-1}, t-1, \phi) & \text{otherwise} \end{cases}$$

- 反演步数可安全减少到去噪步数的 33.3%（$m=3$），几乎无质量退化
- 最后一步反演始终完整计算以保证噪声质量
- 这一发现首次揭示了反演与去噪之间重要性的**高度不对称**

## 实验关键数据

### 主实验：质量与效率对比（PIE-Bench prompt 引导编辑）

| 方法 | 反演方式 | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP-T↑ | FLOPs(T) | Time(s) |
|------|----------|-------|--------|-------|---------|----------|---------|
| P2P | DDIM | 17.87 | 20.88 | 0.72 | 25.13 | 334.4 | 18.75 |
| InfEdit | Inv-free | 28.11 | 5.61 | 0.85 | 25.86 | 124.6 | 2.90 |
| RF-inv | RF-inv | 17.74 | 24.40 | 0.66 | 26.31 | 1111.6 | 13.56 |
| RF-Edit | RF-Solver | 20.17 | 18.50 | 0.77 | 26.64 | 2223.2 | 26.23 |
| Flow-Edit | - | 22.20 | 10.49 | 0.85 | 25.80 | 952.8 | 11.84 |
| **SLoC** | RF-inv | **31.97** | **1.96** | **0.94** | 25.37 | 384.0 | 5.96 |
| **SLoC+ISS** | ISS | **31.97** | **1.95** | **0.94** | **25.38** | **264.5** | **4.60** |

- SLoC 在背景一致性上**碾压**所有方法（PSNR 31.97 vs 次优 28.11）
- 相比 RF-inversion 全计算加速 2.95×，相比 RF-Edit 加速 5.70×

### 消融实验

**反演步跳过（ISS）消融**：

| 反演设置 | 去噪设置 | BG LPIPS↓ | FG LPIPS↓ | FG PSNR↑ | Time(s) |
|----------|----------|-----------|-----------|----------|---------|
| Full step | Full step | 1.98 | - | - | 13.27 |
| 2-step skip | Full step | 1.98 | 5.46 | 43.77 | 10.16 |
| **3-step skip** | Full step | **1.98** | **5.29** | **43.99** | **9.31** |
| 4-step skip | Full step | 1.98 | 5.29 | 43.80 | 8.76 |

**TIP + ISS 跨任务消融**：

| 任务 | TIP | ISS | FG FID↓ | FG PSNR↑ | Time(s) |
|------|-----|-----|---------|----------|---------|
| Prompt | × | × | 39.50 | 31.75 | 5.96 |
| Prompt | ✓ | ✓ | 39.21 | 31.76 | 4.60 |
| Dragging | × | × | 20.61 | 33.47 | 7.12 |
| Dragging | ✓ | ✓ | 22.07 | 33.68 | 5.66 |
| Composition | × | × | 12.33 | 39.78 | 7.25 |
| Composition | ✓ | ✓ | 12.35 | 39.80 | 5.66 |

TIP + ISS 在所有任务上保持性能的同时平均额外提速 20%+。

### 关键发现

- **反演冗余远大于去噪**：反演步可减到 33%，去噪不行——这是首次定量验证
- **SLoC 利用编辑先验**：已知 mask 区域作为无损加速的天然引导，优于通用缓存策略
- **极致加速**：与最先进编辑方法（RF-Edit）相比，最高达 10.96× 延迟加速

## 亮点与洞察

1. **问题定义清晰**：将编辑冗余分解为空间冗余和时间冗余两个正交维度，逐一击破
2. **无需训练**：所有模块均为 training-free，即插即用，适配多种编辑任务
3. **数学等价的预处理**：TIP 的离线预计算保证与在线版本严格等价，是真正的"无损加速"
4. **编辑先验的利用**：现有缓存方法面向生成，忽略了编辑任务的 mask 先验——这是关键差异化
5. **首次揭示反演/去噪重要性不对称**：这一发现对编辑流程设计有指导意义

## 局限性

- **依赖 mask 输入**：需要用户提供编辑区域 mask，对无 mask 的全局风格编辑不直接适用
- **基于 FLUX 12B**：实验仅在 FLUX-Dev 上做主实验，SD 系列仅做定性展示
- **编辑质量上限受限于基础编辑方法**：EEdit 加速的是已有编辑 pipeline，编辑效果本身受制于底层方法
- **缓存比例 R% 需手动设置**：未探索自适应缓存比例策略

## 相关工作与启发

- DeepCache、L2C 等层级缓存方案启发了缓存思路，但 EEdit 创新性地引入 token 级 + 编辑先验
- Toca、Duca 的 token-wise cache 是最近相关工作，但未针对编辑任务
- RF-inversion 是本文基础反演方法，ISS 策略在其上实现进一步加速
- 启发：类似的"重要性不对称"分析可推广到视频编辑，前向传播 vs 后向传播的冗余分析

## 评分 ⭐⭐⭐⭐

框架设计优雅，三个模块各自解决明确问题且可组合叠加。空间-时间冗余的双重分析视角新颖，实验覆盖 prompt/drag/composition 三类任务且加速显著。TIP 的数学等价性证明体现了扎实的理论功底。不足在于 mask 依赖和对更多基础模型的验证不足。

<!-- RELATED:START -->

## 相关论文

- [Rethinking Layered Graphic Design Generation with a Top-Down Approach](rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](../../CVPR2025/image_generation/dynamic_motion_blending_for_versatile_motion_editing.md)
- [EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](../../CVPR2025/image_generation/easycraft_a_robust_and_efficient_framework_for_automatic_avatar_crafting.md)

<!-- RELATED:END -->
