---
title: >-
  [论文解读] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification
description: >-
  [CVPR 2026][3D视觉][4D高斯泼溅] 针对4D高斯泼溅在长视频动态场景建模中面临的内存爆炸、时序闪烁和遮挡处理等挑战，提出了基于锚点接力双向混合 (ARBB) 的MoRel框架，通过关键帧锚点的渐进式构建和可学习时序不透明度控制实现了无闪烁、内存有界的长程4D运动重建。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "4D高斯泼溅"
  - "动态场景重建"
  - "长视频建模"
  - "时序一致性"
  - "内存高效"
---

# MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification

**会议**: CVPR 2026  
**arXiv**: [2512.09270](https://arxiv.org/abs/2512.09270)  
**代码**: [https://cmlab-korea.github.io/MoRel/](https://cmlab-korea.github.io/MoRel/)  
**领域**: 3D视觉  
**关键词**: 4D高斯泼溅, 动态场景重建, 长视频建模, 时序一致性, 内存高效

## 一句话总结
针对4D高斯泼溅在长视频动态场景建模中面临的内存爆炸、时序闪烁和遮挡处理等挑战，提出了基于锚点接力双向混合 (ARBB) 的MoRel框架，通过关键帧锚点的渐进式构建和可学习时序不透明度控制实现了无闪烁、内存有界的长程4D运动重建。

## 研究背景与动机

1. **领域现状**：3D高斯泼溅 (3DGS) 已成为新视角合成的主流范式，自然地被扩展到4D动态场景。现有4DGS方法主要分为"全局一次性训练"和"分块训练"两大类。

2. **现有痛点**：
    - **全局一次性训练**（如4DGS、MoDec-GS）：将所有帧放在一起优化，保证了全局时序一致性，但对长视频会导致GPU内存爆炸，高维高斯数量随时间长度不断增长。
    - **分块训练**（如GIFStream）：将长视频分成短片段独立训练，减少了内存开销，但在片段边界处产生时序不连续和外观突变，即"闪烁"伪影。
    - 滑动窗口策略只是局部修补，不能保证全局一致性；时序高斯层次结构虽内存近恒定，但系统复杂度很高。

3. **核心矛盾**：长视频建模中"全局时序一致性"与"有界内存使用"之间的根本矛盾——既要跨越数千帧保持平滑过渡，又不能让内存随帧数线性增长。

4. **本文目标** (a) 有界内存的长程4D建模；(b) 无闪烁的时序一致性；(c) 高效的随机时间访问；(d) 不依赖外部光流等额外线索。

5. **切入角度**：借鉴视频编码中"关键帧+GOP"的思想，在时间轴上周期性放置关键帧锚点 (KfA)，通过双向变形和自适应混合实现平滑过渡。

6. **核心 idea**：用关键帧锚点接力机制取代全局或分块策略，在锚点间学习双向变形并通过可学习不透明度控制进行平滑混合，实现内存有界且无闪烁的长程4D重建。

## 方法详解

### 整体框架
MoRel采用基于锚点的3DGS表示，以稀疏体素网格的锚点定义规范空间。整体训练分为两个阶段（4个训练步骤）：**锚点接力阶段**（GCA训练 → KfA训练）和**双向混合阶段**（PWD训练 → IFB训练）。输入是多视角长视频序列，输出是可实时渲染的4D高斯表示。

### 关键设计

1. **全局规范锚点 (GCA) + 关键帧锚点 (KfA)**:

    - 功能：GCA浏览整个视频训练全局锚点 $\mathbf{A}^{\text{Global}}$，为后续所有KfA提供全局一致的初始化；KfA在时间轴上周期性放置，在各自时间索引附近精细优化。
    - 核心思路：用单个点云初始化GCA（而非时间密集的点云），对全部帧粗略训练。训练完成后根据每个锚点特征方差分配层次级别。所有KfA从GCA初始化而非从头训练，每个KfA负责 $[t_n - \text{GOP}, t_n + \text{GOP}]$ 的时间范围，并引入时间容差 $\epsilon$ 增强鲁棒性。周期性放置KfA类似视频编码关键帧，提供随机访问点和有界内存。
    - 设计动机：确保全局外观一致性，同时通过局部规范空间捕获细节运动，按需加载/卸载KfA保持内存有界。

2. **渐进式窗口变形 (PWD) + 中间帧混合 (IFB)**:

    - 功能：PWD在每个KfA的双向变形窗口内独立学习前向和后向变形场；IFB在相邻KfA之间训练可学习的时序不透明度控制进行平滑混合。
    - 核心思路：PWD中每个KfA通过动态加载/卸载独立训练，避免"反向污染"——后续chunk训练修改前序chunk依赖的锚点。变形场使用归一化相对时间 $\tau_n \in [-1, 1]$。IFB阶段冻结所有锚点属性和变形场，仅训练时序偏移 $o_{n,k}^{\text{dir}}$ 和衰减速度 $d_{n,k}^{\text{dir}}$，时序不透明度为 $w_{n,k}^{\text{dir}} = \exp[-\lambda_{\text{decay}} \cdot d_{n,k}^{\text{dir}} \cdot |\tau_n - o_{n,k}^{\text{dir}}|]$。
    - 设计动机：PWD彻底防止分块间干扰（chunk-wise训练的根本缺陷），IFB通过可学习参数自适应处理不规则运动（如遮挡），比简单线性插值效果更好。

3. **特征方差引导分层致密化 (FHD)**:

    - 功能：基于锚点特征方差控制致密化节奏，平衡内存使用和高频细节保留。
    - 核心思路：GCA训练后根据特征方差 $\sigma_k^2 = \text{Var}(\hat{f}_k)$ 将锚点分为3个层次（低频/中频/高频）。训练早期低频锚点权重为1，高频锚点权重较低 $\lambda_L$；随训练进行通过线性插值 $\eta_t$ 逐步提升高频权重。梯度统计按层次加权 $g_L^{j_n^S} = g^{j_n^S} \cdot w_L^{j_n^S}$ 作为致密化判据。
    - 设计动机：高频区域在训练早期不稳定，过早致密化会产生冗余锚点；延迟高频致密化既控制内存又保证最终细节质量。

### 损失函数 / 训练策略
采用标准3DGS重建损失（L1 + SSIM）。四个训练阶段顺序执行。关键是按需加载/卸载机制：任何时刻最多加载1-2个KfA和变形场，确保训练和渲染内存有界。

## 实验关键数据

### 主实验
构建SelfCap_LR数据集（5个场景，>3500帧），具有更大的平均运动幅度和更宽的空间范围。

| 方法 | 类型 | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ | tOF↓ | 训练内存↓ |
|------|------|-----------|-----------|------------|------|----------|
| 4DGS | 全局 | 18.95 | 0.648 | 0.402 | 0.222 | ~18,000MB |
| MoDec-GS | 全局 | 19.61 | 0.643 | 0.391 | 0.249 | ~22,000MB |
| LocalDyGS | 全局 | 20.64 | 0.652 | 0.371 | 0.215 | ~12,000MB |
| GIFStream | 分块 | 19.02 | 0.653 | 0.405 | 0.539 | ~9,000MB |
| 4DGS_chunk | 分块 | 19.31 | 0.656 | 0.389 | 0.680 | ~4,500MB |
| **MoRel** | **本文** | **21.00** | **0.664** | **0.355** | **0.203** | **~6,000MB** |

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练内存 | 渲染内存 |
|------|-------|-------|--------|---------|---------|
| (a) GCA + 单向变形 | 19.71 | 0.654 | 0.386 | ~12,000 | 156 |
| (b) + KfA | 19.90 | 0.647 | 0.364 | ~4,500 | 94 |
| (c) + PWD + 线性混合 | 20.66 | 0.656 | 0.358 | ~6,500 | 138 |
| (d) + PWD + IFB | 21.07 | 0.672 | 0.342 | ~6,500 | 144 |
| (e) + FHD (完整MoRel) | 21.20 | 0.672 | 0.348 | ~6,000 | 126 |

### 关键发现
- KfA引入后训练内存从~12,000MB降到~4,500MB（62.5%↓），同时LPIPS从0.386降到0.364。
- IFB相比线性混合PSNR提升0.41dB，可学习不透明度控制对不规则运动至关重要。
- FHD将渲染内存从144MB降至126MB，同时保持质量。
- MoRel在tOF上取得最优0.203，远优于分块方法的0.539/0.680。

## 亮点与洞察
- **锚点接力思想优雅**：借鉴视频编码关键帧概念，既解决内存问题又自然提供随机时间访问，对流媒体系统很实用。
- **PWD解决"反向污染"**：分块训练的根本问题是后续训练破坏先前表示，PWD通过独立BDW训练彻底避免。
- **FHD的设计哲学可迁移**：用特征方差作为频率复杂度代理控制致密化节奏，可迁移到静态3DGS场景重建。

## 局限与展望
- 4阶段顺序训练流程总时间较长，能否并行化部分阶段值得探索
- GOP选择是固定的，能否根据运动复杂度自适应调整
- 主要在自采集数据集上评估，泛化性有待验证
- 未讨论对瞬间出现/消失物体等极端运动的处理能力

## 相关工作与启发
- **vs 4DGS (全局)**：PSNR高2.05dB，tOF低8.6%，且不会内存爆炸
- **vs GIFStream (分块)**：IFB彻底解决边界闪烁，tOF从0.539降至0.203
- **vs TGH (层次)**：系统复杂度更低，不需要CPU-GPU流化

## 评分
- 新颖性: ⭐⭐⭐⭐ 锚点接力和PWD策略新颖，但基本思想来自视频编码
- 实验充分度: ⭐⭐⭐⭐ 消融充分，主数据集是自采集的
- 写作质量: ⭐⭐⭐⭐⭐ 图示清晰，逻辑分明
- 价值: ⭐⭐⭐⭐ 对长视频4D重建有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](../../AAAI2026/3d_vision/gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
