---
title: >-
  [论文解读] TransiT: Transient Transformer for Non-line-of-sight Videography
description: >-
  [ICCV 2025][机器人][非视距成像] 设计了 TransiT 架构，通过瞬态信号压缩、帧间特征融合和时空 Transformer，实现从稀疏快速扫描（16×16、0.4ms/点）的 NLOS 瞬态信号实时重建 64×64 分辨率的隐藏场景视频（10 FPS），并提出基于 MMD 的迁移学习方法弥合合成与真实数据的分布差距。
tags:
  - "ICCV 2025"
  - "机器人"
  - "非视距成像"
  - "NLOS视频重建"
  - "Transformer"
  - "瞬态信号"
  - "迁移学习"
---

# TransiT: Transient Transformer for Non-line-of-sight Videography

**会议**: ICCV 2025  
**arXiv**: [2503.11328](https://arxiv.org/abs/2503.11328)  
**代码**: 即将开源  
**领域**: 机器人/计算成像  
**关键词**: 非视距成像, NLOS视频重建, Transformer, 瞬态信号, 迁移学习

## 一句话总结

设计了 TransiT 架构，通过瞬态信号压缩、帧间特征融合和时空 Transformer，实现从稀疏快速扫描（16×16、0.4ms/点）的 NLOS 瞬态信号实时重建 64×64 分辨率的隐藏场景视频（10 FPS），并提出基于 MMD 的迁移学习方法弥合合成与真实数据的分布差距。

## 研究背景与动机

### 领域现状

非视距（NLOS）成像通过在中继墙面上发射激光并用单光子探测器记录瞬态信号（transients）来重建被遮挡物体。在自动驾驶、灾后搜救等场景中具有重要价值。现有已有多种静态 NLOS 重建方法（f-k、LCT、深度学习方法），但动态 NLOS 视频重建仍面临帧率与质量的矛盾。

### 现有痛点

**帧率与质量的矛盾**：高分辨率扫描（如 64×64）需要较长时间，导致低帧率（2 FPS）；减少扫描密度或时间可提高帧率，但牺牲信噪比和细节

**快速扫描引入的失真**：振镜的最小响应时间（~0.4ms）导致激光在扫描点之间形成连续路径，而非理想的逐点扫描，引入路径积分失真

**合成数据与真实数据的域差距**：真实测量受硬件特性（激光、振镜、SPAD）影响，与仿真数据分布差异大

**现有学习方法不适用于动态场景**：现有超分辨率网络（如 USM）主要针对静态 NLOS 重建，在动态场景中表现有限

### 核心切入

NLOS 视频本质上仍是视频——直接利用 Video Transformer 的时空建模能力来处理 NLOS 瞬态信号序列，而非先上采样瞬态信号再用传统算法重建。

## 方法详解

### 整体框架

TransiT 的 pipeline 包含三个核心模块：(1) 瞬态压缩：用线性层将每个扫描点的时间维度直方图压缩为 32 维特征；(2) 特征融合：利用当前帧与前一帧的特征差异增强时序动态信息；(3) 时空 Transformer：8 个 ViT block（8 heads）+ 时空位置编码，最终通过线性层输出 64×64 重建图像。

### 关键设计

#### 1. **瞬态压缩（Transient Compression）**
- **功能**：将原始瞬态信号（可能有数百个 time bins）压缩为低维特征向量
- **核心思路**：使用线性层直接在时间轴上压缩，将高维瞬态直方图映射为 $\mathcal{F} \in \mathbb{R}^{16 \times 16 \times 32}$
- **设计动机**：与之前方法（如 USM）先上采样瞬态信号至 64×64 再重建不同，直接压缩可显著降低计算量。实验表明压缩到 32 维不会损失性能——在快速扫描的动态场景中，瞬态直方图的有价值信息集中在少数峰值位置和幅度上

#### 2. **特征融合（Feature Fusion）**
- **功能**：利用相邻帧之间的差异信息增强动态场景的表示
- **核心公式**：$\mathcal{F}_{fuse} = \text{concat}(\mathcal{F}_t, \mathcal{F}_t - \mathcal{F}_{t-1})$
- **设计动机**：从稀疏数据中恢复动态场景的精细细节非常困难。帧间差异突出了场景中的运动变化（新出现/消失的部分），为 Transformer 提供显式的时序变化线索

#### 3. **快速扫描失真模型**
- **功能**：建模振镜响应时间导致的实际测量与理想测量之间的偏差
- **核心公式**：理想瞬态 $\tau(\bar{\mathbf{x}}_n, t) = \frac{1}{r^4} \iiint_\Omega \rho(\mathbf{x}) \cdot \delta(2\|\bar{\mathbf{x}}_n - \mathbf{x}\| - tc) d\mathbf{x}$。快速扫描下，实际测量是相邻扫描点路径上的积分：$\hat{\tau}(\bar{\mathbf{x}}_n, t) = \frac{1}{\|S\|} \int_S \tau(\bar{\mathbf{x}}_n^{\mathbf{s}}, t) d\mathbf{s}$
- **设计动机**：训练时通过该模型将 64×64 的理想瞬态转化为 16×16 的失真瞬态，使 TransiT 学会从失真数据中恢复高分辨率重建

#### 4. **MMD 迁移学习**
- **功能**：弥合合成数据与真实测量数据之间的分布差距
- **核心公式**：$\mathcal{L}_{MMD} = \left\| \frac{1}{n}\sum_{i=1}^n \phi(\mathcal{F}_{real}^i) - \frac{1}{m}\sum_{j=1}^m \phi(\mathcal{F}_{syn}^j) \right\|^2$，总损失 $\mathcal{L}_{total} = \mathcal{L}_{imaging} + \lambda \mathcal{L}_{MMD}$
- **设计动机**：真实 NLOS 系统涉及材质反射特性、硬件噪声等仿真难以完美建模的因素。MMD 是自监督的——只需真实测量数据，无需对应的 GT，且仅需少量真实样本（200帧）即可有效对齐

### 训练策略

两阶段训练：
1. **阶段一**：在合成数据集（10万帧）上用 MSE loss 训练 1000 epochs，24×A800 GPU，约 24 小时
2. **阶段二**：用 200 帧真实数据，在 8×A800 GPU 上用 $\mathcal{L}_{total}$（$\lambda=0.01$）微调 100 epochs，约 2 小时

推理：单张 RTX 3090，每帧约 0.6ms，支持 10 FPS 实时重建。

## 实验关键数据

### 主实验（合成数据，动态场景）

| 物体 | 方法 | ED↓ | CS↑ | SSIM↑ | PSNR↑ |
|------|------|------|------|-------|-------|
| Character | f-k | 0.1286 | 0.7876 | 0.6677 | 17.86 |
| Character | PnP | 0.0923 | 0.8575 | 0.6764 | 20.77 |
| Character | **TransiT** | **0.0520** | **0.9418** | **0.9227** | **25.87** |
| Propeller | f-k | 0.3180 | 0.6854 | 0.1902 | 10.01 |
| Propeller | PnP | 0.2707 | 0.7800 | 0.2818 | 11.39 |
| Propeller | **TransiT** | **0.0904** | **0.9781** | **0.8211** | **20.91** |
| Human | f-k | 0.1136 | 0.6265 | 0.5791 | 19.00 |
| Human | PnP | 0.1018 | 0.6939 | 0.6706 | 19.92 |
| Human | **TransiT** | **0.0415** | **0.9272** | **0.8041** | **29.45** |

### 消融实验（静态场景，无失真）

| 物体 | 方法 | ED↓ | CS↑ | SSIM↑ | PSNR↑ |
|------|------|------|------|-------|-------|
| Character | f-k | 0.1352 | 0.6657 | 0.1224 | 17.37 |
| Character | PnP | 0.0763 | 0.8639 | 0.8852 | 22.34 |
| Character | USM | 0.0548 | 0.9317 | 0.9163 | 25.19 |
| Character | **TransiT** | **0.0531** | **0.9567** | **0.9261** | **25.49** |
| Human | f-k | 0.1525 | 0.4828 | 0.0957 | 16.34 |
| Human | PnP | 0.0754 | 0.6925 | 0.8515 | 22.49 |
| Human | USM | 0.0483 | 0.8641 | 0.9227 | 24.24 |
| Human | **TransiT** | **0.0241** | **0.9652** | **0.9361** | **26.39** |

### 关键发现

- **TransiT 在动态场景上大幅领先**：Human 场景上 PSNR 比 PnP 高 9.5 dB，SSIM 高 0.13
- **Propeller 场景最具挑战性**：快速旋转的螺旋桨导致极严重的运动模糊，TransiT 仍能恢复清晰轮廓（PSNR 20.91 vs PnP 11.39）
- **无失真静态场景下 TransiT 同样最优**：证明 TransiT 的优势不仅来自失真建模，还来自 Transformer 的表示能力
- **瞬态压缩到 32 维不损失性能**：验证了快速扫描场景中瞬态信号信息高度集中
- **MMD 迁移学习有效**：仅用 200 帧真实数据微调即可显著提升真实场景重建质量

## 亮点与洞察

1. **失真建模的系统性**：从物理光传输模型出发，推导了快速扫描的路径积分失真公式，并将其集成到训练数据生成中——这种"建模失真→合成训练→迁移学习→真实部署"的 pipeline 可推广到其他计算成像场景
2. **极简而高效的时序融合**：帧间差分 + concat 的特征融合方式极其简单，却能有效捕获运动信息——比如复杂的光流估计或 3D 卷积方案
3. **大规模合成数据集的构建**：10万帧、2K+ 运动序列的合成 NLOS 视频数据集，覆盖多种物体和运动类型，对社区有价值
4. **实时推理能力**：每帧 0.6ms 的推理速度远超实时需求，为实际部署提供充足的计算余量

## 局限与展望

1. **真实结果存在块状伪影**：可能由隐藏物体的材质（训练用漫反射，实际可能有镜面反射）导致
2. **仅支持 16×16→64×64 的固定放大比**：更灵活的输入/输出分辨率支持有待探索
3. **蛇形扫描策略的限制**：相邻帧之间的扫描路径导致方向性失真，更智能的扫描策略可能进一步提升质量
4. **单 SPAD 限制帧率**：虽然 TransiT 的推理可达 10 FPS，但硬件扫描本身（102ms/帧）是帧率瓶颈
5. **未处理多次反射和遮挡**：当前方法假设简单的 confocal 设置，复杂遮挡场景未被讨论

## 相关工作与启发

- LCT 方法（O'Toole et al.）建立了 confocal NLOS 的标准 3D 反卷积框架，TransiT 在此基础上用学习方法替代了物理重建
- PnP 方法利用即插即用去噪技术从 16×16 恢复视频（4 FPS），TransiT 在速度和质量上都大幅超越
- USM 的瞬态超分方法为静态场景设计，TransiT 的帧间融合设计使其在动态场景中表现更优
- Video Transformer（ViViT、TimeSformer）的时空注意力机制被成功迁移到 NLOS 瞬态信号处理中

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 Video Transformer 应用于 NLOS 视频重建是新颖的交叉应用，快速扫描失真建模有独创性
- **实验充分度**: ⭐⭐⭐⭐ — 合成 + 真实数据评估、动态 + 静态场景、消融实验覆盖充分
- **写作质量**: ⭐⭐⭐⭐ — 物理建模和方法描述清晰，但论文领域跨度大（光学+深度学习），读者需要背景知识
- **价值**: ⭐⭐⭐⭐ — 对 NLOS 视频重建领域有实质推进，10 FPS 实时重建具有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/robotics/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](../../AAAI2026/robotics/touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[NeurIPS 2025\] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](../../NeurIPS2025/robotics/memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)
- [\[CVPR 2026\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2026/robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)

</div>

<!-- RELATED:END -->
