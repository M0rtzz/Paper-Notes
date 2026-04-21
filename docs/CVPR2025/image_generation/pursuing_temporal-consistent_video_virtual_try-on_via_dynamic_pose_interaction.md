---
title: >-
  [论文解读] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction
description: >-
  [CVPR 2025][图像生成][视频虚拟试穿] 提出 DPIDM（Dynamic Pose Interaction Diffusion Models），通过骨架姿态适配器将人体和服装的同步姿态注入去噪网络，设计分层注意力模块建模帧内人-衣姿态空间交互和帧间人体姿态时序动态，配合时序正则化注意力损失增强时序一致性，在 VVT 数据集上 VFID 达到 0.506，相比 SOTA 提升 60.5%。
tags:
  - CVPR 2025
  - 图像生成
  - 视频虚拟试穿
  - 扩散模型
  - 姿态交互
  - 时序一致性
  - 注意力机制
---

# Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction

**会议**: CVPR 2025  
**arXiv**: [2505.16980](https://arxiv.org/abs/2505.16980)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 视频虚拟试穿, 扩散模型, 姿态交互, 时序一致性, 注意力机制

## 一句话总结

提出 DPIDM（Dynamic Pose Interaction Diffusion Models），通过骨架姿态适配器将人体和服装的同步姿态注入去噪网络，设计分层注意力模块建模帧内人-衣姿态空间交互和帧间人体姿态时序动态，配合时序正则化注意力损失增强时序一致性，在 VVT 数据集上 VFID 达到 0.506，相比 SOTA 提升 60.5%。

## 研究背景与动机

**领域现状**：视频虚拟试穿 (VVTON) 旨在将指定服装"穿"到视频中的目标人物身上。早期 GAN-based 方法（CP-VTON、ClothFormer 等）采用"变形+融合"两阶段范式，受限于变形估计的准确性。近期扩散模型方法（TunnelTry-on、ViViD、GPD-VVTO 等）利用预训练文生图模型的生成能力，通过参考 U-Net 提取服装细节特征，并加入时序注意力层增强帧间一致性。

**现有痛点**：现有视频试穿方法大多仅通过在空间注意力后插入标准时序注意力来处理视频，**忽略了人体与服装之间关键的时空姿态交互**。具体来说：(1) 每帧中人体姿态与服装覆盖范围之间的**空间对齐**（如衣服的褶皱应随人体姿态变化）被忽略；(2) 整个视频中人体姿态的**时序动态**（人的动作是一系列连贯姿态的连续变化）未被充分建模。当试穿的服装与原来穿着的风格差异较大时，这些问题尤为突出。

**核心矛盾**：视频试穿需要同时保证两个目标：(1) 服装视觉保真度——服装的图案、纹理、颜色需与参考图完全一致；(2) 时序一致性——跨帧的服装外观和人体动作需流畅自然。现有方法在两者之间难以平衡，尤其是动作幅度大时。

**本文目标**：将空间姿态对齐和时序姿态动态纳入扩散模型的注意力机制，从根本上改善视频试穿的质量。

**切入角度**：引入人体和服装的同步骨架姿态作为显式引导，通过分层注意力模块分别处理帧内空间交互和帧间时序动态。

**核心 idea**：用 Pose Adapter 将姿态信息注入到空间和时序注意力的 Q/K/V 中（而非仅在输入层加入），实现每一层都有姿态感知的精细控制。

## 方法详解

### 整体框架

DPIDM 采用双分支架构：Main U-Net（基于 SD v1.5 Inpainting 模型初始化）处理 9 通道输入（噪声潜变量 4ch + 无衣潜变量 4ch + 无衣 mask 1ch）；Garment U-Net（基于 SD v1.5 初始化）提取服装精细特征并注入 Main U-Net。DW-Pose 估计视频中人体姿态，自训练的服装姿态估计器提取与人体关键点对应的服装关键点。姿态通过轻量级 Pose Encoder + Pose Adapter 注入 Main U-Net 的每层注意力模块。

### 关键设计

1. **姿态感知空间注意力 (PASA)**:

    - 功能：捕获帧内人体-服装姿态的空间交互，实现姿态引导的服装变形
    - 核心思路：在原始的空间自注意力中，人体特征 $f_h$ 和服装特征 $f_g$ 拼接后计算注意力。PASA 在此基础上引入 Pose Adapter：将人体姿态嵌入 $p_h$ 和服装姿态嵌入 $p_g$ 拼接为 $p = [p_h, p_g]$，通过两层 FC（down→GELU→up，up 层零初始化保持初始特征空间）映射为 $\text{Adpt}(p)$，加到特征上再做自注意力：$h = \text{Attn}(\psi_q(f + \text{Adpt}(p)), \psi_k(f + \text{Adpt}(p)), \psi_v(f + \text{Adpt}(p)))$。
    - 设计动机：相比现有 PoseGuider 方法仅在第一层加入姿态图像，PASA 在每一层的注意力中都注入姿态信息，实现更精细的多尺度姿态控制。同时人体+服装姿态的同步注入使空间对齐有了显式的几何引导。

2. **时序移位注意力 (TSA)**:

    - 功能：以低计算成本捕获相邻帧之间的短程时序关系
    - 核心思路：借鉴 Latent-Shift 思想，在时序维度上将前 $L$ 帧的 patch token 位移到当前帧，构成移位特征 $h_{\text{shift}}$，与当前帧特征 $h$ 拼接作为注意力的 K 和 V：$\hat{h} = \text{Attn}(\psi_q(h), \psi_k([h, h_{\text{shift}}]), \psi_v([h, h_{\text{shift}}]))$。
    - 设计动机：直接的 3D 空间-时间联合注意力复杂度为 $O((H \times W \times T)^2)$，TSA 通过简单的 shift 操作将时序信息融入 2D 注意力框架，复杂度仅为 $O((H \times W)^2)$ 级别。

3. **姿态感知时序注意力 (PATA) + 时序正则化注意力损失 (TRA)**:

    - 功能：PATA 建模视频中人体姿态的长程时序动态；TRA 进一步约束帧间注意力图的一致性
    - 核心思路：PATA 与 PASA 类似，将人体姿态嵌入通过 Pose Adapter 加到交叉注意力输出上，再做标准时序注意力，使时序建模感知姿态变化。TRA 损失通过最小化连续帧间 PASA 注意力图的差异来增强时序一致性：$\mathcal{L}_{\text{TRA}} = \sum_i^N \sum_{j=2}^T \gamma_i |\mathcal{A}_i^{(j)} - \mathcal{A}_i^{(j-1)}|$。
    - 设计动机：标准时序注意力不考虑人体姿态的连续变化，导致服装外观在动作剧烈帧跳跃。PATA 让时序建模感知"动作从哪里来、到哪里去"，TRA 则从损失层面直接约束帧间结构一致性。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L} = \mathcal{L}_{\text{LDM}} + \lambda \mathcal{L}_{\text{TRA}}$，图像数据 $\lambda=0$，视频数据 $\lambda=10^{-3}$
- 采用图像-视频联合训练策略：交替进行图像训练（仅更新 PASA+CA）和视频训练（仅更新 TSA+PATA），降低显存开销并加速收敛
- 姿态关键点 5% 概率随机 drop，强制模型从邻帧推断姿态，增强鲁棒性和时序一致性
- 训练 80k 步，batch size 32，24 帧/序列，16×A100 GPU
- 推理使用 DDIM sampler，classifier-free guidance scale 1.5，长视频用滑动窗口

## 实验关键数据

### 主实验

| 方法 | VVT VFID_I ↓ | VVT LPIPS ↓ | ViViD VFID_I ↓ | ViViD LPIPS ↓ |
|------|-------------|-------------|----------------|---------------|
| ClothFormer | 3.967 | 0.081 | - | - |
| TunnelTry-on | 3.345 | 0.054 | - | - |
| ViViD | 3.405 | 0.068 | 1.894 | 0.118 |
| GPD-VVTO | 1.280 | 0.056 | - | - |
| **DPIDM** | **0.506** | **0.041** | **0.488** | **0.081** |

VVT 上 VFID 从 1.280 降至 0.506，**相对提升 60.5%**；ViViD 上相比 ViViD 相对提升 74.2%。

### 消融实验

| 配置 | SSIM↑ | LPIPS↓ | VFID_I↓ | VFID_R↓ |
|------|-------|--------|---------|---------|
| (a) Baseline (标准时序注意力) | 0.893 | 0.084 | 3.451 | 2.435 |
| (b) + PAA (姿态感知注意力) | 0.925 | 0.050 | 1.068 | 0.153 |
| (c) + PAA + TSA | 0.929 | 0.043 | 0.721 | 0.075 |
| (d) + PAA + TSA + TRA | **0.930** | **0.041** | **0.506** | **0.047** |

### 关键发现

- **PAA 贡献最大**：从 (a) 到 (b)，VFID 从 3.451 降到 1.068，仅此一个模块已超越所有 SOTA 方法，验证了时空姿态交互建模的核心价值
- TSA 和 TRA 主要改进 VFID（时序一致性指标），对 SSIM/LPIPS 影响较小——说明它们的作用集中在帧间平滑
- DPIDM 在图像试穿任务上也取得 SOTA（VITON-HD FID 8.15, KID 0.32），说明姿态感知设计对单帧质量也有帮助
- 定性分析显示 DPIDM 在大幅度动作下仍能保持服装图案一致和自然褶皱

## 亮点与洞察

- **Pose Adapter 在每层注入的设计**：相比在输入层一次性加入姿态图，在每层注意力中注入姿态信息实现了多尺度的精细控制——这个想法类似 ControlNet 但更轻量，可以迁移到其他需要空间条件控制的视频生成任务
- **图像-视频联合训练策略**：交替训练空间和时序模块，既降低了显存需求，又避免了两类模块之间的干扰。这种训练策略对其他空间-时序联合模型有参考价值
- **姿态 drop-out**：训练时随机丢弃姿态关键点迫使模型参考邻帧，这个简单的 trick 同时增强了对姿态估计误差的鲁棒性和时序一致性
- **服装姿态估计器**：训练了一个专门的服装关键点检测器来建立人体-服装姿态对应，这在之前的试穿工作中罕见

## 局限与展望

- 需要 16×A100 训练，计算资源要求很高
- 服装姿态估计器的标注数据是手工标注的，扩展到更多服装类型时成本较高
- 推理时每帧都需要完整的扩散去噪过程，视频推理速度较慢
- 对背面人物的处理因 mask 不准确而排除了这类场景
- 目前仅处理单人场景，多人场景下的姿态交互更复杂

## 相关工作与启发

- **vs GPD-VVTO**: GPD-VVTO 将服装特征融入时序注意力来增强保真度，但不考虑姿态。DPIDM 通过 PASA+PATA 同时建模空间和时序的姿态交互，VFID 从 1.280 降至 0.506
- **vs ViViD**: ViViD 引入了大规模视频试穿数据集和参考 U-Net 架构。虽然其 SSIM 在 VVT 上最高（得益于更多训练数据），但 VFID 远不如 DPIDM（3.405 vs 0.506），说明时序一致性是其短板
- **vs TunnelTry-on**: TunnelTry-on 通过 Focus-Tunnel 裁剪对齐解决了人不在画面中心的问题，但仍用标准时序注意力处理帧间关系。DPIDM 的分层姿态注意力在时序一致性上大幅领先

## 评分

- 新颖性: ⭐⭐⭐⭐ 姿态感知注意力+分层时空建模的组合有新意，服装姿态估计也是新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ VVT+ViViD+VITON-HD 三个数据集，完整消融，定性分析充分
- 写作质量: ⭐⭐⭐⭐ 架构图清晰，方法描述详尽，但符号略多
- 价值: ⭐⭐⭐⭐ VFID 60.5% 的巨大提升证明了姿态交互建模对视频试穿的核心价值

<!-- RELATED:START -->

## 相关论文

- [Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [OmniVTON: Training-Free Universal Virtual Try-On](../../ICCV2025/image_generation/omnivton_training-free_universal_virtual_try-on.md)
- [BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)

<!-- RELATED:END -->
