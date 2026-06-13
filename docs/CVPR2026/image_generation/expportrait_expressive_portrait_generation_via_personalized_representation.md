---
title: >-
  [论文解读] ExpPortrait: Expressive Portrait Generation via Personalized Representation
description: >-
  [CVPR 2026][图像生成][人像动画] 提出高保真度的个性化头部表征（静态身份偏移 + 动态表情偏移），解决 SMPL-X 等参数化模型表达力不足的问题，结合身份自适应表情迁移模块和 DiT 生成器，在人像视频自驱动和跨身份重演任务上取得 SOTA 表现。 人像视频生成的核心挑战是在精细表情控制和身份一致性之间取得平…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "人像动画"
  - "个性化头部表征"
  - "表情迁移"
  - "Transformer"
  - "SMPL-X"
---

# ExpPortrait: Expressive Portrait Generation via Personalized Representation

**会议**: CVPR 2026  
**arXiv**: [2602.19900](https://arxiv.org/abs/2602.19900)  
**代码**: 无  
**领域**: 人像生成 / 面部重演  
**关键词**: 人像动画, 个性化头部表征, 表情迁移, 扩散Transformer, SMPL-X

## 一句话总结

提出高保真度的个性化头部表征（静态身份偏移 + 动态表情偏移），解决 SMPL-X 等参数化模型表达力不足的问题，结合身份自适应表情迁移模块和 DiT 生成器，在人像视频自驱动和跨身份重演任务上取得 SOTA 表现。

## 研究背景与动机

人像视频生成的核心挑战是在精细表情控制和身份一致性之间取得平衡。现有中间信号存在根本性缺陷：

**2D 关键点**：信号稀疏，缺乏几何细节，大姿态下不稳定

**3D 参数化模型（SMPL-X/FLAME）**：低秩线性近似，预定义 blendshape 无法建模高频非线性动态（如皱纹），导致身份与表情严重混淆

**隐式运动特征**：弱可控、解耦不足，容易发生身份泄露

核心矛盾：参数化模型的低维模板子空间限制了表达力，无法捕捉个体特异性解剖结构和动态皱纹，导致身份保持与表达丰富性难以兼得。

## 方法详解

### 整体框架

人像视频生成卡在精细表情控制和身份一致性的矛盾上，而矛盾的根子是中间控制信号信息密度不够。ExpPortrait 的思路是与其改进生成器、不如先把中间表征做强：它在 SMPL-X 粗模型上叠加个性化的高细节头部表征，再用一个身份自适应模块把驱动表情迁移到目标身份，最后以个性化法线图为条件训练 DiT 视频生成器。三步分别解决"表征不够细""跨身份表情不兼容""如何渲染成视频"。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    SMPLX["SMPL-X 粗网格 V"] --> NEUT["目标中性网格<br/>V_neutral = V^s + 静态身份偏移 Δg"]
    DRV["驱动信号<br/>表情系数 ψ + 下颌姿态 ω"] --> TRANS["身份自适应表情迁移<br/>条件 MLP 预测动态偏移 Δf"]
    NEUT --> TRANS
    TRANS --> REP["个性化头部表征<br/>V^s + Δg + 动态表情偏移 Δf"]
    NEUT --> REP
    REP --> NORM["渲染个性化法线图<br/>参考帧 + 驱动序列"]
    NORM --> DIT["DiT 视频生成器<br/>法线图为条件微调预训练模型"]
    DIT --> OUT["人像视频"]
```

### 关键设计

**1. 个性化头部表征：在 SMPL-X 上叠静态身份偏移 + 动态表情偏移**

SMPL-X/FLAME 是低秩线性近似，预定义 blendshape 建模不了皱纹这类高频非线性动态，身份和表情还会严重混淆。作者在粗网格上叠两个互补的偏移场：静态全局偏移 $\Delta_g^s \in \mathbb{R}^{N_s \times 3}$ 只管表情无关的个性化几何（脸型、发际线、肩部轮廓），约束在非面部区域；动态逐帧偏移 $\Delta_f^s(i) \in \mathbb{R}^{N_s \times 3}$ 只管每帧表情相关的动态（皱纹、微表情），约束在面部区域。两者叠到上采样后的高分辨率网格得到 $\widetilde{V}^s(i) = V^s + \Delta_g^s + \Delta_f^s(i)$，其中 $V^s = \mathcal{B}(V) \in \mathbb{R}^{N_s \times 3}$ 是经重心插值上采样的网格（$N_s \gg N$）。靠空间约束（面部 vs 非面部）+ 时间正则（最小幅值惩罚 + Laplacian 平滑）让两路偏移自然解耦。优化目标包含稀疏关键点损失 $\mathcal{L}_{\text{ldmk}} = \|\Pi(L_{\text{3D}}(i), \mathbf{c}) - L_{\text{2D}}(i)\|_2^2$、稠密法线/深度监督 $\mathcal{L}_{\text{normal}} = \|\hat{N}_i - N_i\|_1$、$\mathcal{L}_{\text{depth}} = \|\hat{D}_i - D_i\|_1$，以及表情系数正则、位移幅值惩罚和 Laplacian 平滑。

**2. 身份自适应表情迁移：用条件化 MLP 让表情适配目标解剖结构**

跨身份重演时直接搬偏移会出问题——儿童不该继承老人的深皱纹模式。作者先用驱动信号编码器把表情系数 $\boldsymbol{\psi} \in \mathbb{R}^{F \times 100}$ 和下颌姿态 $\boldsymbol{\omega} \in \mathbb{R}^{F \times 3}$ 编成每帧条件码 $Q = \mathcal{E}(\boldsymbol{\psi}, \boldsymbol{\omega}) \in \mathbb{R}^{F \times D}$，再用一个顶点级 MLP 以目标身份的中性网格 $V_{\text{neutral}} = V^s + \Delta_g^s$ 和驱动码 $q_i$ 为条件预测个性化动态偏移：$\Delta_f^s(i) = \mathcal{G}(V_{\text{neutral}}, q_i) \in \mathbb{R}^{N_s \times 3}$。因为预测时显式吃进了目标身份的中性网格，迁移表情的同时就自动适配了目标的解剖结构，而非"一刀切"地照搬源身份的形变。

**3. DiT 视频生成器：以个性化法线图为条件微调预训练视频模型**

有了高保真控制信号，最后一步是把它渲染成视频。作者微调基于 LDM 的 DiT，控制信号是参考帧法线图 $N^R$ 加驱动序列法线图 $N_{1:F}^D$：3D 卷积 pose encoder 提时空特征，2D 卷积 reference encoder 提外观线索，训练用标准噪声预测损失 $\mathcal{L}_{\text{ldm}} = \mathbb{E}_{z_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2]$。因为条件信号本身已经携带了个性化的几何细节，生成器不必再去"猜"皱纹和身份，身份泄露和表情僵硬的问题自然缓解。

### 损失函数 / 训练策略

- 数据：VFHQ + CelebV-HQ + HDTF，共 4000 视频约 10 小时，512×512 分辨率
- 先 SMPL-X 重建 + 几何联合优化获取个性化头部模型
- 表情迁移模块冻结后训练扩散模型
- 30 epochs，4×A800 GPU，batch size 1/GPU，学习率 $10^{-4}$
- 评估数据集：RAVDESS（20视频）+ NeRSemble（80视频）

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | L1↓ | AED↓ | APD↓ | CSIM↑ |
|------|-------|-------|--------|-----|------|------|-------|
| LivePortrait | 23.29 | 0.830 | 0.373 | 0.046 | 0.129 | 0.021 | 0.830 |
| Follow-Your-Emoji | 25.69 | 0.841 | 0.236 | 0.029 | 0.147 | 0.015 | 0.803 |
| X-NeMo | 21.56 | 0.781 | 0.324 | 0.048 | 0.137 | 0.018 | 0.830 |
| **Ours** | **26.55** | **0.859** | **0.184** | **0.022** | **0.132** | **0.009** | **0.835** |

自驱动任务多项指标大幅领先：PSNR +0.86（vs F-Y-E），LPIPS -0.052，APD 仅 0.009。

### 跨身份重演

| 方法 | AED↓ | APD↓ | CSIM↑ |
|------|------|------|-------|
| LivePortrait | 0.286 | 0.230 | 0.729 |
| X-NeMo | 0.171 | 0.021 | 0.722 |
| **Ours** | **0.211** | **0.013** | **0.729** |

在表情准确性（AED/APD）和身份保持（CSIM）之间取得最佳平衡。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SMPL-X baseline | 表情僵硬、面部动态受限 | 标准参数化模型的表达力天花板 |
| 直接偏移迁移 | 表情弱化、不自然 | 不同身份偏移不兼容 |
| **完整方案** | 丰富表情+高保真 | 个性化表征+自适应迁移 |

### 关键发现

- 个性化头部表征显著优于标准 SMPL-X，表情丰富度和身份保真度同时提升
- 隐式运动方法（Hunyuan Portrait, X-NeMo）虽能产生逼真结果，但存在严重身份泄露
- 显式控制方法（AniPortrait, F-Y-E）受限于稀疏/低秩信号，表达力不足
- 表情迁移模块比直接偏移转移产生更生动自然的表情

## 亮点与洞察

- **中间表征的天花板决定生成质量**：与其改进生成器，不如提升控制信号的信息密度和可控性
- **静态+动态解耦设计**：通过空间约束（面部/非面部）和时间正则化（零均值动态偏移）实现，无需额外标注
- **身份自适应机制**：条件化预测避免了"一刀切"式的表情迁移，符合面部解剖学差异

## 局限与展望

- **未建模口腔内部**：舌头等细节无法精确生成
- **眼球运动不精细**：缺乏细粒度眼球运动捕捉
- 训练数据量有限（~10小时），可能限制泛化到极端姿态和表情
- 未扩展到全身动画场景

## 相关工作与启发

- 与 LivePortrait 的隐式关键点方法相比，本文的显式 3D 表征更可控且不会发生身份泄露
- 与 Follow-Your-Emoji 的 FLAME 驱动相比，个性化偏移场捕捉更多高频细节
- 启发：个性化表征思想可推广到手部、全身动画等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 个性化偏移场+身份自适应迁移的设计有创新性
- 实验充分度: ⭐⭐⭐⭐ 自驱动+跨驱动全面评估，消融清晰，定性对比说服力强
- 写作质量: ⭐⭐⭐⭐ 表述清楚，管线图和公式化表述规范
- 价值: ⭐⭐⭐⭐ 为高保真人像动画提供了更好的中间表征方案，具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] Unified Vector Floorplan Generation via Markup Representation](unified_vector_floorplan_generation_via_markup_representation.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2025\] Composing Parts for Expressive Object Generation](../../CVPR2025/image_generation/composing_parts_for_expressive_object_generation.md)

</div>

<!-- RELATED:END -->
