---
title: >-
  [论文解读] LUCAS: Layered Universal Codec Avatars
description: >-
  [CVPR 2025][3D视觉][codec avatar] 提出 LUCAS，首个将人脸和头发解耦为分层 mesh 的通用先验 Avatar 模型，通过共享表情编码 + 独立解码实现自然的面部-头发交互，同时支持实时 mesh 渲染（45 FPS mobile）和高保真 Gaussian 渲染…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "codec avatar"
  - "layered representation"
  - "universal prior model"
  - "face-hair disentanglement"
  - "Pixel Codec Avatar"
  - "Gaussian splatting"
  - "real-time rendering"
---

# LUCAS: Layered Universal Codec Avatars

**会议**: CVPR 2025  
**arXiv**: [2502.19739](https://arxiv.org/abs/2502.19739)  
**代码**: [https://lsn33096.github.io/LUCAS/](https://lsn33096.github.io/LUCAS/)  
**领域**: 3D视觉  
**关键词**: codec avatar, layered representation, universal prior model, face-hair disentanglement, Pixel Codec Avatar, Gaussian splatting, real-time rendering

## 一句话总结

提出 LUCAS，首个将人脸和头发解耦为分层 mesh 的通用先验 Avatar 模型，通过共享表情编码 + 独立解码实现自然的面部-头发交互，同时支持实时 mesh 渲染（45 FPS mobile）和高保真 Gaussian 渲染，在跨身份零样本驱动中达到 SOTA。

## 研究背景与动机

**领域现状**: Codec Avatar 通过 VAE 架构实现高保真 3D 头部虚拟化身重建。Pixel Codec Avatar (PiCA) 实现了像素级高效渲染，但仅支持个性化训练；URAvatar 等通用模型实现了跨身份泛化但头发建模质量差。

**现有痛点**: (1) **单 mesh 表征**的根本缺陷: 人脸和头发共用同一 UV 空间，头发被分配到极小的 UV 区域，导致长发重建严重不足；(2) 平滑正则化强制面部和头发耦合运动 — 当皱眉时头发应自然下垂，但单 mesh 无法实现独立变形；(3) PiCA 是逐个体训练的个性化模型，扩展性差；(4) 现有 UPM 的 guide mesh 不准确，导致 Gaussian 锚点偏移。

**核心矛盾**: 面部和头发具有根本不同的几何/运动特性 — 面部是刚性基底+表情形变，头发是柔性悬挂+物理动态 — 用同一个 mesh 表征无法同时优化二者。

**本文切入角度**: 将 PiCA 扩展为通用模型（uPiCA），再通过分层表征解耦面部和头发，各自拥有独立的超网络、解码器和像素解码器。

## 方法详解

### 整体框架

三大组件:
1. **身份超网络** (ℰ_id^face / ℰ_id^hair): 从中性纹理和几何生成身份特定偏置
2. **共享表情编码器** (ℰ_exp): 统一编码表情变化
3. **分层解码器** (𝒟^face / 𝒟^hair): 独立解码面部和头发的几何+外观

### 关键设计

**1. Avatar 去头发处理（Dehairing）**
- **功能**: 构建去头发后的光头几何基准，作为独立头发层的底座。
- **核心机制**: 从 5 个天然光头受试者出发，迭代构建线性可变形模型（EM + 因子分析），逐步扩展到有头发的受试者 — 用已知的非头发区域推断被头发遮盖区域的光头几何。
- **设计动机**: 精确的光头几何是分层表征的基础 — 头发 mesh 必须建立在准确的头皮表面之上才能正确变形。

**2. 通用分层先验模型（Universal Layered Prior Model）**
- **功能**: 将 uPiCA 扩展为面部和头发独立建模的分层架构。
- **核心机制**:
    - 共享表情编码器提取统一表情码 $z$（确保面部-头发同步控制）
    - 面部解码器: $\mathbf{g}^{face} = \mathcal{D}_g^{face}(z, \eta)$, $\mathbf{e}^{face} = \mathcal{D}_e^{face}(z, \omega, \eta)$
    - 头发解码器额外接收头部姿态 $h$: $\mathbf{g}^{hair} = \mathcal{D}_g^{hair}(z, \eta, h)$, $\mathbf{e}^{hair} = \mathcal{D}_e^{hair}(z, \omega, \eta, h)$
    - 多 mesh 联合渲染: 两个 mesh 拼接后统一光栅化，用面部/头发 mask 分别送入独立像素解码器
- **设计动机**: 头发运动受头部姿态+面部表情双重影响（皱眉→头发下垂），故头发解码器需要同时接收 $z$（表情）和 $h$（头部姿态）。

**3. 分层 Mesh 上的 Gaussian 渲染**
- **功能**: 利用分层 mesh 的精确几何作为 Gaussian 锚点，提升高保真渲染质量。
- **核心机制**: 在分层 PiCA guide mesh 顶点上参数化 Gaussian，面部和头发各有独立的超网络和解码器。解码器输出每个 Gaussian 的位置偏移 $\delta t_k$、旋转 $q_k$、缩放 $s_k$、颜色 $d_k^c$ 和不透明度 $o_k$。
- **关键正则化**: Delta 位置损失 $\mathcal{L}_\Delta$ 防止头发 Gaussian 漂移到面部区域、面部光头区域的 Gaussian 漂移到头发区域。

### 损失函数

$$\mathcal{L}_{total} = \lambda_{pica}\mathcal{L}_{pica} + \lambda_{gs}\mathcal{L}_{gs} + \lambda_{dehair}\mathcal{L}_{dehair}$$

- $\mathcal{L}_{pica}$: 分层重建损失（光度 $\mathcal{L}_I$ + 深度 $\mathcal{L}_D$ + 法线 $\mathcal{L}_N$ + mesh tracking $\mathcal{L}_M$ + 平滑 $\mathcal{L}_S$ + KL $\mathcal{L}_{KL}$ + 分割 $\mathcal{L}_{seg}$）
- $\mathcal{L}_{gs}$: Gaussian 渲染损失 + 缩放正则 + delta 位置损失
- $\mathcal{L}_{dehair}$: 去头发几何损失（初期大权重+训练中衰减）

## 实验关键数据

### 主实验 — 定量比较

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| †PiCA (mesh, 个性化) | 32.05 | 0.8895 | 0.2678 |
| †LUCAS (mesh, 个性化) | **33.52** | **0.9044** | **0.2479** |
| †LUCAS (gs, 个性化) | **35.20** | **0.9286** | **0.2407** |
| *uPiCA (mesh, 通用) | 32.56 | 0.8971 | 0.2594 |
| *LUCAS (mesh, 通用) | 33.03 | 0.9073 | 0.2537 |
| *URAvatar (gs, 通用) | 33.12 | 0.9034 | 0.2464 |
| ***LUCAS (gs, 通用)** | **34.56** | **0.9201** | **0.2394** |

LUCAS gs 通用模型超越 URAvatar +1.4 dB PSNR。

### 消融实验

| 配置 | 训练 PSNR↑ | 未见 PSNR↑ |
|---|---|---|
| w/o 表情码 (头发) | 34.10 | 31.91 |
| w/o 头发分割 | 34.03 | 31.80 |
| **Full Model** | **34.50** | **32.58** |

表情码对头发+0.4 dB；头发分割正则对细发丝重建至关重要。

### 关键发现

1. **分层 > 单 mesh**: LUCAS mesh 比 uPiCA（相同架构基础）在所有指标上提升，尤其长发场景。
2. **通用 > 个性化**: LUCAS 通用模型的 Gaussian 渲染（34.56 PSNR）已接近甚至超过个性化 PiCA + Gaussian 的水平。
3. **表情码对头发有必要**: 去掉表情码后头发无法随表情自然变形（如皱眉时头发下垂）。
4. **分层 mesh 提升 Gaussian 质量**: 更精确的锚点几何让 Gaussian 不需要大幅偏移即可拟合目标，减少伪影。

## 亮点与洞察

1. **首个 mesh-based UPM**: 之前通用模型多依赖体积表征或 Gaussian，LUCAS 首次实现 mesh 级别的 UPM，支持移动端 45 FPS 实时渲染。
2. **分层解耦的物理合理性**: 面部是骨骼-肌肉驱动的刚性变形，头发是重力+惯性驱动的柔性变形 — 分层建模符合物理先验。
3. **共享表情码 + 独立解码**: 优雅地平衡了面部-头发的协同（同步控制）和独立性（各自变形）。
4. **从 5 个光头迭代扩展**: 去头发处理的渐进式策略非常实用。

## 局限与展望

1. 极端头发变形（如甩头）时仍存在退化，尤其是零样本驱动中未见过的姿态。
2. 仅在 Meta 内部多视角捕捉系统（110 相机、76 身份）上训练，数据获取门槛极高。
3. 未涉及重光照（relighting）能力。
4. 去头发处理依赖 HRNet 分割的准确性，对发际线模糊的造型可能出错。
5. 双分支架构增加了参数量和训练复杂度，对资源受限场景不友好。

## 相关工作与启发

- **PiCA (Ma et al.)**: 像素级解码 + 实时渲染 → LUCAS 继承其高效渲染同时扩展为通用+分层。
- **URAvatar (Cao et al.)**: Gaussian UPM → 渲染质量好但 mesh 基础差，LUCAS 用分层 mesh 提供更好的 Gaussian 锚点。
- **MEGANE (Lombardi et al.)**: 面部+眼镜的分层建模 → LUCAS 首次将分层推广到面部+头发，难度更大（头发是非刚性的）。
- **启发**: 分层解耦思路可推广到全身 avatar（身体+衣物+配饰各自独立层）以及动物 avatar（身体+毛发分层）。

## 评分

⭐⭐⭐⭐ — 首个 mesh-based UPM + 分层面发解耦很有工程价值，实验全面（个性化/通用/消融/驱动），支持实时移动端渲染；但数据获取门槛高、极端变形处理不足。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](../../ICCV2025/3d_vision/haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)

</div>

<!-- RELATED:END -->
