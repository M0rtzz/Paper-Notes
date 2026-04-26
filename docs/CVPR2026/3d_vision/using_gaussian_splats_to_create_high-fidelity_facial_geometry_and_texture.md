---
title: >-
  [论文解读] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture
description: >-
  [CVPR2026][3D视觉][Gaussian Splatting] 提出一套基于改进 Gaussian Splatting 的人脸重建管线：通过软约束和语义分割监督将高斯与三角网格紧耦合，从仅 11 张未标定图像重建高精度三角面片几何，并利用 PCA 先验 + 可重光照高斯模型分离光照获取去光照 albedo 纹理，最终兼容标准图形管线（MetaHuman）。
tags:
  - CVPR2026
  - 3D视觉
  - Gaussian Splatting
  - 人脸几何重建
  - 去光照纹理
  - 语义分割约束
  - 神经纹理
  - MetaHuman
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture

**会议**: CVPR2026  
**arXiv**: [2512.16397](https://arxiv.org/abs/2512.16397)  
**代码**: 未开源（Epic Games / Stanford）  
**领域**: 3D视觉 / 人脸重建  
**关键词**: Gaussian Splatting, 人脸几何重建, 去光照纹理, 语义分割约束, 神经纹理, MetaHuman

## 一句话总结

提出一套基于改进 Gaussian Splatting 的人脸重建管线：通过软约束和语义分割监督将高斯与三角网格紧耦合，从仅 11 张未标定图像重建高精度三角面片几何，并利用 PCA 先验 + 可重光照高斯模型分离光照获取去光照 albedo 纹理，最终兼容标准图形管线（MetaHuman）。

## 背景与动机

1. **需求驱动**：VR/游戏/影视对高保真、可控、可重光照的人脸数字化需求持续增长，但现有方法通常依赖多相机标定或光舞台，难以大规模民主化使用
2. **NeRF 的局限**：NeRF 隐式表示难以精确分离几何与纹理，且不直接产出三角网格，无法无缝接入标准图形管线
3. **原始 3DGS 的不足**：标准 Gaussian Splatting 虽然显式但高斯与底层几何解耦——高斯可以自由形变来拟合图像，导致生成的网格质量差
4. **纹理光照耦合**：在无光舞台条件下，从少量图像中分离 albedo 与光照是严重欠约束问题，现有方法常产生烘焙阴影
5. **标准管线兼容性**：工业图形管线经过数十年软硬件优化已高度成熟，神经渲染方法需要转化为网格+纹理才能在实时应用中发挥价值
6. **少量输入的挑战**：相比长视频或多视角设置，仅使用 11 张图像重建高质量人脸几何和纹理，对正则化和约束设计提出更高要求

## 方法详解

### 整体框架

输入 → iPhone 后置摄像头拍摄单目视频 → 选取 11 个预定义姿态的帧 → 初始化粗几何（MetaHuman Animator）→ 改进的 Gaussian Splatting 训练 → 几何细化 → 纹理重建与去光照 → 输出三角网格 + 去光照纹理 → MetaHuman 转换

### 改进的 Gaussian Splatting 模型

**核心设计**：每个三角面片绑定恰好一个高斯，禁用密集化和剪枝，保持高斯与面片的一一对应。训练过程中 **不** 联合优化网格顶点，将高斯优化与网格变形解耦。

**软约束正则化** ($\mathcal{L}_{\text{reg}}$)：基于 Laplacian 平滑思想，对每个高斯的几何特征 $\mathbf{z}_i$ 鼓励其与边邻域高斯的均值一致：

$$\mathcal{L}_{\text{reg}} = \sum_i \left\| \mathbf{z}_i - \frac{1}{|\mathcal{E}(i)|} \sum_{j \in \mathcal{E}(i)} \mathbf{z}_j \right\|^2$$

分别对三种特征施加约束：

- **中心位移** $\mathcal{L}_{\text{reg}}^{\text{center}}$：高斯中心与面片质心的偏移量保持邻域平滑
- **局部法线** $\mathcal{L}_{\text{reg}}^{\text{normal}}$：高斯局部法线跨网格平滑变化（用 UV 坐标重建一致坐标系解决不一致问题）
- **边界位移** $\mathcal{L}_{\text{reg}}^{\text{boundary}}$：高斯外边界点到面片质心的距离保持邻域平滑，约束高斯形状和轮廓

**语义分割监督** ($\mathcal{L}_{\text{seg}}$)：训练 Mask2Former 分割网络（1600 个 MetaHuman 合成数据），将面部分为面部/鼻/唇/眼/耳等语义区域。每个高斯继承其所属三角面的标签，通过 alpha 混合构建预测分割图，与网络预测对比计算损失。这防止高斯"滑动"到错误语义区域。

**眼球正则化** ($\mathcal{L}_{\text{eyes}}$)：惩罚眼球高斯与眼窝高斯的交叉干涉，防止眼球高斯遮挡眼窝导致几何不准。

### 三角面片几何细化

训练完成后固定相机外参，迭代细化网格：

1. 重新优化高斯参数获取监督信息（高斯外边界点 $\mathbf{x}_i^*$）
2. 通过最小化 $\mathcal{L}_{\text{centroid}} = \sum_i \| \mathbf{v}_i^{\text{centroid}} - \mathbf{x}_i^* \|^2$ 变形网格顶点
3. 两轮迭代：第一轮优化 MetaHuman PCA 系数，第二轮优化单个顶点位置

### 神经纹理方案

将高斯从世界空间变换到 UV 纹理空间，用正交相机沿法线方向 splatting，颜色仍依赖世界空间视角方向。这允许在标准图形管线中以视角依赖神经纹理的形式使用 Gaussian Splatting，无需修改管线其他部分。

### 损失函数体系

- **图像重建**：$\mathcal{L}_{\text{img}} = 0.8 \cdot \mathcal{L}_1 + 0.2 \cdot \mathcal{L}_{\text{D-SSIM}}$
- **几何约束**：$\mathcal{L}_{\text{reg}}^{\text{center/normal/boundary}}$、$\mathcal{L}_{\text{scale}}$
- **语义**：$\mathcal{L}_{\text{seg}}$（$\lambda=50$）
- **眼球**：$\mathcal{L}_{\text{eyes}}$（$\lambda=20$）
- **光照/纹理**：$\mathcal{L}_{\text{lighting}}$、$\mathcal{L}_{\text{rotation}}$、$\mathcal{L}_{\text{blending}}$、$\mathcal{L}_{\text{view}}$

### 去光照纹理生成

- 用球谐函数建模环境光照，含遮挡图和法线图修正
- PCA 先验（MetaHuman 前 20 个基函数）正则化 albedo 纹理
- 可学习混合权重 $\beta_p$ 控制高斯与网格纹理的贡献比例，正则化趋向零以偏好网格纹理
- 训练后关闭视角依赖颜色和光照，从目标图像高通滤波恢复高频细节

## 实验关键数据

### 几何重建对比

| 方法 | 语义对齐 | 侧视轮廓 | 中性表情 | 数据需求 |
|------|---------|---------|---------|---------|
| **Ours** | ✅ 精确 | ✅ 准确 | ✅ 直接获得 | 11张图像 |
| NextFace | ❌ 语义偏移 | ❌ 侧视失败 | ✅ | 多张图像 |
| NHA | ❌ 纹理滑动 | ⚠️ 一般 | ❌ 过拟合表情 | 多张图像 |
| CoRA | ⚠️ 鼻/下颌伪影 | ⚠️ 边界模糊 | ✅ | 闪光灯采集 |

### 消融实验

| 消融项 | 影响 |
|--------|------|
| 去掉语义分割 | 高斯滑动到错误区域，几何出现伪构 |
| 去掉软约束 | 高斯与面片解耦，大小形状不规则，网格质量差 |
| 去掉眼球损失 | 眼球高斯遮挡眼窝，眼窝几何过小 |
| 去掉遮挡图 | 去光照纹理中残留烘焙阴影（鼻下、唇缝） |

### 去光照纹理质量

- 不同光照条件下去光照结果高度一致（Fig.16 两列 de-lit 纹理视觉接近）
- 在新光照条件下重光照效果优于 CoRA（CoRA 纹理残留更多烘焙光照）
- 支持异构数据联合训练（户外+闪光灯），进一步提升刚性对齐和几何精度

## 亮点

1. **极少数据**：仅需 11 张 iPhone 自拍图像即可重建高质量人脸，真正实现"民主化"人脸数字化
2. **软约束设计精妙**：中心/法线/边界三组 Laplacian 约束让高斯与网格紧耦合，既保留 3DGS 的拟合能力又确保几何质量
3. **语义分割监督**：利用 MetaHuman 合成数据训练分割网络，零成本获取语义标注，防止纹理滑动
4. **神经纹理创新**：将高斯变换到纹理空间作为视角依赖神经纹理，对工业图形管线零侵入
5. **去光照管线完整**：PCA 先验 + 球谐光照 + 遮挡图 + 高频恢复，在无光舞台条件下获得高质量 albedo
6. **端到端 MetaHuman 兼容**：输出直接可用于 UE5 标准管线，支持动画和重光照
7. **text-driven 扩展**：演示了 ChatGPT 生成图像 → Veo 3 生成视频 → 管线重建的文本驱动资产创建

## 局限与展望

1. **去光照精度有限**：无光舞台条件下仍难完全去除阴影，细粒度几何细节（如皱纹）在去光照过程中被牺牲
2. **眼部重建困难**：眼睛和眼睑区域高斯重叠严重，分割粒度不够精细，需要更好的 landmark 预测
3. **头发/颈部未处理**：框架聚焦面部，头发和颈部区域的高斯无结构化约束，不参与几何优化
4. **依赖 MetaHuman 拓扑**：整个管线与 MetaHuman 模板强绑定，泛化到其他拓扑需要额外工作
5. **合成数据域差距**：分割网络在 MetaHuman 合成数据上训练，对真实世界极端光照/遮挡场景的鲁棒性未充分验证

## 与相关工作的对比

- **vs. NeRF 方法**（HeadNeRF, HQ3DAvatar 等）：NeRF 隐式表示无法直接输出网格，本文显式约束高斯到三角面片，直接获得标准管线兼容输出
- **vs. Gaussian Avatar**（Qian et al.）：Gaussian Avatars 联合优化网格和高斯，本文解耦两者，独立约束后再用高斯驱动网格变形，更灵活且几何更精确
- **vs. 2DGS / SuGaR**：2DGS 用扁平高斯+深度蒸馏，SuGaR 用 SDF 正则化；本文用语义分割+软约束，更直接地建立语义对应
- **vs. CoRA**（Han et al.）：CoRA 需要闪光灯采集且结果有鼻/颌伪影和残留光照；本文仅需普通拍摄，去光照更彻底
- **vs. NextFace / NHA**：NextFace 侧视失败，NHA 表情过拟合导致中性表情不可用；本文在所有视角和中性表情上均优

## 评分

- 新颖性: ⭐⭐⭐⭐ — 软约束+语义分割+神经纹理的组合设计新颖，尤其是将高斯变换到纹理空间的想法有原创性
- 实验充分度: ⭐⭐⭐⭐ — 消融全面，多种对比方法，但缺少定量指标（PSNR/SSIM 等）和更大规模的用户研究
- 写作质量: ⭐⭐⭐⭐⭐ — 行文清晰，公式推导严谨，图示丰富且信息量大
- 价值: ⭐⭐⭐⭐ — 对工业界（尤其 Epic/MetaHuman 生态）有直接落地价值，学术贡献在于系统化整合多项技术

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ECCV 2024\] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/texture-gs_disentangling_the_geometry_and_texture_for_3d_gaussian_splatting_edit.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)

<!-- RELATED:END -->
