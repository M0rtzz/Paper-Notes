---
title: >-
  [论文解读] D3-Human: Dynamic Disentangled Digital Human from Monocular Video
description: >-
  [CVPR 2025][人体理解][数字人重建] D3-Human 提出了一种从单目视频重建解耦（服装+人体）数字人几何的方法，通过定义人体流形上的有符号距离场（hmSDF）在无需3D服装先验的条件下实现了可见区域的服装-人体精确分割，约20分钟生成解耦模板并支持换装和动画应用。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "数字人重建"
  - "服装解耦"
  - "单目视频"
  - "隐式-显式混合表示"
  - "hmSDF"
---

# D3-Human: Dynamic Disentangled Digital Human from Monocular Video

**会议**: CVPR 2025  
**arXiv**: [2501.01589](https://arxiv.org/abs/2501.01589)  
**代码**: [https://ustc3dv.github.io/D3Human/](https://ustc3dv.github.io/D3Human/)  
**领域**: 人体理解 / 3D重建  
**关键词**: 数字人重建, 服装解耦, 单目视频, 隐式-显式混合表示, hmSDF

## 一句话总结

D3-Human 提出了一种从单目视频重建解耦（服装+人体）数字人几何的方法，通过定义人体流形上的有符号距离场（hmSDF）在无需3D服装先验的条件下实现了可见区域的服装-人体精确分割，约20分钟生成解耦模板并支持换装和动画应用。

## 研究背景与动机

**领域现状**：从视频重建穿着服装的人体3D几何一直是图形学和计算机视觉的热门方向，广泛应用于虚拟现实、增强现实、全息通信、影视制作和游戏开发。从单目视频进行重建由于设备简单而更具实用价值。

**现有痛点**：现有方法主要分为两类，都有明显局限：（1）隐式表示方法（如 SDF、NeRF）可以重建高质量的穿衣人体几何，但通常将服装和人体重建为不可分离的整体，无法用于换装或动画等需要编辑的应用；（2）显式表示方法依赖预定义模板（如 SMPL），可以分离人体和服装，但受限于模板的表达能力，无法处理多样的服装类型。少数尝试解耦的方法要么依赖3D扫描模板，要么使用 UDF 表示但在单视角稀疏监督下表现不佳。

**核心矛盾**：服装对人体的遮挡导致"可见区域的细节还原"和"不可见区域的合理补全"之间的矛盾。同时，从2D单视角图像中获取3D服装-人体分割信息本身就极具挑战性。

**本文目标**：直接从短单目视频中重建高保真、解耦的服装和人体几何，不使用任何3D服装模板先验。

**切入角度**：作者观察到水密的穿衣人体表面可以被一条封闭曲线分割为服装和人体两部分。因此，如果能在已重建的人体表面上定义一个连续的分割函数，就可以只用2D人体解析（human parsing）信息来获得3D分割。

**核心 idea**：在完整穿衣人体的水密表面上定义一个"人体流形有符号距离场"（hmSDF），用于将表面点分类为服装或人体。hmSDF 仅需2D parsing mask 监督，结合显式 SMPL 模型补全被遮挡的人体区域，实现无需3D先验的完整解耦重建。

## 方法详解

### 整体框架

D3-Human 的输入是一段包含运动中穿衣人物的短视频序列 $\{I_t | t=1,...,N\}$。方法分为两阶段：（1）**模板生成阶段**——在标准空间中利用 hmSDF 重建解耦的服装和人体模板，仅使用 LBS 变形和2D mask 监督；（2）**细节变形阶段**——引入非刚性变形场和感知法线损失来优化每帧的服装和人体细节，使用两个独立的 MLP 分别建模服装和人体的非刚性变形。

### 关键设计

1. **人体流形有符号距离场（hmSDF）**:

    - 功能：在已重建的水密穿衣人体表面上区分服装区域和人体区域
    - 核心思路：传统 SDF 定义在3D空间中，而 hmSDF 定义在人体表面流形上。具体地，用 DMTet（Deep Marching Tetrahedra）的混合表示建模完整穿衣人体——一个四面体网格 $(V_T, T)$ 加上神经隐式 SDF 函数 $s_\eta(x)$。在此表面 $S_\eta = \{x | s_\eta(x) = 0\}$ 上，定义映射 $\nu: S_\eta \to \mathbb{R}$，其中 $\nu(x) < 0$ 表示人体区域，$\nu(x) > 0$ 表示服装区域，$\nu(x) = 0$ 为边界线。与 GShell 的 mSDF 不同，hmSDF 考虑了边界两侧的点（同时处理服装和人体）
    - 设计动机：直接使用隐式 UDF 表示服装在单视角稀疏监督下表现很差——UDF 在零水平面处不可微，且对噪声敏感。hmSDF 利用已有的水密表面作为"载体"，将3D分割降维为表面上的分类问题，大幅降低了问题难度

2. **区域聚合算法（Region Aggregation）**:

    - 功能：修正 hmSDF 分割中因2D human parsing 不一致导致的错误
    - 核心思路：由于不同帧的 human parsing mask 可能存在噪声和帧间不一致，hmSDF 的分割结果可能会出现碎片化——正确的人体区域中混入了小块的服装碎片，反之亦然。作者提出一个基于连通分量分析的修正策略：首先用深度优先搜索找出每个类别的所有连通子图，然后根据每个子图的顶点数量判断其真正类别——小的碎片（顶点数少的连通分量）被识别为错误分割，合并到对面类别中。这确保了最终的服装和人体区域各自连通且无孔洞
    - 设计动机：这是应对"弱监督信号"（仅有2D mask）的实用工程设计，简单但有效地解决了多视角不一致性问题

3. **解耦非刚性变形场**:

    - 功能：分别建模服装和人体在不同帧中的细微变形
    - 核心思路：在模板生成后，使用两个独立的 MLP 分别建模服装和人体的非刚性变形。对于帧 $t$，变形定义为 $x_t = D(x, h_t, E(x); \phi)$，其中 $x$ 是标准空间中的点，$h_t$ 是帧 $t$ 的潜码，$E(x)$ 是位置编码。然后通过共享的 LBS（Linear Blend Skinning）变换到观测空间。关键点是服装和人体的非刚性变形是分开学习的——服装有独立的褶皱动态，而人体有自身的肌肉变形
    - 设计动机：服装和人体遵循不同的运动规律——人体运动由骨架驱动，而服装运动还受重力、惯性和碰撞影响。分离建模使得各自的变形更精确

### 损失函数 / 训练策略

训练分两阶段，损失函数组成如下：

**模板生成阶段**：
- RGB Loss $\mathcal{L}_{color}$：渲染 RGB 与输入的 L1 距离，分别计算服装和人体像素
- Mask Loss $\mathcal{L}_{mask}$：渲染 mask 与 SAM2 提取的 GT mask 的 MSE
- Eikonal Loss $\mathcal{L}_{eik}$：确保 SDF 梯度范数接近1
- Encourage Hole Opening $\mathcal{L}_{hole}$：鼓励 hmSDF 在袖口等位置形成开口
- Regularize Holes $\mathcal{L}_{reg}$：防止开口过大

**细节变形阶段**（额外增加）：
- 感知法线损失 $\mathcal{L}_{per}$：使用 Sapiens 预测的法线图作为 GT，计算 MobileNetV2 多层级感知损失
- 碰撞惩罚 $\mathcal{L}_{collision}$：防止服装穿透人体
- 几何正则化 $\mathcal{L}_{n\_consist} + \mathcal{L}_{laplacian}$：法线一致性 + 拉普拉斯平滑

### 遮挡感知可微渲染

一个重要的技术细节：当从同一视角渲染时，服装可能遮挡人体部分区域。如果分别渲染服装 mask，会得到与2D GT 不一致的结果（因为GT mask考虑了遮挡而单独渲染不考虑）。D3-Human 同时渲染标记了类别标签的服装和人体 mesh，通过光栅化自动处理遮挡关系，生成遮挡感知的2D标签。

## 实验关键数据

### 主实验（合成数据定量评估，Chamfer Distance ×10⁻³）

| 方法 | Female1 All | Female3 All | Male1 All | Male2 All | 可解耦 |
|------|-----------|-----------|---------|---------|--------|
| REC-MV | 1.789 | 1.461 | 1.945 | 1.201 | 服装only |
| BCNet | 5.561 | 5.681 | 4.802 | 2.853 | ✓ |
| DELTA | 1.388 | — | 1.702 | 1.132 | ✓ |
| SelfRecon | 3.420 | 2.249 | 1.310 | 1.454 | ✗ |
| GoMAvatar | 7.319 | 5.058 | 2.382 | 3.163 | ✗ |
| **D3-Human** | **最优** | **最优** | **最优** | **最优** | ✓ |

### 消融实验

| 配置 | 视觉效果 | 说明 |
|------|---------|------|
| hmSDF | 精确服装形状 | 准确的袖口开口和服装边界 |
| 隐式 UDF 替代 | 大量孔洞和缺陷 | 腹部区域大洞，袖口无法正确开口 |
| Perceptual Normal Loss | 光滑且保留细节 | 感知损失保持全局一致性 |
| MSE Normal Loss | 粗糙且有噪声 | 逐点损失导致不平滑 |
| COS Normal Loss | 类似 MSE 的问题 | 角度损失同样不够平滑 |

### 关键发现

- hmSDF 相对于隐式 UDF 的核心优势在于：UDF 在零水平面不可微、表面提取受限于流形约束、且对噪声（如遮挡导致的 mask 错误）敏感。hmSDF 通过在水密表面上定义分割解决了这三个问题
- 感知法线损失（Perceptual Normal Loss）比逐点的 MSE/Cosine 损失更有效，因为它关注特征级一致性而非像素级精确度，产生更平滑且保留细节的结果
- D3-Human 生成解耦模板约20分钟，完整序列数小时，相比 REC-MV/DressRecon 的超过24小时有显著效率提升
- 解耦重建的直接应用价值：换装（交换不同人的服装）和基于物理的动画（使用 HOOD 模拟服装动态）

## 亮点与洞察

- **hmSDF 的优雅定义**：在已有的水密表面上定义分割距离场，将3D分割降维为表面上的分类问题，避免了直接用 UDF 建模开放曲面的诸多困难。这种"先重建整体、再分割部分"的策略也可应用于其他场景分解任务（如室内场景中分割家具和墙壁）
- **仅需2D监督的3D分割**：只利用2D human parsing（SAM2 提供）就实现了3D的服装-人体分割，不需要任何3D服装先验或3D标注。这大大降低了数据需求
- **解耦变形建模**：用独立的 MLP 分别建模服装和人体的非刚性变形，物理上更合理（服装有独立的褶皱动态），结果也更精确

## 局限与展望

- 目前只处理单人场景的短视频，对多人交互或长视频序列的处理能力尚未验证
- 人体解析的质量直接影响 hmSDF 的分割准确性，在复杂遮挡或非常规服装下可能退化
- 依赖 SMPL 参数的准确估计（使用 CLIFF），SMPL 拟合失败时整个重建会失败
- 对于宽松服装（如长裙），服装与人体之间的大间隙可能导致碰撞惩罚失效
- 当前方法不建模服装纹理和材质，仅重建几何，对于完整的虚拟试衣应用还需要扩展

## 相关工作与启发

- **vs SelfRecon**: SelfRecon 用 SDF 重建穿衣人体但无法解耦，几何细节也较为平均。D3-Human 在保证解耦的同时达到了更好的几何质量
- **vs REC-MV**: REC-MV 可以重建服装但不包含人体，且人体直接使用 SMPL 导致穿穿透。D3-Human 同时重建并优化了服装和人体
- **vs DELTA**: DELTA 基于 SCARF 用 NeRF 表示服装，可以解耦但服装几何质量差（NeRF 难以提取平滑几何）。D3-Human 的 mesh 表示产生了更干净的几何
- **vs GoMAvatar**: GoMAvatar 用 Gaussians-on-Mesh 表示，mesh 质量粗糙且不可解耦。D3-Human 在两方面都显著优于它

## 评分

- 新颖性: ⭐⭐⭐⭐ hmSDF 的定义是核心创新，从水密表面上做分割是一个优雅的 insight
- 实验充分度: ⭐⭐⭐⭐ 定量定性对比全面，消融实验有说服力，但真实数据缺少定量评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但部分公式推导可以更简洁
- 价值: ⭐⭐⭐⭐ 解耦重建是实际应用中的刚需，换装和动画的展示很有说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)
- [\[ACL 2025\] TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments](../../ACL2025/human_understanding/transbench_breaking_barriers_for_transferable_graphical_user_interface_agents_in.md)
- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](../../ICCV2025/human_understanding/monocular_facial_appearance_capture_in_the_wild.md)

</div>

<!-- RELATED:END -->
