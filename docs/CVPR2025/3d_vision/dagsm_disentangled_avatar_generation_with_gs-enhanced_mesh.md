---
title: >-
  [论文解读] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh
description: >-
  [CVPR 2025][3D视觉][虚拟人生成] 提出 DAGSM，一种文本驱动的解耦数字人生成方法，通过 GS-enhanced Mesh（GSM）分别表示人体和各件衣物，支持换装、真实动画和纹理编辑。
tags:
  - CVPR 2025
  - 3D视觉
  - 虚拟人生成
  - 解耦衣物
  - 3D高斯网格
  - 文本驱动
  - 物理仿真
---

# DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh

**会议**: CVPR 2025  
**arXiv**: [2411.15205](https://arxiv.org/abs/2411.15205)  
**代码**: [项目页面](https://dagsm.github.io/)  
**领域**: 3D视觉 / 数字人生成  
**关键词**: 虚拟人生成, 解耦衣物, 3D高斯网格, 文本驱动, 物理仿真

## 一句话总结

提出 DAGSM，一种文本驱动的解耦数字人生成方法，通过 GS-enhanced Mesh（GSM）分别表示人体和各件衣物，支持换装、真实动画和纹理编辑。

## 研究背景与动机

- 现有文本驱动3D人体生成方法将人体和衣物作为单一模型生成，无法换装，动画不真实（衣物粘附身体），用户对衣物组合的控制有限
- SDS 直接生成的纹理质量低（过度平滑、颜色过饱和），缺乏视觉吸引力
- 需求：可动画、可换装、高质量纹理、支持多种衣物拓扑（裙子 vs 裤子）的解耦数字人

## 方法详解

### 整体框架

三阶段pipeline：(1) 生成穿内衣的人体（基于 SMPL-X + 2DGS）；(2) 逐件生成衣物（先生成网格代理，再绑定 2DGS 生成纹理）；(3) 视觉一致性纹理精修。

### 关键设计

1. **GS-enhanced Mesh (GSM) 混合表示**:
    - 功能：将 2D Gaussian Splatting 绑定在网格三角面上
    - 核心思路：每个 2DGS 在其绑定三角面的局部坐标系中定义位置（重心坐标 $\lambda_1, \lambda_2$ + 法向偏移 $z$），渲染时世界坐标为 $\hat{\mu} = \lambda_1 x_A + \lambda_2 x_B + (1-\lambda_1-\lambda_2)x_C + z\vec{n}$；使用 UV 特征图 $(\mathcal{U}_c, \mathcal{U}_\alpha)$ 存储颜色和透明度，方便编辑
    - 设计动机：结合网格的物理仿真能力和高斯的高质量渲染，使衣物运动更真实，纹理编辑更方便

2. **SAM-based 衣物分离过滤**:
    - 功能：在衣物生成过程中去除非衣物高斯，实现身体-衣物解耦
    - 核心思路：为每个高斯分配类别属性 $o$（0=身体，1=衣物），用 SAM 获取穿衣人体图像的语义 mask 作为标签，通过 MSE 损失优化 $o$；每 500 迭代删除 $o < 0.5$ 的高斯
    - 设计动机：SDS 优化不可避免地会在衣物区域生成部分人体，SAM 提供的语义信息帮助精确分离

3. **视觉一致纹理精修**:
    - 功能：提升 SDS 生成纹理的质量和多视角一致性
    - 核心思路：提出跨视角注意力机制保持纹理风格一致性；设计入射角加权去噪策略（IAW-DE），根据入射角调整每像素去噪强度
    - 设计动机：直接 SDS 生成的纹理过度平滑且多视角不一致，精修阶段使用 Stable Diffusion 3 的 RFDS 损失提升质量

### 损失函数 / 训练策略

- 人体颜色分支：$\mathcal{L}_{\mathcal{G}_b} = \mathcal{L}_{\text{rfds}}^{I_b} + \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s + \lambda_r \mathcal{L}_r$（含位置/尺度/旋转正则）
- 衣物生成：$\mathcal{L}_{\mathcal{G}_m} = \mathcal{L}_{\text{rfds}}^{I_a} + \mathcal{L}_{\text{sam}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$
- 距离正则 $\mathcal{L}_{\text{dis}}$ 约束高斯到网格面的距离
- 平滑正则 $\mathcal{L}_{\text{smooth}}$ 保证衣物表面平滑
- 使用 RFDS loss（适配 rectified-flow 模型如 SD3）替代传统 SDS

## 实验关键数据

### 主实验

| 方法 | 纹理质量 | 解耦能力 | 换装支持 | 真实动画 |
|------|---------|---------|---------|---------|
| TADA | 中等 | ✗ | ✗ | 有限 |
| HumanGaussian | 较好 | ✗ | ✗ | 有限 |
| TELA | 中等 | ✓(NeRF) | ✓ | 不真实 |
| SO-SMPL | 较差 | ✓(受限) | ✓(受限) | 有限 |
| DAGSM | **最好** | **✓** | **✓** | **真实** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o SAM 过滤 | 身体-衣物混杂 | 无法清晰分离边界 |
| w/ SAM 过滤 | 清晰分离 | 语义引导有效 |
| w/o 跨视角注意力 | 多视角纹理不一致 | 各视角独立去噪导致 |
| w/ 跨视角注意力 | 一致性好 | 跨视角特征共享有效 |
| w/o IAW-DE | 侧面纹理质量差 | 入射角大的区域信号弱 |
| w/ IAW-DE | 均匀高质量 | 加权策略补偿侧面信号 |

### 关键发现

- GSM 表示支持物理仿真驱动的真实衣物运动（如自然下摆飘动），远优于传统骨骼驱动
- 支持多种衣物材质文本描述（蕾丝、牛仔、羊毛、透明织物），纹理多样性优秀
- 可通过提供参考图像实现精确外观控制
- 支持直接修改 UV 纹理图进行手动编辑

## 亮点与洞察

- 首次实现文本驱动的完全解耦数字人生成（身体+多件独立衣物），每件衣物可独立替换
- GSM 表示巧妙结合了网格（结构、物理仿真）和高斯（渲染质量、复杂纹理）的优势
- 顺序生成（先身体后衣物）的设计简洁有效，衣物以身体为条件自然避免穿模
- 使用 RFDS loss 替代 SDS，利用 SD3 等 rectified-flow 模型的更强先验

## 局限与展望

- 衣物网格提取依赖 TSDF 算法，对复杂拓扑可能不够精确
- 生成速度受限于多阶段优化过程
- 衣物物理仿真需要额外仿真器支持
- 多层衣物（如外套+上衣）的遮挡关系处理需进一步验证

## 相关工作与启发

- SMPL-X → 人体先验；2DGS → 高质量表面重建
- RFDS loss → 更强的蒸馏信号；SAM → 语义分割辅助
- TELA → 解耦思路的先驱（但用 NeRF 限制了动画真实性）

## 评分

- 新颖性: ⭐⭐⭐⭐ GSM 混合表示设计精巧，解耦生成 pipeline 实用
- 实验充分度: ⭐⭐⭐⭐ 展示了多样化生成结果，支持换装和动画演示
- 写作质量: ⭐⭐⭐⭐ 方法描述详细，图示清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了虚拟人生成中的实际痛点，换装和物理动画的支持大幅提升实用性

<!-- RELATED:START -->

## 相关论文

- [Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)
- [MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)

<!-- RELATED:END -->
