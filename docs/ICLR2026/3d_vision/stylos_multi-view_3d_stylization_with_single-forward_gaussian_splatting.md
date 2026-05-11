---
title: >-
  [论文解读] Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting
description: >-
  [ICLR 2026][3D视觉][3D风格迁移] Stylos 提出了一个单次前馈的3D风格迁移框架，通过共享Transformer骨干的双路径设计（几何自注意力+风格交叉注意力）和体素级3D风格损失，实现从未标定输入的零样本3D风格化，支持单视角到数百视角的扩展。
tags:
  - ICLR 2026
  - 3D视觉
  - 3D风格迁移
  - 高斯溅射
  - 跨视角一致性
  - 体素风格损失
  - 前馈模型
---

# Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting

**会议**: ICLR 2026  
**arXiv**: [2509.26455](https://arxiv.org/abs/2509.26455)  
**代码**: [https://github.com/HanzhouLiu/Stylos](https://github.com/HanzhouLiu/Stylos)  
**领域**: 3D视觉  
**关键词**: 3D风格迁移, 高斯溅射, 跨视角一致性, 体素风格损失, 前馈模型

## 一句话总结

Stylos 提出了一个单次前馈的3D风格迁移框架，通过共享Transformer骨干的双路径设计（几何自注意力+风格交叉注意力）和体素级3D风格损失，实现从未标定输入的零样本3D风格化，支持单视角到数百视角的扩展。

## 研究背景与动机

3D风格迁移旨在保持场景几何和跨视角一致性的同时迁移参考风格。现有方法存在三层限制：

**NeRF/3DGS方法需逐场景优化**：StyleRF、StyleGaussian等虽比NeRF更高效，但仍需逐场景拟合，无法实现真正的实时3D风格化

**泛化能力弱**：现有方法局限于场景特定训练，无法推广到未见过的类别、场景和风格

**2D风格损失缺乏3D一致性**：经典的Gram矩阵或AdaIN（通道统计量匹配）在图像级别操作，不能显式保证多视角结构一致性

最接近的相关工作Styl3R (Wang et al., 2025b) 虽提出前馈框架，但设计仅针对2-8个输入视角，不特别关注强多视角一致性。

## 方法详解

### 整体框架

Stylos 以VGGT为几何骨干，引入Style Aggregator分支通过CrossBlock融合内容和风格特征。几何属性（深度、位姿）仅由骨干推导，风格仅影响颜色球谐系数，实现几何与风格的解耦。

### 关键设计

1. **CrossBlock 风格-内容融合模块**

   在标准Transformer Block的自注意力和MLP之间插入交叉注意力操作：内容token为Query，风格token为Key/Value。提出三种拓扑策略：
    - **Frame CrossBlock**：每个视角独立与风格交互，保持视角特有结构
    - **Global CrossBlock**：拼接所有视角为全局序列，自注意力确保多视角几何一致性，交叉注意力广播风格信息
    - **Hybrid CrossBlock**：先Frame再Global

   实验表明 **Global CrossBlock 效果最好**（Pizza场景PSNR提升0.79dB），因为全局自注意力保证跨视角一致性同时交叉注意力全局广播风格。

2. **体素级3D风格损失 (Voxel-level 3D Style Loss)**

   将多视角渲染特征通过可微反投影融合到体素网格 $\mathcal{G}_b^l$，然后在体素空间计算风格统计量与参考风格的匹配：

    $\mathcal{L}_{\text{sty}}^{3D} = \frac{1}{B} \sum_{b=1}^B \sum_{l=1}^5 \alpha_l \left(\|\mu(\mathcal{G}_b^l) - \mu(\mathcal{S}_b^l)\|_2^2 + \|\sigma(\mathcal{G}_b^l) - \sigma(\mathcal{S}_b^l)\|_2^2\right)$

   相比图像级风格损失（每帧独立匹配，不保证跨视角一致性）和场景级损失（2D特征拼接，仍在2D空间），体素级损失在3D空间直接编码几何并强制跨视角风格一致性。

3. **预测头设计**

    - 几何头：DPT回归头输出位置、尺度、旋转、不透明度
    - 风格头：颜色头预测球谐系数
    - 辅助头：VGGT相机头估计内外参，深度头预测场景几何

### 损失函数 / 训练策略

**阶段1 - 几何预训练**：用VGGT权重初始化，端到端学习几何。随机选择一个输入视角做颜色抖动作为风格参考（避免恒等映射）。损失：$\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}}$

**阶段2 - 风格化微调**：冻结几何模块，仅更新Style Aggregator和颜色头。损失：
$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}}^{3D} + \lambda_{\text{cnt}} \mathcal{L}_{\text{content}} + \lambda_{\text{clip}} \mathcal{L}_{\text{clip}} + \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}$$

## 实验关键数据

### 主实验

| 数据集/场景 | 指标 | Stylos | StyleGaussian | Styl3R | 说明 |
|--------|------|------|----------|------|------|
| T&T Short LPIPS↓ | 一致性 | **0.033-0.047** | 0.031-0.038 | - | 竞争性 |
| T&T Long LPIPS↓ | 一致性 | **0.153** | 0.157 | - | 长程一致性更好 |
| CO3D ArtScore↑ | 艺术质量 | **9.15** | - | - | 体素损失最高 |
| CO3D 重建PSNR↑ | 重建 | 21.68 | - | - | Global CrossBlock |

### 消融实验

| 配置 | Short RMSE↓ | ArtScore↑ | 说明 |
|------|---------|---------|------|
| Image-level 风格损失 | 0.038 | 4.78 | 基线 |
| Scene-level 风格损失 | 0.036 | 9.12 | +4.34 ArtScore |
| 3D Voxel-level 损失 | **0.034** | **9.15** | 三维最优 |

### 关键发现

- Global CrossBlock 在所有测试类别上优于 Frame 和 Hybrid 变体
- 体素级3D风格损失在一致性和艺术质量上均优于2D风格损失
- 每批视角数在32以内时质量稳定，超过64时出现边缘伪影（训练设置最多24视角）
- Image-level损失有时完全无法迁移风格（如donut场景）

## 亮点与洞察

1. **几何-风格解耦**：骨干特征仅驱动几何，CrossBlock仅影响颜色，概念清晰且模块化
2. **2D→3D风格损失演进**：系统性地从图像级→场景级→体素级推进，提供了清晰的消融路径
3. **可扩展性强**：框架天然支持1到数百视角，仅调整批大小即可
4. **基于VGGT的强几何基础**：利用预训练3D基础模型确保高质量几何

## 局限与展望

- 超过32视角时质量下降，可能需要更大训练批次覆盖
- 仅评估了静态场景，动态场景风格化是未来方向
- 风格参考仅支持单张图像，多风格参考可能提供更丰富的控制
- 体素化步骤的分辨率对风格质量的影响需要更多分析

## 相关工作与启发

- VGGT (Wang et al., 2025a) 和 AnySplat (Jiang et al., 2025) 提供了强大的无姿态3D重建基础
- ArtFlow (An et al., 2021) 的特征级风格/内容损失被有效扩展到3D体素空间
- 体素级统计量匹配的思路可能适用于其他需要3D一致性的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 体素级3D风格损失和CrossBlock设计有创新，但整体框架是成熟组件的组合
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估，消融系统性强，但基线对比可以更丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，但部分描述可以更简洁
- 价值: ⭐⭐⭐⭐ 首个真正可扩展的单次3D风格化方法，实用价值明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2026/3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](../../CVPR2026/3d_vision/reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](../../CVPR2026/3d_vision/brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[ICLR 2026\] Station2Radar: Query-Conditioned Gaussian Splatting for Precipitation Field](station2radar_query_conditioned_gaussian_splatting_for_precipitation_field.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->
