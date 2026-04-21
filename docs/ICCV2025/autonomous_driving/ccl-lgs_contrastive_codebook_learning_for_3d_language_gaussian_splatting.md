---
title: >-
  [论文解读] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting
description: >-
  [ICCV 2025][自动驾驶][3D高斯喷溅] 提出CCL-LGS框架，通过零样本跟踪器实现跨视角掩码关联，并利用对比码本学习（CCL）模块蒸馏出类内紧凑、类间可区分的语义特征，从而解决基于2D先验的3D语义场重建中因遮挡、模糊和视角变化导致的跨视角语义不一致问题。
tags:
  - ICCV 2025
  - 自动驾驶
  - 3D高斯喷溅
  - 开放词汇语义分割
  - 对比学习
  - 码本学习
  - 跨视角一致性
---

# CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2505.20469](https://arxiv.org/abs/2505.20469)  
**代码**: https://epsilontl.github.io/CCL-LGS/  
**领域**: 3D场景理解 / 自动驾驶  
**关键词**: 3D高斯喷溅, 开放词汇语义分割, 对比学习, 码本学习, 跨视角一致性

## 一句话总结

提出CCL-LGS框架，通过零样本跟踪器实现跨视角掩码关联，并利用对比码本学习（CCL）模块蒸馏出类内紧凑、类间可区分的语义特征，从而解决基于2D先验的3D语义场重建中因遮挡、模糊和视角变化导致的跨视角语义不一致问题。

## 研究背景与动机

近年来，3D高斯喷溅（3DGS）结合视觉语言模型（如CLIP、SAM）在3D开放词汇场景理解中取得显著进展。现有方法通常采用"投影监督"范式：将3D语义渲染到2D视角后与预训练VLM提取的特征进行比较。然而，这一范式严重依赖2D语义特征在不同视角之间保持一致的假设。

实际场景中存在三类典型问题：

**遮挡**：同一物体在不同视角中可见部分不同，导致CLIP提取的语义特征不一致

**图像模糊**：运动模糊使得掩码质量下降，语义编码变得嘈杂

**视角依赖变化**：光照、反射等因素导致同一物体在不同视角下呈现不同外观

现有方法（LangSplat、LEGaussians、GOI等）虽通过3D几何一致性部分缓解了这些问题，但由于仍依赖2D监督，显著的输入特征不一致仍会传播到3D空间，导致渲染伪影。CCL-LGS的核心创新在于不直接使用不完美的CLIP特征，而是通过显式建模跨视角语义对齐来解决这一被忽视的根本问题。

## 方法详解

### 整体框架

CCL-LGS包含三个阶段：
1. **双层语义特征提取**：利用SAM生成多尺度掩码，通过CLIP提取像素级语义特征
2. **对比码本学习（CCL）**：通过零样本跟踪器进行掩码关联，然后用对比损失组织和精炼语义特征
3. **3D高斯语义场优化**：将精炼后的语义信息集成到3DGS中，使用交叉熵损失进行端到端优化

### 关键设计

#### 1. 双层语义特征提取
- **功能**：从多视角图像中提取精确且多尺度的语义特征
- **核心思路**：使用SAM的32×32均匀点提示生成三类掩码（子部分、部分、整体），然后合并为两组聚合掩码 $M_o^{sp}$（子部分+部分）和 $M_o^{wp}$（整体+部分）。对每个像素 $v$，其语义特征为：

$$F_i(v) = \text{CLIP}(I_t \odot M_i(v))$$

- **设计动机**：相比LangSplat依赖三个独立尺度（引入尺度错误），本方法在统一框架内整合多尺度信息，既保持精确边界又减少计算开销

#### 2. 对比码本学习（CCL）模块
- **功能**：将不同视角中同一物体的语义特征对齐，同时保持不同物体间的可区分性
- **核心思路**：分为两步：
    - **掩码关联**：使用SAM2将首帧的 $K$ 个掩码传播到所有帧，通过IoU匹配为每个掩码分配类别标签 $y_i \in \{1,2,...,K,-1\}$（IoU > 0.5时匹配，否则标记为-1）
    - **对比损失**：构建码本 $T = \{T_j\}_{j=1}^N$，对每个特征 $F_i$ 找最相似原型：

$$j^* = \underset{j}{\text{argmax}} \cos(F_i, T_j)$$

$$L_{\text{max}} = 1 - \cos(F_i, T_{j^*})$$

  - 同类特征施加拉近损失：$L_{\text{pull}} = 1 - \cos(T_{j_i}, T_{j_k})$（当 $y_i = y_k \neq -1$）
  - 异类特征施加推远损失：$L_{\text{push}} = \text{ReLU}(\cos(T_{j_i}, T_{j_k}) - m)$（当 $y_i \neq y_k$，均不为-1）
- **设计动机**：与自编码器相比，码本方法隐式约束相似特征映射到同一条目，提供更强的一致性约束；对比损失确保同类聚合、异类分离，有效缓解不完美掩码引入的语义歧义

#### 3. 3D高斯语义场优化
- **功能**：利用训练好的码本构建最终的3D语义场
- **核心思路**：将每个像素的语义特征转换为离散索引映射 $\mathcal{M} \in \mathbb{R}^{H \times W}$，通过轻量MLP解码器和softmax层生成语义分布 $\hat{\mathcal{M}} \in \mathbb{R}^{H \times W \times N}$，使用交叉熵损失优化：

$$\mathcal{L}_{CE} = \text{CE}(\hat{\mathcal{M}}, \mathcal{M})$$

推理时，通过文本查询 $\tau$ 的CLIP编码计算相关性图：$p(\tau|v) = \frac{\exp(\tilde{F}(v) \cdot \varphi(\tau) / \|\tilde{F}(v)\| \|\varphi(\tau)\|)}{\sum_{s \in \mathcal{T}} \exp(\tilde{F}(v) \cdot \varphi(s) / \|\tilde{F}(v)\| \|\varphi(s)\|)}$

### 损失函数 / 训练策略

总损失为三项加权和：

$$L = L_{\text{max}} + \lambda_{\text{pull}} L_{\text{pull}} + \lambda_{\text{push}} L_{\text{push}}$$

训练设置：$\lambda_{\text{pull}} = \lambda_{\text{push}} = 0.25$，margin $m = 0.7$，低维特征维度 $d_f = 8$，Adam优化器（lr=0.001），训练30000轮。

## 实验关键数据

### 主实验

在LERF数据集上的mIoU比较：

| 方法 | Ramen | Figurines | Teatime | Waldo Kitchen | 平均 |
|------|-------|-----------|---------|---------------|------|
| LangSplat | 51.2 | 44.7 | 65.1 | 44.5 | 51.4 |
| GOI | 52.6 | 44.5 | 63.7 | 41.4 | 50.6 |
| 3D VL-GS | 61.4 | 58.1 | 73.5 | 54.8 | 62.0 |
| **CCL-LGS** | **62.3** | **61.2** | 71.8 | **67.1** | **65.6** |

在3D-OVS数据集上的mIoU比较：

| 方法 | Bed | Bench | Sofa | Lawn | 平均 |
|------|-----|-------|------|------|------|
| LangSplat | 92.5 | 94.2 | 90.0 | 96.1 | 93.2 |
| 3D VL-GS | **96.8** | **97.3** | **95.5** | **97.9** | **96.9** |
| **CCL-LGS** | 97.3 | 95.0 | 92.3 | 96.1 | 95.2 |

### 消融实验

在LERF数据集上的CCL模块消融：

| 配置 | Ramen | Figurines | Teatime | Kitchen | 平均 | 说明 |
|------|-------|-----------|---------|---------|------|------|
| Baseline | 46.8 | 57.1 | 60.8 | 61.0 | 56.4 | 纯码本压缩 |
| +Pull loss | 48.0 | 58.0 | 70.1 | 62.0 | 59.5 | 增强类内一致性 |
| +Push loss | 55.1 | 61.0 | 66.0 | 59.3 | 60.4 | 减少假激活 |
| **Full model** | **62.3** | **61.2** | **71.8** | **67.1** | **65.6** | 两项损失互补 |

### 关键发现

1. Pull loss在处理遮挡/部分可见物体（如"glass of water"）时提升类内一致性效果尤为显著
2. Push loss有效减少混淆区域的假激活（如"kamaboko"周围）
3. 在遮挡和模糊较少的3D-OVS数据集上，数据增强策略（3D VL-GS）获益更大；而在挑战性更强的LERF数据集上，CCL-LGS的跨视角一致性优势更为明显

## 亮点与洞察

- **核心洞察**：首次关注2D-to-3D语义场重建中被忽视的跨视角特征对齐问题，指出几何一致性约束不足以解决语义不一致
- **巧妙设计**：利用SAM2的零样本跟踪能力实现掩码关联，无需额外训练即可获得跨视角对应关系
- **对比码本**：结合码本量化和对比学习的双重优势，既保证特征一致性又保持类别可区分性

## 局限与展望

- 依赖SAM和SAM2的质量，不完美掩码仍会影响最终结果
- 在简单场景（3D-OVS）上未能超越3D VL-GS的数据增强策略
- 码本大小 $N$ 与实际物体类别数 $K$ 独立设置，如何自适应确定仍是开放问题
- 未探索与基于DINO的方法结合的方案

## 相关工作与启发

- **LangSplat**和**LEGaussians**是主要对比基线，它们分别使用自编码器和码本进行特征压缩，但未解决跨视角一致性问题
- 视频目标分割方法（XMem、SAM2）被创新性地引入到3D场景理解中，用于建立跨视角对应关系
- 对于需要高质量3D语义的下游任务（机器人操作、自动驾驶），该方法提供了更可靠的语义场

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次关注跨视角语义不一致并提出对比码本解决方案
- 实验充分度: ⭐⭐⭐⭐ 消融完整，可视化分析充分，但数据集规模较小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 解决了3D开放词汇理解中的实际瓶颈问题

<!-- RELATED:START -->

## 相关论文

- [GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multi-view_robust_physical_adversarial_camouflage_g.md)
- [Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [Generative Gaussian Splatting for Unbounded 3D City Generation](../../CVPR2025/autonomous_driving/generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)

<!-- RELATED:END -->
