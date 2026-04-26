---
title: >-
  [论文解读] FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting
description: >-
  [CVPR 2026][图像分割][CAD B-rep 学习] 提出 FoV-Net，首个在 CAD B-rep 学习中同时捕获局部表面几何和全局结构上下文的旋转不变框架，通过局部参考系 UV 网格(LRF UV)和视场光线投射(FoV)描述子实现了在任意 $\mathbf{SO}(3)$ 旋转下的鲁棒分类和分割。
tags:
  - CVPR 2026
  - 图像分割
  - CAD B-rep 学习
  - 旋转不变性
  - 光线投射
  - 图注意力网络
  - 3D 分割
---

# FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting

**会议**: CVPR 2026  
**arXiv**: [2602.24084](https://arxiv.org/abs/2602.24084)  
**代码**: [GitHub](https://github.com/UGent-CVAMO/fovnet)  
**领域**: 分割  
**关键词**: CAD B-rep 学习, 旋转不变性, 光线投射, 图注意力网络, 3D 分割

## 一句话总结

提出 FoV-Net，首个在 CAD B-rep 学习中同时捕获局部表面几何和全局结构上下文的旋转不变框架，通过局部参考系 UV 网格(LRF UV)和视场光线投射(FoV)描述子实现了在任意 $\mathbf{SO}(3)$ 旋转下的鲁棒分类和分割。

## 研究背景与动机

CAD 边界表示(B-rep)为图结构，面为节点、共享边为连接，天然适合 GNN 处理。现有 B-rep 学习方法的核心挑战在于设计既能捕获局部表面几何又能编码全局结构上下文的描述子。

UV-Net 开创性地将 UV 参数域采样为网格，存储全局坐标系下的绝对坐标 $(x,y,z)$ 和法向量 $(n_x, n_y, n_z)$，成为后续许多方法的基础。然而这种依赖全局坐标的方式带来了严重的**旋转敏感性**问题。作者的实验揭示了一个惊人的现象：

> 在对齐基准上准确率超过 95% 的模型，在任意 $\mathbf{SO}(3)$ 旋转下可能**崩塌至 10%**。

这在实际制造流水线中是不可接受的——CAD 模型来自不同来源，分析必须对任意朝向鲁棒。旋转增强虽能部分缓解但无法覆盖所有旋转且有计算成本。更重要的是，即使在对齐数据上，旋转不变性对分割任务也至关重要：面在不同位置的变化可能导致虚假的位姿关联，在小数据场景下损害泛化能力。

## 方法详解

### 整体框架

FoV-Net 的输入表示由两部分组成：
1. **LRF UV 网格**：在每个面的局部参考系中编码本征表面几何
2. **FoV 网格**：通过光线投射捕获面周围的 3D 结构上下文

每种描述子由轻量 CNN 提取特征，与面属性拼接后通过 MLP 融合为统一的面嵌入，最终在 B-rep 图上通过图注意力网络(GAT)传播信息。

### 关键设计

1. **局部参考系 UV 网格(LRF UV)**：核心创新是将标准 UV 网格从全局坐标系转换到每个面的局部坐标系。在面中心 $\mathbf{o}$ 构建正交基：$\mathbf{N}$ 为法向量，$\mathbf{U}$ 为 U 方向切向量投影归一化，$\mathbf{V} = \mathbf{N} \times \mathbf{U}$。变换 $\mathbf{p}' = \mathbf{R}_f^\top(\mathbf{p} - \mathbf{o})$ 使得同一面在不同位姿下产生完全相同的描述子。这利用了 B-rep 面天然提供面片离散化和参考方向的优势，避免了点云 LRF 构建中的噪声和一致性问题。结果为 $n_u \times n_v \times 7$ 的张量（相对坐标 + 法向量 + 裁剪掩码）。设计动机：解耦局部几何与全局位姿，保证旋转不变性。

2. **视场(FoV)描述子**：LRF UV 虽然保留了局部几何但丢失了结构上下文。FoV 通过光线投射恢复这一信息。从面中心 $\mathbf{o}$ 沿法向量半球发射光线，按仰角 $n_{\text{el}}$ 和方位角 $n_{\text{az}}$ 离散化（默认 $6 \times 12$，对应 15° 和 30° 步长）。每条光线记录三个量：(1) 命中标志，(2) 到交点的距离，(3) 光线方向与表面法向的点积（入射角编码）。由于光线原点和方向在局部参考系中定义，它们随面共同旋转，保证旋转不变性。

    - **外向视觉(OV)**：沿 $\mathbf{N}$ 方向半球投射，探测面的外部环境
    - **内向视觉(IV)**：沿 $-\mathbf{N}$ 方向半球投射，探测实体内部结构。对于水密实体，内向光线密集相交，提供丰富的距离信息
   
   OV + IV 提供互补的旋转不变结构上下文表示，且天然具有平移不变性。

3. **轻量网络架构**：

    - OV/IV 编码器：2 层 CNN ($32 \to 64$ 通道) + 全局平均池化 + 线性投影到 64 维，方位轴使用圆形填充处理周期性
    - LRF UV 编码器：3 层 CNN ($32 \to 64 \to 128$) + 全局平均池化 + 64 维
    - 面属性：独热编码面类型(6D) + 面积 = 7D
    - 2 层融合 MLP（隐藏维度 256）→ 64 维面嵌入
    - 3 层 GAT（4 注意力头，64 维）在 B-rep 图上传播信息
   
   设计动机：省略了边特征（消融实验表明增益极小），降低计算开销。

### 损失函数 / 训练策略

- 分类：全局最大池化 + 2 层 MLP 分类头
- 分割：面嵌入直接送入逐面预测头
- 优化器：Adam (lr=0.001, batch size 64)，早停策略 (patience 30)
- 单卡 NVIDIA RTX A5000(24GB)，每组实验重复 5 次取均值±标准差

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | FoV-Net (旋转) | FoV-Net (原始) | UV-Net (旋转) | AAGNet (旋转) |
|--------|------|------|----------------|----------------|---------------|---------------|
| SolidLetters | 分类 | Acc% | **96.35** | 96.35 | 8.94 | 14.03 |
| TraceParts | 分类 | Acc% | **100.00** | 100.00 | 45.67 | 91.33 |
| Fusion360 | 分割 | Acc% | **91.72** | 91.72 | 69.13 | 79.85 |
| Fusion360 | 分割 | IoU% | **73.81** | 73.81 | 37.07 | 53.42 |
| MFCAD++ | 分割 | Acc% | **99.33** | 99.33 | 35.44 | 80.13 |
| MFCAD++ | 分割 | IoU% | **97.81** | 97.81 | 18.79 | 64.70 |

FoV-Net 旋转前后性能完全一致；UV-Net 在 SolidLetters 上从 97.10% 崩塌到 8.94%。

### 消融实验

| 配置 | SolidLetters Acc% | 说明 |
|------|-------------------|------|
| FoV-Net 完整 | 96.35 | LRF UV + FoV 互补 |
| 仅 FoV 网格 | 95.79 | 结构上下文信息丰富 |
| 仅 LRF UV | 94.39 | 局部几何也很强 |
| 仅 OV | 92.92 | 外部视觉 |
| 仅 IV | 93.40 | 内部视觉略优于外部 |
| 仅面属性 | 70.68 | 简单属性不够 |
| 仅拓扑 | 37.72 | 图结构远远不够 |

### 关键发现

- FoV 网格和 LRF UV 高度互补，组合优于任一单独使用
- 即使很小的 FoV 分辨率（如 $4 \times 2$）也能产生有效特征；但单条光线($1 \times 1$)崩塌至 75%
- **数据效率惊人**：FoV-Net 在 MFCAD++ 上仅用 50 个样本即达 80% 准确率，UV-Net 需要约 10000 个样本
- 旋转增强虽提升旋转鲁棒性，但代价是整体性能下降——FoV-Net 避免了这种权衡

## 亮点与洞察

1. **问题揭示力度强**：首次系统量化了 B-rep 学习中旋转敏感性的严重程度，95%→10% 的崩塌极具冲击力
2. **光线投射用于描述子构建**：将计算机图形学的经典技术创造性地引入 B-rep 学习，利用 CAD 内核的精确求交能力，从面中心观察周围结构
3. **内/外双视觉互补**：外向光线探测环境、内向光线探测内部结构，两者组合提供完整的 3D 上下文
4. **B-rep 面的 LRF 优势**：B-rep 面天然提供参数化和参考方向，LRF 构建比点云简单得多且更可靠
5. **数据效率**：在工业 CAD 领域（知识产权限制数据可用性），低数据需求是重要优势

## 局限与展望

- 光线投射基于 PythonOCC CAD 内核（CPU），仅支持 CPU 并行化，未利用 GPU 加速
- 当前实验聚焦于中等复杂度的单零件 B-rep，更大装配体的可扩展性待验证
- FoV 网格使用等角 3D→2D 映射，存在类似地理投影的极点畸变，球面 CNN 可能提供更均匀的方向参数化
- 未处理 UV 轴翻转/交换带来的歧义性（UV-Net 通过 D2 等变卷积缓解）
- 省略了边特征，在 B-rep 生成等边特征是关键的任务中可能受限

## 相关工作与启发

- UV-Net 开创了 UV 网格范式但未解决旋转问题——FoV-Net 通过 LRF 转换彻底解决
- 点云旋转不变方法（如 LRF + 全局上下文组合）的思路被成功迁移到 B-rep 域
- 光线投射在 CAD 领域已有工具可达性估计的应用，FoV-Net 将其扩展为通用的面描述子
- FoV 描述子在对比学习预训练和无监督 CAD 检索中有很大潜力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个旋转不变的 B-rep 学习框架，光线投射描述子的思路非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、分类+分割、旋转/原始对比、详细消融、数据效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题揭示有力，可视化清晰，结构严谨
- 价值: ⭐⭐⭐⭐⭐ 解决了 B-rep 学习中长期存在的旋转敏感性问题，工业应用前景广阔

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection](pointer-cad_unifying_b-rep_and_command_sequences_via_pointer-based_edges_faces_s.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[CVPR 2026\] PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](prue_a_practical_recipe_for_field_boundary_segmentation_at_scale.md)
- [\[CVPR 2025\] V-CLR: View-Consistent Learning for Open-World Instance Segmentation](../../CVPR2025/segmentation/v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)

<!-- RELATED:END -->
