---
title: >-
  [论文解读] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees
description: >-
  [AAAI 2026][可解释性][Shapley值] 提出 ShapBPT，将**数据感知的二叉分割树（BPT）**作为层次联盟结构与 Owen 近似的 Shapley 值结合，实现与图像形态学对齐的特征归因，比现有 Shapley 方法收敛更快、形状识别更准确，并通过 20 人用户研究确认解释更受人类偏好。
tags:
  - "AAAI 2026"
  - "可解释性"
  - "Shapley值"
  - "二叉分割树"
  - "特征归因"
  - "图像分类"
---

# ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees

**会议**: AAAI 2026  
**arXiv**: [2602.07047](https://arxiv.org/abs/2602.07047)  
**代码**: [https://github.com/amparore/shap_bpt](https://github.com/amparore/shap_bpt)  
**领域**: 可解释AI / 计算机视觉  
**关键词**: Shapley值, 二叉分割树, 可解释性, 特征归因, 图像分类

## 一句话总结

提出 ShapBPT，将**数据感知的二叉分割树（BPT）**作为层次联盟结构与 Owen 近似的 Shapley 值结合，实现与图像形态学对齐的特征归因，比现有 Shapley 方法收敛更快、形状识别更准确，并通过 20 人用户研究确认解释更受人类偏好。

## 研究背景与动机

1. **领域现状**：像素级特征归因是计算机视觉可解释AI（XCV）的重要工具。SHAP 基于博弈论计算 Shapley 值，LIME 通过预分割计算区域重要性。
2. **现有痛点**：(a) SHAP 使用数据无关的轴对齐（AA）网格层次结构，不利用图像的多尺度形态结构，导致收敛慢且与实际物体形状对齐差；(b) LIME 依赖预定义分割，一旦分割不准确则解释差且无法自适应细化；(c) 层次 Shapley 方法从未利用数据感知的层次结构用于 CV 任务。
3. **核心矛盾**：Owen 近似的计算代价随层次深度指数增长（$O(4^d)$），需要尽量少的递归分割就能到达相关区域。AA 网格无法做到这一点。
4. **本文目标**：让 Shapley 归因方法利用图像的形态学信息，用更少的计算预算达到更好的物体定位和形状识别。
5. **切入角度**：将 MPEG-7 编码中的 BPT 算法重新用于构建数据感知的层次联盟结构，像素按颜色均匀性和空间邻近性从底向上合并。
6. **核心 idea**：颜色和形状相似的像素区域很可能有相同的 Shapley 值，用 BPT 将它们合并能最大化 Owen 公式的效率。

## 方法详解

### 整体框架

输入图像先构建 BPT 层次结构（底向上逐步合并相邻且颜色相似的像素区域）。然后在该 BPT 层次上应用 Owen 近似公式递归计算 Shapley 归因值。采用自适应分裂策略——每次分裂 Shapley 值总和最大的区域，在给定评估预算下最大化解释精度。

### 关键设计

1. **BPT 层次联盟结构生成**

    - 功能：构建与图像形态学对齐的二叉分层分割。
    - 核心思路：从 $n$ 个像素出发，每次合并距离最小的两个相邻分区。距离函数 $dist(T_i, T_j) = clr^2(T_i, T_j) \cdot area(T_i, T_j) \cdot \sqrt{pr(T_i, T_j)}$，其中 $clr^2$ 为合并区域的颜色范围平方和、$area$ 和 $pr$ 分别为面积和周长。合并 $n-1$ 次后得到完整二叉树。颜色相似、空间相邻的像素优先合并——确保分区边界与物体轮廓对齐。
    - 设计动机：满足两个关键需求——R1：尽量少的递归分割即到达相关区域（形态学对齐的分割只需少量切割即可分离物体和背景）；R2：分区不能预先固定（BPT 的自适应分裂可根据 Shapley 值分布动态细化）。

2. **Owen 近似的 Shapley 值计算**

    - 功能：在二叉层次联盟结构上高效计算像素级归因值。
    - 核心思路：递归公式 $\Omega_i(Q, T) = \frac{1}{2}\Omega_i(Q \cup T_2, T_1) + \frac{1}{2}\Omega_i(Q, T_1)$（$i \in T_1$），不可分分区则均匀分配边际贡献 $\frac{1}{|T|}\Delta_T(Q)$。自适应分裂策略在给定预算 $b$ 下，每次迭代分裂 Shapley 值总和最大的分区（每次分裂需 2 次模型评估）。
    - 设计动机：精确 Shapley 值计算是 #P-hard 的。Owen 近似通过层次分组大幅降低代价，但仍依赖层次结构的质量。BPT 的形态学对齐让少量递归即可准确定位。

3. **评估框架**

    - 功能：全面评估解释质量。
    - 核心思路：响应型指标——$AUC^+$（逐步加入高 Shapley 值像素时模型输出的 AUC）和 $AUC^-$（逐步移除时）。真值型指标——$AU\text{-}IoU$（IoU 曲线下面积）和 $max\text{-}IoU$（最大 IoU）。

### 损失函数 / 训练策略

无需训练。BPT 构建和 Shapley 值计算均为确定性算法。主要超参为评估预算 $b$（模型调用次数），实验中测试 100、500、1000。

## 实验关键数据

### 主实验

7 个实验覆盖多种 CV 任务和模型（ResNet50、SwinViT、ViT、YOLO、CNN、VAE-GAN），在 ImageNet-S50 (E1) 上的结果：

| 方法 | AUC+↑ | AUC-↓ | AU-IoU↑ | max-IoU↑ | 时间(s) |
|------|-------|-------|---------|----------|--------|
| GradCAM | 中等 | 中等 | 中等 | 中等 | 快 |
| LIME-1000 | 中等 | 中等 | 较好 | 较好 | 慢 |
| AA-1000 (SHAP) | 中等 | 中等 | 中等 | 中等 | 中等 |
| **BPT-1000** | **最优** | **最优** | **最优** | **最优** | 中等 |
| BPT-500 | 次优 | 次优 | 次优 | 次优 | 较快 |
| BPT-100 | 可比AA-500 | 可比AA-500 | 优于AA-500 | 优于AA-500 | 快 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| BPT vs AA（同预算） | BPT 显著更优 | 数据感知层次压倒性优势 |
| BPT-100 vs AA-1000 | BPT-100 可比 | 10倍更少的预算达到相同效果 |
| 距离函数消融 | $clr^2 \times area \times \sqrt{pr}$ 最优 | 颜色+面积+周长综合效果最好 |
| 用户研究(E8) | BPT 被偏好 | 20人研究显著偏好 BPT 解释 |

### 关键发现

- BPT 在所有实验和所有指标上一致优于 AA 和其他方法。
- BPT-100（100次模型调用）在物体定位上可比 AA-1000（1000次调用），收敛速度提升约 10 倍。
- 对 Vision Transformer（E3、E7）尤其有效——其他方法在 ViT 上产生混乱的显著性图，BPT 仍然清晰聚焦。
- 在异常检测（E6）和目标检测（E4）等不同 CV 任务上同样适用。
- 20 人用户研究确认 BPT 解释更受人类偏好。

## 亮点与洞察

- **将 MPEG-7 的 BPT 重用于可解释 AI**：这是一个优雅的跨领域技术迁移——BPT 最初用于视频编码的多尺度表示，这里用它作为 Shapley 值的层次联盟结构。
- **数据感知 + 理论保证**：BPT 提供数据感知的分割，Owen 公式提供理论保证，两者结合既有效又有理论基础。
- **10 倍效率提升**：用 100 次模型评估达到 1000 次 AA 方法的效果，对实际部署中模型调用成本高的场景（如 API 调用）意义重大。

## 局限与展望

- BPT 构建本身需要遍历所有像素对，对高分辨率图像可能成为瓶颈。
- 当前使用颜色作为合并准则，对纹理主导的场景（如材质识别）可能不够。
- 仅适用于图像数据，未探索扩展到视频或 3D 数据的可能性。
- BPT 是确定性的——不同的初始超像素可能给出不同结果，鲁棒性分析不足。

## 相关工作与启发

- **vs SHAP Partition Explainer**：使用 AA 网格，不利用数据信息，ShapBPT 收敛更快且形状更准确。
- **vs LIME**：LIME 依赖固定预分割无法细化；ShapBPT 的 BPT 支持自适应分裂。
- **vs GradCAM**：GradCAM 快但模糊，ShapBPT 形状更清晰但需更多计算。
- 该框架可迁移到医学影像解释（X-ray、CT 等），其中物体边界清晰且形态学信息丰富。

## 评分

- 新颖性: ⭐⭐⭐⭐ BPT 与 Shapley 值的结合是首创
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个实验覆盖多任务多模型 + 用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，示例图示清晰
- 价值: ⭐⭐⭐⭐ 对 XAI 领域的实际贡献大，开源且有 pip 包

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[AAAI 2026\] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)

</div>

<!-- RELATED:END -->
