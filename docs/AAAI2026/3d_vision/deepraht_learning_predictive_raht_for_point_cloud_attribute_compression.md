---
title: >-
  [论文解读] DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression
description: >-
  [AAAI 2026][3D视觉][点云压缩] 提出首个端到端可微的 RAHT（Region Adaptive Hierarchical Transform）框架 DeepRAHT，用于有损点云属性压缩，通过可学习的预测模型和基于 Laplace 分布的码率代理实现了超越 G-PCC 标准和现有深度学习方法的压缩性能。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "点云压缩"
  - "属性压缩"
  - "RAHT"
  - "端到端学习"
  - "可变码率"
---

# DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression

**会议**: AAAI 2026  
**arXiv**: [2601.12255](https://arxiv.org/abs/2601.12255)  
**代码**: [有](https://github.com/zb12138/DeepRAHT)  
**领域**: 3D视觉  
**关键词**: 点云压缩, 属性压缩, RAHT, 端到端学习, 可变码率

## 一句话总结

提出首个端到端可微的 RAHT（Region Adaptive Hierarchical Transform）框架 DeepRAHT，用于有损点云属性压缩，通过可学习的预测模型和基于 Laplace 分布的码率代理实现了超越 G-PCC 标准和现有深度学习方法的压缩性能。

## 研究背景与动机

点云属性压缩（PCAC）是 3D 数据处理的关键环节。RAHT 作为 MPEG G-PCC 标准的核心变换方法，性能优秀且复杂度低。但现有 RAHT 在深度学习中的应用面临几个问题：

**非可微性**：G-PCC 中的 RAHT 基于 C++ 实现，不可微分，无法端到端训练

**预测缺失**：3DAC（首个学习 RAHT 系数的方法）只用手工 RAHT 生成变换系数再学习熵编码，忽略了 G-PCC 标准中至关重要的预测 RAHT

**仅优化码率**：由于非可微性，3DAC 只能优化码率而无法联合优化失真

**鲁棒性差**：现有方法对数据方差敏感，需多个模型覆盖不同码率点

**预测 RAHT 的可学习性未被探索**：预测可以显著降低变换系数的不确定性，编码残差比编码系数更高效

## 方法详解

### 整体框架

DeepRAHT 的核心流程：

1. **多尺度生成**：对输入点云 $P_0$ 进行 $s$ 次 $2 \times 2 \times 2$ 步长的 sum-pooling，得到 $\{P_1, ..., P_s\}$
2. **自顶向下编码**：从最粗尺度 $s$ 开始，每个尺度应用变换模型（Haar）和可选的预测模型
3. **边编码边重建**：重建的 $\hat{A}_m$ 用于下一层的 DC 重建和预测
4. **解码**：与重建过程完全一致，保证可逆性

### 关键设计

#### 基于稀疏卷积的可微 RAHT（变换模型）

核心创新是用 Minkowski 稀疏张量和稀疏卷积实现可微的 dyadic RAHT：

**Haar 变换**：对每个 $2 \times 2 \times 2$ 体素，将 8 个节点分解为 1 个直流系数（DC）和 7 个交流系数（AC）。按 Z→Y→X 轴依次进行二分分解：

$$\begin{bmatrix} g_L \\ g_H \end{bmatrix} = \frac{1}{\sqrt{w_1+w_2}} \begin{bmatrix} \sqrt{w_1} & \sqrt{w_2} \\ -\sqrt{w_2} & \sqrt{w_1} \end{bmatrix} \begin{bmatrix} g_1 \\ g_2 \end{bmatrix}$$

其中 $w_1, w_2$ 是节点包含的原始点数，作为自适应权重。

**稀疏卷积实现**：
- Z 轴分解：$\text{Zconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,1,2))$
- Y 轴分解：$\text{Yconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,2,1))$
- X 轴分解：$\text{Xconv} \equiv \text{Conv}(i=1, o=2, k=s=(2,1,1))$
- 初始卷积核权重为单位矩阵 $I_2$

关键性质：**DC 等价于下一尺度的归一化属性**：$DC_m \equiv g_{LLL} = A_{m+1,i}/\sqrt{w_{m+1,i}}$，因此 DC 无需编码（已在更高尺度编码），只需编码 7 个 AC 系数。

逆 Haar 变换使用 ConvolutionTranspose 实现。

#### 预测模型

G-PCCv14 使用逆距离加权（IDW）预测，但使用同层兄弟节点会引入自回归依赖、增加解码时间。DeepRAHT 仅使用父层尺度进行预测：

**IDW 预测**（稀疏卷积实现）：
$$\text{IDW}(\hat{a}_m) \equiv \text{Conv}(\text{Unpool}(\hat{a}_m), k=3^3, s=1^3)$$

其中卷积核权重按距离设置比例：中心:面:边:角 = 4:3:2:1。

**预测补偿模块**：利用祖父层（$m+1$）的预测误差来补偿当前预测，避免自回归问题：
$$a'_{m-1} = \text{Comp}(\hat{a}_m - \text{IDW}(\hat{a}_{m+1})) + \text{IDW}(\hat{a}_m)$$

补偿模块由多层线性层和稀疏卷积组成（隐藏层维度 128，核大小 $3^3$），包含一个步长为 2 的转置卷积。预测后对 AC 系数取残差进行编码：$r_{m-1} = AC_{m-1} - AC'_{m-1}$。

补偿模块可根据预测性能决定是否启用（以 $s$ bits 信号通知解码端），保证性能下界为 G-PCCv14。

#### 熵编码器（码率代理）

现有方法使用 bottleneck 熵模型，但对数据方差敏感。DeepRAHT 使用**零游程编码**（zero run-length coding）替代，因为 RAHT 残差高度集中在零附近。

由于游程编码不可微，提出基于 Laplace 分布的码率代理：
$$q(r) = \int_{r-0.5}^{r+0.5} \mathcal{L}_{\mu,\sigma}(r)dr$$

参数 $\alpha=0.425, \mu=0, \sigma=0.2$ 通过拟合实际数据获得，决定系数达 0.991。

**可变码率优势**：只需调整量化步长 $qs$ 即可实现不同码率，无需训练多个模型（$qs = \{8,10,12,...,224\}$），而 3DAC 和 TSC-PCAC 需要每个码率点单独训练。

### 损失函数 / 训练策略

总损失函数：
$$\ell = \ell_{bits} + \lambda(\ell_{recon} + \ell_{pred})$$

- $\ell_{recon} = \|a_0 - \hat{a}_0\|_2^2$：端到端重建误差
- $\ell_{pred} = \sum_m \|(a_m - a'_m)\|_2^2$：预测损失，加速收敛
- $\ell_{bits} = -\sum_m \log_2 q(r_m/qs)$：码率代理损失
- $\lambda = 1/255$，$qs = 8$，Adam 优化器，学习率 0.0001，batch size 1
- 训练数据：RWTT 数据集（568 个真实世界物体）
- YUV 色彩空间压缩

## 实验关键数据

### 主实验

**BD-BR 增益（%，负值=节省码率，anchor = G-PCCv14）：**

| 方法 | Owlii Avg | 8iVSLF Avg | MPEG Avg | 总平均 |
|------|-----------|-----------|----------|--------|
| G-PCCv23 | -20.0 | -17.5 | -11.6 | -16.4 |
| 3DAC | -66.6 | -70.9 | -62.7 | -66.7 |
| TSC-PCAC | -12.8 | -68.5 | -73.2 | -51.5 |
| Unicorn | -7.1 | -10.9 | -4.0 | -7.3 |
| **DeepRAHT** | — | — | — | **基准** |

注：DeepRAHT 相比 G-PCCv23 平均节省 16.4% 码率，相比 Unicorn 节省 7.3% 码率，色度分量改进更大（U: 20.5%, V: 20.8%）。

**复杂度对比（8iVSLF，平均 325 万点/帧）：**

| 方法 | 编码时间 | 解码时间 | 模型大小 | GPU内存 |
|------|---------|---------|---------|---------|
| 3DAC | 38.45s | 51.71s | 1MB×5 | 10GB |
| TSC-PCAC | 7.86s | 26.87s | 148MB×5 | 22GB |
| Unicorn | 20.86s | 14.99s | 65MB×3 | 16GB |
| **DeepRAHT** | **6.03s** | **5.74s** | **88MB×1** | **8GB** |

### 消融实验

**loot_viewdep 上的消融（BD-rate gain vs G-PCCv14）：**

| 配置 | BD-rate gain |
|------|-------------|
| Vanilla RAHT（无预测）| 基准 |
| RAHT+Pred（IDW，≈G-PCCv14） | -48.2% |
| **RAHT+Pred+Comp（DeepRAHT）** | **-24.6%** (vs G-PCCv14) |
| vs G-PCCv23 | **-16.6%** |

### 关键发现

- 预测补偿模块在不使用兄弟上下文的情况下，性能甚至超越 G-PCCv23 的兄弟预测
- 码率代理的拟合精度极高（$R^2=0.991$），有效替代 bottleneck 熵模型
- DeepRAHT 是唯一在所有数据上都能成功压缩的深度学习方法（其他方法在某些大或稀疏点云上会失败）
- 单一模型覆盖 10 个码率点，而竞争方法需 3-5 个模型
- 可逆性保证失真仅来自量化，视觉上比 Unicorn 保留更多纹理细节

## 亮点与洞察

1. **端到端可微 RAHT 的首次实现**：将 G-PCC 标准中的核心算法完全用稀疏卷积重写，打通了深度学习与传统标准的桥梁
2. **性能下界保证**：框架结构与 G-PCCv14 完全一致，通过可选补偿模块和信号位确保不会比 G-PCCv14 差
3. **可变码率的优雅方案**：利用游程编码对 Laplace 分布的鲁棒性，单一模型通过调节量化步长即可覆盖宽码率范围
4. **DC = 下一尺度归一化属性**的等价关系，是避免冗余编码的关键理论基础
5. **实用性极强**：编解码最快、显存最少、鲁棒性最好

## 局限与展望

1. 训练仅在 RWTT 数据集上进行，泛化到 LiDAR 和动态点云有待验证
2. Batch size 限制为 1，大规模训练效率有瓶颈
3. 预测模型仅使用父/祖父尺度，未探索更远距离的上下文
4. 仅处理颜色属性，法线、反射率等其他属性的适用性未验证
5. 与 Gaussian Splatting 数据的结合（作者提及的潜在应用）尚未实验

## 相关工作与启发

- **G-PCC (tmc13v23)**：工业标准，DeepRAHT 与之结构对齐并超越，展示了学习方法替代手工设计的潜力
- **3DAC**：首个学习 RAHT 系数的方法，但非端到端且无预测——DeepRAHT 正是针对这两个缺陷的完整解决方案
- **Unicorn**：当前 SOTA 深度学习框架，用平均池化获取多尺度。DeepRAHT 的 RAHT 分解提供了更理论化的多尺度方案
- 启发：传统信号处理工具（如 Haar 小波变换）与深度学习的深度融合是压缩领域的重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ （端到端可微 RAHT 和码率代理的设计新颖，但整体框架遵循 G-PCC 结构）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个数据集全面评测 + 复杂度对比 + 可变码率 + 鲁棒性验证 + 消融）
- 写作质量: ⭐⭐⭐⭐ （技术描述精确，公式推导完整）
- 价值: ⭐⭐⭐⭐⭐ （直接对标工业标准 G-PCC，实用价值高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] Point Cloud Quantization through Multimodal Prompting for 3D Understanding](point_cloud_quantization_through_multimodal_prompting_for_3d_understanding.md)

</div>

<!-- RELATED:END -->
