---
title: >-
  [论文解读] Identity-preserving Distillation Sampling by Fixed-Point Iterator
description: >-
  [CVPR 2025][3D视觉][score distillation] 提出 Identity-preserving Distillation Sampling (IDS)，通过不动点迭代正则化（FPR）修正文本条件分数函数中导致身份丢失的梯度误差，生成引导噪声替代随机噪声，在 2D 图像编辑和 3D NeRF 编辑中实现结构和姿态的高度保持。
tags:
  - CVPR 2025
  - 3D视觉
  - score distillation
  - image editing
  - NeRF editing
  - identity preservation
  - fixed-point iteration
  - DDS
---

# Identity-preserving Distillation Sampling by Fixed-Point Iterator

**会议**: CVPR 2025  
**arXiv**: [2502.19930](https://arxiv.org/abs/2502.19930)  
**代码**: [https://github.com/shhh0620/IDS](https://github.com/shhh0620/IDS)  
**领域**: 3d_vision / image_generation  
**关键词**: score distillation, image editing, NeRF editing, identity preservation, fixed-point iteration, DDS

## 一句话总结

提出 Identity-preserving Distillation Sampling (IDS)，通过不动点迭代正则化（FPR）修正文本条件分数函数中导致身份丢失的梯度误差，生成引导噪声替代随机噪声，在 2D 图像编辑和 3D NeRF 编辑中实现结构和姿态的高度保持。

## 研究背景与动机

**领域现状**: Score Distillation Sampling (SDS) 通过蒸馏预训练扩散模型的分数函数，实现文本驱动的 3D 生成和图像编辑。Delta Denoising Score (DDS) 通过减去源-目标分数差来缓解 SDS 的模糊问题。

**现有痛点**: (1) SDS 的随机噪声导致梯度方向不稳定，结果过度饱和和模糊；(2) DDS 虽然能减少非文本对齐特征的噪声，但文本条件分数 $\epsilon_\phi^{src}$ 本身并不精确指向源图像，导致累积误差丢失源的身份信息（如背景、姿态、结构）；(3) CDS 和 PDS 试图最大化互信息来保持一致性，但未分析分数本身的固有误差。

**核心矛盾**: 文本条件分数 $\epsilon_\phi^{src}$ 被期望提供从噪声潜变量到源图像的梯度方向，但实际上文本提示可以对应无数不同图像，因此分数指向的后验均值 $\mathbf{z}_{0|t}^{src}$ 与源图像 $\mathbf{z}^{src}$ 存在显著差异（特别在 $t$ 较大时）。这种错误累积导致编辑结果的结构性变化。

**本文切入角度**: 从数值分析中的不动点迭代出发，通过迭代修正源潜变量使分数函数精确对齐到源图像，从根本上解决梯度误差问题。

## 方法详解

### 整体框架

1. 采样随机噪声 $\epsilon$ 和时间步 $t$，构建源潜变量 $\mathbf{z}_t^{src}$
2. 运行 FPR: 迭代更新 $\mathbf{z}_t^{src}$ 使后验均值 $\mathbf{z}_{0|t}^{src}$ 逼近源图像 $\mathbf{z}^{src}$
3. 从优化后的 $\mathbf{z}_t^{src*}$ 提取引导噪声 $\epsilon^*$
4. 用 $\epsilon^*$ 替代随机噪声构建目标潜变量 $\mathbf{z}_t^{trg*}$，计算 IDS 更新方向

### 关键设计

**1. 不动点迭代正则化（FPR）**
- **功能**: 通过迭代更新源潜变量 $\mathbf{z}_t^{src}$，使得由 Tweedie 公式计算的后验均值与源图像一致。
- **核心公式**:
    - 后验均值: $\mathbf{z}_{0|t}^{src} = \frac{1}{\sqrt{\alpha_t}}(\mathbf{z}_t^{src} - \sqrt{1-\alpha_t} \epsilon_\phi^{src})$
    - FPR 损失: $\mathcal{L}_{FPR} = d(\mathbf{z}^{src}, \mathbf{z}_{0|t}^{src})$（欧氏距离）
    - 更新: $\mathbf{z}_t^{src} \leftarrow \mathbf{z}_t^{src} - \lambda \nabla_{\mathbf{z}_t^{src}} \mathcal{L}_{FPR}$
- **迭代 N 次**: 每次更新后重新计算 CFG 分数 $\epsilon_\phi^{src}$ 和后验均值
- **设计动机**: 如果分数正确估计为指向 $\mathbf{z}^{src}$ 的梯度，后验均值应包含源图像的充分信息。通过最小化两者差异来"修正"分数。
- **为什么更新潜变量而非噪声**: 实验发现更新 $\mathbf{z}_t^{src}$ 能保留更多内容细节（因为分数以潜变量为输入）。

**2. 引导噪声替代随机噪声**
- **功能**: FPR 收敛后，从优化的 $\mathbf{z}_t^{src*}$ 反解引导噪声 $\epsilon^* = \frac{1}{\sqrt{1-\alpha_t}}(\mathbf{z}_t^{src*} - \sqrt{\alpha_t}\mathbf{z}^{src})$
- **核心思路**: 引导噪声 $\epsilon^*$ 不再是随机高斯噪声，而是"对齐到源图像身份的结构化噪声"；用它来生成目标潜变量 $\mathbf{z}_t^{trg*}$，使梯度方向包含身份一致性约束。
- **设计动机**: DDS 中源和目标共享同一随机 $\epsilon$，但 $\epsilon$ 是无约束的，可能指向任意方向；$\epsilon^*$ 经过 FPR 修正后隐含了源图像的身份信息。

**3. IDS 更新规则**
- **功能**: 用修正后的潜变量替代 DDS 中的原始潜变量:
  $$\nabla_\theta \mathcal{L}_{IDS} = \mathbb{E}_{t,\epsilon}[(\epsilon_\phi^\omega(\mathbf{z}_t^{trg*}, y^{trg}, t) - \epsilon_\phi^\omega(\mathbf{z}_t^{src*}, y^{src}, t)) \frac{\partial \mathbf{z}^{trg}}{\partial \theta}]$$
- **可逆性验证**: IDS 编辑后的图像可以通过逆向编辑完美恢复到源图像（DDS 无法做到），证明梯度方向被正确修正。

### 损失函数

- FPR: $\mathcal{L}_{FPR} = \|\mathbf{z}^{src} - \mathbf{z}_{0|t}^{src}\|_2^2$
- 编辑: 使用 IDS 梯度更新目标图像/NeRF 参数
- 超参: $\lambda$ 控制 FPR 正则化强度，N 控制迭代次数（通常 N=3）

## 实验关键数据

### 主实验 — 图像编辑

**结构保持指标（LPIPS↓越低越好）**:

| 方法 | cat→pig LPIPS↓ | cat→squirrel LPIPS↓ | IP2P LPIPS↓ | IP2P PSNR↑ |
|---|---|---|---|---|
| P2P | 0.42 | 0.46 | 0.47 | 20.88 |
| PnP | 0.52 | 0.52 | 0.39 | 23.81 |
| DDS | 0.28 | 0.30 | 0.24 | 26.02 |
| CDS | 0.25 | 0.26 | 0.21 | 27.35 |
| **IDS** | **0.22** | **0.24** | **0.19** | **29.25** |

**用户偏好与 GPT 评分**:

| 方法 | 文本对齐↑ | 身份保持↑ | 质量↑ | GPT-Text↑ | GPT-Preserve↑ |
|---|---|---|---|---|---|
| DDS | 20.30% | 10.82% | 16.23% | 7.60 | 7.51 |
| CDS | 17.02% | 16.72% | 17.08% | 8.26 | 8.00 |
| **IDS** | **43.83%** | **60.49%** | **51.67%** | **8.97** | **9.00** |

### NeRF 编辑

| 方法 | CLIP↑ | 文本偏好↑ | 保持偏好↑ | 质量偏好↑ |
|---|---|---|---|---|
| DDS | 0.1596 | 36.88% | 28.37% | 32.62% |
| CDS | 0.1597 | 22.70% | 23.40% | 21.28% |
| **IDS** | **0.1626** | **40.42%** | **48.23%** | **46.10%** |

### 消融实验 — 计算复杂度

| 设置 | LPIPS↓ | CLIP↑ | Time(s/img) | Memory(GB) |
|---|---|---|---|---|
| DDS (200步) | 0.240 | 0.293 | 22.45 | 6.27 |
| CDS (200步) | 0.210 | 0.287 | 59.31 | 8.83 |
| IDS (200步, FPR=3) | 0.190 | 0.277 | 107.77 | 8.63 |
| IDS (100步, FPR=3) | 0.165 | 0.265 | 54.04 | 8.63 |

### 关键发现

1. **身份保持大幅提升**: IDS 在用户偏好的"身份保持"维度获得 60.49%（vs DDS 10.82%），改进极其显著。
2. **可逆性**: IDS 编辑→逆编辑可以几乎完美恢复源图像，DDS 无法做到，证明梯度方向被正确修正。
3. **FPR 迭代次数的权衡**: 仅需 N=3 次迭代即可获得显著效果，增加迭代次数收益递减但计算成本线性增加。
4. **对 NeRF 编辑同样有效**: 3D 场景编辑对一致性要求更高，IDS 的结构保持优势在 3D 中更加明显（深度图也更干净）。
5. **收敛性更好**: IDS 100 步即可超越 DDS 200 步的效果，总计算量低于 CDS。

## 亮点与洞察

- 从数值分析视角（不动点迭代）切入解决扩散模型编辑中的身份保持问题，理论基础扎实
- 可逆性测试是非常漂亮的验证方式：如果编辑→逆编辑能完美恢复，说明梯度方向无误
- 引导噪声替代随机噪声的设计直觉清晰：不是添加更多约束，而是从根本上修正误差来源
- 对文本条件分数"为什么不精确"的分析透彻：后验均值 vs 源图像的差异随时间步增大可视化地展示了问题

## 局限与展望

- FPR 的 N 次迭代导致计算开销增加（每次迭代需要额外的 diffusion model 前向传播）
- 依赖 DDS 框架，继承了其对文本提示对质量的依赖
- 仅在 Stable Diffusion v1.5 上实验，未验证对更新的模型（如 SDXL、FLUX）的适用性
- 超参 $\lambda$ 对结果敏感，过大导致过度保持（不编辑），过小则退化为 DDS
- 缺乏对大尺度编辑（如改变物体类别和场景结构）的讨论

## 相关工作与启发

- **SDS (DreamFusion)**: 分数蒸馏的开创性工作，但梯度噪声导致模糊
- **DDS**: 通过源-目标差分减少噪声，但未解决分数误差的根本问题
- **CDS/PDS**: 通过互信息最大化保持一致性，但计算开销大且效果有限
- **启发**: 扩散模型中"分数函数不精确"是一个被低估的问题，不动点迭代提供了一个通用的修正框架

## 评分

⭐⭐⭐⭐ — 理论分析深入、方法动机清晰、实验验证充分（图像+NeRF+用户研究+可逆性测试），是 SDS 系列方法的重要改进；计算开销增加是主要代价。

<!-- RELATED:START -->

## 相关论文

- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](../../ICCV2025/3d_vision/identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)
- [Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)

<!-- RELATED:END -->
