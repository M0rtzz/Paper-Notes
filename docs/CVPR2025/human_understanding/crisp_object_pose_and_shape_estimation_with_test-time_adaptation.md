---
title: >-
  [论文解读] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation
description: >-
  [人体理解] 提出 CRISP，一个类别无关的物体姿态与形状估计 pipeline，核心创新在于基于 active shape model 的优化校正器和 correct-and-certify 自训练策略，可在测试时自适应弥合大的域差距。
tags:
  - 人体理解
---

# CRISP: Object Pose and Shape Estimation with Test-Time Adaptation

| 属性 | 值 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2412.01052](https://arxiv.org/abs/2412.01052) |
| 代码 | [项目页面](https://web.mit.edu/sparklab/research/crisp_object_pose_shape/) |
| 领域 | 人体理解 / 3D 物体感知 |
| 关键词 | pose estimation, shape reconstruction, SDF, test-time adaptation, self-training, corrector |

## 一句话总结

提出 CRISP，一个类别无关的物体姿态与形状估计 pipeline，核心创新在于基于 active shape model 的优化校正器和 correct-and-certify 自训练策略，可在测试时自适应弥合大的域差距。

## 研究背景与动机

### 领域现状

物体姿态和形状估计是增强现实、机器人和太空自主对接等应用的关键能力。现有方法大致分为：实例级（需要 CAD 模型）、类别级（需要类别先验）、以及近年兴起的类别无关方法。条件扩散模型在 3D 重建上展现潜力，但推理速度过慢。

### 现有痛点

1. **缺乏泛化能力**：类别级方法需要特定类别先验，难以扩展到新类别
2. **域差距问题严重**：训练数据（尤其是合成数据）与真实测试环境分布差异大，导致估计结果不可用甚至在安全关键场景中造成危险
3. **自训练不稳定**：已有自监督方法（如 Chamfer loss）在自训练中需要合成数据稳定训练，不适合真实部署

### 本文目标

1. 设计一个**类别无关**的姿态与形状估计 pipeline
2. 提供**优化校正器**来修正神经网络估计误差
3. 实现**无需合成数据**的测试时自训练，弥合 sim-to-real 域差距

### 切入角度与核心 idea

将形状解码器近似为 active shape model（已知形状的凸组合），将形状校正问题转化为约束线性最小二乘问题，从而高效求解；然后用 correct-and-certify 范式生成伪标签进行自训练。

## 方法详解

### 整体框架

CRISP pipeline 包含三个层次：(1) 编码器-解码器形状估计（DINOv2 backbone + FiLM-conditioned SDF 解码器）；(2) DPT 网络估计 pose-normalized coordinates (PNC) 用于姿态估计；(3) 优化校正器 + 可观测正确性证书 + 自训练。

### 关键设计 1：FiLM-conditioned 形状估计

- **功能**：从单张 RGB-D 图像估计物体的隐式形状表示（SDF）
- **核心思路**：编码器用预训练的 DINOv2 ViT 提取图像特征，经 MLP 回归潜在形状码 $\mathbf{h}$；解码器用正弦激活 MLP + FiLM 条件化生成 SDF
- **设计动机**：FiLM 条件化比拼接条件化产生更好的隐式场重建效果，且无需类别标签。类别无关设计使得 pipeline 可以用任意 CAD 模型集合训练
- **姿态估计**：DPT 网络直接回归每个像素的 pose-normalized coordinates (PNC)，再用 Arun 方法求解 SE(3) 姿态。注意不做 scale normalization，避免自训练中 PNC 退化

### 关键设计 2：优化校正器与 Active Shape Decoder

- **功能**：修正网络估计的姿态和形状误差
- **核心思路**：建立双层优化问题，先用梯度下降校正 PNC（固定形状码），再用投影梯度下降校正形状码（投影到训练形状的单纯形 $S_K$ 上）。关键创新是构造 active shape decoder：$f_a(\mathbf{z}|\mathbf{c}) = c_0 d_0 f_d(\mathbf{z}|\mathbf{h}) + \sum_{k=1}^{K} c_k d_k f_d(\mathbf{z}|\mathbf{h}_k)$，将形状校正转化为约束线性最小二乘问题 $\min_{\mathbf{c} \geq 0, \mathbf{1}^T\mathbf{c}=1} \|\mathbf{F}(\mathbf{Z})\mathbf{D}\mathbf{c}\|^2$
- **设计动机**：训练好的形状解码器在潜在空间插值时行为良好，但外推时产生不合理形状（可视化验证）。因此约束形状码在已知形状的凸包内，并利用线性结构实现高效内点法求解
- **多视角扩展**：聚合多视角的 PNC 校正结果，用于更精准的形状估计

### 关键设计 3：Correct-and-Certify 自训练 (CRISP-ST)

- **功能**：无需合成数据的测试时域适应
- **核心思路**：三步流程——(1) 用校正器修正网络输出；(2) 用可观测正确性证书 $\text{oc}(\hat{\mathbf{Z}}, \hat{\mathbf{h}}) = \mathbb{I}\{[|f_d(\hat{\mathbf{z}}_i|\hat{\mathbf{h}})|]_p < \epsilon\}$ 检验修正结果质量；(3) 通过质量检验的修正结果作为伪标签训练网络
- **设计动机**：标准自训练容易因错误伪标签而崩溃。正确性证书通过验证深度点与隐式形状的几何一致性来过滤不可靠估计。随训练推进，越来越多估计通过检验，形成良性循环
- **损失函数**：$L_h = \|\hat{\mathbf{h}} - \mathbf{h}\|^2$, $L_z = \sum_i \|\hat{\mathbf{z}}_i - \mathbf{z}_i\|^2$，形状解码器冻结

## 实验关键数据

### YCBV 数据集

**形状估计 ($e_{shape}$ ↓)**：

| 方法 | Mean | Median | AUC@3cm | AUC@5cm |
|------|------|--------|---------|---------|
| Shap-E | 0.099 | 0.052 | 0.05 | 0.17 |
| CRISP-Syn | 0.045 | 0.032 | 0.18 | 0.35 |
| CRISP-Syn-ST (LSQ) | **0.037** | **0.024** | **0.25** | **0.43** |
| CRISP-Real | 0.026 | 0.016 | 0.40 | 0.58 |

**姿态估计 (ADD-S)**：自训练后 CRISP-Syn-ST 显著缩小了与 CRISP-Real 的差距

### 关键发现

- 自训练（仅 5 epochs）即可将 sim-to-real gap 大幅缩小：$e_{shape}$ 从 0.045 降至 0.037
- LSQ 求解器通常比 BCD 求解器表现更好（得益于线性最小二乘的高效精确求解）
- Active shape decoder 中包含网络估计的 $\mathbf{h}$ 作为"基底"是关键，去掉后性能下降明显
- 支持 SPE3R（卫星）和 NOCS（生活物品）多种场景，验证了类别无关的泛化性

### 消融实验

- 单视角 vs 多视角校正器：多视角显著提升形状估计质量
- 形状退化检测：$\mathbf{F}(\mathbf{Z})^T\mathbf{F}(\mathbf{Z})$ 最小特征值可作为形状可辨识性指标

## 亮点与洞察

1. **Active Shape Decoder 理论优雅**：将非凸优化转化为约束线性最小二乘，有闭式解，兼具理论保证和实用效率
2. **Correct-and-certify 范式完整**：校正→认证→自训练的三步闭环，无需合成数据辅助
3. **形状解码器凸包行为的发现**：外推不可靠、插值可靠这一观察，启发了投影到单纯形的策略
4. **类别无关设计**：无需类别标签，真正可扩展到任意物体

## 局限性

1. 校正器依赖良好的初始化，网络估计太差时校正器可能收敛到局部最优
2. Active shape decoder 的基底数 K 受限于训练集 CAD 模型数量，对于高度多样化的测试物体可能表达力不足
3. 自训练需要一定数量的观测才能稳定，在极少样本场景可能效果有限
4. 实时推理时校正器的额外计算开销需要考虑

## 相关工作与启发

- **NOCS**（CVPR 2019）：开创性的类别级姿态形状估计，CRISP 的 PNC 与之类似但不做 scale normalization
- **RePoNet**：利用可微渲染训练，CRISP 用 correct-and-certify 作为替代
- **Talak et al. 的 correct-and-certify**：CRISP-ST 扩展了这一范式到同时估计姿态和形状
- **启发**：test-time adaptation 思路可推广到其他 3D 感知任务（如手部姿态、人体姿态估计），关键是设计好校正器和质量证书

## 评分

⭐⭐⭐⭐ — 理论与系统设计均扎实，active shape model 的线性化思想优雅，自训练策略实用。形状解码器凸包行为的 insight 特别有启发性

<!-- RELATED:START -->

## 相关论文

- [RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](../../ICCV2025/human_understanding/wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)

<!-- RELATED:END -->
