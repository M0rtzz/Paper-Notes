---
title: >-
  [论文解读] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures
description: >-
  [3D视觉] TeHOR 利用文本描述作为语义引导，通过预训练扩散模型的 Score Distillation Sampling 联合优化 3D 人体和物体的几何与纹理，突破了传统方法对接触信息的依赖，实现了包括非接触交互在内的准确且语义一致的 3D 重建。
tags:
  - 3D视觉
---

# TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures

## 基本信息

- **会议**: CVPR 2026
- **arXiv**: [2602.19679](https://arxiv.org/abs/2602.19679)
- **代码**: [项目主页](https://hygenie1228.github.io/TeHOR/)
- **领域**: 3D视觉 / 人体-物体重建
- **关键词**: 3D Human-Object Reconstruction, Text-Guided Optimization, Score Distillation Sampling, 3D Gaussian Splatting, Human-Object Interaction

## 一句话总结

TeHOR 利用文本描述作为语义引导，通过预训练扩散模型的 Score Distillation Sampling 联合优化 3D 人体和物体的几何与纹理，突破了传统方法对接触信息的依赖，实现了包括非接触交互在内的准确且语义一致的 3D 重建。

## 研究背景与动机

从单张图像联合重建 3D 人体和物体是人体行为理解的关键任务，在机器人、AR/VR 和数字内容创作中有广泛应用。现有方法存在两个根本性局限：

**过度依赖接触信息**：现有方法（如 PHOSA、CONTHO、InteractVLM）主要利用人体-物体接触区域作为交互推理的核心线索，通过迭代拟合强制接触区域的几何邻近。然而，现实世界中大量交互是非接触的（如注视、指向物体），接触信息完全失效。即使存在接触，错误的接触预测也会直接导致重建失败。

**忽视全局外观上下文**：现有方法的拟合过程主要靠局部几何近邻驱动，忽略了人体和物体的外观线索（颜色、阴影等）所提供的全局交互上下文，导致全局不合理的结果（如物体朝向错误、人体视线不对齐）。

## 方法详解

### 整体框架

TeHOR 采用两阶段框架：**重建阶段**（初始化）和 **HOI 优化阶段**（联合精调）。

| 阶段 | 目标 | 关键技术 |
|------|------|----------|
| 重建阶段 | 获取初始 3D 人物/物体/背景和文本提示 | GPT-4 文本生成、LHM 人体重建、InstantMesh 物体重建 |
| HOI 优化阶段 | 联合优化几何与纹理（200步迭代） | SDS 外观损失、接触损失、碰撞损失 |

**阶段一：重建阶段**

- **文本生成**：使用 GPT-4 从输入图像提取两种文本提示——$P_{\text{holistic}}$（全局交互描述，如"一个人在草地上骑自行车"）和 $P_{\text{contact}}$（接触的身体部位，如"右手, 左手"）
- **人体重建**：SmartEraser 去除物体 → SAM 分割人体 → LHM 生成初始 3D Gaussian 属性 $\phi_h$（40,000 个锚点均匀采样在 SMPL-X 表面）→ Multi-HMR 估计 SMPL-X 姿态 $\theta$ 和体型 $\beta$
- **物体重建**：SmartEraser + SAM 分离物体 → InstantMesh 重建 3D mesh（先用 Zero123++ 生成 6 视角图像，再通过三平面网络重建）→ 转换为 3D Gaussian 属性 $\phi_o$ → ZoeDepth 深度对齐估计物体位姿 $(R, t, s)$
- **背景重建**：SmartEraser 去除人体和物体，得到 2D 背景图，用于构建逼真的前视图和新视角渲染

### 3D 表示

人体和物体分别用 3D Gaussian 集合 $\Phi_h$ 和 $\Phi_o$ 表示：

- **人体 Gaussians**：参数化为 Gaussian 属性 $\phi_h$ + SMPL-X 姿态 $\theta$ + 体型 $\beta$。$\phi_h$ 在标准姿态下定义，每个 Gaussian 锚定到 SMPL-X mesh 表面点，通过 Linear Blend Skinning (LBS) 驱动动画。手部和面部沿用原始 SMPL-X 蒙皮权重，其余部位采用邻近顶点的平均权重
- **物体 Gaussians**：参数化为 Gaussian 属性 $\phi_o$ + 旋转 $R$ + 平移 $t$ + 缩放 $s$，在标准空间定义后通过仿射变换得到最终位置

选择 3D Gaussians 而非传统 mesh 的优势：(1) 高斯能更好建模高保真视觉外观，为外观损失提供更丰富信号；(2) 灵活的拓扑无关结构允许更有效地优化人物-物体空间关系。

### 核心损失函数设计

总损失函数由四项组成：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{appr}} + \mathcal{L}_{\text{contact}} + \mathcal{L}_{\text{collision}}$$

**1) 重建损失 $\mathcal{L}_{\text{recon}}$**：前视角渲染与输入图像间的 MSE，包括 RGB 图像重建误差和人体/物体轮廓与分割 mask 的误差，确保重建结果在输入视角下与原图一致。

**2) 外观损失 $\mathcal{L}_{\text{appr}}$（核心创新）**：基于 Score Distillation Sampling (SDS) 策略，利用预训练 StableDiffusion-v2.1 的视觉先验，将新视角渲染与 $P_{\text{holistic}}$ 语义对齐：

$$\nabla_{\Phi}\mathcal{L}_{\text{appr}} = \mathbb{E}\left[w_t\left(\hat{\epsilon}_t(\mathbf{x}_t; P_{\text{holistic}}) - \epsilon_t\right)\frac{\partial \mathbf{x}_t}{\partial \Phi}\right]$$

其中 $t$ 为噪声级别，$\mathbf{x}_t$ 为加噪后的渲染图像，$w_t$ 为权重因子。该损失最小化扩散模型预测噪声 $\hat{\epsilon}_t(\cdot)$ 与真实噪声 $\epsilon_t$ 的差距，驱使 3D Gaussian 的渲染结果向文本条件下的合理外观分布靠拢。

关键实现细节：

- 在球面坐标 $(r, \upsilon, \psi)$ 均匀采样视角：全身视角 $r \in [1.0, 2.5]$，$\upsilon \in [-30°, 30°]$，$\psi \in [-180°, 180°]$；上半身放大视角以 SMPL-X 脊柱为球心，$r \in [0.7, 1.5]$
- 分类器无关引导 (CFG) 尺度 15.0，噪声时间步在 $[0.02, 0.98]$ 内随机采样
- 梯度裁剪最大范数 1.0

该设计的两大优势：(a) 文本描述超越接触信息，能推理非接触交互（如接飞盘、注视物体）；(b) 像素级密集梯度提供细粒度空间监督，远优于 CLIP 的单向量全局编码。

**3) 接触损失 $\mathcal{L}_{\text{contact}}$**：根据 $P_{\text{contact}}$ 确定接触身体部位对应的 Gaussian 中心点集 $V_{h,c}$，最小化其与最近物体点 $V_o$ 的距离：

$$\mathcal{L}_{\text{contact}} = \frac{1}{|V_{h,c}|}\sum_{v_h \in V_{h,c}} d(v_h, V_o) \cdot \mathbb{1}[d(v_h, V_o) < \tau]$$

阈值 $\tau = 10$ cm，保证局部物理合理性。仅对距离小于阈值的点计算梯度，避免将远处无关点强行拉近。

**4) 碰撞损失 $\mathcal{L}_{\text{collision}}$**：惩罚人体和物体之间的穿模（interpenetration），计算人体顶点在物体 mesh 内部的比例，确保物理合理性。

### Gaussians-to-Mesh 转换

优化完成后需将 3D Gaussians 转换为 mesh 用于评估（与现有 mesh-based 方法公平比较）。由于 Gaussians 偏离底层 base mesh，接触区域可能出现不一致。解决方案：识别人体-物体 Gaussian 距离 < 5 cm 的接触区域，选取对应 mesh 顶点并最小化其间距至零，实现接触一致的转换。

## 实验

### 数据集与指标

- **Open3DHOI**：开放词汇野外 3D HOI 数据集，2.5K+ 图像，133 类物体（仅评估用）
- **BEHAVE**：室内 3D HOI 数据集，8 名受试者 × 20 个物体，测试集 4.5K 图像
- **指标**：$\text{CD}_{\text{human}}$ / $\text{CD}_{\text{object}}$（Chamfer 距离, cm↓）、Contact（F1↑）、Collision（穿模率↓）

### 主实验：与 SOTA 比较（Tab. 4）

| 方法 | CD↓_human (O3D) | CD↓_obj (O3D) | Contact↑ (O3D) | Coll.↓ (O3D) | CD↓_human (BH) | CD↓_obj (BH) | Contact↑ (BH) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PHOSA | 5.342 | 49.180 | 0.243 | 0.044 | 5.758 | 46.003 | 0.257 |
| LEMON+PICO | 5.948 | 25.889 | 0.335 | 0.078 | 6.159 | 22.585 | 0.082 |
| InteractVLM | 5.252 | 24.238 | 0.392 | 0.054 | 5.770 | 19.197 | 0.379 |
| HOI-Gaussian | 5.111 | 19.363 | 0.348 | 0.070 | 5.748 | 21.774 | 0.371 |
| **TeHOR** | **4.941** | **16.701** | **0.412** | **0.047** | **5.615** | **17.339** | **0.412** |

全面超越所有 SOTA。Open3DHOI 上物体 CD 从 19.363→16.701（↓13.7%），Contact F1 从 0.392→0.412。

### 非接触场景评估（Tab. 5）

| 方法 | CD↓_human | CD↓_object | Collision↓ |
|------|:---:|:---:|:---:|
| PHOSA | 5.401 | 65.537 | 0.028 |
| InteractVLM | 5.390 | 46.819 | 0.011 |
| HOI-Gaussian | 5.244 | 25.374 | 0.037 |
| **TeHOR** | **4.958** | **17.546** | **0.005** |

非接触场景优势更显著，物体 CD 从 25.374→17.546（↓30.8%），验证文本语义引导的关键作用。

### 消融实验

**文本引导优化效果（Tab. 1）**：

| 设置 | CD↓_human | CD↓_obj | Contact↑ | Collision↓ |
|------|:---:|:---:|:---:|:---:|
| 优化前 | 5.252 | 31.268 | 0.305 | 0.040 |
| 优化（无文本） | 5.028 | 20.348 | 0.374 | 0.052 |
| **优化（完整）** | **4.941** | **16.701** | **0.412** | **0.047** |

**损失函数配置消融（Tab. 2）**：

| $\mathcal{L}_{\text{appr}}$ | $\mathcal{L}_{\text{contact}}$ | CD↓_obj | Contact↑ |
|:---:|:---:|:---:|:---:|
| ✗ | ✓ | 22.094 | 0.330 |
| ✓ | ✗ | 19.849 | 0.374 |
| CLIP 替代 | ✓ | 18.504 | 0.366 |
| **✓ (SDS)** | **✓** | **16.701** | **0.412** |

关键发现：SDS 外观损失显著优于 CLIP loss——CLIP 编码为单一 1D 向量无法建模密集空间关系，SDS 提供像素级密集梯度。

**渲染组件消融（Tab. 3）**：3D Gaussians→Mesh 使 CD_obj 恶化至 25.162；去除 2D 背景使 CD_obj 恶化至 18.196，说明完整场景上下文对扩散先验至关重要。

## 亮点

- **突破接触依赖范式**：首次将文本描述引入 3D 人体-物体联合重建，支持非接触交互推理（注视、指向、接飞盘等）
- **SDS 外观优化**：利用预训练扩散模型视觉先验，通过多视角 SDS 实现细粒度语义对齐，消融验证远优于 CLIP
- **首个纹理联合重建**：据称首个同时重建人体和物体完整 3D 纹理的框架，可直接生成沉浸式数字资产
- **实验设计完善**：一般场景与非接触场景分别评估，5 组消融实验充分验证各组件有效性

## 局限性

- 依赖 GPT-4、StableDiffusion、LHM、InstantMesh 等多个外部模型，依赖链长且推理成本高
- 每样本约 134 秒（单张 RTX 8000），200 步优化使实时应用困难
- 外观损失主要提供全局引导，对局部细节（小配件、微妙表面变形）监督不足
- 缺乏纹理质量的量化评估指标（无同时标注几何+纹理的 3D HOI 数据集）

## 评分

⭐⭐⭐⭐ — 清晰识别现有方法的根本局限（接触依赖+忽视全局外观），提出的文本引导 SDS 优化方案新颖且有效。在一般和非接触场景均全面 SOTA，消融实验设计系统完善。主要扣分在优化效率和对多个外部模型的长依赖链。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](../../ICCV2025/3d_vision/placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](../../ICCV2025/3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/3d_vision/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

<!-- RELATED:END -->
