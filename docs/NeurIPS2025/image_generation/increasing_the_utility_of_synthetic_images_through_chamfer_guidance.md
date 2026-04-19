---
title: >-
  [论文解读] Increasing the Utility of Synthetic Images through Chamfer Guidance
description: >-
  [NeurIPS 2025][图像生成][Chamfer Distance] 提出 Chamfer Guidance——一种免训练的推理时引导方法，利用少量真实样本作为参照，通过 Chamfer 距离同时优化合成图像的质量（fidelity）和多样性（diversity），在 ImageNet-1k 上仅用 32 张真实图片即可达到 97.5% Precision 和 92.7% Coverage，并在下游分类器训练中带来最高 16% 的准确率提升。
tags:
  - NeurIPS 2025
  - 图像生成
  - Chamfer Distance
  - 扩散模型引导
  - 合成训练数据
  - 图像多样性
  - 分布匹配
  - 免训练推理引导
---

# Increasing the Utility of Synthetic Images through Chamfer Guidance

**会议**: NeurIPS 2025  
**arXiv**: [2508.10631](https://arxiv.org/abs/2508.10631)  
**领域**: 图像生成 / 合成数据  
**关键词**: Chamfer Distance, 扩散模型引导, 合成训练数据, 图像多样性, 分布匹配, 免训练推理引导  

## 一句话总结

提出 Chamfer Guidance——一种免训练的推理时引导方法，利用少量真实样本作为参照，通过 Chamfer 距离同时优化合成图像的质量（fidelity）和多样性（diversity），在 ImageNet-1k 上仅用 32 张真实图片即可达到 97.5% Precision 和 92.7% Coverage，并在下游分类器训练中带来最高 16% 的准确率提升。

## 研究背景与动机

### 合成数据的质量-多样性困境

条件图像生成模型（如 Stable Diffusion）已能生成极度逼真的图像，自然被寄予厚望用作下游模型的合成训练数据。然而研究发现，随着模型规模增大、人类偏好对齐增强，**图像质量提升的同时多样性显著下降**，这严重限制了合成数据的实际效用。

### 现有方法的不足

**基于训练的方法**（微调、ReFL 等）：需要额外训练成本，且实验表明增加真实样本数量并不能持续提升性能

**无参考引导方法**（CFG、APG、CADS 等）：仅在采样过程中调整质量/多样性权衡，不参考目标数据分布，多样性提升有限且可能产生"无根据的多样性"

**c-VSG**：虽然引入少量真实图片做参考，但基于 Vendi Score 的度量无法随参考样本数量有效扩展，且需要平衡两个超参数

### 核心洞察

作者指出关键问题在于：**多样性应该相对于目标数据分布来定义，而非绝对的变化量**。将 3D 视觉中经典的 Chamfer 距离引入图像分布匹配，天然地将质量和多样性统一到一个距离度量中——前向项鼓励覆盖所有真实模式（多样性），反向项鼓励每个生成样本忠于真实分布（质量）。

## 方法详解

### 整体框架

Chamfer Guidance 的核心思路：在扩散模型的逆向采样过程中，每隔若干步利用 Chamfer 距离对当前去噪中间结果施加梯度引导，使生成的一批图像在特征空间中与少量真实参考图像的集合尽可能匹配。

工作流程如下：

1. 预先准备 $k$ 张真实参考图像（$k$ 可低至 2），用 DINOv2 提取特征向量集合 $\mathcal{X}$
2. 扩散采样从高斯噪声 $x_T$ 开始逆向去噪
3. 每隔 $G_{\text{freq}}=5$ 步，利用 DDIM 近似获得当前步的去噪图像 $\hat{x}_{0,t}$
4. 将 $\hat{x}_{0,t}$ 同样投影到 DINOv2 特征空间得到集合 $\mathcal{Y}$
5. 计算 $\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \mathcal{Y})$ 并反向传播梯度更新 $x_t$
6. 继续正常去噪直到生成最终图像

### 关键设计：Chamfer 距离的双向匹配

Chamfer 距离由两项组成：

$$\mathcal{L}_{\text{Chamfer}}(\mathcal{X},\mathcal{Y}) = \underbrace{\frac{1}{|\mathcal{X}|}\sum_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}}\|x-y\|^2}_{\text{覆盖项（Coverage）}} + \underbrace{\frac{1}{|\mathcal{Y}|}\sum_{y\in\mathcal{Y}}\min_{x\in\mathcal{X}}\|x-y\|^2}_{\text{保真项（Fidelity）}}$$

- **覆盖项**：要求每个真实样本都有对应的生成样本与之接近 → 防止模式坍缩，鼓励多样性
- **保真项**：要求每个生成样本都与某个真实样本接近 → 防止生成离群点，保证质量

这种双向匹配的设计使得 Chamfer 距离天然地将 Precision 和 Coverage 两个目标统一起来。

### 特征空间选择

在计算 Chamfer 距离前，将图像投影到 **DINOv2 (ViT-L)** 特征空间。选择 DINOv2 的理由：

- 自监督学习产生的特征空间在语义相似度上优于 CLIP 和 Inception
- 更好地捕捉人类感知中的相似性
- 在对象结构和背景信息上取得更好的平衡

### 引导方程

将 Chamfer 距离作为引导信号嵌入扩散模型的采样过程：

$$\nabla_{x_t}\log p_\theta(x_t|c,\mathcal{X}) = \nabla_{x_t}\log p_\theta(x_t|c) - \gamma\nabla_{x_t}\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \hat{x}_{0,t})$$

- $\gamma$ 控制引导强度
- 使用 DDIM 近似得到 $\hat{x}_{0,t}$ 以避免完成全部 $T$ 步的计算开销
- 引导频率 $G_{\text{freq}}=5$，即每 5 步引导一次

### 计算效率优势

一个重要发现：对于 LDM3.5M，Chamfer Guidance 在 **不使用 CFG**（$\omega=1.0$）的情况下即可达到 SOTA——这意味着无需额外的无条件模型前向传播，实现约 **31% 的 FLOPs 减少**。

## 实验关键数据

### 主实验：ImageNet-1k 分布匹配（表 1）

| 方法 | 模型 | $k$ | $F_1$(P,C)↑ | Precision↑ | Coverage↑ | FDD↓ | FID↓ |
|------|------|-----|-------------|------------|-----------|------|------|
| CFG ($\omega$=7.5) | LDM1.5 | – | 0.709 | 0.862 | 0.603 | 248.7 | 16.1 |
| APG | LDM1.5 | – | 0.723 | 0.855 | 0.626 | 217.9 | 13.4 |
| c-VSG | LDM1.5 | 2 | 0.660 | 0.788 | 0.568 | 236.3 | 10.7 |
| Chamfer fine-tuning | LDM1.5 | 32 | 0.766 | 0.898 | 0.668 | 210.0 | 15.5 |
| **Chamfer Guidance** | **LDM1.5** | **2** | **0.886** | **0.947** | **0.833** | **156.2** | **13.7** |
| **Chamfer Guidance** | **LDM1.5** | **32** | **0.931** | **0.950** | **0.912** | **113.3** | **8.9** |
| CFG ($\omega$=2.0) | LDM3.5M | – | 0.727 | 0.872 | 0.623 | 231.9 | 15.7 |
| **Chamfer Guidance** | **LDM3.5M** | **2** | **0.912** | **0.964** | **0.864** | **134.3** | **8.9** |
| **Chamfer Guidance** | **LDM3.5M** | **32** | **0.950** | **0.975** | **0.927** | **121.4** | **9.6** |

**关键发现**：Chamfer Guidance 仅用 2 张真实图片就大幅超越所有基线方法，且性能随 $k$ 增加持续提升——这一扩展性是其他方法（包括 fine-tuning）不具备的。

### 消融实验：CFG 强度 $\omega$ 的影响（表 4, LDM1.5, $k$=32）

| $\omega$ | $F_1$(P,C)↑ | Precision↑ | Coverage↑ | FDD↓ | FID↓ |
|----------|-------------|------------|-----------|------|------|
| 1.0（无 CFG） | 0.899 | 0.923 | 0.876 | 117.8 | 9.8 |
| **2.0** | **0.931** | **0.950** | **0.912** | **113.3** | **8.9** |
| 7.5 | 0.925 | 0.957 | 0.894 | 153.1 | 14.4 |

**结论**：$\omega=2.0$ 最优；$\omega=1.0$（纯条件模型，无 CFG）已接近 SOTA，证实可省去无条件模型计算。

### 下游分类器训练（表 3）

| 真实图片数 | 引导方式 | 模型 | IN-1k Acc | IN-Sketch | IN-V2 |
|-----------|---------|------|-----------|-----------|-------|
| 0 | CFG $\omega$=2 | LDM1.5 | 47.67 | 20.49 | 40.33 |
| 0 | Chamfer $k$=32 | LDM1.5 | **54.91** | **28.08** | **46.43** |
| 0 | CFG $\omega$=2 | LDM3.5M | 37.83 | 17.60 | 34.07 |
| 0 | Chamfer $k$=32 | LDM3.5M | **53.66** | **34.44** | **45.46** |
| 32k | CFG $\omega$=2 | LDM1.5 | 59.07 | 25.04 | 49.77 |
| 32k | Chamfer $k$=32 | LDM1.5 | **63.81** | **32.34** | **53.84** |

**关键发现**：

- 纯合成数据训练：Chamfer Guidance 带来 +7~16% 的准确率提升
- 混合训练（32k 真实 + 1.3M 合成）：进一步提升至 63.81%
- OOD 泛化（ImageNet-Sketch）：LDM3.5M + Chamfer 甚至超越在完整 1.3M ImageNet 真实数据上训练的结果

### 地理多样性实验（表 2, GeoDE）

| 方法 | $k$ | Avg $F_1$↑ | Worst-Reg $F_1$↑ | Avg Coverage↑ |
|------|-----|-----------|-----------------|---------------|
| LDM1.5 baseline | – | 0.412 | 0.346 | 0.374 |
| c-VSG (CLIP) | 2 | 0.435 | 0.412 | 0.446 |
| **Chamfer (DINOv2)** | **4** | **0.500** | **0.469** | **0.459** |

Chamfer Guidance 在平均 $F_1$ 上提升约 7%，最差区域覆盖提升约 4.9%，有效缓解了地域偏见。

## 亮点与洞察

1. **概念优雅**：将 3D 点云匹配的经典工具（Chamfer 距离）迁移到图像分布匹配，双向项自然对应 Precision 和 Coverage，无需手动平衡两个目标
2. **数据效率极高**：仅 2 张真实参考图即可获得显著提升，且性能随参考数量持续扩展——这在 fine-tuning 和 c-VSG 方法中未观察到
3. **免训练、即插即用**：不修改模型权重，可直接应用于任何扩散/流匹配模型
4. **CFG-free 能力**：在 LDM3.5M 上无需 CFG 即可达到 SOTA，节省 31% 计算量
5. **推理时计算 > 训练时计算**：与 LLM 领域的 test-time compute scaling 趋势一致，推理时引导比 fine-tuning 更有效
6. **缩小模型间差距**：新旧模型（LDM1.5 vs LDM3.5M）在使用 Chamfer Guidance 后性能差距大幅缩小

## 局限性

1. **仅支持类条件生成**：目前方法设计针对 class-conditional 模型，未直接支持开放文本 text-to-image 生成
2. **批量生成假设**：Chamfer 距离需要同时处理一批生成样本，不适用于单图生成场景
3. **评估指标固有偏差**：依赖 DINOv2/Inception 等预训练特征提取器计算分布指标，这些指标本身存在偏差
4. **特征空间依赖**：方法效果与选择的投影特征空间紧密相关（DINOv2 vs CLIP 结果有明显差异）
5. **真实参考图的选择**：论文未深入讨论参考图的选择策略对结果的影响

## Training / Inference

### 训练阶段

Chamfer Guidance 是一种**完全免训练**的方法，不修改扩散模型的任何权重。作为对比，论文同时测试了两种训练方案：

- **Vanilla fine-tuning**：使用真实参考图像作为微调数据，通过标准去噪损失（Eq.1）微调 U-Net / LoRA。LDM1.5 微调整个 U-Net，LDM3.5M 使用 LoRA（rank=4，应用于 attention 的 K/Q/V/O）。学习率 $10^{-6}$，最多 5000 步，每 1000 步保存 checkpoint。
- **Chamfer fine-tuning（ReFL 风格）**：将负 Chamfer 距离作为 reward 信号，在去噪后期（$T_1=30, T_2=39$, 共 $T=40$ 步）随机选择一个 timestep 计算 reward gradient 进行微调。$\lambda=10^{-3}$。

实验结论：两种 fine-tuning 方法性能均不如推理时 Chamfer Guidance，且无法随参考样本数扩展。

### 推理阶段

具体推理流程：

1. **初始化**：从高斯噪声 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，准备 $k$ 张真实参考图的 DINOv2 特征集合 $\mathcal{X}$
2. **常规去噪**：使用 diffusers 默认 sampler，总共 40 步
3. **周期性引导**：每隔 $G_{\text{freq}}=5$ 步执行一次引导：
    - 用 DDIM 近似获得当前去噪图像 $\hat{x}_{0,t}$
    - 解码到像素空间并用 DINOv2 提取特征 $\mathcal{Y}$
    - 计算 $\nabla_{x_t}\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \mathcal{Y})$ 并更新噪声 $x_t$
4. **CFG 设置**：LDM1.5 使用 $\omega=2.0$；LDM3.5M 使用 $\omega=1.0$（无需无条件模型，节省 31% FLOPs）
5. **硬件**：$k \leq 8$ 时单张 H100 GPU；$k \in \{16, 32\}$ 时使用多 GPU

**计算开销分析**：Chamfer Guidance 的额外开销来自 DINOv2 前向传播 + Chamfer 距离梯度反传，但对于 LDM3.5M 可省去 unconditional model 的前向传播，总体 FLOPs 反而更低。

## 相关工作与启发

- **与 c-VSG 的对比**：c-VSG 用 Vendi Score 度量多样性，需要存储中间结果并平衡两个目标项（ungrounded diversity + grounded diversity 两个超参）；Chamfer Guidance 将 Precision 和 Coverage 统一在一个公式里，更简洁、易调参，且可随参考样本数扩展
- **与 ReFL (Reward Feedback Learning) 的关系**：Chamfer 距离可作为 reward 信号用于 fine-tuning（Chamfer fine-tuning），但实验表明推理时引导效果显著优于训练时引导——这与 LLM 领域 test-time compute scaling 趋势一致
- **与 ICL 的类比**：方法类似于 LLM 中的 few-shot in-context learning，用少量示例在推理时适配，无需修改模型。关键区别：c-VSG 也尝试了类似的 few-shot 引导但无法随样本数扩展
- **Reference-free 方法的局限**：APG、CADS、Limited Interval、Particle Guidance 等方法在 ImageNet-1k 上仅在 base LDM 的 Coverage 基础上提升 1-3%，因为它们定义的多样性是绝对变化量而非相对于目标分布的覆盖
- **未来方向**：(1) 基于检索的 text-to-image 扩展管线——用大规模文本-图像数据库建立离线检索索引，推理时 top-k 检索作为参考；(2) 零样本自举管线——先生成候选集，自动选择使特征空间"覆盖直径"最大的子集作为引导参考

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | Chamfer 距离用于扩散引导是新视角，但整体为已有组件的巧妙组合 |
| 理论深度 | ⭐⭐⭐ | 方法直观但缺乏收敛性等理论分析 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | 多模型、多数据集、下游任务、消融实验非常全面 |
| 工程实用性 | ⭐⭐⭐⭐ | 免训练即插即用，但目前限于类条件生成 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图表丰富，动机论述充分 |
| **综合** | **⭐⭐⭐⭐** | 实用性强、实验扎实的工作，将经典距离度量优雅引入生成引导 |

## My Notes

- 这篇工作的核心贡献在于**发现 Chamfer 距离作为引导信号的可扩展性（scalability）**：c-VSG 和 fine-tuning 方法在增加参考样本后性能饱和甚至下降，而 Chamfer Guidance 持续提升。这说明度量函数的选择比引导框架本身更关键。
- **"推理时计算 > 训练时计算"** 的结论值得关注：同样的 Chamfer 距离作为目标函数，用于推理引导（Chamfer Guidance）比用于训练（Chamfer fine-tuning/ReFL）效果好得多。可能原因是 fine-tuning 容易过拟合少量参考图导致多样性下降，而推理引导每次作用于新的噪声样本。
- **特征空间选择的微妙影响**：DINOv2 在 object-centric 任务上优于 CLIP，但在 GeoDE 地理多样性任务中 CLIP 的 CLIPScore 更高（因为引导天然对齐 CLIP 子空间）。这暗示**不同下游任务可能需要不同的投影空间**。
- 方法的一个隐含限制是**batch size 与参考集大小的关系**：Chamfer 距离在两个集合大小相差很大时可能退化（覆盖项主导或保真项主导），论文未讨论生成 batch size 的选择策略。
- **可扩展到其它生成任务**：Chamfer 距离引导可以直接迁移到视频生成（帧级特征集合匹配）、3D 生成（本就是 Chamfer 距离的发源地）等领域，是一个有潜力的通用框架。
- **与 data-free distillation 的联系**：在没有真实数据的场景下，可以通过自举（self-bootstrapping）构造参考集，这为合成数据迭代式自我提升提供了新思路。

<!-- RELATED:START -->

## 相关论文

- [Enhancing Diffusion Model Guidance through Calibration and Regularization](enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)
- [Privacy Amplification Through Synthetic Data: Insights from Linear Regression](../../ICML2025/image_generation/privacy_amplification_through_synthetic_data_insights_from_linear_regression.md)
- [UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [Synthetic Perception: Can Generated Images Unlock Latent Visual Prior for Text-Centric Reasoning?](../../ICML2025/image_generation/synthetic_perception_can_generated_images_unlock_latent_visual_prior_for_text-ce.md)
- [Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)

<!-- RELATED:END -->
