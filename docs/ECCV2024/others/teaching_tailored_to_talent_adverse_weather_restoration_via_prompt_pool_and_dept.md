---
title: >-
  [论文解读] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint
description: >-
  [ECCV 2024][Adverse Weather Restoration] 提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。
tags:
  - ECCV 2024
  - Adverse Weather Restoration
  - 提示学习
  - Depth-Anything
  - 扩散模型
  - All-in-One Restoration
---

# Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint

**会议**: ECCV 2024  
**arXiv**: [2409.15739](https://arxiv.org/abs/2409.15739)  
**代码**: https://github.com/Ephemeral182/T3-DiffWeather (有)  
**领域**: 图像恢复 / 恶劣天气去除  
**关键词**: Adverse Weather Restoration, Prompt Pool, Depth-Anything, diffusion model, All-in-One Restoration

## 一句话总结

提出 T3-DiffWeather，一种基于 diffusion 的 all-in-one 恶劣天气恢复框架，通过 prompt pool 让网络自主组合 sub-prompts 构建实例级 weather-prompts 来建模多样化天气退化，同时利用 Depth-Anything 特征约束 general prompts 来建模场景信息，仅需 2 步采样即达到 SOTA，计算量仅为 WeatherDiffusion 的 1/52。

## 研究背景与动机

恶劣天气图像恢复（去雾、去雨、去雪、去雨滴）是图像恢复的重要方向。现实世界中天气退化具有**不可预测性和组合性**——同一场景可能同时存在雾霾、雨滴、雪花等多种退化，且每种退化的程度和形态各异。

**现有痛点**：

**早期 all-in-one 方法**（TransWeather、All-in-One）使用共享 query 或 NAS，无法显式建模不同天气退化之间的相似性和差异性，难以自适应处理未知天气组合

**WeatherDiffusion** 首次将 diffusion 引入天气恢复并取得 SOTA，但直接用退化图像作为 condition 信息不够丰富，且需要 25 步采样，推理慢

**PromptIR** 用共享可学习 prompt 适配不同退化，但共享参数导致不同退化间的干扰，忽略了实例级退化差异

**缺乏场景建模**：现有方法只关注退化理解，忽略了退化图像背后的场景信息对重建的指导作用

**核心矛盾**：如何在一个统一模型中既灵活建模多样化天气退化的差异与共性，又充分利用场景先验信息来指导 diffusion 去噪？

**核心 idea**："因材施教"（Teaching Tailored to Talent）——用 prompt pool 让网络自主选择 sub-prompts 构建针对特定退化的 weather-prompts，用 Depth-Anything 约束的 general prompts 提供场景条件，两类 prompt 作为 condition 指导 diffusion 重建退化残差。

## 方法详解

### 整体框架

T3-DiffWeather 的 pipeline：输入退化图像 $\mathbf{y}$ → 计算退化残差 $\mathbf{r}_d = \mathbf{x} - \mathbf{y}$（训练时）→ 将残差作为 diffusion 的重建目标 → 使用两类 prompts 作为 condition 通过 cross-attention 注入 diffusion 网络的 latent layer → DDIM 2 步采样得到预测残差 $\mathbf{r}_d^{sample}$ → 恢复图像 $\hat{\mathbf{x}} = \mathbf{r}_d^{sample} + \mathbf{y}$。

关键创新：**重建退化残差而非干净图像**，因为 t-SNE 可视化显示退化残差比背景更具有判别性，且退化是恢复的主要难点。

### 关键设计

1. **Prompt Pool 构建 Weather-Prompts**:

    - 功能：设计一个包含 $N=20$ 个 sub-prompts 的 prompt pool $\mathcal{P} = \{\mathcal{P}_s^i\}_{i=1}^N$，每个 sub-prompt $\mathcal{P}_s^i \in \mathbb{R}^{L_s \times D}$（$L_s=64$ tokens）。网络根据输入退化图像自主选择 top-$k$（$k=5$）个最相关的 sub-prompts，拼接构成 weather-prompts $\mathcal{P}_w$
    - 核心思路：每个 sub-prompt 配有一个可学习的 key $\mathcal{K}_s^i \in \mathbb{R}^{1 \times D}$。将退化残差 embedding $\mathcal{F}_e$ 做 spatial mean pooling 得到 $\mathcal{F}_e^{mean} \in \mathbb{R}^{1 \times D}$，计算与各 key 的 cosine similarity $\delta(\mathcal{K}_s^i, \mathcal{F}_e)$，选择 top-$k$ 相似的 sub-prompts 拼接：$\mathcal{P}_w = \bigcup_{i=1}^k \mathcal{K}_s^i$，其中 $\delta(\mathcal{K}_s^i, \mathcal{F}_e) \geq \delta(\mathcal{K}_s^{i+1}, \mathcal{F}_e)$
    - 设计动机：不同天气退化之间既有**共性**（如雾霾遮蔽、对比度降低）又有**差异**（如雨条纹 vs. 雪花颗粒）。Prompt pool 允许网络通过共享某些 sub-prompts 来捕获共性（如雨和雨滴的某些 sub-prompt 选择频率相似），通过独立 sub-prompts 来建模差异
    - t-SNE 可视化验证：不同天气条件下构建的 weather-prompts 既保持了各自的聚类特性，又体现了跨天气的重叠

2. **Depth-Anything 约束的 General Prompts**:

    - 功能：设计一组 general prompts $\mathcal{P}_g \in \mathbb{R}^{L_g \times D}$（$L_g=256$ tokens），通过 cross-attention 与 Depth-Anything 的中间层特征 $\mathcal{F}_d$ 交互，得到场景感知的 $\mathcal{P}_{gd} = \text{softmax}(\frac{\mathcal{Q}_g \mathcal{K}_d^T}{\sqrt{\mathcal{D}}}) \mathcal{V}_d$
    - 核心思路：t-SNE 显示清洁图像的场景特征在潜在空间中共享共性（与退化特征截然不同）。Depth-Anything 在极端退化图像上的深度估计依然稳健，其中间层特征具有**退化不变性**，能可靠地表征场景结构
    - 设计动机：对比了 DINO、DINOv2 和 Depth-Anything 的中间特征鲁棒性，Depth-Anything 最优（因为其在 DINOv2 基础上用大规模数据训练了深度估计，获得了对退化的额外鲁棒性）
    - 架构选择：使用 Depth-Anything ViT-S（仅 115 MB），在性能和内存间取最佳平衡

3. **两阶段 Cross-Attention 注入**:

    - 功能：将 weather-prompts 和 general prompts 分两次通过 cross-attention 注入 diffusion 网络的 latent feature
    - 公式：$\mathcal{F}_e' = \text{CA}(\mathcal{F}_e, \mathcal{P}_w)$，$\hat{\mathcal{F}_e} = \text{CA}(\mathcal{F}_e', \mathcal{P}_{gd})$
    - 设计动机：类似 Stable Diffusion 中 text embedding 的注入方式，先处理退化信息再融入场景信息，自然且高效

4. **对比 Prompt Loss (Contrastive Prompt Loss)**:

    - 功能：约束两类 prompts 的表征——weather-prompts 的 key 应远离 general prompts 的 key（互为 negative），同时 general prompts 的 key 应靠近 Depth-Anything 特征（positive）
    - 公式：$\mathcal{L}_{cp} = \frac{1}{b} \frac{1}{k} \sum_{j=1}^b \sum_{i=1}^k [\gamma(\mathcal{K}_{gd}, \mathcal{F}_d^{mean}) - \gamma(\mathcal{K}_s^i, \mathcal{K}_{gd})]$，其中 $\gamma(\cdot) = 1 - \delta(\cdot)$
    - 设计动机：两类 prompts 有本质不同的设计目标（退化建模 vs. 场景建模），天然互为 negative pair，无需额外构造负样本

5. **退化残差重建目标**:

    - 功能：diffusion 的重建目标从干净图像 $\mathbf{x}$ 改为退化残差 $\mathbf{r}_d = \mathbf{x} - \mathbf{y}$
    - 训练目标：$\mathcal{L}_{res} = \mathbb{E}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}}(\mathbf{x}-\mathbf{y}) + \sqrt{1-\bar{\alpha}}\boldsymbol{\epsilon}, \mathbf{y}, \mathbf{c})\|_2^2$
    - 设计动机：残差比完整图像更容易学习，因为退化区域的信息密度更高；heatmap 可视化证实模型确实关注了退化区域

### 损失函数 / 训练策略

总损失函数包含四项：

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{res} + \lambda_2 \mathcal{L}_{cp} + \lambda_3 \|(\mathbf{r}_d^{sample} + \mathbf{y}) - \mathbf{x}\|_{psnr} + \lambda_4 \mathcal{L}_{cp}^{sample}$$

- $\mathcal{L}_{res}$：噪声估计损失（残差重建）
- $\mathcal{L}_{cp}$：对比 prompt 损失（噪声估计阶段）
- PSNR 重建损失：对采样结果的显式监督
- $\mathcal{L}_{cp}^{sample}$：采样阶段的对比 prompt 损失
- 所有 $\lambda$ 均设为 1

训练配置：AllWeather 数据集（18,069 图），Adam 优化器，初始学习率 1.5e-4，cosine annealing，800K 迭代，2×A800 GPU，DDIM 采样 1000 timesteps 训练 / 2 步推理。

## 实验关键数据

### 主实验

**去雪（Snow100K）：**

| 方法 | Snow100K-S PSNR/SSIM | Snow100K-L PSNR/SSIM |
|------|---------------------|---------------------|
| Restormer | 35.03/0.9487 | 30.52/0.9092 |
| WeatherDiff64 | 35.83/0.9566 | 30.09/0.9041 |
| AWRCP | 36.92/0.9652 | 31.92/0.9341 |
| **T3-DiffWeather** | **37.51/0.9664** | **32.37/0.9355** |

**去雨（Outdoor-Rain）：**

| 方法 | PSNR | SSIM |
|------|------|------|
| Restormer | 29.97 | 0.9215 |
| WeatherDiff64 | 29.64 | 0.9312 |
| AWRCP | 31.39 | 0.9329 |
| **T3-DiffWeather** | **31.99** | **0.9365** |

**去雨滴（RainDrop）：**

| 方法 | PSNR | SSIM |
|------|------|------|
| UDR-S2Former | 32.64 | 0.9427 |
| AWRCP | 31.93 | 0.9314 |
| **T3-DiffWeather** | **32.66** | **0.9411** |

**计算效率对比（256×256 分辨率）：**

| 方法 | 参数量 | GFLOPs |
|------|--------|--------|
| WeatherDiffusion | 113.68M | 248.4G × 25 steps |
| Refusion | 131.4M | 63.4G × 50 steps |
| **T3-DiffWeather** | **69.38M** | **59.82G × 2 steps** |

### 消融实验

**Prompt Pool 消融（Outdoor-Rain）：**

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| 无 prompt pool | 31.05 | 0.9325 | baseline |
| 无 matched keys | 31.72 | 0.9349 | 无选择机制 |
| PromptIR 式共享 prompt | 31.38 | 0.9330 | 共享参数干扰 |
| **Prompt pool (ours)** | **31.99** | **0.9365** | 自主选择最优 |

**General Prompts 约束消融：**

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| 无 General Prompts | 31.52 | 0.9342 | 缺少场景信息 |
| 无 Depth-Anything 约束 | 31.67 | 0.9349 | 无显式约束 |
| DINO 约束 | 31.77 | 0.9357 | 鲁棒性不足 |
| DINOv2 约束 | 31.82 | 0.9359 | 次优 |
| **Depth-Anything 约束** | **31.99** | **0.9365** | 最佳鲁棒性 |

**对比 Prompt Loss 消融：**

| 配置 | PSNR | SSIM |
|------|------|------|
| 无 CPL | 31.71 | 0.9350 |
| 无 Negative $\gamma$ | 31.81 | 0.9359 |
| 无 Positive $\gamma$ | 31.77 | 0.9358 |
| **完整 CPL** | **31.99** | **0.9365** |

### 关键发现

- **Prompt pool 贡献最大**：相比无 prompt pool 提升 0.94 dB PSNR（31.05→31.99），且优于 PromptIR 的共享 prompt 方式
- **Depth-Anything > DINOv2 > DINO**：场景约束的选择很重要，Depth-Anything 的退化不变性最强
- **2 步采样即够**：由于条件信息足够丰富且重建目标是残差（比完整图像简单），仅需 2 步 DDIM 采样
- **Pool size 和 top-k 敏感性**：pool size 20、top-k 5 为最佳平衡点，过大的 pool 引入冗余，过大的 k 导致过拟合
- **ViT-S 足够**：Depth-Anything ViT-S（115MB）vs ViT-L（1314MB），PSNR 仅差 0.07 dB

## 亮点与洞察

- **"因材施教"的 prompt 设计哲学**：weather-prompts 和 general prompts 分别建模退化和场景，职责清晰，通过对比 loss 保证两者不混淆
- **重建残差而非干净图像**：这个看似简单的改变让 diffusion 仅需 2 步采样，计算量降为 SOTA 的 1/52，非常 practical
- **借力 Depth-Anything 的退化不变特征**：首次将深度估计模型的中间特征用于图像恢复任务的场景约束，思路新颖
- **Sub-prompt 选择频率的可视化**解释性很好——可以看出不同天气共享了哪些 attributes（如雨和雨滴选择频率相似），哪些是独特的
- **可迁移思路**：prompt pool + key matching 的机制可以推广到其他需要处理多种退化/任务的 all-in-one 模型

## 局限与展望

- **训练时需要 paired data**：依赖 clean-degraded 图像对，限制了在真实场景的训练
- **Depth-Anything 作为额外依赖**：虽然用的是 ViT-S，仍增加了模型复杂度和推理开销
- **仅在天气退化上验证**：未扩展到其他类型的图像退化（如噪声、模糊、压缩伪影）
- **prompt pool size 固定**：未探索动态增长的 prompt pool
- **real-world 评测有限**：real-world 数据集上的 PSNR 提升相对较小（~0.6 dB）

## 相关工作与启发

- **vs WeatherDiffusion**: 同为 diffusion-based，但 WeatherDiffusion 用退化图像作 condition 信息不够丰富，需 25 步采样。T3-DiffWeather 通过丰富的 prompt condition 和残差重建目标，仅需 2 步即超越，PSNR 提升 1.68 dB（Snow100K-S）
- **vs PromptIR**: PromptIR 用共享 prompt 适配不同退化，无法区分退化间的差异。T3-DiffWeather 的 prompt pool 自主选择机制更灵活，PSNR 高 0.61 dB
- **vs AWRCP**: AWRCP 是 ICCV 2023 SOTA，T3-DiffWeather 在所有 benchmark 上均超越（Snow100K-S +0.59 dB，Outdoor-Rain +0.60 dB）
- **vs TransWeather**: TransWeather 用固定 weather-type query，无法自适应组合。T3-DiffWeather 的 prompt pool 设计理念更先进

## 评分

- 新颖性: ⭐⭐⭐⭐ Prompt pool + Depth-Anything constraint 的组合设计新颖，残差重建目标简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 四个 synthetic benchmark + 两个 real-world 数据集，详细消融（prompt pool/general prompts/CPL/pool size/top-k/DA架构），计算效率对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富（t-SNE、sub-prompt 选择频率、heatmap），motivation 和 Discussion 写得好
- 价值: ⭐⭐⭐⭐⭐ 1/52 的计算量达到 SOTA，对 diffusion-based restoration 的实用化有重要推动；prompt pool 设计思路可广泛迁移

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)
- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[ECCV 2024\] Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)

<!-- RELATED:END -->
