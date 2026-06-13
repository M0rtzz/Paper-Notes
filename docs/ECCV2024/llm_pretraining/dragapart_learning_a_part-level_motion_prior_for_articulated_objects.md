---
title: >-
  [论文解读] DragAPart: Learning a Part-Level Motion Prior for Articulated Objects
description: >-
  [ECCV 2024][预训练][部件级运动先验] DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。
tags:
  - "ECCV 2024"
  - "预训练"
  - "部件级运动先验"
  - "拖拽控制"
  - "铰接物体"
  - "扩散模型"
  - "域随机化"
---

# DragAPart: Learning a Part-Level Motion Prior for Articulated Objects

**会议**: ECCV 2024  
**arXiv**: [2403.15382](https://arxiv.org/abs/2403.15382)  
**代码**: 有 (dragapart.github.io)  
**领域**: LLM预训练  
**关键词**: 部件级运动先验, 拖拽控制, 铰接物体, 扩散模型, 域随机化

## 一句话总结

DragAPart 提出了一种以拖拽为交互接口的图像生成器，能够响应部件级别的交互（如开关抽屉/门），而非仅仅移动整个物体。通过新的合成数据集 Drag-a-Move、多分辨率拖拽编码和域随机化策略，模型在仅用合成数据训练的情况下能良好泛化到真实图像和未见类别。

## 研究背景与动机

### 为什么需要部件级运动先验

现有的可变形物体建模通常依赖特定物体类型的参数化模板（如 SMPL 用于人体、SMAL 用于哺乳动物），缺乏通用性。一个理想的**通用运动基础模型**应该能够理解任何物体的运动——无论是人、水母、毛巾还是家具。然而，不存在一个能统一表达所有物体姿态的通用模板。

**核心假设**：运动模型不需要参考模板，只需要理解物体的可能物理构型及其转换。**拖拽**提供了一种无需模板即可探测运动先验的方式——指定单个物理点如何移动，让模型"填补"其余部件的合理运动。

### 现有拖拽方法的失败

现有的拖拽控制图像/视频生成方法（DragGAN、DragDiffusion、DragonDiffusion、DragNUWA 等）虽然在物体重新定位上表现不错，但在**部件级运动**上完全失败：
- 它们倾向于**整体移动物体**（如平移/缩放），而非打开抽屉或旋转门
- 原因：大规模预训练数据不可避免地纠缠了相机视角变化和多物体交互，难以分辨不同部件的独立运动

### 拖拽编码的根本问题

先前方法用稀疏光流图表示拖拽，存在两个致命缺陷：

**终点定位模糊**：Transformer 可以通过空间位置感知拖拽起点 $u$，但终点 $v$ 只编码为起点位置的位移向量，空间感知弱

**多拖拽冲突**：所有拖拽编码在同一张光流图中，降分辨率后不同拖拽可能重叠，导致不同部件的运动信号互相干扰

## 方法详解

### 整体框架

DragAPart 基于预训练的 Stable Diffusion (SD) 进行微调：
- **输入**：一张物体中心的 RGB 图像 $y$ + 一组拖拽 $\mathcal{D}$
- **输出**：反映拖拽效果的新图像 $x \sim \mathbb{P}(x|y, \mathcal{D})$
- **训练数据**：合成数据集 Drag-a-Move（基于 GAPartNet 的 3D 铰接物体）
- **架构**：在 SD U-Net 的每个 Transformer block 注入拖拽编码

条件注入方式：参考图像 $y$ 通过相同网络 $\Phi$ 的无噪声前向传播，将其 keys/values 替代生成过程中的 self-attention keys/values，实现参考-生成间的 cross attention。

训练目标为标准去噪损失：

$$\min_\Phi \mathbb{E}_{(x,y,\mathcal{D}),t,\epsilon \sim \mathcal{N}(0,1)} \left[\|\epsilon - \Phi(z_t, t, y, \mathcal{D})\|_2^2\right]$$

### 关键设计

#### 1. **多分辨率拖拽编码 (Multi-Resolution Drag Encoding)**

核心创新。与先前方法不同，本文为每个拖拽分配**独立通道**，且**分别编码起点和终点**：

对于单个拖拽 $d = (u, v)$，在 LDM 的第 $l$ 个 block 对应分辨率 $(h_l, w_l)$ 下，编码为 $F_l(u,v) \in \mathbb{R}^{2 \times h_l \times w_l}$：
- 全填 $-1$（无拖拽标记），仅在起点 $u$ 对应的 latent 像素位置记录子像素偏移
- 终点 $v$ 同样独立编码为 $F_l(v,u)$
- 最终拖拽编码：$F_l(\mathcal{D}) = \bigoplus_{(u,v) \in \mathcal{D}} F_l(u,v) \oplus F_l(v,u) \in \mathbb{R}^{4N \times h \times w}$

**设计优势**：
- 每个拖拽占用独立通道 → 不同部件的运动信号**不会重叠**
- 起点和终点**分别空间编码** → Transformer 能精确感知两端位置
- 在每个 block 以**对应分辨率**注入 → 替代了先前的高斯模糊 + 卷积降采样
- 不足 $N$ 个拖拽时用 $-1$ 填充

#### 2. **域随机化 (Domain Randomization)**

GAPartNet 中的合成物体纹理不具真实感且缺乏变化。直接训练会导致模型"作弊"——在各处生成相同颜色像素，而非真正理解像素的来源。

解决方案：训练时**同时**使用原始纹理渲染和**随机单色纹理**渲染（每个部件被随机着色）。

**设计动机**：随机纹理迫使模型不能依赖颜色一致性来简单复制像素，必须隐式理解 3D 几何结构和部件对应关系。消融实验证实，域随机化在 OOD 数据上将 PSNR 从 18.03 提升至 19.74，泛化能力显著增强。

#### 3. **Drag-a-Move 数据集**

基于 GAPartNet 构建，包含 763 个 3D 模型、16 个类别（有"Hinge Handle"、"Slider Drawer"、"Hinge Lid"、"Hinge Door"标注的物体）。

数据生成流程：
- 每个物体渲染 48 个铰接动画，每动画 36 帧 → 共 4000 万唯一图像对
- 两种动画模式：①部分关节锁定（闭合态），剩余关节在极端状态间过渡；②所有关节随机状态
- 拖拽生成：从运动关节的子树中采样一个部件，按位移概率采样表面点，投影到像素空间
- 同时渲染随机纹理版本用于域随机化

### 下游应用

#### 运动分析
给定 3D 网格 $\mathcal{M} = \mathcal{M}_\text{static} \cup \mathcal{M}_\text{moving}$ 和 3D 拖拽，通过最小化多视角渲染与 DragAPart 生成图像的差异来估计运动参数（旋转/平移型关节、旋转轴/平移方向）：

$$\arg\min_{t, p_{\text{motion}}} \frac{1}{K} \sum_{k=1}^{K} \|R(\mathcal{M}_\text{static} \cup T(\mathcal{M}_\text{moving}; t, p_{\text{motion}}), C_k) - x_k\|_2^2$$

使用网格搜索而非梯度优化避免局部最优。

#### 可动部件分割
提取去噪器 $\Phi$ 的内部特征：对同一图像分别执行有/无拖拽条件的前向传播，计算特征差异，该差异代表拖拽在扩散模型内部表示中的效果，可用于粗分割响应拖拽的可动部件。

## 实验关键数据

### 主实验

Drag-a-Move 和 Human3.6M 上的定量对比：

| 方法 | Drag-a-Move PSNR↑ | SSIM↑ | LPIPS↓ | Human3.6M PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|------------------|-------|--------|
| iPoke | 16.79 | 0.883 | 0.150 | 21.43 | 0.856 | 0.258 |
| DragDiffusion | 15.30 | 0.773 | 0.226 | 18.06 | 0.707 | 0.280 |
| DragonDiffusion | 17.63 | 0.852 | 0.162 | 19.45 | 0.765 | 0.258 |
| DragNUWA | 13.58 | 0.765 | 0.277 | 15.16 | 0.668 | 0.292 |
| **DragAPart (本文)** | **21.38** | **0.925** | **0.066** | **23.82** | **0.870** | **0.091** |

DragAPart 在所有指标上大幅领先：PSNR +3.75 (vs DragonDiffusion)，LPIPS 降低 59% (-0.096)。

### 消融实验

**架构与拖拽编码消融**：

| 编码方式 | DiT PSNR↑ | SSIM↑ | LPIPS↓ | SD PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-----------|-------|--------|----------|-------|--------|
| 卷积仅输入层 | 19.56 | 0.910 | 0.095 | 19.97 | 0.914 | 0.077 |
| 卷积每层注入 | 20.58 | 0.922 | 0.078 | 21.10 | 0.925 | 0.067 |
| **多分辨率编码每层注入** | **21.11** | **0.925** | **0.074** | **21.38** | **0.925** | **0.066** |

**域随机化消融**：

| 训练数据 | I.D. PSNR↑ | SSIM↑ | LPIPS↓ | O.O.D. PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-----------|-------|--------|-------------|-------|--------|
| 无域随机化 | 21.38 | 0.925 | 0.066 | 18.03 | 0.897 | 0.104 |
| **有域随机化** | **21.82** | **0.928** | 0.066 | **19.74** | **0.920** | **0.083** |

域随机化在 I.D. 上也有轻微提升 (+0.44 PSNR)，而 O.O.D. 提升显著 (+1.71 PSNR)。

### 关键发现

1. **多分辨率编码 > 卷积编码**：在两种架构（SD 和 DiT）上均一致提升，证明通道分离 + 空间精确编码比模糊 + 卷积更有效
2. **每层注入远优于仅输入层注入**：PSNR 差距超过 1.4 (SD)，多层注入让运动信息在各分辨率精确传播
3. **预训练 U-Net (SD) 略优于从头训练的 DiT**：得益于亿级图像预训练的泛化先验
4. **域随机化是泛化到真实数据的关键**：没有它的模型在真实图像和未见类别上严重退化
5. **定性观察**：先前拖拽方法倾向于复制 handle 点外观到目标点或整体缩放/平移物体，而 DragAPart 产生物理合理的部件级运动

## 亮点与洞察

1. **拖拽作为运动探针**：将拖拽重新定义为探测通用运动先验的接口，而非简单的图像编辑工具。单个拖拽即可隐式激活整个运动学链的合理响应
2. **少量合成数据 + 预训练模型的威力**：仅 763 个合成 3D 物体（远少于 NVS 的百万级物体），配合预训练 SD 的视觉先验，即可泛化到真实图像和未见类别
3. **域随机化的简单有效**：随机着色这一极简策略竟能显著桥接合成-真实域差距，迫使模型学习结构而非纹理
4. **运动分析的新范式**：不直接预测运动参数，而是通过生成多视角拖拽结果再优化 3D 运动参数，巧妙利用了生成模型的视角一致性

## 局限与展望

1. **跨视角不一致**：未显式约束同一物体在不同视角/拖拽条件下的生成一致性
2. **人体与物体分开训练**：目前在日常物体和人体上训练独立模型，缺乏统一的通用运动先验
3. **合成数据规模有限**：仅 763 个物体、16 个类别，部分小众类别的泛化可能不足
4. **拖拽上限 $N$ 固定**：最大拖拽数 $N$ 预设，超过时无法处理
5. **仅生成静态图像**：未扩展到视频生成，无法展示连续运动轨迹

## 相关工作与启发

- **ControlNet** 式条件注入思路被创造性地应用于拖拽编码
- GAPartNet 的部件标注为学习铰接物体运动提供了关键数据基础
- 域随机化策略源自 sim-to-real 迁移文献（Tobin et al.），在扩散模型微调中同样有效
- 可动部件分割利用扩散模型内部特征差异，启发了"差分特征"做语义分析的新思路

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将拖拽重新定义为部件级运动探针的问题设定非常新颖，多分辨率编码设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 定量和定性实验充分，消融有力，但真实数据评估仍以定性为主
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机阐述清晰，可视化精美，论证逻辑严密
- **实用价值**: ⭐⭐⭐⭐ — 运动分析和部件分割展示了应用前景，但部件类型覆盖有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](../../ICML2025/llm_pretraining/position_the_future_of_bayesian_prediction_is_prior-fitted.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)

</div>

<!-- RELATED:END -->
