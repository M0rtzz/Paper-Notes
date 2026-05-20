---
title: >-
  [论文解读] FE2E: From Editor to Dense Geometry Estimator
description: >-
  [CVPR 2026][3D视觉][深度估计] 本文系统分析了图像编辑模型与生成模型在稠密几何估计任务中的微调行为差异，发现编辑模型具有天然的结构先验优势，并基于此提出 FE2E 框架，首次将 DiT 架构的图像编辑模型适配为深度和法线联合估计器…
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "深度估计"
  - "法线估计"
  - "图像编辑模型"
  - "扩散模型"
  - "DiT"
---

# FE2E: From Editor to Dense Geometry Estimator

**会议**: CVPR 2026  
**arXiv**: [2509.04338](https://arxiv.org/abs/2509.04338)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 深度估计, 法线估计, 图像编辑模型, 扩散模型, DiT

## 一句话总结
本文系统分析了图像编辑模型与生成模型在稠密几何估计任务中的微调行为差异，发现编辑模型具有天然的结构先验优势，并基于此提出 FE2E 框架，首次将 DiT 架构的图像编辑模型适配为深度和法线联合估计器，在零样本场景下大幅超越现有 SOTA（ETH3D 上 AbsRel 降低 35%）。

## 研究背景与动机

1. **领域现状**：单目稠密几何估计（深度和法线）是 3D 视觉的核心任务。近年来，以 Marigold 为代表的方法借助 Stable Diffusion 的预训练生成先验，在有限数据下取得了令人印象深刻的零样本预测效果。另一类以 DepthAnything 系列为代表的方法则走数据驱动路线，使用大规模数据（62.6M图像）训练通用估计器。

2. **现有痛点**：生成模型（text-to-image）本身设计目标是从文本生成图像，其内部特征并不天然对齐几何结构。微调过程中需要从零开始"重塑"特征，学习过程不稳定且存在性能瓶颈。数据驱动方法虽然有效，但对大规模标注数据的依赖限制了其泛化性。

3. **核心矛盾**：稠密几何估计本质上是 image-to-image 任务，但现有方法却基于 T2I 生成模型做微调——这是任务范式与模型范式之间的不匹配。

4. **本文目标**（1）验证图像编辑模型是否比生成模型更适合做稠密几何估计；（2）解决编辑模型适配为确定性预测器时遇到的训练目标、数值精度和计算效率问题。

5. **切入角度**：作者从一个直觉出发——图像编辑模型天然理解输入图像结构，同时保持了生成模型的能力，理应比 T2I 模型更适合做 dense prediction。通过系统的特征演化分析和训练动态对比验证了这一假设。

6. **核心 idea**：用图像编辑模型替代生成模型作为稠密几何估计的基础模型，并通过一致速度流匹配、对数量化和零成本联合估计三个技术适配编辑器为估计器。

## 方法详解

### 整体框架
FE2E 基于 Step1X-Edit（一个基于 DiT 架构的 SOTA 图像编辑模型）。输入是 RGB 图像，输出是对应的深度图和法线图。整个流程为：VAE 编码器将输入图像和几何标注编码到潜空间，DiT 学习从固定起点到目标潜表示的恒速直线路径，最后 VAE 解码器将预测结果解码回像素空间。关键创新在于利用 DiT 编辑模型的并行输出区域同时预测深度和法线，无需额外计算开销。

### 关键设计

1. **编辑模型 vs. 生成模型的系统分析**

    - 功能：验证编辑模型作为稠密估计基础模型的优越性
    - 核心思路：对 Step1X-Edit（编辑器）和 FLUX（生成器）在相同设置下微调，通过可视化 DiT 不同层（Block1/20/35）的特征演化和训练损失曲线进行对比。发现编辑模型的初始特征就已经对齐图像几何结构，微调过程只是"精炼"和"聚焦"已有能力；而生成模型的特征从混沌状态开始"重塑"，训练震荡且存在约 0.08 的损失瓶颈
    - 设计动机：为"From Editor to Estimator"范式提供理论基础，解释为什么编辑模型能用更少数据达到更好效果

2. **一致速度流匹配（Consistent Velocity Flow Matching）**

    - 功能：将编辑模型的随机性流匹配目标改造为适合确定性预测的训练目标
    - 核心思路：原始流匹配中，模型学习所有可能路径上的瞬时速度场，导致全局速度场非线性、积分路径弯曲，离散求解器引入累计误差。FE2E 做了两步简化：（1）要求速度方向和大小在整条路径上保持一致，使训练目标 $\mathcal{L} = \mathbb{E}[\|\mathbf{v} - f_\theta(\mathbf{z}^x)\|^2]$ 完全不依赖时间步 $t$；（2）将随机高斯起点固定为零向量 $\mathbf{z}_0^y = \mathbf{0}$，消除采样随机性。推理时直接 $\mathbf{z}_1^y = f_\theta(\mathbf{z}^x)$，一步完成，无需迭代求解
    - 设计动机：几何估计是确定性任务（只有唯一 GT），不需要生成模型的多样性。恒速直线路径从根本上消除了弯曲轨迹的离散化误差，同时大幅提升推理速度

3. **对数标注量化（Logarithmic Annotation Quantization）**

    - 功能：解决 BF16 精度训练与深度估计高精度需求之间的矛盾
    - 核心思路：现代编辑模型均以 BF16 精度训练，对 RGB 输出足够（1/256 精度），但直接用于深度标注会产生严重量化误差。例如在 Virtual KITTI 数据集（0-80m 深度范围）中，均匀量化到 [-1,1] 后在 0.1m 处的 AbsRel 误差高达 1.6。逆深度量化（disparity）在近处精度极高但远处完全不可用（39m 和 78m 映射为同一值）。FE2E 采用对数量化 $D_{log} = \ln(D_{GT} + 1e{-6})$，再配合百分位归一化 $\mathbf{y}_D = \langle((D_{log} - D_{log,2})/(D_{log,98} - D_{log,2}) - 0.5) \times 2\rangle$，在远近距离都能保持均衡的低误差（AbsRel 约 0.013）
    - 设计动机：让 BF16-only 模型也能做高精度深度估计，避免之前方法强制使用 FP32 导致的成本增加和先验继承不良

### 损失函数 / 训练策略
- 主损失为恒速流匹配损失，在潜空间计算
- 联合估计时，左侧输出区域监督深度，右侧监督法线：$\mathcal{L}_{fm} = \mathbb{E}(\|\mathbf{v}_D - p_l\|^2 + \|\mathbf{v}_N - p_r\|^2)$
- 额外使用辅助分散损失(dispersion loss)鼓励不同样本的特征在隐空间中分散
- 训练使用 LoRA（rank=64, α=32），冻结 DiT 以外的所有参数，AdamW 优化器，学习率 1e-4，训练 30 epoch
- 在单张 RTX 4090 上即可训练（启用梯度检查点），使用 H20 GPU 约 1.5 天完成

## 实验关键数据

### 主实验

| 数据集 | 指标(AbsRel↓) | FE2E | Lotus-D (之前SOTA) | DepthAnything V2 | 提升 |
|--------|------|------|----------|------|------|
| NYUv2 | AbsRel | **4.1** | 5.1 | 4.5 | 19.6% vs Lotus-D |
| KITTI | AbsRel | **6.6** | 8.1 | 7.4 | 18.5% |
| ETH3D | AbsRel | **3.8** | 6.1 | 13.1 | **37.7%** |
| ScanNet | AbsRel | **4.4** | 5.5 | - | 20.0% |
| DIODE | AbsRel | **22.8** | 22.8 | 26.5 | 持平 |

训练数据仅 71K 图像，远少于 DepthAnything V2 的 62.6M。

### 消融实验

| ID | 配置 | KITTI AbsRel | ETH3D AbsRel | 说明 |
|------|---------|------|------|------|
| 2 | DirectAdapt (Step1X-Edit) | 9.5 | 5.6 | 基线 |
| 3 | + Consistent Velocity | 8.8 | 5.0 | CV贡献 -7%/-10% |
| 4 | + Fixed Start | 8.6 | 4.8 | FS进一步提升 |
| 6 | + Logarithmic Quant | **6.8** | **3.9** | 对数量化贡献 -19%/-13% |
| 7 | FLUX+改进方法 | 7.1 | 4.5 | 编辑模型仍优于生成模型 |
| 8 | FE2E完整 (+联合估计) | **6.6** | **3.8** | 联合估计带来额外增益 |
| 9 | FLUX-Kontext+完整方法 | 6.7 | 3.6 | 验证可扩展到其他编辑模型 |

### 关键发现
- 对数量化是贡献最大的单项改进，在 KITTI 上单独贡献 19% 的误差降低
- 编辑模型 vs 生成模型的差距在所有设置下都很一致（ID2 vs ID1, ID6 vs ID7）
- 联合估计不仅零成本，还能在困难场景（蝴蝶平面结构、远景建筑）带来可视的质量提升
- 方法可扩展到 FLUX-Kontext 等其他编辑器（ID9），甚至表现更好

## 亮点与洞察
- **"From Editor to Estimator"范式的系统论证**：不只是换个模型，而是通过特征演化可视化和训练动态分析，从机理层面解释了为什么编辑模型更适合做稠密估计。这一发现可能启发更多 I2I 任务利用编辑模型先验
- **零成本联合估计的巧妙设计**：DiT 编辑模型的输入拼接机制导致 50% 的输出被丢弃，FE2E 利用这部分"废弃"输出做第二个任务的预测，零额外计算成本实现多任务学习
- **BF16 精度下的对数量化方案**：完美解决了"模型只有 BF16 权重但任务需要高数值精度"这一实际工程难题，方案通用性强

## 局限与展望
- 训练数据局限于合成数据（Hypersim + Virtual KITTI），未使用真实世界数据
- 法线估计性能提升相比深度估计没有那么显著
- 仅支持仿射不变的深度预测，未扩展到度量深度
- 对数量化方案虽然均衡，但在极近和极远距离上都不是最优的，可以考虑分段自适应量化

## 相关工作与启发
- **vs Marigold/Lotus**：它们基于 Stable Diffusion (T2I)，FE2E 基于 Step1X-Edit (编辑模型)，在相似数据量下 FE2E 大幅领先，证明了基础模型选择比算法改进更关键
- **vs DepthAnything V2**：后者使用 100 倍数据量，FE2E 仍在 ETH3D 等数据集上显著领先，说明编辑先验可以弥补数据量不足
- **vs Diffusion-E2E-FT**：E2E-FT 提出端到端微调去噪架构，FE2E 进一步推进到编辑模型+恒速流匹配，是这一路线的延续

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统论证并实现"编辑模型→估计器"的范式转换
- 实验充分度: ⭐⭐⭐⭐⭐ 5个深度基准+4个法线基准+详尽消融，证据链完整
- 写作质量: ⭐⭐⭐⭐ 图表可视化出色，分析有深度
- 价值: ⭐⭐⭐⭐⭐ 为稠密预测任务的基础模型选择提供了新范式和理论支撑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[AAAI 2026\] Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine](../../AAAI2026/3d_vision/free-form_scene_editor_enabling_multi-round_object_manipulation_like_in_a_3d_eng.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[ECCV 2024\] Diffusion Model is a Good Pose Estimator from 3D RF-Vision](../../ECCV2024/3d_vision/diffusion_model_is_a_good_pose_estimator_from_3d_rf-vision.md)

</div>

<!-- RELATED:END -->
