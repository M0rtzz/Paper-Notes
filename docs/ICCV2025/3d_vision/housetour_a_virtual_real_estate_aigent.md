---
title: >-
  [论文解读] HouseTour: A Virtual Real Estate A(I)gent
description: >-
  [ICCV 2025][3D视觉][Camera Trajectory Generation] 提出 HouseTour，给定一组已知位姿的室内图像，联合生成类人的3D相机轨迹和房地产文字描述，通过 Residual Diffuser 进行基于扩散的轨迹规划并将空间特征集成到 Qwen2-VL-3D 中生成3D-grounded文本摘要。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "Camera Trajectory Generation"
  - "Real Estate"
  - "扩散模型"
  - "VLM"
  - "3D Gaussian Splatting"
---

# HouseTour: A Virtual Real Estate A(I)gent

**会议**: ICCV 2025  
**arXiv**: [2510.18054](https://arxiv.org/abs/2510.18054)  
**代码**: [https://house-tour.github.io/](https://house-tour.github.io/)  
**领域**: 3D视觉 / 视觉语言模型 / 轨迹生成  
**关键词**: Camera Trajectory Generation, Real Estate, Diffusion, VLM, 3D Gaussian Splatting  

## 一句话总结
提出 HouseTour，给定一组已知位姿的室内图像，联合生成类人的3D相机轨迹和房地产文字描述，通过 Residual Diffuser 进行基于扩散的轨迹规划并将空间特征集成到 Qwen2-VL-3D 中生成3D-grounded文本摘要。

## 研究背景与动机

房产视频导览是美国房地产市场（价值3.43万亿美元）的关键工具。然而，制作专业导览视频需要：(1) 具备高端摄影设备的专业经纪人实地拍摄；(2) 精心撰写空间描述文案。这一过程劳动密集且成本高昂。

**现有方法的局限**：
- 视觉语言模型（VLMs）在几何推理方面能力不足，无法理解3D空间布局
- 现有3D数据集的相机轨迹针对重建任务设计（近距离贴近物体表面、抖动运动），不适合展示整体空间
- 场景描述数据集只罗列家具及其关系，缺乏对空间布局、建筑特征、材质、氛围的专业描述

**核心目标**：让普通用户无需专业设备，仅上传一组手机拍摄的照片，即可自动生成专业级房产导览视频+文字描述。

## 方法详解

### 整体框架
给定稀疏有序相机位姿 $\mathcal{C}=[c_1,...,c_{N_c}]$ 和对应RGB图像 $\mathcal{I}$：
1. **Residual Diffuser**：生成平滑的类人导览轨迹 $\tau$ ($N > N_c$ 帧)
2. **Qwen2-VL-3D**：利用轨迹空间特征+视觉tokens生成房地产文字摘要 $\Sigma$
3. **3DGS渲染**：沿生成轨迹渲染最终导览视频

### Residual Diffuser — 基于扩散的轨迹规划

核心创新：不直接学习绝对轨迹（因为不同房产布局差异巨大），而是学习**相对于样条插值的残差**。

**公式化**：$\tilde{p} = \mathcal{S} + \Delta p$，其中 $\mathcal{S}$ 是已知位姿间的样条插值，$\Delta p$ 是预测残差。对于已知位姿时间步，残差为零向量。

**反向扩散过程**：
$$\begin{cases} \vec{0} = \delta(p^i) & \text{if } i \in t_\tau \\ p_\theta(\Delta p_{t-1}^i | \Delta p_t^i, \mathcal{S}) = \mathcal{N}(\Delta p_{t-1}^i; \mu_\theta, \Sigma_\theta) & \text{else} \end{cases}$$

**轨迹损失**：
- 平移：在均匀采样的密集样条点上计算L2范数
- 旋转：在SO(3)流形上使用测地线损失
$$\mathcal{L}_\theta = \mathbb{E}_{t,\tau,\epsilon}\left[\|\epsilon_{pos} - \epsilon_\theta(pos_t,t)\|^2 + d_{geo}(\epsilon_{rot}, \epsilon_\theta(rot_t,t))\right]$$

关键设计：在连续相机位姿间用Horner方法高效评估样条段上的均匀采样点，将轨迹建模为连续函数而非离散点序列。

### Qwen2-VL-3D — 3D感知的文本生成

**两阶段训练**：
1. **LoRA微调**：在96帧多图像输入上微调Qwen2-VL，学习房地产描述的语言风格和建筑术语
2. **空间特征集成**：
    - 添加特殊token `<|traj_start|>`, `<|traj_pad|>`, `<|traj_end|>`
    - 将Residual Diffuser的去噪位姿 $p_0^i$ 和瓶颈层特征 $f_0^i$ 拼接后通过线性层映射到VLM的语言嵌入空间
    - 每帧使用单个token编码空间信息

### HouseTour数据集
- 1639个导览视频，涵盖从公寓到多层别墅的多样房产
- 1298个视频带文字描述（半数有时间戳），878个场景有3D重建
- 提供场景级人类轨迹、密集点云和专业房地产描述

## 实验

### 端到端性能

| 方法 | R@75cm ↑ | Rot. Score ↑ | BT ↑ | SLS ↑ |
|------|----------|-------------|------|-------|
| Baseline (Catmull-Rom + Qwen2-VL SFT) | 57.1 | 96.8 | 71.4 | 71.7 |
| **HouseTour** | **60.2** | **97.1** | **79.5** | **76.0** |

HouseTour在所有指标上超越基线，特别是文本生成（BT +8.1）。

### 轨迹生成消融

| 方法 | R@50cm ↑ | R@1m ↑ | Euclidean ↓ | DTW ↓ | Geodesic ↓ |
|------|----------|--------|-------------|-------|-----------|
| Linear Interp. | 41.2% | 59.8% | 145.8 | 192.1 | 0.20 |
| Catmull-Rom | 45.9% | 64.7% | 106.2 | 146.3 | 0.10 |
| **Residual Diffuser** | **46.2%** | **69.4%** | **73.9** | **128.8** | **0.09** |

**关键发现**：
- Residual Diffuser在R@1m上显著优于插值方法（69.4% vs 64.7%），表明大误差更少
- 欧几里德距离降低30%+，残差学习显著优于直接轨迹学习
- 在更高不确定性的区域（远离已知位姿处），Residual Diffuser的优势更明显

## 亮点与洞察
1. **残差扩散建模**：将轨迹生成从学习绝对位置转为学习相对于样条的残差，优雅解决了跨场景的变化布局问题
2. **新联合评估指标SLS**：首次提出空间-语言联合评分（翻译回忆+旋转分数+Bradley-Terry的调和均值）
3. **三模态VLM**：将语言、视觉、3D定位三种模态集成到VLM中，实现空间感知的文本生成
4. **实用性强**：端到端系统可直接应用于房地产和旅游行业

## 局限性
- 依赖手机拍摄的图像质量和覆盖范围
- 3DGS渲染在稀疏视图下质量有限（作者将此声明为非本文范围）
- 数据集仅覆盖房地产场景，未扩展到其他导览场景（博物馆、旅游景点等）
- 文本生成仍依赖LoRA微调，对新领域的泛化需进一步验证

## 相关工作
- 长视频理解：TimeChat等处理长序列但缺乏建筑领域知识
- 轨迹规划：Diffuser系列用于机器人决策，但处理固定环境而非变化布局
- 3D视觉语言：ScanRefer、DenseCap描述物体关系，但忽略空间布局和建筑特征

## 评分
- **创新性**: ★★★★☆ — 残差扩散轨迹规划+三模态VLM的组合新颖
- **实用性**: ★★★★★ — 直接面向价值万亿美元的房地产市场
- **实验**: ★★★★☆ — 提出新数据集和评估指标，消融充分
- **写作**: ★★★★☆ — 结构清晰，问题定义明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)

</div>

<!-- RELATED:END -->
