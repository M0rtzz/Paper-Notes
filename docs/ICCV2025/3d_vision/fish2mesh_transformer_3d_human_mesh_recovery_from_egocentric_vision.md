---
title: >-
  [论文解读] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision
description: >-
  [ICCV 2025][3D视觉][第一人称视觉] 本文提出Fish2Mesh，一个鱼眼感知的Transformer模型，通过等距矩形投影的自我中心位置编码（EPE）将鱼眼图像的球面几何信息嵌入Swin Transformer，实现从头戴鱼眼相机的第一人称视角准确恢复3D人体mesh。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "第一人称视觉"
  - "鱼眼纠偏"
  - "人体mesh重建"
  - "SMPL"
  - "位置编码"
---

# Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision

**会议**: ICCV 2025  
**arXiv**: [2503.06089](https://arxiv.org/abs/2503.06089)  
**代码**: 有（项目网站）  
**领域**: 3D视觉  
**关键词**: 第一人称视觉, 鱼眼纠偏, 人体mesh重建, SMPL, 位置编码

## 一句话总结
本文提出Fish2Mesh，一个鱼眼感知的Transformer模型，通过等距矩形投影的自我中心位置编码（EPE）将鱼眼图像的球面几何信息嵌入Swin Transformer，实现从头戴鱼眼相机的第一人称视角准确恢复3D人体mesh。

## 研究背景与动机

**领域现状**：第一人称人体估计从双目立体视觉发展到单目系统，但主流研究集中在关节姿态估计。人体mesh恢复（HMR）相比关节估计能捕获更完整的体型和体量信息，但在第一人称鱼眼视角下挑战更大。

**现有痛点**：(1) 第一人称数据集稀缺且标注困难；(2) 鱼眼镜头引入严重的空间畸变，尤其在图像边缘；(3) 自遮挡问题严重——手臂挡住躯干、头部限制下半身可见性；(4) 当前SOTA方法EgoHMR使用扩散模型，输出不确定性大，不适合实时XR和机器人交互场景，且未处理鱼眼畸变。

**核心矛盾**：标准Transformer的位置编码假设规则网格，无法表达鱼眼投影的非线性空间扭曲。直接在鱼眼图像上使用标准模型会丢失3D空间上下文。

**本文目标**：设计一个原生理解鱼眼几何的Transformer架构，在第一人称鱼眼视角下准确回归SMPL参数和3D人体mesh。

**切入角度**：将鱼眼图像视为球面投影，通过等距矩形投影-球面坐标转换生成可学习的3D位置编码表，编码每个像素的球面空间信息。

**核心 idea**：用等距矩形投影将鱼眼图像的2D像素坐标转换为3D球面坐标 $(x_{3D}, y_{3D}, z_{3D})$，离散化后构建可学习的位置编码表，替换Swin Transformer中的标准位置偏置，使模型天然感知鱼眼几何畸变。

## 方法详解

### 整体框架
输入鱼眼RGB图像，先计算等距矩形投影的EPE位置编码，与图像patch一起送入Swin Transformer的patch merging层。经4个Swin块提取层次化特征后，送入三个任务头分别预测SMPL形状/姿态参数、相机变换和全局朝向。辅助3D/2D关节损失辅助训练。

### 关键设计

1. **自我中心位置编码（EPE）**:

    - 功能：将鱼眼几何的3D球面信息编码到Transformer中
    - 核心思路：先用等距矩形投影公式将2D鱼眼像素 $(x_{2D}, y_{2D})$ 转换为球面经纬度 $(\lambda, \varphi)$，再转换为3D球面坐标 $x_{3D} = R \cdot \sin(\varphi) \cdot \cos(\lambda)$ 等。将连续坐标离散化后查询可学习的嵌入表 $POS[x_{3D}, y_{3D}, z_{3D}]$，加到图像token上。同时移除Swin Transformer原有的相对位置偏置（因为EPE已包含位置信息）
    - 设计动机：标准位置编码无法表达球面畸变信息，EPE直接将3D空间约束注入模型，使self-attention可以在几何正确的特征空间上操作

2. **多任务头联合优化**:

    - 功能：同时回归SMPL参数和辅助关节坐标
    - 核心思路：三个任务头分别输出SMPL shape参数 $\Theta_s$、pose参数 $\Theta_p$、相机变换 $\Pi$ 和全局朝向 $O$。同时预测3D关节坐标和2D投影作为辅助监督，确保3D-2D一致性
    - 设计动机：SMPL参数空间高维抽象，辅助关节损失提供更直接的几何约束，帮助模型更快收敛

3. **弱监督数据增强**:

    - 功能：缓解第一人称数据稀缺问题
    - 核心思路：使用预训练的4D-Human模型从第三人称相机图像生成SMPL伪标签，配合基于prompt的自然动作采集系统，生成覆盖日常活动（含真实头部运动和自遮挡）的训练数据
    - 设计动机：真正的第一人称数据极其稀缺且标注成本高，弱监督策略显著扩大训练集

### 损失函数 / 训练策略
$\mathcal{L} = a(\mathcal{L}_{SMPL} + \mathcal{L}_{orient}) + b \cdot \mathcal{L}_{3D} + c \cdot \mathcal{L}_{2D}$，其中SMPL损失为shape和pose参数的L2损失，朝向和3D/2D关节为L1损失。端到端从头训练。

## 实验关键数据

### 主实验

| 模型 | MPJPE↓(mm) | MPVPE↓(mm) | PA-MPJPE↓(mm) | PA-MPVPE↓(mm) |
|---|---|---|---|---|
| 4DHuman | 390.0 | 521.3 | 90.0 | 129.8 |
| EgoHMR(扩散) | 较差 | 较差 | 竞争力 | 竞争力 |
| **Fish2Mesh** | **最优** | **最优** | **最优** | **最优** |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| 无EPE（标准位置编码） | 下降 | 鱼眼畸变未处理 |
| 无辅助3D/2D损失 | 下降 | 缺少几何约束 |
| 无数据增强 | 显著下降 | 训练数据不足 |
| 完整Fish2Mesh | 最优 | 所有组件互补 |

### 关键发现
- EPE位置编码是性能提升的最大贡献因素，证明几何感知是鱼眼HMR的核心
- 确定性回归（本文）比扩散生成（EgoHMR）更适合实时应用场景
- 弱监督数据增强对补偿第一人称数据稀缺至关重要
- 模型在多个第一人称数据集上一致超越SOTA

## 亮点与洞察
- **EPE的几何直觉**：将鱼眼畸变不再视为需要纠正的错误，而是将其球面本质通过3D位置编码编入模型。比先去畸变再处理更优雅且保留更多信息
- **从姿态估计到mesh恢复的提升**：指出不同数据集关键点定义不一致（COCO 32点 vs H36M 17点），mesh恢复天然避免这个问题
- **实用性导向**：明确针对实时XR和机器人交互场景设计，回避了扩散模型的不确定性缺陷

## 局限与展望
- 需要已知鱼眼镜头参数（焦距、视场角等），不同设备需调整
- 训练数据仍受限于实验室环境，真实野外场景表现有待验证
- 仅处理单帧，未利用时序信息改善遮挡下的估计
- 可扩展到多人场景或结合手部/面部估计

## 相关工作与启发
- **vs EgoHMR**: 使用扩散模型但输出不确定性高且不处理鱼眼畸变；Fish2Mesh用确定性回归+几何位置编码
- **vs FisheyeViT**: 用patch undistortion处理鱼眼，但patch切分引入大量预处理开销；EPE更简洁高效
- EPE的思路可迁移到任何使用鱼眼/全景相机的视觉Transformer任务

## 评分
- 新颖性: ⭐⭐⭐⭐ EPE位置编码将鱼眼几何原生编入Transformer是创新设计
- 实验充分度: ⭐⭐⭐ 数据集较少，基线对比有限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题动机表述明确
- 价值: ⭐⭐⭐⭐ 对XR/机器人场景的第一人称感知有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](../../CVPR2025/3d_vision/prompthmr_promptable_human_mesh_recovery.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](../../CVPR2025/3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[CVPR 2025\] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](../../CVPR2025/3d_vision/heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
