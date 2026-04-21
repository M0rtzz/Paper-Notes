---
title: >-
  [论文解读] ProtoDepth: Unsupervised Continual Depth Completion with Prototypes
description: >-
  [CVPR 2025][3D视觉][持续学习] ProtoDepth提出基于原型（Prototype）的持续学习方法，通过冻结预训练模型并为每个新域学习轻量原型集来调制隐层特征，在室内和室外场景中将遗忘率降低超过50%。
tags:
  - CVPR 2025
  - 3D视觉
  - 持续学习
  - 深度补全
  - 原型学习
  - 灾难性遗忘
  - 无监督
---

# ProtoDepth: Unsupervised Continual Depth Completion with Prototypes

**会议**: CVPR 2025  
**arXiv**: [2503.12745](https://arxiv.org/abs/2503.12745)  
**代码**: [protodepth.github.io](https://protodepth.github.io/)  
**领域**: 3D Vision  
**关键词**: 持续学习, 深度补全, 原型学习, 灾难性遗忘, 无监督

## 一句话总结

ProtoDepth提出基于原型（Prototype）的持续学习方法，通过冻结预训练模型并为每个新域学习轻量原型集来调制隐层特征，在室内和室外场景中将遗忘率降低超过50%。

## 研究背景与动机

深度补全（从RGB图像和稀疏点云预测稠密深度图）在自动驾驶和机器人中广泛应用。无监督学习范式（不需要GT深度）天然适合持续学习场景。然而，当模型在非平稳分布的数据序列上训练时，会发生灾难性遗忘——在新域上训练后，之前学到的知识严重退化。

现有持续学习方法的局限：
- **正则化方法**（EWC等）：通过限制重要参数更新来缓解遗忘，但在大域转移时效果有限
- **回放方法**（Replay）：存储旧数据周期性重训，受限于内存和隐私约束
- **架构方法**：分配任务特定子网络，但参数量可能超过原始模型

核心洞察：如果冻结预训练模型的所有权重，就能保证**零遗忘**。问题变为如何用轻量参数适配新域。与NLP中的prompt不同，图像没有自然的token化尺度，因此prompt-based方法仅适用于ViT。本文提出的原型方法**架构无关**，同时适用于CNN和Transformer。

## 方法详解

### 整体框架

ProtoDepth冻结预训练深度补全模型 $f_\theta$，为每个新遭遇的域 $\mathcal{D}_k$ 学习一组原型集（prototype set）。原型集通过全局乘性偏置和局部加性偏置来变换隐层特征以适配新域分布。对于domain-agnostic设置（测试时不知域身份），额外学习域描述符来自动选择最匹配的原型集。

### 关键设计

**1. 全局原型（Global Prototype）+ 局部原型（Local Prototype）**

- **功能**：全局原型学习从预训练数据分布到新域分布的变换；局部原型捕获细粒度特征，可根据输入选择性查询
- **核心思路**：全局原型用 $1 \times 1$ 深度卷积实现乘性偏置，对隐层每个通道进行缩放。局部原型通过key-value机制工作：冻结的query与原型集中的key计算余弦相似度，加权选择对应的value作为加性偏置注入隐层特征
- **设计动机**：域转移可建模为全局的分布偏移（如不同传感器的系统偏差）加上局部的内容差异（如不同场景的几何特性）。全局+局部的组合能灵活表达各种域间差异

**2. 域描述符与原型集选择机制**

- **功能**：在测试时域身份未知的情况下，自动为输入样本选择最合适的原型集
- **核心思路**：为每个域学习一个描述符向量。推理时，将输入样本通过冻结编码器提取描述符，与所有已学习的域描述符计算余弦相似度，选择最匹配域的原型集
- **设计动机**：不同于需要知道域身份的domain-incremental设置，domain-agnostic设置更接近真实场景。描述符学习通过对比损失实现——同域样本描述符相近，异域样本描述符远离

**3. 训练目标设计**

- **功能**：在无监督深度补全的损失基础上增加域描述符的对比学习损失
- **核心思路**：无监督损失 $\mathcal{L} = w_{ph}\ell_{ph} + w_{sz}\ell_{sz} + w_{sm}\ell_{sm}$（光度一致性 + 稀疏深度一致性 + 局部平滑正则）。域描述符通过类InfoNCE的对比损失训练，使同域内样本描述符聚集
- **设计动机**：三项无监督损失是标准的SfM自监督信号，不需要GT深度；对比损失使描述符空间有区分性，支持推理时的域识别

### 损失函数

$$\mathcal{L} = w_{ph}\ell_{ph} + w_{sz}\ell_{sz} + w_{sm}\ell_{sm} + \lambda_{desc}\mathcal{L}_{desc}$$

其中 $\ell_{ph}$ 结合L1和SSIM衡量相邻帧光度重建误差，$\ell_{sz}$ 衡量预测深度与稀疏点云的L1距离，$\ell_{sm}$ 对深度梯度施加边缘感知平滑约束。

## 实验关键数据

### 主实验：室外序列（KITTI→Waymo→VKITTI），VOICED骨干

| 方法 | 平均遗忘MAE(%)↓ | 平均遗忘RMSE(%)↓ | 平均MAE(mm)↓ | 平均RMSE(mm)↓ |
|------|---------------|----------------|-------------|-------------|
| Finetuned | 8.828 | 6.131 | 63.352 | 125.28 |
| EWC | 9.439 | 8.014 | 63.787 | 126.706 |
| Replay | 6.154 | 4.688 | 64.305 | 126.714 |
| ProtoDepth-A | 2.439 | 3.598 | 56.971 | 118.132 |
| **ProtoDepth** | **0.000** | **0.000** | **56.359** | **115.153** |

### 消融：Domain-Incremental vs Domain-Agnostic

| 设置 | VOICED MAE遗忘(%) | FusionNet MAE遗忘(%) |
|------|-------------------|---------------------|
| ProtoDepth (已知域) | 0.000 | 0.000 |
| ProtoDepth-A (未知域) | 2.439 | 1.282 |

### 关键发现

- **Domain-incremental**设置下ProtoDepth实现**零遗忘**（冻结模型+独立原型集）
- **Domain-agnostic**设置下遗忘率降低52.2%（室内）和53.2%（室外），超越所有baseline
- 每个域仅增加<5%的原始模型参数量，远轻于现有架构方法
- 原型方法架构无关，在VOICED（CNN）和FusionNet（CNN+Sparse Conv）上都有效
- 域描述符的选择准确率在90%以上，是domain-agnostic设置有效的关键

## 亮点与洞察

1. **冻结+原型的范式**将持续学习从"如何安全更新参数"转变为"如何用轻量偏置适配新域"，从根本上消除了遗忘问题
2. **全局乘性+局部加性**的双重偏置设计直觉清晰：全局捕获系统性域差异（如传感器标定），局部捕获内容依赖的特征变化
3. 首次将原型方法引入无监督持续3D重建任务，拓展了原型学习的应用边界

## 局限与展望

- 域描述符需要在训练时同时训练，无法完全在线学习新域
- 原型集的大小和数量需要手动设定，未自适应调整
- 当前仅考虑深度补全任务，未扩展到其他3D感知任务（如语义理解）
- 极端域差异（如室内→极端天气户外）的适配效果有待验证

## 相关工作与启发

- **与VPT/L2P的关系**：Prompt-based方法将可学习token拼接到ViT输入，但不适用于CNN。ProtoDepth的原型在隐空间中操作，架构无关
- **与LoRA的关系**：LoRA通过低秩矩阵微调权重，仍可能造成微小遗忘。ProtoDepth冻结所有权重实现零遗忘
- **启发**：对于持续学习场景，"不改变模型，只添加轻量适配器"的思路值得推广到更多视觉任务

## 评分

⭐⭐⭐⭐

首次提出无监督持续深度补全的专用方法，原型范式设计优雅且理论清晰。零遗忘（domain-incremental）和>50%遗忘减少（domain-agnostic）的结果令人印象深刻。参数开销极低（<5%/域）。局限在于域描述符学习的灵活性和任务范围的拓展。

<!-- RELATED:START -->

## 相关论文

- [P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)
- [Open-World Amodal Appearance Completion](open-world_amodal_appearance_completion.md)
- [CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](../../ICCV2025/3d_vision/cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)

<!-- RELATED:END -->
