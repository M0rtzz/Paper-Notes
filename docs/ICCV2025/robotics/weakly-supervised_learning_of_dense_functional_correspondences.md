---
title: >-
  [论文解读] Weakly-Supervised Learning of Dense Functional Correspondences
description: >-
  [ICCV 2025][机器人][稠密功能对应] 定义了"稠密功能对应"（Dense Functional Correspondence）任务——基于物体功能（如"倒水"）在不同类别物体之间建立像素级稠密对应，并提出一种弱监督学习框架，通过 VLM 伪标注功能部件 + 多视角对比学习来蒸馏功能和结构知识到新模型中。
tags:
  - ICCV 2025
  - 机器人
  - 稠密功能对应
  - 弱监督学习
  - 视觉语言模型
  - 对比学习
  - 机器人操控
---

# Weakly-Supervised Learning of Dense Functional Correspondences

**会议**: ICCV 2025  
**arXiv**: [2509.03893](https://arxiv.org/abs/2509.03893)  
**代码**: [项目页面](https://dense-functional-correspondence.github.io/)  
**领域**: 机器人  
**关键词**: 稠密功能对应, 弱监督学习, 视觉语言模型, 对比学习, 机器人操控

## 一句话总结

定义了"稠密功能对应"（Dense Functional Correspondence）任务——基于物体功能（如"倒水"）在不同类别物体之间建立像素级稠密对应，并提出一种弱监督学习框架，通过 VLM 伪标注功能部件 + 多视角对比学习来蒸馏功能和结构知识到新模型中。

## 研究背景与动机

建立跨图像的像素级对应关系是形状重建、图像编辑和机器人操控等任务的基础。当前方法在以下三个层次面临递增的挑战：

**同一物体不同视角**：多视角对应，相对简单

**同类别不同实例**：如两只猫之间的对应，现有方法（NOCS、关键点检测）已有较好解决方案

**不同类别物体**：如水壶和瓶子的功能性对应，**这是最具挑战性但最重要的场景**

**关键洞察**："形式服从功能"（form follows function）——执行相似功能的物体部件（如水壶的壶嘴和瓶子的瓶口）在形状和外观上往往具有相似性，即使整体物体外观差异很大。这为跨类别物体建立稠密对应关系提供了天然的桥梁。

现有方法的不足：

- **自监督表示**（DINOv2、Stable Diffusion）：在同类别物体间建立对应效果好，但跨类别时准确度显著下降
- **视觉语言模型**（CogVLM、ManipVQA）：可以零样本检测功能部件的边界框，但无法进行细粒度的像素级对应推理
- **关键点方法**（Lai et al.）：仅定义 5 个关键点，无法捕捉高度不同物体之间的细微相似性
- **可供性学习**（Affordance）：仅识别单张图像中的交互区域，不能建立跨图像的稠密对应

## 方法详解

### 整体框架

整个方法分为三个阶段：

1. **评估数据集构建**：通过 3D 物体对齐推导 2D 稠密功能对应标注
2. **训练数据集构建**：利用 VLM (CogVLM) 伪标注 + GPT-4 生成提示来获取大规模功能部件标签
3. **模型训练**：在冻结的 DINOv2 特征之上训练功能条件化 MLP，结合功能部件对比损失和多视角空间对比损失

### 关键设计

1. **稠密功能对应的形式化定义**:

    - 功能：给出严格的数学定义，将"功能性相似"转化为可计算的 3D 距离
    - 核心思路：给定功能 $\mathcal{F}$ 和图像对 $(I_1, I_2)$，定义功能对应映射 $f(I_1, I_2; \mathcal{F}): M(I_1;\mathcal{F}) \to M(I_2;\mathcal{F})$，最小化 $\sum_{p \in M(I_1;\mathcal{F})} \|\pi^{-1}(p) - \pi^{-1}(f(p))\|_2$，其中 $\pi^{-1}$ 是像素到 3D 表面点的反投影
    - 设计动机：通过 3D 对齐来定义 2D 对应，避免了人工标注稠密对应的不可行性，同时提供了自然的评估基准

2. **VLM 伪标注管线**:

    - 功能：利用大规模预训练 VLM 自动标注 Objaverse 3D 资产的功能部件
    - 核心思路：GPT-4 生成类别-功能-部件的文本提示 → CogVLM 对多视角渲染图预测边界框 → 将 2D 标签反投影聚合到 3D 点云 → 后处理生成 2D 像素级伪标签。涵盖 24 个功能类别、160 个物体类别、8285 个 3D 资产
    - 设计动机：手工标注稠密对应不可行，利用 VLM 的零样本能力进行伪标注，结合多视角聚合和 3D 一致性来提高标签质量

3. **功能条件 MLP + 双重对比学习**:

    - 功能：训练一个以功能文本为条件的特征提取网络，同时学习功能语义和空间结构
    - 核心思路：模型 $g_\theta(p|I,\mathcal{F})$ 在 DINOv2（多层特征加权）和 CLIP 文本嵌入之上加 3 层 MLP。训练包含：
        - **功能部件对比损失** $\mathcal{L}_{func}$：InfoNCE 损失，功能部件像素为正对，非功能部件为负对，且非功能部件之间也互相排斥
      $\mathcal{L}_{func} = -\log\frac{e^{\text{sim}(p_1^+, p_2^+)/\tau}}{e^{\text{sim}(p_1^+, p_2^+)/\tau} + e^{\text{sim}(p_1^+, p_2^-)/\tau} + e^{\text{sim}(p_1^-, p_2^-)/\tau}}$
        - **多视角空间对比损失** $\mathcal{L}_{spatial}$：同一物体不同视角的像素对应为正对，防止特征坍缩
      $\mathcal{L}_{spatial} = -\log\frac{e^{\text{sim}(q, q_+^\prime)/\tau}}{e^{\text{sim}(q, q_+^\prime)/\tau} + e^{\text{sim}(q, q_-^\prime)/\tau}}$
        - 可选的 **掩码预测损失** $\mathcal{L}_{mask}$
    - 设计动机：单独使用功能对比损失会导致模式坍缩（整个壶嘴特征相同），空间对比损失保留结构信息（壶嘴顶部和底部应有不同特征）。两者互补不可或缺

### 损失函数 / 训练策略

- 最终损失 $\mathcal{L} = \mathcal{L}_{func} + \lambda_{spatial}\mathcal{L}_{spatial} + \lambda_{mask}\mathcal{L}_{mask}$
- $\lambda_{spatial} = 10$，$\lambda_{mask} = 1$
- DINOv2-B backbone 冻结，仅训练 MLP（3层，1024维隐层）
- Adam 优化器，学习率 $1 \times 10^{-4}$，batch size 50 对图像，每张图 128 个采样点
- 随机颜色背景增强

## 实验关键数据

### 主实验

**合成评估数据集（1800+ 图像对，24 功能，85% 跨类别）：**

| 方法 | Norm.Dist↓ | PCK@23p↑ | Best F1@23p↑ | AP@23p↑ |
|------|-----------|---------|-------------|--------|
| Chance | 0.310 | 0.165 | 0.416 | 0.256 |
| DINO | 0.212 | 0.381 | 0.578 | 0.381 |
| SD-DINO | 0.227 | 0.376 | 0.563 | 0.341 |
| CogVLM + DINO | 0.180 | 0.416 | 0.678 | 0.556 |
| **Ours (full)** | **0.170** | **0.486** | **0.768** | **0.685** |

**真实评估数据集（HANDAL，500+ 图像对，13 功能）：**

| 方法 | Norm.Dist↓ | PCK@23p↑ | Best F1@23p↑ | AP@23p↑ |
|------|-----------|---------|-------------|--------|
| DINO | 0.206 | 0.408 | 0.589 | 0.382 |
| CogVLM + DINO | 0.172 | 0.440 | 0.695 | 0.561 |
| **Ours (full w/ mask)** | **0.153** | **0.501** | **0.808** | **0.730** |

### 消融实验

| 配置 | Norm.Dist↓ | PCK@23p↑ | AP@23p↑ | 说明 |
|------|-----------|---------|--------|------|
| Functional only | 0.228 | 0.287 | 0.441 | 模式坍缩，结构信息丢失 |
| Spatial only | 0.204 | 0.470 | 0.412 | 缺乏功能语义 |
| Full (w/o mask) | 0.170 | 0.486 | 0.685 | 功能+空间互补 |
| Full (w/ mask) | 0.172 | 0.480 | 0.684 | 掩码在真实数据上有提升 |

### 关键发现

1. **纯合成数据训练可泛化到真实图像**：在合成 Objaverse 数据上训练的模型在 HANDAL 真实数据集上也表现优异
2. **功能和空间对比缺一不可**：单独使用功能损失导致模式坍缩（PCK仅0.287），单独使用空间损失缺乏功能理解（AP仅0.412）
3. **推理速度优势**：模型推理比 CogVLM 快约 50 倍，比 ManipVQA 快约 1000 倍
4. ManipVQA 用功能名称提示（ManipVQA-F）效果远差于用部件名称提示（ManipVQA-P），说明零样本功能推理仍然困难

## 亮点与洞察

- **任务定义贡献** ：首次形式化定义了"稠密功能对应"任务，填补了稀疏功能关键点对应和类内稠密对应之间的空白
- **巧妙的数据策略**：利用 GPT-4 + CogVLM 的组合进行大规模伪标注，结合 3D 一致性校验，将人工干预降至最低
- **互补蒸馏**：将 VLM 的语义理解能力和 DINOv2 的空间对应能力蒸馏融合到轻量级 MLP 中
- **评估方法设计**：通过 3D 物体对齐自动推导 2D 稠密对应真值，提供了可扩展的评估管线
- **实用价值**：对机器人模仿学习（如将一个物体上的操作演示迁移到另一类物体）具有直接应用价值

## 局限与展望

1. 当前假设输入图像已经完成物体分割，实际应用中需要额外的分割模块
2. 功能分类的粒度（24个功能）可能不够覆盖所有实际需求
3. Objaverse 资产质量参差不齐，部分资产的功能部件标注可能存在噪声
4. 评估仅限于工具类和容器类物体，是否适用于更广泛的物体类别（如家具、电子设备）有待验证
5. 未探索功能对应在下游机器人操控任务中的实际效果

## 相关工作与启发

- 与 NOCS 等类内对应方法相比，功能对应是一种更高层次的抽象，跨越了类别边界
- VLM 伪标注 + 对比学习蒸馏的范式可以推广到其他需要稠密标注但标注成本高的任务
- 3D 对齐定义 2D 对应的思路可以推广到其他涉及 3D 结构的对应任务
- 与 affordance grounding 的区别：affordance 关注"如何与物体交互"，functional correspondence 关注"不同物体间功能等价部件的对齐"

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 全新的任务定义 + 创新的数据构建管线 + 巧妙的训练方法
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据集评估全面，消融清晰，但缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法流程连贯，图表精美
- 价值: ⭐⭐⭐⭐⭐ 开辟了全新研究方向，对机器人操控领域有重要应用价值

<!-- RELATED:START -->

## 相关论文

- [Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)

<!-- RELATED:END -->
