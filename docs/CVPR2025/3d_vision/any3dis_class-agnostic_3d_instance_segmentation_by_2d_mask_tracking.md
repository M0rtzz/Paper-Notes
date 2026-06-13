---
title: >-
  [论文解读] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking
description: >-
  [CVPR 2025][3D视觉][3D实例分割] 提出Any3DIS，通过3D感知的2D掩码跟踪（利用SAM-2追踪每个超点在多帧中的2D分割）替代传统的无监督合并策略，并用动态规划优化3D Proposal，在ScanNet200和ScanNet++上的类别无关、开放词汇、开放式3D实例分割任务中均取得SOTA。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D实例分割"
  - "2D掩码跟踪"
  - "SAM-2"
  - "类别无关"
  - "动态规划优化"
---

# Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking

**会议**: CVPR 2025  
**arXiv**: [2411.16183](https://arxiv.org/abs/2411.16183)  
**代码**: [https://any3dis.github.io/](https://any3dis.github.io/)  
**领域**: 3D视觉 / 3D实例分割  
**关键词**: 3D实例分割, 2D掩码跟踪, SAM-2, 类别无关, 动态规划优化

## 一句话总结
提出Any3DIS，通过3D感知的2D掩码跟踪（利用SAM-2追踪每个超点在多帧中的2D分割）替代传统的无监督合并策略，并用动态规划优化3D Proposal，在ScanNet200和ScanNet++上的类别无关、开放词汇、开放式3D实例分割任务中均取得SOTA。

## 研究背景与动机
3D实例分割是计算机视觉的核心任务，在自主导航、增强现实、场景理解中有广泛应用。现有的类别无关3D实例分割方法（如Open3DIS）采用"先分割后合并"的策略——在每帧上用SAM等模型生成密集2D掩码，再提升到3D空间通过启发式准则合并。这带来两个核心痛点：（1）过度分割，产生大量冗余的3D Proposal；（2）合并不一致性，来自不同视角的2D分割结果缺乏一致性，合并质量差。根本原因在于启发式合并过程无法利用视频中物体的时序一致性信息。本文的核心idea是"跟踪替代合并"——用SAM-2的视频跟踪能力直接获得跨帧一致的2D掩码轨迹，每条轨迹自然对应一个3D物体。

## 方法详解
Any3DIS将3D实例分割简化为：先用2D视频分割与跟踪得到每个物体的跨帧掩码轨迹，再将轨迹提升为3D Proposal并通过优化精炼。

### 整体框架
输入为3D点云和对应的RGB-D帧序列。首先对点云进行超点分割和最远点采样选取初始超点。对每个超点，找到其在所有帧中可见性最高的"枢轴视角"，在该视角用SAM-2分割目标物体并向前后帧跟踪，得到完整的2D掩码轨迹。然后将轨迹提升为3D超点集合形成候选Proposal，最后通过动态规划算法优化超点选择，输出精炼的3D掩码。迭代处理未被覆盖的超点直到所有物体被分割。

### 关键设计
1. **3D感知的2D掩码跟踪 (3D-Aware 2D Mask Tracking)**:
    - 功能：为每个采样超点生成跨所有帧的一致2D掩码轨迹
    - 核心思路：对每个超点 $\mathbf{S}_l$，计算其在每帧中的可见点数并用邻居超点可见性进行加权得到直方图 $\psi_t^l = |\rho_t^l| \cdot s_t^l$，选择直方图最大值对应帧为枢轴视角。在枢轴视角中用FPS采样3个投影点作为SAM-2的点提示，获得2D分割后作为掩码提示进行双向（前向+后向）跟踪
    - 设计动机：传统方法从第一帧开始分割容易遗漏后续帧出现的物体，且初始掩码可能不是最大可见区域。邻居超点加权确保选择的视角不仅当前超点可见，且整体物体都可见良好。对于物体消失后重现的情况，利用超点投影点重新提供查询点

2. **3D Proposal精炼优化 (3D Mask Optimization)**:
    - 功能：从候选超点集合中选择最优子集形成最终3D Proposal
    - 核心思路：定义二值优化问题——最大化选中超点在所有视角2D掩码内的投影点数（正项），最小化在掩码外的投影点数（负项）。由于穷举NP-hard，采用动态规划：按顺序遍历所有视角，每一步选择"保留当前方案"或"加入该视角所有新可见超点"中目标值更高的选项
    - 设计动机：直接选择所有IOU超过阈值的超点不关心多视角一致性——某超点可能在一个视角重叠度高但在其他视角很低。DP算法虽非全局最优但高效且实践效果超越所有先前方法

3. **迭代物体采样 (Iterative 3D Object Sampling)**:
    - 功能：确保场景中所有物体都被覆盖
    - 核心思路：每轮处理后标记已分配超点，对未分配超点重复FPS采样→跟踪→优化流程，直到无剩余自由超点
    - 设计动机：单轮采样可能遗漏被大物体遮挡或较小的物体

### 损失函数 / 训练策略
方法是免训练的（training-free），使用SAM-2 (ViT-L)进行零样本推理。关键超参数：2D-3D遮挡深度阈值0.1，3D提升IOU阈值 $\tau=0.5$，RGB-D帧采样间隔10帧。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|------|----------|------|
| ScanNet++ | Class-Agnostic | AP | 22.2 | 20.7 (Open3DIS) | +1.5 |
| ScanNet200 | Class-Agnostic (2D only) | AP | 32.5 | 31.5 (Open3DIS) | +1.0 |
| ScanNet200 | Class-Agnostic (3D+2D) | AP | 42.5 | 41.5 (Open3DIS) | +1.0 |
| ScanNet200 | Open-Vocab (3D+2D) | AP | 25.8 | 23.7 (Open3DIS) | +2.1 |
| ScanNet++ | Open-Ended | AP | 20.1 | 18.4 (Point-Wise) | +1.7 |
| ScanNet200 | Open-Ended | AP | 19.1 | 16.0 (Point-Wise) | +3.1 |

### 消融实验

| 配置 | AP | AP50 | AP25 | 说明 |
|------|-----|------|------|------|
| 基线(无3D-AMT, 无MaskOpt) | 17.0 | 30.1 | 42.6 | 第一帧枢轴+全部超点 |
| +3D-AMT | 17.9 | 31.2 | 45.1 | 3D感知枢轴选择 |
| +MaskOpt | 20.3 | 33.7 | 44.5 | DP优化超点选择 |
| +3D-AMT +MaskOpt (完整) | 22.2 | 35.8 | 47.0 | 两者互补 |

### 关键发现
- MaskOpt贡献最大（+3.3 AP），说明精炼超点选择比改善跟踪起点更重要
- 在掩码优化策略对比中，DP优化（AP 22.2）远优于top-1视角（19.8）或top-10视角（20.8）的穷举方案
- 在开放词汇设置下，尾部类别（APtail=26.4）的提升尤为显著，说明方法对稀有类别更友好
- 在ScanNet++上即使不使用3D Proposal，仅2D方案也超越了使用SAM-HQ的Open3DIS

## 亮点与洞察
- "跟踪替代合并"的思路简洁而深刻——利用SAM-2的时序一致性天然解决了多视角2D掩码不一致的问题
- 3D感知的枢轴视角选择非常实用，邻居可见性加权是一个巧妙的trick
- 动态规划解NP-hard问题的设计思路值得学习，虽然不保证最优但实践效果出色
- 免训练方案使方法可直接推广到新场景和新数据集

## 局限与展望
- 强烈依赖SAM-2的分割和跟踪质量，SAM-2的2D错误会传播到3D结果
- DP算法每步只能选择"全部加入/不加入"，无法对单个超点做精细选择，可能非全局最优
- 对于高度遮挡或重叠的物体，跟踪可能丢失或混淆
- 可以考虑引入量子计算或更高级的优化算法改进NP-hard问题的求解

## 相关工作与启发
- 与Open3DIS的关系：共享超点+2D掩码提升到3D的基本范式，但用跟踪替代合并，根本性改变了pipeline
- SAM-2的引入是关键，其视频分割能力将2D到3D的提升从独立帧扩展到时序一致性
- 启发：Foundation model在3D任务中的应用仍有巨大潜力，关键是如何设计利用其优势的pipeline

## 补充分析

### 方法复杂度
- 每个超点需要计算在T帧中的投影直方图，复杂度与帧数线性相关
- DP优化的时间复杂度为 $O(T \cdot L)$，其中L为超点数，实际运行高效
- 迭代采样通常2-3轮即可覆盖场景中绝大部分物体

### 与2D跟踪模型的通用性
- 当前使用SAM-2 ViT-L作为分割和跟踪backbone
- 方法中SAM-2可替换为其他视频分割模型（如XMem等），框架具有通用性
- SAM-2的记忆窗口限制（7帧）通过投影点重查询机制得到解决
- 未来SAM系列模型精度提升将直接惠及本方法

## 评分
- 新颖性: ⭐⭐⭐⭐ "跟踪替代合并"的核心思路新颖且直觉良好，但整体框架仍基于超点+2D掩码的已有范式
- 实验充分度: ⭐⭐⭐⭐⭐ 三种评估设置（类别无关/开放词汇/开放式），两个数据集，消融完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码和公式推导规范
- 价值: ⭐⭐⭐⭐ 在免训练3D实例分割方向有重要的方法论贡献，对后续工作有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ASSIST-3D: Adapted Scene Synthesis for Class-Agnostic 3D Instance Segmentation](../../AAAI2026/3d_vision/assist-3d_adapted_scene_synthesis_for_class-agnostic_3d_instance_segmentation.md)
- [\[CVPR 2025\] Category-Agnostic Neural Object Rigging](category-agnostic_neural_object_rigging.md)
- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](../../ICCV2025/3d_vision/cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[CVPR 2025\] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation](relation3d_enhancing_relation_modeling_for_point_cloud_instance_segmentation.md)

</div>

<!-- RELATED:END -->
