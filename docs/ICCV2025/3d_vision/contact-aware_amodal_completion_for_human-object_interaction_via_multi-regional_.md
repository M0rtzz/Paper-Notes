---
title: >-
  [论文解读] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting
description: >-
  [ICCV 2025][3D视觉][非模态补全] 提出首个面向人物交互（HOI）场景的非模态补全框架，利用人体拓扑和接触信息通过凸包操作识别遮挡区域，结合多区域修复策略在预训练扩散模型上无需额外训练即可完成高质量的遮挡物体补全。
tags:
  - ICCV 2025
  - 3D视觉
  - 非模态补全
  - 人物交互
  - 多区域修复
  - 扩散模型
  - 接触估计
---

# Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting

**会议**: ICCV 2025  
**arXiv**: [2508.00427](https://arxiv.org/abs/2508.00427)  
**代码**: 无  
**领域**: 3D视觉 / 人物交互理解  
**关键词**: 非模态补全, 人物交互, 多区域修复, 扩散模型, 接触估计

## 一句话总结

提出首个面向人物交互（HOI）场景的非模态补全框架，利用人体拓扑和接触信息通过凸包操作识别遮挡区域，结合多区域修复策略在预训练扩散模型上无需额外训练即可完成高质量的遮挡物体补全。

## 研究背景与动机

非模态补全（Amodal Completion）是推断被部分遮挡物体完整外观的任务，对理解复杂人物交互场景至关重要。现有方法的主要问题：

**修复区域不准确**：直接将遮挡者（如人体）的mask作为修复区域，通常远大于实际遮挡区域，导致扩散模型生成过度扩展或不准确的补全

**缺乏HOI先验**：现有方法未利用人物交互的独特特征——可见区域呈凹形、人体拓扑可获取、接触点提供关键空间关系

**单区域修复局限**：传统方法只能处理单个mask区域，无法对不同遮挡概率的区域施加差异化策略

## 方法详解

### 整体框架

流水线分两个核心阶段：
1. **遮挡区域识别**（Occluded Region Identification）：利用接触信息和凸包操作将遮挡区域分为主区域$M_p$（高概率遮挡）和次区域$M_s$（低概率遮挡）
2. **多区域修复**（Multi-Regional Inpainting）：在预训练Stable Diffusion v2修复模型上，对两个区域施加差异化去噪策略

### 关键设计

1. **接触感知凸包（Contact-aware Convex Hull）**：

    - 对人体mask $M_{human}$和物体mask $M_{object}$做膨胀操作，得到遮挡边界mask $M_{boundary}$
    - 将$M_{boundary}$与接触图$M_{contact}$合并：$C = M_{boundary} \cup M_{contact}$
    - 对点集$C$计算凸包$H = \text{ConvexHull}(C)$
    - 主区域mask：$M_p = M_{in} \cap M_{hull}$，即凸包内与遮挡者重叠的区域
    - 次区域mask：$M_s = M_{in} \setminus M_p$，即遮挡者mask内凸包外的区域
    - 设计动机：HOI中物体被遮挡的部分通常在接触点附近，凸包操作能精确定位关键修复区域

2. **多区域修复策略**：

    - 扩展标准SD-inpaint流程以处理多区域mask：
    $I_{out} = F_{T \to T'}(I_{in}, M_p, \mathcal{P}) \,|\, F_{T' \to 0}(I_{in}, M_p \cup M_s, \mathcal{P})$
    - 其中$T' = \lfloor T \cdot r \rfloor$，$r$为强度参数（默认0.5）
    - 第一阶段（$T \to T'$）：仅在主区域$M_p$做去噪，建立粗略结构
    - 第二阶段（$T' \to 0$）：在$M_p \cup M_s$上做去噪，基于主区域的初始结构逐步细化
    - 无需额外训练，完全基于预训练SD-inpaint模型
    - 设计动机：利用扩散模型"先建结构后加细节"的特性，确保主区域先获得合理补全，次区域在此基础上自然衔接

3. **野外数据处理流水线（In-the-Wild Pipeline）**：

    - 使用SAM生成人体和物体mask（替代ground-truth分割）
    - 使用HMR模型估计SMPL参数获取人体关节3D坐标
    - 使用VLM生成交互描述（如"a man is holding an object with both hands"）并提取相关SMPL关节ID
    - 将3D关节坐标投影到2D空间生成接触mask
    - 设计动机：消除对ground-truth标注的依赖，使方法可用于真实场景

### 损失函数 / 训练策略

本方法无需训练，完全基于预训练Stable Diffusion v2 Inpainting模型的推理流程。关键超参数为强度参数$r$（控制次区域修复的介入时序）和DDIM调度器步数$T=50$。

## 实验关键数据

### 主实验 - 非模态补全性能对比（表格）

| 方法 | BEHAVE CLIP↑ | BEHAVE mIoU↑ | InterCap CLIP↑ | InterCap mIoU↑ | Win-rate |
|------|-------------|-------------|---------------|---------------|---------|
| Naive outpainting | 27.34 | 50.92% | 27.55 | 52.07% | 94.0% |
| LaMa | 25.97 | 60.47% | 26.43 | 51.38% | 92.4% |
| Inst-Inpaint | 26.08 | 63.71% | 26.12 | 57.54% | 88.0% |
| pix2gestalt | 23.45 | 69.58% | 26.14 | 68.32% | 68.0% |
| Xu et al. | 26.34 | 71.03% | 26.21 | 69.23% | 65.8% |
| **Ours** | **26.91** | **77.64%** | **26.97** | **72.34%** | - |

### 消融实验 - 区域策略与强度参数（表格）

| 方法 | 区域 | r值 | CLIP↑ | mIoU↑ |
|------|------|-----|-------|-------|
| Naive outpainting | 全画面 | - | 27.34 | 50.92% |
| Human mask（单区域） | $M_p \cup M_s$ | 1.0 | 26.27 | 69.98% |
| 凸包无接触 | $M_p$ | 0.0 | 26.43 | 75.24% |
| 凸包含接触 | $M_p$ | 0.0 | 26.63 | 76.11% |
| **Ours多区域** | **{$M_p, M_s$}** | **0.5** | **26.91** | **77.64%** |
| Ours + GT接触 | {$M_p, M_s$} | 0.5 | 27.07 | 80.15% |

| 遮挡率 | r=0.0 mIoU | r=0.1 mIoU | r=0.5 mIoU | r=0.9 mIoU | r=1.0 mIoU |
|--------|-----------|-----------|-----------|-----------|-----------|
| 轻度(10-40%) | 84.97% | **85.44%** | 84.70% | 80.33% | 72.45% |
| 重度(40-70%) | 70.20% | 71.54% | **72.93%** | **73.94%** | 68.33% |
| 总计 | 76.11% | 77.10% | **77.64%** | 76.50% | 69.98% |

### 关键发现

- **mIoU大幅领先**：在BEHAVE上达到77.64%，比最强baseline（Xu et al. 71.03%）高出6.6个百分点
- **用户偏好压倒性**：在1-on-1用户研究中，对所有baseline的胜率均超过65%
- **接触信息关键**：加入接触信息使凸包mIoU从75.24%提升至76.11%
- **多区域策略有效**：从单区域到多区域，mIoU提升1.5个百分点以上
- **in-the-wild流水线可靠**：无GT接触时与有GT仅差2.5个百分点mIoU
- **r=0.5为最优折中**：轻度遮挡偏好小r，重度遮挡偏好大r，0.5在总体上最优
- **3D重建应用**：非模态补全后的3D GS重建质量显著提升

## 亮点与洞察

1. **首次针对HOI的非模态补全**：填补了一个重要的研究空白，HOI场景的独特几何约束被巧妙利用
2. **无需训练的多区域修复**：扩展了标准inpainting流程，用不同强度的噪声处理不同优先级区域，设计简洁有效
3. **接触+凸包的物理先验**：将几何先验自然融入修复流程，显著提升了遮挡区域定位精度
4. **实用性强**：in-the-wild流水线结合SAM+HMR+VLM，无需任何标注即可工作

## 局限与展望

- 主要在室内单人-单物场景验证，对多人多物的复杂场景泛化性未知
- 单图处理，缺乏时序一致性，不适用于视频任务
- 依赖扩散模型的修复能力，对未见过的物体类型可能补全效果不佳
- 凸包假设对某些非凸遮挡模式可能不适用
- 未来可扩展到视频帧间一致的非模态补全，以及更复杂的多人交互场景

## 相关工作与启发

- 与pix2gestalt等通用非模态补全方法形成互补：通用方法不考虑HOI特有的物理约束
- 多区域差异化修复的思路可推广到其他需要分区域处理的图像编辑任务
- 凸包+接触点的区域识别方法可用于机器人抓取场景的遮挡物体理解
- 3D GS重建的应用展示了非模态补全在下游任务中的价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将非模态补全专门应用于HOI，多区域修复策略新颖实用
- **实验充分度**: ⭐⭐⭐⭐ 两个数据集+用户研究+丰富消融+3D应用，但数据集偏限（室内）
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，流水线图示直观，公式推导完整
- **价值**: ⭐⭐⭐⭐ HOI理解和3D重建的实用工具，开创了新的研究方向

<!-- RELATED:START -->

## 相关论文

- [Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [Open-World Amodal Appearance Completion](../../CVPR2025/3d_vision/open-world_amodal_appearance_completion.md)
- [Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
