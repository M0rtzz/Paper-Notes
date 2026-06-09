---
title: >-
  [论文解读] Scene-Agnostic Pose Regression for Visual Localization
description: >-
  [CVPR 2025][视觉定位] 提出"场景无关位姿回归"（SPR）新任务范式，以序列首帧为坐标原点回归后续帧的相对位姿，避免了APR需重训练、RPR需检索数据库、VO存在累积漂移的困境，并建立了200K全景图的360SPR大规模数据集和双分支SPR-Mamba模型。
tags:
  - "CVPR 2025"
  - "视觉定位"
  - "位姿回归"
  - "场景无关"
  - "全景图"
  - "Mamba"
---

# Scene-Agnostic Pose Regression for Visual Localization

**会议**: CVPR 2025  
**arXiv**: [2503.19543](https://arxiv.org/abs/2503.19543)  
**代码**: [https://github.com/junweizheng93/SPR](https://github.com/junweizheng93/SPR)  
**领域**: LLM评测  
**关键词**: 视觉定位, 位姿回归, 场景无关, 全景图, Mamba

## 一句话总结

提出"场景无关位姿回归"（SPR）新任务范式，以序列首帧为坐标原点回归后续帧的相对位姿，避免了APR需重训练、RPR需检索数据库、VO存在累积漂移的困境，并建立了200K全景图的360SPR大规模数据集和双分支SPR-Mamba模型。

## 研究背景与动机

**领域现状**：视觉定位主要有三种范式——绝对位姿回归(APR)直接从图像预测相对于场景坐标系的6DoF位姿；相对位姿回归(RPR)预测查询图和参考图之间的相对位姿；视觉里程计(VO)根据前一帧位姿和当前帧预测当前位姿。

**现有痛点**：APR学习场景特定特征，无法泛化到未知环境，更换场景必须重新训练；RPR泛化能力较好但推理时需要大规模参考图像数据库做检索，当参考图与查询图重叠不足时性能急剧下降；VO在开环轨迹中存在不可避免的累积漂移。

**核心矛盾**：三种范式存在"泛化能力 vs 推理依赖 vs 累积误差"的三难困境——没有一种方法能同时做到免重训练、免数据库、无累积漂移。

**本文目标**：(1) 定义新的SPR任务范式来同时解决三个问题；(2) 建立大规模全景数据集；(3) 设计有效的SPR模型。

**切入角度**：将序列首帧定义为坐标原点，回归查询帧相对于首帧（而非前一帧）的位姿。这样坐标系与场景解耦（不是场景绝对坐标），模型学习的是帧间相对特征而非场景特定特征。同时因为每帧位姿都直接相对于首帧回归，不依赖前一帧的预测结果，消除了累积漂移。

**核心 idea**：以序列首帧为origin，利用序列中所有前驱帧的信息来回归任意查询帧相对于origin的位姿，同时使用全景图来最大化视觉信息和帧间重叠。

## 方法详解

### 整体框架

SPR-Mamba接受一个沿轨迹采集的全景图序列, I_2, ..., I_q$作为输入，输出查询帧$相对于首帧$的6DoF相机位姿$\mathbf{T}_q$。模型由DINO特征提取器（冻结）和两个互补分支（局部分支+全局分支）组成。

### 关键设计

1. **局部分支（Local Branch）**:

    - 功能：学习相邻帧间的逐帧相对位姿
    - 核心思路：计算相邻帧DINO特征的差异（$帧产生-1$个差异向量），通过多层线性层处理。训练时附加辅助平移和旋转头输出帧间相对位姿作为额外监督。辅助头在推理时可移除
    - 设计动机：局部分支提供精细的短程运动信息，弥补全局分支可能在长距离直接回归时丢失的细粒度位移信息

2. **全局分支（Global Branch）**:

    - 功能：学习查询帧到首帧的全局相对位姿
    - 核心思路：多个Mamba块堆叠处理整个序列的DINO特征。利用Mamba的SSM特性，最后一个Mamba块的最后hidden state聚合了从$到$的所有信息。选取该hidden state与局部分支的输出特征融合后，通过平移头和旋转头输出最终位姿
    - 设计动机：Mamba能顺序处理变长序列且推理时复杂度线性，适合持续接收新帧。全局分支直接回归相对于origin的位姿，避免了VO的累积漂移

3. **360SPR大规模数据集**:

    - 功能：为SPR任务提供大规模训练和评估数据
    - 核心思路：使用Habitat模拟器在270个室内场景中采集200K+全景图和3.6M针孔图像。模拟三种机器人高度（0.1m/0.5m/1.7m），轨迹长度3-20m，每个采样点18张针孔图缝合为全景图。三名检查员交叉验证质量，清洗耗时300+小时
    - 设计动机：现有全景定位数据集360Loc仅4个场景、不到10K张全景图，远不能满足鲁棒定位需求

### 损失函数 / 训练策略

训练损失包含全局分支的平移和旋转回归损失，以及局部分支辅助头的帧间位姿监督。DINO提取特征后冻结，仅训练Mamba块和各回归头。SPR-Mamba支持任意长度序列推理。

## 实验关键数据

### 主实验

360SPR数据集（未知场景）：

| 范式 | 方法 | TE中位(m)↓ | RE中位(°)↓ |
|------|------|-----------|-----------|
| APR | PoseNet | 30.25 | 47.15 |
| APR | Marepo | 27.98 | 48.12 |
| RPR | PanoPose | 10.91 | 20.01 |
| RPR | FAR | 11.85 | 21.04 |
| **SPR** | **SPR-Mamba** | **~3-4** | **~5-7** |

在未知场景上，SPR-Mamba相比APR和RPR平移误差降低7m+、旋转误差降低16°+。

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅全局分支 | 缺少局部细粒度信息，误差较高 |
| 仅局部分支 | 存在类VO累积效应，误差中等 |
| 双分支融合 | 局部+全局互补，误差最低 |
| 360Loc训练→测通用 | +4.08m/+4.67°，数据量和多样性不足 |

### 关键发现

- 全景图比窄视角图像显著降低定位误差——随FoV增大APR/RPR/SPR三种范式误差都下降
- 360SPR数据集的场景多样性至关重要
- 多高度训练提升鲁棒性——固定高度训练在其他高度性能明显下降
- SPR范式在开环轨迹上完全没有累积漂移

## 亮点与洞察

- **SPR任务定义本身是最大贡献**：用首帧做origin的思路简单但巧妙，一举解决APR/RPR/VO三种范式各自的核心问题
- **Mamba用于位姿回归**：SSM顺序处理特性让最后hidden state自然聚合全局信息，推理可持续接收新帧且复杂度线性
- **全景图的必要性有定量验证**：给出了FoV从小到大误差下降的完整曲线图

## 局限与展望

- 数据集基于模拟器，与真实世界存在domain gap
- SPR要求完整轨迹序列输入，不适用于单帧定位
- 首帧的选择对结果有影响但未深入讨论
- 室外大规模场景效果未知
- 与feature matching类方法未做对比

## 相关工作与启发

- **vs PoseNet(APR)**: PoseNet学场景特定特征，换场景失效；SPR学帧间相对特征，场景无关
- **vs RelPose-GNN(RPR)**: RPR需要检索数据库找参考图，SPR只需序列本身
- **vs DeepVO(VO)**: VO累积漂移；SPR每帧直接回归到origin

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 任务定义新颖且有效
- 实验充分度: ⭐⭐⭐⭐ 对比了多种基线，数据集规模大
- 写作质量: ⭐⭐⭐⭐⭐ 范式对比一目了然
- 价值: ⭐⭐⭐⭐ 数据集和范式对视觉定位社区有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](../../NeurIPS2025/others/revisiting_agnostic_boosting.md)
- [\[NeurIPS 2025\] MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](../../NeurIPS2025/others/metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)
- [\[CVPR 2025\] VinaBench: Benchmark for Faithful and Consistent Visual Narratives](vinabench_benchmark_for_faithful_and_consistent_visual_narratives.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
