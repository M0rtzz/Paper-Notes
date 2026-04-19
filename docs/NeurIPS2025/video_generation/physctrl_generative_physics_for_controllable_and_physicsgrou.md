---
title: >-
  [论文解读] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation
description: >-
  [NeurIPS 2025][图像生成][物理驱动视频生成] PhysCtrl用扩散模型学习四种材料（弹性/沙/橡皮泥/刚体）的物理动力学分布，将动态表示为3D点轨迹，在55万合成动画上训练含时空注意力+物理约束的扩散模型，生成的轨迹驱动预训练视频模型实现力和材料参数可控的高保真物理视频生成。
tags:
  - NeurIPS 2025
  - 图像生成
  - 物理驱动视频生成
  - 扩散模型
  - 3D点轨迹
  - 材料仿真
  - 力控制
---

# PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation

**会议**: NeurIPS 2025  
**arXiv**: [2509.20358](https://arxiv.org/abs/2509.20358)  
**代码**: [项目页](https://cwchenwang.github.io/physctrl)  
**领域**: 图像/视频生成  
**关键词**: 物理驱动视频生成, 扩散模型, 3D点轨迹, 材料仿真, 力控制

## 一句话总结

PhysCtrl用扩散模型学习四种材料（弹性/沙/橡皮泥/刚体）的物理动力学分布，将动态表示为3D点轨迹，在55万合成动画上训练含时空注意力+物理约束的扩散模型，生成的轨迹驱动预训练视频模型实现力和材料参数可控的高保真物理视频生成。

## 研究背景与动机

**领域现状**：现代视频生成模型能产出照片级视频，但缺乏物理合理性和3D可控性。

**现有痛点**：传统物理仿真器（MPM等）计算成本高、超参敏感、数值不稳定；直接将仿真器与视频模型结合需要手动调参且可能切换不同仿真器。

**核心矛盾**：如何既保持物理合理性又避免传统仿真器的局限？

**本文目标** 将物理先验嵌入扩散模型，支持快速正向/逆向推理，以物理参数和外力作为控制信号。

**切入角度**：回答两个基本问题——什么表示适合控制视频模型？→3D点轨迹；如何嵌入多材料物理先验？→时空注意力扩散模型+物理约束。

**核心 idea**：用扩散模型学习物理动力学潜在分布，以3D点轨迹为桥梁连接物理世界和视频生成。

## 方法详解

### 整体框架

单张图片→SAM分割+SV3D多视角+LGM重建3D点云→条件扩散模型生成3D点轨迹→投影2D作为tracking video→DaS视频模型生成最终视频。

### 关键设计

1. **时空注意力扩散模型**:
    - 功能：学习四种材料的物理动力学分布 $p(\mathcal{P}|c)$
    - 核心思路：2048点×24帧轨迹，条件含初始点云/力/作用点/杨氏模量/泊松比/地面高度/材料类型。Spatial-Temporal Attention Block：空间注意力（同帧内点间self-attention，注入物理条件token用AdaLN）→时间注意力（同一点跨帧self-attention）
    - 设计动机：模拟粒子动力学——先整合邻近粒子信息再时间传播，反映MPM的P2G/G2P循环

2. **物理约束训练损失**:
    - 功能：将MPM的变形梯度更新公式作为显式物理约束
    - 核心思路：$\mathcal{L}_{phys}$ 约束 $\mathbf{F}_p^{f+1} \approx g(\hat{\mathbf{x}}_p^f)\mathbf{F}_p^f$（变形梯度一致性），加上 $\mathcal{L}_{floor}$ 防穿透
    - 设计动机：物理损失作为正则化确保轨迹物理合理性

3. **大规模合成数据集**:
    - 功能：55万动画覆盖四种材料（150K弹性+各100K沙/橡皮泥/刚体/重力）
    - 核心思路：ObjaverseXL高质量3D对象+MPM/刚体仿真器，2048点×24帧
    - 设计动机：多样化数据是学习物理分布的基础

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{diff} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{phys}\mathcal{L}_{phys} + \lambda_{floor}\mathcal{L}_{floor}$。Base 6层256维/Large 12层512维，AdamW lr=1e-4。DDIM 25步约1-3秒，4步约0.13-0.48秒。

## 实验关键数据

### 主实验
| 方法 | SA↑ | PC↑ | VQ↑ |
|------|-----|-----|-----|
| DragAnything | 2.9 | 2.8 | 2.8 |
| ObjCtrl-2.5D | 1.5 | 1.3 | 1.4 |
| Wan2.1 | 3.8 | 3.7 | 3.6 |
| CogVideoX | 3.2 | 3.2 | 3.1 |
| **PhysCtrl** | **4.5** | **4.5** | **4.3** |

| 方法 | vIoU↑ | CD↓ | Corr↓ |
|------|-------|-----|-------|
| Motion2VecSets | 24.92% | 0.2160 | 0.1064 |
| MDM | 53.78% | 0.0159 | 0.0240 |
| **Ours** | **77.59%** | **0.0028** | **0.0015** |

### 消融实验
| 配置 | vIoU↑ | CD↓ | 说明 |
|------|-------|-----|------|
| w/o 空间注意力 | 33.76% | 0.2348 | 空间交互至关重要 |
| w/o 时间注意力 | 53.63% | 0.0480 | 时间一致性核心 |
| w/o 物理损失 | 76.30% | 0.0030 | 物理约束进一步提升 |
| 完整模型 | **77.59%** | **0.0028** | - |

### 关键发现
- 用户研究：物理合理性81%偏好率远超所有baseline
- 空间注意力移除后vIoU断崖下降（77.59%→33.76%）
- 泊松比ν对生成轨迹影响可忽略（与PhysDreamer一致）
- 仅4个扩散步即可获得高质量轨迹

## 亮点与洞察
- 将物理仿真参数化为条件生成问题，避免传统仿真器限制
- 3D点轨迹作为中间表示——既灵活通用又能直接控制视频模型
- 时空注意力设计优雅地反映物理仿真计算结构
- 支持物理参数反演（inverse problem），仅需2分钟估计杨氏模量

## 局限与展望
- 主要处理单物体，多物体交互仅有初步实验
- 仅四种材料，不含流体
- 视频模型先验可能与物理轨迹冲突
- 薄结构处理能力不足

## 相关工作与启发
- **vs PhysGaussian**: 场景特定需高质量3D重建；PhysCtrl学习通用先验
- **vs PhysGen/PhysMotion**: 依赖仿真器生成动力学；PhysCtrl将先验嵌入扩散模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次以扩散模型学习多材料物理动力学分布
- 实验充分度: ⭐⭐⭐⭐ GPT-4o评估+用户研究+消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰
- 价值: ⭐⭐⭐⭐ 物理可控视频生成重要一步

<!-- RELATED:START -->

## 相关论文

- [Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)
- [PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation](../../ECCV2024/video_generation/physdreamer_physics-based_interaction_with_3d_objects_via_video_generation.md)
- [FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](../../ICCV2025/video_generation/fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](../../CVPR2026/video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)

<!-- RELATED:END -->
