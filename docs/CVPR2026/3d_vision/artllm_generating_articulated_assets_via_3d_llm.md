---
title: >-
  [论文解读] ArtLLM: Generating Articulated Assets via 3D LLM
description: >-
  [CVPR 2026][3D视觉][Articulated Object] ArtLLM 将铰接物体生成建模为语言生成问题，使用 3D 多模态 LLM 从点云自回归预测部件布局和运动关节参数（离散化为 token），再结合 XPart 生成高保真部件几何，在 PartNet-Mobility 数据集上显著超越现有方法（mIoU 0.69, 推理仅需 19 秒）。
tags:
  - CVPR 2026
  - 3D视觉
  - Articulated Object
  - 3D LLM
  - URDF
  - Autoregressive
  - Part-Aware Generation
---

# ArtLLM: Generating Articulated Assets via 3D LLM

**会议**: CVPR 2026  
**arXiv**: [2603.01142](https://arxiv.org/abs/2603.01142)  
**代码**: [https://authoritywang.github.io/artllm](https://authoritywang.github.io/artllm) (有)  
**领域**: 3D 视觉 / 铰接物体生成  
**关键词**: Articulated Object, 3D LLM, URDF, Autoregressive, Part-Aware Generation

## 一句话总结
ArtLLM 将铰接物体生成建模为语言生成问题，使用 3D 多模态 LLM 从点云自回归预测部件布局和运动关节参数（离散化为 token），再结合 XPart 生成高保真部件几何，在 PartNet-Mobility 数据集上显著超越现有方法（mIoU 0.69, 推理仅需 19 秒）。

## 研究背景与动机

**领域现状**：交互式数字环境（游戏、机器人、仿真）依赖铰接 3D 物体，其功能来源于部件几何和运动结构。现有方法存在根本局限。

**两类方法的痛点**：
   - **优化重建方法**（PARIS, VideoArtGS, ArtGS）：需要逐物体慢速关节拟合，通常只能处理单关节简单物体
   - **检索装配方法**（SINGAPO, CAGE, URDFormer）：从固定部件库中检索拼装，几何重复性高、泛化差

**3D 生成的断层**：通用 3D 生成模型（Trellis, Hunyuan3D）已能生成高质量几何，部件级生成（XPart, OmniPart）也取得进展。但这些模型**不理解运动结构**——生成的部件不知道该怎么动，导致几何和运动脱节。

**切入角度**：需要一个**统一理解几何和铰接的方案**。LLM 天然适合处理变长的结构化序列，利用 3D LLM 的序列建模和推理能力来自回归预测铰接蓝图。

**核心 idea**：将 URDF 铰接结构离散化为 token 序列，训练 3D LLM 从点云自回归生成"部件布局 + 运动关节"的统一蓝图，再驱动部件生成模型合成几何。

## 方法详解

### 整体框架
三阶段流水线：
1. **ArtLLM**：输入点云 → 3D LLM → 预测部件 AABB + 关节参数的 token 序列
2. **部件几何合成**：预测的 bounding box → XPart 生成高保真部件 mesh
3. **物理关节限制修正**：碰撞检测 → 层级搜索精确限制角

### 关键设计

1. **铰接结构的语言建模**：

    - 每个部件用 AABB 参数化：$\text{BBox}(x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max})$
    - 关节定义包括类型、父子连接、轴方向、轴位置、运动范围
    - 支持 4 种关节：Revolute, Continuous, Prismatic, Screw
    - 生成顺序：先预测所有部件 bounding box，再预测所有关节定义

2. **连续参数离散化（Quantization）**：

    - **设计动机**：LLM 本质上预测离散 token，直接回归连续值数值不稳定
    - Bounding box 坐标：$[-1,1]$ 范围量化为 128 个 bin
    - 关节原点：128 个 bin；旋转角度范围：48 个 bin（$[-2\pi, 2\pi]$）；平移距离：64 个 bin
    - **关节轴方向**：构建 128 个条目的离散码本——先在 XY/YZ/XZ 平面均匀采样（覆盖轴对齐方向），再在 Fibonacci 球上用 FPS 补充
    - 这种层级码本设计对主轴对齐方向有更密集覆盖，同时保持其他方向的灵活性

3. **多任务多阶段 SFT**：

    - 三个任务：仅部件布局预测 / 给定布局预测关节 / 端到端预测
    - **两阶段训练**：
        - 阶段一：仅训练部件布局预测（用 P3SAM 预训练权重初始化点云编码器）
        - 阶段二：在三个任务上联合 SFT
    - **设计动机**：先建立部件级几何理解的基础，再在此基础上学习运动推理

4. **物理约束关节限制修正**：

    - 单时间步的几何预测无法感知运动动态，可能导致部件碰撞
    - 对旋转关节：在预测范围内铰接子部件，计算与其他静态部件的碰撞体积
    - 碰撞体积导数出现尖峰的角度即为碰撞点 → 层级搜索精确角度 → 设为修正后的限制
    - 类似处理平移关节

### 损失函数 / 训练策略
- 标准交叉熵损失用于 SFT
- 多任务数据混合比 3:2:5
- 余弦学习率调度，最大 1e-5，warmup 0.03
- 数据增强：75% 概率随机缩放（$s \in [0.8, 1.05]$）和旋转（90°/180°/270°）
- 阶段一：8×H20 GPU，50 epoch（~8h）；阶段二：8×H20 GPU，30 epoch（~15h）
- 3D 编码器：Point Transformer v3；LLM 骨干：Qwen3 0.6B

## 实验关键数据

### 主实验（PartNet-Mobility，7 类 77 个物体）
| 方法 | mIoU↑ | CD↓ | Type Acc↑ | Joint-Axis-Err↓ | Joint-Pivot-Err↓ | Range-IoU↑ | Graph Acc↑ | Time(s) |
|------|-------|-----|-----------|-----------------|-------------------|------------|------------|---------|
| URDFormer | 0.123 | 0.249 | 0.607 | 0.738 | 0.610 | 0.703 | 0.079 | 183 |
| SINGAPO | 0.433 | 0.044 | 0.765 | 0.245 | 0.257 | 0.526 | 0.456 | 84 |
| ArtAny | 0.338 | 0.072 | 0.846 | 0.453 | 0.536 | **0.865** | 0.614 | 522 |
| **ArtLLM** | **0.688** | **0.028** | **0.908** | **0.127** | **0.080** | 0.740 | **0.774** | **19** |

### 消融实验
| 配置 | IoU | Type Acc | Axis Err | Pivot Err | Range IoU | Graph Acc |
|------|-----|----------|----------|-----------|-----------|-----------|
| Full | **0.473** | **0.898** | **0.141** | **0.135** | **0.582** | **0.780** |
| A: 无离散化 | 0.352 | 0.823 | 0.277 | 0.235 | 0.575 | 0.775 |
| B: 无多任务 | 0.464 | 0.825 | 0.289 | 0.131 | 0.510 | 0.737 |
| C: 无数据增强 | 0.412 | 0.894 | 0.142 | 0.138 | 0.577 | 0.754 |
| D: 无多阶段 | 0.463 | 0.890 | 0.143 | 0.175 | 0.511 | 0.780 |

### 关键发现
- ArtLLM 在推理速度上快一个数量级（19s vs 84-522s），适合大规模仿真环境
- 离散化（A）对坐标和方向相关属性的影响最大（IoU 0.352 vs 0.473）
- 多任务学习（B）提升了轴方向以外的所有指标，说明不同难度任务的共同训练有互补效果
- 物理限制修正有效消除了自碰撞（定性结果），且不影响推理速度
- Real2Sim 应用成功：重建的铰接资产在 SAPIEN 仿真器中复现了真实机器人操作行为

## 亮点与洞察
- **铰接 = 语言**：将 URDF 格式的运动结构自然映射为 token 序列，充分利用 LLM 的序列建模优势
- **离散化策略的精心设计**：关节轴方向的层级码本、不同物理量的不同量化精度，体现了对问题结构的深刻理解
- **多任务多阶段训练**：简单有效地解耦了几何理解和运动推理
- **端到端实用价值**：从图像/文本到可仿真的铰接资产的完整流水线

## 局限与展望
- 训练数据的物体类别仍有限（43 类），对车辆、机器人等复杂类别泛化不足
- 未建模物理属性（质量、摩擦系数等），可作为未来扩展
- 关节限制修正是后处理步骤，理想情况应在生成过程中感知碰撞
- 依赖 XPart 进行部件生成，bounding box 预测不准时可能截断几何

## 相关工作与启发
- SINGAPO 和 URDFormer 是直接竞品，均基于固定部件库，ArtLLM 通过生成彻底摆脱此限制
- 与 SpatialLM 类似的 3D LLM 编码器-Projector 架构
- 离散化 + 自回归的思路可推广到其他结构化 3D 预测任务（如场景图生成、装配规划）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用 3D LLM 端到端生成多关节铰接资产，范式创新
- 实验充分度: ⭐⭐⭐⭐ 定量比较充分，消融完整，有 Real2Sim 验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，离散化设计描述详细
- 价值: ⭐⭐⭐⭐⭐ 对机器人学习和仿真有直接且显著的应用价值

<!-- RELATED:START -->

## 相关论文

- [3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience](3drawagent_teaching_llm_to_draw_in_3d_with_early_contrastive_experience.md)
- [ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [MeshArt: Generating Articulated Meshes with Structure-Guided Transformers](../../CVPR2025/3d_vision/meshart_generating_articulated_meshes_with_structure-guided_transformers.md)
- [FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)
- [Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks](mind_the_hitch_dynamic_calibration_and_articulated_perception_for_autonomous_tru.md)

<!-- RELATED:END -->
