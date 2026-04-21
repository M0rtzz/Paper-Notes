---
title: >-
  [论文解读] GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion
description: >-
  [CVPR 2026][机器人抓取] 提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。
tags:
  - CVPR 2026
  - 机器人抓取
  - 潜在扩散策略
  - 抓取先验
  - 模仿学习
  - 泛化
---

# GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion

**会议**: CVPR 2026  
**arXiv**: [2602.22862](https://arxiv.org/abs/2602.22862)  
**代码**: 有（Project Page）  
**领域**: 强化学习  
**关键词**: 机器人抓取, 潜在扩散策略, 抓取先验, 模仿学习, 泛化

## 一句话总结
提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。

## 研究背景与动机
在机器人操作流程中，抓取是实现物理交互的关键初始步骤。基于模仿学习的视觉运动策略（如 Diffusion Policy、ACT）在通用操作任务上展现了潜力，但在抓取子任务上往往不如专门的抓取检测方法，因为对整个抓取动作序列的建模本质上更复杂。

现有的将抓取先验整合到模仿学习中的方法（如 Robograsp、GPA-RAM）存在两个问题：(1) 仅将 grasp pose 作为条件输入拼接到策略模型中，导致 grasp pose 与输出动作序列的关联弱，难以提供有效指导；(2) 低语义的 grasp pose 与高维视觉输入之间存在模态不匹配，策略模型难以充分提取抓取空间分布信息。另一方面，数据驱动方法 GraspVLA 虽然性能强但需要 160 块 RTX 4090 训练 10 天生成 1B 帧模拟数据，成本极高。

核心 idea：借鉴图像生成中潜在扩散模型的成功经验，在动作潜空间中注入精确的目标 grasp pose 来引导动作生成，同时用 graspness map 作为视觉线索指导扩散过程，将静态的目标 grasp pose 和动态的动作序列桥接到共享潜空间。

## 方法详解

### 整体框架
GraspLDP 采用两阶段训练：(1) Action Latent Learning 阶段：用 VAE 将动作序列编码到紧凑潜空间，并在 decoder 中注入 grasp pose 先验引导动作重建；(2) Diffusion on Latent Action Space 阶段：在潜空间上训练扩散策略，用 graspness map 视觉线索调节去噪过程，并附加自监督重建目标强化线索利用。

### 关键设计

1. **Grasp Guidance in Latent Space（潜空间抓取引导）**:

    - 功能：用轻量 VAE 将动作序列压缩为紧凑潜特征，在 decoder 端拼接目标 grasp pose 进行重建
    - 核心思路：编码 $\mathbf{Z} = \mathcal{E}(A)$，解码 $\hat{A} = \mathcal{D}(\mathbf{Z} \oplus \mathcal{G})$，损失为 $\mathcal{L}_{VAE} = \text{MSE}(A, \hat{A}) + \lambda \mathcal{L}_{KL}$。扩散模型在紧凑的潜特征上去噪，而非直接在高维动作空间操作
    - 设计动机：将 grasp pose 直接作为条件会稀释引导强度并增加训练难度。通过在潜空间中注入，grasp pose 在 decoder 阶段对动作重建产生直接的、强约束的影响。同时潜空间维度更低，加速推理

2. **Visual Graspness Cue（视觉抓取性线索）**:

    - 功能：从预训练 graspness 网络获取点云的逐点抓取性得分，反投影到像素空间形成 graspness map，叠加到手腕相机图像上作为视觉线索
    - 核心思路：$O_{cue}(j,k) = \text{masked\_color}$ if $M(j,k) > \tau$，否则保留原始像素。同时在每个反向扩散步骤重建 $O_{cue}$ 作为自监督目标：$\mathcal{L}_{Recon.} = \text{MSE}(O_{cue}, \hat{O}_{cue})$，总损失 $\mathcal{L}_{LDP} = \mathcal{L}_{Diff.} + \lambda_{Recon.} \mathcal{L}_{Recon.}$
    - 设计动机：graspness map 是几何驱动的、光照不变的抓取可行性指标，能引导末端执行器朝向可抓取区域移动。自监督重建确保模型真正关注这些视觉线索而非忽略它们

3. **Heuristic Pose Selector (HPS)**:

    - 功能：推理时从抓取检测器预测的候选 grasp pose 中选择最合适的作为引导
    - 核心思路：先通过碰撞检测和 NMS 过滤，保留 top-k 质量候选。计算当前末端执行器位姿 $P$ 与候选 $\mathcal{G}_j$ 的 SE(3) 测地距离 $d_{\mathcal{G}_j, W} = \sqrt{\xi^\top W \xi}$，选择距离最小的候选 $\mathcal{G}^* = \arg\min d(\mathcal{G}_j)$
    - 设计动机：联合考虑抓取质量和运动学接近性，平衡抓取可行性和轨迹平滑性，避免不合理的 pose 引导降低成功率

### 损失函数 / 训练策略
两阶段：Stage 1 训练 VAE（MSE + KL），Stage 2 训练潜在扩散策略（扩散损失 + 自监督重建损失）。训练数据为 LIBERO benchmark 上约 12K 高质量抓取演示。

## 实验关键数据

### 主实验

| 方法 | In Domain | Spatial Gen. | Object Gen. | Visual Gen. | 平均 |
|------|-----------|-------------|-------------|-------------|------|
| Diffusion Policy | 62.8 | 48.9 | 11.4 | 16.3 | 34.9 |
| GraspVLA | 50.8 | 49.5 | 46.8 | 51.7 | 49.7 |
| Ours Baseline (CG) | 72.3 | 59.1 | 48.3 | 47.7 | 56.9 |
| **GraspLDP** | **80.3** | **71.1** | **58.2** | **64.6** | **68.6** |

### 消融实验

| 配置 | ID SR | SG SR | OG SR | VG SR |
|------|-------|-------|-------|-------|
| GraspLDP (full) | 80.3 | 71.1 | 58.2 | 64.6 |
| w/o Graspness Cue | 77.4 (-2.9) | 67.3 (-3.8) | 54.2 (-4.0) | 57.5 (-7.1) |
| w/o Latent Guidance w/ CG | 73.5 (-6.8) | 62.2 (-8.9) | 52.3 (-5.9) | 54.5 (-10.1) |
| w/o Latent Guidance | 60.6 (-19.7) | 49.8 (-21.3) | 21.2 (-37.0) | 19.4 (-45.2) |
| w/o GC & LG | 55.1 (-25.2) | 46.2 (-24.9) | 16.0 (-42.2) | 15.7 (-48.9) |

### 关键发现
- GraspLDP 在域内抓取成功率比 Diffusion Policy 提升 17.5%，空间/物体/视觉泛化分别提升 22.2%、46.8%、48.3%
- Latent Guidance 是最关键组件，移除后 OG/VG 暴跌至 ~20%，说明潜空间 grasp pose 引导对泛化至关重要
- Graspness Cue 在 Visual Generalization 上的提升最大（-7.1%），因为几何抓取性线索对光照变化具有鲁棒性
- HPS 比 random/highest/nearest 选择策略更优，联合考虑质量和运动学接近性是必要的
- 推理延迟仅比 Diffusion Policy 多 ~15%，远快于 GraspVLA

## 亮点与洞察
- 将图像生成中的潜在扩散思想迁移到机器人动作生成，在潜空间中注入先验是比条件拼接更有效的引导方式
- graspness map 作为视觉 prompt 的设计简洁有效，自监督重建确保信息利用
- 真实世界实验中，GraspLDP 在杂乱场景的 Scene Completion Rate 达到 96.2%，接近开环方法 AnyGrasp 的 92.3%

## 局限与展望
- 依赖预训练的抓取检测网络（如 AnyGrasp），若检测器在新物体上失效则 pose 先验变差
- 目前仅针对抓取子任务，未扩展到完整的长序列操作
- VAE 训练和扩散训练分为两阶段，端到端联合训练可能更优

## 相关工作与启发
- Diffusion Policy 提出了扩散模型用于动作生成的范式，GraspLDP 将其扩展到潜空间并注入任务先验
- GraspVLA 是数据驱动路线（1B 帧），GraspLDP 走先验注入路线，在更少数据下更高效
- 潜空间引导的思路可推广到其他需要先验知识的操作任务（如装配、工具使用）
- PPI 用离散关键位姿引导连续动作生成，GraspLDP 进一步在潜空间中实现更精细的引导
- GSNet 的 graspness 概念被复用为视觉 prompt，展示了抓取检测与策略学习的协同潜力

## 补充细节
- 真实世界实验中杂乱场景抓取：GraspLDP 的 SCR 达到 96.2%，4 个场景平均 SR 80%
- 推理时 graspness 计算仅需 36ms，latent decode 不到 1ms，整体延迟可控制在 ~100ms
- VAE 使用非对称解码器，encoder 不接受 grasp pose，decoder 接受——信息流设计确保 latent 编码动作本身、decoder 注入 pose 调制
- GFE（Grasp Frame Error）指标创新地基于 SE(3) 测地距离评估策略跟随 grasp pose 引导的精度
- 训练数据仅 12K 演示，远少于 GraspVLA 的 1B 帧，但泛化性能全面超越

## 评分
- 新颖性: ⭐⭐⭐⭐ 潜空间 grasp pose 注入和 graspness 视觉线索的自监督重建有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实世界、多维度泛化评估、详细消融、HPS 消融均完善
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 解决了策略抓取泛化的实际问题，OG +46.8% 的提升对实际部署意义重大

## 关键术语
- **Graspness**: 点云中每个点的可抓取性得分，几何驱动的抓取可行性度量
- **Latent Diffusion Policy**: 在 VAE 编码的动作潜空间上进行扩散去噪
- **SE(3) Geodesic Distance**: 特殊欧氏群上的测地距离，统一衡量旋转和平移差异

<!-- RELATED:START -->

## 相关论文

- [Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [Formal Verification of Diffusion Auctions](../../AAAI2026/reinforcement_learning/formal_verification_of_diffusion_auctions.md)
- [Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_with_multimodal_latent_replay_and_incremental_adjust.md)
- [Object-Centric Latent Action Learning](../../AAAI2026/reinforcement_learning/object-centric_latent_action_learning.md)
- [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](../../NeurIPS2025/reinforcement_learning/egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)

<!-- RELATED:END -->
