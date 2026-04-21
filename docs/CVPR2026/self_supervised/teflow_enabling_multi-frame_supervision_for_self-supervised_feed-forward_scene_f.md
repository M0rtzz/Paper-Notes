---
title: >-
  [论文解读] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation
description: >-
  [CVPR 2026][自监督学习][场景流] 提出TeFlow——首个将多帧监督引入自监督前馈场景流估计的方法：通过时序集成策略构建运动候选池并基于共识投票聚合时序一致的监督信号，在Argoverse 2上Three-way EPE达3.57cm（媲美优化方法Floxels）同时保持实时推理（8s vs 24min），较SeFlow++提升22.3%。
tags:
  - CVPR 2026
  - 自监督学习
  - 场景流
  - 自监督
  - 多帧监督
  - 时序集成
  - 前馈网络
  - 点云
---

# TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation

**会议**: CVPR 2026  
**arXiv**: [2602.19053](https://arxiv.org/abs/2602.19053)  
**代码**: [github.com/KTH-RPL/OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow)  
**领域**: 自监督学习 / 自动驾驶  
**关键词**: 场景流, 自监督, 多帧监督, 时序集成, 前馈网络, 点云

## 一句话总结

提出TeFlow——首个将多帧监督引入自监督前馈场景流估计的方法：通过时序集成策略构建运动候选池并基于共识投票聚合时序一致的监督信号，在Argoverse 2上Three-way EPE达3.57cm（媲美优化方法Floxels）同时保持实时推理（8s vs 24min），较SeFlow++提升22.3%。

## 研究背景与动机

**领域现状**：场景流估计LiDAR点云中每个点的3D运动。现有自监督方法分为两类：(1) **优化方法**（NSFP、EulerFlow）——利用多帧长时约束优化场景特定模型，精度高但延迟极大（小时到天级别）；(2) **前馈方法**（SeFlow、ZeroFlow）——单次前向推理高效，但训练目标仅来源于两帧点对应，容易受遮挡、噪声、稀疏观测影响产生不稳定的监督信号。

**核心矛盾**：多帧监督有潜力提供更稳定的训练信号，但朴素地将两帧目标扩展到多帧是无效的——帧间点对应剧烈变化，产生不一致的信号。如论文Figure 1b所示，两帧监督信号方向在时间上剧烈抖动，即使真实运动平滑，两帧估计也因遮挡和噪声而剧烈波动。

**现有尝试的不足**：ZeroFlow通过知识蒸馏从慢速"教师"生成伪标签，但需7.2 GPU月计算量。SeFlow改进两帧损失函数但仍受限于两帧信号天花板。多帧架构（Flow4D、DeltaFlow）在监督学习中有效，但在自监督下仍受两帧目标约束。

**TeFlow的切入**：不是设计更好的两帧损失，而是**挖掘多帧间时序一致的运动线索**——构建候选运动池+共识投票→产生稳定的多帧监督信号，让前馈模型首次在自监督下充分利用多帧架构的时序建模能力。

## 方法详解

### 整体框架

TeFlow基于DeltaFlow多帧骨干网络，输入5帧LiDAR点云（经ego-motion对齐），预测残余流 $\mathcal{F}_{res}$。训练时，点云先分为静态/动态区域（由DUFOMap提供），动态点聚类为簇 $\mathcal{C}_j$（由HDBSCAN预计算）。对每个动态簇，TeFlow通过时序集成生成可靠的监督目标 $\bar{\mathbf{f}}_{\mathcal{C}_j}$，结合静态损失和几何一致性损失进行训练。

### 关键设计

1. **运动候选池构建 (Motion Candidate Generation)**
    - 功能：为每个动态簇生成多样化的运动假设，平衡稳定性与数据驱动的几何证据
    - 核心思路：候选池由两类来源组成——(a) **内部候选** $\hat{\mathbf{f}}_{\mathcal{C}_j}$：当前网络预测的簇内平均流，作为稳定锚点；(b) **外部候选** $\mathbf{f}^{t'}_{\mathcal{C}_j,k}$：从每个时间帧 $t'$ 中通过最近邻搜索找到Top-K个最大位移对应点，按时间间隔归一化：$\mathbf{f}^{t'}_{\mathcal{C}_j,k} = \frac{\mathcal{NN}(\mathbf{p}_k, \mathcal{P}_{t',d}) - \mathbf{p}_k}{t' - t}$
    - 设计动机：内部候选让训练保持稳定不漂移，外部候选从多帧几何中挖掘真实运动线索。选Top-K最大位移过滤噪声点。时间归一化确保不同时间间隔的候选可比较

2. **共识投票与流聚合 (Consensus Voting)**
    - 功能：从候选池中提取最可靠的运动估计，过滤不一致的噪声候选
    - 核心思路：构建共识矩阵 $\mathbf{M}_{ab} = \mathbf{1}[\cos(\mathbf{f}_a, \mathbf{f}_b) > \tau_{cos}]$ 衡量方向一致性，结合可靠性权重 $w_i = \gamma^{m_i}(1 + \|\mathbf{f}_i\|_2^2)$（时间衰减+位移幅度加权）。投票得分 $\mathbf{S} = \mathbf{M}\mathbf{w}$，选得分最高者为共识赢家 $a^\dagger$，最终监督目标为与赢家方向一致的所有候选的加权平均：$\bar{\mathbf{f}}_{\mathcal{C}_j} = \frac{\sum_b \mathbf{M}_{a^\dagger b} w_b \mathbf{f}_b}{\sum_b \mathbf{M}_{a^\dagger b} w_b}$
    - 设计动机：单一候选不可靠→投票聚合利用多数一致性；时间衰减 $\gamma=0.9$ 优先近帧；大位移候选更有信息量→加权后共识信号显著比两帧信号稳定（Figure 1b对比）

3. **动态簇损失 (Dynamic Cluster Loss)**
    - 功能：公平地监督不同尺寸的动态物体，避免大物体因包含更多点而主导训练
    - 核心思路：结合点级损失（所有点平均L2）和簇级损失（先簇内平均再跨簇平均）：$\mathcal{L}_{dcls} = \frac{1}{|\mathcal{P}_\mathcal{C}|}\sum_j\sum_{\mathbf{p}_i \in \mathcal{C}_j}\|\hat{\mathbf{f}}_i - \bar{\mathbf{f}}_{\mathcal{C}_j}\|^2_2 + \frac{1}{N_c}\sum_j(\frac{1}{|\mathcal{C}_j|}\sum_{\mathbf{p}_i \in \mathcal{C}_j}\|\hat{\mathbf{f}}_i - \bar{\mathbf{f}}_{\mathcal{C}_j}\|^2_2)$
    - 设计动机：仅点级损失→行人等小物体被淹没（Table 5验证PED类上涨53%）；仅簇级损失→大物体细粒度对齐不足（OTHER类误差暴增82%）→两者结合取最优

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{total} = \mathcal{L}_{dcls} + \mathcal{L}_{static} + \mathcal{L}_{geom}$

- $\mathcal{L}_{static}$：静态点残余流趋零
- $\mathcal{L}_{geom}$：多帧Chamfer距离确保warped点云与邻帧几何对齐
- 训练配置：Adam优化器，lr=0.002，batch size=20，10×RTX 3080，15 epoch，约15-20小时

## 实验关键数据

### 主实验：Argoverse 2测试集排行榜

| 方法 | 类型 | #帧 | 运行时间/seq | Three-way EPE↓ | Dynamic Norm↓ | PED↓ |
|------|------|:---:|:---:|:---:|:---:|:---:|
| NSFP | 优化 | 2 | 60m | 6.06 | 0.422 | 0.722 |
| EulerFlow | 优化 | all | 1440m | 4.23 | 0.130 | 0.195 |
| Floxels | 优化 | 13 | 24m | 3.57 | 0.154 | 0.195 |
| ZeroFlow | 前馈 | 3 | 5.4s | 4.94 | 0.439 | 0.808 |
| SeFlow | 前馈 | 2 | 7.2s | 4.86 | 0.309 | 0.464 |
| SeFlow++ | 前馈 | 3 | 10s | 4.40 | 0.264 | 0.367 |
| **TeFlow** | **前馈** | **5** | **8s** | **3.57** | **0.205** | **0.253** |

- TeFlow EPE 3.57cm = Floxels（优化方法），但速度150×（8s vs 24min）
- 动态指标较SeFlow++提升22.3%，行人类别降31%

### 消融实验：帧数与损失项

| 损失组合 | Dynamic Norm Mean↓ | CAR↓ | PED↓ | Three-way EPE↓ |
|---------|:---:|:---:|:---:|:---:|
| 仅 $\mathcal{L}_{geom}$ | 0.386 | 0.317 | 0.297 | 8.85 |
| $\mathcal{L}_{geom} + \mathcal{L}_{static}$ | 0.458 | 0.321 | 0.481 | 6.37 |
| $\mathcal{L}_{dcls}$ only | 0.303 | 0.254 | 0.285 | 8.53 |
| $\mathcal{L}_{static} + \mathcal{L}_{dcls}$ | 0.313 | 0.233 | 0.296 | 4.84 |
| **全部三项** | **0.265** | **0.198** | **0.295** | **4.43** |

| 帧数 | Dynamic Norm Mean↓ | Three-way EPE Mean↓ |
|:---:|:---:|:---:|
| 2 (SeFlow) | 0.408 | 6.35 |
| 2 (TeFlow) | 0.353 | 5.98 |
| 4 | 0.283 | 4.57 |
| **5** | **0.265** | **4.43** |
| 6 | 0.269 | 4.55 |
| 8 | 0.300 | 5.40 |

### 关键发现

- 即使同为2帧，TeFlow也比SeFlow优13.5%→候选池+簇级损失的贡献
- 5帧为最优窗口，6帧开始性能微降，8帧明显退化→过远帧引入噪声
- 仅动态簇损失在动态物体上很强（Mean 0.303）但静态区域EPE 8.53→必须结合静态损失
- 候选池消融：仅内部（0.455）< 仅外部（0.321）< 两者结合（0.265）→内部锚定+外部证据互补
- nuScenes上同样SOTA：Dynamic Norm 0.395 vs SeFlow++ 0.509，行人误差降33.8%

## 亮点与洞察

- **核心insight精准**：多帧自监督的关键困难不是架构而是监督信号质量→时序一致性挖掘是正确的解法
- **候选池+投票机制优雅**：将不可靠的多源运动估计聚合为可靠信号，无需额外网络或复杂优化
- **簇级损失简单有效**：用一行额外代码解决大小物体不平衡问题，对行人等小物体提升幅度巨大
- **效率-精度帕累托最优**：在实时方法中精度最高，在高精度方法中速度最快，成功弥合了两类方法的鸿沟

## 局限与展望

- 依赖外部模块进行静态/动态分割（DUFOMap）和动态聚类（HDBSCAN）→分割错误可能级联传播
- 超过5帧后性能下降→共识机制对远距帧的利用还不够精细
- 候选归一化假设线性运动→曲线运动（如转弯车辆）的候选质量受限
- 未探索将时序集成策略应用于模型推理阶段（当前仅用于训练）

## 相关工作与启发

- **vs EulerFlow**：EulerFlow优化连续ODE精度极高（EPE 4.23）但需1440分钟→TeFlow 3.57cm仅需8秒→实际部署中TeFlow是唯一可行的高精度方案
- **vs SeFlow/SeFlow++**：改进两帧损失但受限于两帧信号天花板→TeFlow从信号源头提升质量
- **vs ZeroFlow**：知识蒸馏需7.2 GPU月生成伪标签→TeFlow完全端到端自监督
- **启发**：共识投票的多帧信号挖掘范式可迁移到其他自监督视觉任务（光流、深度估计）；簇级损失的思路对所有涉及物体尺度不平衡的3D感知任务有参考价值

## 评分

⭐⭐⭐⭐⭐ (5/5)

综合评价：精准定位了自监督前馈场景流的核心瓶颈（多帧监督信号不稳定），提出简洁优雅的时序集成解决方案。实验全面（Argoverse 2 + nuScenes + Waymo），消融透彻（帧数/损失组合/候选池/损失形式），定量定性结果均令人信服。首次在自监督前馈方法中达到优化方法精度的同时保持实时效率——开创了新的帕累托前沿。

<!-- RELATED:START -->

## 相关论文

- [AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](../../CVPR2025/self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)
- [Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](../../ECCV2024/self_supervised/scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](../../ACL2025/self_supervised/shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)
- [Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)

<!-- RELATED:END -->
