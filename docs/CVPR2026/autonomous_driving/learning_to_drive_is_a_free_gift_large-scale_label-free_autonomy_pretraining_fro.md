---
title: >-
  [论文解读] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
description: >-
  [CVPR 2026][自动驾驶][自动驾驶预训练] 提出LFG（Learning to drive is a Free Gift），一个完全无标签、教师引导的自动驾驶预训练框架，从大规模无姿态YouTube驾驶视频中学习几何、语义和运动感知的统一伪4D表示，在NAVSIM基准上仅用单目前视相机即超越多相机+LiDAR的BEV方法（PDMS 85.2），并展示了出色的数据效率（10%标签即达81.4 PDMS）。
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "自动驾驶预训练"
  - "无标签学习"
  - "视频基础模型"
  - "4D场景理解"
  - "规划"
---

# Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos

**会议**: CVPR 2026  
**arXiv**: [2602.22091](https://arxiv.org/abs/2602.22091)  
**代码**: [项目页面](https://lfg-ai.github.io/)  
**领域**: 自动驾驶  
**关键词**: 自动驾驶预训练, 无标签学习, 视频基础模型, 4D场景理解, 规划

## 一句话总结

提出LFG（Learning to drive is a Free Gift），一个完全无标签、教师引导的自动驾驶预训练框架，从大规模无姿态YouTube驾驶视频中学习几何、语义和运动感知的统一伪4D表示，在NAVSIM基准上仅用单目前视相机即超越多相机+LiDAR的BEV方法（PDMS 85.2），并展示了出色的数据效率（10%标签即达81.4 PDMS）。

## 研究背景与动机

互联网上存在海量的第一人称驾驶视频（如YouTube），但这些数据没有任何标注——没有3D标注、没有相机姿态、没有LiDAR、没有语义分割标签。现有自动驾驶方法虽然在scaling up时性能提升，但仍**严重依赖标注数据**（专家轨迹、LiDAR扫描、里程计、语义标注）。

一个自然的问题是：**能否像GPT训练在无标注文本语料上一样，从大规模无标注驾驶视频中预训练出强大的自动驾驶表示？**

已有的自监督方法（如PPGeo）主要依赖帧间一致性损失（光度一致性等），隐式假设场景是静态的，无法捕获**动态物体**——而动态物体恰恰是驾驶中最关键的。同时，大型世界模型（如UniPAD、ViDAR）仍需一定程度的监督标签。

LFG的核心洞察是：**安全的反应式驾驶不仅需要理解当前场景的3D结构，更需要预测短期未来的几何、运动和语义演变**。因此LFG采用前馈式的"当前+未来"联合预测框架，利用多个专门的教师模型（π³、SegFormer、SAM2、CoTracker3）提供伪监督，在不需要任何真实标注的情况下学习统一的4D驱动表示。

## 方法详解

### 整体框架

LFG 想验证一件事：能不能像 GPT 吃无标注文本一样，从海量无标注、无姿态的 YouTube 驾驶视频里预训练出强大的自驾表示。它的核心洞察是——安全的反应式驾驶不只要懂当前场景的 3D 结构，更要预测短期未来的几何、运动和语义演变，所以 LFG 采用前馈式的"当前+未来"联合预测。架构建在 π³（前馈 3D 重建模型，约 10 亿参数）之上：π³ 编码器吃 $N$ 帧无姿态 RGB，输出潜在场景 token $\mathbf{Z}_{1:N}$；一个因果自回归 Transformer 从观测 token 外推出 $M$ 帧未来 token $\mathbf{Z}_{N+1:N+M}$；一个共享解码器再把全部 $N+M$ 帧 token 解码成 5 种输出模态——点云图 $P_t$（每像素的 3D 世界坐标）、相机姿态 $T_t \in \mathbb{R}^{4 \times 4}$（自车轨迹）、语义分割 $S_t \in \mathbb{R}^{7 \times H \times W}$（道路/车辆/行人/建筑/植被/天空/背景 7 类）、置信度图 $C_t \in [0,1]^{H \times W}$、运动掩码 $M_t \in [0,1]^{H \times W}$。模型总参数约 1.45B，在 RTX 5090 上以 5Hz 运行。监督则完全来自多个现成专家教师（π³、SegFormer、SAM2、CoTracker3），不需要任何真实标注。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["N 帧无姿态 YouTube RGB（只看前 N 帧）"] --> B["π³ 编码器<br/>输出潜在场景 token Z(1:N)"]
    B --> C["因果自回归 Transformer<br/>外推未来 M 帧 token Z(N+1:N+M)"]
    C --> D["共享解码器<br/>解码全部 N+M 帧"]
    D --> E["5 模态输出<br/>点云图 / 相机姿态 / 语义 / 置信度 / 运动掩码"]
    E --> F["规划表示（下游 NAVSIM 微调）"]

    T1["π³ 教师蒸馏<br/>全知教师看全部帧供几何+姿态伪监督"] -. 伪监督 .-> D
    T2["SegFormer 语义蒸馏<br/>Cityscapes 预训练供 7 类语义伪标签"] -. 伪监督 .-> D

    subgraph MM["运动掩码生成流水线"]
        direction TB
        H["Grounded SAM2<br/>检测首帧人/车实例"] --> I["CoTracker3<br/>跨帧跟踪 2D 轨迹"]
        I --> J["π³ 点云反投影<br/>算 3D 位移→阈值判动态→逐像素掩码"]
    end
    MM -. 伪监督 .-> D
```

### 关键设计

**1. π³ 教师蒸馏（几何+姿态）：用全知教师逼学生从部分观测外推未来**

要学会"预测未来场景结构"，得有个能看到未来的老师来出题。π³ 教师模型能看到全部 $N+M$ 帧，而 LFG 学生只看前 $N$ 帧——教师把所有帧的点云图、置信度图和相机姿态都当作伪监督喂下来，逼学生从部分观测里同时预测当前和未来的 3D 几何。几何项用 $\mathcal{L}_{\text{point}} = \alpha \|\mathbf{P} - \widehat{\mathbf{P}}\|_1$，姿态项用帧对之间的相对旋转（SO(3) 测地距离）加平移（Huber 损失）。这种"全知教师 + 受限学生"的设置，正是让 LFG 学会外推而非单纯重建当前帧的关键。

**2. SegFormer 语义蒸馏：白送一份场景语义理解**

驾驶决策离不开语义，但视频本身没有分割标签。LFG 拉来预训练 SegFormer（Cityscapes）当语义教师，为每帧生成伪标签；同样地，教师能看到所有帧（包括未来帧的真实 RGB），LFG 却只看前 $N$ 帧却要预测全部 $N+M$ 帧的语义分割，并用加权 BCE 损失处理类别不平衡。于是场景语义理解被"免费"蒸馏进了同一个学生网络。

**3. 运动掩码生成流水线：把动态物体从静态环境里解耦出来**

此前依赖帧间一致性（光度一致性等）的自监督方法隐含假设场景静态，恰恰丢掉了驾驶里最关键的动态物体。LFG 用一条完全无标注的自动化流水线显式标出动态区域：先用 Grounded SAM2 检测第一帧中的人和车辆实例，再用 CoTracker3 跨帧跟踪它们的 2D 轨迹 $\mathbf{u}_t^{(i)}$，借 π³ 教师的点云图把 2D 轨迹反投影到 3D，算每个实例的 3D 时序位移 $d_t^{(i)} = \|\bar{\mathbf{p}}_{t+1}^{(i)} - \bar{\mathbf{p}}_t^{(i)}\|_2$，位移超过阈值 $\tau_{\text{motion}}$ 且至少持续 $K_{\min}$ 帧的实例标为动态，最后把实例级运动指示转成逐像素运动掩码 $\mathbf{M}_t$。这一步直接补上了帧间一致性方法的核心缺陷。

**4. 因果自回归 Transformer：把"预测未来"做成 next-token prediction**

把未来预测形式化成序列建模，就能复用 NLP 那套范式。这个 4 层、8 头注意力、dropout 0.1 的 Transformer 接收编码器输出的场景 token，用因果注意力（只能看过去和当前 token、不能看未来）逐步预测未来帧的潜在表示，等于把"预测未来场景"变成了 next-token prediction 问题。消融显示移除这个自回归头后 10% 数据性能从 81.4 掉到 77.7，说明未来预测能力是规划的关键。

### 损失函数 / 训练策略

总损失为当前帧损失 + 加权的未来帧损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{current}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}$$

每部分包含4个子损失：$\lambda_{\text{seg}}\mathcal{L}_{\text{seg}} + \lambda_{\text{pose}}\mathcal{L}_{\text{pose}} + \lambda_{\text{point}}\mathcal{L}_{\text{point}} + \lambda_{\text{motion}}\mathcal{L}_{\text{motion}}$

- 未来帧权重 $\omega = 10.0$（强调外推能力）
- AdamW优化器，lr=1e-4，cosine退火
- BF16混精度训练，32×A100 GPU，40k iterations
- 3阶段训练：① 几何+姿态自回归 → ② 加入语义头 → ③ 加入运动头
- 训练数据：OpenDV YouTube驾驶数据集子集（约200万样本），多帧率训练（2/5/10 Hz）

## 实验关键数据

### 主实验 — NAVSIM规划基准

| 方法 | 输入 | NC | DAC | TTC | EP | PDMS |
|------|------|-----|-----|-----|-----|------|
| UniAD | 6Cam | 97.8 | 91.9 | 92.9 | 78.8 | 83.4 |
| TransFuser | 3Cam+LiDAR | 97.7 | 92.8 | 92.0 | 79.2 | 84.0 |
| Hydra-MDP | 3Cam+LiDAR | 96.9 | 94.0 | 94.0 | 78.7 | 84.7 |
| DiffusionDrive | 3Cam+LiDAR | 96.8 | 95.4 | 94.7 | 82.0 | 88.1 |
| **LFG (ours)** | **1Cam** | **98.2** | 93.7 | 94.4 | 79.1 | **85.2** |

### 数据效率对比（PDMS↑）

| 方法 | 输入 | 1%数据 | 10%数据 | 100%数据 |
|------|------|--------|---------|----------|
| PPGeo | 1Cam | 61.5 | 65.6 | 74.6 |
| π³ | 1Cam | 56.2 | 77.5 | 82.8 |
| DINOv3 | 1Cam | 60.0 | 75.8 | 81.4 |
| **LFG** | **1Cam** | **66.3** | **81.4** | **85.2** |

### 消融实验

| 配置 | 1%数据 | 10%数据 | 100%数据 | 说明 |
|------|--------|---------|----------|------|
| 完整LFG | 66.3 | 81.4 | 85.2 | 基准 |
| +2×预训练数据 | 76.6 | 82.3 | 84.8 | 低标签时提升巨大 |
| +更长预测horizon | 80.5 | 84.4 | 84.8 | 1%数据提升14+ |
| -语义/运动头 | 64.8 | 77.1 | 84.6 | 语义和运动监督重要 |
| -自回归头 | 66.3 | 77.7 | 84.2 | 未来预测能力关键 |

### 关键发现

- **单目前视 > 多相机+LiDAR**: LFG仅用1个前视相机在NAVSIM上即超越UniAD(6相机)和Hydra-MDP(3相机+LiDAR)——证明大规模视频预训练能弥补传感器配置的劣势
- **数据效率极高**: 10%标签即达81.4 PDMS，与DINOv3的100%数据性能持平
- **语义分割超越教师**: 在KITTI-360上LFG的分割质量（PA 0.947, mIoU 0.768）超过了提供伪标签的SegFormer教师（PA 0.926, mIoU 0.677）——说明多任务联合学习有互益效果
- **运动预测可自纠正**: LFG能正确识别静止停放的车辆，而伪GT运动标签错误地将其标为动态——学生模型表现优于教师标签

## 亮点与洞察

- **"免费的午餐"理念**：证明了YouTube驾驶视频可以作为强大的、免费的自动驾驶预训练数据源，不需要任何标注、姿态或LiDAR
- **多教师协同蒸馏**的范式很优雅：用不同专家模型的优势互补生成伪监督，避免了单一教师的局限
- **next-token prediction在驾驶场景中的应用**：将几何和运动的未来预测转化为token序列预测，与NLP中的范式统一
- 实验证明**短期未来预测**是驾驶规划的关键特征——移除自回归头后10%数据性能从81.4降至77.7

## 局限与展望

- 仅预测**短期未来**（3-6帧），长期推理能力有限，可扩展到多尺度时间horizon
- 仅使用**单前视相机**——虽然这是YouTube视频的特性，但多视角训练（如新的PhysicalAI数据集）可能进一步提升
- 教师模型（π³等）本身的质量上限了学生模型的性能天花板
- 运动掩码生成流水线依赖多个大模型（SAM2+CoTracker3+π³），计算成本高
- 1.45B参数、5Hz推理速度离实时部署仍有距离

## 相关工作与启发

- π³ (feedforward 3D重建) 提供了LFG的骨干，LFG的创新在于加入了**时序建模**和**多模态伪监督**
- 与ViDAR（未来点云预测作为预训练任务的先驱）相比，LFG不需要任何LiDAR数据
- PPGeo证明了几何先验对驾驶有帮助，但其帧间一致性假设（静态场景）是根本局限
- DINOv3/GPT的大规模无标注预训练范式在视觉-驾驶领域的成功验证

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 完全无标注的YouTube视频预训练→超越多传感器方法，范式很有突破性
- **实验充分度**: ⭐⭐⭐⭐ 多任务评估（分割/深度/轨迹/规划）、数据效率和消融都很充分
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、图表精美，整体narrative流畅
- **价值**: ⭐⭐⭐⭐⭐ 为自动驾驶预训练开辟了"internet-scale free data"的路线，影响力大

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SimScale: Learning to Drive via Real-World Simulation at Scale](simscale_learning_to_drive_via_real-world_simulation_at_scale.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](../../ICLR2026/autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)

</div>

<!-- RELATED:END -->
