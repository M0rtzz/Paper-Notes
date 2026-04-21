---
title: >-
  [论文解读] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories
description: >-
  [AAAI 2026][交互视频生成] 提出 Mask2IV，一个两阶段解耦框架——先预测交互者和物体的 mask 运动轨迹，再基于轨迹生成视频——实现了无需密集 mask 标注的、以交互为中心的可控视频生成，支持人-物交互和机器人操作两个场景。
tags:
  - AAAI 2026
  - 交互视频生成
  - mask轨迹
  - 人物交互
  - 机器人操作
  - 两阶段扩散
---

# Mask2IV: Interaction-Centric Video Generation via Mask Trajectories

**会议**: AAAI 2026  
**arXiv**: [2510.03135](https://arxiv.org/abs/2510.03135)  
**代码**: [项目页面](https://reagan1311.github.io/mask2iv)  
**领域**: 视频理解 / 视频生成  
**关键词**: 交互视频生成, mask轨迹, 人物交互, 机器人操作, 两阶段扩散

## 一句话总结

提出 Mask2IV，一个两阶段解耦框架——先预测交互者和物体的 mask 运动轨迹，再基于轨迹生成视频——实现了无需密集 mask 标注的、以交互为中心的可控视频生成，支持人-物交互和机器人操作两个场景。

## 研究背景与动机

### 领域现状

扩散模型在视频生成领域取得了重大进展，能从文本或图像提示生成高质量视频。在具身智能（Embodied AI）领域，生成逼真的人-物交互或机器人-物交互视频序列具有重要价值，可为模仿学习、可供性学习等下游任务提供视觉先验。

### 现有痛点

**文本条件控制不精确**：现有以文本为条件的方法（如 EgoVid、LEGO）缺乏对交互细节的精细控制——无法指定与哪个物体交互、手部应在哪个位置

**Mask 条件方法有两大致命缺陷**：
   - **实用性差**：如 InterDyn 需要用户提供**逐帧的密集手部 mask 序列**作为控制信号。获取这些 mask 需要先录制或合成想要控制的交互视频——形成了"先有鸡还是先有蛋"的悖论
   - **只关注手部**：仅用手部 mask 限制了交互建模范围，无法精确指定目标物体或捕捉手-物接触的细粒度信息

**缺少统一框架**：人-物交互和机器人操作通常作为独立问题研究，没有统一的解决方案

### 核心矛盾

Mask 是有效的交互控制信号（形状明确、运动可追踪），但**获取密集 mask 标注的成本与视频生成本身一样高**。如何在利用 mask 优势的同时消除其标注依赖，是核心技术矛盾。

### 本文切入角度

**解耦轨迹预测和视频生成**：第一阶段自动预测交互轨迹（mask 序列），第二阶段基于预测的轨迹生成视频。用户只需提供初始图像、目标物体 mask 和文本/位置条件，无需任何密集标注。

## 方法详解

### 整体框架

Mask2IV 将交互视频生成分解为两个阶段：

**Stage 1：交互轨迹生成**（Interaction Trajectory Generation）
- 输入：初始帧 $I$、物体 mask $M$、条件信号（文本 $T$ 或目标位置 mask $P$）
- 输出：mask 轨迹序列 $S \in \mathbb{R}^{N \times H \times W \times 3}$

**Stage 2：轨迹条件视频生成**（Trajectory-conditioned Video Generation）  
- 输入：初始帧 $I$、预测的 mask 轨迹 $S$
- 输出：交互视频 $V \in \mathbb{R}^{N \times H \times W \times 3}$

### 关键设计

#### 1. 交互轨迹生成（Stage 1）

**功能**：预测交互者（手/机械臂）和物体的联合运动轨迹。

**核心思路**：
- 将初始帧 $I$ 和物体 mask $M$ 通过 VAE 编码器编码为潜空间特征
- 物体 mask 先经**颜色编码**转为 RGB 格式（因 VAE 需要三通道输入）
- 如果初始帧中包含交互者（手/机械臂），使用 GroundedSAM 分割并分配不同颜色，使模型能区分角色
- 编码后的潜特征拼接到噪声潜变量，输入视频扩散模型
- **冻结时间注意力层**以保留运动先验，微调其余参数

**两种条件变体**：

**(a) 文本条件轨迹生成（TT-Gen）**：
- 文本提示通过 CLIP 编码，经交叉注意力注入模型
- 可区分细微交互意图：如"拿起"vs"放下"，"推"vs"拉"

**(b) 位置条件轨迹生成（PT-Gen）**：
- 目标位置 mask $P$ 编码后插入最后一帧的槽位
- 初始物体 mask 潜变量分配给第一帧
- 中间帧填充零值，模型自动插值生成连贯轨迹

**设计动机**：将问题简化为先预测 mask 运动——模型只需关注运动动力学，不需处理外观细节。这大幅降低了直接生成复杂交互视频的难度。

#### 2. 轨迹条件视频生成（Stage 2）

**功能**：基于预测的 mask 轨迹合成最终视频。

**核心思路**：
- 轨迹 $S$ 通过 VAE 编码为特征张量 $f_s$
- $f_s$ 与噪声潜变量和首帧特征拼接后输入扩散模型

**两个针对性设计**：

**(a) 随机扰动增强鲁棒性**：
- 训练时以概率 $p=0.2$ 对 mask 进行膨胀或腐蚀（核大小随机从 {3,5,7} 选择）
- 使模型不过度依赖 mask 的精确形状，提高泛化能力

**(b) 接触区域加权损失**：
- 定义接触图 $m_c = (\delta(m_h) \cap m_o) \cup (m_h \cap \delta(m_o))$，其中 $\delta(\cdot)$ 是膨胀操作
- 用接触图重新加权扩散目标函数：

$$w = (1 - m_c) + \lambda \cdot m_c$$

$$\mathcal{L} = \mathbb{E}_{z,S,\epsilon,t}[\|w \odot (\epsilon - \epsilon_\theta(z, f_\psi(S), t))\|_2^2]$$

其中 $\lambda=5$，使接触区域的损失权重为非接触区域的 5 倍。

**设计动机**：
- 随机扰动解决训练（真实 mask）与推理（预测 mask）之间的分布差异
- 接触加权解决手-物边界区域合成最困难的问题——这恰好是交互的核心地带

#### 3. 基准构建

**功能**：构建带逐帧分割标注的训练/评估基准。

- **HOI4D**（人-物交互）：剪裁带时间戳的视频片段，基于手和物体位移的运动分数过滤低动态视频（去除后 5%），使用文本模板 "a hand {verbing} an {object}"
- **BridgeData V2**（机器人操作）：用 GroundingDINO 检测物体 + SAM2 视频分割提取机械臂和物体 mask；通过时间帧间 mIoU 低的物体识别为被操作物体（因其位置和形状在变化）

### 损失函数 / 训练策略

- 基于 DynamiCrafter 构建，增加额外卷积通道支持 mask 潜变量
- 16 帧、320×512 分辨率
- AdamW 优化器，学习率 1e-5，batch size 8
- 推理使用 DDIM 采样器，50 步去噪
- 接触权重 $\lambda = 5$

## 实验关键数据

### 主实验

| 方法 | 会议 | FVD↓ (HOI/Robot) | LPIPS↓ | PSNR↑ | SSIM↑ | V2V-Sim↑ | T2V-Sim↑ |
|------|------|-----------------|--------|-------|-------|----------|----------|
| DynamiCrafter | ECCV24 | 554/861 | 0.516/0.375 | 13.48/14.21 | 0.553/0.571 | 0.473/0.867 | 0.146/0.215 |
| DynamiCrafter-ft | ECCV24 | 169/198 | 0.206/0.166 | 20.49/19.80 | 0.721/0.775 | 0.814/0.957 | 0.199/0.223 |
| CosHand | ECCV24 | 163/175 | 0.209/0.123 | 20.67/21.81 | 0.725/0.809 | 0.837/0.969 | 0.191/0.220 |
| InterDyn | CVPR25 | 172/208 | 0.207/0.145 | 20.71/21.16 | 0.730/0.802 | 0.794/0.955 | 0.172/0.219 |
| **Mask2IV** | **本文** | **150/156** | **0.178/0.111** | **21.48/22.30** | **0.741/0.815** | **0.847/0.971** | **0.200/0.220** |

- HOI4D 上 FVD 降低 8.7%（150 vs 163），BridgeData V2 上降低 10.9%（156 vs 175）
- 所有指标上均优于所有基线

### 消融实验

| 配置 | FVD↓ | LPIPS↓ | PSNR↑ | SSIM↑ | 说明 |
|------|------|--------|-------|-------|------|
| ControlNet | 157.38 | 0.182 | 21.49 | 0.747 | 辅助网络方案 |
| MaskLatent | 130.07 | 0.157 | 22.33 | 0.760 | 直接拼接潜变量，更优 |
| +object mask | 115.14 | 0.132 | 23.85 | 0.802 | 加入物体轨迹，大幅提升 |
| +random d/e | 108.80 | 0.124 | 24.16 | 0.802 | 随机膨胀/腐蚀增强 |
| +contact loss | **104.61** | **0.126** | **24.37** | **0.804** | 接触加权损失 |

（消融在 HOI4D 上使用真实 mask 轨迹进行）

### 关键发现

1. **直接拼接 mask latent 优于 ControlNet**：训练更稳定，早期收敛更快
2. **物体轨迹的加入带来最大提升**：FVD 从 130→115（-11.5%），证明仅建模手部运动不够，需要联合建模手-物运动
3. **随机扰动确实提升鲁棒性**：FVD 从 115→109
4. **接触加权损失进一步改善质量**：FVD 从 109→105
5. **灵活的物体指定能力**：同一场景通过修改 mask 可生成与不同物体的交互
6. **文本和位置条件互补**：文本适合描述动作类型，位置适合精确空间控制

## 亮点与洞察

1. **解耦设计的双重优势**：降低生成难度 + 提供更灵活的控制（用户可以修改预测的轨迹）
2. **接触区域加权损失的创新**：精准聚焦交互最关键的区域（手-物接触边界），利用 mask 的几何信息定义加权图
3. **统一框架覆盖人和机器人**：同一方法处理两种交互场景，仅条件类型不同（文本 vs 位置）
4. **物体识别的巧妙方法**：在 BridgeData V2 中用时间帧间 mIoU 低来识别被操作物体，无需额外标注
5. **颜色编码的设计细节**：将 mask 转为不同颜色的 RGB 图像输入 VAE，使模型能区分交互者和物体角色

## 局限与展望

1. **分辨率受限**：320×512 较低，高分辨率下计算开销会显著增加
2. **仅 16 帧**：短时交互可以，但长序列操作（如多步组装）难以覆盖
3. **两阶段推理的误差累积**：第一阶段预测的轨迹质量直接影响最终视频质量
4. **不支持多步交互**：如"先拿起再放下"这类连续动作未被建模
5. **缺少人体评估**：完全依赖自动指标，无人工主观评价视频质量
6. **数据集规模**：HOI4D 动作类别有限（主要是抓取），更多样的交互类型待探索

## 相关工作与启发

- **Mask 作为中间表示的优势**：比光流更语义化，比边界框更精确，是控制交互生成的理想信号
- **轨迹预测与视频生成的解耦**：思路可迁移到其他控制信号（如骨架、关键点）
- **接触图的定义方法**：基于 mask 的膨胀交集，简单但有效地定义了接触区域，可用于其他手-物交互任务
- **与机器人学的结合**：生成的交互视频可直接用于训练视觉运动策略，是 sim-to-real 之外的另一条数据获取路径

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 两阶段解耦和接触加权损失有新意，但大框架基于已有扩散模型
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集、完整消融，但缺少人工评估
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机和方法阐述极为清晰，图表设计精美
- **价值**: ⭐⭐⭐⭐ — 对具身智能的数据生成有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](../../CVPR2025/video_generation/through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](../../CVPR2026/video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)
- [U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](../../CVPR2026/video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](../../CVPR2025/video_generation/hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)

<!-- RELATED:END -->
