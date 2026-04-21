---
title: >-
  [论文解读] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning
description: >-
  [CVPR 2026][目标检测][video customization] 提出 DreamVideo-Omni，在单一 DiT 架构中统一多主体身份定制和全运动控制（全局 bbox + 局部轨迹 + 相机运动），通过条件感知 3D RoPE、Group/Role Embeddings 解决多主体歧义，并设计潜空间身份奖励反馈学习（LIReFL）在任意去噪步提供密集身份奖励，绕过 VAE 解码器实现高效身份强化。
tags:
  - CVPR 2026
  - 目标检测
  - video customization
  - multi-subject
  - motion control
  - identity preservation
  - reward learning
---

# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**代码**: [https://dreamvideo-omni.github.io](https://dreamvideo-omni.github.io/)  
**领域**: 视频生成 / 可控生成  
**关键词**: video customization, multi-subject, motion control, identity preservation, reward learning

## 一句话总结

提出 DreamVideo-Omni，在单一 DiT 架构中统一多主体身份定制和全运动控制（全局 bbox + 局部轨迹 + 相机运动），通过条件感知 3D RoPE、Group/Role Embeddings 解决多主体歧义，并设计潜空间身份奖励反馈学习（LIReFL）在任意去噪步提供密集身份奖励，绕过 VAE 解码器实现高效身份强化。

## 研究背景与动机

**领域现状**：大规模扩散模型已能生成高保真视频，但精确控制多主体身份和多粒度运动仍是开放挑战。现有方法分成两个独立方向：主体定制方法（如 ConsisID、Phantom）保持外观但运动不可控；运动控制方法（如 Wan-Move、Tora）精确控制运动但无法指定主体外观。

**现有痛点**：少数尝试统一的方法（DreamVideo-2、VACE）面临三大困境：(1) 运动控制粒度有限——大多只支持单一信号（bbox 或 depth 或稀疏轨迹），无法同时覆盖全局位置、局部动态和相机运动；(2) 多主体控制歧义——所有条件信号被不加区分地注入，模型不知道哪个运动属于哪个主体；(3) 身份退化——运动控制要求像素动态变化，而身份保持要求与静态参考一致，标准扩散重建损失无法调和这一矛盾。

**核心矛盾**：身份保持要求"像参考图像"（像素一致性），运动控制要求"与参考不同"（时序变化），这两个目标从梯度方向上就是矛盾的。低级别重建损失无法表达"身份一致但姿态不同"这种高级语义。

**切入角度**：(1) 用显式绑定机制（Group/Role Embeddings）解决多主体歧义；(2) 将身份保持从低级像素一致性提升到与人类偏好对齐的高级语义一致性，通过训练专门的身份奖励模型来提供梯度信号。

**核心 idea**：通过 Group/Role Embeddings 显式绑定运动信号与主体身份，并训练基于 VDM 的潜空间身份奖励模型在去噪过程中任意时间步提供密集身份反馈。

## 方法详解

### 整体框架

基于 Wan2.1-1.3B T2V DiT，渐进式两阶段训练：Stage 1（Omni-Motion & Identity SFT）集成参考图像、全局 bbox、局部轨迹三元组进行联合训练；Stage 2（Latent Identity Reward Feedback Learning）训练潜空间身份奖励模型 LIRM，在去噪过程中直接提供身份保持的奖励信号。训练数据为自建的 2.12M 视频数据集，评估在新建的 DreamOmni Bench（1,027 视频）上进行。

### 关键设计

1. **条件感知 3D RoPE + Group/Role Embeddings**:

    - 功能：处理异质输入（视频帧、参考图像、轨迹）并解决多主体控制歧义
    - 核心思路：3D RoPE 的空间维度保持标准索引，但时间维度按输入类型分配：视频帧用连续索引 $t \in [0, T-1]$（标记为序列），参考图像用共享固定索引 $t_{ref}$（标记为静态条件），padding 用无效索引 $t_{pad}$，轨迹与视频帧共享索引（保持时空对齐）。Group embedding 为每个 $\langle$参考, bbox, 轨迹$\rangle$ 三元组分配唯一标识，确保运动信号与正确主体绑定；Role embedding 区分"外观资产"（object embedding）和"控制信号"（control embedding）
    - 设计动机：去掉条件感知 RoPE 直接导致训练崩溃。去掉 Group/Role Embeddings 后运动控制精度大幅下降，尤其多主体模式

2. **层级运动注入（Hierarchical BBox Injection）**:

    - 功能：将 bbox 条件从输入到每层都注入，而非仅在输入层叠加一次
    - 核心思路：bbox 先通过 3D VAE 编码为潜变量 $\mathbf{z}_{box}$，初始叠加 $\mathbf{h}_0 = \mathbf{z}_t + \mathcal{Z}_{in}(\mathbf{z}_{box})$，每层输出再叠加 $\mathbf{h}_{l+1} = \text{Block}_l(\mathbf{h}_l) + \mathcal{Z}_l(\mathbf{z}_{box})$，$\mathcal{Z}_l$ 是 layer-specific zero-convolutions。不增加 token 序列长度
    - 设计动机：消融显示去掉层级注入后多主体 mIoU 从 0.570 暴跌到 0.289（−49.3%），证明仅在输入层注入不足以在深层保持空间控制

3. **潜空间身份奖励模型（LIRM）+ 奖励反馈学习（LIReFL）**:

    - 功能：在潜空间中评估视频-参考图像的身份一致性，并在去噪过程中提供密集梯度反馈
    - 核心思路：LIRM 架构 = 预训练 VDM 前 8 层（backbone）+ 身份交叉注意力层 + MLP 预测头。参考图像特征作为 Query $\mathbf{Q} = f_{ref}\mathbf{W_Q}$，视频时空特征作为 Key/Value，通过交叉注意力计算身份对齐度 $r_t = \mathcal{H}(\mathbf{h}_{attn} + \mathbf{Q})$。用 ~27,500 视频的人类偏好数据和 BCE 损失训练。LIReFL 在去噪中随机采样时间步 $t_m$，执行单步梯度可传播的去噪，将中间潜变量送入冻结 LIRM 得到奖励，损失 $\mathcal{L} = \mathcal{L}_{sft} + 0.1 \cdot \mathcal{L}_{LIReFL}$
    - 设计动机：(1) 基于 VDM 而非静态图像编码器（CLIP/DINO）——VDM 具有时空先验，能区分"身份一致但姿态不同"和"copy-paste"；关键发现：参考图像必须作为 Query（作为 KV 则准确率从 0.720 暴跌到 0.455）。(2) 在潜空间操作完全绕过 VAE 解码器，训练效率显著提升。(3) 在任意时间步 $t_m$ 而非仅最终步提供反馈，覆盖整个去噪过程的结构性信息

### 损失函数 / 训练策略

Stage 1: $\mathcal{L}_{sft} = \mathbb{E}[(1 + \lambda_1 \mathbf{M}) \cdot \|\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathcal{C}, t)\|_2^2]$，bbox 内区域权重放大（$\lambda_1=2$）。Stage 2: LIRM 训练 ~4,000 步，LIReFL 训练 3,400 步。64 × A100 GPU 用于 SFT，16 × A100 用于 RL 阶段。

## 实验关键数据

### 主实验（DreamOmni Bench）

| 方法 | R-CLIP↑ | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ | CLIP-T↑ |
|------|---------|---------|---------|-------|------|---------|
| DreamVideo-2 | 0.731 | 0.429 | 0.157 | 0.212 | 24.05 | 0.297 |
| **DreamVideo-Omni** | **0.739** | **0.499** | **0.301** | **0.558** | **9.31** | **0.308** |

### 运动控制精度（DreamOmni Bench）

| 方法 | 单主体 mIoU↑ | 单主体 EPE↓ | 多主体 mIoU↑ | 多主体 EPE↓ |
|------|-------------|------------|-------------|------------|
| Tora (1.1B) | 0.163 | 31.74 | 0.162 | 32.84 |
| Wan-Move (14B) | 0.507 | 14.43 | 0.541 | 9.02 |
| **Ours (1.3B)** | **0.558** | **9.31** | **0.570** | **6.08** |

### 消融实验

| 配置 | mIoU↑ | EPE↓ | Face-S↑ |
|------|-------|------|---------|
| w/o Hierarchical BBox | 0.289 | 13.88 | - |
| w/o Group & Role Emb. | 0.458 | 10.38 | - |
| w/o Condition-aware RoPE | 训练崩溃 | - | - |
| w/o LIReFL (Stage 1 only) | - | - | 0.271 |
| Full DreamVideo-Omni | 0.570 | 6.08 | 0.329 |

### 关键发现

- **1.3B 参数超越 14B 模型**：DreamVideo-Omni（1.3B）在单主体和多主体运动控制上全面超越 Wan-Move（14B），EPE 降低 35%（单主体）/ 33%（多主体），证明精准的条件注入比模型规模更重要
- **层级 BBox 注入至关重要**：去掉后 mIoU 暴跌 49.3%，说明在 DiT 中空间控制信号必须在每层都被强化
- **LIRM 中 Query 方向至关重要**：参考图像作为 Query（0.720 准确率）远胜于作为 Key/Value（0.455）——因为"从参考身份出发查找视频中的对应"比"从视频出发查找参考"更有效

## 亮点与洞察

- **Group/Role Embeddings 解决多主体歧义**：将控制信号与主体的显式绑定思想优雅而通用，可迁移到任何需要多实体独立控制的生成任务
- **潜空间身份奖励绕过 VAE 瓶颈**：这是 ReFL 在视频生成中的关键突破——像素空间计算奖励的 GPU 开销使视频 ReFL 几乎不可行，潜空间操作使其成为可能。且在任意时间步提供反馈（而非仅最终步）覆盖了更多结构性信息
- **统一相机运动和局部运动为轨迹控制**：用背景像素的轨迹表示相机运动，避免了显式 3D 相机参数估计和额外训练数据，大大简化了管线

## 局限与展望

- **训练成本高**：Stage 1 需要 64×A100 训练 40K 步，Stage 2 需要 16×A100，总计算量巨大
- **数据构建管道复杂**：需要 RAFT 光流、RAM++ 标签、Qwen3-VL 描述、GroundingDINO 检测、SAM2 分割、CoTracker3 轨迹追踪——完整复现困难
- **评估主要在自建 Bench 上**：DreamOmni Bench 是新建基准，社区认可度尚需建立
- **仅基于 Wan2.1-1.3B**：更大的 VDM 基础模型（14B）可能进一步释放性能

## 相关工作与启发

- **vs DreamVideo-2**: DreamVideo-2 仅支持单主体+粗粒度 bbox 控制，身份保持弱（Face-S 0.157 vs 0.301），运动精度差（EPE 24.05 vs 9.31）。本文的 Group/Role Embeddings 和 LIReFL 是差距来源
- **vs Wan-Move**: Wan-Move 用 14B 参数实现精确轨迹控制但不支持身份定制。本文 1.3B 模型在运动指标上全面超越，说明架构设计比参数量更重要
- **vs IPRO / Identity-GRPO**: 都需要将潜变量解码为像素再计算身份奖励，GPU 开销大且反馈受限于最后几步。LIReFL 完全在潜空间操作是关键优势

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 第一个统一多主体定制+全运动控制+潜空间身份 RL 的框架
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准、多维度指标、用户研究、详尽消融
- 写作质量: ⭐⭐⭐⭐ 系统完整但细节极多
- 价值: ⭐⭐⭐⭐⭐ 定义了视频定制+运动控制的新 SOTA 和评估标准

<!-- RELATED:START -->

## 相关论文

- [PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)
- [MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](mokus_leveraging_cross-modal_knowledge_transfer_for_knowledge-aware_concept_cust.md)

<!-- RELATED:END -->
