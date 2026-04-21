---
title: >-
  [论文解读] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning
description: >-
  [CVPR 2026][目标检测][视频定制生成] 提出 DreamVideo-Omni，通过两阶段渐进训练范式（全运动身份监督微调 + 潜空间身份奖励反馈学习），在单一 DiT 架构中首次统一实现多主体定制与全粒度运动控制（全局包围盒 + 局部轨迹 + 相机运动）。
tags:
  - CVPR 2026
  - 目标检测
  - 视频定制生成
  - 多主体身份保持
  - 全运动控制
  - 潜空间强化学习
  - DiT
---

# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**代码**: [项目页](https://dreamvideo-omni.github.io)  
**领域**: 视频生成 / 多主体定制  
**关键词**: 视频定制生成, 多主体身份保持, 全运动控制, 潜空间强化学习, DiT

## 一句话总结

提出 DreamVideo-Omni，通过两阶段渐进训练范式（全运动身份监督微调 + 潜空间身份奖励反馈学习），在单一 DiT 架构中首次统一实现多主体定制与全粒度运动控制（全局包围盒 + 局部轨迹 + 相机运动）。

## 研究背景与动机

**领域现状**：大规模扩散模型在文本到视频生成方面取得突破，但真实应用场景要求在生成高保真视频的同时精确控制多主体身份和多粒度运动。现有方法要么只关注主体定制（如 ConsisID、VideoMage），要么只关注运动控制（如 Tora、Wan-Move），鲜有统一框架。

**痛点**：当前统一尝试面临三大瓶颈——(a) **运动控制粒度有限**：大多仅依赖单一信号（包围盒/深度图/稀疏轨迹），无法同时控制全局位置、局部动态和相机运动；(b) **控制歧义**：多主体场景中，模型无法辨别哪个运动信号对应哪个主体；(c) **身份退化**：引入运动控制后，身份保真度下降，因为身份保持要求像素一致性而运动控制要求像素动态变化，传统扩散重建损失无法调和此矛盾。

**核心矛盾**：身份保持（鼓励像素级静态一致）与运动控制（要求像素动态演变）目标天然对立，标准扩散损失不足以同时满足两者。

**要解决什么**：在单一框架中同时实现多主体定制 + 全运动控制（全局 + 局部 + 相机），且不牺牲身份保真度。

**切入角度**：(a) 将运动信号显式绑定到对应主体消除歧义；(b) 用基于人类偏好的强化学习（而非重建损失）来优化身份保持，因为身份评估本质上是主观的感知对齐。

**核心 idea**：两阶段范式——第一阶段用结构化三元组 ⟨参考主体, 全局包围盒, 局部轨迹⟩ 联合训练，引入 group/role embeddings 消除歧义；第二阶段训练潜空间身份奖励模型（LIRM）在潜空间直接计算身份奖励，绕过 VAE 解码器进行高效强化学习。

## 方法详解

### 整体框架

基于 Wan2.1-1.3B T2V DiT 预训练模型，采用两阶段渐进训练：

- **Stage 1（全运动身份 SFT）**：在 DiT 中统一注入参考图像、包围盒、轨迹三类条件信号，联合训练单/多主体定制 + 全局/局部运动控制 + 相机运动控制。
- **Stage 2（潜空间身份强化学习）**：先训练 LIRM 奖励模型，再用 Reward Feedback Learning 在潜空间直接优化身份保持。

### 关键设计

#### 1. 条件感知 3D RoPE（Condition-Aware 3D RoPE）

- **功能**：为异构输入（视频帧、参考图像、轨迹、padding）分配不同的时间位置索引。
- **核心思路**：视频帧用顺序时间索引 $t \in [0, T-1]$；参考图像用共享的特殊时间索引 $t_{\text{ref}}$ 使模型将其视为静态条件；padding 用无效索引 $t_{\text{pad}}$ 使模型忽略；轨迹继承视频帧索引保证时空对齐。
- **设计动机**：直接拼接异构 token 会导致位置编码混乱，消融实验显示去掉后训练崩溃（R-DINO 从 0.499 暴跌至 0.139）。

#### 2. Group & Role Embeddings

- **功能**：解决多主体场景中"哪个运动信号控制哪个主体"的歧义。
- **核心思路**：每个控制单元 ⟨参考主体, 包围盒, 轨迹⟩ 分配唯一的 group embedding；另设 role embedding 区分"外观资产"（参考图像）和"运动引导"（包围盒/轨迹）。
- **设计动机**：无此设计时多主体 mIoU 从 0.532 降至 0.459，EPE 从 6.80 升至 20.69。

#### 3. 层级包围盒注入（Hierarchical BBox Injection）

- **功能**：将包围盒潜表示通过逐层零卷积注入到每个 DiT block 的输出中。
- **核心思路**：$\bm{h}_0 = \bm{z}_t + \mathcal{Z}_{\text{in}}(\bm{z}_{\text{box}})$，$\bm{h}_{l+1} = \text{Block}_l(\bm{h}_l) + \mathcal{Z}_l(\bm{z}_{\text{box}})$，每层有独立的零卷积。
- **设计动机**：仅在输入层融合不够，去掉后多主体 mIoU 从 0.532 暴跌至 0.289。

#### 4. 潜空间身份奖励模型（LIRM）

- **功能**：在潜空间评估生成视频与参考图像的身份一致性，提供奖励信号。
- **核心思路**：以预训练 VDM 前 8 层为 backbone，参考图像特征作为 Q、视频特征作为 K/V 做交叉注意力，MLP 头预测标量奖励。用 BCE 损失训练，基于 ~27.5K 人工标注的视频 win-lose 对。
- **设计动机**：相比 CLIP/DINO 等静态编码器，VDM backbone 具备时空先验，能感知运动下的身份一致性；潜空间操作避免了 VAE 解码开销。

### 损失函数 / 训练策略

- **Stage 1 损失**：加权扩散损失 $\mathcal{L}_{\text{sft}} = \mathbb{E}[(1 + \lambda_1 \mathbf{M}) \cdot \|\epsilon - \epsilon_\theta(\bm{z}_t, \mathcal{C}, t)\|_2^2]$，其中 $\lambda_1=2$ 加强前景区域学习。
- **Stage 2 损失**：$\mathcal{L} = \mathcal{L}_{\text{sft}} + \lambda_2 \mathcal{L}_{\text{LIReFL}}$，$\lambda_2=0.1$。LIReFL 从高斯噪声初始化，先无梯度去噪到随机中间步 $t_m$，再执行一步有梯度去噪，冻结 LIRM 计算奖励并反向传播。SFT 损失作为正则化防止 reward hacking。
- **训练规模**：Stage 1 在 64×A100 上训练 40K 步；LIRM 训练 4K 步；LIReFL 微调 3.4K 步（16×A100）。

## 实验关键数据

### 主实验

**表1：DreamOmni Bench 联合定制+运动控制对比**

| 方法 | R-CLIP↑ | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ | CLIP-T↑ |
|------|---------|---------|---------|-------|------|---------|
| DreamVideo-2 | 0.731 | 0.429 | 0.157 | 0.212 | 24.05 | 0.297 |
| **DreamVideo-Omni** | **0.739** | **0.499** | **0.301** | **0.558** | **9.31** | **0.308** |

**表2：运动控制对比（单主体 / 多主体）**

| 方法 | 单主体 mIoU↑ | 单主体 EPE↓ | 多主体 mIoU↑ | 多主体 EPE↓ |
|------|-------------|------------|-------------|------------|
| Tora (1.1B) | 0.163 | 31.74 | 0.162 | 32.84 |
| Wan-Move (14B) | 0.507 | 14.43 | 0.541 | 9.02 |
| **DreamVideo-Omni (1.3B)** | **0.558** | **9.31** | **0.570** | **6.08** |

1.3B 参数的 DreamVideo-Omni 全面超越 14B 的 Wan-Move。

### 消融实验

**表3：各组件消融（单主体模式）**

| 配置 | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ |
|------|---------|---------|-------|------|
| w/o Cond-Aware 3D RoPE | 0.139 | 0.039 | 0.274 | 30.22 |
| w/o Group & Role Emb. | 0.486 | 0.254 | 0.524 | 26.24 |
| w/o Hierarchical BBox | 0.508 | 0.257 | 0.400 | 31.84 |
| Stage 1 Only | 0.483 | 0.251 | 0.556 | 10.53 |
| w/o LIReFL (Stage2 SFT only) | 0.487 | 0.266 | 0.561 | 10.01 |
| **Full Model** | **0.499** | **0.301** | **0.558** | **9.31** |

### 关键发现

1. **Condition-Aware 3D RoPE 是根基**：去掉后所有指标灾难性下降，训练直接崩溃。
2. **Group/Role Embeddings 对多主体至关重要**：去掉后多主体 EPE 从 6.80 涨至 20.69（3× 退化）。
3. **层级注入 vs 输入级融合差距巨大**：多主体 mIoU 从 0.532 降至 0.289。
4. **LIReFL 有效提升身份保真度**：同样是 Stage 2 训练，纯 SFT 增益有限，LIReFL 在多主体 Face-S 上额外提升 0.013，R-DINO 提升 0.012。
5. **全时间步奖励 > 最后 3 步**：在所有时间步施加奖励反馈比仅在最后 3 步去噪时效果更好。
6. **涌现能力**：尽管基于 T2V 模型训练，自然涌现零样本 I2V 生成和首帧条件轨迹控制能力。
7. **用户研究**：在联合任务中总体质量偏好率达 89.2%（vs DreamVideo-2 的 10.8%）。

## 亮点与洞察

- **首个多主体定制+全运动控制统一框架**：一个 DiT 同时处理主体外观、全局运动、局部动态和相机运动。
- **Group/Role Embeddings 的绑定机制**：简洁优雅地解决多主体控制歧义问题，对每个 ⟨主体, 包围盒, 轨迹⟩ 三元组分配 group embedding，将信号显式绑定到主体。
- **潜空间奖励学习**：避免 VAE 解码的巨大开销，使视频级 ReFL 真正可行；VDM backbone 比 CLIP/DINO 更适合评估运动下的身份一致性。
- **相机运动 = 背景轨迹**：不需要额外 3D 相机参数，直接复用轨迹控制机制来控制相机运动，减少训练开销。
- **数据集（2.12M）和 Benchmark（1027 视频）**均为社区新贡献。

## 局限与展望

1. 基于 1.3B 模型，视频质量上限受限于 base model 能力，可扩展到更大模型。
2. 分辨率仅 480×832、49 帧，距离高清长视频还有距离。
3. LIRM 的人工标注（27.5K 对）成本较高，可探索自动化偏好数据生成。
4. 相机运动控制通过背景轨迹间接实现，对精确 3D 相机参数控制可能不够精细。
5. 多主体超过 2-3 个时的扩展性和质量需要更多验证。

## 相关工作与启发

- **DreamVideo-2**：前作，仅支持单主体+包围盒控制，本文全面升级。
- **Wan-Move**：14B 参数的 I2V 轨迹控制模型，本文 1.3B 即超越，说明架构设计>参数规模。
- **IPRO / Identity-GRPO**：类似的身份强化学习思路，但在像素空间计算奖励开销大，且仅限最后去噪步反馈。本文的潜空间方案更高效且可全时间步反馈。
- **PRFL**：同期的潜空间奖励建模工作，但面向通用视频质量，本文聚焦身份保持。
- **启发**：(1) 潜空间奖励学习的范式可推广到其他视频控制任务；(2) Group/Role embedding 的显式绑定思路可用于其他多条件生成场景。

## 评分

⭐⭐⭐⭐⭐ 工程和方法完成度极高的系统性工作：首次统一多主体定制与全运动控制，两阶段范式设计合理，消融充分，数据集和 Benchmark 均有贡献，1.3B 超越 14B 的结果令人印象深刻。

<!-- RELATED:START -->

## 相关论文

- [MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)

<!-- RELATED:END -->
