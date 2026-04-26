---
title: >-
  [论文解读] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model
description: >-
  [CVPR2026][模型压缩][紧凑离散tokenizer] 提出 CompACT，将每张图像压缩至仅 8 个离散 token（约 128 bits），通过冻结预训练视觉编码器保留规划关键语义信息、生成式解码补充感知细节，使基于世界模型的规划速度提升约 40 倍且精度不降。
tags:
  - CVPR2026
  - 模型压缩
  - 紧凑离散tokenizer
  - 世界模型
  - 潜空间规划
  - 极端压缩
  - 语义编码
  - 生成式解码
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model

**会议**: CVPR2026  
**arXiv**: [2603.05438](https://arxiv.org/abs/2603.05438)  
**代码**: [kdwonn/CompACT](https://kdwonn.github.io/CompACT)  
**领域**: 模型压缩 / 世界模型 / 表征学习  
**关键词**: 紧凑离散tokenizer, 世界模型, 潜空间规划, 极端压缩, 语义编码, 生成式解码

## 一句话总结

提出 CompACT，将每张图像压缩至仅 8 个离散 token（约 128 bits），通过冻结预训练视觉编码器保留规划关键语义信息、生成式解码补充感知细节，使基于世界模型的规划速度提升约 40 倍且精度不降。

## 背景与动机

1. **世界模型的规划瓶颈**：现有世界模型（如 NWM）将每帧编码为数百个 token（SD-VAE 需 784 token），Attention 的二次复杂度导致规划延迟高达 3 分钟/episode，无法用于实时控制
2. **token 数量决定计算代价**：在 MPC 规划过程中需要大量前向推理（~1920 次 rollout），token 数直接影响吞吐
3. **重建保真度 ≠ 规划所需**：传统 tokenizer 优先保留纹理、光照等高频细节，但规划任务只需空间布局、物体关系等高层语义
4. **扩散模型的迭代去噪开销**：连续潜空间需要数百步迭代去噪，进一步拖慢规划速度
5. **现有压缩方案的局限**：FlexTok 等 1D tokenizer 虽支持可变长度，但仍以重建保真度为目标，非面向规划优化
6. **信息论下界支持极端压缩**：作者从互信息角度证明，规划充分表征的最低熵为 $H(\mathbf{a}^*)$，远小于 $H(\mathbf{o})$，理论上仅需百余 bit 即可

## 方法详解

### 整体框架

三阶段流水线：(a) 训练 CompACT tokenizer 将图像映射为紧凑离散 token；(b) 在紧凑潜空间训练动作条件世界模型；(c) 测试时通过 MPC（CEM 优化）在潜空间做规划。

### 编码器：语义编码（ℰ_compact）

- **冻结 DINOv3-B** 提取语义 patch 特征，不做微调（微调反而导致 rFID 从 2.40 退化到 5.22）
- **Latent Resampler**：N 个可学习 query token（N=8 或 16）通过 cross-attention 从 DINOv3 输出中蒸馏高层语义
- **Finite Scalar Quantization (FSQ)**：levels 为 [8,8,8,5,5,5]，每个 token 约 16 bits，总共 128~256 bits/帧
- 设计核心：视觉基础模型已抽象掉低层细节 → cross-attention 仅能提取语义信息 → 天然实现"只保留规划关键信息"

### 解码器：生成式解码（𝒟_compact）

- 不直接从 8/16 token 重建像素（信息不足，ill-posed）
- 以 MaskGIT VQGAN（196 token / 帧）为目标 tokenizer
- 用 masked generative modeling：训练时随机 mask 目标 token，以 compact token 为条件恢复
- 推理时从全 mask 序列开始，基于置信度迭代 unmask，无需迭代去噪
- 训练损失：$\mathcal{L}_{\text{tok}} = -\mathbb{E}[\log p(\mathbf{z}^\psi | \mathbf{z}, M(\mathbf{z}^\psi))]$

### 世界模型训练

- 导航任务：自回归 DiT，固定长度历史窗口 + 历史 token 随机 mask（diffusion forcing 思想）
- 机器人操作任务：block-causal transformer 并行预测多帧
- 世界模型损失：$\mathcal{L}_{\text{world}} = -\mathbb{E}[\log p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{a}_t, M(\mathbf{z}_{t+1}))]$
- 规划时通过 CEM 搜索最优动作序列，代价函数可在像素空间（LPIPS）或潜空间（L1）计算

## 实验关键数据

### 重建质量（ImageNet 验证集）

| Tokenizer | 类型 | #tok | rFID↓ | IS↑ |
|---|---|---|---|---|
| SD-VAE | 连续 | 1024 | 0.64 | 223.8 |
| MaskGIT-VQGAN | 离散 | 256 | 1.83 | 186.7 |
| FlexTok | 离散 | 16 | 5.60 | 114.9 |
| **CompACT** | **离散** | **16** | **2.40** | **209.0** |
| **CompACT** | **离散** | **8** | **3.21** | **207.5** |

### 导航规划（RECON 基准）

| Tokenizer | #tok | ATE↓ | RPE↓ | 延迟(s)↓ |
|---|---|---|---|---|
| SD-VAE | 784 | 1.262 | 0.354 | 178.78 |
| FlexTok | 64 | 1.484 | 0.400 | 16.68 |
| FlexTok | 16 | 1.625 | 0.446 | 14.48 |
| **CompACT** | **16** | **1.330** | **0.390** | **5.78** |
| **CompACT** | **8** | **1.373** | **0.401** | **4.83** |

CompACT-16 比 SD-VAE 快 **~31×**，CompACT-8 快 **~37×**，且精度相当。

### 消融实验

- **去掉生成式解码**（单步前馈解码）：rFID 从 2.40 暴增到 28.80
- **解冻 DINOv3 微调**：rFID 退化到 5.22，规划 ATE 从 1.330 退化到 1.472
- **历史 masking**：关闭后 ATE 从 1.330 退化到 1.480
- **潜空间代价函数**：ATE 略降（1.379 vs 1.330），但延迟从 5.78s 降至 2.15s（整体 ~80× over SD-VAE）

### 动作条件视频预测（RoboNet）

| 模型 | #tok | APE↓ | 延迟(s)↓ |
|---|---|---|---|
| MaskGIT-VQGAN | 256 | 0.3383 | 3.826 |
| **CompACT** | **16** | **0.1122** | **0.740** |

APE 降低 3×，速度提升 5.2×，验证紧凑 token 对动作相关信息的保持。

## 亮点

1. **极端压缩比**：8 token / 128 bits 编码一帧，较 SD-VAE 压缩近 100×，且规划精度持平
2. **冻结编码器的反直觉洞察**：不微调 DINOv3 反而更好，微调会使语义退化
3. **信息论支撑**：从互信息角度严格证明规划充分表征的最低信息量远小于完整图像熵
4. **模块化 token 注意力**：每个 token 自发关注语义一致的区域（物体、末端执行器等），无需显式监督
5. **跨骨干泛化**：SigLIP-2、MAE、DINOv3 均有效，方法不依赖特定视觉基础模型

## 局限与展望

1. **闭环操作仅有初步验证**：仅在 RoboMimic Lift 上测试（56% 成功率），缺乏复杂接触和长horizon任务验证
2. **解码器依赖 MaskGIT VQGAN 质量**：最终重建上限受目标 tokenizer 约束
3. **未在大规模自驾/游戏场景验证**：仅覆盖室内导航和桌面操作
4. **无闭环 IDM 集成**：IDM 未利用实时观测和本体感知信息
5. **离散量化的信息损失不可控**：FSQ 的 level 选择目前靠经验，缺乏自适应机制

## 与相关工作的对比

| 方法 | 特点 | 与 CompACT 的区别 |
|---|---|---|
| NWM (Bar et al.) | SD-VAE 784 token + 扩散世界模型 | 延迟高 ~180s，CompACT 快 40× |
| FlexTok | 可变长度 1D tokenizer | 面向重建非规划，同 16 tok 时 rFID 5.60 vs 2.40 |
| IRIS / iVideoGPT | 离散 token + 条件压缩 | 依赖前帧条件，不适合长horizon规划 |
| DINO-WM | DINO 特征做世界模型 | 仍用大量 token，未做极端压缩 |
| TA-TiTok | 32 token 紧凑 tokenizer | 未针对规划优化，无世界模型验证 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 极端压缩 + 冻结编码器 + 生成式解码的组合设计新颖，信息论分析有理论深度
- 实验充分度: ⭐⭐⭐⭐ — 重建/规划/视频预测/消融齐全，但闭环操作验证偏弱
- 写作质量: ⭐⭐⭐⭐⭐ — 动机-方法-实验逻辑链清晰，Proposition 1 和消融设计精炼
- 价值: ⭐⭐⭐⭐ — 为世界模型实时规划提供了实用路径，40× 加速有工程意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] WPT: World-to-Policy Transfer via Online World Model Distillation](wpt_world-to-policy_transfer_via_online_world_model_distillation.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[ICCV 2025\] TokenBridge: Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_v.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)

<!-- RELATED:END -->
