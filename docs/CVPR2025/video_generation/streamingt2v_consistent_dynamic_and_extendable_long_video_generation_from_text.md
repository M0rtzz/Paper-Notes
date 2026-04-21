---
title: >-
  [论文解读] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text
description: >-
  [CVPR 2025][图像生成][长视频生成] 提出 StreamingT2V，一种自回归文本到长视频生成方法，通过短期记忆模块（CAM）和长期记忆模块（APM）实现长达 2 分钟以上（1200+ 帧）的无缝、高运动量视频生成。
tags:
  - CVPR 2025
  - 图像生成
  - 长视频生成
  - 自回归视频合成
  - 时序一致性
  - 扩散模型
  - 文本到视频
---

# StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

**会议**: CVPR 2025  
**arXiv**: [2403.14773](https://arxiv.org/abs/2403.14773)  
**代码**: [GitHub](https://github.com/Picsart-AI-Research/StreamingT2V)  
**领域**: 视频生成 (Video Generation)  
**关键词**: 长视频生成, 自回归视频合成, 时序一致性, 扩散模型, 文本到视频

## 一句话总结

提出 StreamingT2V，一种自回归文本到长视频生成方法，通过短期记忆模块（CAM）和长期记忆模块（APM）实现长达 2 分钟以上（1200+ 帧）的无缝、高运动量视频生成。

## 研究背景与动机

文本到视频扩散模型在短视频生成（≤16帧）方面取得了显著进展，但在生成长视频时面临严重挑战。直接在长视频上训练模型在计算上不可行（仅 16 帧就需要 260K 步训练和 4.5K batch size）。现有的自回归扩展方案主要存在以下问题：

1. **帧拼接方案**：将前一片段最后几帧的噪声 latent 与当前片段拼接，条件太弱导致片段间不一致
2. **CLIP 图像嵌入方案**：如 SVD、I2VGen-XL 使用 CLIP 图像编码器提取前一帧特征，但 CLIP 会丢失重建所需的关键信息
3. **稀疏编码器方案**：如 SparseCtrl 需要在条件帧后补零帧，输入不一致导致输出不一致

更关键的是，所有现有方法在自回归应用时都会产生**视频停滞**（video stagnation）——背景冻结、运动消失，视频几乎变成静态图像。此外，随着自回归过程推进，对象外观变化和视频质量衰退也是严重问题。

StreamingT2V 的核心 idea 是：通过引入**短期记忆**（利用前一片段的多帧特征通过注意力机制条件化当前生成）和**长期记忆**（从首帧提取高层特征防止遗忘初始场景），同时保持运动的丰富性和片段间的一致性。

## 方法详解

### 整体框架

StreamingT2V 的流水线分为三个阶段：
1. **初始化阶段**：使用现成的 text-to-video 模型（如 Modelscope）生成首个 16 帧片段
2. **流式 T2V 阶段**：通过 CAM 和 APM 自回归生成后续帧，每次生成 16 帧并与前一片段重叠 8 帧
3. **流式精化阶段**：使用高分辨率视频增强模型（如 MS-Vid2Vid-XL）通过随机混合方法自回归增强长视频至 720×720 分辨率

### 关键设计

1. **条件注意力模块 (CAM) — 短期记忆**:
    - 功能：利用前一片段最后 F_cond=8 帧的特征条件化当前片段的生成
    - 核心思路：由特征提取器（帧编码器 + UNet 编码器层的权重副本）和特征注入器构成。注入时在 UNet 的每个 skip connection 处，让当前帧的特征通过时序多头交叉注意力 (T-MHA) attend to CAM 提取的特征。Q 来自 UNet skip connection 特征，K/V 来自 CAM 特征
    - 设计动机：注意力机制相比拼接方式能更有效地借用前一片段的内容信息，同时不限制当前帧的运动。输出层零初始化确保训练初期不影响基础模型

2. **外观保持模块 (APM) — 长期记忆**:
    - 功能：从首个片段的锚点帧提取高层场景和对象特征，在整个自回归过程中保持外观一致性
    - 核心思路：(i) 将锚点帧的 CLIP 图像 token 通过 MLP 扩展为 k=16 个 token，与文本 CLIP token 拼接后通过投影块得到混合编码 x_mixed；(ii) 每个交叉注意力层引入可学习权重 α_l（初始化为 0），通过 SiLU(α_l)·x_mixed + x_text 加权混合图像和文本信息
    - 设计动机：仅依赖前一帧的方法会逐步遗忘初始场景特征，导致外观变化和质量衰退。使用固定锚点帧提供全局一致性先验，避免误差累积

3. **随机混合增强 (Randomized Blending)**:
    - 功能：在使用高分辨率模型增强长视频时确保相邻片段间的平滑过渡
    - 核心思路：将长视频分为 24 帧一组（重叠 8 帧）的片段。在每个去噪步骤中，相邻片段共享重叠区域的噪声。然后在重叠区域随机采样切割位置 f_thr，将前一片段的 1:F-f_thr 帧与当前片段的 f_thr+1:F 帧拼接。对于重叠区域的每一帧，以概率 1-f/(O+1) 使用前一片段的 latent
    - 设计动机：朴素独立增强每个片段会导致过渡不一致；仅共享噪声仍有对齐问题；随机混合通过概率性地融合重叠区域的 latent 有效消除片段间的不一致

### 损失函数 / 训练策略

- 使用标准扩散模型去噪损失训练 CAM 和 APM
- CAM 的输出投影层 P_out 零初始化，APM 的权重 α_l 初始化为 0，确保训练初期不干扰基础模型
- 流式精化阶段无需额外训练，直接使用 SDEdit 方法在预训练的高分辨率模型上进行

## 实验关键数据

### 主实验

在 50 个文本 prompt 的测试集上，生成 240 帧视频进行评估：

| 方法 | MAWE↓ | SCuts↓ | CLIP↑ |
|------|-------|--------|-------|
| SparseCtrl | 6069.7 | 5.48 | 29.32 |
| I2VGenXL | 2846.4 | 0.4 | 27.28 |
| DynamiCrafterXL | 176.7 | 1.3 | 27.79 |
| SEINE | 718.9 | 0.28 | 30.13 |
| SVD | 857.2 | 1.1 | 23.95 |
| FreeNoise | 1298.4 | 0 | 31.55 |
| OpenSora | 1165.7 | 0.16 | 31.54 |
| OpenSoraPlan | 72.9 | 0.24 | 29.34 |
| **StreamingT2V** | **52.3** | **0.04** | **31.73** |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| 仅 CAM | 平滑片段过渡 | 注意力机制优于拼接方式 |
| CAM + APM | 保持外观一致性 | 锚点帧防止场景遗忘 |
| 朴素拼接增强 | 明显过渡不一致 | X-T slice 可视化显示严重断裂 |
| 共享噪声增强 | 略有改善但仍有对齐问题 | 单独不足以解决 |
| 随机混合增强 | 平滑无缝过渡 | 概率混合有效消除不一致 |

### 关键发现

1. MAWE 指标显著优于所有竞争方法（比第二名 OpenSoraPlan 低约 30%）
2. FreeNoise 虽然 SCuts=0，但生成的是近乎静态的视频，运动几乎为零
3. 使用 CLIP 图像编码器的方法（SVD、DynamiCrafterXL、I2VGenXL）在自回归设置下 CLIP 分数很低，可能因为 CLIP 图像编码器在生成图像上存在域偏移
4. StreamingT2V 的指标在时间推移中保持稳定（120-220帧范围内 MAWE: 43-46, CLIP: 31.79-32.45）
5. 方法可泛化到 DiT 架构（如 OpenSora）

## 亮点与洞察

1. **短期+长期记忆的互补设计**：CAM 确保相邻片段的平滑过渡，APM 防止长程外观漂移，两者协同解决了长视频生成的核心挑战
2. **注意力机制 vs 拼接的优势**：CAM 的交叉注意力设计不需要将条件帧补零到与目标帧等长，避免了 SparseCtrl 等方法的输入不一致问题
3. **随机混合的巧妙**：将确定性融合转变为概率性混合，每个去噪步在不同位置切换，使过渡自然且稳健
4. **运动停滞问题的揭示**：系统性地发现并论证所有现有自回归方法都会导致视频停滞

## 局限与展望

1. 生成视频的帧率和分辨率仍受限于基础模型的能力
2. 依赖于初始生成质量——如果首帧质量不高，APM 会将不良特征传播到所有后续帧
3. 自回归生成速度较慢，每个片段都需要完整的去噪过程
4. MAWE 作为新提出的指标，其全面性和与人类感知的相关性还需更多验证
5. 未来可探索与 DiT 架构的更深度集成，以及端到端训练增强管线

## 相关工作与启发

- **vs FreeNoise**: FreeNoise 通过重用噪声向量实现帧间一致性，但代价是运动几乎消失；StreamingT2V 通过注意力机制在保持一致性的同时维持运动
- **vs SparseCtrl**: SparseCtrl 使用类 ControlNet 的稀疏编码器，需要零填充导致输入不一致；CAM 通过交叉注意力自然处理不同帧数
- **vs SVD/DynamiCrafter**: 这些方法使用 CLIP 图像嵌入进行条件化，但 CLIP 丢失了关键的重建信息；CAM 直接在特征空间进行注意力交互，保留更丰富的信息
- **vs OpenSora/OpenSoraPlan**: 基于 Transformer 的方法虽能生成更长视频（384帧），但运动量有限且仍有片段过渡问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 短期+长期记忆的自回归框架设计新颖，随机混合方法简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 与 9 种方法对比，提出了新指标 MAWE/SCuts，有消融和稳定性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题分析深入，动机阐述充分
- 价值: ⭐⭐⭐⭐ 长视频生成是重要问题，方法实用且已开源，对后续研究有参考价值

<!-- RELATED:START -->

## 相关论文

- [LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [ViReS: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [AnimateAnything: Consistent and Controllable Animation for Video Generation](animateanything_consistent_and_controllable_animation_for_video_generation.md)
- [World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)

<!-- RELATED:END -->
