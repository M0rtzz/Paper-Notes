---
title: >-
  [论文解读] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation
description: >-
  [CVPR 2026][video editing] AutoCut 提出了一个端到端的广告视频编辑框架，通过残差向量量化（RQVAE）将视频、音频和文本统一到共享的离散 token 空间中，在 Qwen3-8B 上进行多模态对齐和监督微调，实现了视频选择、排序、脚本生成和背景音乐选择四项任务的统一处理，在多项指标上超越 GPT-4o 基线。
tags:
  - CVPR 2026
  - video editing
  - 多模态
  - Residual VQ
  - Advertisement
  - Controllable Generation
---

# AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation

**会议**: CVPR 2026  
**arXiv**: [2603.28366](https://arxiv.org/abs/2603.28366)  
**代码**: [https://github.com/AdAutoCut/Autocut](https://github.com/AdAutoCut/Autocut) (有)  
**领域**: 视频理解 / 视频编辑  
**关键词**: video editing, Multimodal LLM, Residual VQ, Advertisement, Controllable Generation

## 一句话总结
AutoCut 提出了一个端到端的广告视频编辑框架，通过残差向量量化（RQVAE）将视频、音频和文本统一到共享的离散 token 空间中，在 Qwen3-8B 上进行多模态对齐和监督微调，实现了视频选择、排序、脚本生成和背景音乐选择四项任务的统一处理，在多项指标上超越 GPT-4o 基线。

## 研究背景与动机

**领域现状**：短视频已成为数字广告的主要载体，但制作流程涉及脚本编写、素材拍摄、剪辑和后期，成本高、门槛高。

**现有方法的三大障碍**：
   - **松散的多模态耦合**：视频、音频、文本的表征弱对齐，无法进行统一推理
   - **缺乏可解释控制**：模型不提供结构化或离散表示，难以调整叙事节奏和内容重点
   - **割裂的理解与生成**：多模态理解和生成被视为独立过程，优化不一致

**MLLM 的机遇与限制**：多模态大语言模型有潜力统一感知、理解和创作，但**受限于上下文窗口长度**，难以直接处理大规模视频检索和编辑。

**核心 idea**：将视频和音频特征通过 RQVAE 离散化为 token 后与文本 token 统一，构建共享词表，让 LLM 在统一的 token 空间中进行多模态推理和生成。

## 方法详解

### 整体框架
两阶段训练：
1. **多模态对齐**：冻结 LLM 骨干，仅更新新引入的多模态 embedding 层（~700K 样本）
2. **监督微调 SFT**：全参数微调进行任务特定行为学习（~100K 策划样本）

推理时：LLM 生成 token 序列 → 视频 token 检索最近邻片段 + 音频 token 解码 + 文本直接输出 → ffmpeg 合成

### 关键设计

1. **多模态编码与离散化**：

    - **视频编码器**：ResNet-50（对比学习预训练），提取帧级语义 embedding
    - **音频编码器**：PANNs（Wavegram-Logmel-CNN），在 AudioSet 上预训练
    - **RQVAE 离散化**：残差向量量化将连续 embedding 压缩为离散 token
        - 码本大小：$256 \times 8$（每帧/每段音频编码为 8 个 token）
        - 重建质量：视频 cos similarity 0.89，音频 0.96
        - 训练损失：$\mathcal{L}_{rec} = 1 - \cos(\hat{f}, f)$
    - **设计动机**：RQVAE 通过逐级残差逼近实现高效压缩，8 个 token 的配置在重建质量和 token 长度之间取得良好平衡

2. **统一 Token 空间**：

    - 视频 token、音频 token 和文本 token 共享一个扩展词表
    - 多模态对齐阶段用标准 NTP 损失训练：$\mathcal{L}_{NTP} = -\sum_t \log P(x_t | x_{<t})$
    - 对齐阶段冻结 LLM 骨干，仅更新新 embedding 层 → 稳定训练
    - **设计动机**：统一 token 空间使得跨模态推理简化为序列建模问题

3. **四项任务的统一建模**：

    - **视频选择**：从候选池中选择相关片段（CSA 指标）
    - **视频排序**：将片段排列为连贯时序序列（CRA 指标）
    - **脚本生成**：生成与视觉内容对齐的广告文案（SQ + WCD 指标）
    - **背景音乐选择**：检索匹配多模态上下文的 BGM（MSS 指标）

4. **检索与渲染**：

    - 视频 token → FAISS 最近邻搜索匹配素材库中的片段
    - 音频 token → 解码或检索
    - ffmpeg 拼接、转场、字幕叠加 → 最终 MP4

### 训练数据
- **对齐数据**：~700K 筛选后的广告视频（高互动量、有人声）
- **SFT 数据**：~100K 高质量策划样本（时长 <120s，片段 2-60s，Qwen-VL 评估高视觉-文本相关性）
- 数据解析：ASR 提取对齐时间戳、1fps 帧采样、pydub 音频分离

## 实验关键数据

### 主实验（364 测试视频）

| 方法 | CSA↑ | CRA↑ | VSC↑ | SQ↑ | WCD↓ | MSS↑ |
|------|------|------|------|-----|------|------|
| Qwen3-8B (Caption) | 0.137 | 0.016 | 0.931 | 80.0 | 5.26 | – |
| Qwen3-8B (Caption+SFT) | 0.569 | 0.030 | 1.123 | 59.2 | 6.82 | – |
| Qwen2.5-VL-32B | 0.665 | 0.025 | **0.998** | 78.3 | 12.51 | – |
| GPT-4o + MGSV | 0.269 | 0.078 | 1.136 | 83.0 | 7.75 | 0.266 |
| **AutoCut** | **0.659** | **0.107** | 1.036 | **84.6** | **3.02** | **0.348** |

### 消融实验

| 配置 | CSA↑ | CRA↑ | VSC↑ | SQ↑ | WCD↓ |
|------|------|------|------|-----|------|
| SFT only | 0.478 | 0.082 | 1.004 | 83.2 | 4.43 |
| emb+full+sft | **0.717** | 0.058 | 0.967 | 79.0 | 4.50 |
| **emb+sft (ours)** | 0.659 | **0.107** | **1.036** | **84.6** | **3.02** |

### 关键发现
- AutoCut 的 CRA（片段排序准确率）大幅领先所有基线（0.107 vs 0.078），说明 token 化多模态表征能更好地捕捉时序结构
- WCD（脚本-视频时间一致性）3.02 远优于 GPT-4o 的 7.75，体现了联合多模态建模的时间对齐优势
- 人类评估中 AutoCut 在所有 5 个维度上优于 GPT-4o（88% 总体胜率）
- 额外预训练阶段（emb+full+sft）反而降低了 CRA 和 SQ，说明有限质量的预训练语料会引入噪声
- 成本优势显著：处理 100 个视频 AutoCut ~$0.015 vs GPT-4o ~$2.5

## 亮点与洞察
- **"离散化即统一"的思路**：通过 RQVAE 将所有模态统一到 token 空间后，问题简化为 NTP，简洁优雅
- **两阶段训练策略（对齐 + SFT）**比三阶段（对齐 + 预训练 + SFT）更好，说明数据质量 > 数据量
- 低帧率 token 用于推理 + 高帧率帧用于检索的双轨设计兼顾效率和精度
- BGM 选择任务的引入是亮点——视频编辑中音频选择长期被忽视

## 局限与展望
- 视频动作和音频节奏的细粒度同步仍有不足（偶尔出现不同步）
- 控制粒度仅达到片段级别，不支持帧级或情感级编辑
- RQVAE 的重建虽然余弦相似度高，但信息丢失的影响在下游任务中可能被放大
- 评测依赖 GPT-4o 作为 judge（VSC, SQ 指标），存在评估偏差的风险

## 相关工作与启发
- VC-LLM 也用 MLLM 做广告视频生成，但依赖多分辨率时空推理
- MGSV 是唯一具有音频匹配能力的基线
- 与 NExT-GPT 等"任意模态"LLM 相比，AutoCut 更专注于编辑场景的实际约束
- 离散化 + 检索的方式可推广到其他视频创作场景（短剧、Vlog 等）

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态离散化统一框架在广告编辑领域是新方案，但核心组件相对成熟
- 实验充分度: ⭐⭐⭐⭐ 自动指标 + 人工评估 + 消融，但测试集仅 364 视频
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但 evaluation metrics 的定义较分散
- 价值: ⭐⭐⭐⭐ 对广告视频自动化生产有直接实用价值，成本优势显著

<!-- RELATED:START -->

## 相关论文

- [U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)

<!-- RELATED:END -->
