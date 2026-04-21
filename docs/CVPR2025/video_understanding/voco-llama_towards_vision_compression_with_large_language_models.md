---
title: >-
  [论文解读] VoCo-LLaMA: Towards Vision Compression with Large Language Models
description: >-
  [CVPR 2025][视频理解][视觉Token压缩] 提出 VoCo-LLaMA，首个利用 LLM 自身能力压缩视觉 token 的方法，通过在视觉和文本 token 之间插入 VoCo token 并修改注意力掩码实现注意力蒸馏，以单个 token 实现 576 倍压缩率同时保持 83.7% 性能。
tags:
  - CVPR 2025
  - 视频理解
  - 视觉Token压缩
  - 大语言模型
  - 注意力蒸馏
  - KV Cache复用
---

# VoCo-LLaMA: Towards Vision Compression with Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2406.12275](https://arxiv.org/abs/2406.12275)  
**代码**: 无  
**领域**: 视频理解 / 多模态 (Video Understanding / Vision-Language Models)  
**关键词**: 视觉Token压缩, 大语言模型, 注意力蒸馏, KV Cache复用, 视频理解

## 一句话总结

提出 VoCo-LLaMA，首个利用 LLM 自身能力压缩视觉 token 的方法，通过在视觉和文本 token 之间插入 VoCo token 并修改注意力掩码实现注意力蒸馏，以单个 token 实现 576 倍压缩率同时保持 83.7% 性能。

## 研究背景与动机

视觉语言模型（VLM）在多模态任务上取得了巨大成功，但面临**上下文窗口受限**和**高分辨率/视频输入计算成本高**的瓶颈。例如 LLaVA-1.6 处理 672×672 分辨率图像需要 2880 个视觉 token，占据超过一半的上下文长度。随着输入图片数量增加或视频帧数增多，文本可用的上下文窗口被进一步压缩。

现有视觉压缩方法（Q-Former、Re-sampler、平均池化等）使用**外部模块**压缩视觉 token，然后强迫 LLM 理解压缩后的 token。这种"外压内学"方式存在根本问题：LLM 理解视觉 token 的方式和压缩学习过程是脱节的，导致高压缩率下严重的视觉信息损失。

VoCo-LLaMA 的核心创新是：**让 LLM 自己来做视觉压缩**。通过在视觉指令微调阶段引入 Vision Compression (VoCo) token，并修改注意力掩码使文本 token 只能通过 VoCo token 间接获取视觉信息，LLM 自然地学会将视觉理解蒸馏到 VoCo token 的 transformer activation 中。这保证了压缩和理解使用同一套模型参数和范式。

## 方法详解

### 整体框架

输入序列为 $(\mathcal{V}, VoCo, \mathcal{T}) = (V_0, ..., V_n, VoCo, T_0, ..., T_m)$。训练时通过注意力掩码实现两阶段信息流：VoCo token 可以看到所有视觉 token，但文本 token 只能看到 VoCo token（看不到原始视觉 token）。推理时分两步前向：第一步压缩视觉token为 VoCo cache，第二步用 VoCo cache + 文本 token 完成任务。

### 关键设计

1. **注意力蒸馏压缩机制（Attention Distillation Compression）**:
    - 功能：让 LLM 自身将视觉 token 的信息蒸馏到紧凑的 VoCo token 中
    - 核心思路：修改注意力掩码 $M_{ij}$：文本 token $i \in \mathcal{T}$ 到视觉 token $j \in \mathcal{V}$ 设为 False（禁止直接交互），文本到 VoCo 设为 True，VoCo 到视觉保持 causal attention 的 True。优化目标为最小化 KL 散度 $E_{\mathcal{V},\mathcal{T}}[D_{KL}(p_{LM_o}(y|\mathcal{V},\mathcal{T}) \| p_{VoCo-LLaMA})]$，其中 $p_{VoCo-LLaMA} = p_{LM}(y|LM(\mathcal{V}, VoCo), \mathcal{T})$。实现上只需修改注意力掩码矩阵，极其简洁
    - 设计动机：外部压缩模块（Q-Former等）的压缩范式与 LLM 的理解范式不一致，导致信息损失；让 LLM 自己压缩确保压缩和理解使用同一模型和范式

2. **VoCo Cache 复用机制**:
    - 功能：缓存压缩后的视觉表征，支持同一图像的多任务复用
    - 核心思路：推理分两阶段——第一阶段输入 [视觉token, VoCo token]，将视觉信息压缩为 VoCo token 上的 KV Cache；第二阶段输入 [文本token]，加载 VoCo Cache 即可。同一图像的 VoCo Cache 可在不同任务间复用。相比缓存完整视觉 token 的 KV Cache，存储量减少 99.8%
    - 设计动机：避免每次查询都重新处理大量视觉 token，实现真正的计算效率提升

3. **时序建模扩展到视频（Temporal Modeling Extension）**:
    - 功能：将 VoCo-LLaMA 从图像压缩扩展到视频理解
    - 核心思路：将视频分为多个帧段，每段独立压缩为 VoCo Cache $Cache_t = LM(\mathcal{V}_t, VoCo_t)$，然后将所有帧的 VoCo Cache 拼接为时序序列 $\mathcal{F} = \{Cache(VoCo_1), ..., Cache(VoCo_k)\}$。在此基础上继续训练使模型关注时序相关性，学习 $p(y|\mathcal{F}, \mathcal{T})$。单个 VoCo token 使得相同上下文长度可处理约 200 倍的视频帧
    - 设计动机：视频帧数巨大，直接处理全部视觉 token 远超 LLM 上下文长度

### 损失函数 / 训练策略

- **训练损失**: KL 散度蒸馏 + 标准自回归语言建模（SFT）
- **训练数据**: LLaVA-filtered CC3M（对齐阶段）+ 多任务指令数据（VoCo 压缩阶段）+ WebVid + Video-ChatGPT QA（视频阶段）
- **模型配置**: CLIP-ViT-L 视觉编码器 + Vicuna-7B LLM
- **训练技巧**: 使用梯度检查点减少显存

## 实验关键数据

### 主实验

图像基准（576 个视觉 token 压缩为 1 个 VoCo token）：

| 方法 | Token数 | GQA | MMB | MME | POPE | VQAv2 | 平均保持率 |
|------|---------|-----|-----|-----|------|-------|----------|
| 上界(无压缩) | 576 | 61.1 | 64.0 | 1487 | 85.0 | 77.7 | 100% |
| **VoCo-LLaMA** | **1** | **57.0** | **58.8** | **1323** | **81.4** | **72.3** | **83.7%** |
| Avg Pool + Linear | 1 | 52.9 | 55.5 | 1210 | 79.1 | 65.0 | 64.1% |
| Q-Former | 1 | 51.1 | — | — | — | — | <64% |

### 消融实验

| 配置 | 说明 |
|------|------|
| 多VoCo token (1→4→16) | 更多token提升质量，但压缩率降低 |
| 无注意力掩码修改 | 文本直接看视觉token，VoCo不学压缩 |
| 仅外部压缩（Q-Former） | 性能显著下降，证明 LLM 内在压缩的优势 |

### 关键发现

- 576 倍压缩率下保持 83.7% 性能，显著优于 Q-Former (约 64%) 和平均池化 (64.1%)
- 推理效率：KV Cache 存储减少 99.8%，FLOPs 减少 94.8%，推理时间减少 69.6%
- 视频理解方面，通过 VoCo 压缩可处理约 200 倍更多视频帧，在 Video-ChatGPT 基准上超越之前方法
- VoCo Cache 复用使得同一图像的多任务查询无需重复编码

## 亮点与洞察

- **"让LLM自己压缩"的思路极具启发性**：LLM 已经学会理解视觉 token，那它自然也最擅长判断哪些信息是核心的。这种内在压缩比外部模块强制压缩更加自然且信息损失小
- **极度简洁的实现**：整个方法的核心就是修改注意力掩码矩阵——将文本到视觉的注意力设为 False 即可。无需引入任何新模块、新参数、新架构

## 局限与展望

- 当前仅在 7B 规模模型验证，更大模型（70B+）的效果未知
- 极致压缩（1 token）仍损失约 16% 性能，对精确视觉推理任务影响较大
- 视频时序建模依赖简单的 cache 拼接，缺少显式的时序注意力机制
- 未探索动态压缩率——根据图像复杂度自适应选择 VoCo token 数量

## 相关工作与启发

- **vs Q-Former (BLIP-2)**: Q-Former 用外部可学习查询压缩视觉 token，与 LLM 理解范式脱节；VoCo-LLaMA 让 LLM 自身做压缩，保持率高出约 20 个百分点
- **vs Chat-UniVi**: Chat-UniVi 使用平均池化+线性层做压缩，方法简单但信息损失大；VoCo-LLaMA 利用 LLM 的全部注意力机制做压缩
- **vs 文本压缩方法 (AutoCompressor, ICAE)**: 类似思路但应用于文本领域；VoCo-LLaMA 是首个将此范式引入视觉模态的工作

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次让 LLM 自身压缩视觉 token，思路简洁且有效
- 实验充分度: ⭐⭐⭐⭐ 多基准评估+消融+效率分析+视频扩展
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导简洁
- 价值: ⭐⭐⭐⭐⭐ 576 倍压缩保持 83.7% 性能，对 VLM 效率提升有重大意义

<!-- RELATED:START -->

## 相关论文

- [Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [LLAVIDAL: A Large Language Vision Model for Daily Activities of Living](llavidal_a_large_language_vision_model_for_daily_activities_of_living.md)
- [On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

<!-- RELATED:END -->
