---
title: >-
  [论文解读] Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing
description: >-
  [CVPR2026][视频理解][视频冗余去除] 提出 AutoGaze——一个仅 3M 参数的轻量自回归模块，在 ViT 之前以多尺度方式选择最少量 patch 并去除时空冗余，实现 4×-100× token 压缩和最高 19× ViT 加速，使 MLLM 可扩展到 1K 帧 4K 分辨率视频。
tags:
  - CVPR2026
  - 视频理解
  - 视频冗余去除
  - 自回归注视
  - 多尺度patch选择
  - 视觉token压缩
  - 长视频高分辨率理解
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing

**会议**: CVPR2026  
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**代码**: [autogaze.github.io](https://autogaze.github.io/)  
**领域**: 视频理解  
**关键词**: 视频冗余去除, 自回归注视, 多尺度patch选择, 视觉token压缩, 长视频高分辨率理解

## 一句话总结

提出 AutoGaze——一个仅 3M 参数的轻量自回归模块，在 ViT 之前以多尺度方式选择最少量 patch 并去除时空冗余，实现 4×-100× token 压缩和最高 19× ViT 加速，使 MLLM 可扩展到 1K 帧 4K 分辨率视频。

## 研究背景与动机

**视频存在大量时空冗余**：视频中静态背景在连续帧间高度重复，但现有 MLLM 对每一帧的每个像素一视同仁地处理，浪费大量计算。

**现有 token 压缩仅作用于模型内部**：已有方法（如 ToMe、LongVU、STORM 等）仅在 ViT 内部或 ViT-LLM 之间剪枝/合并 token，ViT 仍需处理完整视频，形成效率瓶颈。

**启发式剪枝效果受限**：基于注意力分数等启发式方法的效果不如学习型方法；而涉及搜索和推理的方法额外开销大，限制了可扩展性。

**长视频高分辨率需求迫切**：真实应用（监控、自动驾驶、机器人）需要处理分钟级 4K 视频，但现有模型在高时空分辨率下因计算成本无法扩展。

**人类视觉的启示**：人眼通过"扫视"（saccade）选择性地注视运动物体和细节区域，跳过静态背景，实现高效实时场景理解。

**缺乏高分辨率长视频评测基准**：现有 benchmark（VideoMME、MLVU 等）仅关注长视频但不要求高分辨率，无法评估模型在高分辨率长视频上的能力。

## 方法详解

### 整体框架

AutoGaze 是一个 3M 参数的轻量模块（卷积编码器 + 自回归 Transformer 解码器），位于 ViT **之前**。给定视频，AutoGaze 逐帧编码并自回归地解码 patch 索引：

1. **帧编码**：卷积编码器对当前帧提取特征
2. **自回归注视**：解码器基于当前帧特征和历史帧/已选 patch 信息，逐步输出 patch 索引（词表为 {1,...,V}）
3. **自动停止**：解码器附带一个 head 预测"当前已选 patch 能否将帧重建到阈值内"，一旦预测损失低于用户设定阈值 ε 则停止当前帧的注视
4. **多尺度注视**：词表包含多个尺度的 patch，模型可为低细节区域选粗尺度、高细节区域选细尺度
5. **多 token 预测**：采用多头同时输出多个 patch 索引，加速推理

### 关键设计

- **重建目标**：使用定制 VideoMAE（block-causal attention）作为重建器，以像素重建损失 + 感知损失的加权和衡量重建质量
- **跨帧信息传递**：解码第 t 帧时可参考 1..t 帧的特征和 1..t-1 帧的已选 patch，避免重复选择冗余 patch
- **任意分辨率/时长推理**：将视频切分为 16×224×224 的时空 tile，每个 tile 独立运行 AutoGaze 后合并结果
- **ViT 适配**：通过对不同尺度分别做 patch embedding 再拼接送入 ViT，使标准图像 ViT 接受多尺度 patch 输入；同时将图像 ViT 扩展为视频 ViT（16 帧 token 拼接为一个序列）

### 训练流程（两阶段）

**阶段一：NTP 预训练**

- 贪心搜索收集约 25 万条近似最优注视序列（800K 视频，16 帧 224 分辨率）
- 使用标准 next-token prediction 交叉熵损失训练 patch 索引预测
- 同时用 ℓ₂ 损失监督每步的重建损失预测

**阶段二：RL 后训练**

- 采用简化的 on-policy GRPO 算法，以重建损失为 reward
- 优势函数 A 为未来帧负重建损失的折扣和，在 group 内归一化
- 使重建损失预测进一步校准

### 损失函数

- 预训练：$L_{NTP} = -\sum_{t,k} \log \pi_\theta(\tilde{p}_k^t \mid X^{1:t}, \tilde{p}_{1:k-1}^t)$ + 重建损失预测的 ℓ₂ 损失
- 后训练：$L_{GRPO} = -\sum_{t,k} \frac{\pi_\theta(p_k^t)}{\pi_{\theta_{detached}}(p_k^t)} A_k^t$ + 重建损失预测监督

## 实验

### 主要结果

| 模型 | 最大帧数 | 最大分辨率 | VideoMME (w/o sub) | VideoMME (w/ sub) | MLVU | HLVid |
|---|---|---|---|---|---|---|
| GPT-4o | - | - | 71.9 | 77.2 | 64.6 | 49.3 |
| Qwen2.5-VL-7B | 48 | 896 | 65.1 | 71.6 | 70.2 | 48.1 |
| VideoChat-Flash | 10000 | 448 | 65.3 | 69.7 | 74.7 | 46.6 |
| NVILA-8B-Video | 256 | 448 | 64.2 | 70.0 | 70.1 | 42.5 |
| **+ AutoGaze** | **1024** | **3584** | **67.0** | **71.8** | **71.6** | **52.6** |

NVILA + AutoGaze 在 VideoMME 上达到 67.0%，HLVid 上从 42.5% 提升至 52.6%（+10.1%），超过 GPT-4o 的 49.3%。

### 与 token 压缩方法对比

在 128 帧、6.25% 选择率下：

| 方法 | ViT 延迟 | LLM 延迟 | VideoMME |
|---|---|---|---|
| 无压缩 | 2.20s | 1.42s | 53.4 |
| ToMe | 2.23s | 0.11s | 51.5 |
| STORM | 2.18s | 0.15s | 52.7 |
| LongVU | 2.17s | 0.12s | 52.2 |
| **AutoGaze** | **0.55s** | **0.10s** | **52.3** |

AutoGaze 是唯一能显著降低 ViT 延迟的方法（4× 加速），同时保持 LLM 端最低延迟。

### 消融实验

**训练流程消融**：仅 NTP 预训练 gazing ratio 0.102，仅 RL 后训练 0.209，两者结合 0.094（最优）。

**模型设计消融**：多 token 预测 k=10 在效率和质量间平衡最佳（0.193s 延迟，gazing ratio 0.094）；多尺度注视将 gazing ratio 从 0.220 降至 0.094，效率提升 2.3×。

### 关键发现

- **注视偏好运动区域**：光流越大的 patch 被选中概率越高，各尺度均成立
- **细粒度尺度对应高细节区域**：细尺度选择的 patch 拉普拉斯方差更高（相关系数 ρ=0.12, p<0.001）
- **OOD 泛化性强**：在 CCTV、机器人操作、风格迁移等分布外视频上仍能稳健追踪变化区域
- **30FPS 4K 视频仅需约 1% patch**：冗余度随 FPS 和分辨率增加而增大

## 亮点

- **在 ViT 之前** 去除冗余，真正解决了效率瓶颈问题，而非只优化 LLM 端
- 仅 3M 参数，开销极低，即插即用
- 自回归框架统一了 patch 选择和停止决策，无需手工阈值调参
- 两阶段训练（NTP + RL）设计简洁有效，RL 阶段在 NTP 基础上进一步提升 ~10%
- 提出首个高分辨率长视频 QA 基准 HLVid（268 QA，5 分钟 4K 视频）
- 可扩展到 1K 帧 4K 分辨率，HLVid 上超越 GPT-4o

## 局限性

- 重建目标为代理任务，可能与下游语义理解目标不完全一致
- 仅在 NVILA-8B 上验证集成效果，未验证对其他 MLLM（如 Qwen2.5-VL、InternVL）的通用性
- 多尺度注视需要修改 ViT 的 patch embedding 和位置编码，对现有模型有一定侵入性
- HLVid 仅 268 个 QA，规模较小，评测可靠性有待更大规模验证
- tile 化处理可能在 tile 边界引入信息不连续

## 相关工作

- **视频理解 MLLM**：NVILA、Qwen2.5-VL、LongVILA、VideoChat-Flash 等扩展到长视频但受限于低分辨率
- **空间 token 压缩**：ToMe（token 合并）、VisionZip、FastV（注意力剪枝）——仅作用于模型内部
- **时空 token 压缩**：STORM（时空路由）、LongVU（跨帧去重）、FastVID——同样不减少 ViT 输入
- **自适应 tokenization**：FlexTok、AnyRes 等灵活分配 token 数量，但 tokenizer 本身开销大
- **视频表示学习**：V-JEPA2、VideoMAE 等自监督方法提供重建能力基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 在 ViT 之前做自回归 patch 选择是全新思路
- 实验充分度: ⭐⭐⭐⭐⭐ — 行为分析、效率对比、SOTA 对比、消融实验和新 benchmark 均齐全
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，图表丰富，motivation 充分
- 价值: ⭐⭐⭐⭐⭐ — 解决视频 MLLM 的核心效率瓶颈，实用性极强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)

<!-- RELATED:END -->
