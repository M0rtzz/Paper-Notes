---
title: >-
  [论文解读] VIRTUE: Visual-Interactive Text-Image Universal Embedder
description: >-
  [ICLR 2026][图像分割][提示学习] 提出 VIRTUE，将分割模型 SAM2 与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务（+3.1%-8.5%）和 5 个 SCaR 任务（+15.2%-20.3%）上均达到 SOTA。
tags:
  - ICLR 2026
  - 图像分割
  - 提示学习
  - embedding model
  - SAM2
  - VLM
  - visual-interactive
  - retrieval
---

# VIRTUE: Visual-Interactive Text-Image Universal Embedder

**会议**: ICLR 2026  
**arXiv**: [2510.00523](https://arxiv.org/abs/2510.00523)  
**代码**: [GitHub](https://github.com/sony/virtue)  
**领域**: 图像分割（多模态嵌入/视觉交互）  
**关键词**: visual prompt, embedding model, SAM2, VLM, visual-interactive, retrieval

## 一句话总结

提出 VIRTUE，将分割模型 SAM2 与 VLM 结合构建视觉交互式通用嵌入器，支持用户通过点/框/掩码指定兴趣区域产生实体级+全局级联合嵌入，并构建百万级 SCaR 基准评估视觉交互检索能力，在 36 个 MMEB 任务（+3.1%-8.5%）和 5 个 SCaR 任务（+15.2%-20.3%）上均达到 SOTA。

## 研究背景与动机

**嵌入模型的交互局限**：现有 VLM 嵌入模型（VLM2Vec/GME/LamRA）仅支持文本指令交互，缺乏视觉交互能力（点/框/掩码等 visual prompt）

**视觉提示的价值**：在生成模型中已广泛使用（SAM、GroundingDINO），但嵌入模型尚未探索。视觉提示可提供精确空间定位用于细粒度理解

**裁剪的不足**：直觉的 ROI 裁剪方案会丢失全局场景上下文——"桌上的沙拉叉"裁剪后失去"桌"的信息，导致在需要组合推理的检索中失败

**同一图像不同实体需求**：同一图像中的狗和猫需要不同嵌入，但整体嵌入无法区分

**缺乏评估基准**：没有评估视觉交互嵌入能力的公开基准

## 方法详解

### 架构: SAM2 + VLM (Qwen2-VL) + 分割-语言连接器

### 三路嵌入融合

1. **分割嵌入 $H_s$**：SAM2 的 prompt encoder 处理视觉提示（点/框/掩码）→ mask decoder 生成 64×64 特征图 $F_s$ → Conv2D 压缩到 $|S|$ 个 token → MLP 投影到 LLM 维度 $d$
2. **视觉嵌入 $H_v$**：VLM 的 vision encoder 提取全局上下文
3. **文本嵌入 $H_t$**：LLM 的文本嵌入层处理指令文本

### 无视觉提示时的策略

均匀采样 $N$ 个点作为替代输入 SAM2，利用 SAM2 的自动分割能力提取多实体级特征，确保在传统非交互任务上也有增益

### 训练方案

- 拼接 $[H_s, H_v, H_t]$ → LLM → 取最后 token 的 hidden state → InfoNCE 对比学习
- SAM2 和 vision encoder 冻结，仅训练 LoRA（rank=8）+ 分割-语言连接器（从头训练）
- 20 个 MMEB 训练集，batch size 1024

### SCaR Benchmark（百万级新基准）

- 任务：给定图像+ROI 框，检索描述该实体在全局场景中的标题
- 来源：RefCOCO+/RefCOCOg/VisualGenome/COCO-Stuff/ADE20K 五个数据集
- 规模：957K 训练 + 47K 评估样本
- 负样本：GPT-4V 生成——对象/关系/场景三种元素替换策略，每样本 9 个干扰项
- 多阶段质量控制：GPT-4V 验证 + WordNet 同义词检测 + 人工审查

## 实验关键数据

### MMEB Overall (36 tasks)

| 模型 | 参数 | IND | OOD | Overall |
|------|------|-----|-----|---------|
| VLM2Vec-2B | 2B | 60.7 | 57.3 | 59.7 |
| **VIRTUE-2B** | 2B | **69.7** | **58.8** | **64.8** |
| VLM2Vec-7B | 7B | 71.4 | 58.1 | 65.5 |
| UniME-7B | 7B | 68.4 | 57.9 | 66.6 |
| **VIRTUE-7B** | 7B | **74.4** | **61.4** | **68.6** |

### SCaR (5 visual-interactive tasks)

| 模型 | RefCOCOg | RefCOCO+ | COCO-Stuff | VG | ADE20K |
|------|---------|----------|-----------|-----|--------|
| VLM2Vec-7B | 56.2 | 52.1 | 45.3 | 42.8 | 38.1 |
| **VIRTUE-7B** | **75.1** | **70.8** | **62.5** | **59.4** | **55.9** |

### 消融实验

| 配置 | MMEB Overall | SCaR Avg | 说明 |
|------|-------------|---------|------|
| 无分割嵌入 | 65.5 | 52.1 | VLM2Vec 基线 |
| + 裁剪 ROI | 65.8 | 54.3 | 裁剪帮助有限 |
| + 全SAM2特征 | 67.1 | 63.2 | 实体级信息有效 |
| **+ 完整 VIRTUE** | **68.6** | **68.2** | 最佳 |

### 关键发现

- 分割嵌入在非交互场景下（均匀采样点）也提供实体级信息增益
- 即使在传统 MMEB 任务（无视觉提示）上也有 3.1%-8.5% 提升
- SAM2 作为结构化先验比裁剪更精确地捕捉实体语义（避免包含背景、跨实体等问题）

## 亮点与洞察

- **新交互范式**：首次将视觉提示（点/框/掩码）引入嵌入模型，定义了全新问题空间
- **SCaR 基准**：百万级数据 + GPT-4V 生成高质量干扰项 + 多阶段过滤，是可靠的评估工具
- **兼顾通用性**：无视觉提示时自动采样点策略保证了在传统任务上的竞争力
- **实用性强**：SAM2 冻结 + LoRA 微调，训练成本可控

## 局限与展望

- SAM2 增加推理计算开销（额外的分割前向传播）
- SCaR 仅评估 I2T 检索，未覆盖 I2I 视觉交互场景
- 均匀采样点的自动策略可能不是最优的实体发现方式（可考虑自动目标检测驱动）
- 分割-语言连接器需从头训练，增加了训练复杂度

## 相关工作与启发

- **VLM2Vec/GME/LamRA**：VLM 嵌入模型基线，仅支持文本交互
- **CLIP/SigLIP/OpenCLIP**：双塔嵌入模型，全局匹配无区域感知
- **SAM2**：作为实体级特征提取器被引入嵌入学习

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视觉交互嵌入 = 全新问题定义 + 新基准
- 实验充分度: ⭐⭐⭐⭐⭐ 36+5 任务 + 大量消融 + 两种规模模型
- 写作质量: ⭐⭐⭐⭐ 清晰系统，基准构建过程透明
- 价值: ⭐⭐⭐⭐⭐ 开辟视觉交互嵌入新方向 + 高质量基准

<!-- RELATED:START -->

## 相关论文

- [Universal Multi-Domain Translation via Diffusion Routers](universal_multi-domain_translation_via_diffusion_routers.md)
- [VSC: Visual Search Compositional Text-to-Image Diffusion Model](../../ICCV2025/segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [Live Interactive Training for Video Segmentation](../../CVPR2026/segmentation/live_interactive_training_for_video_segmentation.md)
- [RegionReasoner: Region-Grounded Multi-Round Visual Reasoning](regionreasoner_region-grounded_multi-round_visual_reasoning.md)
- [Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)

<!-- RELATED:END -->
