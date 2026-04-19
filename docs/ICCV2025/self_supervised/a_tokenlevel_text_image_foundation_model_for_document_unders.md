---
title: >-
  [论文解读] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)
description: >-
  [ICCV 2025][自监督学习][token-level对齐] 提出首个 token 级别文本图像基础模型 TokenFD，通过在 2000 万图像、18 亿 BPE token-mask 对上进行 token 级视觉-语言对齐预训练，实现 image-as-text 语义能力，并基于此构建文档理解 MLLM TokenVL，在 OCRBench 上得分 860（8B 组最高），在 DocVQA 等十项 VQA 任务上平均提升 8.8%。
tags:
  - ICCV 2025
  - 自监督学习
  - token-level对齐
  - 视觉基础模型
  - 文档理解
  - OCR
  - 多模态大模型
---

# A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)

**会议**: ICCV 2025  
**arXiv**: [2503.02304](https://arxiv.org/abs/2503.02304)  
**代码**: [Token-family/TokenFD](https://github.com/Token-family/TokenFD)  
**领域**: 自监督/表示学习  
**关键词**: token-level对齐, 视觉基础模型, 文档理解, OCR-free, 多模态大模型

## 一句话总结

提出首个 token 级别文本图像基础模型 TokenFD，通过在 2000 万图像、18 亿 BPE token-mask 对上进行 token 级视觉-语言对齐预训练，实现 image-as-text 语义能力，并基于此构建文档理解 MLLM TokenVL，在 OCRBench 上得分 860（8B 组最高），在 DocVQA 等十项 VQA 任务上平均提升 8.8%。

## 背景与动机

1. **领域现状**: 通用视觉基础模型（VFM）如 CLIP、DINO、SAM 被广泛用于多模态大模型的视觉编码器。但这些模型都是在图像级（CLIP/DINO）或像素级（SAM）监督下训练的。

2. **现有痛点**: 对于包含密集小文字的文档图像，图像级 VFM 无法精确感知细粒度文本内容，导致下游 OCR 相关任务中出现基本的感知错误。有些方法尝试引入 SAM 作为额外高分辨率编码器，但双 VFM 组合导致 token 数量翻倍，既昂贵又缺乏灵活性。

3. **核心矛盾**: 当前不存在 token 粒度的细粒度文本图像基础模型。从图像级到像素级之间存在一个关键空白——token 级别的对齐，即每个 BPE 子词与其在图像中对应区域的精确映射。

4. **本文目标** (1) 构建首个 token 级图像文本数据集；(2) 训练首个 token 级 VFM；(3) 将其应用于文档理解 MLLM 的构建。

5. **切入角度**: 利用 BPE tokenizer 将文本拆分为子词，为每个子词构造像素级 mask，实现 token 粒度的视觉-语言对齐——比 CLIP 的图像级对齐精细得多，比 SAM 的像素级分割有更强的语义。

6. **核心 idea**: 在 token（BPE 子词）粒度上对齐视觉特征和语言嵌入，使 VFM 获得"图像即文本"的语义能力。

## 方法详解

### 整体框架

三层产品系列：(1) **TokenIT** 数据集——2000 万图像 + 18 亿 token-mask 对；(2) **TokenFD** 基础模型——ViT + 反卷积上采样 + token 级对比学习，实现 image-as-text 对齐；(3) **TokenVL** MLLM——以 TokenFD 为视觉编码器 + InternLM 为 LLM，两阶段训练（token 对齐预训练 + 指令微调）。

### 关键设计

1. **TokenIT 数据集构建**:

    - 功能：构建首个 token 级图像文本数据集
    - 核心思路：四步流水线——①文本图像分割（自然场景用微调 SAM，文档用无监督聚类）；②文本识别（SOTA OCR 获取转录）；③BPE Tokenizer 拆分子词；④将字符级 mask 合并为 token 级 mask。每个样本包含原图、mask 图和 JSON 文件（记录 BPE token 信息）
    - 设计动机：覆盖自然场景、文档、表格、图表、代码、GUI 等多种类型，三轮人工检验历时 4 个月确保质量

2. **TokenFD 预训练**:

    - 功能：实现 token 级视觉-语言对齐
    - 核心思路：输入图像经 ViT 编码器提取特征，两层反卷积上采样 4x 到更高分辨率，再线性投射到与语言嵌入相同维度。对每个 BPE token-mask 对，通过 mean pooling 在 mask 区域提取 token 级视觉特征 $\mathbf{t}_i$，用简单的 token 嵌入层（无需复杂文本编码器）获取语言嵌入 $\mathbf{e}_i$。三个损失函数联合优化：距离损失 $\mathcal{L}_{dis}$（L1 距离）、相似性损失 $\mathcal{L}_{sim}$（余弦相似度）、sigmoid 对比损失 $\mathcal{L}_{sig}$（类 SigLIP）
    - 设计动机：不同于 CLIP 需要复杂文本编码器，TokenFD 直接用 token 嵌入层对齐——因为操作粒度已经是 BPE 子词，无需上下文编码

3. **TokenVL（MLLM）**:

    - 功能：构建文档理解 MLLM
    - 核心思路：两阶段训练。**阶段1 LLM-guided Token Alignment**：自回归 VQA 训练（隐式对齐）+ token 对齐分支（显式空间对齐，从 LLM 中间层提取视觉-语言特征在 token 级对齐）。**阶段2 SFT**：取消 token 对齐分支避免推理开销，在 VQA 数据上全参数微调。还设计了 token abstractor 用可学习 token 在每个窗口内自适应压缩视觉特征
    - 设计动机：token 对齐分支在训练时强制 LLM 更多参考图像内容而非依赖语义上下文推测，推理时移除无额外开销

### 损失函数 / 训练策略

- TokenFD 预训练：AdamW + cosine schedule，基础 lr=5e-4，在 TokenIT 上训练 2 epochs，64 张 H800 GPU
- TokenVL 阶段1：冻结 InternLM，训练 TokenFD + token abstractor，lr=2e-4，1 epoch
- TokenVL 阶段2：全参数可训练，lr=1e-5

## 实验关键数据

### 主实验（TokenFD 零样本/线性探测）

| 任务 | 方法 | 零样本 avg | 线性探测 avg |
|------|------|-----------|-------------|
| 文本分割 | CLIP-L-1024px | 15.81 | - |
| | SAM-H | - | 34.51 |
| | InternViT2.5 | - | 42.21 |
| | **TokenFD** | **34.59** | **48.77** |
| 文本检索 | CLIP-L | - | 3.62 |
| | InternViT2.5 | - | 13.29 |
| | **TokenFD** | - | **63.62** |

### TokenVL 文档理解

| 模型 | 参数 | DocVQA | InfoVQA | ChartQA | TextVQA | OCRBench |
|------|------|--------|---------|---------|---------|----------|
| InternVL2.5-2B | 2B | 88.7 | 60.9 | 79.2 | 74.3 | 804 |
| **TokenVL-2B** | **2B** | **89.9** | **61.0** | **81.1** | **76.4** | **821** |
| InternVL2.5-8B | 8B | 93.0 | 77.6 | 84.8 | 79.1 | 822 |
| **TokenVL-8B** | **8B** | **94.2** | **76.5** | **86.6** | **79.9** | **860** |

### 消融实验

| 配置 | DocVQA | ChartQA | 说明 |
|------|--------|---------|------|
| w/o token abstractor, w/o TA | 93.1 | 86.5 | 基线 |
| w/ token abstractor, w/o TA | 93.8 | 86.5 | +token abstractor |
| w/ token abstractor, w/ TA | **94.2** | **86.6** | 完整模型 |

### 关键发现

- TokenFD 在零样本文本分割上比 CLIP 高 18.78%，在文本检索上比 InternViT2.5 高 50%+——token 级对齐对文本相关任务的优势压倒性
- TokenVL-8B OCRBench 得分 860，比 InternVL2.5 高 38 分，比 TextHawk2 高 76 分——token 级 VFM 显著提升文档理解能力
- Token alignment 分支在全图文本识别中显著降低编辑距离，证实显式空间对齐的有效性

## 亮点与洞察

- **VFM 粒度谱系的补全**：从 CLIP（图像级）→ SAM（像素级）→ TokenFD（token 级），形成完整的视觉基础模型粒度谱系。Token 级恰好填充了语义和空间之间的关键空白
- **简单但有效的语言编码**：不需要 CLIP 那样的复杂文本编码器，一个简单 token 嵌入层就够了——因为粒度已经是 BPE 子词，不需要上下文理解
- **数据工程的重要性**：18 亿高质量 token-mask 对的构建历时 4 个月三轮审核，体现了基础模型研究中数据质量的关键作用

## 局限与展望

- 预训练需要 64 张 H800 GPU，计算资源门槛较高
- 仅提供 2B 和 8B 两个 TokenVL 版本，未探索更大规模模型
- Token 级 mask 的质量依赖上游 OCR 和分割模型的准确性
- 仅在文档/OCR 相关任务上验证，对通用视觉理解的影响未探索

## 相关工作与启发

- **vs CLIP**: CLIP 图像级对齐在密集文本场景精度不足，TokenFD 的 token 级对齐在文本分割上零样本超 CLIP 两倍
- **vs InternVL2.5**: InternVL2.5 用 InternViT 作为通用 VFM，TokenFD 专为文本图像设计，在文档任务上一致超越
- **vs SAM**: SAM 像素级分割但缺乏语义能力，TokenFD 既有空间精度又有语义对齐

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个 token 级文本图像基础模型，填补了 VFM 粒度谱系的关键空白
- 实验充分度: ⭐⭐⭐⭐⭐ 文本分割/检索/VQA/OCRBench 全面覆盖，消融详细
- 写作质量: ⭐⭐⭐⭐ 结构清晰，产品系列层次分明
- 价值: ⭐⭐⭐⭐⭐ 对文档理解领域有重大推动，数据集+模型+代码全开源

<!-- RELATED:START -->

## 相关论文

- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](../../CVPR2026/self_supervised/d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](../../ECCV2024/self_supervised/marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2025/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)

<!-- RELATED:END -->
