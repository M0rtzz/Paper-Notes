---
title: >-
  [论文解读] Imagine How To Change: Explicit Procedure Modeling for Change Captioning
description: >-
  [ICLR 2026][变化描述] 提出 ProCap 框架，将变化描述从静态图像对比较重新定义为动态过程建模：第一阶段通过帧插值和掩码重建训练过程编码器学习时空变化动力学，第二阶段用可学习过程查询隐式推断变化过程，在三个数据集上超越 SOTA。
tags:
  - ICLR 2026
  - 变化描述
  - 过程建模
  - 帧插值
  - 掩码重建
  - 可学习查询
  - 视觉语言
---

# Imagine How To Change: Explicit Procedure Modeling for Change Captioning

**会议**: ICLR 2026  
**arXiv**: [2603.05969](https://arxiv.org/abs/2603.05969)  
**代码**: [GitHub](https://github.com/BlueberryOreo/ProCap)  
**领域**: llm_nlp  
**关键词**: 变化描述, 过程建模, 帧插值, 掩码重建, 可学习查询, 视觉语言  

## 一句话总结

提出 ProCap 框架，将变化描述从静态图像对比较重新定义为动态过程建模：第一阶段通过帧插值和掩码重建训练过程编码器学习时空变化动力学，第二阶段用可学习过程查询隐式推断变化过程，在三个数据集上超越 SOTA。

## 研究背景与动机

**变化描述（Change Captioning）**是生成文本描述两张相似图像差异，应用于遥感监测、医学诊断、城市规划和工业质控。

现有方法根本局限：

**静态图像对建模**：仅比较"前"和"后"，忽略变化的**动态过程**

**缺失时间线索**：无法理解变化是"如何发生的"

**编码器局限**：各种差异提取器和对齐机制，但都是空间比较而非时空建模

**关键洞察**：两张图像间存在隐含的连续过渡过程，包含丰富的时空动力学。例如物体位移可通过中间帧揭示运动轨迹。

## 方法详解

### 整体框架

ProCap 两阶段：
- **第一阶段：显式过程建模（EPM）**：学习变化过程的时空动力学
- **第二阶段：隐式过程描述（IPC）**：用可学习查询代替显式中间帧

### 第一阶段：显式过程建模

**过程生成模块**：预训练帧插值模型递归生成中间帧。FI 模型预测双向光流，生成扭曲图像对，Transformer 产生软掩码和残差合成中间帧。

**置信度帧采样**：选"语义等距"关键帧——与起始帧和结束帧语义距离相等的帧得分最高。平方差项确保无论更接近哪端都被惩罚。

**过程建模模块**：Transformer 编码器 + 图像 tokenizer。输入含视觉流（patch 特征）、文本流（caption token）和特殊 token（帧一致性 + 跨模态对齐）。

**多粒度掩码**（4 种随机选一）：
1. 整帧掩码：迫使用 caption 重建
2. 随机 patch 掩码：学习分布式表示
3. 块内掩码：学习局部纹理
4. 块外掩码：学习区域-场景关系

### 损失函数

L_PRO = L_msm + L_align + L_csy

- L_msm：掩码位置预测离散 token（交叉熵）
- L_align：区分匹配/不匹配的 caption-过程对
- L_csy：区分正常/打乱的帧序列

### 第二阶段：隐式过程描述

核心：k*n_I 个可学习过程查询替代显式中间帧。利用第一阶段学到的时空理解，从图像对隐式推断变化过程。Transformer 解码器翻译为 caption。推理无需帧插值。

### 训练策略

第二阶段自回归损失端到端训练。推理仅额外 k*n_I 参数（k=2 时开销可忽略）。

## 实验关键数据

### 主实验

**三数据集 SOTA 对比**（表1，CIDEr）：

| 方法 | CLEVR-Change | Spot-the-Diff | Image-Editing |
|------|-------------|---------------|---------------|
| DUDA (2019) | 112.3 | 32.5 | 22.8 |
| SCORER+CBR (2023) | 126.8 | 38.9 | 33.4 |
| MCT-CCDiff (2025) | 131.7 | 41.7 | 38.3 |
| FINER (LLM, 2024) | 137.2 | 61.8 | 50.5 |
| LLaVA-1.5+RP (LLM) | — | 43.2 | 60.9 |
| **ProCap (Ours)** | **135.6** | **42.7** | **40.6** |

非 LLM 方法中全面领先，与 LLM 方法差距显著缩小。

### 消融实验

**组件消融**（CLEVR-Change CIDEr）：

| EPM | IPC | k | CIDEr |
|-----|-----|---|-------|
| N | N | 0 | 108.4 |
| Y | N | 0 | 112.7 |
| N | Y | 1 | 106.2 |
| Y | Y | 1 | **128.5** |

两者结合 CIDEr +20.1（108.4 -> 128.5）。

**查询长度 k**：

| k | TPS | CIDEr |
|---|-----|-------|
| 1 | 766 | 128.5 |
| 2 | 699 | **135.6** |
| 4 | 461 | 128.7 |
| 7 | 271 | 130.5 |

k=2 最优且效率合理。

**损失消融**（CLEVR / StD CIDEr）：

| msm | align | csy | CLEVR | StD |
|-----|-------|-----|-------|-----|
| Y | N | N | 127.5 | 29.7 |
| Y | N | Y | 128.6 | 36.3 |
| Y | Y | Y | **135.6** | **42.7** |

完整组合在 StD 上比仅 msm 提升 13.0。

### 关键发现

1. **过程建模远优于静态比较**
2. **预训练+查询协同**：预训练提供时空理解，查询提供高效推理
3. **轻量但强大**：非 LLM 接近甚至超越 LLM 方法
4. **跨场景泛化**：合成/自然/开放三类均强劲

## 亮点与洞察

1. **范式转移**：从"静态空间比较"到"动态时空过程建模"
2. **两阶段精巧**：训练用显式帧，推理用隐式查询——兼顾表征和效率
3. **置信度采样创意**：选"语义等距"帧聚焦关键时刻
4. **多粒度掩码**：帧级到 patch 级多尺度理解
5. **非 LLM 竞争力**：证明架构创新而非规模也能显著提升

## 局限与展望

1. **帧插值质量依赖**：FI 质量直接影响上限
2. **假设变化可插值**：物体突然出现/消失无法通过光流建模
3. **LLM 解码器缺失**：与 LLM 结合可能更大提升
4. **仅限两张图像**：未扩展到视频变化描述
5. **置信度采样需预定义相似度函数**

## 相关工作与启发

- **DUDA** [Park et al., 2019]：开创性框架——ProCap 根本扩展范式
- **FINER** [Zhang et al., 2024]：LLM 增强——ProCap 无需 LLM 达可比性能
- **VideoMAE** [Han et al., 2022]：视频掩码自编码——ProCap 过程建模受启发
- **VQGAN** [Esser et al., 2021]：图像 tokenizer——用于重建目标
- **RIFE** [Lu et al., 2022]：帧插值——用于显式过程生成

## 评分

| 维度 | 评分 |
|------|------|
| 理论深度 | ⭐⭐⭐ |
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体评价 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Optimal Online Change Detection via Random Fourier Features](../../NeurIPS2025/llm_pretraining/optimal_online_change_detection_via_random_fourier_features.md)
- [\[ICLR 2026\] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization](recon_robust_symmetry_discovery_via_explicit_canonical_orientation_normalization.md)
- [\[ICLR 2026\] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](../../NeurIPS2025/llm_pretraining/how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[ICML 2025\] How to Synthesize Text Data without Model Collapse?](../../ICML2025/llm_pretraining/how_to_synthesize_text_data_without_model_collapse.md)

<!-- RELATED:END -->
