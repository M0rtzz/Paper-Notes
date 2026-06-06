---
title: >-
  [论文解读] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning
description: >-
  [NEURIPS2025][LLM安全][多模态] 提出 HoloLLM，首次将 LiDAR、红外、毫米波雷达、WiFi 等稀有传感模态接入多模态大语言模型（MLLM），通过 Universal Modality-Injection Projector（UMIP）在数据稀缺条件下实现传感模态与文本的高效对齐…
tags:
  - "NEURIPS2025"
  - "LLM安全"
  - "多模态"
  - "Human Sensing"
  - "LiDAR"
  - "mmWave"
  - "WiFi"
  - "Modality Alignment"
---

# HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning

**会议**: NEURIPS2025  
**arXiv**: [2505.17645](https://arxiv.org/abs/2505.17645)  
**代码**: [NTUMARS/HoloLLM](https://github.com/NTUMARS/HoloLLM)  
**领域**: 自动驾驶  
**关键词**: Multimodal LLM, Human Sensing, LiDAR, mmWave, WiFi, Modality Alignment  

## 一句话总结
提出 HoloLLM，首次将 LiDAR、红外、毫米波雷达、WiFi 等稀有传感模态接入多模态大语言模型（MLLM），通过 Universal Modality-Injection Projector（UMIP）在数据稀缺条件下实现传感模态与文本的高效对齐，在人体动作问答和描述任务上较现有 MLLM 提升约 30%。

## 背景与动机
智能家居中的具身智能体（家用机器人、智能家电等）需要通过多种传感输入理解人体行为，并以自然语言进行交互。现有 Vision-Language Model（VLM）虽然在视觉语言感知上表现出色，但仅依赖视觉模态在遮挡、弱光、隐私场景下会失效。例如，当有人在障碍物后方摔倒时，摄像头完全无法检测，而毫米波雷达和 WiFi 信号则不受影响。因此，将 LiDAR、红外、毫米波、WiFi 等互补传感模态引入 MLLM 是实现鲁棒人体感知的关键方向。

## 核心问题
将稀有传感模态接入 MLLM 面临两大挑战：

1. **数据稀缺**：RGB/深度图像有数百万级网络配对数据用于预训练 projector，而 mmWave、WiFi 等传感数据仅有实验室采集的数千个样本，无法通过大规模预训练实现模态-文本对齐
2. **信号异质性**：不同传感器的物理设计（波长、频率）差异巨大，信号表征与常见模态截然不同，通用 Transformer 编码器难以学到有区分力的特征

## 方法详解

### 整体框架
HoloLLM 由四部分组成：(1) CLIP 视觉编码器作为通用编码器生成预对齐初始嵌入；(2) 针对每种模态设计的 Tailored Encoder 提取细粒度特征；(3) UMIP 将两种特征融合为与文本对齐的多模态 token；(4) LLaMA2-7B 作为 LLM 接收多模态 token 和文本指令进行推理。

### Universal Modality-Injection Projector（UMIP）
UMIP 是本文核心模块，采用"粗到细"的渐进式特征增强策略：

- **粗粒度查询生成**：利用 CLIP ViT-L（经 LAION 图像-文本对比预训练）作为统一编码器，为任意模态生成初始嵌入 $\mathbf{Y}_{CLIP}^m$。由于 CLIP 天然具备文本对齐能力和跨模态迁移性，这些嵌入可视为"预对齐"的。通过自适应平均池化将其下采样为固定数量的查询 $\mathbf{Q}^m$
- **细粒度键值生成**：每种模态配备专用编码器（视觉/深度/红外用 ResNet18，LiDAR/mmWave 用 PointNet，RFID 用 1D Temporal ResNet18，WiFi 用 MetaFi），提取异质特征 $\mathbf{Y}_T^m$ 并转换为 Key/Value
- **迭代式粗到细交叉注意力**：UMIP 包含 $L=8$ 个 block，每个 block 依次执行自注意力、交叉注意力（查询从细粒度 Key/Value 中自适应提取文本对齐特征）、前馈网络（将增强后的查询投射到 LLM 文本空间），输出作为下一个 block 的增强查询
- 最终输出 $\mathbf{Z}^m = \text{MLP}(\mathbf{Q}_L^m)$，维度从 CLIP 的 1024 映射到 LLM 的 4096

### 两阶段训练策略
- **阶段一**：冻结 CLIP，用 HAR 分类损失预训练各模态的 Tailored Encoder
- **阶段二**：冻结 Tailored Encoder，微调 Tokenizer 和 UMIP，联合优化分类损失与 next-token prediction 损失

### 与现有 Projector 设计的对比
- **Modality-Specific Projector**（PointLLM、ImageBind-LLM 等）：每种模态一个编码器+一个 projector，需要大量模态-文本配对数据预训练
- **Universal Projector**（OneLLM）：统一编码器+统一 projector，不需要模态专用预训练，但缺乏异质特征捕获能力
- **UMIP（本文）**：统一编码器生成预对齐初始嵌入 + 专用编码器提取细粒度特征 + 渐进式交叉注意力融合，兼顾数据高效和特征区分力

### 数据构建流水线
针对 MM-Fi 和 XRF55 两个多模态人体传感数据集，设计了人类-VLM 协作标注流水线：
- **Action QA**：人类专家标注 5 个问题模板，GPT-4o 改写扩充到 15 个，随机采样生成选项式问答对
- **Action Caption**：均匀采样少量样本由人类标注，其余通过 LLaVA-Video 进行 in-context learning 自动生成描述
- 所有文本标注在同一数据样本的不同模态间共享，降低标注成本

## 实验关键数据

### 数据集与评测设置
- 数据集：MM-Fi（Video, Depth, LiDAR, mmWave, WiFi 五种模态）和 XRF55（Video, Depth, Infrared, RFID, WiFi 五种模态）
- 三种评测设置：Random Split、Cross-Subject、Cross-Environment
- 指标：Action QA/Recognition 用 Accuracy，Action Caption 用 METEOR

### 主要结果（MM-Fi，Cross-Environment 最难设置）

| 方法 | Action QA Avg | Action Caption Avg |
|------|:---:|:---:|
| Tokenpacker | 4.6% | 3.8% |
| Honeybee | 1.7% | 10.4% |
| OneLLM | 5.0% | 9.3% |
| ImageBind | 16.7% | 17.3% |
| **HoloLLM** | **56.4%** | **22.6%** |

HoloLLM 在 QA 任务上比最强基线 ImageBind 高出约 40 个百分点。在 Random Split 设置下 QA 平均达 86.5%，比 ImageBind 高 40.3%。

### 消融实验（Cross-Environment）
- Baseline（CLIP + Q-Former）：MM-Fi Action QA 仅 6.2%
- +Tailored Encoder：提升至 46.6%（+40.4%），说明专用编码器对异质传感特征至关重要
- +UMIP：进一步提升至 56.4%（+9.8%），尤其在需要深度语言理解的 QA 任务上增益显著
- 在 XRF55 上也观察到类似趋势：Baseline QA 3.5% → +Tailored Encoder 11.8% → +UMIP 12.4%
- tSNE 可视化显示 HoloLLM 的多模态 token 在动作类别上聚类效果显著优于 Baseline 和 OneLLM

## 亮点
1. **首创性**：首次将 mmWave、WiFi、RFID 等稀有传感模态接入 MLLM 进行语言驱动的人体感知与推理
2. **UMIP 设计精巧**：利用 CLIP 预对齐嵌入作为"粗查询"、专用编码器特征作为"细参考"，通过迭代交叉注意力渐进融合，避免了大规模模态-文本预训练的数据需求
3. **完整 benchmark 建设**：建立了首个多传感模态语言驱动人体感知基准，包含 Action Recognition、QA、Caption 三个任务和三种跨域设置
4. **tSNE 可视化**清晰展示了 UMIP 生成的多模态 token 在类别维度聚类性好、且与文本 token 对齐

## 局限与展望
1. 任务范围有限，仅覆盖动作识别/QA/描述，未涉及任务规划、动作生成等更复杂的具身任务
2. WiFi、RFID 等模态在跨被试/跨环境设置下泛化能力弱，性能接近随机
3. 数据集规模仍然较小（MM-Fi 和 XRF55 均为实验室级别），是否可扩展到大规模真实场景有待验证
4. LLM 固定为 LLaMA2-7B，未探索更大或更新的基座模型

## 与相关工作的对比
- **OneLLM**：同样使用统一编码器+统一 projector，但缺少针对异质传感特征的专用编码器，在传感模态上 QA 仅 3-5%
- **ImageBind**：通过对比学习对齐多模态嵌入，为深度/红外配备专用编码器，在部分模态上表现尚可但远逊于 HoloLLM
- **Tokenpacker / Honeybee**：仅使用共享 projector，完全无法捕捉传感模态特异性特征，QA 近乎随机
- HoloLLM 的 UMIP 介于 modality-specific projector 和 universal projector 之间，兼顾通用性和模态特异性

## 启发与关联
- UMIP 的"粗到细"渐进对齐思路可推广到其他数据稀缺的模态-语言对齐场景（如触觉、EEG）
- 人类-VLM 协作标注流水线是低成本生成稀有模态文本配对的有效范式
- 对于自动驾驶/具身智能领域，如何将雷达、LiDAR 等传感器的感知能力与 LLM 的推理能力结合是值得探索的方向

## 评分
- 新颖性: 8/10 — 首次将多种稀有传感模态接入 MLLM，UMIP 设计有创新性
- 实验充分度: 7/10 — 两个数据集、三种设置、消融和可视化齐全，但数据集规模有限
- 写作质量: 8/10 — 问题定义清晰，方法描述合理，图表精美
- 价值: 7/10 — 开辟了传感模态+MLLM 新方向，但实用性受限于数据规模和任务范围

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[NeurIPS 2025\] Poly-Guard: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset](poly-guard_massive_multi-domain_safety_policy-grounded_guardrail_dataset.md)

</div>

<!-- RELATED:END -->
