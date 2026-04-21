---
title: >-
  [论文解读] Where, What, Why: Towards Explainable Driver Attention Prediction
description: >-
  [ICCV 2025][自动驾驶][驾驶员注意力预测] 本文提出了"可解释驾驶员注意力预测"新范式，构建了首个大规模 W³DA 数据集并设计了 LLada 框架，将空间注意力预测（Where）、语义解析（What）和认知推理（Why）统一在一个端到端的大语言模型驱动架构中。
tags:
  - ICCV 2025
  - 自动驾驶
  - 驾驶员注意力预测
  - 可解释性
  - 多模态大语言模型
  - 认知推理
  - 注视预测
---

# Where, What, Why: Towards Explainable Driver Attention Prediction

**会议**: ICCV 2025  
**arXiv**: [2506.23088](https://arxiv.org/abs/2506.23088)  
**代码**: [github.com/yuchen2199/Explainable-Driver-Attention-Prediction](https://github.com/yuchen2199/Explainable-Driver-Attention-Prediction)  
**领域**: 自动驾驶 / 注意力建模  
**关键词**: 驾驶员注意力预测, 可解释性, 多模态大语言模型, 认知推理, 注视预测

## 一句话总结

本文提出了"可解释驾驶员注意力预测"新范式，构建了首个大规模 W³DA 数据集并设计了 LLada 框架，将空间注意力预测（Where）、语义解析（What）和认知推理（Why）统一在一个端到端的大语言模型驱动架构中。

## 研究背景与动机

驾驶员注意力建模是自动驾驶和认知科学的基础挑战。现有方法（如 ACT-Net、FBLNet）主要关注预测驾驶员"看哪里"——生成空间热力图来回归注视点分布。然而，这种范式存在根本性局限：

**仅有浅层隐式表示**：空间热力图本质上是像素空间的回归，只提供注意力的"位置"信息，无法揭示驾驶员为什么关注某个区域

**缺少语义理解**：无法回答驾驶员"在看什么"——是前方车辆、红绿灯还是行人？

**缺少认知解释**：无法回答"为什么关注"——是遵守交通规则、确保驾驶安全还是导航到目的地？

例如，驾驶员在十字路口关注红灯（遵守交通规则）、关注即将通过的骑行者（确保安全）、关注前方道路（规划路线），这些背后的认知动机此前从未被建模。

本文的核心贡献是将驾驶员注意力预测从单一的"Where"扩展为"Where + What + Why"三位一体的可解释范式，通过整合空间、语义和认知知识实现更全面的注意力理解。

## 方法详解

### 整体框架

LLada（Large Language model-driven driver attention）包含四个核心组件：
1. **预训练视觉编码器** $\mathcal{F}_{\text{vis}}$：CLIP-ViT-L + 线性投影器
2. **大语言模型** $\mathcal{F}_{\text{LLM}}$：Vicuna-7B
3. **特殊注意力 token** [ATTN]：编码高层认知线索
4. **认知感知注意力解码器** $\mathcal{F}_{\text{dec}}$：将认知信息解码为像素级注意力图

### 关键设计

1. **W³DA 数据集构建**：

    - 整合四个主流驾驶注意力数据集：DR(eye)VE（正常驾驶）、LBW（正常驾驶）、BDDA（安全关键场景）、DADA-2000（交通事故）
    - 总计 69,980 个关键样本，来自 3,548 个视频场景
    - **注意力感知关键帧选择**：不采用统一帧率采样，而是基于三个标准筛选关键帧——(a) 驾驶场景语义相似度（CLIP CLS token 余弦相似度），(b) 注意力空间分布 KL 散度，(c) 注意力区域语义相似度。当 KL 散度超阈值或语义相似度低于阈值时选为关键帧
    - **半自动标注流程**：使用视觉和上下文提示引导 Qwen-VL-Max 生成初步标注（链式推理：先数注意力区域数→再描述内容→最后解释原因），然后由人类专家验证和修改

2. **[ATTN] token 设计**：

    - 在 LLM 词表中新增特殊 [ATTN] token
    - 在注意力预测时，LLM 输出序列中包含 [ATTN] token，其对应嵌入通过 MLP 投影后送入注意力解码器
    - 设计动机：[ATTN] token 在 LLM 内部编码了高层认知线索（通过和语言 token 的自注意力交互），有效地将文本空间的语义/因果推理信息传递给视觉空间的注意力图生成

3. **认知感知注意力解码器**：

    - 通过交叉注意力机制整合 [ATTN] 嵌入 $\mathbf{h}_{\text{attn}}$ 和视觉特征 $\mathbf{h}_{\text{vis}}$：
   
    $\mathbf{h}_{\text{dec}}' = \mathbf{h}_{\text{vis}} + \text{Repeat}(CA(\mathbf{h}_{\text{attn}}, \mathbf{h}_{\text{vis}}))$
   
    - 将增强后的视觉特征重塑为 3D 特征图，经过 5 层 3×3 卷积降维后通过双线性上采样生成全分辨率注意力图
    - 设计动机：[ATTN] 嵌入包含"为什么看这里"的认知信息，通过交叉注意力将其注入到视觉特征中，使得解码器不仅知道"看哪里"，还理解"为什么看"

### 损失函数 / 训练策略

总损失为注意力图损失和文本解释损失的加权和：

$$\mathcal{L} = \lambda_{\text{map}} \mathcal{L}_{\text{map}} + \lambda_{\text{txt}} \mathcal{L}_{\text{txt}}$$

- 注意力图损失：$\mathcal{L}_{\text{map}} = \lambda_{\text{bce}} \text{BCE}(\hat{\mathbf{A}}, \mathbf{A}) + \lambda_{\text{kl}} \text{KL}(\hat{\mathbf{A}}, \mathbf{A})$
- 文本损失：$\mathcal{L}_{\text{txt}} = \lambda_{\text{what}} \text{CE}(\hat{\mathcal{S}}, \mathcal{S}) + \lambda_{\text{why}} \text{CE}(\hat{\mathcal{E}}, \mathcal{E})$

训练配置：4 张 A100 GPU，DeepSpeed 引擎，AdamW（lr=3e-4），LoRA 微调 LLM，视觉编码器冻结，注意力解码器从头训练。

## 实验关键数据

### 主实验 — W³DA 注意力图预测

| 方法 | 类型 | KLdiv ↓ (正常) | CC ↑ (正常) | KLdiv ↓ (关键) | KLdiv ↓ (事故) |
|------|------|--------------|------------|--------------|--------------|
| GBVS | 传统 | 2.572 | 0.294 | 2.238 | 2.826 |
| ERFNet | DNN | 1.979 | 0.558 | 1.593 | 2.181 |
| ConvNeXt | DNN | 2.042 | 0.570 | 1.765 | 3.049 |
| GazeXplain† | 多任务 | 2.578 | 0.477 | 2.769 | 3.109 |
| **LLada†** | **多任务** | **1.219** | **0.583** | **1.230** | **1.927** |

LLada 在所有场景和指标上全面超越现有方法。在 KLdiv 上分别优于第二名 ERFNet 38.4%（正常）、22.8%（关键）、11.7%（事故）。

### 消融实验 — Where/What/Why 联合推理的互增益

**Where 对 What/Why 的影响**（文本解释生成质量）：

| 设置 | BLEU ↑ | METEOR ↑ | ROUGE ↑ | CIDEr-R ↑ |
|------|--------|----------|---------|-----------|
| 无 Where（仅 What+Why） | 下降显著 | 下降显著 | 下降显著 | 下降显著 |
| **有 Where（完整 LLada）** | **最高** | **最高** | **最高** | **最高** |

**What/Why 对 Where 的影响**（注意力图预测质量）：

| 设置 | KLdiv ↓ | CC ↑ |
|------|---------|------|
| 仅 Where | 基线 | 基线 |
| Where + What | 改善 | 改善 |
| **Where + What + Why** | **最优** | **最优** |

Where、What、Why 三者相互增强：空间注意力提供位置线索辅助语义/因果推理，语义/因果推理反过来提升注意力定位精度。

### 关键发现

1. **跨域泛化性**：仅在 W³DA 上训练的 LLada，在完整 DR(eye)VE/BDDA/DADA 测试集上超越大多数专属训练的模型（KLdiv 分别改善 29.8%、20.7%、5.5%）
2. **文本解释质量**：LLada 在所有文本指标上全面超越 GazeXplain 和两阶段基线方法（注意力预测器+微调 LLaVA），CIDEr-R 提升超过 50%
3. **定性对比**：LLada 正确关注了车前行人并生成了上下文相关的认知解释（"评估行人运动速度以避免碰撞"），而 GazeXplain 遗漏了此关键区域

## 亮点与洞察

- 从"Where"到"Where + What + Why"的范式扩展是本文最核心的概念贡献，将注意力预测从像素回归提升到认知理解的层面
- W³DA 数据集的构建方法论（关键帧选择 + MLLM 辅助标注 + 人工验证）为类似大规模标注任务提供了可复用的 pipeline
- [ATTN] token 的设计巧妙地在 LLM 的文本空间和视觉空间之间架起了桥梁，使得语言推理可以直接指导像素级预测
- 消融实验清晰地证明了三个维度的互增益关系，验证了统一框架的必要性

## 局限与展望

- 当前仅使用单帧图像作为输入，未利用视频的时序信息（驾驶场景天然是时序的）
- W³DA 的 What/Why 标注依赖 MLLM API（Qwen-VL-Max），可能存在系统性偏差
- 注意力解码器较简单（5 层卷积），可能限制了精细空间细节的恢复
- Vicuna-7B 作为 LLM backbone 相对较小，使用更强的 LLM 可能进一步提升推理质量
- 未探索视频级的时序认知推理（如追踪注意力转移的因果链）

## 相关工作与启发

- 与 GazeXplain 的区别：GazeXplain 的解释仅停留在语义层面（描述注视目标），LLada 进一步到认知层面（解释注视原因）
- 与 LISA（推理分割）的关系：类似地使用 LLM 输出的特殊 token 来控制像素级预测，但应用于注意力预测而非语义分割
- 对智能驾驶培训的启发：可解释的注意力模型可以告诉新手司机"应该看哪里、看什么、为什么要看"
- 对自动驾驶可解释性的启发：为 AV 系统的注意力决策提供人类可理解的解释

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （Where+What+Why 新范式 + 首个可解释注意力数据集）
- 实验充分度: ⭐⭐⭐⭐⭐ （多数据集/多场景/多指标/多基线 + 跨域泛化验证）
- 写作质量: ⭐⭐⭐⭐⭐ （动机清晰，图示优秀，实验全面）
- 价值: ⭐⭐⭐⭐⭐ （开辟了注意力预测的新方向，数据集和方法均有长期影响力）

<!-- RELATED:START -->

## 相关论文

- [DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance](../../CVPR2026/autonomous_driving/drivergaze360_omnidirectional_driver_attention_with_object-level_guidance.md)
- [Where am I? Cross-View Geo-localization with Natural Language Descriptions](where_am_i_cross-view_geo-localization_with_natural_language_descriptions.md)
- [ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)
- [SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)
- [DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)

<!-- RELATED:END -->
