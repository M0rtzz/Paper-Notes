---
title: >-
  [论文解读] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation
description: >-
  [ICCV 2025][图像分割][全景场景图生成] 提出SPADE——一种面向开放词汇全景场景图生成（PSG）的空间感知去噪网络，通过DDIM逆向校准将预训练扩散模型适配为PSG特定的空间先验提取器，并设计关系图Transformer捕获长程和局部上下文，在闭集和开集场景中均大幅超越SOTA，尤其在空间关系预测上表现突出。
tags:
  - ICCV 2025
  - 图像分割
  - 全景场景图生成
  - 开放词汇
  - 扩散模型
  - 空间关系推理
  - Transformer
---

# SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation

**会议**: ICCV 2025  
**arXiv**: [2507.05798](https://arxiv.org/abs/2507.05798)  
**代码**: 无（项目页面提及）  
**领域**: 图像分割  
**关键词**: 全景场景图生成, 开放词汇, 扩散模型, 空间关系推理, 图Transformer

## 一句话总结

提出SPADE——一种面向开放词汇全景场景图生成（PSG）的空间感知去噪网络，通过DDIM逆向校准将预训练扩散模型适配为PSG特定的空间先验提取器，并设计关系图Transformer捕获长程和局部上下文，在闭集和开集场景中均大幅超越SOTA，尤其在空间关系预测上表现突出。

## 研究背景与动机

全景场景图生成（PSG）将实例分割与关系理解统一为 subject-predicate-object 三元组。近年来基于VLM的开放词汇方法取得显著进展，但存在一个被忽视的关键问题：

**VLM的空间推理缺陷**：多项研究表明CLIP、BLIP、GLIP等VLM在空间关系理解上存在先天不足（因训练数据中缺乏空间描述），导致模型难以判断"左/右/上方/下方"等空间关系

**距离敏感性**：作者系统性实验发现，当两个物体距离较远（中心间距>1/3图像宽度）时，VLM-based模型的空间关系预测性能急剧下降（如OpenPSG的R@50从43.7降至37.1）

**缺乏上下文推理**：现有方法主要关注设计视觉提示来提取VLM知识，忽略了关系对之间的空间和语义上下文信息

**直接使用扩散模型的不足**：虽然扩散模型具有出色的空间组合能力，但预训练知识未针对PSG任务优化，直接使用效果不佳

**核心动机**：能否将扩散模型的空间知识注入VLM，而不损害其固有的开放世界识别能力？

## 方法详解

### 整体框架

SPADE是一个两阶段方法：
- **第一阶段**：逆向引导校准（Inversion-guided Calibration）——将预训练扩散模型适配为PSG特定的去噪网络
- **第二阶段**：空间感知上下文推理（Spatial-aware Context Reasoning）——通过关系图Transformer生成高质量关系查询

### 关键设计

1. **逆向引导校准（Inversion-guided Calibration）**：

    - **逆向空间先验提取**：将真实图像通过DDIM逆向过程转为确定性噪声 $z$，再用教师扩散模型（BELM）在关系提示 "[subject] is [predicate] [object]..." 条件下进行确定性采样，得到交叉注意力图 $A_i$ 作为空间先验
    - **隐式文本编码器**：由于推理时无文本描述，用CLIP图像编码器 + MLP适配器替代文本编码器：$f_i = \epsilon_\phi(x_i, \mathrm{MLP} \circ \mathrm{CLIP_{image}}(x_i))$
    - **LoRA校准**：仅更新UNet交叉注意力层的低秩矩阵 $\Delta\mathbf{W}_k = \mathbf{B} \times \mathbf{D}$（$\mathbf{B} \in \mathbb{R}^{m_{in} \times r}$, $\mathbf{D} \in \mathbb{R}^{r \times m_{out}}$），保留预训练知识
    - **校准损失**：$\mathcal{L}_{cal} = \frac{1}{N}\sum_{i=1}^{N}(\lambda\|A_i - A_i'\|_1)$，对齐真实图像上的交叉注意力图与逆向过程得到的先验
    - 设计动机：DDIM逆向过程天然保留输入图像的空间结构；LoRA微调可在注入PSG空间知识的同时最大限度保留扩散模型的世界知识

2. **空间感知关系图Transformer（RGT）**：

    - **空间-语义图构建**：基于实例掩码的空间距离（相邻=1）和特征余弦相似度（>阈值=1）构建图 $G \in \mathbb{R}^{N \times N}$
    - **长程上下文学习**（$\mathrm{RGT_g}$）：分别对邻居 $\mathcal{P}(r)^+$ 和非邻居 $\mathcal{P}(r)^-$ 计算自注意力，然后融合：$\mathbf{q}_r \leftarrow \mathbf{q}_r + \mathrm{RGT}(\mathbf{q}_r)_{\mathcal{P}^+} + \mathrm{RGT}(\mathbf{q}_r)_{\mathcal{P}^-}$，再用MLP融合所有特征
    - **局部上下文学习**（$\mathrm{RGT_l}$）：用GCN在图 $G$ 上聚合局部邻域信息：$\hat{\mathbf{q}}_r = \mathrm{GCN}(G, \mathbf{q}_r'; \mathbf{W}_l)$
    - **关系查询构造**：基于余弦距离选择相近目标对构建关系查询 $\Psi_r$，辅助loss $\mathcal{L}_{rqc}$ 优化选择质量
    - 设计动机：仅建模相连对象不够，非相连对象间也可能存在关系（如"远处的人看着飞机"）；长程+局部双路推理覆盖不同空间尺度

3. **开放词汇关系预测**：

    - 使用CLIP文本编码器对目标/谓词类别模板编码，与特征做相似度分类
    - 双路预测融合：扩散模型特征分数 $\mathbf{P}_o$ + CLIP池化特征分数 $\mathbf{P}_o'$，通过几何均值融合：$\mathbf{P}^o_{\text{final}} = \mathbf{P}_o^\alpha \cdot \mathbf{P}_o'^{(1-\alpha)}$
    - 关系预测同理，使用subject+object的联合掩码进行池化
    - 设计动机：扩散模型擅长空间推理但开放词汇能力有限，CLIP擅长开放世界识别但空间推理弱，互补融合

### 损失函数 / 训练策略

- 第一阶段（校准）：仅 $\mathcal{L}_{cal}$（L1对齐交叉注意力图），更新MLP适配器和LoRA参数
- 第二阶段：$\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{rqc}}\mathcal{L}_{rqc} + \lambda_{\text{mask}}L_{\text{mask}}$
- 超参数：$\alpha=0.34$, $\eta=0.65$, $\lambda_{rqc}=0.6$, $\lambda_{mask}=1$
- 扩散模型和CLIP参数在第二阶段冻结
- 共训练80个epoch，第60个epoch降低学习率
- 4×A100 GPU训练

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | SPADE | OpenPSG | OvSGTR | 提升 |
|---|---|---|---|---|---|
| PSG闭集 | R/mR@100 | **54.3/51.7** | 49.3/47.5 | 41.4/28.3 | +5.0/+4.2 |
| PSG开集(OvR) | R/mR@50 | **26.7/23.3** | 21.2/19.8 | 19.3/12.4 | +5.5/+3.5 |
| PSG开集(OvR) | R/mR@100 | **31.8/25.8** | 25.1/21.4 | 22.8/14.0 | +6.7/+4.4 |
| VG开集(OvR) | R/mR@100 | **29.9/13.9** | 25.7/12.1 | 26.7/5.7 | +4.2/+1.8 |
| PSG开集(OvD+R) | R@50 | **22.7** | 11.4 | 19.1 | +3.6 |

### 消融实验

| 配置 | R/mR@50 (PSG闭集) | 说明 |
|---|---|---|
| 无RGT组件 | 30.5/26.3 | Baseline |
| +长程邻居学习(LCNL) | 35.6/32.2 | 邻居上下文显著提升 |
| +长程非邻居(LCNNL) | 37.1/34.4 | 非邻居关系也有贡献 |
| +局部学习(LCL) | 40.3/35.6 | GCN局部推理补充长程 |
| +全部+$\mathcal{L}_{rqc}$ | **45.1/41.2** | 辅助损失进一步优化 |

| 校准策略 | OvR R@50 | OvD+R R@50 | 说明 |
|---|---|---|---|
| 无校准(直接用预训练UNet) | 15.3 | 10.1 | 预训练知识不匹配PSG |
| 无LoRA(全参数微调) | 18.8 | 12.7 | 破坏预训练知识 |
| 无逆向(随机噪声) | 21.0 | 15.9 | 缺乏空间结构先验 |
| 完整方法 | **26.7** | **22.7** | 逆向+LoRA最优 |

### 关键发现

- **空间关系的系统性分析**：SPADE在远距离关系(DR)上R@50达42.3，OpenPSG仅37.1，差距从6.6缩小到4.2，说明SPADE有效改善了远距离空间推理
- **开放词汇模块消融**：扩散特征+池化特征的融合比单独使用任一方都好（26.7 vs 12.5 vs 21.4）
- **逆向过程是关键**：随机采样噪声（21.0）远不如确定性逆向（26.7），因为逆向过程保留了原图的空间布局

## 亮点与洞察

- **问题发现有深度**：系统性地揭示了VLM在空间关系推理上的缺陷，并量化了距离对性能的影响
- **创新性地利用扩散模型的逆向过程**：DDIM逆向天然保留空间结构，这一特性被巧妙地转化为PSG的空间先验
- **长程+局部双路上下文推理**：邻居/非邻居的分别建模 + GCN局部聚合，全面覆盖不同尺度的关系
- **扩散+判别双路分类**：空间推理靠扩散模型，开放识别靠CLIP，互补融合发挥各自优势
- **LoRA校准的必要性**：全参数微调反而不如LoRA，验证了保留预训练知识的重要性

## 局限与展望

- 两阶段训练流程较复杂，需要先独立校准UNet再训练RGT
- 扩散模型推理的计算开销较大，未报告推理速度
- 空间-语义图的构建依赖固定阈值，不够灵活
- 仅在PSG和VG两个数据集验证，未在更多场景图生成基准上评测
- 关系提示的设计（"[subject] is [predicate] [object]"）可能限制了能捕获的关系类型

## 相关工作与启发

- 与OpenPSG、OvSGTR等VLM-based方法相比，SPADE首次引入扩散模型增强空间推理
- LoRA校准策略（仅微调交叉注意力层的低秩矩阵）可推广到其他需要适配扩散模型的下游任务
- 关系图Transformer中邻居/非邻居分别推理的思想，可应用于HOI检测等结构化预测任务
- 与DiffusionSG等工作不同，SPADE利用的是逆向过程的注意力图而非生成能力

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 发现VLM空间推理缺陷，创新性利用扩散逆向过程补充空间先验
- **实验充分度**: ⭐⭐⭐⭐ 闭集+开集+空间关系分析+多维消融，缺少效率分析
- **写作质量**: ⭐⭐⭐⭐ 动机阐述清晰，方法描述系统，图表丰富
- **价值**: ⭐⭐⭐⭐ 揭示了VLM空间推理的核心问题，提供了扩散模型+VLM融合的新范式

<!-- RELATED:START -->

## 相关论文

- [OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](../../ECCV2024/segmentation/openpsg_openset_panoptic_scene_graph_generation_via_large_mu.md)
- [DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_comprehensive_panoptic_scene_graph_generation_in_realtime.md)
- [SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)

<!-- RELATED:END -->
