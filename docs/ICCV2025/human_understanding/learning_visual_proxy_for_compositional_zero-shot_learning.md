---
title: >-
  [论文解读] Learning Visual Proxy for Compositional Zero-Shot Learning
description: >-
  [ICCV 2025][人体理解][Compositional Zero-Shot Learning] 提出 Visual Proxy（视觉代理）概念，在 CZSL 任务中首次引入文本引导的视觉类中心，并通过跨模态联合学习（CMJL）协同优化文本原型与视觉代理，在四个 CZSL 基准上达到闭世界 SOTA。
tags:
  - ICCV 2025
  - 人体理解
  - Compositional Zero-Shot Learning
  - Visual Proxy
  - 跨模态
  - CLIP
  - VLM
---

# Learning Visual Proxy for Compositional Zero-Shot Learning

**会议**: ICCV 2025  
**arXiv**: [2501.13859](https://arxiv.org/abs/2501.13859)  
**代码**: [codefish12-09/VP_CMJL](https://github.com/codefish12-09/VP_CMJL)  
**领域**: Human Understanding / Zero-Shot Learning  
**关键词**: Compositional Zero-Shot Learning, Visual Proxy, Cross-Modal Learning, CLIP, VLM

## 一句话总结

提出 Visual Proxy（视觉代理）概念，在 CZSL 任务中首次引入文本引导的视觉类中心，并通过跨模态联合学习（CMJL）协同优化文本原型与视觉代理，在四个 CZSL 基准上达到闭世界 SOTA。

## 研究背景与动机

组合零样本学习（CZSL）旨在通过已见的属性-物体组合（如"绿色衣服""红色苹果"）泛化到未见组合（如"红色衣服""绿色苹果"）。基于 CLIP 的现有方法通过文本-图像匹配进行分类，但存在两大核心问题：

**持续的模态鸿沟（Modality Gap）**：尽管各种 prompt 设计和融合方法已部分缩小文本与视觉空间的距离，但完全消除不可能。在 top-1 检索场景下，真实跨模态对的距离可能超过假阳性对，导致语义相似的组合被混淆（如"成熟苹果" vs "未成熟苹果"）。

**文本原型缺乏细粒度视觉信息**：每个类别的文本原型仅来自一个组合标签，而对应的图像实例包含丰富的纹理、光照、形状变化。这种语义-视觉不对称使得文本原型无法捕捉区分相似组合所需的细粒度信息。

核心洞察：CZSL 的本质是图像分类，最优类中心应在视觉空间中。但直接学习视觉中心困难（高方差），因此利用结构化的文本空间引导视觉中心学习。

## 方法详解

### 整体框架

VP-CMJL 由三个模块组成：
1. **Textual Prototype Learning Module**：三路框架（属性/物体/组合）+ 跨模态解耦模块
2. **Visual Proxy Learning Module**：文本引导的视觉代理学习 + MLP 解耦
3. **Cross-Modal Joint Learning Module**：KL 散度约束协同优化双模态

使用冻结的 CLIP ViT-L/14 作为视觉和文本编码器。

### 关键设计

1. **Textual Prototype Learning（文本原型学习）**：

    - 三路可学习 prompt：属性 $\theta^a$、物体 $\theta^o$、组合 $\theta^c$，前缀初始化为 "a photo of"
    - **跨模态解耦模块 (AD-CA / OD-CA)**：使用多头交叉注意力将全局图像特征 $f_v^{cls}$ 分解为与文本原型对齐的属性/物体特征。Query 来自图像特征，Key/Value 来自文本原型，输出通过 FFN + LayerNorm + 残差连接
    - **注意力分数增强概率计算**：属性和物体分支的分类概率同时考虑文本-图像余弦相似度和注意力分数 $s^a/s^o$:
    $p_t(y_i|x) = \frac{\exp((f_t^y \cdot t_i^y + s_i^y)/\tau_t)}{\sum_k \exp((f_t^y \cdot t_k^y + s_k^y)/\tau_t)}$

2. **Visual Proxy Learning（视觉代理学习）**：

    - **初始化**：使用 CLIP 文本编码器的词嵌入初始化视觉代理 $v_i^a = E_l(w_i^a)$（实验验证 CLIP 初始化优于 BERT/GPT）
    - **组合代理构造**：拼接属性和物体代理后通过全连接层投影 $v_{i,j}^c = E_c([v_i^a, v_j^o])$
    - **MLP 解耦**：视觉模态内部使用简单 MLP 解耦（而非 cross-attention），因为这是模态内学习
    - **对比训练**：通过 softmax 温度缩放的余弦相似度进行类内吸引、类间排斥
    - 关键理论支撑：CLIP 的最优类中心在视觉与文本空间的交叠区域，但由于模态鸿沟，该中心仍受偏差影响。视觉代理直接在视觉空间学习，更贴近图像分类的最优解

3. **Cross-Modal Joint Learning（跨模态联合学习，CMJL）**：

    - **训练**：以文本原型分布为目标，视觉代理分布为近似，用 KL 散度约束：
    $\mathcal{L}_{kl} = D_{KL}(P_t \| P_v)$
      总损失 $\mathcal{L} = \alpha(\mathcal{L}_t + \mathcal{L}_v) + \beta \mathcal{L}_{kl}$
    - **推理**：融合两个模态的概率：
    $p(y_{i,j}|x) = p_t(y_{i,j}|x) + \lambda p_v(y_{i,j}|x)$
      最终预测为属性、物体、组合三路概率之和的 argmax

### 损失函数 / 训练策略

- 文本路径：$\mathcal{L}_t = \gamma_{ao}(\mathcal{L}_t^a + \mathcal{L}_t^o) + \gamma_c \mathcal{L}_t^c$
- 视觉路径：$\mathcal{L}_v = \gamma_{ao}(\mathcal{L}_v^a + \mathcal{L}_v^o) + \gamma_c \mathcal{L}_v^c$
- 总损失：$\mathcal{L} = \alpha(\mathcal{L}_t + \mathcal{L}_v) + \beta \mathcal{L}_{kl}$
- 训练 20 epochs，CLIP ViT-L/14，NVIDIA A800 GPU

## 实验关键数据

### 主实验 (表格)

**Closed-World Results（Best HM / AUC）**

| 方法 | C-GQA HM | C-GQA AUC | UT-Zappos HM | UT-Zappos AUC | MIT-States HM | MIT-States AUC |
|------|----------|-----------|--------------|---------------|---------------|----------------|
| Troika (CVPR'24) | 29.4 | 12.4 | 54.6 | 41.7 | 39.3 | 22.1 |
| IMAX (TPAMI'25) | 29.8 | 12.8 | 54.2 | 40.6 | 39.1 | 21.9 |
| CDS-CZSL (CVPR'24) | 28.1 | 11.1 | 52.7 | 39.5 | 39.2 | 22.4 |
| **VP-CMJL (Ours)** | **34.9** | **16.3** | **58.5** | **47.9** | **40.4** | **23.3** |

在 C-GQA 上 HM 提升 +5.5%，AUC 提升 +3.9%；UT-Zappos 上 HM +3.9%，AUC +6.2%。

**VAW-CZSL（新大规模数据集）**

| 方法 | S | U | HM | AUC |
|------|---|---|-----|-----|
| CAILA (WACV'24) | 41.6 | 49.2 | 34.6 | 17.2 |
| **VP-CMJL** | **47.8** | **51.1** | **38.2** | **20.7** |

### 消融实验 (表格)

**组件消融（UT-Zappos / MIT-States, Closed-World）**

| TP | VP | UT-Zappos HM | UT-Zappos AUC | MIT-States HM | MIT-States AUC |
|----|-----|--------------|---------------|---------------|----------------|
| ✓ | ✓ | **58.5** | **47.9** | **40.4** | **23.3** |
| ✓ | ✗ | 51.9 | 37.8 | 37.8 | 20.8 |
| ✗ | ✓ | 55.3 | 42.1 | 37.6 | 20.7 |

去除 VP 导致 UT-Zappos AUC 下降 10.1%；去除 TP 下降 5.8%。

**解耦模块消融**

| i2t 解耦 | i2v 解耦 | UT-Zappos HM | MIT-States HM |
|----------|----------|--------------|---------------|
| CA | MLP | **58.5** | **40.4** |
| CA | CA | 54.7 | 39.6 |
| MLP | MLP | 58.5 | 38.8 |
| MLP | CA | 55.7 | 39.6 |

CA 适合跨模态对齐（文本原型），MLP 适合模态内学习（视觉代理）。

### 关键发现

- **视觉代理的引入是关键**：在 UT-Zappos 上 AUC 从 37.8 提升到 47.9（+26.7%）
- **双模态训练的隐式增强效应**：即使推理时去掉一个模态，性能退化也小于训练时去掉，说明联合优化促进了表征的相互增强
- **Open-World 同样有效**：C-GQA 开放世界 HM +4.6%，UT-Zappos +6.7%
- **t-SNE 可视化**：VP-CMJL 的视觉特征空间更紧凑、类边界更清晰

## 亮点与洞察

1. **视觉代理概念的提出**：首次在 CZSL 中引入文本引导的可学习视觉类中心，弥合了文本原型缺乏细粒度视觉信息的根本缺陷
2. **跨模态 KL 散度约束**：简洁有效的协同优化策略，利用文本的语义稳定性引导视觉代理的学习方向
3. **注意力分数增强分类**：在传统余弦相似度基础上引入 cross-attention 分数，同时编码查询图像与所有类别的关系
4. **解耦模块的恰当选择**：跨模态用 CA（需要对齐），模态内用 MLP（简单变换即可），体现了对问题本质的理解

## 局限与展望

- MIT-States 开放世界性能仅具竞争力，未超越 CDS-CZSL（该方法专门针对开放世界设计了剪枝技术）
- 视觉代理的温度参数 $\tau_v$ 和融合权重 $\lambda$ 需人工调节
- 组合代理通过拼接+全连接生成，可能无法捕捉属性-物体间的非线性交互
- 未见组合的视觉代理通过属性/物体代理拼接获得，缺乏直接来自视觉空间的监督
- 未探索更大规模 VLM（如 SigLIP、EVA-CLIP）的效果

## 相关工作与启发

- **CSP / DFSP / Troika**：基于 CLIP 的 CZSL 方法族，VP-CMJL 在其三路框架上添加了视觉代理
- **CDS-CZSL**：考虑属性特异性的方法，在开放世界有优势
- **视觉中心学习**：传统方法通过平均图像特征获得原型，但受视角/光照影响大；VP-CMJL 通过文本引导避免此问题
- 启发：在 VLM 时代，不应只学文本端的 prompt，同时学习视觉端的类中心可能是提升分类性能的通用策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ 视觉代理概念新颖，跨模态联合学习策略设计合理
- **实验充分度**: ⭐⭐⭐⭐ 4 个数据集、开放/闭世界、组件消融、解耦模块消融、可视化分析
- **写作质量**: ⭐⭐⭐⭐ 动机分析深入（类中心理论），方法描述清晰
- **价值**: ⭐⭐⭐⭐ CZSL 任务上的显著提升，视觉代理思想可推广至其他 VLM 分类场景

<!-- RELATED:START -->

## 相关论文

- [NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [Conformal Prediction for Zero-Shot Models](../../CVPR2025/human_understanding/conformal_prediction_for_zero-shot_models.md)
- [VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](../../ICML2025/human_understanding/visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)
- [In-Context Compositional Learning via Sparse Coding Transformer](../../NeurIPS2025/human_understanding/in-context_compositional_learning_via_sparse_coding_transformer.md)
- [External Knowledge Injection for CLIP-Based Class-Incremental Learning](external_knowledge_injection_for_clip-based_class-incremental_learning.md)

<!-- RELATED:END -->
