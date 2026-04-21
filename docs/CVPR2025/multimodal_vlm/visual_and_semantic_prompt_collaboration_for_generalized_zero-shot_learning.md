---
title: >-
  [论文解读] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning
description: >-
  [CVPR 2025][多模态][广义零样本学习] 提出视觉语义提示协作网络（VSPCN），通过在预训练ViT中同时学习视觉提示和语义提示，并设计浅层弱融合+深层强融合机制，高效适配ViT提取语义相关的判别性视觉特征，在CUB/SUN/AWA2三个GZSL基准上均达到SOTA。
tags:
  - CVPR 2025
  - 多模态
  - 广义零样本学习
  - 视觉语义提示
  - 提示学习
  - ViT适配
  - 知识迁移
---

# Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning

**会议**: CVPR 2025  
**arXiv**: [2503.23030](https://arxiv.org/abs/2503.23030)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 广义零样本学习, 视觉语义提示, Prompt Tuning, ViT适配, 知识迁移

## 一句话总结

提出视觉语义提示协作网络（VSPCN），通过在预训练ViT中同时学习视觉提示和语义提示，并设计浅层弱融合+深层强融合机制，高效适配ViT提取语义相关的判别性视觉特征，在CUB/SUN/AWA2三个GZSL基准上均达到SOTA。

## 研究背景与动机

广义零样本学习（GZSL）需要同时识别已见类和未见类，核心依赖语义信息（如类别属性）将知识从已见类迁移到未见类。传统方法用预训练骨干独立提取视觉和语义特征后对齐，但独立提取导致对齐不充分。

微调骨干的方法虽能增强视觉-语义交互，但面临两个问题：(1) 语义-视觉交互仅在网络最后层进行，对浅层影响有限；(2) 在有限的seen-class数据上微调整个骨干容易过拟合unseen classes。基于Transformer的ZSL方法（如ZSLViT）虽支持多层交互，但全量微调仍面临过拟合风险。

本文的关键洞察：不同于只学视觉提示的VPT，同时学习视觉提示（编码判别视觉信息）和语义提示（编码类别语义信息），两者协作提取语义相关的视觉特征，且通过prompt tuning避免了全量微调的过拟合风险。

## 方法详解

### 整体框架

VSPCN以预训练ViT-Base（ImageNet-1k）为骨干，输入包含5部分：CLS token、视觉提示 $f_{vp}$、语义提示 $f_{sp}$、图像token和共享语义属性 $S$（由Glove编码）。浅层（前 $l=6$ 层）使用弱融合机制初始化提示，深层（第7层起）使用强融合机制持续更新提示。ViT骨干参数冻结，仅训练提示相关的少量参数。

### 关键设计

1. **弱提示融合（Weak Prompt Fusion）**:
    - 功能：在网络浅层为随机初始化的提示注入基础信息
    - 核心思路：弱视觉提示融合（WVPF）用cross-attention从图像token聚合信息：$\tilde{f}_{vp}^0 = \text{softmax}(\frac{Q_v^0 {K_v^0}^T}{\sqrt{D}}) V_v^0$，query来自视觉提示，key/value来自图像token。弱语义提示融合（WSPF）类似地从共享语义属性 $S$ 中聚合信息。融合后拼接 $\tilde{F}^0 = [f_{cls}^0, \tilde{f}_{vp}^0, \tilde{f}_{sp}^0, f_1^0, \ldots, f_{N_v}^0]$ 送入后续ViT层
    - 设计动机：浅层特征相对初级，简单的cross-attention就足以为提示提供初步信息基础。弱融合仅在输入层执行一次

2. **强提示融合（Strong Prompt Fusion）**:
    - 功能：在深层持续补充提示信息，防止语义影响随层数增加而衰减
    - 核心思路：使用带注意力偏置的transformer更新提示：$\tilde{f}_{vp}^l = [\alpha_v \text{softmax}(\frac{Q_v^l {K_v^l}^T}{\sqrt{D}}) + (1-\alpha_v) \text{softmax}(B_v^l)] V_v^l + f_{vp}^l$，其中 $B_v^l \in \mathbb{R}^{N_v}$ 是可学习偏置，$\alpha_v$ 控制注意力与偏置的权重比。语义提示类似地与adapter更新后的属性 $S^l$ 融合。视觉提示在融合时仅与图像token交互，不受其他token干扰
    - 设计动机：弱融合的语义信号随网络加深会衰减。强融合在每一层重新注入视觉和语义信息。attention bias提供了可学习的先验位置信息，补充注意力机制

3. **语义属性适配器（Semantic Adapter）**:
    - 功能：学习实例自适应的语义特征
    - 核心思路：用cross-attention让语义属性与当前图像token交互：$S^l = \alpha_a \text{softmax}(\frac{Q_a^l {K_a^l}^T}{\sqrt{D}}) V_a^l + (1-\alpha_a) S^{l-1}$，query来自上层语义属性，key/value来自当前图像token
    - 设计动机：全局共享的语义属性对所有图像一视同仁。adapter根据不同图像动态调整属性权重，实现从类级别到实例级别的语义特征自适应

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{BASE} + \lambda_{CED}\mathcal{L}_{CED} + \lambda_{SKD}\mathcal{L}_{SKD}$：

- **基础损失** $\mathcal{L}_{BASE} = \mathcal{L}_{CLS} + \gamma \mathcal{L}_{AR}$：分类交叉熵（CLS token与语义原型相似度）+ 语义回归MSE（对齐CLS token到ground truth原型）
- **交叉熵散度损失** $\mathcal{L}_{CED}$：鼓励视觉提示学习与CLS token互补的判别信息。$\mathcal{L}_{ED} = \log(\frac{\mathcal{L}_{CE}(f_{vp}^M) + \mathcal{L}_{CE}(f_{cls}^M)}{\mathcal{L}_{KL}(\delta(f_{vp}^M), \delta(f_{cls}^M))} + 1)$，分子保证两者都准确，分母鼓励两者分布不同
- **语义知识蒸馏损失** $\mathcal{L}_{SKD}$：JSD散度+欧氏距离，将语义提示对齐到对应类别语义原型

推理时使用校准策略：$\tilde{y} = \arg\max_{\hat{y}}(f_{cls}^M \cdot a_{\hat{y}}^T + \tau \mathbb{I}_{\hat{y} \in \mathcal{Y}^u})$ 平衡已见/未见类偏差。

使用Adam优化器，学习率0.001，权重衰减0.0001，在NVIDIA RTX A4000上训练。

## 实验关键数据

### 主实验（GZSL Harmonic Mean H）

| 数据集 | VSPCN | ZSLViT | PSVMA | ZSCLR | MSDN | 提升vs次优 |
|--------|-------|--------|-------|-------|------|-----------|
| CUB | **75.7** | 73.6 | 73.8 | 72.4 | 68.1 | +1.9 |
| SUN | **53.8** | 47.3 | 52.3 | 48.7 | 41.3 | +1.5 |
| AWA2 | **77.6** | 74.2 | 75.4 | 73.4 | 67.7 | +2.2 |
| CUB (CZSL Acc) | **80.6** | 78.9 | - | 77.8 | 76.1 | +1.7 |
| SUN (CZSL Acc) | **75.3** | 68.3 | - | 66.3 | 65.8 | +7.0 |

### 消融实验

| 配置 | CUB H | SUN H | AWA2 H | 说明 |
|------|-------|-------|--------|------|
| Baseline (ViT only) | 59.3 | 45.2 | 65.0 | 无提示 |
| +视觉提示+WVPF+SVPF | 72.7 | 51.4 | 68.6 | 视觉提示贡献+13.4 |
| +语义提示+WSPF+SSPF | 73.9 | 52.2 | 76.2 | 语义提示贡献更大+14.6 |
| +双提示无融合 | 65.6 | 48.9 | 67.2 | 不融合反而不如单提示 |
| 完整VSPCN | **75.7** | **53.8** | **77.6** | 全部组件协同最优 |
| 无adapter(绿色✓) | 74.9 | 52.9 | 72.2 | adapter在AWA2上贡献5.4% |

### 关键发现

- 语义提示贡献（H提升14.6%/7.0%/11.2%）大于视觉提示（13.4%/6.2%/3.6%），说明注入语义信息对GZSL更关键
- 双提示不融合时性能反而低于单提示——说明融合机制才是协作的关键
- 即使使用ImageNet-1k ViT-Base，也超过了ImageNet-21k ViT-Large方法（71.0% vs 75.7% on CUB）
- 最优 $\alpha_v=0.05$（视觉提示几乎全依赖注意力），$\alpha_s=0.8$（语义提示更依赖偏置），两类提示信息融合模式截然不同

## 亮点与洞察

- 将prompt tuning从"训练效率工具"提升为"防止过拟合的根本方案"——冻结骨干比微调骨干更不容易过拟合已见类
- 弱融合→强融合的分层设计符合直觉：浅层特征初级用简单融合，深层特征成熟用复杂融合
- attention map可视化清晰展示了互补关系：视觉提示关注局部区域，语义提示关注语义相关区域，CLS综合两者
- t-SNE可视化显示VSPCN的类内紧凑性和类间分离度明显优于ZSLViT

## 局限与展望

- 未使用CLIP等大规模预训练VLM作为骨干，在更大模型上的效果未知
- 仅使用Glove作为语义属性编码，可尝试BERT/CLIP text encoder等更强编码器
- 超参数较多（$\lambda_{CED}$, $\lambda_{SKD}$, $\alpha_v$, $\alpha_s$, $\alpha_a$, $\gamma$, $\eta_1$, $\eta_2$），不同数据集需分别调优
- 代码未开源，可复现性受限

## 相关工作与启发

- 与VPT/SP等只学单一提示的方法相比，"双提示协作"是关键创新，可迁移到其他视觉-语义任务
- 生成式方法（GAN/VAE/Diffusion）和嵌入式方法各有优劣，VSPCN作为嵌入式方法已能匹敌甚至超越生成式
- 可尝试将VSPCN与生成式方法结合：用prompt-tuned特征指导生成unseen class特征

## 评分

- 新颖性: ⭐⭐⭐⭐ 双提示协作+分层融合设计新颖，但整体仍在attention范畴内
- 实验充分度: ⭐⭐⭐⭐⭐ 三数据集+详细消融+超参分析+可视化+与生成式方法全面对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图表质量高
- 价值: ⭐⭐⭐⭐ 在GZSL领域取得新SOTA，方法设计有启发性

<!-- RELATED:START -->

## 相关论文

- [UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [SpLIP: Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](../../ECCV2024/multimodal_vlm/elevating_all_zeroshot_sketchbased_image_retrieval_through_m.md)

<!-- RELATED:END -->
