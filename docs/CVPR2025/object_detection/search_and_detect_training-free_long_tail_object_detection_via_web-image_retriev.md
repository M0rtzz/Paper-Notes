---
title: >-
  [论文解读] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval
description: >-
  [CVPR 2025][目标检测][开放词汇检测] SearchDet提出了一种完全免训练的长尾目标检测框架，通过从Web检索正负样本图像、注意力加权查询生成、SAM区域提议和热力图联合定位，在ODinW上比GroundingDINO提升48.7% mAP、在LVIS上提升59.1% mAP，展示了利用Web作为外部动态记忆进行推理阶段增强的巨大潜力。
tags:
  - CVPR 2025
  - 目标检测
  - 开放词汇检测
  - 长尾检测
  - Web图像检索
  - 免训练
  - SAM
---

# Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval

**会议**: CVPR 2025  
**arXiv**: [2409.18733](https://arxiv.org/abs/2409.18733)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 开放词汇检测, 长尾检测, Web图像检索, 免训练, SAM

## 一句话总结

SearchDet提出了一种完全免训练的长尾目标检测框架，通过从Web检索正负样本图像、注意力加权查询生成、SAM区域提议和热力图联合定位，在ODinW上比GroundingDINO提升48.7% mAP、在LVIS上提升59.1% mAP，展示了利用Web作为外部动态记忆进行推理阶段增强的巨大潜力。

## 研究背景与动机

**领域现状**：开放词汇目标检测（OVD）旨在检测任意文本标签描述的物体。当前主流方法如GroundingDINO、GLIP、T-Rex2等通过大规模图文预训练获得了强大的零样本能力，但性能仍受限于预训练数据的分布覆盖——对训练中见过的常见物体表现好，对长尾罕见物体表现差。

**现有痛点**：要进一步提升这些模型在长尾物体上的性能，需要昂贵的持续预训练或任务特定微调。例如，GroundingDINO在检测"Mountain Dew瓶子"这类细粒度概念时会完全失败，因为模型的参数化记忆中缺乏足够的相关视觉经验。

**核心矛盾**：模型的知识被冻结在参数中，无法动态扩展；而Web搜索引擎可以实时获取任意概念的视觉示例，但之前没有有效方法将这种外部检索能力整合到目标检测pipeline中。

**本文目标**：设计一种完全不需要训练的检测框架，利用Web检索的图像在推理阶段动态增强检测能力。

**切入角度**：作者观察到搜索引擎本身就是一个不断扩展的物体"视觉记忆库"——给任何文本标签，Google都能返回高质量的相关图像。这种检索式的视觉表示天然具备长尾覆盖能力。

**核心 idea**：对于目标标签，从Web检索正样本和负样本图像，用注意力机制加权融合为查询嵌入，再结合SAM区域提议和相似性热力图进行联合定位，实现完全免训练的高精度检测。

## 方法详解

### 整体框架

给定一张输入图像和目标标签，SearchDet分四步工作：(1) 从Web检索该标签的正样本图像和由LLM生成负标签的负样本图像；(2) 用注意力加权的方式生成经过调整的查询嵌入；(3) 利用SAM生成区域提议并通过自适应阈值筛选；(4) 生成相似性热力图，与筛选后的SAM区域取交集输出最终检测框。

### 关键设计

1. **正负样本Web检索与注意力查询生成**:

    - 功能：从Web获取目标物体和干扰物体的视觉表示，生成精炼的查询嵌入
    - 核心思路：首先用搜索引擎检索5张正样本图像和5张负样本图像（负标签由LLM生成，如"surfboard"的负标签是"waves"）。用DINOv2的CLS token分别嵌入查询图像、正样本和负样本。计算查询与每个正/负嵌入的余弦相似度作为注意力权重：$\alpha_{pos,i} = \text{softmax}(S_{pos,i})$，然后加权聚合：$A_{pos} = \sum_i \alpha_{pos,i} e_{pos,i}$，最终查询为 $q_{adjusted} = A_{pos} - A_{neg}$。注意力机制确保与查询图像更相关的检索图像获得更高权重。
    - 设计动机：负样本至关重要——检索"surfboard"的图像通常也包含海浪，如果不减去负嵌入，模型会把海浪也当作目标。相比简单的均值池化减法，注意力加权能更精准地适配输入图像的具体场景。

2. **SAM区域提议与频率自适应阈值**:

    - 功能：生成候选区域并自适应判断目标是否存在
    - 核心思路：用HQ-SAM在输入图像上生成全部分割区域。对每个区域，用DINOv2嵌入其mask后的图像得到区域嵌入，计算与查询嵌入的欧氏距离。将距离排序后分桶（每桶包含$m$个距离，$m$为查询数），如果某个桶中单一mask占比超过80%则选为候选。关键创新在于验证步骤：计算候选mask在桶内的平均距离 $\mu_{D_j}$，与参考分布（各查询嵌入间的互距离分布）的均值 $\mu_R$ 比较，若偏差超过3个标准差则拒绝。
    - 设计动机：简单百分位阈值在目标不存在时仍会输出结果导致假阳性。频率自适应方法根据距离分布的形状动态调整，在目标不存在时距离分布均匀——不会有某个mask在低距离桶中主导。

3. **热力图生成与联合定位**:

    - 功能：提供独立于SAM的检测信号并精化定位
    - 核心思路：将输入图像的DINOv2 patch特征上采样，用同样的注意力加权方法生成查询嵌入（但这次用patch特征而非CLS token），计算查询与每个patch的余弦相似度生成热力图。热力图二值化后与SAM区域取交集——每个通过筛选的SAM区域只有在与热力图有非空交集时才输出边框。这种联合策略互补性强：SAM可能遗漏物体部分区域，热力图可以补充；反之热力图的边界可能不精确，SAM提供更准确的分割边界。
    - 设计动机：单独依赖SAM区域提议时，如果SAM漏检了目标物体的区域，就无法检测到；而单独用热力图又缺乏精确的物体边界。两者结合提供了更高的召回率和更精确的定位。

## 实验关键数据

### 主实验

| 方法 | 骨干 | COCO | LVIS | ODinW-35 | Roboflow100 |
|------|------|------|------|----------|-------------|
| GLIP-L | Swin-L | 49.8 | 26.9 | 23.4 | 8.6 |
| GroundingDINO-L | Swin-L | 48.4 | 27.4 | 22.3 | 8.3 |
| T-Rex2 (Text) | Swin-L | 52.2 | 45.8 | 22.0 | 10.5 |
| T-Rex2 (Visual-G) | Swin-L | 46.5 | 45.3 | 27.8 | 18.5 |
| **SearchDet (Ours)** | DINOv2-L | **59.3** | 43.6 | **33.1** | **27.9** |

在10-shot设置下，SearchDet达到61.4 mAP，比之前SOTA的DE-ViT (52.9)高16.1%。

### 消融实验

| 配置 | COCO mAP | 说明 |
|------|---------|------|
| 完整方法 | 59.34 | — |
| 仅正样本（无负样本） | 45.80 | 下降22.82%，负样本至关重要 |
| 无热力图精化 | 51.07 | 下降13.94%，热力图提供关键补充 |
| 均值池化替代注意力加权 | 55.47 | 下降6.5%，注意力适配更有效 |

### 关键发现

- **负样本的重要性**压倒性地最大（去掉后mAP从59.3降到45.8），说明在开放词汇检测中消除视觉干扰是关键
- 检索图像数量与性能正相关：从1张到10张正/负样本，mAP从49.7稳步增长到59.3，增幅19.4%
- 检索图像的嵌入在不同日期获取时高度一致（余弦相似度>0.9），说明Web检索的稳定性足以支撑实际应用
- 在ODinW和Roboflow100这类领域多样化数据集上优势最大（提升100%+），说明Web检索的长尾覆盖能力是其核心竞争力

## 亮点与洞察

- **将Web搜索引擎作为"外部视觉记忆"**是最核心的洞察：模型参数是静态的，但Web是动态增长的——任何新物体只要有网页图片就能被检测，无需重训练
- **LLM生成负标签**的设计简单但有效：利用语言模型的世界知识自动推断"surfboard"常与"waves"共现，省去了人工设计负类的成本
- **频率自适应阈值**是比固定阈值更鲁棒的方案：它能处理"目标不在图中"的情况，避免假阳性，这对实际部署很重要
- 整个pipeline可以迁移到视觉问答：检索相关图像作为few-shot示例来增强VLM的推理能力

## 局限与展望

- 每张图像处理约3秒（V100 GPU），加上Web检索延迟，实时性不足
- 在LVIS上略低于T-Rex2，可能因为只用5张检索图像不够覆盖1203个类别的语义空间
- 依赖Google搜索引擎，带来隐私和可用性问题；离线版本需要预构建大规模图像库
- 对模糊标签（如"20"而非"20 dollar bill"）敏感，需要更好的标签描述策略
- 未来方向：预构建离线检索库消除Web依赖、结合VLM实现更智能的标签扩展、与在线学习结合逐步精化检索质量

## 相关工作与启发

- **vs GroundingDINO / GLIP**: 这些方法将知识编码到参数中，对长尾物体支持有限；SearchDet通过外部检索动态扩展知识，在长尾场景优势巨大
- **vs T-Rex2**: T-Rex2也使用视觉提示，但需要训练阶段的对齐；SearchDet完全免训练，更灵活
- **vs DE-ViT**: 同为免训练few-shot检测，但DE-ViT需要手动提供支持图像；SearchDet自动从Web获取，更实用

## 评分

- 新颖性: ⭐⭐⭐⭐ Web检索+免训练检测的组合有新意，各组件均为现有模型的创造性组合
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、完整消融和稳定性分析，但缺少推理速度的详细分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整，但部分符号使用不够统一
- 价值: ⭐⭐⭐⭐ 展示了Web检索增强的巨大潜力，为免训练检测开辟了新方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[CVPR 2025\] Interpreting Object-level Foundation Models via Visual Precision Search](interpreting_object-level_foundation_models_via_visual_precision_search.md)
- [\[CVPR 2025\] SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection](simltd_simple_supervised_and_semi-supervised_long-tailed_object_detection.md)
- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](../../CVPR2026/object_detection/beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](../../ICCV2025/object_detection/anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)

<!-- RELATED:END -->
