---
title: >-
  [论文解读] Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification
description: >-
  [ECCV 2024][医学图像][few-shot learning] 提出 PEMP 框架，将病理学先验知识（视觉样例 + 文本描述）融入 patch 级和 slide 级的 prompt 中，结合 CLIP 进行多实例 prompt learning，在少样本弱监督 WSI 分类任务上平均超越 SOTA 方法 4%。
tags:
  - ECCV 2024
  - 医学图像
  - few-shot learning
  - 提示学习
  - Whole Slide Image
  - Multiple Instance Learning
  - 视觉语言
---

# Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification

**会议**: ECCV 2024  
**arXiv**: [2407.10814](https://arxiv.org/abs/2407.10814)  
**代码**: 无  
**领域**: 医学图像分析 / 病理图像  
**关键词**: few-shot learning, Prompt Learning, Whole Slide Image, Multiple Instance Learning, Vision-Language Model

## 一句话总结

提出 PEMP 框架，将病理学先验知识（视觉样例 + 文本描述）融入 patch 级和 slide 级的 prompt 中，结合 CLIP 进行多实例 prompt learning，在少样本弱监督 WSI 分类任务上平均超越 SOTA 方法 4%。

## 研究背景与动机

病理切片（WSI）分类在癌症诊断中至关重要，但由于 WSI 的 gigapixel 分辨率，通常将其切分为大量 patch，采用多实例学习（MIL）范式进行弱监督分类。现有 MIL 方法需要大量训练样本，然而在临床实践中，由于患者隐私、罕见病种、新兴疾病等原因，往往只能获取极少量的 WSI 数据。

Few-shot Weakly Supervised WSI Classification（FSWC）应运而生，但面临核心矛盾：训练样本极少（如 2/4/8/16/32 张 slide），且仅有 slide 级标注。基于 CLIP 等视觉-语言模型的 prompt learning 是有前景的方向，但现有方法（如 CoOp、TOP）要么只关注 patch 级 prompt，要么仅使用文本 prompt，忽视了病理领域高度专业化的视觉特征与对应术语之间的关联。

核心 idea：像病理学家从教科书中学习一样，将任务相关的**视觉样例**和**文本描述**作为先验知识，同时注入视觉和文本 prompt 的 patch 级和 slide 级，双侧知识增强引导模型在少样本下准确识别关键病理模式。

## 方法详解

### 整体框架

PEMP 基于冻结的 CLIP 模型，包含三个学习过程：
- **Visual Prompt Learning**：在视觉侧融入 patch 级和 slide 级的病理图像样例
- **Textual Prompt Learning**：在文本侧融入对应的病理语言描述
- **Two-level Prompt Alignment**：通过对比学习对齐视觉和文本 prompt

输入为 WSI 及其 slide 标签，输出为分类预测概率。中间经过 patch 特征提取 → 视觉样例匹配 → Messenger Layer（patch 间建模）→ Summary Layer（聚合为 slide 特征）→ 与文本特征对比分类。

### 关键设计

1. **视觉先验知识注入（Visual Prompt Construction）**:

    - 功能：为每个分类任务构建典型的 patch 和 slide 视觉样例作为固定 prompt
    - 核心思路：由病理专家从权威教材中挑选代表性图像，如预后差的宫颈癌病理特征包括"高级别异型性"、"血管侵犯"、"坏死"等 patch 样例，以及"模糊肿瘤边界、低间质比"等 slide 样例
    - 设计动机：在少样本场景下，仅靠有限训练数据难以获取有效知识，引入外部典型样例可引导模型关注任务相关的关键病理模式
    - 实现：用 CLIP 图像编码器提取样例特征 $z_l = \phi_{img}(e_l)$，通过余弦相似度匹配每个 patch 最相似的样例，拼接为增强特征 $f_{i,j}^e$

2. **Messenger Layer 与 Summary Layer**:

    - 功能：建模同一 slide 内 patch 间关系，并聚合 patch 特征为 slide 特征
    - Messenger Layer：轻量级自注意力层，输入增强后的 patch 特征 $F_i^e$，通过标准注意力机制 $F_i^{ML} = \text{softmax}(\frac{QK^\top}{\sqrt{d_w}})V$ 捕获 patch 间的空间关系
    - Summary Layer：基于 attention pooling 的聚合层，通过可学习权重 $a_{i,j} = \frac{\exp(w^\top \tanh(Vf_{i,j}^\top))}{\sum_j \exp(w^\top \tanh(Vf_{i,j}^\top))}$ 将所有 patch 特征加权求和为 slide 特征 $F_i^S$
    - 和之前方法的区别：TOP 等方法直接用平均池化或简单注意力，缺乏 patch 间的交互建模

3. **文本先验知识注入（Textual Prompt Construction）**:

    - 功能：在文本侧构建三层结构化 prompt——Slide Task Token、Slide-level Descriptive Token、Patch-level Descriptive Token
    - 每层都包含固定的病理描述和可学习 prompt 向量（如 $[\alpha]_1[\alpha]_2\ldots[\alpha]_M$），分别对应任务类别描述、slide 级病理特征描述、patch 级病理特征描述
    - 设计动机：病理图像的专业术语对 CLIP 来说可能是"unseen"的，仅靠文本难以激活正确特征；配合视觉样例形成跨模态对齐

4. **双层对齐对比学习（Two-level Alignment）**:

    - 总损失函数：$\mathcal{L}_{total} = \mathcal{L}_t + \lambda_1 \mathcal{L}_s + \lambda_2 \mathcal{L}_p$
    - $\mathcal{L}_t$：slide 视觉特征与 slide 文本特征的对齐（完成分类任务）
    - $\mathcal{L}_s$：slide 级视觉样例与 slide 级文本描述的对齐
    - $\mathcal{L}_p$：patch 级视觉样例与 patch 级文本描述的对齐
    - 基本形式为标准对比损失：$\mathcal{L} = -\sum_{F_i} \log \frac{\exp(\text{sim}(F_i, T_y)/\tau)}{\sum_{i=1}^{U} \exp(\text{sim}(F_i, T_i)/\tau)}$

### 损失函数 / 训练策略

- 三部分 AC-Loss 形式一致，均为负对数似然对比损失
- CLIP 的图像编码器和文本编码器参数冻结，仅更新可学习 prompt 向量、Messenger Layer、Summary Layer 和 projector
- 推理时通过 softmax 计算视觉特征与各类别文本特征的匹配概率

## 实验关键数据

### 主实验

**任务1：宫颈癌生存预后预测（C-index）**

| 数据集 | 方法 | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|--------|------|---------|---------|--------|--------|--------|
| In-house | TOP (NeurIPS'23) | 0.652 | 0.608 | 0.574 | 0.539 | 0.508 |
| In-house | **PEMP (ours)** | **0.667** | **0.637** | **0.614** | **0.587** | **0.562** |
| TCGA-CESC | TOP | 0.611 | 0.597 | 0.566 | 0.536 | 0.518 |
| TCGA-CESC | **PEMP (ours)** | **0.637** | **0.624** | **0.602** | **0.577** | **0.551** |

**任务2：淋巴结转移预测（AUC）**

| 数据集 | 方法 | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|--------|------|---------|---------|--------|--------|--------|
| In-house | TOP | 0.825 | 0.819 | 0.801 | 0.787 | 0.762 |
| In-house | **PEMP** | **0.849** | **0.838** | **0.824** | **0.801** | **0.783** |
| TCGA-CESC | TOP | 0.799 | 0.761 | 0.744 | 0.708 | 0.679 |
| TCGA-CESC | **PEMP** | **0.818** | **0.795** | **0.760** | **0.726** | **0.704** |

**任务3：圆细胞肿瘤亚型分类（AUC）**

| 方法 | 32-shot | 16-shot | 8-shot | 4-shot | 2-shot |
|------|---------|---------|--------|--------|--------|
| TOP | 0.682 | 0.652 | 0.633 | 0.584 | 0.560 |
| **PEMP** | **0.751** | **0.718** | **0.685** | **0.643** | **0.625** |

罕见病分类上 PEMP 平均 AUC 提升 6.2%，优势尤为显著。

### 消融实验

| 配置 | 32-shot | 2-shot | 说明 |
|------|---------|--------|------|
| w/o v&t em.（退化为 CoOp） | 0.641 | 0.490 | 无任何视觉/文本样例 |
| w/o vision em. | 0.655 | 0.511 | 仅用文本描述 |
| w/o text em. | 0.658 | 0.533 | 仅用视觉样例 |
| w/o Summary Layer | 0.632 | 0.487 | 用平均池化替代 |
| w/o Messenger Layer | 0.664 | 0.554 | 无 patch 间交互 |
| w/o Slide-level Prompts | 0.656 | 0.525 | 无 slide 级 prompt |
| w/o AC-Loss | 0.660 | 0.549 | 无样例对齐损失 |
| **PEMP (full)** | **0.667** | **0.562** | 完整模型 |

### 关键发现

- **Summary Layer（Attention Pooling）贡献最大**：移除后性能下降最严重（32-shot 下降 3.5%），说明 MIL 聚合方式是关键
- **视觉和文本样例互补**：单独移除任一侧都会掉点，但文本样例移除（w/o text em.）在极少样本时影响更大
- **双侧知识增强在极少样本（2-shot）时优势更明显**：full model 相比 CoOp 在 2-shot 上提升 7.2%

## 亮点与洞察

- **病理教科书式学习范式**：模仿病理医生从教科书学习的过程，引入视觉样例+文本描述，是非常自然且有效的先验知识注入方式
- **双层双侧设计**：patch+slide 两个粒度 × vision+text 两个模态的完整覆盖，系统性很强
- **高可解释性**：通过可视化 patch/slide 样例的匹配结果，可以看到模型学到了正确的病理模式（如血管侵犯、坏死等）
- **可迁移的思路**：这种"引入领域专家知识作为 prompt"的范式可以推广到其他需要专业知识但数据稀缺的领域

## 局限与展望

- **依赖病理专家提供样例和描述**：需要人工构建每个任务的视觉和文本先验，扩展新任务时有额外成本
- **样例数量和质量的敏感性**未充分分析：不同数量/质量的样例对性能影响如何？
- **仅针对分类任务**：未探索在 WSI 检测/分割等任务上的适用性
- **CLIP 的局限**：CLIP 预训练数据中医学图像比例有限，更强的病理 VLM（如 PLIP、CONCH）可能进一步提升

## 相关工作与启发

- **vs TOP (NeurIPS'23)**: TOP 仅在文本侧引入语言描述，缺乏视觉侧知识；PEMP 双侧增强更全面
- **vs CoOp (IJCV'22)**: CoOp 仅用可学习文本 prompt，不含领域知识；PEMP 引入静态+可学习的混合 prompt
- **vs MI-Zero / PLIP**: 这些工作聚焦于构建大规模病理 VLM，但缺乏面向 FSWC 的有效适配策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 双层双侧病理知识增强 prompt learning 设计新颖，但核心组件（注意力、对比学习）相对标准
- 实验充分度: ⭐⭐⭐⭐ 三个临床任务、五个数据集、完整消融实验、可视化分析，但缺乏与更多 VLM 的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述合理，图示直观
- 价值: ⭐⭐⭐⭐ 对少样本病理图像分析有实际临床价值，可解释性强，适合实际部署

<!-- RELATED:START -->

## 相关论文

- [Unleashing the Power of Prompt-driven Nucleus Instance Segmentation](unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)
- [TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data](tip_tabularimage_pretraining_for_multimodal_classification_w.md)
- [Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multimodal_contrastive_learning_with_exper.md)

<!-- RELATED:END -->
