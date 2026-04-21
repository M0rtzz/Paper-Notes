---
title: >-
  [论文解读] Zero-shot Object Counting with Good Exemplars (VA-Count)
description: >-
  [ECCV 2024][多模态][零样本目标计数] 提出VA-Count框架，通过样本增强模块（EEM）利用Grounding DINO发现高质量正负样本，结合噪声抑制模块（NSM）用对比学习区分正负密度图，实现零样本目标计数在FSC-147和CARPK上的SOTA表现。
tags:
  - ECCV 2024
  - 多模态
  - 零样本目标计数
  - 视觉语言预训练
  - 对比学习
  - 样本增强
  - 噪声抑制
---

# Zero-shot Object Counting with Good Exemplars (VA-Count)

**会议**: ECCV 2024  
**arXiv**: [2407.04948](https://arxiv.org/abs/2407.04948)  
**代码**: [GitHub](https://github.com/HopooLinZ/VA-Count) (有)  
**领域**: 多模态VLM  
**关键词**: 零样本目标计数, 视觉语言预训练, 对比学习, 样本增强, 噪声抑制

## 一句话总结

提出VA-Count框架，通过样本增强模块（EEM）利用Grounding DINO发现高质量正负样本，结合噪声抑制模块（NSM）用对比学习区分正负密度图，实现零样本目标计数在FSC-147和CARPK上的SOTA表现。

## 研究背景与动机

目标计数在安防监控等场景中至关重要。传统方法局限于特定类别（人群、车辆等），无法泛化到未见类别。现有的类别无关计数方法可分为：

**Few-shot方法**：需要少量标注框，但新类别仍需标注，实际部署受限

**Reference-free方法**：无需标注但无法指定计数类别，容易受背景噪声干扰

**Zero-shot方法**：仅需类别名称即可计数，最具实用价值

现有零样本计数方法的核心问题在于**无法有效识别高质量样本**：
- **图文关联方法**（如CLIP-Count）：直接用CLIP对齐文本和图像，难以精确表示非典型形状的目标类别
- **样本发现方法**（如ZSC）：通过文本生成原型匹配图像块，但块的选择是任意的，无法准确框出完整物体，且局限于训练集中的预定义类别

核心矛盾：如何在不需要手动标注的情况下，自动发现能准确代表目标类别的高质量样本，并建立鲁棒的视觉关联。VA-Count的核心idea是利用VLP模型（Grounding DINO）的开放词汇检测能力来发现样本，再通过二分类过滤和对比学习来提升样本质量并抑制噪声。

## 方法详解

### 整体框架

VA-Count由两个核心模块组成：
- **样本增强模块（EEM）**：负责从图像中发现并筛选高质量的正样本和负样本
- **噪声抑制模块（NSM）**：利用对比学习区分正负样本的密度图，抑制错误样本的影响

整体流程：输入图像 → Grounding DINO检测候选框 → 单物体过滤器筛选 → 正负样本分离 → Counter生成正负密度图 → 对比损失+密度损失联合优化

### 关键设计

1. **Grounding DINO引导的候选框选择**:

    - 功能：利用Grounding DINO的开放词汇检测能力，为任意类别生成候选边界框
    - 核心思路：
        - 正样本检测：输入特定类别文本（如"dog"），Grounding DINO输出候选框及其置信度
        - 负样本检测：输入通用文本"object"，检测图像中所有物体的框
        - 去重：通过IoU阈值（τ_iou=0.5）过滤与正样本重叠的负样本框
    - 设计动机：Grounding DINO在大规模数据上预训练，具备强大的开放类别检测能力，保证框架对任意类别的适应性

2. **单物体样本过滤器**:

    - 功能：确保每个样本框中仅包含一个目标物体
    - 核心思路：构建一个二分类器δ(·)，使用冻结的CLIP ViT-B/16提取特征，接FFN进行二分类判断框中是单个还是多个物体
    - 训练数据构造：
        - 单物体正样本：训练集中的标注样本
        - 多物体负样本：随机裁剪的图像块和整图
        - 数据按7:3划分训练/验证，类别不重叠以保持类别无关性
    - 设计动机：Grounding DINO的高置信度框可能包含多个物体（多物体框的置信度可能高于单物体框），这会破坏后续视觉关联学习，必须严格筛选

3. **Counter网络（特征交互与密度图生成）**:

    - 功能：基于正负样本分别生成密度热力图
    - 核心思路：基于CounTR架构，使用图像特征作为Query，样本特征的线性投影作为Key/Value进行交叉注意力融合
    - 融合公式：F_fuse = Γ_fuse(F_query, W^k·F_key, W^v·F_value)
    - 解码器将融合特征上采样到原图尺寸，输出密度热力图
    - 正负样本分别选取置信度最高的3个框作为最终样本

### 损失函数 / 训练策略

**对比损失 L_C**：
- 正密度图与GT密度图的相似度应最大化
- 负密度图与GT密度图的相似度应最小化
- 采用InfoNCE形式：L_C = -log(exp(sim(D^p, D^g)) / (exp(sim(D^p, D^g)) + exp(sim(D^n, D^g))))

**密度损失 L_D**：
- 正样本密度图与GT密度图的逐像素均方误差
- L_D = (1/HW) Σ||D^p - D^g||²

**总损失**：L_total = L_C + L_D

**训练策略**：
- 采用CounTR的两阶段训练：MAE预训练 + AdamW微调
- 学习率：10⁻⁵，batch size：8
- Grounding DINO置信度阈值 τ_l = 0.02
- 单物体分类器训练100个epoch，学习率1e-4

## 实验关键数据

### 主实验：FSC-147零样本计数

| 方法 | 类型 | Val MAE↓ | Val RMSE↓ | Test MAE↓ | Test RMSE↓ |
|------|------|---------|----------|----------|-----------|
| ZSC | Zero-shot | 26.93 | 88.63 | 22.09 | 115.17 |
| CLIP-Count | Zero-shot | 18.79 | 61.18 | 17.78 | 106.62 |
| PseCo | Zero-shot | 23.90 | 100.33 | 16.58 | 129.77 |
| **VA-Count** | **Zero-shot** | **17.87** | 73.22 | **17.88** | 129.31 |
| CounTR | Few-shot(3) | 13.13 | 49.83 | 11.95 | 91.23 |
| CACViT | Few-shot(3) | 10.63 | 37.95 | 9.13 | 48.96 |

### 跨域实验：CARPK

| 方法 | 类型 | F→C MAE↓ | F→C RMSE↓ | 说明 |
|------|------|----------|-----------|------|
| FamNet | Few-shot(3) | 28.84 | 44.47 | 跨域few-shot |
| RCC | Zero-shot | 21.38 | 26.61 | - |
| CLIP-Count | Zero-shot | 11.96 | 16.61 | - |
| Grounding DINO | Zero-shot | 29.72 | 31.60 | 直接用G-DINO计数 |
| G-DINO + 过滤器 | Zero-shot | 18.54 | 21.71 | 加单物体过滤器 |
| **VA-Count** | **Zero-shot** | **10.63** | **13.20** | SOTA |

### 消融实验

| G(·) | Φ(·) | L_D | L_C | Val MAE | Test MAE | 说明 |
|------|------|-----|-----|---------|----------|------|
| ● | ○ | ○ | ○ | 52.82 | 54.48 | 仅Grounding DINO检测 |
| ● | ● | ○ | ○ | 52.12 | 54.27 | +单物体过滤器（略有提升） |
| ● | ● | ● | ○ | 19.63 | 18.93 | +密度损失（大幅下降） |
| ● | ● | ● | ● | **17.87** | **17.88** | +对比损失（继续提升） |

### 关键发现
- VA-Count在FSC-147零样本设置中MAE最低，证明样本发现策略优于ZSC
- 跨域CARPK实验表现突出，零样本性能接近few-shot方法（MAE 10.63 vs 10.44）
- 单物体过滤器在CARPK上效果显著，将Grounding DINO的MAE从29.72降至18.54（减少约10）
- 对比损失的引入在密度损失基础上进一步降低MAE约2个点
- 密度损失是最关键的组件，引入后MAE从52降至~19

## 亮点与洞察
- **巧妙利用VLP模型**：将Grounding DINO的开放词汇检测能力引入零样本计数，突破了依赖固定类别原型的局限
- **正负样本双通道设计**：不仅利用正样本建立关联，还通过负样本（非目标类别物体）进行对比学习抑制噪声，类似于分类中的"教模型什么不是"
- **单物体过滤器简单有效**：一个简单的二分类器解决了Grounding DINO框可能包含多物体的关键问题
- **跨域泛化性强**：FSC-147→CARPK的迁移性能优于大多数baseline

## 局限与展望
- RMSE指标不如CLIP-Count，说明存在少数极端误差样本，鲁棒性有待提升
- Grounding DINO本身的计算开销较大，推理效率可能是瓶颈
- 单物体分类器的训练数据构造方式较简单（随机裁剪），可能引入类别偏差
- 在密集场景中Grounding DINO的检测质量可能下降，影响样本质量
- 固定选取top-3样本，缺乏自适应机制
- 可探索用SAM替代或辅助Grounding DINO进行更精确的实例分割

## 相关工作与启发
- **CounTR**：Counter网络的基础架构来源，两阶段MAE预训练+微调的训练策略
- **CLIP-Count**：用CLIP编码文本和图像建立语义关联，VA-Count在此基础上引入显式样本发现
- **Grounding DINO**：核心外部工具，提供开放词汇检测能力
- **BMNet**：双线性匹配网络用于精细相似度评估，启发了特征交互的设计
- 启发：VLP模型作为"外部知识源"为零样本任务提供了新范式，关键在于如何处理其输出的噪声

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将Grounding DINO引入零样本计数，正负样本双通道+对比学习的设计有创意
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+跨域实验+逐组件消融+定性分析，较为全面
- 写作质量: ⭐⭐⭐ 方法描述稍显冗长，符号定义较多，可读性一般
- 价值: ⭐⭐⭐⭐ 零样本计数实用价值高，思路可迁移到其他零样本视觉任务

<!-- RELATED:START -->

## 相关论文

- [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](metaprompting_for_automating_zeroshot_visual_recognitio.md)
- [SpLIP: Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zeroshot_sketchbased_image_retrieval_through_m.md)
- [MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_visionlanguage_mod.md)
- [RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](../../ACL2025/multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)
- [Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](../../ICLR2026/multimodal_vlm/bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)

<!-- RELATED:END -->
