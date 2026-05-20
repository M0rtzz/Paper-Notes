---
title: >-
  [论文解读] Zero-Shot Object Counting with Good Exemplars (VA-Count)
description: >-
  [ECCV 2024][LLM/NLP][零样本计数] 提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。
tags:
  - "ECCV 2024"
  - "LLM/NLP"
  - "零样本计数"
  - "视觉-语言预训练"
  - "样例增强"
  - "噪声抑制"
  - "Grounding DINO"
---

# Zero-Shot Object Counting with Good Exemplars (VA-Count)

**会议**: ECCV 2024  
**arXiv**: [2407.04948](https://arxiv.org/abs/2407.04948)  
**代码**: [https://github.com/HopooLinZ/VA-Count](https://github.com/HopooLinZ/VA-Count)  
**领域**: LLM/NLP  
**关键词**: 零样本计数, 视觉-语言预训练, 样例增强, 噪声抑制, Grounding DINO

## 一句话总结

提出 VA-Count，一种基于视觉关联的零样本物体计数框架，通过 Grounding DINO 驱动的样例增强模块和对比学习噪声抑制模块，为任意类别建立高质量样例与图像间的鲁棒视觉关联。

## 研究背景与动机

- 零样本物体计数（ZOC）旨在仅知道类别名称的情况下计数图像中的物体，无需标注
- 现有 ZOC 方法的核心问题：无法有效识别高质量样例（exemplars）
- 两类现有方法各有不足：
  1. **图文对齐方法**（CLIP-Count, VLCount）：依赖直接的图文关联，对非典型形状的类别表示不足
  2. **类别相关样例搜索**（ZSC）：依赖任意 patch 选择，无法准确勾勒完整物体；且受限于预定义类别
- **本文目标**：用检测驱动的样例发现方法（Grounding DINO），同时融合文本和视觉表示

## 方法详解

### 整体框架

两大核心模块协同工作：
1. **样例增强模块（EEM）**：用 Grounding DINO 发现正负样例，并用单物体分类器过滤
2. **噪声抑制模块（NSM）**：用对比学习区分正负样例的密度图，减少错误样例的影响

### 关键设计

**样例增强模块（EEM）**：

*Grounding DINO 引导的框选择*：
- 正样本：输入图像 + 正文本标签（具体类别名） → 获取正候选框 B^p
- 负样本：输入图像 + 负文本标签（"object"） → 获取负候选框 B^n
- Logits 阈值 τ_l = 0.02

*去重过滤*：
- 对负样本框，用 IoU 阈值 τ_iou = 0.5 去除与正样本框重叠的部分

*单物体样例过滤*：
- 二分类器 δ(·) = FFN(CLIP-ViT(b))
- 用冻结的 CLIP-ViT-B/16 + 可训练 FFN 判断候选框是否恰好包含一个物体
- 训练数据：正样本=训练集中标注的单物体样例，负样本=随机裁剪 patch + 整图
- 确保样例的干净性

**噪声抑制模块（NSM）**：

*计数器 Γ(·)*：
- 基于 CounTR 架构：图像编码器 + 样例-图像交互模块 + 解码器
- 图像特征为 Query，样例特征线性投影为 Key/Value
- 分别用正/负样例生成正/负密度图

*对比学习*：
- L_C = -log(exp(sim(D^p,D^g)) / (exp(sim(D^p,D^g)) + exp(sim(D^n,D^g))))
- 最大化正密度图与真值相似度，最小化负密度图与真值相似度
- 总损失：L_total = L_C + L_D

### 损失函数 / 训练策略

- 密度损失 L_D：MSE(D^p, D^g)
- 对比损失 L_C：正负密度图与真值的对比学习
- 优化器：AdamW，学习率 10⁻⁵，batch size 8
- 两阶段训练：MAE 预训练 + 微调
- 每张图选 top-3 正样例和 top-3 负样例

## 实验关键数据

### 主实验

FSC-147 数据集：

| 方法 | 类型 | Val MAE | Val RMSE | Test MAE | Test RMSE |
|------|------|---------|----------|----------|-----------|
| CLIP-Count | 零样本 | 17.78 | 55.43 | 18.97 | 95.93 |
| VLCount | 零样本 | 18.20 | 60.63 | 19.18 | 103.29 |
| **VA-Count** | **零样本** | **最优** | **最优** | **最优** | **最优** |

CARPK 数据集（跨数据集泛化）：VA-Count 同样超越现有零样本和少样本方法。

### 消融实验

| 组件 | MAE 变化 |
|------|---------|
| 无 EEM（随机 patch） | 显著下降 |
| 无单物体过滤器 | 下降（多物体样例引入噪声） |
| 无 NSM（无对比学习） | 下降（错误样例未被抑制） |
| EEM + NSM 完整 | 最优 |

### 关键发现

1. Grounding DINO 提供了显著优于随机 patch 选择的样例质量
2. 单物体过滤至关重要——DINO 置信度高的框可能包含多个物体
3. 对比学习有效区分了正负样例对密度图的影响
4. "object" 作为通用负文本标签能检测到非目标类别的物体

## 亮点与洞察

- **检测驱动的样例发现**：将 Grounding DINO 的通用检测能力引入零样本计数，实现了任意类别的高质量样例获取
- **正负样例对比学习**：不仅学习"什么是目标"，还学习"什么不是目标"，双向约束更鲁棒
- **模块化设计**：EEM 和 NSM 独立可控，易于理解和扩展
- **实用的负样本策略**：用 "object" 检测所有物体，去除与正类重叠的即为负样本

## 局限性 / 可改进方向

- 依赖 Grounding DINO 的检测质量，对 DINO 难以检测的对象（极小/遮挡严重）可能失败
- 单物体分类器需要额外训练数据和训练过程
- 每张图仅选 3 个正/负样例，更多样例是否能进一步提升？
- 计算成本相对较高（Grounding DINO + CLIP + CounTR）
- 未来方向：端到端训练整个流水线、用 SAM 替代/增强样例定位

## 相关工作与启发

- CounTR 的 MAE 预训练 + 样例匹配框架是本文的基础架构
- Grounding DINO 的引入展示了大规模 VLP 模型在下游任务中的即插即用价值
- 正负样例对比学习的思路可推广到其他需要样例匹配的任务（如少样本分割）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 3.5 |
| 技术深度 | 3.5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总分 | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/llm_nlp/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[ECCV 2024\] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection](adaclip_adapting_clip_with_hybrid_learnable_prompts_for_zero.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](../../ACL2025/llm_nlp/bilingual_zero-shot_stance_detection.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)

</div>

<!-- RELATED:END -->
