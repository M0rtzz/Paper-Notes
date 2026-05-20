---
title: >-
  [论文解读] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval
description: >-
  [CVPR 2026][目标检测][组合图像检索] 提出Object-Anchored Composed Image Retrieval（OACIR）新任务和OACIRR大规模基准（160K+四元组），以及AdaFocal框架通过上下文感知注意力调制器自适应地增强对锚定实例区域的关注…
tags:
  - "CVPR 2026"
  - "目标检测"
  - "组合图像检索"
  - "实例级一致性"
  - "注意力调制"
  - "细粒度检索"
  - "视觉锚定"
---

# Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval

**会议**: CVPR 2026  
**arXiv**: [2604.05393](https://arxiv.org/abs/2604.05393)  
**代码**: [项目页](https://hahajun1101.github.io/OACIR/)  
**领域**: 目标检测/图像检索  
**关键词**: 组合图像检索, 实例级一致性, 注意力调制, 细粒度检索, 视觉锚定

## 一句话总结
提出Object-Anchored Composed Image Retrieval（OACIR）新任务和OACIRR大规模基准（160K+四元组），以及AdaFocal框架通过上下文感知注意力调制器自适应地增强对锚定实例区域的关注，在实例级检索保真度上大幅超越现有方法。

## 研究背景与动机
**领域现状**：组合图像检索（CIR）通过参考图像+修改文本的多模态查询实现灵活检索，在电商和交互搜索中广泛应用。

**核心痛点**：CIR本质上优先语义匹配，参考图像仅作为粗粒度视觉锚点——在存在视觉相似干扰项时，**无法可靠地检索用户指定的特定实例**。

**实际需求**：数字记忆检索、长期身份追踪等场景中，保证**具体实例的保真度**比宽泛语义对齐更为关键。

**核心矛盾**：需要同时完成（1）三源信息的组合推理（锚定实例+全局场景+文本修改）和（2）从充满视觉相似干扰项的gallery中精确区分目标实例。

**核心idea**：通过显式的bounding box视觉锚定 + 自适应注意力增强机制，将CIR从语义级提升到实例级。

## 方法详解

### 整体框架
查询分支：$(I_r, B_r, T_m)$ → 图像编码器 → CAAM预测调制标量 $\beta$ → 注意力激活机制增强实例区域 → 多模态编码器 → 查询表示 $f_q$
目标分支：$I_t$ → 图像编码器 → 多模态编码器 → 目标表示 $f_t$
训练：对比学习损失对齐两个分支的表示

### 关键设计
1. **Context-Aware Attention Modulator (CAAM)**：

    - 将参考图像和修改文本编入多模态编码器，同时注入 $K$ 个可学习的**上下文探针token**
    - 探针token通过与多模态输入的交互学习上下文线索
    - Transformer-based CRM（Contextual Reasoning Module）聚合推理后，线性映射输出**调制标量 $\beta$**
    - **设计动机**：实例关注的强度应随查询上下文动态变化——如果修改文本要求大幅场景变化，应适度放宽实例关注；如果仅改变背景，应强化实例关注

2. **Attention Activation Mechanism**：
   在查询分支的交叉注意力中，将 $\beta$ 作为动态偏置注入：
    $\{\hat{q}_m\} = \text{Softmax}\left(\frac{QK^T + \beta \cdot M_{B_r}}{\sqrt{d_k}}\right)V$
   其中 $M_{B_r}$ 是与bounding box空间对齐的二值mask。$\beta>0$ 增强实例区域注意力，实现自适应聚焦。

3. **OACIRR基准构建**（四阶段流水线）：

    - **图像对收集**：从DeepFashion2、Stanford Cars、Products-10K、Google Landmarks v2提取同实例跨语境图像对
    - **图像对过滤**：去除过于相似的对（防止捷径学习）+ 过滤类别中心图像
    - **四元组标注**：MLLM生成修改文本 + Grounding模型标注bounding box
    - **Gallery构建**：定向挖掘hard-negative（类别相关但实例不同的干扰项）

### 损失函数 / 训练策略
- Contrastive Alignment Loss：batch内对比学习，最大化正确查询-目标对的余弦相似度
- 差异化学习率：CAAM用1e-4，多模态编码器用1e-5
- 温度参数 $\tau = 0.07$

## 实验关键数据

### 主实验（OACIRR基准，ViT-G骨干）

| 方法 | Fashion $R_{ID}@1$ | Car $R_{ID}@1$ | Product $R_{ID}@1$ | Landmark $R_{ID}@1$ | Avg |
|------|-----------|----------|------------|-------------|-----|
| GME (7B) | 44.98 | 63.11 | 83.44 | 77.11 | 62.53 |
| SPRC (CIRR训练) | 28.62 | 25.13 | 54.39 | 40.41 | 37.30 |
| SPRC (OACIRR训练) | 65.25 | 72.87 | 86.05 | 76.32 | 74.05 |
| **AdaFocal** | **77.15** | **78.42** | **91.86** | **82.92** | **79.00** |

### 消融实验

| 配置 | $R_{ID}@1$ | R@1 | Avg | 说明 |
|------|-----------|-----|-----|------|
| 无CAAM（$\beta=0$） | 77.74 | 58.39 | 74.91 | 基线 |
| 平均池化+冻结探针 | 79.70 | 59.84 | 76.39 | 简单聚合不足 |
| Transformer CRM+可学习探针 | **82.59** | **62.88** | **79.00** | 推理能力+任务适应 |

### 关键发现
- OACIRR数据集训练使SPRC从37.30%跃升至74.05%：**实例一致性数据**是关键
- AdaFocal在此基础上再提升+4.95%：**自适应注意力调制**有效
- $R@1$与$R_{ID}@1$差距揭示现有方法的主要失败模式是**实例误识别**

## 亮点与洞察
- 将CIR从语义级推进到实例级，是检索领域的重要范式转变
- OACIRR是首个跨四个领域的大规模实例级组合检索基准，具有很高的社区价值
- CAAM的上下文感知调制机制优雅地平衡了实例保真度与组合推理

## 局限与展望
- Bounding box标注增加了用户交互成本，未来可探索自动实例锚定
- 当前仅支持单实例锚定，多实例场景待扩展
- 未探索视频级实例追踪检索

## 相关工作与启发
- 与ReID（行人重识别）的实例一致性目标一致但更通用
- 注意力偏置注入思路来自生成模型（如Prompt-to-Prompt），成功迁移到检索任务
- 对商品搜索、数字资产管理等应用有直接价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 新任务定义+新基准+新方法，三位一体
- 实验充分度: ⭐⭐⭐⭐⭐ 跨多范式对比、详尽消融、定性分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集构建流程详细
- 价值: ⭐⭐⭐⭐⭐ 问题定义和基准贡献将推动检索领域发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[CVPR 2025\] Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval](../../CVPR2025/object_detection/search_and_detect_training-free_long_tail_object_detection_via_web-image_retriev.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2026\] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)

</div>

<!-- RELATED:END -->
