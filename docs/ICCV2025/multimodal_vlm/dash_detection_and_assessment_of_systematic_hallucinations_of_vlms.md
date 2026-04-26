---
title: >-
  [论文解读] DASH: Detection and Assessment of Systematic Hallucinations of VLMs
description: >-
  [ICCV 2025][多模态][视觉语言模型] 提出DASH自动化流水线，通过LLM生成文本查询（DASH-LLM）和扩散模型优化图像查询（DASH-OPT）两种策略，在ReLAION-5B中系统性地发现VLM的假阳性对象幻觉聚类，共发现19k+聚类和950k+图像，并构建了更具挑战性的DASH-B基准。
tags:
  - ICCV 2025
  - 多模态
  - 视觉语言模型
  - 对象幻觉
  - 系统性错误检测
  - 扩散模型优化
  - 大规模基准
---

# DASH: Detection and Assessment of Systematic Hallucinations of VLMs

**会议**: ICCV 2025  
**arXiv**: [2503.23573](https://arxiv.org/abs/2503.23573)  
**代码**: [https://YanNeu.github.io/DASH](https://YanNeu.github.io/DASH)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 对象幻觉, 系统性错误检测, 扩散模型优化, 大规模基准

## 一句话总结
提出DASH自动化流水线，通过LLM生成文本查询（DASH-LLM）和扩散模型优化图像查询（DASH-OPT）两种策略，在ReLAION-5B中系统性地发现VLM的假阳性对象幻觉聚类，共发现19k+聚类和950k+图像，并构建了更具挑战性的DASH-B基准。

## 研究背景与动机
VLM（如PaliGemma、LLaVA-NeXT）在多模态任务上表现出色，但会产生**对象幻觉**——将图中不存在的物体错误地识别为存在。现有基准（POPE、AMBER）存在两个根本问题：

**数据集过于局限**：依赖MSCOCO等小规模标注数据集（仅80个对象类），无法反映VLM在开放世界中的真实应用场景。POPE基准在当前模型上的TNR已达96%，接近饱和。

**缺乏系统性评估**：无法区分幻觉是随机偶发还是模型的系统性缺陷。如果某VLM在特定类型的图像上持续产生幻觉，这意味着存在根本性缺陷需要修复。

核心矛盾：VLM的应用场景是开放世界的，但评测基准却局限于封闭数据集。穷举测试ReLAION-5B中所有图片-物体组合是不现实的，需要一种**有效的搜索策略**来发现系统性幻觉。

核心idea：构建全自动流水线DASH，通过生成针对性的查询（文本/图像），在大规模数据集中检索能触发VLM幻觉的真实图像，并通过聚类找到语义相似的系统性错误模式。

## 方法详解

### 整体框架
DASH包含四个阶段：查询生成（DASH-LLM或DASH-OPT）→ 探索（Exploration，kNN检索）→ 利用（Exploitation，扩展检索）→ 聚类（Clustering）。整个流水线无需人工标注，完全自动化。

### 关键设计
1. **DASH-LLM（基于文本的查询生成）**:

    - 功能：利用LLM（Llama 3.1-70B）为每个对象类生成50条文本查询
    - 核心思路：要求LLM生成可能导致VLM误识别的场景描述（如"消防船"容易让模型幻觉出"水炮"），但不能在描述中提及目标对象本身
    - 设计动机：FP-幻觉通常源于对象间的共现关联（如圣诞装饰→Baumkuchen蛋糕），LLM训练于大规模文本语料，能有效捕捉这些关联
    - 局限：与特定VLM无关（不针对模型特异性错误），且文本查询的CLIP检索可能找不到合适图像

2. **DASH-OPT（基于优化的图像查询生成）**:

    - 功能：通过优化扩散模型的输入变量，生成能同时骗过VLM但不包含目标对象的图像
    - 核心思路：使用单步扩散模型（蒸馏后的LDM），联合优化两个目标：
        - VLM损失：$L_{\text{vlm}}(C) = -\log p_{\text{vlm}}(\text{"Yes"} \mid q(C), \text{qstnOBJ})$，最大化VLM回答"Yes"的概率
        - 检测器损失：$L_{\text{det}}(C) = -\log(1 - p_{\text{det}}(\text{OBJ} \mid q(C)))$，最小化开放世界检测器（OWLv2）的目标对象置信度
        - 总目标：$\min_C L_{\text{vlm}}(C) + L_{\text{det}}(C)$
    - 设计动机：直接在像素空间优化会产生对抗样本而非自然图像，在扩散模型的潜空间中优化可保证生成的图像落在"自然图像流形"上。与DASH-LLM不同，DASH-OPT是模型特异性的，能发现更多意外的幻觉模式

3. **探索-利用-聚类流程**:

    - 探索阶段：对每个查询在ReLAION-5B上做kNN检索（CLIP相似度），获取候选图像，过滤掉检测器认为包含目标对象的图像和未触发VLM幻觉的图像
    - 利用阶段：对探索阶段的成功图像做二次kNN检索（每张50个近邻），验证幻觉是否能转移到语义相似的图像上，使用DreamSim去除近重复项
    - 聚类阶段：使用CLIP嵌入空间中的层次聚类（平均链接）合并相似的预聚类

### 损失函数 / 训练策略
- OWLv2检测器使用极低阈值（保守策略），确保标注的"不含目标对象"尽可能可靠
- 人工验证显示：DASH仅5.2%的图像实际包含目标对象（vs POPE的25.5%误标率）
- 微调策略：每个对象采样200张DASH图像（训练回答"No"）+ 400张正样本（训练回答"Yes"）

## 实验关键数据

### 主实验

| 模型 | 方法 | 总图像数 | 总聚类数 | 每对象平均聚类 | 每聚类平均图像 |
|------|------|---------|---------|--------------|--------------|
| PaliGemma | DASH-LLM | 99.3K | 1892 | 5.0 | 52.5 |
| PaliGemma | DASH-OPT | **221.7K** | **3895** | **10.3** | 56.9 |
| LLaVA-NeXT Vicuna | DASH-LLM | 162.4K | 3632 | 9.6 | 44.7 |
| LLaVA-NeXT Vicuna | DASH-OPT | **252.0K** | **4632** | **12.2** | 54.4 |
| LLaVA-NeXT Mistral | DASH-OPT | 133.8K | 3229 | 8.5 | 41.5 |

### 消融实验 / 迁移与基准

| 配置 | DASH-B Acc. | DASH-B TNR | POPE TNR | 说明 |
|------|-----------|-----------|---------|------|
| PaliGemma2-3B | 68.9% | 40.9% | 97.3% | DASH-B远比POPE困难 |
| Ovis2-8B | 71.4% | 44.8% | 94.9% | 幻觉问题仍严重 |
| LLaVa-OneVision | 75.1% | 60.1% | 95.8% | — |
| GPT-4o-mini | **86.3%** | **76.7%** | — | 最强模型仍有23%假阳性 |
| PaliGemma (微调前) | 56.4% | — | 87.2% | — |
| PaliGemma (微调后) | **68.0%** (+11.6%) | — | 86.4% | DASH数据微调有效 |

### 关键发现
- **DASH-OPT远优于DASH-LLM**：发现更多聚类（约2倍）和更多样化的幻觉模式（如"豹纹"→leopard、"卡通青蛙"→dam）
- **幻觉可跨模型迁移**：PaliGemma发现的幻觉图像在LLaVA-NeXT Vicuna上有43-49%的迁移率
- **LLM骨架显著影响幻觉率**：Vicuna > Mistral > Llama，且这与模型整体的"Yes"倾向正相关
- **视觉编码器影响更大**：CLIP (60%迁移率) > SigLIP (43%)
- **模型规模影响较小**：Qwen2-VL 7B (19%) vs 72B (18%)，但72B在相同TPR下幻觉更少
- **POPE基准已饱和**：当前VLM的TNR达96%，而DASH-B的TNR仅48.6%，揭示了大量被忽视的幻觉

## 亮点与洞察
1. **开放世界视角**：首次在web-scale数据集（ReLAION-5B, 50亿图像）上系统评估VLM幻觉，发现现有基准严重低估了问题的严重性
2. **全自动无标注流水线**：使用检测器替代人工验证，误差率（5.2%）远低于POPE的标签噪声（25.5%）
3. **DASH-OPT的巧妙设计**：利用扩散蒸馏模型实现单步生成，计算高效；在潜空间优化保证生成自然图像而非对抗样本
4. **实用价值**：发现的系统性幻觉模式可直接用于微调改进模型（DASH-B精度提升11.6%）

## 局限与展望
1. ReLAION-5B虽然庞大但仍无法覆盖所有自然图像分布，某些场景可能缺乏足够的语义邻居来构成聚类
2. 保守的检测器阈值可能导致对最先进VLM的评估出现偏差——当VLM的能力接近检测器时
3. 目前的微调策略是独立任务式的，未与VLM的正常训练流程整合（如curriculum learning）
4. 仅关注False Positive幻觉（错误说"是"），未系统研究False Negative（错误说"否"）

## 相关工作与启发
- **与Spurious ImageNet的关系**：后者研究分类器中的伪相关特征，DASH将此思路扩展到VLM的开放世界场景
- **与DiG-IN的关系**：DiG-IN也用优化引导的图像生成来调试模型，但使用多步扩散过程且仅找分类器差异
- **启发**：系统性幻觉的发现暗示VLM的训练数据中存在强共现偏差——未来或可在预训练阶段就通过数据去偏来根本性解决

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新的系统性幻觉检测范式，DASH-OPT设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 380个对象类、多个VLM、人工验证、迁移分析、微调实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化优秀，但部分实验细节需参考附录
- 价值: ⭐⭐⭐⭐⭐ 揭示了VLM幻觉远比想象中严重，DASH-B有望成为新标准基准

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/multimodal_vlm/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[NeurIPS 2025\] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](../../NeurIPS2025/multimodal_vlm/ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)
- [\[ICCV 2025\] CLIPSym: Delving into Symmetry Detection with CLIP](clipsym_delving_into_symmetry_detection_with_clip.md)

<!-- RELATED:END -->
