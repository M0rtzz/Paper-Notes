---
title: >-
  [论文解读] VisualLens: Personalization through Task-Agnostic Visual History
description: >-
  [NeurIPS 2025][多模态推荐] 提出VisualLens框架，利用用户日常拍摄的与任务无关的视觉历史(task-agnostic visual history)，通过频谱用户画像(spectrum user profile)和多模态大模型实现跨领域个性化推荐，在新建的Google Review-V和Yelp-V数据集上Hit@3超越GPT-4o 2-5%。
tags:
  - NeurIPS 2025
  - 多模态推荐
  - 视觉历史
  - 个性化
  - MLLM
  - 用户画像
---

# VisualLens: Personalization through Task-Agnostic Visual History

**会议**: NeurIPS 2025  
**arXiv**: [2411.16034](https://arxiv.org/abs/2411.16034)  
**代码**: 无  
**领域**: recommender  
**关键词**: 多模态推荐, 视觉历史, 个性化, MLLM, 用户画像

## 一句话总结
提出VisualLens框架，利用用户日常拍摄的与任务无关的视觉历史(task-agnostic visual history)，通过频谱用户画像(spectrum user profile)和多模态大模型实现跨领域个性化推荐，在新建的Google Review-V和Yelp-V数据集上Hit@3超越GPT-4o 2-5%。

## 研究背景与动机
**领域现状**: 现有推荐系统主要依赖用户的物品交互历史(如购买记录、点击日志)或文本信号进行推荐。多模态推荐虽有进展(如UniMP)，但仍局限于特定领域的物品级历史，无法实现跨领域的通用个性化推荐。

**现有痛点**: 物品级交互历史不总是可获取的(冷启动问题)，且不具备跨任务泛化能力——电商平台知道你的购物偏好，却对你的餐厅喜好一无所知。现有多模态推荐方法严重依赖结构化的物品特征(item features)，无法处理非结构化的视觉信号。

**核心矛盾**: 用户日常拍摄的照片蕴含丰富的偏好信息(如对建筑风格、食物类型的偏好)，但这些照片与推荐任务之间存在巨大的语义鸿沟——照片可能与当前查询完全无关，且噪声大、信息密度低。

**本文目标**: (1) 如何从与任务无关的日常照片中提取用户偏好？(2) 如何在有限的模型上下文窗口内处理大量视觉历史？(3) 缺乏评测此类系统的数据集。

**切入角度**: 作者受Vannevar Bush的Memex概念启发，假设用户的视觉历史(日常照片)蕴含跨领域的偏好信号，可用于通用个性化推荐。关键洞察是将每张图片表示为"频谱三元组"(原始图像→字幕→方面词)，在信息丰富度和语义清晰度之间取得平衡。

**核心 idea**: 用频谱用户画像将日常照片压缩为(图像, 字幕, 方面词)三元组，通过图像网格化和联合训练实现基于视觉历史的跨领域推荐。

## 方法详解

### 整体框架
VisualLens包含离线用户画像生成和在线推荐两大阶段。离线阶段：对用户视觉历史中的每张图片生成CLIP编码、字幕和方面词，构建频谱用户画像。在线阶段：给定推荐查询q，(1) 通过CLIP相似度检索与q相关的图片，(2) 将检索到的图片组成d×d网格图并生成查询相关用户画像，(3) 将画像与候选项匹配打分排序。整个框架是模块化的，各组件可独立替换为更强的模型。

### 关键设计
1. **频谱用户画像 (Spectrum User Profile)**:

    - 功能：将每张图片表示为三元组 (raw image, caption, aspect words)，构成从信息丰富到语义清晰的频谱
    - 核心思路：用CLIP ViT-L/14@336px编码图像向量，用LLaVA-v1.6-8B生成≤30词的简洁字幕，再提取关键方面词(如dome, balcony)
    - 设计动机：原始图像信息丰富但噪声大，纯文本描述信息损失多；三元组在两者间取得平衡，消融实验证明三种表示均有贡献
    - 所有模块均为即插即用设计，可替换为更强的编码器或生成模型

2. **图像网格化 (Image Gridification)**:

    - 功能：将检索到的w张图片排列为d×d网格(d²=w)，作为单张图片输入多模态模型
    - 核心思路：选取w=64张图片构成8×8网格，每张子图标注编号(1~64)以关联对应的字幕和方面词；不足64张时用黑色背景填充
    - 设计动机：直接输入64张图片会远超MLLM上下文窗口限制(如PaliGemma对896×896图片生成4096 token)，网格化将多图问题转化为单图理解问题
    - 配合多图字幕预训练(DOCCI数据集)确保模型能准确识别网格中每个子图

3. **历史检索 (History Retrieval)**:

    - 功能：从用户视觉历史中检索与当前查询最相关的图片
    - 核心思路：对每个推荐类别c，随机采样n=10K个候选项，平均其CLIP视觉编码得到类别向量v_c，然后计算用户历史图片与类别向量的余弦相似度，选top-w张
    - 设计动机：用户视觉历史多样且嘈杂，大量图片与当前查询无关；检索使模型聚焦于相关视觉信号，消融实验显示这是贡献最大的组件

4. **迭代方面词精炼 (Iterative Aspect Word Refinement)**:

    - 功能：通过多轮迭代提升方面词与用户偏好的对齐度
    - 核心思路：初始方面词由LLaVA-v1.6生成(𝒲⁰)，每轮用Llama-3.1-70B结合ground truth筛选有用方面词(𝒲ʲ)，约4轮收敛得到训练目标𝒲
    - 设计动机：零样本生成的方面词质量不稳定，可能包含无关词(如"blue sky")，迭代精炼使方面词更贴合推荐任务需求

### 损失函数 / 训练策略
- **多图字幕预训练**: 在DOCCI数据集(15,000+图片)上LoRA微调，让模型学会对网格中每张子图分别描述，为后续网格化推荐奠定基础
- **联合训练**: 方面词生成损失(CE) + 候选匹配损失(BCE)联合优化：ℒ_joint = ℒ_asp + λ·ℒ_pred，其中λ=2
- **LoRA微调**: 在预训练backbone(MiniCPM-V2.5/PaliGemma)上进行参数高效微调

## 实验关键数据

### 主实验 (Google Review-V / Yelp-V)

| 方法 | 模态 | 大小 | GR-V Hit@1 | GR-V Hit@3 | GR-V Hit@10 | GR-V MRR | Yelp Hit@1 | Yelp Hit@3 | Yelp Hit@10 | Yelp MRR |
|------|------|------|------------|------------|-------------|----------|------------|------------|-------------|----------|
| Random | - | - | 7.6 | 21.0 | 55.0 | 21.2 | 13.0 | 33.6 | 72.7 | 30.0 |
| UniMP | T+I | 3B | 13.8 | 34.1 | 73.0 | 30.5 | 22.4 | 48.5 | 85.0 | 38.3 |
| GPT-4o | T+I | - | 17.1 | 37.3 | 80.1 | 34.3 | 26.1 | 54.5 | 90.5 | 41.7 |
| **VisualLens (8B)** | T+I | 8B | **18.5** | **38.9** | **82.3** | **35.4** | **28.3** | **59.1** | **91.0** | **44.9** |
| Human | - | - | 22.0 | 45.0 | - | - | 36.0 | 66.0 | - | - |

### 消融实验 (PaliGemma 3B backbone)

| 配置 | GR-V Hit@3 | GR-V MRR | Yelp Hit@3 | Yelp MRR | 说明 |
|------|------------|----------|------------|----------|------|
| 完整VisualLens | 36.3 | 33.5 | 58.8 | 44.3 | 全部组件 |
| 去掉联合训练 | 35.8 | 33.0 | 57.9 | 43.3 | Hit@3 -0.5/-0.9 |
| 去掉迭代精炼+联合训练 | 35.2 | 32.5 | 57.5 | 42.9 | Hit@3 -1.1/-1.3 |
| 去掉字幕 | 34.7 | 31.9 | 55.3 | 41.2 | 字幕贡献显著 |
| 去掉方面词 | 33.9 | 31.2 | 53.9 | 40.4 | 方面词贡献最大 |
| 仅保留图像 | 32.5 | 29.6 | 48.2 | 38.8 | 文本增强非常关键 |
| 去掉检索(全部图片) | 27.9 | 25.9 | 45.7 | 36.8 | 检索模块贡献最大(-7%/-12%) |

### 关键发现
- VisualLens (8B) 在Hit@3上超越GPT-4o 1.6%(GR-V)和4.6%(Yelp-V)，填补了与人类标注~75%的差距
- 历史检索是最重要的组件，去掉后Hit@3下降7-12%；方面词的贡献大于字幕
- MRR在用户历史图片数达到~100张后趋于饱和，在候选数超过50后趋于稳定
- 跨历史长度和跨类别的迁移性良好，长历史测试MRR最高(GR-V: 38.0 vs 35.4)
- 模糊类别(如"area""station")推荐效果最差；通用类别(如"museum""hotel")效果最好
- 相邻类别间存在迁移学习效应，如"deli"和"takeout"受益于与"restaurant"类的相似性

## 亮点与洞察
- **问题定义新颖**: 首次系统研究"任务无关的视觉历史→跨领域个性化"这一问题，创建了两个专用benchmark(GR-V: 15.69M训练样本, Yelp-V: 4.12M)，填补了该方向的评测空白。
- **频谱表示设计巧妙**: (图像, 字幕, 方面词)三元组在信息量和清晰度之间形成频谱，图像网格化巧妙解决了MLLM上下文窗口限制，使8B模型可同时处理64张图片。

## 局限与展望
- 各模块(编码器、字幕生成等)未使用最优模型，框架注重模块化而非极致性能
- 仅处理静态图片，未涵盖视频、音频等更丰富的模态
- 评测仅限QA格式的推荐任务，未覆盖排序列表、对话式推荐等场景
- 隐私问题：从日常照片推断用户偏好在实际部署中需要严格的隐私保护机制

## 相关工作与启发
- **vs UniMP (Wei et al., 2024a)**: UniMP是当前SOTA多模态推荐，但依赖物品级交互历史；VisualLens使用任务无关的视觉历史，在相同参数量下超越UniMP ~5-10% Hit@3
- **vs GPT-4o**: 直推GPT-4o在视觉推荐上表现强劲，但VisualLens 8B模型通过专门的频谱画像和联合训练仍可超越1.6-4.6% Hit@3
- **vs ReLLa (Lin et al., 2024)**: ReLLa通过检索增强文本推荐；VisualLens将检索思想扩展到视觉模态，检索模块贡献最大

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出任务无关视觉历史驱动的推荐框架，问题定义和数据集都是全新的
- 实验充分度: ⭐⭐⭐⭐ 两个大规模数据集、多个baseline、详细消融和迁移性分析
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，图表丰富，但部分公式符号可更简洁
- 价值: ⭐⭐⭐⭐ 开辟了视觉历史个性化的新方向，但实际部署的隐私挑战需进一步解决

<!-- RELATED:START -->

## 相关论文

- [MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)
- [Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](../../ACL2025/recommender/beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)
- [QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval](../../ICML2025/recommender/qure_query-relevant_retrieval_through_hard_negative_sampling_in_composed_image_r.md)
- [ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [EMPATHIA: Multi-Faceted Human-AI Collaboration for Refugee Integration](empathia_multi-faceted_human-ai_collaboration_for_refugee_integration.md)

<!-- RELATED:END -->
