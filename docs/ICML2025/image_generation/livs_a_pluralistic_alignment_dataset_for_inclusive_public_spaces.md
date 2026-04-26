---
title: >-
  [论文解读] LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces
description: >-
  [ICML2025][图像生成][Pluralistic Alignment] 通过两年社区参与式研究，构建了包含 37,710 对多标准偏好标注的 LIVS 数据集，用于文本到图像模型在包容性城市公共空间设计中的多元对齐，并用 DPO 微调 SDXL 验证其有效性。
tags:
  - ICML2025
  - 图像生成
  - Pluralistic Alignment
  - 文生图
  - DPO
  - Intersectionality
  - 城市规划
  - 包容性设计
  - 社区参与
---

# LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces

**会议**: ICML2025  
**arXiv**: [2503.01894](https://arxiv.org/abs/2503.01894)  
**代码**: [mid-space.one](https://mid-space.one)  
**领域**: 数据集 / 公平性 / 多元对齐  
**关键词**: Pluralistic Alignment, 文生图, DPO, Intersectionality, 城市规划, 包容性设计, 社区参与

## 一句话总结

通过两年社区参与式研究，构建了包含 37,710 对多标准偏好标注的 LIVS 数据集，用于文本到图像模型在包容性城市公共空间设计中的多元对齐，并用 DPO 微调 SDXL 验证其有效性。

## 研究背景与动机

- **现有 T2I 对齐的局限**：现有对齐框架（如 Simulacra、Pic-a-Pic、ImageReward）通常假设单一的全局"好/坏"标准，主要关注美学或内容审核，忽视了多元社区在无障碍、安全、包容等方面的差异化需求。
- **交叉性（Intersectionality）被忽视**：生成模型通常为"平均用户"校准，可能系统性地排除或错误表征历史上被边缘化的群体；不同身份交叉（如残障+种族）的用户对公共空间有截然不同的评判标准。
- **本地知识缺失**：全球性众包数据难以捕捉特定社区的历史、空间和文化背景，在城市规划等领域尤为突出。
- **核心目标**：提出多元对齐范式（Pluralistic Alignment），通过社区驱动的多标准标注数据集，让 T2I 模型生成结果反映本地化、交叉性的多样化价值观。

## 方法详解

### 1. 社区参与式数据集构建

历时两年，与蒙特利尔 30 个社区组织合作，开展 11 场工作坊、34 次访谈、5 批标注：

- **概念收集**→ 634 个初始概念（涵盖照明、无障碍设施、多语言标识等）
- **标准提炼**→ 经由语义聚合、投票讨论，凝练为 **6 大核心标准**：
    - **Accessibility**（无障碍）：坡道、电梯、触觉指引
    - **Safety**（安全）：照明、视野、防护
    - **Comfort**（舒适）：座椅、遮阳、噪音
    - **Invitingness**（吸引力）：绿化、开放布局
    - **Inclusivity**（包容性）：多语标识、文化需求
    - **Diversity**（多样性）：人群多元、用途丰富

### 2. 提示词与图像生成

- 工作坊收集 440 条人类撰写提示词，另用 GPT-4o 三种策略扩展至约 2,910 条合成提示
- 对每条提示用 SDXL 生成至多 20 张图像，基于 CLIP 相似度贪心选取最具差异性的 4 张
- 最终保留 **13,462 张图像**用于标注

### 3. 多标准偏好标注

- 每次展示一对图像，随机分配 6 个标准中的 3 个
- 标注者使用滑块 $[-1, +1]$ 表示偏好方向和强度，0 表示中立
- 5 批标注共产生 **37,710 对图像级比较**，约 113,130 条标准级标注

### 4. DPO 微调

将多标准标注合并为二值偏好信号，用多数投票解决标准间冲突：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x_w, x_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(x_w)}{\pi_{\text{ref}}(x_w)} - \beta \log \frac{\pi_\theta(x_l)}{\pi_{\text{ref}}(x_l)} \right) \right]$$

其中 $x_w, x_l$ 分别为偏好/非偏好图像，$\pi_\theta$ 为微调模型，$\pi_{\text{ref}}$ 为基线 SDXL。

## 实验关键数据

### Case Study I：多标准 DPO 是否改善对齐

| 评估结果 | 数量 | 占比 |
|:--|:--|:--|
| 偏好 DPO 模型 | 700 | 32% |
| 偏好基线 SDXL | 300 | 14% |
| 中立/无偏好 | 1,100 | 50% |

- 标注量较多的标准（Comfort、Invitingness）DPO 提升更明显
- Inclusivity、Diversity 中立率较高，主观性更强

### Case Study II：身份影响偏好

- 大多数参与者轻微偏好 DPO 输出；后期加入（未参与标准制定工作坊）的参与者对两个模型无明显偏好
- 有行动障碍的标注者更倾向 DPO 输出（部分无障碍特征被捕获）

### Case Study III：提示词来源的影响

- 人类撰写提示词（Method 0）产生的 **中立标注更少**，即视觉差异更显著
- GPT-4o 生成的提示词中立率更高，可能因缺乏本地语境细节

### Case Study IV：交叉身份的评分差异

- 不同交叉身份群体（如残障×种族）在 Accessibility、Safety 等标准上给出了系统性不同的评分
- 证实单一全局对齐目标无法捕获多元偏好

## 亮点与洞察

1. **真正的社区参与**：两年、30 个组织、634 个概念→6 标准的提炼过程极具说服力，远超常见的众包标注
2. **多标准偏好学习**：首次将交叉性视角引入 T2I 对齐，6 个标准覆盖了城市公共空间的核心维度
3. **中立标注的信号价值**：约 50% 中立不代表失败，而是反映社区价值观的异质性和模糊性，为未来对齐方法设计提供重要依据
4. **人类 vs LLM 提示词对比**：揭示了人类创造力在生成差异化视觉输出中的独特作用
5. **可复用方法论**：社区参与+多标准标注+DPO 的流程可推广到文化遗产保护、医疗、教育等领域

## 局限与展望

- **地域局限**：仅覆盖蒙特利尔一个中型多元文化城市，推广到其他文化背景需验证
- **参与者规模小**：仅约 18 位核心标注者，评估集仅 2,200 条；统计显著性有限
- **DPO 的简化**：多标准反馈经多数投票坍缩为二值标签，丢失标准间的冲突和细粒度信息
- **中立标注未被利用**：约 50% 中立标注在训练中被忽略，浪费了潜在的对齐信号
- **视觉局限**：DPO 改善了步道和座椅布局，但对坡道、触觉铺装、多语标识等细节特征生成仍不稳定
- **未来方向**：Pareto 多目标优化、用户个性化对齐层、中立标注的部分奖励信号

## 相关工作与启发

- **偏好数据集**：Simulacra (238K)、Pic-a-Pic (500K+)、ImageReward (137K)、HPS/HPS v2 均为单目标全局标准
- **多维偏好**：MPS（Zhang et al., 2024b）在美学/语义/细节/整体 4 维度上训练，但无社区参与
- **多元对齐理论**：Sorensen et al. (2024) 提出 Overton 多元主义、可操控对齐、分布多元主义，LIVS 在实践层面对接了这些理论
- **DPO**：Rafailov et al. (2024) 的 DPO 框架在此被首次系统应用于城市规划领域的 T2I 对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将交叉性和社区参与引入 T2I 对齐，视角独特
- 实验充分度: ⭐⭐⭐ — 4 个 case study 结构清晰，但参与者规模和数据量偏小
- 写作质量: ⭐⭐⭐⭐ — 方法论描述详尽，社区参与过程透明
- 价值: ⭐⭐⭐⭐ — 为公平性/多元对齐提供了可复用的数据集和方法论范式

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces](diffuse_everything_multimodal_diffusion_models_on_arbitrary_state_spaces.md)
- [\[ICML 2025\] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](taming_diffusion_for_dataset_distillation_with_high_representativeness.md)
- [\[CVPR 2026\] BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment](../../CVPR2026/image_generation/biovita_biological_dataset_model_and_benchmark_for_visual-textual-acoustic_align.md)
- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](../../NeurIPS2025/image_generation/guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[ICML 2025\] Theoretical Guarantees on the Best-of-n Alignment Policy](theoretical_guarantees_on_the_best-of-n_alignment_policy.md)

<!-- RELATED:END -->
