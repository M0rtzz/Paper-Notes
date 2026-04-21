---
title: >-
  [论文解读] DreamRelation: Bridging Customization and Relation Generation
description: >-
  [CVPR 2025][图像生成][关系感知定制生成] DreamRelation 提出了一种关系感知的定制化图像生成框架，通过精心构建的解耦数据引擎、关键点匹配损失（KML）和局部 token 注入三大设计，在保持多目标身份一致性的同时准确生成文本指定的目标间关系（如拥抱、骑行等），在 RelationBench 上全面超越现有方法。
tags:
  - CVPR 2025
  - 图像生成
  - 关系感知定制生成
  - 多目标定制
  - 关键点匹配损失
  - 局部特征注入
  - LoRA微调
---

# DreamRelation: Bridging Customization and Relation Generation

**会议**: CVPR 2025  
**arXiv**: [2410.23280](https://arxiv.org/abs/2410.23280)  
**代码**: https://github.com/shi-qingyu/DreamRelation (有)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 关系感知定制生成, 多目标定制, 关键点匹配损失, 局部特征注入, LoRA微调

## 一句话总结
DreamRelation 提出了一种关系感知的定制化图像生成框架，通过精心构建的解耦数据引擎、关键点匹配损失（KML）和局部 token 注入三大设计，在保持多目标身份一致性的同时准确生成文本指定的目标间关系（如拥抱、骑行等），在 RelationBench 上全面超越现有方法。

## 研究背景与动机

1. **领域现状**：定制化图像生成（Customized Image Generation）已取得显著进展，基于微调（DreamBooth、Custom Diffusion）和基于训练（MS-Diffusion、SSR-Encoder）的方法都能较好地保持用户提供的目标身份。
2. **现有痛点**：现有多目标定制方法忽略了目标之间的关系——当用户给出两个目标图像和一个描述关系的文本（如"拥抱"），生成结果往往无法正确表达这种关系，要么关系缺失，要么目标混淆。
3. **核心矛盾**：（a）缺乏合适的训练数据来解耦身份信息和关系信息；（b）模型设计上，图像 prompt 的控制力过强导致文本中的关系描述被忽视；（c）CLIP 提取的全局特征缺乏局部细节（如"手"的特征），导致重叠场景下目标混淆。
4. **本文目标** 在保持多目标身份的同时，准确生成文本描述的目标间关系——即"关系感知的定制化图像生成"。
5. **切入角度**：作者观察到关系主要体现在目标的姿态变化上，而现有方法裁剪目标作为 image prompt 会带来 copy-paste 效应（姿态不变，关系无法学习），因此需要从数据和模型两个层面同时解决。
6. **核心 idea**：通过关系感知数据引擎解耦身份与关系，再用关键点匹配损失显式引导姿态调整、局部 token 注入避免目标混淆，实现关系感知的定制化生成。

## 方法详解

### 整体框架
DreamRelation 基于 MS-Diffusion 构建。输入包括两张目标图像 $c_i$（image prompt）和一条描述关系的文本 $c_t$（text prompt），输出要求同时保持两个目标的身份并准确反映文本中的关系。整个方法分为三个核心阶段：（1）关系感知数据引擎构建训练三元组 $(x_k, c_i, c_t)$；（2）在 U-Net 的文本交叉注意力层注入 LoRA 进行关系学习；（3）引入 KML 和局部 token 注入增强关系生成和防止目标混淆。

### 关键设计

1. **关系感知数据引擎（Relation-Aware Data Engine）**:

    - 功能：构建用于关系学习的高质量训练三元组数据。
    - 核心思路：利用 DALL-E 3 的多轮对话能力，生成包含相同目标但姿态不同的图像三元组——关系目标图像 $x_k$、独立的身份图像 $c_i$、以及关系描述文本 $c_t$。通过 prompt "The photo of the same" 让 DALL-E 3 记住并保持目标身份。然后使用 X-Pose 标注关键点、SAM 标注 mask、LLaVA 生成 caption。
    - 设计动机：直接从 $x_k$ 裁剪目标作为 $c_i$ 会导致 copy-paste 效应——裁剪图像的姿态与目标图像一致，模型无法学到关系带来的姿态变化。通过生成姿态不同但身份相同的 $c_i$，迫使模型从 $c_t$ 中学习关系信息，实现身份与关系的解耦。

2. **关键点匹配损失（Keypoint Matching Loss, KML）**:

    - 功能：在隐空间中显式监督目标的姿态，引导模型生成与关系匹配的正确姿态。
    - 核心思路：使用 X-Pose 检测 $x_k$ 和 $c_i$ 中每个目标的 17 个关键点。在训练时，通过 U-Net 输出 $\hat{\epsilon}$ 预测 $\hat{z}_0$，然后在 VAE 隐空间中计算 $\hat{z}_0$ 对应关键点位置与 $\mathcal{E}(c_i)$ 对应关键点位置之间的 MSE 损失：$\mathcal{L}_{KML} = \frac{1}{n_{kp}} \mathbb{E} \| \mathcal{E}(c_i)[c_{kp}^{c_i}] - \hat{z}_0[c_{kp}^{x_k}] \|_2^2$。总损失为 $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \cdot \mathcal{L}_{KML}$，$\lambda = 1e{-3}$。
    - 设计动机：关系与姿态密切相关（如"拥抱"需要手臂交叉、"骑车"需要脚在踏板上）。仅靠 diffusion loss 难以精确控制姿态，而 KML 直接在隐空间关键点上施加约束，使模型学会为不同关系调整目标姿态。

3. **局部 Token 注入（Local Token Injection）**:

    - 功能：从 image prompt 中提取局部细粒度特征，防止重叠场景中的目标混淆。
    - 核心思路：修改 CLIP Image Encoder 最后一层获取 dense feature $h_{dense}$，通过自蒸馏方法使局部特征与全局特征对齐。推理时对 dense feature 进行分区池化得到 local tokens $tok_{local}$，与 image-level tokens $tok_{image}$ 拼接后送入 ID extractor：$q = q + \text{Attention}(\text{concat}[q, tok_{image}, tok_{local}])$。
    - 设计动机：CLIP 的全局图像特征过于粗糙，缺乏局部细节（如手部特征），导致在生成"握手"等关系时无法区分两个目标。引入 dense local feature 提供精细的局部信息，有效避免目标混淆。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{denoise} + \lambda \cdot \mathcal{L}_{KML}$
- 仅在 U-Net 的文本交叉注意力层的 $W_q, W_{k_t}, W_{v_t}, W_{out}$ 注入 LoRA（rank=4），冻结其余参数
- 微调 500 步，2 张 A100，batch size=8，耗时 10 分钟
- 可训练参数仅 3.1M，兼容任何基于 SDXL 的模型
- 图像交叉注意力的缩放因子 $\gamma = 0.6$

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | DreamRelation | MS-Diffusion | ReVersion+MS | 提升 |
|------------|------|--------------|--------------|--------------|------|
| RelationBench 单目标 | CLIP-T | **30.6** | 26.5 | 27.8 | +2.8 |
| RelationBench 单目标 | CLIP-R | **21.4** | 18.8 | 19.3 | +2.1 |
| RelationBench 多目标 | CLIP-T | **28.9** | 26.9 | 27.2 | +1.7 |
| RelationBench 多目标 | DINO | **62.1** | 58.8 | 59.7 | +2.4 |

### 消融实验

| 配置 | CLIP-T | CLIP-R | CLIP-I | DINO | 说明 |
|------|--------|--------|--------|------|------|
| Full Model | **28.9** | **20.4** | **75.4** | **62.1** | 完整模型 |
| w/o Relation-aware Data | 27.3 | 19.4 | 75.3 | 59.8 | 使用裁剪数据引擎，CLIP-T 下降 1.6 |
| w/o Local Token Injection | 28.5 | 19.5 | 75.1 | 59.9 | 去掉局部 token，DINO 下降 2.2 |
| w/o Keypoint Matching Loss | 27.4 | 19.2 | 75.2 | 61.2 | 去掉 KML，CLIP-T 下降 1.5 |

### 关键发现
- **数据引擎贡献最大**：去掉关系感知数据引擎后 CLIP-T 和 CLIP-R 均显著下降，说明解耦身份与关系的数据构建是成功的关键。
- **KML 对关系生成至关重要**：去掉 KML 后 CLIP-R 降幅最大（-1.2），证明关键点级别的姿态监督对关系准确性有直接影响。
- **局部 token 防止目标混淆**：去掉局部 token 后 DINO 从 62.1 降至 59.9，说明局部特征对保持身份一致性在重叠场景下尤其重要。
- 方法可直接迁移到 SDXL 基模型，从 text-to-image 的关系生成也有效。

## 亮点与洞察
- **数据解耦思路极其巧妙**：利用 DALL-E 3 的多轮对话能力生成同身份不同姿态的图像对，自然解耦了身份和关系——这种数据构建思路可推广到任何需要解耦多种属性的生成任务。
- **隐空间关键点约束**：KML 不在像素空间而在 VAE 隐空间操作，与 diffusion loss 空间一致，避免了梯度不匹配问题——这种在隐空间引入结构化约束的思路可迁移到姿态控制、人体生成等任务。
- **仅微调 3.1M 参数**：通过只在文本交叉注意力层加 LoRA，实现了极其轻量的微调且兼容所有 SDXL 模型——训练仅 10 分钟。

## 局限与展望
- **目标类别受限**：数据引擎依赖 DALL-E 3 生成常见类别（如动物、玩偶），对罕见物体的身份保持能力未验证。
- **关系类型有限**：实验中的 25 种关系主要是空间和交互关系，对更抽象的关系（如"保护"、"追赶"）的泛化能力未知。
- **两目标限制**：虽然展示了三目标的定性结果，但定量评估仅限两目标，多目标扩展的稳定性待验证。
- **评估指标局限**：CLIP-R 只提取关系词计算相似度，对复杂关系的评估可能不够准确。

## 相关工作与启发
- **vs MS-Diffusion**: MS-Diffusion 用 bounding box 引导多目标生成，但无法处理 box 重叠时的目标混淆，也不考虑关系。DreamRelation 在此基础上增加关系学习能力，通过 KML 和局部 token 解决了重叠混淆。
- **vs ReVersion**: ReVersion 通过文本反转学习关系嵌入，但依赖共现图像且无法定制目标身份。DreamRelation 通过数据解耦同时支持身份保持和关系生成。
- **vs ADI**: ADI 只处理关系生成不支持定制，DreamRelation 是首个同时支持单/多目标定制 + 关系生成的方法。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义"关系感知定制生成"任务并提出完整解决方案，数据引擎解耦思路新颖
- 实验充分度: ⭐⭐⭐⭐ 提出 RelationBench 评估体系，消融充分，但缺少用户研究和更多 baseline 对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机阐述逻辑性强，图示质量高
- 价值: ⭐⭐⭐⭐ 填补了定制生成与关系生成之间的空白，对多目标交互场景生成有实用价值

<!-- RELATED:START -->

## 相关论文

- [AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [LoRACLR: Contrastive Adaptation for Customization of Diffusion Models](loraclr_contrastive_adaptation_for_customization_of_diffusion_models.md)
- [DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)
- [Visual Persona: Foundation Model for Full-Body Human Customization](visual_persona_foundation_model_for_full-body_human_customization.md)

<!-- RELATED:END -->
