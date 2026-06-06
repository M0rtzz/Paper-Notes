---
title: >-
  [论文解读] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models
description: >-
  [ICCV 2025][多模态VLM][联邦学习] 提出FedMVP，在联邦学习场景下通过PromptFormer网络融合图像视觉特征和LLM生成的类别属性文本特征，生成动态多模态视觉提示注入CLIP的视觉编码器，在20个数据集、三种泛化设置下显著超越现有联邦提示学习方法1.57%-2.26%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "联邦学习"
  - "CLIP提示学习"
  - "多模态提示"
  - "视觉提示调优"
  - "跨域泛化"
---

# FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2504.20860](https://arxiv.org/abs/2504.20860)  
**代码**: [https://github.com/mainaksingha01/FedMVP](https://github.com/mainaksingha01/FedMVP)  
**领域**: 多模态VLM  
**关键词**: 联邦学习, CLIP提示学习, 多模态提示, 视觉提示调优, 跨域泛化

## 一句话总结
提出FedMVP，在联邦学习场景下通过PromptFormer网络融合图像视觉特征和LLM生成的类别属性文本特征，生成动态多模态视觉提示注入CLIP的视觉编码器，在20个数据集、三种泛化设置下显著超越现有联邦提示学习方法1.57%-2.26%。

## 研究背景与动机
联邦学习（FL）允许多个客户端协作训练全局模型而不共享数据。CLIP等VLM因其强泛化能力成为FL的理想选择，但其参数量巨大导致通信开销过高。提示学习（prompt tuning）通过仅学习轻量级提示token来适配CLIP，仅需通信 ~0.37% 的参数，天然适合FL场景。

然而，现有FL提示学习方法面临严重的**泛化性退化**问题：

**文本提示学习（TPT，如PromptFL）**：学到的是静态上下文，固定后无法泛化到未见类别

**视觉提示学习（VPT，如FedVPT）**：同样因静态上下文导致泛化受限

**条件化提示方法**：FedTPG仅利用类名文本信息，FedCoCoOp仅利用图像视觉信息——在FL的高异质性场景下，单一模态的上下文信息不足

核心矛盾：FL场景的数据异质性极高（各客户端数据来自不相交的类别和领域），需要提示具备跨类别和跨域的泛化能力，但现有方法的条件化信息来源过于单一。

核心idea：**双模态条件化**——同时利用(1)输入图像的视觉特征和(2)类别的LLM文本属性描述来生成提示，通过交叉注意力融合两种模态信息。属性促进未见类别的泛化（未见类共享已见类的某些属性），图像特征促进未见域的泛化（属性无法描述的纹理/抽象概念）。

## 方法详解

### 整体框架
每个客户端在本地数据上训练PromptFormer网络（唯一可训练模块），生成的多模态视觉提示注入冻结的CLIP视觉编码器。训练后仅将轻量级PromptFormer参数发送到服务器进行FedAvg聚合。

### 关键设计
1. **LLM属性生成**:

    - 功能：使用GPT-4o为每个类别名称生成丰富的文本属性描述
    - 示例："giraffe" → "Exceptionally long neck, unique coat pattern with irregular brown patches, ..."
    - 设计动机：标签名称本身包含的语义信息有限。属性提供了细粒度的类别共性描述——"鸡"的"两条腿"属性可以迁移到未见类"海鸥"

2. **PromptFormer网络**:

    - 功能：将图像patch嵌入和文本属性嵌入通过交叉注意力融合，生成多模态视觉提示
    - 核心架构：
        - 属性嵌入提取：$\mathbf{A}_i = \{\mathcal{E}_t(\text{LLM}(c_k))\}_{k=1}^K$
        - 线性投影对齐维度：$\mathbf{A}' = T_{\text{proj}}(\mathbf{A})$（512→768）
        - 交叉注意力融合：
       $$\mathbf{P}(\mathbf{A}', \mathbf{E}) = \text{FFN}(\text{CrossAttention}(\mathbf{Q}_\mathbf{E}, \mathbf{K}_{\mathbf{A}'}, \mathbf{V}_{\mathbf{A}'}))$$
       其中 $\mathbf{Q}_\mathbf{E} = \mathbf{E}W_\mathbf{Q}$（图像patch为query），$\mathbf{K}_{\mathbf{A}'} = \mathbf{A}'W_\mathbf{K}$（属性为key/value）
        - 4头交叉注意力 + LayerNorm + 两层FFN
    - 设计动机：通过交叉注意力，图像的patch特征学习attend到相关的属性特征。例如，描绘"腿"的patch会关注"四条腿"属性——当出现共享该属性的未见类时，提示自然包含相关信息

3. **视觉提示注入**:

    - 功能：将生成的多模态提示 $\mathbf{P} \in \mathbb{R}^{m \times d_v}$（$m=4$）拼接到视觉编码器的输入
    - 输入重定义：$\mathbf{I} = [\mathbf{z}; \mathbf{E}; \mathbf{P}] \in \mathbb{R}^{(1+b+m) \times d_v}$
    - 设计动机：与FedTPG等注入文本编码器的方法不同，视觉提示调优允许更直接地影响视觉特征表示，且支持实例级别的动态提示（每张图像生成不同的提示）

4. **轻量级LoRA微调**:

    - 功能：当客户端的训练损失初始值低于阈值 $\sigma=0.5$ 时，冻结PromptFormer参数，仅训练注入的LoRA矩阵
    - 设计动机：数据量少的客户端容易过拟合，LoRA将可训练参数减少 $267\times$，同时降低通信开销

### 损失函数 / 训练策略
- **CLIP交叉熵损失**：$\mathcal{L}_{ce} = -\mathbb{E}_{(\mathbf{x},y)} y \log p(y|\mathbf{I})$
- **一致性损失**：$\mathcal{L}_{con} = 1 - \cos(\mathcal{E}_v(\mathbf{I}), \mathcal{E}_v(\mathbf{x}'))$，约束同一图像两种增强视图的表示一致性
- **总损失**：$\mathcal{L}_{total} = \mathcal{L}_{ce} + \alpha \cdot \mathcal{L}_{con}$，$\alpha = 10$
- 文本特征：$\mathbf{t}_k = \mathcal{E}_t([\text{"A photo of [CLASS]"}; \text{LLM}(c_k)])$，拼接手工模板和LLM属性
- SGD优化器，学习率0.003，衰减1e-5，batch size 128，每类8-shot

## 实验关键数据

### 主实验（Base-to-New泛化，9个数据集）

| 方法 | Local Acc | Base Acc | New Acc | HM |
|------|----------|---------|---------|-----|
| ZS-CLIP | 76.72 | 70.51 | 75.78 | 74.24 |
| PromptFL | 81.75 | 74.47 | 71.70 | 75.74 |
| FedTPG | 80.75 | 73.68 | 76.02 | 76.70 |
| FedMaPLe | 81.63 | 74.44 | 70.62 | 75.29 |
| **FedMVP (Ours)** | **81.89** | **75.37** | **77.82** | **78.27** |
| 提升 | +0.14 | +0.90 | **+1.79** | **+1.57** |

### 消融实验 / 域泛化

| 设置 (DomainBed MSST) | PACS | OfficeHome | VLCS | TerraInc | DomainNet | 平均 |
|----------------------|------|-----------|------|---------|----------|------|
| ZS-CLIP | 96.16 | 81.49 | 83.29 | 33.98 | 57.13 | 70.41 |
| FedTPG | 90.99 | 82.78 | 69.77 | 26.79 | 56.82 | 65.43 |
| FedCLIP | 96.29 | 81.74 | 82.70 | 36.58 | 57.85 | 71.03 |
| **FedMVP (Ours)** | **97.28** | **84.15** | **85.12** | **37.36** | **61.17** | **73.02** |
| 提升 | +0.99 | +1.37 | +1.83 | +0.78 | **+2.29** | **+1.99** |

| ImageNet域泛化 (SSMT) | IN | INV2 | IN-S | IN-A | IN-R | 平均 |
|---------------------|------|------|------|------|------|------|
| FedTPG | 69.51 | 62.90 | 47.65 | 49.97 | 76.35 | 59.22 |
| **FedMVP (Ours)** | **70.87** | **63.72** | **50.93** | **51.76** | **77.23** | **60.91** |
| 提升 | +1.36 | +0.82 | **+3.28** | +1.43 | +0.74 | **+1.69** |

| 组件消融 | Base-to-New HM | MSST DG Avg |
|---------|---------------|------------|
| ZS-CLIP | 74.24 | 70.41 |
| $f_\Theta$ only | 75.94 | 71.85 |
| $f_\Theta$ + $\mathcal{L}_{con}$ | 76.27 | 72.14 |
| w/o LoRA | 77.41 | 72.58 |
| **FedMVP (Full)** | **78.27** | **73.02** |

### 关键发现
- **多模态条件化是关键**：FedMVP在未见类上比FedTPG（仅文本条件化）高1.79%，比FedCoCoOp（仅视觉条件化）高11.82%
- **IN-Sketch上提升尤为显著（+3.28%）**：属性在真实图像和素描间保持不变（如"四条腿"），验证了属性特征的跨域迁移能力
- **多数提示学习方法在域泛化上不如ZS-CLIP**：只有FedCLIP和FedMVP超过了零样本基线，说明不恰当的提示学习可能导致源域过拟合
- **LoRA防过拟合显著**：不使用LoRA的FedMVP（w/o LoRA行）HM下降0.86%
- **FedMVP收敛比FedTPG快约10倍**（通信轮次计），虽然每轮传输参数量多2倍但总通信成本更低
- **跨数据集泛化最具挑战**：FedMVP在OxfordPets和StanfordCars上低于ZS-CLIP，可能因为细粒度类别间属性重叠

## 亮点与洞察
1. **首次在FL中引入LLM生成的属性信息**：属性作为类别间的共享知识桥梁，有效促进跨类别泛化
2. **交叉注意力的直觉清晰**：视觉patch作为query去检索相关属性，使得提示在面对新类别时能自动关注对应的共享属性
3. **视觉提示而非文本提示**：与主流方法相反，FedMVP选择在视觉编码器端注入提示，支持实例级别的动态化
4. **LoRA自适应策略**：根据客户端数据量自动切换全参数/LoRA训练，兼顾性能和防过拟合——这是对FL异质性的精细化适配

## 局限与展望
1. 依赖GPT-4o生成属性，增加了部署环节的成本和对外部API的依赖
2. 在细粒度数据集（OxfordPets、StanfordCars）上表现不如ZS-CLIP，属性重叠可能是根因——需要更细粒度的属性设计
3. 仅使用ViT-B/16骨架，未验证更大模型（如ViT-L/14）的表现
4. LoRA的切换阈值 $\sigma=0.5$ 是手动设置的，自适应阈值可能更好
5. 客户端数量增多时的可扩展性和通信效率仍需进一步分析

## 相关工作与启发
- **与FedTPG的关系**：FedTPG仅用类名作文本条件化生成文本提示，FedMVP增加了图像视觉条件和LLM属性条件，且注入视觉编码器而非文本编码器
- **与MaPLe的关系**：MaPLe同时注入文本和视觉编码器，但需要解锁编码器部分权重，不适合FL场景
- **与LaBo/VFC的关系**：这些方法用LLM生成的描述增强CLIP，但不是在FL场景下，FedMVP是首次将此策略引入联邦学习
- **启发**：在高异质性分布式场景中，**多模态条件化信息比任何单一模态都更有效**——这启示未来的FL方法应尽可能利用所有可用的模态信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态条件化+视觉提示调优的组合新颖，PromptFormer设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 20个数据集、三种泛化设置、多个FL基线、详尽消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但符号系统稍复杂
- 价值: ⭐⭐⭐⭐ 为联邦VLM适配提供了实用方案，跨域泛化改进显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)

</div>

<!-- RELATED:END -->
