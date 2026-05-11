---
title: >-
  [论文解读] Attention Prompting on Image for Large Vision-Language Models
description: >-
  [ECCV 2024][多模态][提示学习] 提出Attention Prompting on Image（API），通过辅助VLM（CLIP或LLaVA）根据文本查询生成注意力归因图，将其作为热力图叠加到原始图像上引导LVLM聚焦相关区域，在MM-Vet上提升LLaVA-1.5达3.8%，跨多种LVLM（包括GPT-4V）通用有效。
tags:
  - ECCV 2024
  - 多模态
  - 提示学习
  - 注意力机制
  - LVLM
  - self-reflection
  - model ensemble
---

# Attention Prompting on Image for Large Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2409.17143](https://arxiv.org/abs/2409.17143)  
**代码**: [GitHub](https://github.com/yu-rp/apiprompting)  
**领域**: 多模态VLM  
**关键词**: visual prompting, attention heatmap, LVLM, CLIP, self-reflection

## 一句话总结

提出Attention Prompting on Image（API），用辅助VLM（如CLIP或LLaVA）根据文本查询生成注意力归因热力图，将其叠加到原始图像上作为视觉提示，在无需训练的情况下提升LVLM在多个VL基准上的表现（LLaVA-1.5 在MM-Vet上+3.8%）。

## 研究背景与动机

**领域现状**：LVLM在多种视觉语言任务上表现出色，但在复杂场景中仍难以准确聚焦于回答问题所需的关键区域。视觉提示（如画圈、标记）已被证明可以改善LVLM的感知能力，但现有方法存在明显缺陷。

**现有痛点**：(1) 已有视觉提示方法（FGVP、SoM）基于分割模型生成提示，与文本查询无关——无论问什么同一张图的提示结果不变；(2) 依赖分割mask的提示本质上是instance-level proposals，偏向grounding任务，不适用于通用VQA；(3) 粗暴的mask/blur操作可能破坏图像全局信息。

**核心矛盾**：如何生成与文本查询相关的视觉提示，既能引导LVLM关注正确区域，又不损害其全局理解能力？

**切入角度**：利用辅助VLM的注意力机制从文本查询中提取"应关注哪些图像区域"的信号，将其以柔和的alpha blending热力图方式叠加到原图上。

## 方法详解

### 整体框架

给定输入图像$I$和文本查询$T^i$，API分两步：(1) 用辅助模型$g$（CLIP或LLaVA）生成文本-图像归因图$\Psi \in \mathbb{R}^{P \times P}$；(2) 将归因图转换为像素空间热力图$\Phi$，通过alpha blending叠加到原图得到$I^a$，再送入目标LVLM $f$生成回答。当$g=f$时为自反射模式，$g \neq f$时为模型集成模式。

### 关键设计

1. **从CLIP获取归因图**

    - **直接归因$\Psi^{cls}$**：将CLIP视觉编码器中MSA输出的cls token按残差连接分解，得到每个patch对最终image-level相似度的贡献$\psi_t$，再与文本embedding计算相似度——直接关联到查询中提到的实体
    - **补充归因$\Psi^{comp}$**：利用CLIP最后一层非cls token与文本的相似度反转——高相似度的"空白"patch被视为信息寄存器而非重要区域
    - **融合**：$\Psi = \Psi^{cls} + \Psi^{comp} - \Psi^{comp} \cdot \Psi^{cls}$（soft OR操作）——既高亮查询直接相关的区域，又保留潜在相关的区域
    - 设计动机：两个归因图互补——cls分解精确定位显式实体，comp反转保留隐式相关区域

2. **从LLaVA获取归因图**

    - 直接使用LLaVA最后几层的输出token到图像token的注意力权重
    - 在所有生成token和注意力头上取平均作为归因图
    - 更简洁但依赖LLaVA的可访问性

### 损失函数 / 训练策略

API是纯推理时技术，无需训练。归因图$\Psi$经Resize到像素空间后用均值滤波（kernel size $k$）平滑方块效应，得到最终热力图$\Phi$。$\Phi$作为alpha通道与原图相乘得到输入。

## 实验关键数据

### 主实验

| 推理模型 | 提示方法 | VisWiz | TextVQA | MMMU | MM-Vet | LLaVA-Bench |
|---------|---------|--------|---------|------|--------|-------------|
| LLaVA-1.5 | 无提示 | 60.9 | 48.3 | 35.2 | 32.8 | 71.9 |
| LLaVA-1.5 | FGVP (Mask) | 56.9 | 39.4 | 36.1 | 31.0 | 57.4 |
| LLaVA-1.5 | SoM | 54.2 | 18.8 | 35.6 | 26.4 | 56.1 |
| LLaVA-1.5 | **API (CLIP)** | **61.3** | **48.8** | **37.5** | **35.3** | **74.1** |
| LLaVA-1.5 | **API (LLaVA)** | **61.4** | **48.8** | **37.0** | **36.6** | **74.8** |
| GPT-4V | 无提示 | 59.4 | 50.6 | 50.6 | 67.0 | 102.0 |
| GPT-4V | **API (CLIP)** | **69.5** | **51.5** | **51.0** | **67.7** | **103.3** |

### 消融实验

| 消融项 | MM-Vet (LLaVA) |
|--------|----------------|
| Full API (CLIP) | 35.3 |
| 仅$\Psi^{cls}$ | 34.1 |
| 仅$\Psi^{comp}$ | 33.8 |
| 无平滑滤波 | 34.5 |
| 固定注意力（非query-dependent） | 33.2 |

### 关键发现

- API在LLaVA-1.5上: MM-Vet +3.8%, LLaVA-Bench +2.9%, MMMU +2.3%——所有基准一致提升
- 已有视觉提示方法（FGVP、SoM）在多数基准上反而降低性能（尤其TextVQA降>5%）——因为与查询无关的提示会误导模型
- API对GPT-4V同样有效（VisWiz +10.1%），证明了跨模型通用性
- 两种归因图互补：$\Psi^{cls}$擅长显式实体定位，$\Psi^{comp}$擅长整体区域筛选

## 亮点与洞察

- 极简设计：只需叠加一个热力图到原图，无需训练、无需修改模型架构
- query-dependent是关键区分因素——根据不同问题动态生成不同视觉提示，这是之前方法缺乏的
- $\Psi^{cls}$的cls token分解技术提供了一种理解CLIP内部注意力机制的新视角
- 可视为"自反射"或"模型集成"的轻量实现——通过像素空间而非文本空间传递中间知识

## 局限性 / 可改进方向

- 辅助模型的前向推理增加了延迟——对CLIP约增加1次编码，对LLaVA约增加1次生成
- 热力图的alpha blending强度缺乏自适应调整——不同图像/查询可能需要不同强度
- 对需要精确空间推理（如counting、grounding）的任务提升有限
- CLIP归因图的cls分解基于后几层MSA的近似，理论保证不够严格

## 相关工作与启发

- **vs FGVP/SoM**：基于分割的提示是query-independent的，且偏向grounding任务；API是query-dependent的通用VQA提升方案
- **vs Chain-of-Thought**：CoT在文本空间做自反射；API在像素空间做自反射——两者正交且可联合使用
- **启发**：VLM的注意力信号可以被"外化"为视觉提示——这种"看我看什么"的范式可能适用于多轮对话中的上下文理解增强

## 评分

- 新颖性: ⭐⭐⭐⭐ query-dependent视觉提示+cls token分解，概念简洁有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 4个LVLM×6个基准的全面实验，含GPT-4V和Gemini
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，可视化效果好
- 价值: ⭐⭐⭐⭐ 零训练的通用VLM提升方案，实用性强
---
title: >-
  [论文解读] Attention Prompting on Image for Large Vision-Language Models
description: >-
  [ECCV 2024][多模态][提示学习] API用辅助VLM根据文本查询生成注意力热力图叠加原图，引导LVLM关注相关区域，在MM-Vet上提升LLaVA-1.5达3.8%，跨模型通用。
tags:
  - ECCV 2024
  - 多模态
  - 提示学习
  - LVLM
  - 视觉提示
---

# Attention Prompting on Image for Large Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2409.17143](https://arxiv.org/abs/2409.17143)  
**代码**: [GitHub](https://github.com/yu-rp/apiprompting)  
**领域**: 大视觉语言模型 / 视觉提示  
**关键词**: visual prompting, attention heatmap, LVLM, self-reflection, model ensemble

## 一句话总结

提出Attention Prompting on Image（API），通过辅助VLM（CLIP或LLaVA）根据文本查询生成注意力归因图，将其作为热力图叠加到原始图像上引导LVLM聚焦相关区域，在MM-Vet上提升LLaVA-1.5达3.8%，跨多种LVLM（包括GPT-4V）通用有效。

## 研究背景与动机

**领域现状**：视觉提示（visual prompting）通过在图像上添加圈圈、箭头、遮罩等标注来引导LVLM关注特定区域。已有方法（如FGVP、SoM）依赖分割模型生成标注，无需训练且直觉有效。

**现有痛点**：(1) 已有视觉提示技术仅处理图像本身，不考虑文本查询内容——无论问什么问题，同一张图的视觉提示结果相同；(2) 这导致提示区域与实际问题所需关注的区域不匹配；(3) 基于分割的方法（FGVP/SoM）本质是实例级proposal，不适用于通用VQA任务。

**核心矛盾**：如何让视觉提示随文本查询动态变化，使模型根据不同问题关注图像的不同区域？

**切入角度**：利用LVLM自身的视觉-文本对齐能力生成query-aware的注意力归因图，将其作为视觉提示叠加到图像上。

## 方法详解

### 整体框架

给定图像$I$和文本查询$T^i$，API分两步：(1) **归因图生成**：用辅助VLM $g$（可以是CLIP或LLaVA自身）计算文本查询对图像各patch的重要性得分，生成归因图$\Psi \in \mathbb{R}^{P \times P}$；(2) **热力图叠加**：将归因图上采样到像素空间，经均值滤波平滑后作为alpha通道与原图混合，得到标注图像$I^a$送入推理VLM $f$。若$g=f$则为self-reflection，若$g \neq f$则为模型集成。

### 关键设计

1. **CLIP的cls token分解归因图**

    - 利用ViT的残差连接，将CLIP的图像级相似度分解到各patch的贡献
    - 对深层MSA输出逐patch计算与文本嵌入$\hat{T}$的相似度，得到$\Psi^{cls}$——直接定位与查询实体相关的patch
    - 互补归因图$\Psi^{comp}$：最后一层非cls token与$\hat{T}$的相似度取反——低信息量的"寄存器"token相似度高，有实际内容的patch相似度低
    - 最终归因图 $\Psi = \Psi^{cls} + \Psi^{comp} - \Psi^{cls} \cdot \Psi^{comp}$（软OR操作），兼顾显式实体定位和隐式相关区域保留

2. **LLaVA的注意力权重归因图**

    - 直接取LLaVA深层decoder的cross-attention权重（输出token对图像token的注意力值）
    - 在所有生成token和所有注意力头上取平均，得到每个图像patch的平均被关注程度
    - 比CLIP方案更简单，但需要先做一次推理生成输出序列

### 损失函数 / 训练策略

API是一种**无需训练**的推理时技术，不涉及损失函数或训练过程。核心超参数包括：归因图使用的起始层$L'$、均值滤波核大小$k$。

## 实验关键数据

### 主实验

| 模型 | 提示方法 | MM-Vet | LLaVA-Bench | MMMU |
|------|---------|--------|------------|------|
| LLaVA-1.5 | 无提示 | 32.8 | 71.9 | 35.2 |
| LLaVA-1.5 | FGVP (Mask) | 31.0 (-1.8) | 57.4 (-14.5) | 36.1 (+1.0) |
| LLaVA-1.5 | SoM | 26.4 (-6.4) | 56.1 (-15.8) | 35.6 (+0.4) |
| LLaVA-1.5 | **API (CLIP)** | **35.3 (+2.5)** | **74.1 (+2.2)** | **37.5 (+2.4)** |
| LLaVA-1.5 | **API (LLaVA)** | **36.6 (+3.8)** | **74.8 (+2.9)** | **37.0 (+1.8)** |
| GPT-4V | 无提示 | 67.0 | 102.0 | 50.6 |
| GPT-4V | **API (CLIP)** | **67.7 (+0.7)** | **103.3 (+1.3)** | **51.0 (+0.4)** |

### 消融实验

| 消融项 | MM-Vet | 说明 |
|--------|--------|------|
| 仅$\Psi^{cls}$ | 34.1 | 缺少隐式相关patch |
| 仅$\Psi^{comp}$ | 33.8 | 缺少显式实体定位 |
| $\Psi^{cls} + \Psi^{comp}$（软OR） | 35.3 | 两者互补最优 |
| 无均值滤波 | 33.5 | 矩形mask与物体形状不匹配 |
| 不同起始层$L'$ | $L'=L-2$最优 | 浅层信息不够判别 |

### 关键发现

- API在LLaVA-1.5上比FGVP和SoM显著更好——关键在于query-aware（后两者不看问题）
- FGVP和SoM在大多数模型-数据集组合中反而降低性能——query-agnostic的视觉提示可能造成mismatch
- API在闭源模型GPT-4V和Gemini上同样有效（+0.7%到+11.6%），证明了通用性
- 当辅助模型$g$与推理模型$f$相同时（self-reflection），效果最好（API-LLaVA on LLaVA: +3.8%）

## 亮点与洞察

- 将视觉提示从"query-agnostic"升级到"query-aware"是一个关键的范式转变——同一图，不同问题应highlight不同区域
- cls token分解是一种巧妙的归因方法——利用残差连接的可加性将全局相似度拆解到patch级
- 发现非cls token的高相似度实际是"寄存器"功能——与register token的最新研究一致

## 局限性 / 可改进方向

- 需要额外的一次辅助模型前向推理，推理成本翻倍
- 热力图叠加为像素级乘法，可能丢失被压暗区域的信息——对需要全局理解的任务可能有害
- CLIP的归因图对非实体类查询（如"这幅画的风格是什么"）效果可能有限
- 未测试在视频VQA或多图对比等复杂场景中的效果

## 相关工作与启发

- **vs FGVP/SoM**：FGVP/SoM用分割模型生成固定标注，与文本查询无关；API根据查询动态生成归因热力图
- **vs Self-Reflection**：传统self-reflection在文本空间迭代（反复回答+修改），API在像素空间做self-reflection——更直接
- **vs GradCAM**：GradCAM需要梯度回传，API仅需前向推理；关键创新是cls token分解替代了梯度方法
- **启发**：LVLM的注意力权重是宝贵的"免费"信号——可用于更多场景如注意力蒸馏、token剪枝、hallucination检测

## 评分

- 新颖性: ⭐⭐⭐⭐ query-aware视觉提示是重要的范式升级，cls分解方法有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 5个LVLM×6个基准+详细消融+闭源模型验证
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，两种归因图方案对比全面
- 价值: ⭐⭐⭐⭐ 无训练的即插即用方法，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[ECCV 2024\] Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_vision-language_adapters.md)
- [\[ECCV 2024\] FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_high-resolution_vision-language_models.md)
- [\[ECCV 2024\] Quantized Prompt for Efficient Generalization of Vision-Language Models](quantized_prompt_for_efficient_generalization_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
