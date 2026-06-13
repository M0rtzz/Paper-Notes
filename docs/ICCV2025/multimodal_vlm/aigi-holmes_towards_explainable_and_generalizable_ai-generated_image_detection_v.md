---
title: >-
  [论文解读] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][AI生成图像检测] 提出 AIGI-Holmes，通过构建包含解释性标注的 Holmes-Set 数据集和精心设计的三阶段训练流程（视觉专家预训练 → SFT → DPO），将 MLLM 改造为既能准确检测 AI 生成图像又能提供人类可验证解释的"福尔摩斯"检测器，推理阶段通过协同解码策略进一步增强泛化能力。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "AI生成图像检测"
  - "多模态大语言模型"
  - "可解释检测"
  - "直接偏好优化"
  - "协同解码"
---

# AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: [https://github.com/wyczzy/AIGI-Holmes](https://github.com/wyczzy/AIGI-Holmes)  
**领域**: 多模态VLM  
**关键词**: AI生成图像检测, 多模态大语言模型, 可解释检测, 直接偏好优化, 协同解码

## 一句话总结

提出 AIGI-Holmes，通过构建包含解释性标注的 Holmes-Set 数据集和精心设计的三阶段训练流程（视觉专家预训练 → SFT → DPO），将 MLLM 改造为既能准确检测 AI 生成图像又能提供人类可验证解释的"福尔摩斯"检测器，推理阶段通过协同解码策略进一步增强泛化能力。

## 研究背景与动机

### 问题定义
AI 生成内容（AIGC）技术的快速发展使得高度逼真的 AI 生成图像（AIGI）被滥用于传播虚假信息，威胁公共信息安全。现有检测方法面临两个核心问题：

**缺乏可解释性**：当前检测模型是黑盒的，检测结果难以由人类验证。没有人类可验证的解释导致检测结果不可靠。

**缺乏泛化性**：AIGC 技术迭代极快（如 FLUX、SD3.5、VAR 等），现有方法难以泛化到最新的生成技术。

### 为什么用 MLLM？
MLLM 具备常识理解和自然语言生成能力，能够从语义层面分析视觉内容，是解决可解释性和泛化性的理想候选。但直接使用 MLLM 面临两大挑战：

- **训练数据稀缺**：现有的 AIGI 检测数据集（如 CNNDetection、GenImage、DRCT）仅包含视觉模态，缺乏适用于 MLLM SFT 的指令微调数据集。FakeBench 和 LOKI 虽有初步尝试，但依赖 GPT-4o 标注且规模太小。
- **次优的监督微调**：仅在 SFT 数据集上训练 MLLM 效果有限，因为 MLLM 在图像分类和低级感知任务上能力不足，且 SFT 模型可能机械复制解释模板而非真正理解伪影或语义错误的根因。

## 方法详解

### 整体框架

AIGI-Holmes 的设计包含两个核心部分：**Holmes-Set 数据集**和 **Holmes Pipeline 训练框架**。

在架构上，作者在 LLaVA 基础上增加了一个 NPR（Neighboring Pixel Relationships）视觉专家，用于捕获低级伪影信息。整个模型的输入处理为：
- CLIP 视觉编码器 $F$ 提取高级语义特征 $f_{img}$
- NPR 视觉专家 $R$ 提取低级伪影特征 $f_{npr}$
- 两者通过 projector 注入 LLM：$H = \text{LLM}(\text{proj}([f_{img}, f_{npr}]), f_t)$

### 关键设计一：Holmes-Set 数据集

#### Holmes-SFTSet（65K 图像）
数据来源分为两部分：
1. **已有数据集**：从 CNNDetection、GenImage、DRCT 中选取 45K 图像
2. **专家过滤图像**：使用专家小模型过滤出 20K 含有常见 AI 生成缺陷的图像（文本、人体、人脸、射影几何、常识、物理定律）

标注方法使用 **Multi-Expert Jury**，由 4 个开源 MLLM（Qwen2VL-72B、InternVL2-76B、InternVL2.5-78B、Pixtral-124B）交叉标注和评估：
- **General Positive Prompt**：从高级语义（解剖学、物理定律）和低级维度（纹理、清晰度）分析
- **General Negative Prompt**：生成对抗性标注，与正向标注构成 DPO 数据对 $D_1$
- **Specialist Prompt**：针对 20K 专家过滤图像的特定缺陷标注

质量控制采用 MLLM-as-a-judge 方式，仅保留共识度最高的标注。

#### Holmes-DPOSet（65K + 4K）
为解决 SFT 模型"机械复制模板"的问题，构建人类对齐的偏好数据集：
- $D_1$：来自 General Positive/Negative Prompt 的自然正负对
- $D_2$：**人工修改**，2K 人工标注 + 2K 使用 Specialist Prompt 修改的样本。人类专家对 SFT 模型输出提供修改建议（补充正确信息、删除错误/无关解释），然后用 DeepSeek-V3 执行修改

### 关键设计二：Holmes Pipeline（三阶段训练）

**阶段一：Visual Expert Pre-training（视觉专家预训练）**

目标是让视觉专家在 AIGI 检测领域获得泛化能力。分别对两个视觉编码器进行二分类预训练：
- CLIP-ViT-L/14 使用 LoRA（$r=4, \alpha=8$）微调，从 CLS 特征 $f_{cls}$ 通过 MLP 获得分类结果
- NPR-based ResNet 前两层进行全参数微调，同样通过 MLP 分类
- 损失函数：$l_{clip} = l_{bce}(y_{clip}, y), \quad l_{npr} = l_{bce}(y_{npr}, y)$

**为什么要预训练？** 原始 CLIP 和 ResNet 并非为 AIGI 检测设计，直接用于下游任务时在低级伪影感知和分类能力上不足。预训练使它们具备领域特定的特征提取能力。

**阶段二：Supervised Fine-Tuning（SFT）**

将预训练好的视觉专家集成到 LLM 中，在 Holmes-SFTSet 上进行自回归文本损失训练：
- 冻结视觉专家
- 训练 projector 和 LLM 的 LoRA 组件（rank=128, α=256）
- 损失：$l_{txt} = l_{ce}(H, H_{txt})$

**阶段三：Direct Preference Optimization（DPO）**

在 Holmes-DPOSet ($D = D_1 \cup D_2$) 上进行人类偏好对齐：

$$L_{DPO}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\phi(y_w|x)}{\pi_{base}(y_w|x)} - \log \frac{\pi_\phi(y_l|x)}{\pi_{base}(y_l|x)} \right) \right) \right]$$

DPO 的关键作用是重塑 MLLM 的推理模式，使解释结果与人类判断标准对齐，而非停留在次优微调的水平。

### 关键设计三：Collaborative Decoding（协同解码）

推理阶段，MLLM 与预训练视觉专家联合决策，调整"real"和"fake" token 的 logit 值：

$$\text{logit}_{new}(y=k) = \alpha \cdot \text{logit}_{raw}(y=k) + \beta \cdot \text{logit}(y_{clip}=k) + \gamma \cdot \text{logit}(y_{npr}=k)$$

其中 $\alpha=1, \beta=1, \gamma=0.2$。

**为什么有效？** 通过保留 MLLM 预测的同时引入视觉专家的判断，防止 MLLM 过拟合到已见过的伪造类型，从而提高在未见领域的泛化能力。

### 损失函数 / 训练策略

| 阶段 | 可训练参数 | 损失函数 | 超参数 |
|------|-----------|---------|--------|
| Visual Expert Pre-training | CLIP LoRA(r=4) + ResNet全参 | Binary CE | batch=32, 5 epochs |
| SFT | Projector + LLM LoRA(r=128) | Autoregressive CE | lr=5e-5, batch=16, 3 epochs |
| DPO | Projector + LLM LoRA(r=48) | DPO Loss | lr=5e-7, batch=4, β=0.1, 2 epochs |

## 实验关键数据

### 主实验

论文在三个协议（Protocol-I/II/III）下评估，其中 P3 最具挑战性——训练在扩散模型数据上，测试在全新的自回归生成模型和最先进扩散模型上。

**Protocol-III 检测准确率（Acc. %）**：

| 方法 | VAR | FLUX | Janus-Pro-7B | SD3.5-Large | Mean Acc. | Mean A.P. |
|------|-----|------|--------------|-------------|-----------|-----------|
| CNNSpot | 59.9 | 63.8 | 85.0 | 78.2 | 72.9 | 85.6 |
| NPR | 85.9 | 91.6 | 73.9 | 93.4 | 84.0 | 89.5 |
| UnivFD | 64.3 | 87.8 | 96.4 | 75.7 | 83.6 | 95.9 |
| RINE | 85.0 | 97.8 | 97.2 | 98.9 | 96.2 | 99.5 |
| AIDE | 93.6 | 99.4 | 97.8 | 98.6 | 97.0 | 99.7 |
| **AIGI-Holmes** | **99.6** | **99.4** | **98.0** | **99.9** | **99.2** | **99.9** |

AIGI-Holmes 在所有生成器上均达到 98%+ 的准确率，Mean Acc. 比 AIDE 高 2.2%，比 RINE 高 3.0%。

**解释质量对比（MLLM vs AIGI-Holmes）**：

| 模型 | BLEU-1 | ROUGE-L | CIDEr | ELO Rating |
|------|--------|---------|-------|------------|
| GPT-4o | 0.433 | 0.308 | 0.005 | 10.271 |
| Pixtral-124B | 0.428 | 0.270 | 0.010 | 10.472 |
| AIGI-Holmes (w/o DPO) | 0.445 | 0.315 | 0.023 | 10.670 |
| **AIGI-Holmes (w/ DPO)** | **0.622** | **0.375** | **0.107** | **11.420** |

### 消融实验

**核心组件消融（Acc. %）**：

| VEP-S | DPO | CD | P1 | P3 |
|-------|-----|----|----|-----|
| ✗ | ✗ | ✗ | 83.3 | 90.1 |
| ✓ | ✗ | ✗ | 84.8 | 92.3 |
| ✓ | ✓ | ✗ | 87.4 | 97.6 |
| ✓ | ✗ | ✓ | 90.8 | 98.9 |
| ✓ | ✓ | ✓ | **93.2** | **99.2** |

- Visual Expert Pre-training 在 P3 上提升 +2.2%
- DPO 提升 +0.4%（但 ELO Rating 提升 0.75 分）
- Collaborative Decoding 贡献最大：+1.7%
- 三者组合比基线提升约 10%

**鲁棒性评估（P3 平均 Acc. %）**：

| 方法 | JPEG QF=75 | Gaussian σ=2 | Resize ×0.5 |
|------|-----------|-------------|-------------|
| AIDE | 92.8 | 90.7 | 89.2 |
| RINE | 92.4 | 92.8 | 92.3 |
| **AIGI-Holmes** | **99.0** | **97.9** | **95.9** |

### 关键发现

1. **协同解码是泛化的关键**：Collaborative Decoding 通过引入视觉专家的领域知识，有效防止 MLLM 过拟合到训练见过的伪造类型
2. **DPO 对解释质量至关重要**：虽然 DPO 对检测准确率提升有限（+0.4%），但对解释质量提升显著（ELO +0.75）
3. **MLLM 关注高级语义特征**：即使在 JPEG 压缩、高斯模糊等扰动下，解释质量指标未显著下降，说明 MLLM 不依赖低级伪影而是关注高级语义
4. **多专家交叉验证提升数据质量**：Multi-Expert Jury 方法比单模型标注更可靠

## 亮点与洞察

1. **数据驱动的方法论**：Holmes-Set 是首个包含解释性标注和人类偏好数据的 AIGI 检测数据集，填补了数据空白
2. **三阶段训练设计精妙**：Visual Expert → SFT → DPO 的递进设计，每个阶段解决一个具体问题（特征提取→解释生成→人类对齐）
3. **推理阶段的创新**：Collaborative Decoding 不增加训练成本，仅在推理时引入视觉专家判断，是一种低成本且高效的泛化增强策略
4. **Multi-Expert Jury 数据标注方法**：用多个 MLLM 交叉标注取代昂贵的 GPT-4o/人工标注，降低成本的同时保证质量

## 局限与展望

1. **推理开销**：协同解码需要同时运行 MLLM 和视觉专家，增加了推理时间
2. **基座模型依赖**：基于 LLaVA-1.6-mistral-7B，换用更强的 MLLM 基座可能进一步提升
3. **DPO 数据规模**：人工修改的 DPO 数据仅 4K，扩大规模可能进一步提升解释质量
4. **实时检测场景**：作为 MLLM 方案，难以满足实时检测需求
5. **视频生成检测**：当前聚焦图像，视频 AIGC（如 Sora）检测是未探索的方向

## 相关工作与启发

- **NPR 视觉专家的启发**：低级邻域像素关系对 AIGI 检测很有价值，但需要与高级语义特征结合才能发挥最大作用
- **DPO 在垂直领域的应用**：本文展示了 DPO 不仅适用于通用对话，在垂直检测任务中同样能有效提升输出质量与人类偏好对齐
- **可解释 AI 的新范式**：从"后验解释"转向"生成式解释"，让模型在检测的同时自然地输出推理过程

## 评分

- 新颖性: ⭐⭐⭐⭐ （三阶段训练框架设计独到，但各组件均非全新）
- 实验充分度: ⭐⭐⭐⭐⭐ （三个协议、鲁棒性测试、解释质量评估、消融全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但部分细节需查附录）
- 价值: ⭐⭐⭐⭐⭐ （可解释+泛化的 AIGI 检测是重要且实用的方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](docthinker_explainable_multimodal_large_language_models_with.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

</div>

<!-- RELATED:END -->
