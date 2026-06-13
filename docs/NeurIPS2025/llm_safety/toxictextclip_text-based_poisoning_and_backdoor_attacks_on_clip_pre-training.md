---
title: >-
  [论文解读] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training
description: >-
  [NeurIPS 2025][LLM安全][CLIP安全] 提出 ToxicTextCLIP 框架，通过背景感知选择和背景驱动增强两个模块，在 CLIP 预训练阶段生成高质量对抗文本，实现高达 95.83% 投毒成功率和 98.68% 后门 Hit@1，且能突破 RoCLIP、CleanCLIP、SafeCLIP 三种防御。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "CLIP安全"
  - "数据投毒"
  - "backdoor attack"
  - "文本对抗"
  - "多模态安全"
---

# ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training

**会议**: NeurIPS 2025  
**arXiv**: [2511.00446](https://arxiv.org/abs/2511.00446)  
**代码**: [GitHub](https://github.com/xinyaocse/ToxicTextCLIP/)  
**领域**: AI安全  
**关键词**: CLIP安全, 数据投毒, backdoor attack, 文本对抗, 多模态安全

## 一句话总结

提出 ToxicTextCLIP 框架，通过背景感知选择和背景驱动增强两个模块，在 CLIP 预训练阶段生成高质量对抗文本，实现高达 95.83% 投毒成功率和 98.68% 后门 Hit@1，且能突破 RoCLIP、CleanCLIP、SafeCLIP 三种防御。

## 研究背景与动机

CLIP 模型通过对比学习将大规模网络图文数据对齐，在零样本分类、图文检索、图像生成引导等下游任务中取得了卓越表现。然而，CLIP 对未清洗网络数据的依赖使其面临数据投毒和后门攻击的风险。

现有攻击方法主要集中在**图像模态**：通过视觉 patch、对抗扰动等方式注入后门。而**文本模态**的攻击几乎未被系统研究过，尽管文本在 CLIP 对比学习中同样核心。相比图像在传输中可能被压缩或裁剪从而破坏像素级触发器，文本在采集和分发过程中保持完整，这使得文本触发器更加自然、隐蔽且持久。

已有的文本攻击方法（如 mmPoison）仅简单地将图像描述替换为目标类文本，存在两大挑战：

**语义误对齐**：目标类文本中的背景描述可能与目标类语义不一致，削弱投毒效果

**可扩展性不足**：许多目标类在开源语料中缺乏足够的高质量语义一致文本

## 方法详解

### 整体框架

ToxicTextCLIP 是一个**背景敏感的投毒文本生成框架**，由两个迭代执行的模块组成：背景感知目标文本选择器（Background-aware Target Text Selector）和背景驱动投毒文本增强器（Background-driven Poisoned Text Augmenter）。核心思想是先从语料中筛选出背景语义与目标类一致的高质量文本，再通过编码-解码架构增强并扩充这些候选文本。

投毒数据集的构建方式为：将源类图像 $\bm{x}_A$ 的原始文本描述 $\bm{t}_A$ 替换为与目标类 $B$ 语义对齐的投毒文本 $\bm{t}_{p,B}$。

### 关键设计

1. **背景感知目标文本选择器**：其核心任务是从目标类文本中挑选出背景内容与目标类语义高度一致的候选文本。具体步骤：

    - 对每个候选文本 $\bm{t}_{B,j}$，构建所有可能的背景描述集合 $\mathcal{S}_j$，即从原文中删除至多 $\eta$ 个词的所有组合
    - 通过多模板平均计算稳定的类别语义中心 $\bm{Z}_B = \frac{1}{n}\sum_{i=1}^n E^T(\text{Temp}_i(B))$
    - 对每个背景候选计算评分：$S_{b,j}^* = \arg\max_{S_{b,j} \in \mathcal{S}_j} (\text{Sim}(E^T(S_{b,j}), E^I(\bm{x}_{B,j})) - \text{Sim}(E^T(S_{b,j}), \bm{Z}_B))$
    - 最终按背景与类别中心的相似度降序排列，优先选用背景最一致的文本
    - 设计动机：直接使用目标类文本可能包含与目标类不一致的背景信息，通过显式分离类别相关语义和背景内容，确保投毒文本的语义完整性

2. **背景驱动投毒文本增强器**：解决目标类语料不足和语义对齐不够的问题。包含四个子步骤：

    - **特征编码**：用 CLIP 文本编码器提取候选文本的嵌入 $\bm{f}_j^T$
    - **特征增强**：融合图像特征 $\bm{f}_j^T = \bm{f}_j^T + \lambda \bm{f}_j^I$，其中 $\lambda$ 控制视觉特征的影响程度
    - **Transformer 解码**：使用交叉注意力机制融合文本特征和图像 patch 嵌入，并通过 Diverse Beam Search (DBS) 生成多样化候选，公式为 $\text{Cro\_Att} = \text{softmax}(\frac{Q \cdot (\bm{Z}_{\text{patch}}^I \oplus \bm{f}_j^T)^\mathsf{T}}{\sqrt{d_k}}) * (\bm{Z}_{\text{patch}}^I \oplus \bm{f}_j^T)$
    - **Jaccard 后处理**：通过 Jaccard 距离迭代选择最不相似的候选文本，去除 DBS 产生的冗余样本
    - 设计动机：超过 50% 的 ImageNet 类在 1M 规模语料中无法支持每类 30 条投毒文本的攻击需求

3. **后门攻击扩展**：与投毒攻击不同，后门攻击的目标是让任何包含触发器 $\bm{b}$ 的输入文本都映射到预定义目标类。方法是从目标类采样多张图像，为每张检索其他类别的代表性文本，然后在这些文本后附加触发器，构建多样化的投毒训练对。支持词级（如罕见词 "zx"）和句子级触发器（如 "Please return high-quality results."）

### 损失函数 / 训练策略

- 受害者模型使用 AdamW 优化器、余弦学习率调度（初始 $5 \times 10^{-5}$），批大小 512，训练 10 个 epoch
- 替代模型使用 OpenAI ViT-B/32 CLIP（与受害者模型架构不同，确保黑盒设定）
- 文本解码器为 6 层 Transformer，使用 Adam 优化器、逆平方根调度器，初始学习率 $10^{-3}$，批大小 832，训练 32 个 epoch

## 实验关键数据

### 主实验

| 数据集 | 攻击类型 | 方法 | CA | ASR | Hit@1 | Hit@5 |
|--------|----------|------|-----|------|-------|-------|
| CC3M | 单目标投毒 | mmPoison | 31.52 | 62.50 | - | - |
| CC3M | 单目标投毒 | **ToxicTextCLIP** | 32.23 | **95.83** | - | - |
| CC3M | 词级后门 | Baseline | 31.91 | - | 72.13 | 96.74 |
| CC3M | 词级后门 | **ToxicTextCLIP** | 32.03 | - | **92.66** | 98.87 |
| CC3M | 句级后门 | Baseline | 32.45 | - | 64.41 | 89.64 |
| CC3M | 句级后门 | **ToxicTextCLIP** | 34.67 | - | **98.68** | 99.81 |

### 防御鲁棒性实验

| 攻击类型 | 防御 | 方法 | ASR/Hit@1 |
|----------|------|------|-----------|
| 单目标投毒 | RoCLIP | mmPoison / ToxicTextCLIP | 33.33 / **70.83** |
| 单目标投毒 | CleanCLIP | mmPoison / ToxicTextCLIP | 45.83 / **75.00** |
| 单目标投毒 | SafeCLIP | mmPoison / ToxicTextCLIP | 25.00 / **64.17** |
| 句级后门 | RoCLIP | Baseline / ToxicTextCLIP | 57.82 / **91.15** |
| 句级后门 | CleanCLIP | Baseline / ToxicTextCLIP | 56.29 / **86.63** |

### 消融实验

| 配置 | ASR (无防御) | ASR (RoCLIP) | 说明 |
|------|-------------|-------------|------|
| mmPoison baseline | 62.50 | 33.33 | 基线方法 |
| w/o 选择器 | 87.50 | 62.50 | 去掉背景感知选择 |
| w/o 增强器 | 83.33 | 58.33 | 去掉背景驱动增强 |
| **完整框架** | **95.83** | **70.83** | 两模块协同效果最佳 |

### 关键发现

- 生成的投毒文本困惑度（408.89）低于原始网络文本（755.27），说明生成文本质量更高
- 投毒率很低（每类 35 条，仅占语料 0.003%）即可达到很高攻击成功率
- 模型仅需 2-3 个 epoch 即可被投毒文本影响，说明文本攻击传播极快

## 亮点与洞察

- **首次系统研究** CLIP 预训练阶段的纯文本攻击，揭示了文本模态这一被忽视的攻击面
- 背景语义一致性是文本投毒成功的关键——不仅要文本包含目标类关键词，还需背景描述与目标类视觉场景匹配
- 三种 SOTA 防御（RoCLIP、CleanCLIP、SafeCLIP）均无法有效缓解该攻击

## 局限与展望

- 攻击仅在 ResNet-50 架构的 CLIP 上验证，未测试 ViT 等其他架构
- 依赖替代模型（OpenAI CLIP）进行文本生成，若替代模型与受害者模型差异过大可能影响效果
- 作者建议的防御方向：基于语言模型的文本异常检测、跨模态背景一致性验证

## 相关工作与启发

- 与 mmPoison（Yang et al., 2023b）相比，ToxicTextCLIP 从"简单替换"升级到"背景感知生成"，大幅提升攻击效果
- 启发：多模态模型的安全性需要同时关注两种模态；现有防御主要针对图像触发器设计，需要发展模态感知的防御机制

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究文本投毒CLIP，但攻击思路是渐进式改进
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、三种攻击类型、三种防御、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式规范
- 价值: ⭐⭐⭐⭐ 揭示重要安全漏洞，对多模态安全研究有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[ACL 2025\] Exploring Forgetting in Large Language Model Pre-Training](../../ACL2025/llm_safety/exploring_forgetting_in_large_language_model_pre-training.md)
- [\[ICML 2025\] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP](../../ICML2025/llm_safety/x-transfer_attacks_towards_super_transferable_adversarial_attacks_on_clip.md)
- [\[ICML 2025\] The Ripple Effect: On Unforeseen Complications of Backdoor Attacks](../../ICML2025/llm_safety/the_ripple_effect_on_unforeseen_complications_of_backdoor_attacks.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](../../ICML2025/llm_safety/iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)

</div>

<!-- RELATED:END -->
