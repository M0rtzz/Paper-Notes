---
title: >-
  [论文解读] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs
description: >-
  [CVPR 2026][LLM安全][对抗攻击] 发现 ViT 中 Value 特征相比 Patch 特征具有更解耦的局部语义表示，提出 V-Attack 通过自增强 Value 特征 + 文本引导语义操控实现精确可控的 LVLM 局部语义攻击，ASR 平均提升 36%。
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "对抗攻击"
  - "视觉语言模型"
  - "Value特征"
  - "语义操控"
  - "可控攻击"
---

# V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs

**会议**: CVPR 2026  
**arXiv**: [2511.20223](https://arxiv.org/abs/2511.20223)  
**代码**: [GitHub](https://github.com/Summu77/V-Attack)  
**领域**: AI安全  
**关键词**: 对抗攻击, 视觉语言模型, Value特征, 语义操控, 可控攻击

## 一句话总结

发现 ViT 中 Value 特征相比 Patch 特征具有更解耦的局部语义表示，提出 V-Attack 通过自增强 Value 特征 + 文本引导语义操控实现精确可控的 LVLM 局部语义攻击，ASR 平均提升 36%。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：对抗攻击已从干扰分类预测进化到操控 LVLM 的图像语义。但现有方法在精确操控特定概念时成功率极低——同时改变3个概念，成功率<10%。

**核心发现**：ViT 自注意力使 Patch 特征产生语义纠缠（全局上下文主导，局部语义被稀释），而 Value 特征天然抑制全局上下文通道，保留高熵的解耦局部语义。通道分布分析显示 Patch 特征被少数高激活通道（与 CLS token 相关）主导，而 Value 特征分布均匀。

## 方法详解

### 整体框架

多代理模型 Value 特征提取 → Self-Value Enhancement → Text-Guided Value Manipulation → PGD 迭代生成对抗扰动。

### 关键设计

1. **Value 特征解耦性**：分析 CLIP-L/14 发现 Patch 特征的信息熵在中间层骤降，而 Value 特征始终保持高熵。文本对齐分析显示 V 与特定文本的余弦相似度图有清晰空间对齐（"dog" → 0.28 vs X 的 0.22），V 是更精确的语义操控目标。

2. **Self-Value Enhancement**：对提取的 Value 特征做"自注意力"（Q=K=V 全来自 Value），强化局部语义的内部一致性：$\widetilde{\mathbf{V}}^{(k)} = \text{Attn}(\mathbf{V}^{(k)}, \mathbf{V}^{(k)}, \mathbf{V}^{(k)})$

3. **Text-Guided Value Manipulation**：

    - 用 CLIP 文本编码器编码源/目标概念
    - 计算每个增强 Value token 与源文本的余弦相似度
    - 自适应阈值 $\tau^{(k)}$ 选出与源概念对齐的 token 集合 $\mathcal{I}_{\text{align}}^{(k)}$
    - 损失：$\mathcal{L} = \sum_{k} \sum_{i \in \mathcal{I}_{\text{align}}^{(k)}} [-s_i^{(k)}(t_s) + s_i^{(k)}(t_t)]$
    - PGD 迭代 + 随机裁缩增强迁移性

### 损失函数 / 训练策略

跨多个代理模型（CLIP变体）集成优化，对选出的语义对齐 token 同时远离源概念、靠近目标概念。

## 实验关键数据

### 主实验（局部语义攻击，MS-COCO）

| 方法 | LLaVA CAP | InternVL CAP | DeepseekVL CAP | GPT-4o CAP | Avg |
|------|-----------|-------------|----------------|------------|-----|
| MF-it | 0.051 | 0.040 | 0.040 | 0.028 | 0.040 |
| SSA-CWA | 0.262 | 0.304 | 0.241 | 0.285 | 0.273 |
| M-Attack | 0.370 | 0.405 | 0.483 | 0.544 | 0.450 |
| **V-Attack** | **最高** | **最高** | **最高** | **最高** | **+36%** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 攻击 Patch 特征 X | 低 | 语义纠缠 |
| 攻击 Value 特征 V | 显著提升 | 解耦局部语义 |
| +Self-Enhancement | 进一步提升 | 语义更丰富 |
| +Text-Guided | **最优** | 精确定位+操控 |

### 关键发现

- Value 特征抑制了 Patch 中主导全局信息的高激活通道
- 攻击 Value 比攻击 Patch 平均 ASR 提升 36%
- 在 GPT-4o、GPT-o3 等闭源模型上也有效

## 亮点与洞察

- **深度洞察**：首次揭示 ViT Value 特征的解耦性质，为对抗攻击提供新视角
- **精确可控**：实现单概念级别的精准语义替换（"狗"→"猫"）
- **强迁移性**：白盒代理生成的扰动在黑盒 GPT-4o 上有效

## 局限与展望

- 依赖 CLIP 等白盒代理模型，架构差异大时迁移效果可能下降
- 多概念同时攻击的成功率仍有提升空间
- 需关注此类工具的伦理风险

## 相关工作与启发

- AttackVLM 首次用 CLIP 做 LVLM 对抗攻击，V-Attack 找到更精确的攻击目标
- M-Attack 用裁剪增强+模型集成，V-Attack 从特征选择角度切入
- 对 LVLM 安全防御也有启示：可保护 Value 特征空间

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Value 特征解耦的发现非常深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多场景全面验证（含 GPT-4o）
- 写作质量: ⭐⭐⭐⭐ 分析透彻可视化优秀
- 价值: ⭐⭐⭐⭐ 揭示 LVLM 安全隐患
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility](unsafe2safe_controllable_image_anonymization_for_downstream_utility.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](../../NeurIPS2025/llm_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)

</div>

<!-- RELATED:END -->
