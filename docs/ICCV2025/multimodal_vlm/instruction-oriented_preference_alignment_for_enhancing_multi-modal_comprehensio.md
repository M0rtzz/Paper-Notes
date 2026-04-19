---
title: >-
  [论文解读] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs
description: >-
  [ICCV 2025][多模态][多模态大模型] 提出**指令导向偏好对齐（IPA）**框架，通过自动化偏好构建机制和渐进式偏好数据收集管线，将对齐信号锚定在**指令完成效能**而非仅局限于幻觉因素，在 Qwen2VL-7B 上跨 9 个基准（幻觉评估、通用VQA、文本理解）实现一致性提升。
tags:
  - ICCV 2025
  - 多模态
  - 多模态大模型
  - 偏好对齐
  - 指令遵循
  - DPO
  - 幻觉缓解
---

# Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs

**会议**: ICCV 2025  
**arXiv**: 2503.20309  
**代码**: [数据集](https://huggingface.co/datasets/wangzt-kghl/IPA)  
**领域**: multimodal_vlm  
**关键词**: 多模态大模型, 偏好对齐, 指令遵循, DPO, 幻觉缓解  
**作者**: Zitian Wang, Yue Liao, Kang Rong, Fengyun Rao, Yibo Yang, Si Liu (北航, NUS, KAUST)

## 一句话总结

提出**指令导向偏好对齐（IPA）**框架，通过自动化偏好构建机制和渐进式偏好数据收集管线，将对齐信号锚定在**指令完成效能**而非仅局限于幻觉因素，在 Qwen2VL-7B 上跨 9 个基准（幻觉评估、通用VQA、文本理解）实现一致性提升。

## 研究背景与动机

多模态大语言模型（MLLMs）在视觉-语言理解方面取得了令人瞩目的进展，但 SFT 数据中的噪声或模糊标注会导致模型学习到不正确的信息。**偏好对齐**作为后训练策略应运而生。

然而，现有偏好对齐方法存在根本性问题：

**过度聚焦幻觉因素**：RLHF-V、RLAIF-V、Topic-Overwrite 等方法主要通过注入或检测幻觉来构建偏好对，将改进局限于幻觉缓解

**忽略内在质量维度**：当多个响应在幻觉方面无显著差异时，这些方法缺乏区分能力

**人工/商业模型依赖**：人工标注成本高，依赖 GPT-4V 则财务开销大

作者提出的核心问题：**偏好对中哪些因素真正决定了对齐方向？**

答案：**指令完成效能**——响应是否充分满足了指令的核心要求。例如，对于 "Is there a cat in the image?" 这一问题，"Yes." 和 "The image's lower-left corner displays a gray cat." 在幻觉维度上无差异，但后者通过展示观察逻辑和细节完整度，更好地满足了指令要求。

## 方法详解

### 整体框架

IPA 由两部分组成：**自动化偏好构建机制** + **渐进式偏好数据收集管线**。

### 阶段一：响应采样 (Sampling)

对每个样本 $s = (V, I, r^{ref})$，采用两种策略：

**正常采样**：
$$r^{norm} \sim \pi_G(\cdot | V, I; \theta_{\pi_G})$$

**对比采样**（受 VCD 启发）：
$$r^{cont} \sim \pi_G(\cdot | t(V), I; \theta_{\pi_G}), \quad t \sim \mathcal{T}$$

其中 $t$ 是从扰动集 $\mathcal{T}$（噪声、模糊、缩放等）中随机选择的扰动算子。对比采样旨在增强与 $\pi_G$ 鲁棒性相关的缺陷响应模式多样性。

### 阶段二：反射式响应修正 (Reflective Revision)

通过两个阶段改进响应：

1. **诊断反馈生成**：修正模型 $\pi_R$ 识别初始响应中的关键缺陷
$$fb \sim \pi_R(\cdot | V, I, r, I^{fb}; \theta_{\pi_R})$$

2. **反馈驱动精修**：根据反馈信息精修响应
$$r^{rev} \sim \pi_R(\cdot | V, I, r, fb, I^{rf}; \theta_{\pi_R})$$

修正阶段不仅纠正视觉幻觉，更重要的是**扩展任务相关信息的深度**。

### 阶段三：指令导向验证 (Instruction-Oriented Verification)

核心设计——通过验证过程将对齐信号提取从格式约束中**解耦**：

$$v \sim \pi_V(\cdot | V, I, r, r^{ref}, I^{ver}; \theta_{\pi_V})$$

验证模型评估四个维度：
- **逻辑蕴含**：参考答案能否从响应中逻辑推导出
- **细节完整性**：是否缺失关键细节
- **矛盾检测**：响应与参考是否存在矛盾
- **指令遵从**：是否遵循指令

$v = 1$ 表示通过所有条件（preferred），$v = 0$ 表示存在某些失败模式（dispreferred）。最终构建偏好对 $\mathcal{P}^{resp} = \{(r^w, r^l)\}$。

### 渐进式偏好收集管线

**Round 1**：在种子数据集上执行偏好构建，收集初始偏好数据。无法生成 $\mathcal{R}^{win}$ 的困难样本保留到 $\mathcal{D}_2$。

**Round 2（自我进化）**：用 Round 1 的偏好数据通过 DPO 优化生成器和修正器：
$$\pi_G^+ = \text{DPO}(\pi_G, \mathcal{P}_1^{resp}), \quad \pi_R^+ = \text{DPO}(\pi_R, \mathcal{P}_1^{fb})$$
增强后的模型重新处理困难样本。

**Round 3（参考引导）**：对剩余困难样本，在修正阶段引入参考响应 $r^{ref}$：
$$fb \sim \pi_R(\cdot | V, I, r, r^{ref}, I^{fb}; \theta_{\pi_R})$$

最终收集 **89K 偏好对**。

### 训练细节

- 基座模型：Qwen2VL-7B
- 对齐方式：DPO + LoRA
- 训练 1 epoch
- 数据集涵盖 VQA、文本理解、开放指令等多源多模态样本

## 实验关键数据

### 主实验：跨基准一致性提升

| 模型 | HallBench | POPE | MMHal | MMMU | MMStar | MMVet | MME | LLaVA | OCR |
|------|-----------|------|-------|------|--------|-------|-----|-------|-----|
| Qwen2VL-7B | 50.0 | 86.2 | 3.6 | 53.8 | 60.7 | 63.1 | 1676 | 76.6 | 86.2 |
| **+ IPA** | **54.3** | **87.2** | **3.7** | **54.6** | **61.7** | **64.2** | **1687** | **84.0** | **87.3** |
| 提升 | +4.3 | +1.0 | +0.1 | +0.8 | +1.0 | +1.1 | +11 | +7.4 | +1.1 |
| Qwen2.5VL-7B | 54.7 | 86.2 | 3.7 | 57.6 | 64.7 | 65.6 | 1694 | 75.2 | 87.5 |
| **+ IPA** | **55.7** | **86.6** | **3.9** | **59.8** | **66.5** | **68.3** | **1707** | **87.3** | **88.2** |

在 Qwen2.5VL-7B 上的提升同样一致，说明偏好数据具有**跨模型迁移能力**。

### 与现有方法对比

| 方法 | HallBench | POPE | MMMU | MMStar | MMVet | MME | LLaVA | OCR |
|------|-----------|------|------|--------|-------|-----|-------|-----|
| RLHF-V | 50.2 | 86.4 | 52.9 | 61.0 | 60.4 | 1682 | 76.6 | 86.3 |
| RLAIF-V | 50.2 | 86.5 | 53.8 | 61.1 | 58.2 | 1674 | 78.8 | 86.4 |
| VLFeedback | 52.2 | 84.7 | 53.7 | 60.7 | 60.6 | 1682 | 81.4 | 86.6 |
| MMPR | 53.4 | 86.4 | 54.3 | 58.5 | 61.7 | 1681 | 83.5 | 86.3 |
| **IPA (ours)** | **54.3** | **87.2** | **54.6** | **61.7** | **64.2** | **1687** | **84.0** | **87.3** |

- 幻觉导向方法（RLHF-V、RLAIF-V、Topic-Overwrite）往往在多维评估中出现性能权衡
- IPA 在各维度上实现一致提升，特别是 MMVet (+1.1) 和 LLaVABench (+7.4) 大幅领先

### 消融实验

| 配置 | 平均分 | HallBench | MMMU | MMStar | MMVet | OCR |
|------|--------|-----------|------|--------|-------|-----|
| Baseline | 68.1 | 50.0 | 53.8 | 60.7 | 63.1 | 86.2 |
| Round 1 w/o CS | 69.1 | 52.3 | 53.4 | 61.2 | 61.7 | 87.0 |
| Round 1 | 70.1 | 52.9 | 54.2 | 61.9 | 62.7 | 87.0 |
| + Round 2 | 70.4 | 53.7 | 54.6 | 61.7 | 63.8 | 86.9 |
| + Round 3 w/ RGS | 69.8 | 53.4 | 54.7 | 61.1 | 62.7 | 87.2 |
| **+ Round 3** | **70.5** | **54.3** | **54.6** | **61.7** | **64.2** | **87.3** |

关键发现：
- **对比采样**贡献 1.0 分平均增益，在 HallBench 上贡献 +2.9
- **自我进化**（Round 2）进一步提升 0.3 分
- **参考引导采样（RGS）**会诱发简单模仿，性能反而下降 0.7 分
- **参考引导反馈**（Round 3 正式版）有效回收困难样本

## 亮点与洞察

1. **范式转换**：从幻觉导向→指令导向的偏好对齐，锚定在 MLLM 的指令完成能力上，实现全面性能提升而非单一维度优化
2. **验证机制的巧妙设计**：将偏好判断转化为二值验证任务，通过解耦格式变异来建立清晰的决策边界
3. **渐进式收集管线**：三轮渐进策略最大化困难样本利用率，89K 偏好对覆盖多源多模态场景
4. **跨模型泛化**：在 Qwen2.5VL-7B 上的一致提升证明偏好信号的通用性
5. **对比采样的有效性**：视觉扰动不仅增加负样本多样性，还暴露了模型在多模态理解鲁棒性上的薄弱环节

## 局限性

1. 由于计算资源限制，未在更大规模模型（如数百亿参数）上验证
2. 验证模型和被对齐模型使用相同的 Qwen2VL-7B，可能存在偏差
3. 偏好数据主要基于 Qwen2VL 构建，在更弱的模型（LLaVA-1.5-7B）上需要额外过滤
4. 未探索在线偏好优化（如 online DPO 或 RLHF with PPO）的效果

## 相关工作与启发

- **与 MMPR 的差异**：MMPR 聚焦 CoT 推理和格式化答案匹配，IPA 聚焦指令完成能力
- **与 VLFeedback 的对比**：VLFeedback 依赖 GPT-4V 标注且优化 helpfulness/faithfulness/ethics，IPA 完全自动化且以指令遵从为核心
- **对后续工作的启发**：指令导向的偏好构建范式可以推广到任意多模态任务和 MLLM 架构

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 指令导向的偏好对齐视角新颖，验证机制设计巧妙
- **实验**: ⭐⭐⭐⭐ — 9 个基准全面覆盖，消融充分，跨模型验证增强说服力
- **写作**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，Figure 1 的对比直观有力
- **价值**: ⭐⭐⭐⭐ — 提供了可扩展的偏好数据构建范式，对 MLLM 社区有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference](../../ACL2025/multimodal_vlm/omnialign-v_towards_enhanced_alignment_of_mllms_with_human_preference.md)
- [Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inference-time_search_with_vision_value_model_for_improved_visual_compre.md)
- [Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)

<!-- RELATED:END -->
