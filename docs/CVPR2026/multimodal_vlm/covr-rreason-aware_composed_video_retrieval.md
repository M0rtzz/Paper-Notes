---
title: >-
  [论文解读] CoVR-R: Reason-Aware Composed Video Retrieval
description: >-
  [CVPR 2026][多模态VLM][组合视频检索] CoVR-R 提出了推理优先的零样本组合视频检索框架，利用大型多模态模型（Qwen3-VL）显式推理编辑操作隐含的"后效应"（状态转换、时间阶段、镜头变化等），并构建了包含结构化推理轨迹和困难干扰项的 CoVR-R 基准来评估推理能力…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "组合视频检索"
  - "推理感知检索"
  - "后效应推理"
  - "零样本检索"
  - "大型多模态模型"
---

# CoVR-R: Reason-Aware Composed Video Retrieval

**会议**: CVPR 2026  
**arXiv**: [2603.20190](https://arxiv.org/abs/2603.20190)  
**代码**: [github.com/mbzuai-oryx/CoVR-R](https://github.com/mbzuai-oryx/CoVR-R)  
**领域**: Multimodal / Video-Language Models  
**关键词**: 组合视频检索, 推理感知检索, 后效应推理, 零样本检索, 大型多模态模型

## 一句话总结

CoVR-R 提出了推理优先的零样本组合视频检索框架，利用大型多模态模型（Qwen3-VL）显式推理编辑操作隐含的"后效应"（状态转换、时间阶段、镜头变化等），并构建了包含结构化推理轨迹和困难干扰项的 CoVR-R 基准来评估推理能力，在检索准确率上大幅超越现有方法。

## 研究背景与动机

组合视频检索（CoVR）的目标是，给定一个参考视频和修改文本，找到反映所请求变化的目标视频。现有方法存在关键局限：

**关键词匹配的局限**：大多数方法基于三元组驱动的训练，主要奖励关键词重叠，而忽略了修改文本隐含的后效应（after-effects）。例如，"换成特写镜头"隐含了更紧凑的取景和更短的时长；"煎炸"隐含了烟雾和更快的手部动作。

**从说了什么到必须发生什么的鸿沟**：编辑文本明确说的和目标视频必须展示的之间存在差距，弥合这个差距需要推理——预测连接编辑到可能视频证据的因果链。

**现有基准不评估推理**：先前的 CoVR 数据集强调字面编辑或描述对齐，不评估因果合理性和时间一致性。

**核心动机**：将推理显式引入检索循环，通过预测编辑的后果来驱动目标检索，从"匹配关键词"转向"推理后果"。

## 方法详解

### 整体框架

CoVR-R 采用两阶段"推理-然后-检索"（Reason-then-Retrieve）架构：

- **Stage 1 推理**：使用 Qwen3-VL-8B 根据参考视频 $V_r$ 和编辑文本 $E$，生成结构化的后效应推理轨迹 $R$
- **Stage 2 检索**：将 $(V_r, E, R)$ 转换为效应感知查询嵌入，与预计算的 gallery 嵌入进行余弦相似度检索

整个框架保持 LMM 冻结，不依赖 CoVR 特定监督，实现零样本检索。

### 关键设计

1. **Gallery 视频编码**：对每个视频 $V$ 用 Qwen3-VL 生成详细描述 $D(V)$，提取最后一层 token 嵌入，通过**重要性加权池化**聚合为单个向量。权重按语义信息量分三档：$\alpha_{\text{high}}=1.0$（动作、物体、状态）、$\alpha_{\text{mid}}=0.3$（属性、场景）、$\alpha_{\text{low}}=0.1$（功能词）。所有嵌入 L2 归一化后离线缓存。

2. **推理感知查询编码**（三步骤）：

    - **后效应推理**：提示 Qwen3-VL 根据 $(V_r, E)$ 生成结构化推理轨迹 $R = \{\text{states}, \text{actions}, \text{scene}, \text{camera}, \text{tempo}\}$，每个槽位最多 4 个原子断言
    - **目标描述生成**：以 $(V_r, E, R)$ 为条件，生成假想编辑后视频的完整描述 $D_{\text{target}}$
    - **嵌入提取与池化**：同样提取 token 嵌入并用重要性加权池化聚合

3. **CoVR-R 基准构建**：

    - 从 Dense-WebVid-CoVR 和 Something-Something V2 构建 2800 个高质量三元组
    - 每个三元组配有 schema 约束的推理轨迹和困难干扰项
    - 筛选标准：至少满足两项——时间依赖、状态转换、镜头技巧、隐式因果、低词汇充分性
    - 推理轨迹生成遵循固定槽位顺序（actions → camera → states → scene → tempo），经人工审核校正

### 损失函数 / 训练策略

- **无训练**：整个方法是零样本的，不需要任何任务特定的微调
- 检索排名基于余弦相似度：$s(V) = \mathbf{q}(V_r, E)^\top \mathbf{v}(V)$
- 推理评估引入 LLM-as-a-judge（GPT-4o），在 10 个维度上评分（1-10），取算术平均为总体推理分

## 实验关键数据

### 主实验

**CoVR-R 基准上的零样本对比**

| 方法 | Backbone | R@1 | R@5 | R@10 | R@50 | 推理分 |
|------|----------|-----|-----|------|------|--------|
| CoVR-BLIP | BLIP | 30.30 | 51.07 | 57.05 | 73.82 | 4.85 |
| BSE-CoVR (CA) | BLIP | 37.90 | 57.67 | 64.48 | 79.47 | 6.42 |
| MVFT-JI† | BLIP | 34.40 | 54.15 | 62.30 | 77.40 | 6.28 |
| **Ours** | Qwen-VL | 44.32 | 61.91 | 67.33 | 79.90 | 7.46 |
| **Ours+R** | Qwen-VL | **49.88** | **66.99** | **72.97** | **85.14** | **8.31** |

R@1 较最强基线提升 **+11.98** 个百分点（31.6% 相对提升）。

**Dense-WebVid-CoVR 测试集**

| 方法 | R@1 | R@5 | R@10 | R@50 |
|------|-----|-----|------|------|
| BSE-CoVR (CA) | 48.08 | 73.36 | 81.06 | 93.78 |
| **Ours** | 58.19 | 80.50 | 86.92 | 97.14 |
| **Ours+R** | **61.21** | **83.40** | **89.39** | **97.61** |

R@1 提升 **+13.13** 个百分点，超越所有基线。

### 消融实验

**Token 聚合策略**

| 策略 | R@1 | R@5 | R@50 |
|------|-----|-----|------|
| Last token | 1.51 | 3.57 | 10.14 |
| Mean pooling | 44.87 | 63.67 | 82.44 |
| Max pooling | 35.95 | 52.02 | 93.98 |
| **Weighted (ours)** | **49.88** | **66.99** | **85.14** |

重要性加权池化比均值池化提升 +5.01 R@1。

**模型规模影响**

| 模型 | R@1 | 推理分 |
|------|-----|--------|
| Qwen3-VL-4B | 43.98 | 7.95 |
| Qwen3-VL-8B | 49.88 | 8.31 |
| Qwen3-VL-72B | 55.48 | 9.05 |

性能随模型规模一致提升，8B 是性价比最优选择。

### 关键发现

- 推理增强变体（+R）在 R@1 上比无推理版本提升 +5.56 个百分点，验证了显式后效应预测的价值
- 先前方法在 CoVR-R 上比在标准基准上表现更差（avg R@1 32.05% vs 40.66%），说明推理依赖型编辑构成了独特挑战
- 迭代细化推理（5 轮）仅带来边际收益（R@1: 49.88% → 50.56%），但推理成本增加 5 倍，单次推理为最终选择
- Qwen3 系列在相近参数量下始终优于 Qwen2.5 系列

## 亮点与洞察

- **推理优先范式**：将推理从检索的副产品提升为一等公民，显式预测编辑的"后效应"再进行检索，比端到端的特征融合更可解释
- **无需任务特定训练**：利用通用 LMM 的推理能力实现零样本 CoVR，减少了对标注数据的依赖
- **重要性加权池化**：简单却有效的无参数策略，通过下调功能词、上调语义丰富词的权重，优于所有复杂拼接方案
- **结构化推理记录**：五维度 schema 约束（states/actions/scene/camera/tempo）使推理可验证、可比较，有利于后续研究

## 局限与展望

- 依赖 Qwen3-VL 的视频理解能力，对低质量或极长视频可能效果下降
- Gallery 编码需对每个视频生成描述并提取嵌入，预处理成本较高
- 推理轨迹的质量受限于 LMM 的推理能力，某些微妙的因果链可能被遗漏
- 基准规模（2800 三元组）相对较小，领域覆盖有限
- 与端到端微调方法相比，零样本方法在标准基准上的优势能否在更大规模下保持有待验证

## 相关工作与启发

- 从 CIR（组合图像检索）到 CoVR 的推广引入了时间/因果维度，这是视频理解的核心
- 与 MVFT-JI、CoVR-BLIP 等训练型方法形成互补——推理型和训练型可结合使用
- 重要性加权池化的思路可推广到其他需要从 LMM 生成文本中提取语义嵌入的任务
- 零样本推理检索的范式可能扩展到其他模态（3D、音频等）的组合检索

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 推理优先的零样本 CoVR 框架新颖，基准设计有价值
- **实验充分度**: ⭐⭐⭐⭐ — 两个基准、多维消融、模型规模分析全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，推理记录的形式化定义规范
- **价值**: ⭐⭐⭐⭐ — 推动 CoVR 从关键词匹配向推理驱动转变

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](../../CVPR2025/multimodal_vlm/collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2026\] G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval](g_mixer_geodesic_mixup_based_implicit_semantic_expansion_for_zero_shot_cir.md)
- [\[ICML 2026\] Find, Fix, Reason: Context Repair for Video Reasoning](../../ICML2026/multimodal_vlm/find_fix_reason_context_repair_for_video_reasoning.md)

</div>

<!-- RELATED:END -->
