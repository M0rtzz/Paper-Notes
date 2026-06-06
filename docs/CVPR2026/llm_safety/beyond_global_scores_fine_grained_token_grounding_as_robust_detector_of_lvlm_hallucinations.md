---
title: >-
  [论文解读] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations
description: >-
  [CVPR 2026][LLM安全][hallucination detection] 提出基于 patch 级别的 LVLM 幻觉检测框架，发现幻觉 token 表现出弥散注意力模式和低语义对齐两个特征标志，据此设计注意力弥散分数（ADS）和跨模态接地一致性（CGC）两个轻量指标，检测准确率达 90%。
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "hallucination detection"
  - "LVLM"
  - "注意力机制"
  - "patch-level grounding"
  - "token-level"
---

# Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations

**会议**: CVPR 2026  
**arXiv**: [2604.04863](https://arxiv.org/abs/2604.04863)  
**代码**: 有  
**领域**: LLM安全  
**关键词**: hallucination detection, LVLM, attention dispersion, patch-level grounding, token-level

## 一句话总结

提出基于 patch 级别的 LVLM 幻觉检测框架，发现幻觉 token 表现出弥散注意力模式和低语义对齐两个特征标志，据此设计注意力弥散分数（ADS）和跨模态接地一致性（CGC）两个轻量指标，检测准确率达 90%。

## 研究背景与动机

LVLMs 易产生视觉幻觉——描述图像中不存在的目标或属性。现有检测方法依赖粗粒度全图统计（如聚合整体注意力、输出概率、全局嵌入相似度），这种全局策略存在根本局限：幻觉 token 可能与图像多个局部区域有弱但分散的相关性，聚合后产生欺骗性的高相关度，逃过全局检测器。

核心洞察：忠实的目标 token 必须强烈接地到特定图像区域。因此幻觉检测必须从全图分析转向 patch 级别的接地分析。

## 方法详解

### 整体框架

分析生成 token 与图像 patch 之间的细粒度交互，提取两类结构性特征，训练轻量分类器进行 token 级幻觉检测。

### 关键设计

1. **注意力弥散分数 (ADS)**：量化目标 token 的注意力空间分布紧凑度。保留 top-k% 注意力激活，用 8 连通组件过滤注意力沉没（面积 $< \tau_{ADS}$ 的小 blob 被抑制），计算前景 blob 质量 $m_t^{(n)} = \sum_{c \in \mathcal{C}_t^{(n)*}} \sum_{p \in c} \bar{\mathbf{A}}_t^{(n)}(p)$ 和背景归一化熵 $\hat{H}_t^{(n)} = -\sum_{p \in \mathcal{B}} \mathbf{E}(p) \log \mathbf{E}(p) / \log|\mathcal{P}|$，最终 $ADS_t^{(n)} = (1-m_t^{(n)}) \cdot \hat{H}_t^{(n)}$。低 ADS = 紧凑聚焦（真实目标），高 ADS = 分散弥漫（幻觉）。中间层分离度最强，深层因语言先验主导而差距收敛。

2. **跨模态接地一致性 (CGC)**：逐层计算 token 嵌入与各图像 patch 嵌入的余弦相似度，取 top-k patch 相似度均值。真实 token 与对应区域有尖锐的相似度峰值，幻觉 token 与所有区域相似度都低且弥散。

3. **逐层特征拼接分类**：将所有层的 ADS 和 CGC 拼接为特征向量，训练 XGB/MLP/随机森林分类器。中间层分离度最强（语言先验在深层增强导致差异收敛）。

### 损失函数 / 训练策略

分类器使用交叉熵训练。标签来自 GPT-4o 的语义验证（结合 CHAIR 指标和人工描述）。在 MS-COCO 2014 验证集的 4000 张图像上实验，90/10 划分。

## 实验关键数据

### 主实验（图像字幕任务）

| 方法 | LLaVA-1.5-7B F1 | Qwen2.5-VL-7B F1 | InternVL2.5-8B F1 |
|------|----------------|------------------|-------------------|
| MetaToken | 0.51 | 0.54 | — |
| SVAR | — | — | — |
| Ours (XGB) | **0.90** | **0.88** | **0.88** |

### 关键发现

- 真实 token 在早/中层展现紧凑、定位良好的注意力，幻觉 token 注意力弥散
- 真实 token 与对应 patch 有高语义相似度，幻觉 token 与所有 patch 相似度低
- 两个发现共同指出：幻觉主要源于语言先验的过度依赖而非视觉编码器缺陷
- 中间层特征最具判别力，深层因语言先验主导而差距收敛
- ADS 单独作为分类器即可达到 F1=0.73-0.77，结合 CGC 后达到 F1=0.88-0.90
- 实验在 MS-COCO 2014 验证集 4000 张图像上进行，90/10 划分
- 标签由 GPT-4o 的语义验证结合 CHAIR 指标确定

## 亮点与洞察

- "从全局到局部"的视角转换抓住了幻觉检测的关键
- ADS 的连通组件过滤和背景熵设计优雅地处理了注意力沉没现象
- 可解释性强——可以可视化注意力热图来解释检测结果
- 轻量分类器即可达到 90% 准确率

## 局限与展望

- 依赖模型内部注意力权重，需要白盒访问
- 分类器需要针对每个 LVLM 单独训练
- 仅在目标级幻觉上验证，属性幻觉未涉及
- 使用 XGB/MLP/随机森林等轻量分类器，可解释性强但对复杂模式的捕获能力有限
- CGC 通过逐层计算 token 嵌入与 patch 嵌入的余弦相似度，取 top-k patch 均值，真实 token 呈现尖锐峰值而幻觉 token 弥散且低
- 对现有 SVAR 等全局统计方法的根本局限性分析深刻：弥散但广泛的弱相关聚合后产生欺骗性高的全局分数

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — patch 级幻觉分析的开创性工作
- 技术深度：⭐⭐⭐⭐ — ADS 和 CGC 设计精巧
- 实验充分度：⭐⭐⭐⭐ — 多模型多基准验证
- 实用价值：⭐⭐⭐⭐ — 轻量可解释的幻觉检测器

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](../../ACL2026/llm_safety/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/llm_safety/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](../../ACL2026/llm_safety/finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)

</div>

<!-- RELATED:END -->
