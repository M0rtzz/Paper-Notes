---
title: >-
  ICML2026 AIGC 检测方向7篇论文解读
description: >-
  7篇ICML2026的 AIGC 检测方向论文解读，涵盖 LLM、对抗鲁棒、多模态、推理、机器人、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "AIGC 检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "多模态"
  - "推理"
  - "机器人"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🧪 ICML2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (16)](../../ACL2026/aigc_detection/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/aigc_detection/index.md) · [🔬 ICLR2026 (6)](../../ICLR2026/aigc_detection/index.md) · [🤖 AAAI2026 (2)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (9)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md)

🔥 **高频主题：** LLM ×3 · 对抗鲁棒 ×2

**[AutoBaxBuilder: Bootstrapping Code Security Benchmarking](autobaxbuilder_bootstrapping_code_security_benchmarking.md)**

:   AUTOBAXBUILDER用LLM代理流水线自动生成Web后端安全评测场景、功能测试和端到端安全测试，把人工构建BAXBENCH式任务的成本降低约12倍，并构建出40个新场景的AUTOBAXBENCH来评估当代代码模型的正确性与安全性差距。

**[Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)**

:   SurpMark 把"AI 文本检测"重构成似然无关假设检验：用代理 LM 算 token surprisal 后 k-means 离散成 k 个状态，估计一阶 Markov 转移矩阵，再用广义 Jensen-Shannon 散度（GJS）和预先建好的"人写 / 机写"参考转移矩阵比较，单次前向就给出黑盒、无需重训、无需 per-instance 重采样的判别分数。

**[CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection](core_conflict-oriented_reasoning_for_general_multimodal_manipulation_detection.md)**

:   作者把"多模态假新闻检测"重新定义为"显式捕获模态间或与世界知识之间的冲突"任务，构建了带细粒度冲突标注的 14k 语料 CAC，并提出 CORE 框架通过冲突感知训练（CPT）重塑 MLLM 的概念边界，使其在 DGM4、MDSM、MMFakeBench、NewsCLIPpings 四个数据集上以 100–750 个样本就大幅超过专用 SOTA。

**[Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook](distributional_open-ended_evaluation_of_llm_cultural_value_alignment_based_on_va.md)**

:   DOVE 用率失真变分优化从 1 万篇人类文本中自动构造紧凑的"价值码本"，再用不平衡最优传输度量人类与 LLM 长文本在价值空间上的分布差异，从而在 12 个 LLM 上把"评测—下游任务"相关性从基线 ≤24% 拉到 31.56%。

**[Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)**

:   本文在「单阈值固定协议」下系统暴露 AI 文本检测器在跨数据集/跨生成器 shift 下的脆弱性，并提出把可学注意力加权的手工语言特征与 transformer [CLS] 表征融合，配合 DeBERTa-v3 backbone，在 M4 多域多生成器基准上达到 85.9% balanced accuracy，比强 zero-shot 基线（Fast-DetectGPT、RADAR、Log-Rank）高最多 +7.22。

**[Generating Robust Portfolios of Optimization Models using Large Language Models](generating_robust_portfolios_of_optimization_models_using_large_language_models.md)**

:   本文提出一个轻量、无需训练的算法：用同一个 LLM 同时扮演"随机生成器"和"打分评审"两个角色，把生成概率前缀和达到 $1-\alpha$ 的候选优化模型打包成 portfolio，从理论上证明只要"生成器"或"评审"任一与人类偏好对齐，portfolio 就一定包含高质量优化模型，并在 NL4LP 上用 GPT 验证 portfolio 在最差情况下也稳定优于随机采样。

**[On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)**

:   针对零样本 AI 生成文本检测里"高频 boilerplate 稀释信号"和"单点概率脆弱"两大痼疾，作者提出 Uncertainty / Uncertainty++ 检测器：只在每段文本底部 $\rho$ 分位的低概率 token 上聚合 log-prob，并叠加同一组位置上的 Rényi 熵作为分布形状信号，再在 12 个生成器、7 个数据集上把平均 AUROC 从 Lastde 的 86.49 推到 88.74，且在改写 / 改解码这类扰动下显著更稳。
