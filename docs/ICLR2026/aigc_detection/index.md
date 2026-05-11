---
title: >-
  ICLR2026 AIGC 检测方向6篇论文解读
description: >-
  6篇ICLR2026的 AIGC 检测方向论文解读，涵盖对抗鲁棒、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔎 AIGC 检测

**🔬 ICLR2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (9)](../../ACL2026/aigc_detection/index.md) · [📷 CVPR2026 (1)](../../CVPR2026/aigc_detection/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/aigc_detection/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/aigc_detection/index.md) · [💬 ACL2025 (15)](../../ACL2025/aigc_detection/index.md) · [📷 CVPR2025 (3)](../../CVPR2025/aigc_detection/index.md)

**[Calibrating Verbalized Confidence with Self-Generated Distractors](calibrating_verbalized_confidence_with_self-generated_distractors.md)**

:   提出 DiNCo 方法，通过让 LLM **独立**评估自动生成的干扰选项（合理但错误的替代答案）来暴露其"暗示性偏差"，用干扰项上的总置信度进行归一化，并融合生成一致性与验证一致性两个互补维度，在短文本 QA 和长文本生成任务上显著改善置信度校准。

**[CLARC: C/C++ Benchmark for Robust Code Search](clarc_cc_benchmark_for_robust_code_search.md)**

:   构建首个可编译的 C/C++ 代码检索基准 CLARC（6717 查询-代码对），自动化 pipeline 从 GitHub 提取代码并用 LLM+假设检验生成/验证查询；覆盖标准/匿名化/汇编/WebAssembly 四种检索场景，揭示现有代码嵌入模型过度依赖词汇特征（匿名化后 NDCG@10 从 0.89 降至 0.67）且在二进制级别检索上严重不足。

**[Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)**

:   通过 26 位专业作家对 8618 条表达的 close reading 标注，揭示 n-gram 新颖度不足以衡量文本创造力——约 91% 的高 n-gram 新颖表达并不被认为具有创造性，且开源 LLM 中高 n-gram 新颖度与低语用合理性负相关。

**[DMAP: A Distribution Map for Text](dmap_a_distribution_map_for_text.md)**

:   提出 DMAP（Distribution Map），一种将文本经由语言模型的 next-token 概率排序映射为 $[0,1]$ 区间上 i.i.d. 样本的数学框架，理论证明纯采样文本产生均匀分布，由此可用 $\chi^2$ 检验验证生成参数、揭示概率曲率类检测器在纯采样下彻底失效的根本原因，并可视化后训练（SFT/RLHF）在下游模型中留下的统计指纹。

**[Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)**

:   构建了迄今最大的 AI 生成同行评审数据集（788,984 篇评审），系统评估了 18 种 AI 文本检测方法在同行评审场景下的表现，并提出了利用论文原文作为上下文的 Anchor 检测方法，在低误报率下大幅超越所有基线。

**[PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)**

:   基于欧洲议会13年（2009-2022）共2225条高质量审议记录构建PoliCon基准，通过设计不同投票机制（简单多数/2/3多数/否决权）、权力结构和政治目标（功利主义/罗尔斯主义），系统评估LLM起草政治共识决议的能力，揭示前沿模型在复杂共识任务中的不足及固有党派偏见。
