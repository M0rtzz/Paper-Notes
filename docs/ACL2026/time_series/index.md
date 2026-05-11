---
title: >-
  ACL2026 时间序列方向5篇论文解读
description: >-
  5篇ACL2026的时间序列方向论文解读，涵盖 LLM、时序预测、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**💬 ACL2026** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (6)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (37)](../../ICLR2026/time_series/index.md) · [🤖 AAAI2026 (35)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (58)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md) · [🧪 ICML2025 (26)](../../ICML2025/time_series/index.md)

🔥 **高频主题：** LLM ×2 · 时序预测 ×2

**[A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)**

:   提出FinLangNet框架，通过双模块架构（DeepFM处理静态特征 + 双粒度提示机制的Transformer处理时序行为）实现多尺度信用风险预测，在滴滴金融平台部署后实现KS提升6.3pp和坏账率下降9.9%。

**[Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)**

:   提出 SIVR 框架，通过计算 LLM 隐藏状态跨层的内部方差（广义方差、圆方差、token 熵）作为 token 级特征，用轻量 Transformer 编码器聚合全序列模式来估计不确定性/检测幻觉，显著优于基线且泛化更强。

**[STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)**

:   本文提出 STK-Adapter，通过在 LLM 每一层嵌入三个 MoE 模块（ST-MoE 捕捉时空结构、EA-MoE 建模事件链语义、CMA-MoE 深度跨模态对齐），解决现有方法将 TKG 嵌入与 LLM 浅层对齐导致的时空信息丢失和逐层稀释问题，在四个基准数据集上显著超越 SOTA。

**[Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study](temporal_leakage_in_search-engine_date-filtered_web_retrieval_a_retrospective_fo.md)**

:   本文对 Google 和 DuckDuckGo 的日期过滤器进行系统审计，发现搜索引擎日期过滤在回顾性预测评估中严重失效——71%（Google）和 81%（DuckDuckGo）的问题至少有一个页面包含重大截止日期后信息泄漏，导致预测 Brier 分数从 0.24 虚降至 0.10。

**[Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)**

:   定义 Time-RA 新任务将时间序列异常检测从二分类升级为生成式推理诊断（检测+分类+原因解释），构建首个包含约 4 万样本、10 个领域、20 种异常类型的多模态基准 RATs40K，并通过 AI 反馈标注流程和 LLM 微调验证了该范式的可行性。
