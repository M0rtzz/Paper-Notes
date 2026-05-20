---
title: >-
  ICML2026 时间序列方向10篇论文解读
description: >-
  10篇ICML2026的时间序列方向论文解读，涵盖时序预测、推理、对抗鲁棒、对齐/RLHF、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "推理"
  - "对抗鲁棒"
  - "对齐/RLHF"
  - "问答"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧪 ICML2026** · **10** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (6)](../../ACL2026/time_series/index.md) · [📷 CVPR2026 (6)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/time_series/index.md) · [🤖 AAAI2026 (35)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (57)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×8 · 推理 ×2

**[CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models](combinationts_a_modular_framework_for_understanding_time-series_forecasting_mode.md)**

:   CombinationTS 把时序预测模型解耦为 Input Transformation / Embedding / Encoder / Decoder / Output Transformation 五个正交模块，在共享的"评估条件空间"上做配对蒙特卡洛采样，用边际性能 $\mu$ 和稳定性 $\sigma$ 取代脆弱的单点 MSE，结论是：一旦数据视图（Embedding）设计得好，参数无关的 Identity Encoder 就能打平甚至超过复杂 Transformer，时序预测领域的"SOTA 增益"很大程度上来自看数据的方式而不是建模能力。

**[DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)**

:   针对"未来协变量已知"的时间序列预测 (TSF-X), DAG 设计了一个双通路网络: 一条沿时间维捕获"历史外生→未来外生"的注意力模式并注入到"历史内生→未来内生"的预测里, 另一条沿通道维捕获"历史外生→历史内生"的模式并注入到"未来外生→未来内生"的预测里, 在 12 个公开/新发布 TSF-X 数据集上 10/10 拿下 MSE 最佳, 显著超过 TimeXer、TFT、TiDE、CrossLinear、PatchTST 等。

**[Doubly Outlier-Robust Online Infinite Hidden Markov Model](doubly_outlier-robust_online_infinite_hidden_markov_model.md)**

:   本文提出 BR-iHMM：把"鲁棒观测更新（WoLF）"与"批量化状态推断（degenerate sticky HDP prior）"结合起来，给在线无限隐马模型同时在观测空间和状态空间提供有界的 Posterior Influence Function（PIF），在金融订单簿、电力负荷、合成回归三类含异常值的流式数据上把一步预测 RMSE 最多降低 67%。

**[Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)**

:   Fern 把长期时间序列预测重新表述为「从固定高斯源到数据相关椭球的最优传输」，借助 Brenier 定理把搜索空间限制在 SPD（对称正半定）类 Jacobian 上，用 Householder 反射的低秩谱分解把代价从 $O(n^3)$ 压到 $O(Rn)$，并在非平稳冲击场景下相对 DLinear / Koopa 等基线取得最多 790× 的稳定性提升。

**[FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)**

:   本文把 HiPPO 框架背后的概率测度推广到带可调奇异指数 $\alpha$ 的分数阶幂律测度，从而首次同时拿到「全历史保留 + 近时敏感 + 尺度不变」，并将这一理论落地为 LTI 对角化 SSM——FRACTAL 在 Long Range Arena 上以 87.11% 平均分追平 S5，并在 ListOps 上拿到 61.85%。

**[From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)**

:   作者发现现有 TSF 模型即使预测精度高，其潜空间也常常是"时间错乱"的（Latent Chaos）；他们提出 LatentTSF——先用 AutoEncoder 把观察压到一个高维潜状态空间，然后让任何主流 backbone 在这个空间内做未来预测（Pred + Align 双损失），最后再解码回观察空间——在 6 个标准 benchmark 上稳定降 MSE/MAE，并恢复了潜表征的时间局部性和频谱结构。

**[HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)**

:   给每个特征学一个"身份嵌入"作为持久语义锚点，配合时间-特征双螺旋注意力，在 5 个公开多变量时序数据集 21 个缺失场景上全部拿下第一，比次优的 ImputeFormer 在 ETT-h1 等数据集上多 25% 以上的 MAE 降幅。

**[PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)**

:   针对时间序列问答 (TSQA)，PATRA 在表征端把序列显式拆成 full / trend / season 三类模式，并通过三组可学习对齐 token 与文本做深度交叉对齐；在训练端用 SFT + GRPO 两阶段强化学习，把判别式与生成式任务的奖励统一映射到 $[0,2]$ 解决难度失衡，从而在四类 TSQA 任务上全面超越文本 LLM、ChatTS 等多模态时序 LLM。

**[Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)**

:   作者用 Allen 时间区间代数提出 PRO-DYN 命名法，把任意时序预测模型拆成"前处理 PRO → 动力学 DYN → 后处理 PRO"三段，发现两条经验规律：(i) DYN 必须**可学习且完整**才能打过 LTSF-Linear，(ii) DYN 必须放在**整个流程末端**（PRE-DYN 配置）才能吃到长 lookback 的红利；并通过给 Informer/FEDformer/MICN/FiLM 加一个线性 DYN 层让性能稳定提升，给 iTransformer/PatchTST/Crossformer 把 DYN 挪到前端则性能下降，用实验验证两条规律。

**[TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)**

:   TSRBench 构造了一个覆盖 14 个领域、4 大维度（感知/推理/预测/决策）、15 个任务、4125 道题、同时支持文本/可视化/文本+图/嵌入四种模态输入的时间序列推理基准，系统评测 30+ 主流 LLM、VLM 与 TSLLM，揭示出"scaling 在感知/推理上仍成立但在预测上失效"以及"文本与可视化模态高度互补但当前模型几乎无法融合"等关键结论。
