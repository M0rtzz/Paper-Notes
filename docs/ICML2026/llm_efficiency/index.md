---
title: >-
  ICML2026 LLM 效率方向11篇论文解读
description: >-
  11篇ICML2026的 LLM 效率方向论文解读，涵盖 LLM、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM 效率"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**🧪 ICML2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (20)](../../ACL2026/llm_efficiency/index.md) · [📷 CVPR2026 (4)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md)

🔥 **高频主题：** LLM ×5

**[A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)**

:   本文建立首个显式纳入 KV 缓存显存动态的 LLM 推理排队模型，给出闭形稳定性条件 $\lambda < \mu(1-\delta)$，使运维人员可直接计算所需 GPU 数；在单 GPU、8 GPU 集群与 LongBench 真实数据上验证误差均 $\leq 10\%$。

**[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)**

:   本文指出多轮对话场景下传统 Prefill-Decode 分离架构因每轮都要 P→D 重算并传输 KV 而严重低效，提出 PPD（Prefill-capable Decode）动态路由系统，让 decode 节点根据 SLO 权重决定是否本地处理 Turn 2+ 的 append-prefill，把 Turn 2+ TTFT 降低约 68%。

**[OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)**

:   OServe 把 LLM 服务的「资源分配 + 并行策略 + 请求路由」联合建模为流网络上的双层最大流问题，配合 LSTM 工作负载预测和基于 GPU 互联的 ad hoc 模型切换，应对真实流量在空间（不同请求类型）和时间（成分随时刻变化）两个维度的异质性，端到端 P99 延迟和吞吐相比 vLLM 平均提升 1.5×、最大 2×。

**[PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)**

:   本文提出 PipeSD：把投机解码（speculative decoding）从端云顺序执行改成 token-batch 流水线，并用双阈值 NAV 触发 + 贝叶斯自动调参替代固定 draft 长度，在 5G 带宽的真实端云测试床上拿到 1.16×–2.16× 加速、14–25% 云端能耗下降。

**[Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)**

:   本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。

**[Scout: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States](scout_active_information_foraging_for_long-text_understanding_with_decoupled_epi.md)**

:   Scout 把百万级 token 的长文本理解重新建模为"主动信息觅食"过程，引入与交互轨迹解耦的、带来源锚点的 epistemic state $\mathcal{E}_t$ 作为唯一推理底座，并通过 gap-diagnosed 自评估迭代收缩到查询充分子集，在 LooGLE-v2 和 $\infty$Bench 上既追平甚至超过 Gemini-3-Pro 等前沿模型，又把 token 成本降低到约 $1/8$。

**[SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)**

:   SLAY 把受物理"逆平方相互作用"启发的 Yat-kernel 通过 (1) 球面归一化 (2) Bernstein 定理的 Laplace 积分表示 (3) Gauss-Laguerre 求积 (4) 多项式+指数核张量积正随机特征四步连击线性化，得到 $O(L)$ 时间复杂度且与 softmax 几乎无差异的注意力机制。

**[Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)**

:   本文为新兴的 Attention-FFN 解耦 (AFD) 推理架构提供首个理论框架,基于"prefill 长度有限均值 + decode 长度服从几何分布"的概率工作负载模型,推导出 rA-1F 拓扑下最优 A/F 比的闭式解 $r^*=\max\{r_A, r_C, r_{\text{peak}}\}$,并用 trace-calibrated 模拟器验证理论与实测最优值偏差 <10%。

**[Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)**

:   作者搭了一套基于 NVML 的分阶段 GPU 能耗采集框架，把蒸馏流水线拆成"教师侧 + 学生侧 + 评估"逐段计量，发现一次性运行时教师 logit 缓存 / 合成数据生成才是大头，让 KD 和 synthetic SFT 在 1B–13B OLMo-2 学生上反而比直接 SFT 多耗约 $2.4\times$ 能量，并给出闭式 break-even 公式说明只有当教师产物被复用 $N^*$ 次以上时蒸馏才真"省电"。

**[Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)**

:   本文提出一套训练与推理共享完全相同的分段前向执行语义的长上下文 LLM 框架：跨段只保留固定长度的可微分 KV 尾部 + 一条仅前向的检索旁路，在 LLaMA2-7B 32K/80K 上以约 $6\times$ 更低的 prefill 峰值显存达到与全注意力可比甚至更好的 LongBench/RULER 表现。

**[Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)**

:   本文把现代 LLM 长上下文推理中的稀疏注意力、RAG、压缩上下文记忆等优化统一为四阶段 "Prepare Memory → Compute Relevancy → Retrieval → Apply to Inference" 内存处理流水线，定量证明该流水线占整体延迟 22%-97% 且各阶段计算特性高度异构，并据此提出 GPU-FPGA 异构系统：把规则/算密集操作留 GPU、把稀疏/不规则/访存密集操作 offload 到 FPGA，在 MI210 + Alveo U55C 上取得最多 2.2× 端到端加速和 4.7× 能耗下降。
