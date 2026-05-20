---
title: >-
  ICML2026 机器人/具身智能方向12篇论文解读
description: >-
  12篇ICML2026的机器人/具身智能方向论文解读，涵盖机器人、多模态、导航、推理、Agent、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "机器人/具身智能"
  - "论文解读"
  - "论文笔记"
  - "机器人"
  - "多模态"
  - "导航"
  - "推理"
  - "Agent"
  - "扩散模型"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 机器人/具身智能

**🧪 ICML2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (6)](../../ACL2026/robotics/index.md) · [📷 CVPR2026 (38)](../../CVPR2026/robotics/index.md) · [🔬 ICLR2026 (47)](../../ICLR2026/robotics/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/robotics/index.md) · [🧠 NeurIPS2025 (55)](../../NeurIPS2025/robotics/index.md) · [📹 ICCV2025 (26)](../../ICCV2025/robotics/index.md)

🔥 **高频主题：** 机器人 ×3 · 多模态 ×3 · 导航 ×3 · 推理 ×2 · Agent ×2

**[Decompose and Recompose: Reasoning New Skills from Existing Abilities for Cross-Task Robotic Manipulation](decompose_and_recompose_reasoning_new_skills_from_existing_abilities_for_cross-t.md)**

:   针对"训练任务到全新任务"的零样本机器人操作，作者把 demo 拆成"原子技能-动作对"作为中间表示，再用 dual-library（动态库按视觉/计划相似度检索 + 静态库按 IDF 加权补全缺失技能 token）给 LLM 提供 skill-comprehensive in-context demonstrations，从而把"模仿轨迹"升级为"组合技能推理"。

**[Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)**

:   本文提出 CAPS：把"指令漂移"重新解释为系统性采样误差，用 SNR（=$\log|\mathcal{A}|-\mathcal{H}$）作为元认知开关，仅在高熵"Pivotal Window"触发基于幂分布 $\pi\propto p^\alpha$ 的 Metropolis-Hastings 迭代精修，在 RoboTwin、Simpler-WindowX、Libero-long 上 training-free 超越 OpenVLA 和 TACO。

**[Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models](embodied_interpretability_linking_causal_understanding_to_generalization_in_visi.md)**

:   本文把「视觉—动作归因」重新表述为干预估计问题，提出 ISS（介入显著性分数）和 NMR（干扰物质量比）两个指标，用 Bernoulli 掩码 + 高斯模糊扰动 + Action MSE 代理 KL 散度的方式量化 VLA 策略到底依赖哪些视觉区域，并证明 NMR 与 OOD 任务成功率呈 $r = -0.77$ 的强负相关——是预测 VLA 泛化能力的便宜诊断工具。

**[From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation](from_imagined_futures_to_executable_actions_mixture_of_latent_actions_for_robot_.md)**

:   MoLA 用一组在大规模机器人数据上预训练好的"模态感知逆动力学模型 (IDM)"，把视频生成模型预测出的未来帧翻译成语义/深度/光流三路离散潜动作，再让策略头基于这些动作中心的表征做控制，从而在 CALVIN、LIBERO、LIBERO-Plus 以及真实 UR5e 上把"想象-执行"接口做得既稳又准。

**[HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)**

:   HDFlow 用扩散模型生成稀疏战略子目标、用整流流生成稠密轨迹，再叠加能量引导和流形投影，构建一套快慢分工的双层规划器，把家具组装等长程稀疏奖励任务的成功率拉高 20~30 个百分点。

**[Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)**

:   LaRA-VLA 把 VLA 模型里的文本 CoT 和视觉 CoT 全部内化为连续 latent，通过三阶段 curriculum 训练（显式 CoT → latent 替换 → 动作专家适配）让推理留在 latent 空间里完成，推理延迟相比显式 CoT 降低高达 90%，控制频率重回实时区间。

**[Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering](mitigating_error_accumulation_in_continuous_navigation_via_memory-augmented_kalm.md)**

:   把无人机连续 VLN 的 step-by-step 预测重写成"递归贝叶斯估计 = GRU 先验 + 记忆库似然 + 可学习卡尔曼增益"的闭环, 在 TravelUAV 上仅用 10% 数据微调就把 L1-Full 的 SR 从 17.6% 推到 25.9%, 同时把 100 步后还在不断累积的位置漂移压平到 30–40 米。

**[Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)**

:   本文把匿名多机器人路径规划（MAPF）证明为一类**马尔可夫多边际最优传输（MMOT）**，从而把原本 $K^{T+1}$ 维的传输张量压缩成多项式规模 LP（P1），并通过全单模性保证最优解整数性；再把它推广为 Schrödinger bridge 得到 Sinkhorn 风格 entropic 松弛 P2 产出"影子传输"，最后在影子上做剪枝并解 LP（P3）恢复整数解，在 $K^{1.15}$ 复杂度下实现 3.6×–7.1× 加速、代价差距 <10%。

**[Plan in Sandbox, Navigate in Open Worlds: Learning Physics-Grounded Abstracted Experience for Embodied Navigation](plan_in_sandbox_navigate_in_open_worlds_learning_physics-grounded_abstracted_exp.md)**

:   本文提出 SAGE：在物理约束的语义沙盒里自动合成大量导航任务+IF-THEN 经验规则，用混合提示采样 + 非对称自适应裁剪的 GRPO 把这些经验蒸馏进 VLM 策略，最终在 A-EQA 上把 LLM-Match 成功率从 43.5% 拉到 53.2%（2B）/ 60.2%（4B），并能迁移到真实室内机器人。

**[Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)**

:   本文提出 PLMD：把 BEV 语义图与障碍图合并成 Label Map，用 DDPM 在障碍先验调制下补全未探索区域的语义+障碍标签，作为即插即用模块挂在任意 GON 策略上，在 ON / IIN / MRON 三类任务的 HM3D/MP3D 上一致刷新 SOTA。

**[Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](seeing_realism_from_simulation_efficient_video_transfer_for_vision-language-acti.md)**

:   针对 VLA（vision-language-action）模型在简单扰动下性能崩塌的问题，本文用"提取语义/几何条件 → 改写 caption → 条件视频扩散重渲染"的视频迁移流水线给仿真数据补上视觉与环境多样性，同时配以三段式 velocity caching 把生成时间砍掉 61% 以及 difficulty + diversity 双驱动的 coreset 采样仅选 10% 关键轨迹，最终在 Robotwin 2.0、LIBERO-Plus 和真机上让 RDT-1B / $\pi_0$ 涨 5–15%。

**[STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction](step_warm-started_visuomotor_policies_with_spatiotemporal_consistency_prediction.md)**

:   STEP 给 diffusion policy 接了一个轻量的 "前一段历史动作 + 当前观测 → 下一段动作"的 Transformer 预测器, 用它的输出作为去噪起点 (warm-start), 把 100 步去噪压到 2 步, 又附带一个 "动作变化太小就注一点噪声"的执行死锁防御机制, 在 9 个仿真任务和 2 个真机任务上比 BRIDGER / DDIM 平均提 21.6% / 27.5% 成功率。
