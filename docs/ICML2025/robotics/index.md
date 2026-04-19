---
title: >-
  ICML2025 具身智能方向 15篇论文解读
description: >-
  15篇ICML2025 具身智能方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🤖 具身智能

**🧪 ICML2025** · **15** 篇论文解读

**[BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly](biassemble_learning_collaborative_affordance_for_bimanual_geometric_assembly.md)**

:   提出 BiAssemble 框架，通过学习感知双臂协作的点级可供性（affordance），将几何装配任务分解为抓取→对齐→装配三步，在破碎物体重组任务上超越现有可供性和模仿学习方法，并在真实世界基准上验证。

**[Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)**

:   将 LLM 的自精炼规划过程建模为不动点问题（深度均衡模型），通过隐式微分实现端到端监督训练，无需额外验证器或 RL，并设计嵌套均衡求解实现闭环长程机器人规划。

**[CommVQ: Commutative Vector Quantization for KV Cache Compression](commvq_commutative_vector_quantization_for_kv_cache_compression.md)**

:   提出 CommVQ——通过可加向量量化压缩 KV cache，创新性地设计与 RoPE 可交换的码本并用 EM 算法训练，在 2-bit 下几乎无损、1-bit 下仍保持可用精度，使 LLaMA-3.1 8B 在单张 RTX 4090 上支持 128K 上下文。

**[Efficient Robotic Policy Learning via Latent Space Backward Planning](efficient_robotic_policy_learning_via_latent_space_backward_planning.md)**

:   提出潜在空间反向规划（LBP），从最终目标出发递归预测越来越接近当前状态的中间子目标，在保持任务对齐的同时大幅提升规划效率，在 LIBERO-LONG 仿真和真实机器人长时域任务上达到 SOTA。

**[FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)**

:   提出 FOUNDER 框架，通过学习映射函数将 Foundation Model (FM) 的多模态任务表示对齐到 World Model (WM) 的状态空间，结合时间距离预测器生成奖励信号，实现无需环境奖励的开放式多任务具身决策。

**[Geometric Contact Flows: Contactomorphisms for Dynamics and Control](geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)**

:   提出 Geometric Contact Flows (GCF)，利用黎曼几何和接触几何作为归纳偏置，通过接触微分同胚（contactomorphisms）将具有稳定性/能量守恒等期望性质的潜在接触哈密顿动力学映射到目标动力学，同时利用集成不确定性驱动测地线实现鲁棒泛化和避障。

**[Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)**

:   提出 Hi Robot，一个层次化 VLM 系统：高层 VLM 将复杂用户指令/反馈推理为原子命令，低层 VLA (π0) 执行动作，结合合成数据生成方案，在三类机器人平台上实现了远超 GPT-4o 和扁平 VLA 的开放式指令跟随能力。

**[Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors](internal_causal_mechanisms_robustly_predict_language_model_out-of-distribution_b.md)**

:   利用LLM内部已识别的因果机制来预测模型在分布外输入上的输出正确性，提出反事实模拟和值探测两种方法，在OOD设置中比现有基线平均AUC-ROC提升13.84%。

**[Learning to Stop: Deep Learning for Mean Field Optimal Stopping](learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)**

:   首次在离散时间有限状态空间下形式化并计算求解平均场最优停止（MFOS）问题，证明 MFOS 以 $O(1/N)$ 速率逼近多智能体最优停止（MAOS），并提出两种深度学习算法（直接法 DA 和动态规划法 DPP），在维度高达 300 的 6 个场景中验证有效性。

**[Machine Learning from Explanations](machine_learning_from_explanations.md)**

:   提出一种用简单解释信号（重要输入特征）引导机器学习的方法——通过交替优化预测准确率和注意力对齐的两阶段训练循环，在小数据、类不平衡、虚假特征场景下显著提升性能和稳定性。

**[PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](poisonbench_assessing_large_language_model_vulnerability_to_data_poisoning.md)**

:   提出 PoisonBench——首个系统评估 LLM 在偏好学习阶段面对数据投毒攻击脆弱性的基准，涵盖内容注入与对齐退化两类攻击，在 22 个模型上揭示了投毒比例与攻击效果的对数线性关系及欺骗性对齐的初步证据。

**[STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization](star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector.md)**

:   提出STAR框架，通过旋转增强残差技能量化（RaRSQ）解决VQ-VAE的codebook坍塌问题，并通过因果技能Transformer（CST）建模技能间依赖关系，在LIBERO基准上整体成功率达93.6%，比此前SOTA QueST提升约12%。

**[Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](synthesizing_images_on_perceptual_boundaries_of_anns_for_uncovering_and_manipula.md)**

:   提出 BAM（Boundary Alignment & Manipulation）框架，通过在 ANN 感知决策边界上采样生成图像刺激，系统性地揭示、预测和操控人类个体间的感知差异。

**[Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length](unable_to_forget_proactive_interference_reveals_working_memory_limits_in_llms_be.md)**

:   借鉴认知科学中的前摄干扰（Proactive Interference）范式，发现LLM的信息检索准确率随干扰信息量呈对数线性下降至零，揭示了一种独立于上下文长度的"工作记忆"容量瓶颈，且提示工程无法有效缓解。

**[X-Hacking: The Threat of Misguided AutoML](x_hacking_the_threat_of_misguided_automl.md)**

:   揭示了XAI(可解释AI)领域的新安全威胁"X-hacking"：通过AutoML的管道搜索能力，对抗者可在Rashomon模型集中寻找支持预定结论的解释性结果，Bayesian优化比随机搜索快3倍。
