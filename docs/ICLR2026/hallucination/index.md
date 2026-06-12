---
title: >-
  ICLR2026 幻觉检测论文汇总 · 9篇论文解读
description: >-
  9篇ICLR2026的幻觉检测方向论文解读，涵盖 LLM、多模态、RAG、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "多模态"
  - "RAG"
  - "对抗鲁棒"
item_list:
  - u: "copy-paste_to_mitigate_large_language_model_hallucinations/"
    t: "Copy-Paste to Mitigate Large Language Model Hallucinations"
  - u: "dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis/"
    t: "Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models"
  - u: "enhancing_hallucination_detection_through_noise_injection/"
    t: "Enhancing Hallucination Detection through Noise Injection"
  - u: "hallucination_begins_where_saliency_drops/"
    t: "Hallucination Begins Where Saliency Drops"
  - u: "look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model/"
    t: "Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation"
  - u: "lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals/"
    t: "LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals"
  - u: "shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de/"
    t: "SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense"
  - u: "token-guard_towards_token-level_hallucination_control_via_self-checking_decoding/"
    t: "Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding"
  - u: "veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes/"
    t: "VeriTrail: Closed-Domain Hallucination Detection with Traceability"
item_total: 9
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🔬 ICLR2026** · **9** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (19)](../../ICML2026/hallucination/index.md) · [💬 ACL2026 (27)](../../ACL2026/hallucination/index.md) · [📷 CVPR2026 (18)](../../CVPR2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** LLM ×2 · 多模态 ×2

**[Copy-Paste to Mitigate Large Language Model Hallucinations](copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)**

:   提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

**[Enhancing Hallucination Detection through Noise Injection](enhancing_hallucination_detection_through_noise_injection.md)**

:   在 LLM 中间层的 MLP 激活中注入均匀噪声来近似贝叶斯后验，捕获认知不确定性（epistemic uncertainty），与采样温度捕获的偶然不确定性（aleatoric uncertainty）互补，将 GSM8K 上的幻觉检测 AUROC 从 71.56 提升到 76.14。

**[Hallucination Begins Where Saliency Drops](hallucination_begins_where_saliency_drops.md)**

:   提出 LVLMs-Saliency 梯度感知诊断框架来量化每个输出 token 的视觉锚定强度，发现"当先前输出 token 对下一个 token 预测的显著性降低时，幻觉就会产生"的关键规律，并基于此设计了 SGRS（显著性引导的拒绝采样）+ LocoRE（局部一致性增强）双机制推理时框架，在多个 LVLM 上显著降低幻觉率。

**[Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)**

:   提出 AIR（Adaptive vIsual Reinforcement）框架，通过原型距离的 token 精简 + 最优传输引导的 patch 选择性增强，在推理时无训练地减少 MLLM 幻觉（LLaVA-1.5-7B CHAIR_S: 22→18.4，POPE 准确率 +5.3%），同时保持多模态通用能力。

**[LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)**

:   提出 Lumina 框架，通过"上下文-知识信号"检测RAG系统中的幻觉：用MMD度量**外部上下文利用**程度，用跨层token预测演化度量**内部知识利用**程度，无需超参调优即可泛化。

**[SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)**

:   首次将LVLM对象幻觉系统性追溯到视觉编码器，识别出统计偏差（高频模式token过度强调）、固有偏差（预训练主导对象的残余表示）、脆弱性（微小扰动即导致特征失真）三大问题，并提出SHIELD——一个完全免训练的框架，通过token重加权、token减法和对比解码三策略协同防御，在LLaVA-1.5/InstructBLIP/Qwen-VL上全面超越VCD和OPERA等方法。

**[Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)**

:   提出 Token-Guard，一种基于自检验解码的 token 级幻觉控制方法，通过隐空间中的 token 级/段级评分和迭代修正机制，在解码过程中检测并抑制幻觉生成，F1 平均提升 16.3%。

**[VeriTrail: Closed-Domain Hallucination Detection with Traceability](veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)**

:   提出 VeriTrail——首个为多步生成过程（MGS）提供可追溯性的闭域幻觉检测方法，建模生成过程为 DAG 并沿路径逐层验证，同时构建了首批包含所有中间输出和人工标注的 MGS 数据集。
