---
title: >-
  CVPR2025 LLM 推理方向6篇论文解读
description: >-
  6篇CVPR2025的 LLM 推理方向论文解读，涵盖推理、多模态、少样本学习、目标检测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**📷 CVPR2025** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (37)](../../ACL2026/llm_reasoning/index.md) · [📷 CVPR2026 (12)](../../CVPR2026/llm_reasoning/index.md) · [🔬 ICLR2026 (65)](../../ICLR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (66)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×6

**[Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)**

:   Argus 提出了一种grounded visual CoT机制，通过让MLLM先预测与问题相关的bounding box（RoI），然后重新采样/编码该区域的视觉token作为推理上下文，实现了显式的目标导向视觉注意力，在7B/8B级MLLM中取得视觉推理和目标grounding双料SOTA。

**[CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)**

:   提出 CoT-VLA，将视觉思维链推理引入视觉-语言-动作模型，通过两阶段推理——先预测子目标图像再生成动作序列——结合混合注意力和动作分块策略，在 LIBERO 基准上实现 81.13% 平均成功率，显著超越现有方法。

**[Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)**

:   提出交错模态思维链（ICoT），在推理步骤中穿插图像区域 crop 作为视觉 rationale，通过无参数的 Attention-driven Selection（ADS）从输入图像中智能选取关键区域插入生成序列，在 Chameleon 和 Qwen2-VL 上相比现有多模态 CoT 提升高达 14%。

**[Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval](osrcir_reflective_cot.md)**

:   本文提出OSrCIR，一种免训练的单阶段零样本组合图像检索方法，利用多模态大语言模型直接处理参考图像和修改文本，并通过反思式链式思维推理准确理解用户隐含意图，在多个基准上比现有免训练方法提升1.80%~6.44%。

**[Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)**

:   提出 Chain-of-Thought 引导的风格演化方法（CGSE），通过词→短语→句子三级渐进式风格描述生成，结合特征解耦和类别原型聚类，在五种恶劣天气场景和 Real-to-Art 基准上实现了显著的域泛化检测性能提升。

**[VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection](videoespresso_a_large-scale_chain-of-thought_dataset_for_fine-grained_video_reas.md)**

:   VideoEspresso 构建了一个20万+的大规模视频CoT推理数据集（包含空间bounding box和时间grounding标注），并提出VideoQA-SC混合框架——用1.5B轻量级模型选择平均2.36个核心帧，再用8B推理模型进行两阶段证据提取+答案生成，以仅1.8%的帧数和14.7%的计算量超越了GPT-4o和所有开源LVLM。
