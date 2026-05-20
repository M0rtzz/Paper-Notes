---
title: >-
  ICML2026 视频理解方向8篇论文解读
description: >-
  8篇ICML2026的视频理解方向论文解读，涵盖目标跟踪、推理、异常检测、强化学习、多模态、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "视频理解"
  - "论文解读"
  - "论文笔记"
  - "目标跟踪"
  - "推理"
  - "异常检测"
  - "强化学习"
  - "多模态"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📹 视频理解

**🧪 ICML2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (11)](../../ACL2026/video_understanding/index.md) · [📷 CVPR2026 (77)](../../CVPR2026/video_understanding/index.md) · [🔬 ICLR2026 (22)](../../ICLR2026/video_understanding/index.md) · [🤖 AAAI2026 (33)](../../AAAI2026/video_understanding/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/video_understanding/index.md) · [📹 ICCV2025 (57)](../../ICCV2025/video_understanding/index.md)

🔥 **高频主题：** 目标跟踪 ×3

**[CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)**

:   本文针对视频字幕擦除提出 CLEAR：两阶段训练（Stage I 用 dual encoder + 正交解耦学自监督字幕先验掩码；Stage II 在 Wan2.1 视频扩散模型上加 LoRA + occlusion head 做自适应加权），推理完全不需要任何 mask 或文本检测器，仅训练 0.77% 参数就在中文测试集上把 PSNR 推到 26.80 dB（比最强基线 +6.77 dB），并零样本泛化到 6 种语言。

**[Find, Fix, Reason: Context Repair for Video Reasoning](find_fix_reason_context_repair_for_video_reasoning.md)**

:   本文针对视频推理中"on-policy RL 在能力天花板停滞、off-policy 蒸馏又会熵塌缩"的两难，引入一个冻结的、工具集成的大教师模型在学生 rollout 失败时插入最小化的"证据补丁" (key-frame 区间、错误类型)，让学生在同一问题上重新作答，并把修复后的轨迹通过 chosen-rollout 机制纳入 GRPO 优化。

**[Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection](privacy-aware_video_anomaly_detection_through_orthogonal_subspace_projection.md)**

:   作者提出 OPL（Orthogonal Projection Layer）和加强版 G-OPL，用一个 QR 分解出来的可学习正交子空间，在视频异常检测特征空间中显式投影掉"任务无关变量"和"人脸隐私分量"，同时引入 SSC/ARD/PD/FPD 四个隐私感知指标，在保持/提升 VAD AUC 的前提下让线性 SVM 探针对面部预测的准确率显著下降。

**[RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)**

:   RELO 把视觉单目标跟踪中"哪里是目标"这件事重构成一个空间特征图上的 MDP,把每个空间位置当作 action,用 actor-critic + IoU/AUC 直接奖励替换掉传统的手工中心热图监督,并配合"先 warmup 回归 + 层对齐时序 token 传播"两个稳定化设计,在 LaSOText 上以 57.5% AUC 拿到 SOTA。

**[Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval](revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)**

:   本文针对 Partially Relevant Video Retrieval (PRVR) 中"短查询 vs 长视频"导致的查询歧义与时间稀疏监督问题，提出基于 Dirichlet 分布的层次证据学习框架 Holmes，在视频间用三重原则区分精确/多义/欠定查询并自适应校准标签，在视频内用带 dustbin 的柔性最优传输获得稠密对齐，在 ActivityNet/Charades/TVR 三个数据集上取得 SOTA。

**[STORM: Segment, Track, and Object Re-Localization from a Single Image](storm_segment_track_and_object_re-localization_from_a_single_image.md)**

:   STORM 提出"一张参考图就能跑"的 6D 位姿跟踪框架：用层级化空间融合注意力 HSFA 做参考-查询特征对齐（产出分割掩膜 + SAM3D 网格），再训一个 BCE 二分类的 Tracking Verifier，把其 logit 取负当作能量分数 $E=-g_\theta$，连续 $L=3$ 帧超阈值就触发自动重定位，从而在 LM-O / YCB-V 上把无标注 6D 跟踪精度推到接近 ground-truth 掩膜上限。

**[Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)**

:   OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练，靠 Meta Merger 做模态融合、Dual MoE 把"时空匹配"与"模态融合"两类异质特征显式拆到 T-MoE 与 M-MoE，并用 dissimilarity loss + router clustering 保证它们不塌成同一子空间。

**[VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)**

:   VideoSEAL 发现现有 agentic 长视频 QA 系统存在「答对但没看到证据」的失配问题，并把根因归结为「coupled agent 把规划和回答权混在一起」，提出 planner-inspector 解耦框架：planner 负责长视距证据搜寻、inspector 持有独占回答权并在像素级证据充分时才放行，在 LVBench 上把准确率从 48.2% 拉到 55.1%（↑20.5%）且 LongVideoBench 从 52.2% 升至 62.0%。
