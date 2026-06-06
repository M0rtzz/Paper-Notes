---
title: >-
  [论文解读] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement
description: >-
  [ACL 2026][视频生成][文本到视频生成] 提出 VideoRepair，首个免训练、模型无关的文本到视频自校正框架，通过 MLLM 检测细粒度文本-视频不对齐，保留正确区域并选择性修复问题区域，在 EvalCrafter 和 T2V-CompBench 上跨四种 T2V 骨干模型一致提升对齐质量。
tags:
  - "ACL 2026"
  - "视频生成"
  - "文本到视频生成"
  - "自校正"
  - "局部修复"
  - "文本-视频对齐"
  - "扩散模型"
---

# Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement

**会议**: ACL 2026  
**arXiv**: [2411.15115](https://arxiv.org/abs/2411.15115)  
**代码**: [video-repair](https://video-repair.github.io/)  
**领域**: 视频生成  
**关键词**: 文本到视频生成, 自校正, 局部修复, 文本-视频对齐, 扩散模型

## 一句话总结

提出 VideoRepair，首个免训练、模型无关的文本到视频自校正框架，通过 MLLM 检测细粒度文本-视频不对齐，保留正确区域并选择性修复问题区域，在 EvalCrafter 和 T2V-CompBench 上跨四种 T2V 骨干模型一致提升对齐质量。

## 研究背景与动机

**领域现状**：文本到视频（T2V）扩散模型在生成质量上取得了显著进步，但在遵循复杂文本提示方面仍有困难——特别是涉及多物体、属性绑定和空间关系时。常见错误包括物体数量错误、属性绑定混乱或区域变形。

**现有痛点**：现有的组合式 T2V 方法虽然改善了组合性，但缺乏显式的反馈机制来检测和纠正不对齐。图像领域的修复框架存在计算开销大、依赖外部生成器、或引入视觉不一致等问题。关键问题是：即使生成的视频存在不对齐的部分，其中正确生成的区域往往应该被保留而非重新生成。

**核心矛盾**：全局重新生成浪费了已正确生成的内容，而简单的 inpainting/editing 缺乏语义引导的能力来引入或修正与文本不匹配的实体。需要一种既能精确定位问题区域又能保留忠实内容的机制。

**本文目标**：设计一个免训练的视频修复框架，能自动检测哪里错了、规划如何修复、然后局部修正。

**切入角度**：类比人类修改创作作品的方式——只修改错误部分，保留正确部分。通过 MLLM 生成细粒度的评估问题来识别不对齐区域，然后利用扩散模型本身的重新生成能力来选择性修复。

**核心 idea**：保留正确区域、选择性修复错误区域——将 MLLM 评估反馈转化为可操作的生成指导。

## 方法详解

### 整体框架

VideoRepair 分三阶段：(1) 不对齐检测：从文本提示提取语义元组，生成评估问题集，用 MLLM 回答二值判断识别不对齐区域；(2) 修复规划：确定需保留的实体及其实例数，通过分割模型获取保留区域掩码，为待修复区域生成局部提示；(3) 局部修复：选择性重新初始化噪声，对保留区域和修复区域分别施加不同文本引导，通过联合优化实现无缝融合。

### 关键设计

1. **MLLM 驱动的不对齐检测（Misalignment Detection）**:

    - 功能：自动识别视频中哪些元素与文本提示不匹配
    - 核心思路：从提示中提取语义元组（实体、属性、关系、动作），用 LLM 生成评估问题集 $Q$，分为计数问题 $Q_c$（如"有一只熊吗？"）和其他问题 $Q_{others}$（属性、动作、场景）。MLLM 对初始视频回答这些问题，计数问题返回三元组（判断、提示数量、视频数量），其他问题返回二值判断。汇总为 $[0,1]$ 的对齐分数
    - 设计动机：比简单的物体存在检查更精细——显式捕获数量、属性、时空关系和动作，提供直接指导修复规划的反馈信号

2. **区域保留修复规划（Refinement Planning）**:

    - 功能：确定保留什么、修复什么、用什么提示修复
    - 核心思路：(a) 用 MLLM 基于问答结果识别正确生成的关键实体 $O^*$ 及其保留数量 $N^*$；(b) 用 pointing 提示和分割模型获取实体在各帧的二值掩码 $\mathbf{M}$；(c) 用 LLM 生成排除已保留实体后的局部修复提示 $p^r$
    - 设计动机：将评估反馈转化为可操作的生成指导——掩码精确定义了哪些像素保留、哪些重新生成，局部提示确保修复区域接收正确的语义引导

3. **局部修复与融合（Localized Refinement）**:

    - 功能：在不破坏正确区域的前提下修复问题区域
    - 核心思路：将掩码下采样到潜空间，保留区域使用原始噪声、修复区域重新采样噪声。每步去噪时运行两次扩散模型：保留区域用原提示 $p$、修复区域用局部提示 $p^r$。最终通过联合优化融合：$V_1 = \arg\min_{\tilde{V}} \|M_{pres} \otimes (\tilde{V} - \hat{V}_{pres})\|^2 + \|M_{refine} \otimes (\tilde{V} - \hat{V}_{refine})\|^2$，实现区域边界的无缝过渡
    - 设计动机：单纯的掩码 inpainting 无法引入新实体，单纯的编辑无法自由修正不对齐；双路径去噪+联合优化兼顾了精确控制和全局一致性

### 损失函数 / 训练策略

完全免训练，使用现有的 T2V 扩散模型进行推理。生成 K 个修复候选视频（不同随机种子），通过评估问题得分选出最佳。若分数持平则用 BLIP-BLEU 分数作为 tiebreaker。

## 实验关键数据

### 主实验

| T2V骨干 | 方法 | EvalCrafter Avg↑ | Visual Quality | Motion Quality | Temporal Consistency |
|--------|------|------|----------|------|------|
| Wan 2.1-1.3B | Original | 44.83 | 63.2 | 61.0 | 62.1 |
| Wan 2.1-1.3B | + VideoRepair | 49.01 | 65.1 | 61.6 | 62.0 |
| VideoCrafter2 | Original | 45.97 | 61.8 | 62.6 | 62.9 |
| VideoCrafter2 | + VideoRepair | 48.83 | 62.1 | 62.4 | 62.0 |
| CogVideoX-5B | Original | 45.01 | 65.8 | 61.0 | 61.8 |
| CogVideoX-5B | + VideoRepair | 46.41 | 64.8 | 61.1 | 61.9 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| vs LLM paraphrasing | 43.12-45.81 | 简单改写提示，提升有限甚至下降 |
| vs SLD | 43.72-47.11 | 部分场景有效但严重破坏视觉/时序质量 |
| vs OPT2I | 45.63-48.69 | 提升明显但低于 VideoRepair |
| VideoRepair | 46.41-49.01 | 一致最优且不损害质量指标 |

### 关键发现

- VideoRepair 在所有四种 T2V 骨干上都带来一致提升，验证了模型无关性
- 关键优势在于不损害视觉质量、运动质量和时序一致性——SLD 方法虽然在对齐分数上有时接近，但严重破坏了这些质量指标（如时序一致性从 62.1 降至 21.0）
- 计数（Count）和颜色（Color）子类别提升最显著，这正是当前 T2V 模型最薄弱的环节

## 亮点与洞察

- **"保留正确、修复错误"的范式**：这是一个直觉上很自然但技术上非平凡的思路——相比全局重新生成或简单 inpainting，区域保留修复在效率和质量上都更优。这个范式可以迁移到任何需要后处理校正的生成任务
- **评估反馈驱动的生成**：将 MLLM 的评估问答结果直接转化为修复计划（掩码+提示），建立了评估和生成之间的闭环。这种自校正范式比单纯的人工反馈更具可扩展性
- **免训练+模型无关**：不需要额外训练任何模型，可以即插即用到任何 T2V 扩散模型上

## 局限与展望

- 需要两次扩散模型前向（保留+修复），推理开销翻倍
- 依赖 MLLM 的评估准确性——如果 MLLM 误判对齐状态可能导致不必要的修改或遗漏
- 当前仅支持单轮修复，迭代修复可能导致误差累积
- 可探索：与 T2V 模型的训练过程结合实现在线自校正、引入用户交互反馈

## 相关工作与启发

- **vs SLD/OPT2I**：SLD 使用全局语义引导但严重破坏视觉质量；OPT2I 优化提示但不做像素级修复；VideoRepair 的区域保留策略兼顾了对齐精度和质量保持
- **vs 图像修复/编辑方法**：Inpainting 只能填充而无法引入新实体，editing 无法自由修正不对齐；VideoRepair 的双路径去噪克服了这两个限制

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个免训练视频自校正框架，区域保留修复范式新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四种骨干、两个基准、全面消融和质量指标评测
- 写作质量: ⭐⭐⭐⭐ 三阶段流程图清晰，方法描述系统化
- 价值: ⭐⭐⭐⭐ 为 T2V 生成提供了通用且实用的后处理改善方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](../../CVPR2025/video_generation/phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)
- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)
- [\[AAAI 2026\] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](../../AAAI2026/video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](../../CVPR2026/video_generation/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[ICML 2026\] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](../../ICML2026/video_generation/world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
