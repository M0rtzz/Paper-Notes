---
title: >-
  [论文解读] Visual Planning: Let's Think Only with Images
description: >-
  [ICLR 2026][机器人][视觉规划] 提出Visual Planning——首个纯视觉推理范式：规划过程完全由图像序列表达（无文本中介），用Large Vision Model自回归生成逐步状态图像；引入VPRL两阶段RL框架（随机轨迹初始化探索+GRPO进度奖励优化），在FrozenLake/Maze/MiniBehavior三个导航任务上平均EM超越文本推理方法27%，证明"vision-first"任务中图像推理远优于文本推理。
tags:
  - ICLR 2026
  - 机器人
  - 视觉规划
  - 纯图像推理
  - 大视觉模型
  - GRPO
  - 强化学习
  - 导航
---

# Visual Planning: Let's Think Only with Images

**会议**: ICLR 2026  
**arXiv**: [2505.11409](https://arxiv.org/abs/2505.11409)  
**代码**: [GitHub](https://github.com/yix8/VisualPlanning)  
**领域**: 视觉推理/规划  
**关键词**: 视觉规划, 纯图像推理, 大视觉模型, GRPO, 强化学习, 导航

## 一句话总结

提出Visual Planning——首个纯视觉推理范式：规划过程完全由图像序列表达（无文本中介），用Large Vision Model自回归生成逐步状态图像；引入VPRL两阶段RL框架（随机轨迹初始化探索+GRPO进度奖励优化），在FrozenLake/Maze/MiniBehavior三个导航任务上平均EM超越文本推理方法27%，证明"vision-first"任务中图像推理远优于文本推理。

## 研究背景与动机

**领域现状**：LLM/MLLM在推理中取得巨大进展，但所有推理过程均在文本空间进行——即使输入包含图像，也先将视觉信息描述为文本再推理。认知科学的Dual Coding Theory指出人类认知同时拥有语言和非语言两个独立通道，空间任务中视觉想象比语言更高效。

**现有痛点**：
   - (1) 空间/几何任务中，视觉信息→文本描述→丢失关键空间特征，造成modality gap
   - (2) Visual Sketchpad等方法用工具生成辅助视觉，但推理决策仍在文本空间完成
   - (3) MVoT生成可视化辅助文本推理，但本质仍是文本驱动的tool-use范式
   - (4) 尚无真正的纯视觉推理范式——所有现有方法最终都依赖文本做决策

**切入角度**：彻底去除文本中介→规划=图像序列→每张图代表一个环境状态→动作隐式编码在状态转换中→用纯视觉数据训练的LVM避免语言干扰。

**RL的动机**：RL已在文本推理中展现出显著优于SFT的泛化能力（DeepSeek-R1），但从未被应用于图像生成式推理/规划场景。

**SFT的不足**：监督学习（VPFT）仅模仿训练分布中的轨迹，缺乏对多样action的探索，容易过拟合且无法从错误中学习。

**评估挑战**：视觉输出是高维稀疏的，不像文本token可以直接评判对错，需要设计专门的dynamics interpreter和progress estimator来评估生成图像是否代表有意义的规划进展。

## 方法详解

### 关键设计1: Visual Planning范式——纯视觉自回归规划

- **功能**：将规划过程定义为图像序列生成，每一步预测下一个视觉状态，无需任何文本参与。
- **核心思路**：给定初始状态图像 $v_0$，模型自回归生成规划轨迹 $\hat{\mathcal{T}} = (\hat{v}_1, \ldots, \hat{v}_n)$，每步条件依赖所有历史状态：

$$\hat{v}_i \sim \pi_\theta(v_i \mid v_0, \hat{v}_1, \ldots, \hat{v}_{i-1})$$

- **设计动机**：空间规划任务中，状态转换（如迷宫中的移动）天然适合用图像表示——文字描述坐标/布局不仅冗长且容易出错（实验显示25.7%的坐标描述与真实布局不匹配）。选择LVM-7B（仅在图像/视频上预训练，零文本数据）作为backbone，彻底消除语言监督的confound。

### 关键设计2: 两阶段RL框架VPRL

- **功能**：提出VPRL两阶段训练框架——Stage 1用随机轨迹初始化策略模型获得探索能力，Stage 2用GRPO+进度奖励优化规划策略。
- **核心思路**：

  **Stage 1 (Policy Initialization)**：在环境中执行随机游走收集轨迹，训练模型生成有效的状态转换并保持探索随机性。训练时随机采样合法的下一状态作为监督目标，防止过拟合：

$$\mathcal{L}_{\text{VPFT}}(\theta) = -\mathbb{E}_{(v_{\leq i}, \tilde{v}_{i+1})} \left[ \log \pi_\theta(\tilde{v}_{i+1} \mid v_{\leq i}) \right]$$

  **Stage 2 (GRPO优化)**：行为模型采样 $G$ 个候选下一状态，用奖励函数打分后计算组内相对优势，通过GRPO目标更新策略：

$$\mathcal{J}_{\text{VPRL}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{k=1}^{G} \min\left(\rho^{(k)} A^{(k)},\; \text{clip}(\rho^{(k)}, 1-\epsilon, 1+\epsilon) A^{(k)}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

- **设计动机**：直接用VPFT作为RL的初始策略会导致探索崩溃——teacher-forcing训练的模型entropy迅速趋近零，生成近乎相同的action导致零advantage、无法更新策略。Stage 1用随机轨迹初始化专门解决这个问题，实验验证其entropy接近均匀随机策略且无效动作比例低。

### 关键设计3: 进度奖励函数（Progress Reward）

- **功能**：设计组合奖励函数，同时评估生成视觉状态的合法性（是否违反环境约束）和目标进度（是否更接近目标）。
- **核心思路**：引入dynamics interpreter $\mathcal{D}$ 解析状态转换的动作类型，progress estimator $P$ 量化到目标的剩余距离。将候选状态分为三类并赋予不同奖励：

$$r(v_i, \hat{v}_{i+1}^{(k)}) = \alpha_{\text{opt}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{A}_{\text{opt}}] + \alpha_{\text{nopt}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{A}_{\text{nopt}}] + \alpha_{\text{inv}} \cdot \mathbb{I}[\mathcal{D}(\cdot) \in \mathcal{E}_{\text{inv}}]$$

  其中 $\alpha_{\text{opt}}=1$（最优动作奖励），$\alpha_{\text{nopt}}=0$（非最优但有效），$\alpha_{\text{inv}}=-5$（无效动作重罚）。

- **设计动机**：视觉输出不像文本可以逐token匹配，需要环境级别的语义评估。三档奖励设计既鼓励向目标前进、又允许合法绕路、但严厉惩罚非法状态（如穿墙），有效引导策略在合法空间内探索最优路径。

## 实验关键数据

### 表1: 主实验——各方法在三个导航任务上的表现

| 方法 | 输入→输出 | FrozenLake EM | FrozenLake PR | Maze EM | Maze PR | MiniBehavior EM | MiniBehavior PR | 平均EM | 平均PR |
|------|-----------|-------------|-------------|---------|---------|----------------|----------------|--------|--------|
| Gemini 2.0 Flash Direct | 图+文→文 | 21.2 | 47.6 | 8.3 | 31.4 | 0.7 | 29.8 | 10.1 | 36.3 |
| Gemini 2.0 Flash CoT | 图+文→文 | 27.6 | 52.5 | 6.9 | 29.8 | 4.0 | 31.2 | 12.8 | 37.8 |
| Gemini 2.5 Pro (think) | 图+文→文 | 72.0 | 85.0 | 21.5 | 35.5 | 37.6 | 59.9 | 43.7 | 60.1 |
| Qwen2.5-VL Direct | 图+文→文 | 1.2 | 15.0 | 0.6 | 14.5 | 0.3 | 9.8 | 0.7 | 13.1 |
| Qwen2.5-VL CoT | 图+文→文 | 8.2 | 29.1 | 2.3 | 15.2 | 0.5 | 14.7 | 3.7 | 19.7 |
| Qwen2.5-VL SFT | 图+文→文 | 68.6 | 84.4 | 60.9 | 70.3 | 31.3 | 56.1 | 53.6 | 69.9 |
| LVM VPFT (ours) | 图→图 | 75.4 | 79.5 | 59.0 | 64.0 | 33.8 | 52.2 | 56.1 | 65.2 |
| **LVM VPRL (ours)** | **图→图** | **91.6** | **93.2** | **74.5** | **77.6** | **75.8** | **83.8** | **80.6** | **84.9** |

### 表2: 文本规划变体在FrozenLake上的对比

| 方法 | EM (%) | PR (%) |
|------|--------|--------|
| Qwen2.5-VL SFT Direct | 68.6 | 84.4 |
| Qwen2.5-VL SFT w/ Coordinates | 74.4 | 82.7 |
| Qwen2.5-VL SFT w/ ASCII | 73.1 | 83.4 |
| Qwen2.5-VL GRPO w/ VPRL reward | 54.4 | 69.9 |
| Qwen2.5-VL GRPO w/ PR metric reward | 60.1 | 74.3 |

发现：文本规划即使加入坐标/ASCII等增强表示，RL也无法超越SFT基线→证明瓶颈在modality gap而非训练方法。

## 关键发现

1. **视觉规划全面碾压文本推理**：VPRL平均EM 80.6% vs 文本SFT最佳 53.6%（+27%），在MiniBehavior上差距最大（75.8% vs 31.3%），说明任务越复杂视觉推理优势越大。

2. **文本RL在多模态输入下失效**：与纯文本domain不同，RL用于文本+图像输入的规划任务反而不如SFT（54.4% vs 68.6%），瓶颈在于视觉信息→文本grounding的modality gap，约25%的布局描述与真实不匹配。

3. **随机初始化是RL成功的关键**：VPFT的entropy训练后趋近零导致探索崩溃；Stage 1随机轨迹初始化使entropy接近均匀分布且无效动作率低，为Stage 2 RL提供充分的探索空间。

4. **VPRL大幅减少无效动作**：失败轨迹中包含无效动作的比例——VPFT为61%~78%，VPRL降低至少24%，说明VPRL有效约束模型在合法动作空间内规划。

5. **VPRL在复杂度缩放时更鲁棒**：FrozenLake从3×3到6×6，Gemini 2.5 Pro EM从98%跌至38.8%，而VPRL从97.6%仅降至82.4%，展现出更平缓的性能曲线。

## 亮点

- **"首个纯视觉推理"**：之前所有"视觉推理"工作最终都在文本空间决策，Visual Planning真正实现了全程图像空间推理——人类画草图解空间题的AI版本。
- **首次将RL应用于图像生成式规划**：将DeepSeek-R1的RL→推理成功范式从文本跨模态迁移到图像生成，开辟全新研究方向。
- **Dual Coding Theory的计算验证**：Paivio假设视觉和语言是独立推理通道，本文首次在计算实验层面证实这一认知科学假说。
- **两阶段设计精巧合理**：Stage 1随机初始化解决RL探索问题的方案简洁有效，比直接VPFT初始化好得多。

## 局限性

1. **任务范围有限**：仅验证了三个grid-based导航任务（FrozenLake/Maze/MiniBehavior），尚未扩展到连续空间、3D环境、或真实机器人场景。
2. **环境约束依赖规则解析**：dynamics interpreter和progress estimator目前基于规则（而非学习），限制了向复杂/未知环境的推广。
3. **图像分辨率和复杂度有限**：当前环境图像为简单grid渲染，对于真实场景的高分辨率复杂图像的可扩展性未知。
4. **训练成本未充分讨论**：两阶段RL+GRPO的训练开销、采样效率等关键工程问题缺乏详细分析。
5. **与语言的互补性未探索**：论文将视觉规划定位为文本推理的替代而非补充，但实际中两种模态的融合可能更优。

## 相关工作对比

### vs Visual Sketchpad (Hu et al., 2024)
Visual Sketchpad用外部工具生成草图/可视化辅助MLLM推理，但**推理决策仍完全在文本空间**——视觉只是辅助展示。Visual Planning则完全在图像空间推理，无文本参与，是根本性的范式转变。

### vs MVoT (Li et al., 2025)
MVoT为每个文本推理步生成可视化，但本质上仍是"文本推理+视觉tool-use"：模型先在文本中决定action，再自调用生成可视化验证。Visual Planning不需要文本决策环节，action隐式编码在图像状态转换中，从根本上消除modality gap。

### vs Action-conditional Generative Models (Hafner et al., 2019; Ha & Schmidhuber, 2018)
世界模型（如Dreamer）学习状态转换动力学用于model-based RL，但它们**不执行规划**——需要耦合外部planner。VPRL则是一个自包含的holistic planner，将规划内化于视觉生成流程中，无需外部规划器。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 纯视觉推理范式+VPRL框架均属首创，GRPO首次用于图像生成式规划
- **实验充分度**: ⭐⭐⭐⭐ 3个导航任务+难度缩放+消融+误差分析全面，但任务类型偏窄
- **写作质量**: ⭐⭐⭐⭐⭐ 认知科学动机清晰，范式对比直观，公式和图表规范
- **价值**: ⭐⭐⭐⭐⭐ 开辟纯视觉推理新方向，对多模态推理社区有重要启发意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)
- [\[ICLR 2026\] Sparse Imagination for Efficient Visual World Model Planning](sparse_imagination_for_efficient_visual_world_model_planning.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](../../NeurIPS2025/robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)

<!-- RELATED:END -->
