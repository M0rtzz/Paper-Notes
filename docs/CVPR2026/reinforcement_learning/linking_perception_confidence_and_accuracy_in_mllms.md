---
title: >-
  [论文解读] Linking Perception, Confidence and Accuracy in MLLMs
description: >-
  [CVPR 2026][多模态大模型] 揭示 MLLM 的严重置信度失校准问题（视觉输入退化时准确率暴跌但置信度不变），提出 CDRL（基于原始-噪声图像对的置信度驱动 RL）进行感知敏感性训练，并利用校准后的置信度实现自适应测试时缩放（CA-TTS），在四个基准上平均提升 8.8%。
tags:
  - CVPR 2026
  - 多模态大模型
  - 置信度校准
  - 强化学习
  - 测试时缩放
  - 视觉感知
---

# Linking Perception, Confidence and Accuracy in MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.12149](https://arxiv.org/abs/2603.12149)  
**代码**: https://github.com/anotherbricki/CA-TTS (有)  
**领域**: 强化学习  
**关键词**: 多模态大模型, 置信度校准, 强化学习, 测试时缩放, 视觉感知

## 一句话总结
揭示 MLLM 的严重置信度失校准问题（视觉输入退化时准确率暴跌但置信度不变），提出 CDRL（基于原始-噪声图像对的置信度驱动 RL）进行感知敏感性训练，并利用校准后的置信度实现自适应测试时缩放（CA-TTS），在四个基准上平均提升 8.8%。

## 研究背景与动机
近年来 MLLM 研究主要聚焦于增强视觉感知能力以提升准确率，但一个关键问题被忽视了：**模型知道自己什么时候不知道吗？**

作者设计了一个探测实验：对关键视觉证据逐步添加噪声，同时观察模型的置信度和准确率变化。结果发现：置信度几乎保持不变，但准确率大幅下降。这暴露了 MLLM 的严重置信度失校准——即使视觉感知严重退化，模型仍维持高置信度。

现有 LLM 的置信度校准方法在 token 粒度上操作，但 MLLM 的视觉感知是全局性的（贯穿整个响应），存在粒度不匹配。LLM 校准方法也未考虑视觉组件对校准的影响。

核心 idea：(1) 用原始-噪声图像对训练 RL，通过置信度差异奖励增强感知敏感性，同时用准确性-置信度对齐奖励实现校准；(2) 校准后的置信度天然可作为测试时缩放的路由信号——这是一个"免费午餐"，因为校准本身就带来了 TTS 能力。

## 方法详解

### 整体框架
两阶段：(1) CDRL 训练阶段——用 GRPO 在原始-噪声图像对上训练，增强感知敏感性并校准置信度；(2) CA-TTS 推理阶段——利用校准后的置信度信号自适应调度三个解耦的推理模块（Self-Consistency、Self-Reflection、Self-Check），由 Expert Model 作为 Planner/Voter/Critic 协调。

### 关键设计

1. **Confidence-Driven Reinforcement Learning (CDRL)**:

    - 功能：增强 MLLM 的感知敏感性（对视觉退化要有反应）并校准置信度（正确时高置信，错误时低置信）
    - 核心思路：用 CLIP attention map 对关键视觉区域添加噪声生成图像对 $(i, i')$。置信度定义为 Negative Mean Log-Probability：$C = \frac{1}{T}\sum_{t=1}^T \text{Conf}_{\text{token}_t}$, $\text{Conf}_{\text{token}} = -\frac{1}{k}\sum_{i=1}^k \log p_{(i)}$。置信度校准奖励：$R_{\text{Conf},j} = \underbrace{\alpha \tanh(\beta \cdot \Delta C)}_{\text{Perception Term}} + \underbrace{(2 \cdot R_{\text{Output},j} - 1) \cdot C_j^{norm}}_{\text{Calibration Term}}$
    - 设计动机：Perception Term 奖励原始图和噪声图之间的置信度差异（$\Delta C = C_j - C_j'$），鼓励模型对视觉退化敏感。Calibration Term 在正确时奖励高置信（+$C_j$），错误时惩罚高置信（-$C_j$），实现 accuracy-confidence 对齐

2. **Self-Consistency（自洽性模块）**:

    - 功能：采样多个响应，用置信度加权投票 + Expert Model 外部校准得到稳健答案
    - 核心思路：$V_{internal}[k] = \sum_{i=1}^n C_i \cdot \mathbb{I}(A_i = k)$ 为内部置信度加权投票。Expert Model (Voter) 对候选选项给出外部置信度 $C_{expert}$，综合投票 $V_{final}[k] = V_{internal}^{norm}[k] + \tau_1 \cdot c_k$
    - 设计动机：相比普通多数投票，置信度加权投票能让"确信的正确回答"贡献更大权重，Expert Model 提供独立的外部验证

3. **Self-Reflection（自反思模块）**:

    - 功能：Expert Model 作为 Critic 生成对问题的批评，引导基座模型重新思考
    - 核心思路：$Crit = M_{expert}^{Critic}(i, q, P_{critique})$，$(CoT_{reflect}, A_{reflect}) = M_{base}(i, q, Crit)$，反思答案加权 $\tau_2$ 加入最终投票
    - 设计动机：低置信度的预测可以通过外部引导的反思来纠正

4. **Self-Check（自检模块）**:

    - 功能：在视觉层面进行自检，用 Visual Contrastive Decoding (VCD) 对比原始和噪声图像的输出
    - 核心思路：$\log P_{VCD}(y|i,q) = (1+\alpha) \cdot \log P_\theta(y|i,q) - \alpha \cdot \log P_\theta(y|i',q)$，对比解码的答案加权 $\tau_3$ 加入投票
    - 设计动机：从视觉层面验证推理，噪声图像上的"虚假自信"和原始图像的"真实信号"之间的差异能凸显可靠的视觉推理

### 损失函数 / 训练策略
GRPO 训练，总奖励 $r_j = R_{\text{Conf},j} + R_{\text{Output},j} + R_{\text{Format},j}$。基座模型 Qwen2.5-VL-7B-Instruct，8×H100 全参数微调，训练集 1936 样本。Expert Model 为 Gemini-2.5-Pro。

## 实验关键数据

### 主实验

| 方法 | Math-Vista | Math-Vision | MMStar | MMMU |
|------|-----------|------------|--------|------|
| Pass@1 (base) | 64.7 | 23.0 | 60.2 | 48.8 |
| Majority Voting | 69.8 | 30.1 | 69.0 | 57.5 |
| VL-Rethinker | 74.1 | 30.7 | 63.4 | 55.6 |
| We-Think | 73.3 | 29.7 | 65.1 | 55.7 |
| **Ours (CDRL+CA-TTS)** | **79.5** | **42.4** | **71.3** | **66.3** |

### 消融实验

| 配置 | Math-Vision ALL | 说明 |
|------|----------------|------|
| Training-Free (Pass@1) | 22.96 | 基线 |
| CDRL only | 26.38 | 校准后模型状态更好 |
| CA-TTS only | 37.99 | TTS 框架显著提升 |
| CDRL + CA-TTS | **42.35** | 二者协同，最佳 |

### 关键发现
- CDRL 训练后模型对视觉扰动的置信度下降幅度提升 4-8 倍（如 Noised: -0.32 → -1.39），真正"知道自己不知道"
- CA-TTS 的缩放斜率 $\beta_1 = 3.65$ 是 Majority Voting（1.64）的 2.2 倍、DeepConf（1.19）的 3.1 倍——校准后的置信度使 TTS 更高效
- 即使用 Qwen2.5-VL-7B 自身作为 Expert，也比 Majority Voting 提升显著，不依赖超强 Expert
- 在 MMMU 上 66.3% vs VL-Rethinker 55.6%，提升 10.7 个百分点

## 亮点与洞察
- "模型知道自己不知道吗"的探测实验非常直观有力地揭示了 MLLM 的核心缺陷
- CDRL 的双项奖励设计优雅：Perception Term 用图像对增强敏感性，Calibration Term 将置信度与准确性对齐
- "校准后的置信度是免费午餐"——训练时的校准直接转化为推理时 TTS 的能力，无需额外成本
- CA-TTS 的三个模块完全解耦、顺序无关，都只贡献投票，架构灵活且鲁棒

## 局限与展望
- CA-TTS 依赖 Expert Model（如 Gemini-2.5-Pro），引入了外部 API 成本和延迟
- 训练数据仅 1936 样本，扩大规模可能进一步提升校准质量
- Self-Check 的 VCD 需要对噪声图像额外推理，增加了推理开销
- 三个模块的投票权重 $\tau_1 = \tau_2 = \tau_3 = 0.5$ 为固定值，自适应权重可能更优

## 相关工作与启发
- DeepConf 用置信度做 TTS，但仅用于数学推理且未做校准训练，本文补上了训练环节
- VCD 原用于缓解幻觉，本文将其整合到 TTS 框架中作为视觉自检模块
- 与 ToT 等树搜索方法相比，CA-TTS 的解耦多阶段验证更鲁棒，避免了单点故障

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究 MLLM 的视觉感知-置信度校准问题，CDRL+CA-TTS 框架原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 四个基准、多消融、缩放曲线分析、敏感性实验、case study 均完善
- 写作质量: ⭐⭐⭐⭐ 探测实验引入方式引人入胜，框架描述清晰
- 价值: ⭐⭐⭐⭐⭐ 揭示了 MLLM 的基础性问题并提供系统性解决方案，8.8% 平均提升意义重大

## 关键术语
- **NMLP (Negative Mean Log-Probability)**: 全序列级别的置信度度量，值越低表示越确定
- **Perceptual Bluntness**: 模型对视觉输入退化不敏感的现象
- **VCD (Visual Contrastive Decoding)**: 对比原始和噪声图像的 logit 差异进行解码
- **Free Lunch**: 校准训练免费获得的 TTS 能力提升

<!-- RELATED:START -->

## 相关论文

- [When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?](../../AAAI2026/reinforcement_learning/when_eyes_and_ears_disagree_can_mllms_discern_audio-visual_confusion.md)
- [Spotlight on Token Perception for Multimodal Reinforcement Learning](../../ICLR2026/reinforcement_learning/spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [APPLE: Toward General Active Perception via Reinforcement Learning](../../ICLR2026/reinforcement_learning/apple_toward_general_active_perception_via_reinforcement_learning.md)
- [VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/reinforcement_learning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)

<!-- RELATED:END -->
