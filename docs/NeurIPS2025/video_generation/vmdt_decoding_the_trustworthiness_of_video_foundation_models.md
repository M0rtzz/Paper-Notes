---
title: >-
  [论文解读] VMDT: Decoding the Trustworthiness of Video Foundation Models
description: >-
  [NeurIPS 2025][视频理解][视频基础模型] 提出 VMDT（Video-Modal DecodingTrust），首个统一评估 T2V 和 V2T 视频基础模型在安全、幻觉、公平、隐私和对抗鲁棒性五个维度上可信度的基准平台，涵盖 7 个 T2V 和 19 个 V2T 模型的大规模评测，揭示了模型规模与可信度之间的复杂关系。
tags:
  - NeurIPS 2025
  - 视频理解
  - 视频基础模型
  - 可信度评估
  - 安全性
  - 公平性
  - 对抗鲁棒性
---

# VMDT: Decoding the Trustworthiness of Video Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.05682](https://arxiv.org/abs/2511.05682)  
**代码**: [有](https://sunblaze-ucb.github.io/VMDT-page/)  
**领域**: video_understanding  
**关键词**: 视频基础模型, 可信度评估, 安全性, 公平性, 对抗鲁棒性

## 一句话总结
提出 VMDT（Video-Modal DecodingTrust），首个统一评估 T2V 和 V2T 视频基础模型在安全、幻觉、公平、隐私和对抗鲁棒性五个维度上可信度的基准平台，涵盖 7 个 T2V 和 19 个 V2T 模型的大规模评测，揭示了模型规模与可信度之间的复杂关系。

## 研究背景与动机
**领域现状**: AI 可信度评估主要集中在 LLM 和图像模型上（如 DecodingTrust、MMDT），视频模态缺乏系统性的可信度基准。视频基础模型（VFM）快速发展，但其可信度评估严重滞后。

**现有痛点**: 视频模态具有独特挑战——如时序风险（有害内容仅在连续播放时才显现）、光敏性癫痫触发（闪烁效果在静态帧中无法检测），这些在图像评估中被忽略。

**核心矛盾**: VFM 能力快速提升，但安全对齐、公平性控制与隐私保护机制严重不足，缺乏统一平台来衡量和追踪进展。

**本文目标**: 构建首个涵盖 T2V 和 V2T 双方向的视频模型可信度评估平台，覆盖五大关键维度。

**切入角度**: 参考 DecodingTrust（文本）和 MMDT（图像）的评估框架，针对视频模态特性设计专用数据集和指标。

**核心 idea**: 将视频模型可信度分解为安全、幻觉、公平、隐私、对抗鲁棒性五个正交维度，分别构建数据集和评估方法，形成统一的评测平台。

## 方法详解

### 整体框架
VMDT 平台评估两类模型：T2V（文本到视频，7 个模型）和 V2T（视频到文本，19 个模型），在五个可信度维度上分别设计数据集和评估指标。每个维度均考虑视频模态的独特特性。

### 关键设计
1. **安全性评估（Safety）**:

    - T2V: 780 条 prompt，覆盖 13 个风险类别，包括 vanilla（直接有害指令）和 transformed（看似无害但底层有害）两种场景
    - V2T: 990 条 video-prompt 对，覆盖 6 大类 27 子类风险
    - 特别关注视频特有风险：时序风险（内容连续播放才显示有害性）和物理伤害（光敏性癫痫触发）
    - 核心指标：Bypass Rate (BR, 模型未拒绝率) 和 Harmful content Generation Rate (HGR, 有害生成率)
    - 使用 GPT-4o 作为评判器（人工一致率 86%-88%）

2. **幻觉评估（Hallucination）**:

    - T2V: 1,650 条 prompt；V2T: 1,218 条 prompt
    - 7 种场景：Natural Selection (NS), Distraction (DIS), Misleading (MIS), Counterfactual (CR), Temporal (TMP, 视频专属), Co-Occurrence (CO), OCR
    - 5 类任务：物体识别、属性识别、动作识别、计数、空间理解（+V2T 场景理解）
    - T2V 使用 Qwen2.5-VL-72B-Instruct 评估（Pearson相关 0.765），V2T 采用关键词匹配

3. **公平性评估（Fairness）**:

    - T2V: 1,086 条 prompt；V2T: 5,008 条 video-prompt 对
    - 三个人口统计属性：性别、种族、年龄
    - 三个指标：F₁(g) 社会刻板印象公平性、F₂(g) 决策公平性、O overkill 公平性（追求多样性而牺牲历史准确性）
    - 理想值：F₁=0, F₂=0, O=0

4. **隐私评估（Privacy）**:

    - T2V: 1,000 条 prompt（来自 WebVid-10M 训练语料）评估数据记忆
    - V2T: 200 条驾驶视频评估位置推断能力
    - T2V 指标：ℓ₂ 距离和余弦相似度；V2T 指标：位置推断准确率

5. **对抗鲁棒性评估（Adversarial Robustness）**:

    - T2V: 329 对 benign/adversarial prompt；V2T: 1,523 对
    - 三种攻击算法：贪心、遗传、梯度（T2V）；FMM-Attack（V2T）
    - 五个任务：动作/属性/计数/物体/空间理解
    - 指标：benign accuracy 和 robust accuracy 的性能下降

### 损失函数 / 训练策略
本文为评估性工作（benchmark），不涉及模型训练。评估流程：对每个维度构建数据集 → 模型推理生成输出 → 自动化评估（GPT-4o 或关键词匹配）→ 汇总分析。

## 实验关键数据

### 主实验

**T2V 安全性（HGR，越低越安全）**:

| 模型 | Vanilla HGR | Transformed HGR | 平均 HGR |
|------|-----------|----------------|----------|
| Nova Reel | 0.08 | 0.11 | **0.10** |
| Luma | 0.19 | 0.14 | 0.17 |
| CogVideoX-5B | 0.45 | 0.26 | 0.36 |
| Pika | 0.52 | 0.28 | 0.40 |

**T2V 幻觉（准确率%，越高越好）**:

| 模型 | NS | DIS | MIS | CR | TMP | CO | OCR | Avg |
|------|----|-----|-----|----|-----|----|----|-----|
| Luma | 63.8 | 74.7 | 78.3 | 68.5 | 45.5 | 82.9 | 59.7 | **67.6** |
| Pika | 56.5 | 68.9 | 72.3 | 70.7 | 53.7 | 77.3 | 41.5 | 63.0 |
| Vchitect-2.0 | 58.5 | 66.6 | 47.9 | 28.3 | 46.9 | 35.3 | 59.2 | 49.0 |

**跨维度综合评分**:

| 排名 | T2V 最佳 | T2V 最差 | V2T 最佳 | V2T 最差 |
|------|---------|---------|---------|---------|
| 模型 | Luma | CogVideoX-5B | InternVL2.5-78B | Qwen2.5-VL-3B |
| 综合分 | 70.1 | 55.7 | 72.7 | 65.3 |

### 消融实验

**模型规模与各维度的关系（V2T）**:

| 维度 | 规模效应 | 相关性 |
|------|---------|-------|
| 幻觉 | 规模↑ → 幻觉↓ | 正相关（改善） |
| 对抗鲁棒性 | 规模↑ → 鲁棒性↑ | 正相关（P=0.034） |
| 公平性 | 规模↑ → 不公平↑ | 负相关 |
| 隐私 | 规模↑ → 位置推断↑ | 正相关（Pearson=0.544, P=0.016） |
| 安全性 | 无显著规模效应 | 不相关 |

### 关键发现
1. 所有开源 T2V 模型均缺乏安全拒绝机制（BR=1.00），即使闭源模型也仅有有限的安全防护
2. T2V 模型在 transformed 场景下 HGR 更低，反映的是能力限制而非安全性提升
3. V2T 模型中模型规模是把双刃剑：提升幻觉和鲁棒性表现，但恶化公平性和隐私风险
4. T2V 模型在性别/种族/年龄上存在严重过度代表（偏向男性、白人、年轻人），且偏差比 T2I 更严重
5. 即使最佳模型综合分仅约 70-73，距理想的可信模型差距巨大

## 亮点与洞察
- **首创性**: 首个统一 T2V 和 V2T 的视频可信度评估平台，填补了重要空白
- **视频特有风险发现**: 时序风险和光敏性癫痫触发等无法用图像评估覆盖的安全问题
- **规模悖论**: 首次系统地揭示了 V2T 模型中规模与可信度的非单调关系——更大模型更少幻觉但更不公平
- **开源闭源鸿沟**: 尤其在安全性维度，开源与闭源模型差距巨大
- **跨模态对比**: 系统比较 T2V 与 T2I 模型的公平性差异，发现 T2V 偏向男性而 T2I 偏向女性（过度纠正）

## 局限与展望
1. 评估依赖 GPT-4o 作为裁判，引入了评判模型的偏差（虽然人工一致率约 86-88%）
2. T2V 仅评估了 7 个模型，覆盖不够全面（如 Sora 等未纳入）
3. 隐私评估仅关注像素级记忆和位置推断，未涵盖面部识别等更复杂的隐私风险
4. 对抗攻击方法有限，未考虑更强的自适应攻击
5. 缺乏对模型改进的具体建议和缓解策略

## 相关工作与启发
- **DecodingTrust / TrustGPT**: LLM 可信度评估框架，VMDT 将其扩展到视频模态
- **MMDT**: 多模态（图像） DecodingTrust，VMDT 的直接前驱，多处复用其数据集设计
- **T2VSafetyBench / SafeWatch-Bench**: 视频安全评估工作，VMDT 整合并扩展了其数据和分类体系
- 启发：可信度评估应随模型能力同步发展；视频作为信息最丰富的模态，其可信度挑战也最复杂

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个全面的视频模型可信度平台
- 实验充分度: ⭐⭐⭐⭐⭐ 26 个模型 × 5 个维度的大规模评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现丰富有洞察力
- 价值: ⭐⭐⭐⭐ 对视频模型安全发展具有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)
- [VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [Navigation World Models](../../CVPR2025/video_generation/navigation_world_models.md)
- [Video Diffusion Models Excel at Tracking Similar-Looking Objects Without Supervision](video_diffusion_models_excel_at_tracking_similar-looking_objects_without_supervi.md)
- [Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models](video_killed_the_energy_budget_characterizing_the_latency_and_power_regimes_of_o.md)

<!-- RELATED:END -->
