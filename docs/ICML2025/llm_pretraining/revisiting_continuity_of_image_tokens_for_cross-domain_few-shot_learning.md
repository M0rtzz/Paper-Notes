---
title: >-
  [论文解读] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning
description: >-
  [ICML 2025][cross-domain few-shot] 发现破坏 ViT 图像 token 的连续性（使相邻 patch 像素不再平滑过渡）在源域性能显著下降但在目标域仅略降，揭示连续性帮助学习的大空间模式更难跨域迁移，据此提出简单有效的 ReCIT 方法来缩小域差距。
tags:
  - ICML 2025
  - cross-domain few-shot
  - ViT
  - token continuity
  - spatial patterns
  - domain gap
---

# Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning

**会议**: ICML 2025  
**arXiv**: [2506.03110](https://arxiv.org/abs/2506.03110)  
**代码**: [github.com/shuaiyi308/ReCIT](https://github.com/shuaiyi308/ReCIT)  
**领域**: LLM / NLP（视觉 Transformer）  
**关键词**: cross-domain few-shot, ViT, token continuity, spatial patterns, domain gap

## 一句话总结
发现破坏 ViT 图像 token 的连续性（使相邻 patch 像素不再平滑过渡）在源域性能显著下降但在目标域仅略降，揭示连续性帮助学习的大空间模式更难跨域迁移，据此提出简单有效的 ReCIT 方法来缩小域差距。

## 研究背景与动机
**领域现状**: ViT 通过大规模预训练获得强大的通用表征，但在与源域差距大的下游领域（如医学影像）且训练样本稀少时仍面临挑战，即跨域少样本学习（CDFSL）。

**现有痛点**: 现有 CDFSL 方法主要关注适配和微调策略，忽视了 ViT 在处理图像 token 时的一个有趣现象。

**核心矛盾**: Self-Attention 对 token 顺序不敏感，但图像 token 的空间连续性仍然影响 ViT 的行为——破坏连续性对源域影响大但对目标域影响小。这暗示连续性帮助学到的特征反而是域差距的"元凶"。

**本文解决什么**: 解释这一现象并利用它来改善 CDFSL。

**切入角度**: 分析发现连续性帮助 ViT 学习更大的空间模式（跨 patch 的纹理/结构），这些大模式源域特异性强、难以迁移；而 patch 内的小模式更具域无关性。

**核心 idea**: 通过更好地破坏 token 连续性，引导模型更多依赖小模式（patch 内特征），减少对大模式的依赖。

## 方法详解

### 整体框架
ReCIT (Re-visiting Continuity of Image Tokens) 在训练和推理时对图像 token 的连续性进行控制性破坏：(1) 打乱 patch 的空间排列；(2) 对 patch 边界进行处理以消除跨 patch 的平滑过渡。

### 关键设计
1. **连续性破坏策略**: 通过随机/结构化地重排 patch 顺序来破坏图像 token 的空间连续性。由于 Self-Attention 本身对 token 顺序不敏感，这主要影响的是 patch embedding 阶段——相邻 patch 的像素不再平滑过渡，使得 patch embedding 编码的信息更局限于 patch 内部。设计动机：迫使模型学习 patch 内的局部特征而非跨 patch 的全局模式。

2. **解释性分析**: 从理论和实验两方面解释为什么破坏连续性对源域影响大但对目标域影响小：(a) 源域上，连续性帮助学到的大空间模式（如 ImageNet 中的纹理）贡献了显著性能；(b) 在远域（如医学图像），这些大模式无法迁移，只有小模式有效。因此移除大模式对近域损失大、对远域损失小。

3. **自适应控制**: 可以根据源-目标域距离自适应调节连续性破坏的程度——域差距越大，破坏越彻底。

### 损失函数 / 训练策略
标准的 few-shot learning 训练（meta-learning 或 fine-tuning），核心改变仅在数据预处理阶段。

## 实验关键数据

### 主实验（多个 CDFSL benchmark）
| 方法 | CropDisease | EuroSAT | ISIC | ChestX | 平均 |
|------|------------|---------|------|--------|------|
| 基线 ViT | 中等 | 中等 | 较低 | 较低 | 基线 |
| SOTA 方法 | 较高 | 较高 | 较高 | 中等 | 较高 |
| **ReCIT** | **最高** | **最高** | **最高** | **最高** | **SOTA** |

### 消融实验
| 配置 | 近域性能 | 远域性能 | 说明 |
|------|---------|---------|------|
| 无连续性破坏 | 最高 | 基线 | 标准 ViT |
| 轻微破坏 | 略降 | 提升 | 部分去除大模式 |
| 完全破坏（ReCIT） | 下降 | 最高 | 最大化局部特征 |
| 源域评估 | 显著下降 | N/A | 验证大模式的源域价值 |

### 关键发现
- 连续性破坏对源域性能影响大（-显著%）但对远域仅-略微%
- 域差距越大，ReCIT 的相对增益越大
- 小空间模式比大空间模式具有更好的跨域迁移性

## 亮点与洞察
- 发现了 ViT 中一个被忽视的现象，并提供了深刻的解释
- 方法极其简单（只改变 patch 排列），几乎无额外计算开销
- 解释了为什么 ViT 在大域差距下表现退化的深层原因（大模式不可迁移）
- 提供了一种新的理解 ViT 特征层次的视角

## 局限与展望
- 连续性破坏程度的最优选择依赖域差距的先验知识
- 对域差距小的情况可能有负面影响
- 仅在 CDFSL 上测试，对标准 few-shot 或 full-data 场景效果未知

## 相关工作与启发
- 为理解 ViT 的 patch-based 处理提供了新视角
- 启发：特征的"迁移性"与其空间尺度相关——这个洞察可能适用于其他架构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 发现并解释了一个有趣且有用的现象
- 实验充分度: ⭐⭐⭐⭐ 多个 CDFSL benchmark 验证
- 写作质量: ⭐⭐⭐⭐ 现象-解释-方法的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 对 CDFSL 和 ViT 理解都有贡献

---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：(1) 对 LLM 内部机制的深入理解需求日益增长；(2) 模型效率和可访问性的重要性不断提升；(3) AI 安全和可靠性成为核心关注点。从方法论角度看，本文代表了一种从"黑盒使用"到"白盒理解"的研究范式转变。

### 对未来研究的具体建议
1. 可以将本文的核心思路与其他模态（视觉、语音）结合
2. 考虑在更大规模的模型和数据上验证结论的普适性
3. 探索与强化学习和在线学习结合的可能性
4. 开发自动化的评估和优化工具链


---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：模型能力评估与可靠性保证、参数高效微调与模型压缩、以及 AI 安全与对齐。从方法论角度看，本文代表了对 LLM 深层机制的探索，有助于推动从经验驱动到理论驱动的研究范式转变。

### 对未来研究的具体建议
1. 可以将核心思路与其他模态（视觉、语音、多模态）结合，验证方法的跨模态通用性
2. 在更大规模模型（70B+）和更新的架构（Mixture-of-Experts 等）上验证结论
3. 探索与强化学习、在线学习结合的可能性，实现动态适应
4. 开发自动化评估和优化工具，降低方法的使用门槛
5. 考虑与 LLM alignment 研究的交叉，探索安全性和性能的协同优化

<!-- RELATED:START -->

## 相关论文

- [Learning to Obstruct Few-Shot Image Classification over Restricted Classes](../../ECCV2024/llm_pretraining/learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)
- [3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](../../CVPR2025/llm_pretraining/3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](../../ECCV2024/llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)
- [Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)

<!-- RELATED:END -->
