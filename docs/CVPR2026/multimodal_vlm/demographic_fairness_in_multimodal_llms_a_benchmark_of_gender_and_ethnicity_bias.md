---
title: >-
  [论文解读] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification
description: >-
  [CVPR 2026][多模态][多模态大模型] 首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。
tags:
  - CVPR 2026
  - 多模态
  - 多模态大模型
  - 人脸验证
  - 人口统计公平性
  - 偏见基准
  - 族裔和性别偏差
---

# Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification

**会议**: CVPR 2026  
**arXiv**: [2603.25613](https://arxiv.org/abs/2603.25613)  
**代码**: [项目页面](https://www.idiap.ch/paper/mllm-fairness) (有)  
**领域**: Multimodal / VLM  
**关键词**: 多模态大模型, 人脸验证, 人口统计公平性, 偏见基准, 族裔和性别偏差

## 一句话总结
首次系统性地评估了 9 个开源 MLLM 在人脸验证任务上的人口统计公平性，在 IJB-C 和 RFW 两个 benchmark 上使用 4 种 FMR-based 公平性指标衡量性别和族裔偏差，发现 MLLM 的偏见模式与传统人脸识别系统不同。

## 研究背景与动机
**领域现状**: 多模态大模型 (MLLM) 近来被探索用于人脸验证——通过视觉问答方式判断两张脸是否属于同一人。与传统嵌入式系统不同，MLLM 依赖通用视觉推理能力而非专门训练。

**现有痛点**: 传统人脸识别系统的人口统计偏见已被广泛研究（如 Buolamwini & Gebru 发现深肤色女性错误率更高），但 MLLM 作为人脸验证系统的公平性**完全未被探索**。

**核心矛盾**: MLLM 处理人脸的方式与嵌入式系统根本不同（视觉问答 vs 特征距离），传统系统的偏见模式是否会延续到 MLLM 上是未知的。

**本文目标**: 建立 MLLM 人脸验证公平性的评估基准，发现和分析偏见模式。

**切入角度**: 在标准人脸验证协议（IJB-C、RFW）上，按 4 个族裔组 + 2 个性别组评估 9 个 MLLM，使用多维度公平性指标。

**核心 idea**: MLLM 的偏见模式与传统系统不同——最准确的模型不一定最公平，性能差的模型可能因"均匀高错误率"而看似公平。

## 方法详解

### 整体框架
给定两张人脸图像和文本提示，MLLM 输出相似度分数 $s_{ij} \in [0,1]$。对于多图 template，取所有 $m \times n$ 对的相似度平均值作为 template-level 分数。在不同阈值下计算 FMR、FNMR、EER，并按人口统计组分别评估。

### 关键设计
1. **评估协议设计**: 在 IJB-C 上抽样 10,000 template 对（受限于 MLLM 推理速度，每个模型需约 20 天 H100 计算），按 4 个族裔（African, East Asian, South Asian, Caucasian）和 2 个性别分组。RFW 提供平衡的 4 族裔 × 6000 对。设计动机：确保每个人口统计组有足够的统计样本。

2. **多维度公平性指标体系**: 使用 4 种互补指标——最大 FMR 差值 $\Delta$（绝对）、最大 FMR 比值 $R$（相对）、最大比值与几何均值之比 $M$（多组稳定版）、Gini 系数 $G$（全局不等式度量）。加上 Decidability 指数 $d'$ 衡量得分分布分离度。设计动机：单一指标可能误导，多维度评估更全面——如 $R$ 在最小 FMR 接近零时不稳定，$M$ 通过几何均值缓解这一问题。

3. **零样本评估**: 所有 MLLM 均在零样本设置下评估，不做任何微调或上下文学习，反映其"开箱即用"的公平性。

### 损失函数 / 训练策略
本文是评测工作，不涉及训练。所有模型均使用预训练权重，统一文本提示格式。

## 实验关键数据

### 主实验（IJB-C EER, % ↓）

| 模型 | Global | African | Caucasian | East Asian | South Asian | σ(族裔) |
|------|--------|---------|-----------|------------|-------------|--------|
| FaceLLM-8B | **5.13** | 4.65 | 5.53 | 4.31 | **3.98** | **0.58** |
| Qwen2-VL-7B | 8.54 | 6.34 | 9.14 | 7.49 | 5.97 | 1.23 |
| Qwen2.5-VL-7B | 10.43 | 9.24 | 11.07 | 10.38 | 8.13 | 1.12 |
| Ovis1.5-Llama3-8B | 30.89 | 32.71 | 30.35 | 31.30 | 32.67 | 0.99 |

### 消融实验（RFW 跨族裔对比）

| 模型 | Global EER | African | Caucasian | Asian | Indian | σ |
|------|-----------|---------|-----------|-------|--------|---|
| FaceLLM-8B | **29.46** | 35.25 | 27.87 | **21.23** | 29.13 | 4.98 |
| Qwen2-VL-7B | 34.39 | 39.63 | 34.57 | 24.40 | 34.13 | 5.51 |
| Valley2 | 39.67 | 45.85 | 40.35 | 30.33 | 39.92 | 5.58 |

### 关键发现
- **FaceLLM-8B 一枝独秀**：作为唯一的人脸专用 MLLM，在两个 benchmark 上均远超通用 MLLM（EER 5.13% vs 次优 8.54%）。
- **偏见模式与传统系统不同**：在 RFW 上，所有 MLLM 对 Asian 的 EER 最低（最好表现），而传统系统通常对 Caucasian 表现最好。
- **最准确 ≠ 最公平**：FaceLLM-8B 在 IJB-C 上 σ=0.58（最小族裔差异），但在 RFW 上 σ=4.98，不是最公平的。
- **低准确度的虚假公平**：Ovis1.5-Llama3-8B 族裔差异 σ=0.99 看似很低，但其 EER 高达 30.89%——因为所有群组都"均匀地差"。
- **性别偏差**：大多数 MLLM 对女性 EER 高于男性（差距 1-4%），FaceLLM-8B 差距最小（0.55%）。

## 亮点与洞察
- **填补空白**：首次将人口统计公平性分析从传统人脸识别扩展到 MLLM 领域
- **多维度指标体系**：4 种 FMR-based 指标 + decidability 指数的组合避免了单一指标的误导
- **"虚假公平"的发现**：揭示了低准确度模型可能伪装成公平模型的现象——如 Ovis1.5 各族裔 EER 差异仅 0.99，但总体 EER 高达 30.89%
- **实际意义**：MLLM 正被考虑用于替代传统人脸系统，了解其公平性特征对负责任部署至关重要
- **偏见模式新发现**：MLLM 对 Asian 组表现最好（传统系统通常对 Caucasian 最好），这可能反映了 MLLM 预训练数据的分布差异
- **Qwen2-VL-7B 的性别一致性**：Gender gap 仅 0.10%，可能暗示某些架构设计天然有利于性别公平
- **FaceLLM 的领域专用优势**：EER 5.13% vs 通用模型最优 8.54%，说明人脸专用微调不可或缺

## 局限与展望
- 计算成本极高（每个模型约 20 天 H100），限制了可评估模型数量和 IJB-C 的采样规模。
- 仅评估开源模型，未包含 GPT-4V、Gemini 等闭源 MLLM。
- 公平性分析局限于族裔和性别，未涉及年龄、肤色等更细粒度属性。
- 未深入分析偏见的**来源**（是预训练数据不平衡还是模型架构导致）。
- MLLM 整体人脸验证性能远不如专用系统，实用性有限。- 未探索 in-context learning 或 prompt engineering 是否能减轻偏见
- 跨数据集（IJB-C vs RFW）的偏见模式不一致，需要更大规模统一评估

## 相关工作与启发
- 与 FaceRecBench、FaceXBench 的区别：后两者评估 MLLM 人脸能力的准确性，本文专注**公平性**维度。
- NIST FRVT 系列工作为传统系统建立了公平性评估标准，本文将其扩展到 MLLM。
- 公平性指标选择方面遵循 ISO/IEC 19795-10:2024 标准。
- FaceLLM 的优势表明通用 MLLM 在人脸任务中确实需要领域专用微调

## 技术细节补充
- **Template-level 聚合**: $s(T_p, T_g) = \frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n} s_{ij}$
- **阈值扫描**: $\tau \in [0,1]$，步长 0.005（201 个操作点）
- **Decidability**: $d' = \frac{\mu_{genuine} - \mu_{impostor}}{\sqrt{\frac{1}{2}(\sigma_{genuine}^2 + \sigma_{impostor}^2)}}$
- **IJB-C 采样**: 10,000 对，每模型约 20 天 H100 计算
- **LLaVA-NeXT 失败**: 仅产生 2 个独特相似度分数，无法提供有意义的验证
- **RFW 上所有 MLLM 性能显著下降**（EER 29-50%），说明该 benchmark 更具挑战性
- **4 种公平性指标**: $\Delta$ (差值)、$R$ (比值)、$M$ (与几何均值比值)、$G$ (Gini系数)
- **Gini 系数定义**: $G = \frac{\sum_i\sum_j |e_i - e_j|}{2K\sum_i e_i}$，$G=0$完全公平
- **最大比值的不稳定性**: 当最小组 FMR 接近零时 $R$ 趋于无穷，$M$ 通过几何均值缓解此问题
- **模型家族覆盖**: Idefics3, Ovis, Qwen2-VL, Qwen2.5-VL, Valley, LLaVA-NeXT, FaceLLM 共 6 个家族 9 个模型
- **IJB-C template 结构**: 每个 template 包含 1 张到数百张图像/视频帧
- **零样本评估协议**: 所有模型不做微调、不提供上下文示例
- **TMR 指标**: 在 FMR=10%/1%/0.1% 三个安全等级下报告真正匹配率
- **跨 benchmark 差异**: IJB-C 和 RFW 上偏见模式显著不同，如 FaceLLM 在 IJB-C 上 $\sigma$=0.58 最低但在 RFW 上 $\sigma$=4.98
- **模型参数范围**: 从 2B (Qwen2-VL-2B) 到 8B (FaceLLM-8B)，覆盖不同规模

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统评估 MLLM 人脸验证公平性，发现与传统系统不同的偏见模式
- 实验充分度: ⭐⭐⭐⭐ 9 个模型 × 2 benchmark × 多指标，但受限于计算成本采样规模有限
- 写作质量: ⭐⭐⭐⭐ 指标定义严谨，分析透彻
- 价值: ⭐⭐⭐⭐ 对 MLLM 负责任部署具有重要参考意义

<!-- RELATED:START -->

## 相关论文

- [Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [MultiMM: Cultural Bias Matters — Cross-Cultural Benchmark for Multimodal Metaphors](../../ACL2025/multimodal_vlm/multimm_cultural_metaphor.md)
- [FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)

<!-- RELATED:END -->
