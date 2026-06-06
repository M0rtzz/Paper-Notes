---
title: >-
  [论文解读] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training
description: >-
  [CVPR 2026][人体理解][远程光体积描记] rPPG-VQA 提出首个面向远程心率检测（rPPG）的视频质量评估框架，结合信号级多方法共识 SNR 和场景级 MLLM 干扰识别，配合两阶段自适应采样策略筛选野外视频构建训练集。
tags:
  - "CVPR 2026"
  - "人体理解"
  - "远程光体积描记"
  - "视频质量评估"
  - "无监督学习"
  - "多模态大语言模型"
  - "数据筛选"
---

# rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training

**会议**: CVPR 2026  
**arXiv**: [2604.11156](https://arxiv.org/abs/2604.11156)  
**代码**: [https://github.com/Tianyang-Dai/rPPG-VQA](https://github.com/Tianyang-Dai/rPPG-VQA)  
**领域**: 人体理解  
**关键词**: 远程光体积描记, 视频质量评估, 无监督学习, 多模态大语言模型, 数据筛选

## 一句话总结
rPPG-VQA 提出首个面向远程心率检测（rPPG）的视频质量评估框架，结合信号级多方法共识 SNR 和场景级 MLLM 干扰识别，配合两阶段自适应采样策略筛选野外视频构建训练集。

## 研究背景与动机

**领域现状**：无监督 rPPG 旨在利用无标注视频数据学习非接触式心率检测，但研究主要集中在方法创新，忽略了数据质量问题。

**现有痛点**：(1) 野外视频中运动、光照等噪声可能淹没微弱的生理信号；(2) AI 生成视频完全缺乏真实生理基础；(3) 传统 VQA 评估人类感知质量，与 rPPG 需求脱节；(4) 单一 SNR 指标易被周期性非生理信号（如闪光灯）欺骗。

**核心矛盾**：视觉质量好的视频可能不含可提取的生理信号，而视觉质量差的视频可能仍包含有效信号——传统 VQA 无法区分。

**核心 idea**：双分支评估——信号级用多方法共识 SNR 排除方法偏差，场景级用 MLLM 识别运动/光照等干扰。

## 方法详解

### 整体框架
输入野外视频 → 信号级分支（多种 rPPG 算法提取信号 → 共识 SNR 评分） + 场景级分支（MLLM 评估运动/光照/压缩干扰） → 融合为统一质量分数 → 两阶段自适应采样 → 构建目标训练集。

### 关键设计

1. **信号级多方法共识 SNR**:

    - 功能：评估视频中生理信号的完整性，排除单一算法偏差
    - 核心思路：用多种传统 rPPG 算法（GREEN、ICA、CHROM、POS 等）分别提取信号并估计 SNR，如果真正的生理信号存在则各方法应给出一致的高 SNR（方法无关性），不一致则说明信号不可靠
    - 设计动机：单一 SNR 容易被周期性噪声欺骗（如闪光灯产生类心跳信号），多方法共识可过滤这种假阳性

2. **场景级 MLLM 干扰识别**:

    - 功能：识别信号级指标无法捕获的场景干扰
    - 核心思路：利用 MLLM 对视频帧进行类人场景推理，检测不稳定光照、剧烈运动、相机伪影等复杂干扰，输出干扰评分
    - 设计动机：信号级指标无法区分信号的生理来源，缺乏场景上下文来区分真正的生物信号和混淆伪影

3. **两阶段自适应采样（TAS）**:

    - 功能：从大规模未审查视频池中构建最优训练集
    - 核心思路：Stage 1 用质量阈值过滤低质量视频；Stage 2 用时长感知概率采样平衡质量、多样性和效率
    - 设计动机：简单过滤可能导致训练集不够多样，概率采样在保证质量的同时维持数据多样性

### 损失函数 / 训练策略
用筛选后的训练集训练现有无监督 rPPG 方法（如 ContrastPhys、SiNC），验证 VQA 框架的有效性。

## 实验关键数据

### 主实验

| 采样策略 | PURE 测试集 HR MAE | 说明 |
|---------|-------------------|------|
| All（全部数据） | 高误差 | 低质量数据损害训练 |
| Random | 中等误差 | 随机采样好于全部 |
| rPPG-VQA | 最低误差 | 质量筛选效果显著 |

### 消融实验

| 配置 | HR MAE | 说明 |
|------|--------|------|
| 信号级+场景级 | 最优 | 双分支互补 |
| 仅信号级 | 次优 | 场景干扰遗漏 |
| 仅场景级 | 中等 | 信号质量评估缺失 |

### 关键发现
- 使用全部野外视频训练反而不如使用质量筛选后的子集
- 双分支互补效果明显，单一分支都有盲区
- TAS 策略在保证质量的同时维持了训练集的多样性

## 亮点与洞察
- **首次系统研究 rPPG 的数据质量问题**：填补了无监督 rPPG 中数据侧的空白
- **方法无关的信号质量度量**：利用多算法共识排除偏差，思路可迁移到其他信号处理任务

## 局限与展望
- MLLM 推理的计算成本较高
- 质量阈值的设定需要一定的人工调优
- 未来可探索端到端的质量感知训练框架

## 相关工作与启发
- **vs 传统 VQA (PSNR/SSIM)**: 面向人类感知，与 rPPG 需求脱节
- **vs 信号后验评估**: 需要先提取信号，无法预筛选原始视频

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统解决 rPPG 数据质量问题
- 实验充分度: ⭐⭐⭐⭐ 多种采样策略对比充分
- 写作质量: ⭐⭐⭐⭐ 问题定义精准
- 价值: ⭐⭐⭐⭐ 解锁了无监督 rPPG 利用野外数据的能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[CVPR 2026\] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)

</div>

<!-- RELATED:END -->
