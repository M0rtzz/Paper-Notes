---
title: >-
  [论文解读] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach
description: >-
  [CVPR 2026][语音][矛盾/犹豫识别] 提出面向第 10 届 ABAW 竞赛的多模态矛盾/犹豫（A/H）识别方法，整合场景、面部、音频和文本四种模态，通过 Transformer 融合模块和原型增强分类策略，最佳单模型 MF1 达 83.25%，最终测试集上五模型集成达 71.43%。
tags:
  - CVPR 2026
  - 语音
  - 矛盾/犹豫识别
  - 多模态融合
  - 原型学习
  - 情感计算
  - ABAW竞赛
---

# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach

**会议**: CVPR 2026  
**arXiv**: [2603.12848](https://arxiv.org/abs/2603.12848)  
**代码**: [LEYA-HSE/ABAW10-BAH](https://github.com/LEYA-HSE/ABAW10-BAH)  
**领域**: 语音/音频  
**关键词**: 矛盾/犹豫识别, 多模态融合, 原型学习, 情感计算, ABAW竞赛

## 一句话总结
提出面向第 10 届 ABAW 竞赛的多模态矛盾/犹豫（A/H）识别方法，整合场景、面部、音频和文本四种模态，通过 Transformer 融合模块和原型增强分类策略，最佳单模型 MF1 达 83.25%，最终测试集上五模型集成达 71.43%。

## 研究背景与动机
矛盾/犹豫（Ambivalence/Hesitancy, A/H）识别是情感计算中的困难任务，与决策不确定性、抵抗和行为改变动机波动密切相关。A/H 的核心难点在于：

**跨模态不一致性**：A/H 状态常表现为模态间的矛盾——一个人说的话、说话的方式和表情可能不一致

**细粒度行为信号**：不同于基本情绪（如高兴、惊讶），A/H 更加细微，需要多模态综合建模

**文本主导但不充分**：先前研究表明文本是最强单模态线索，但仅靠文本无法捕获 A/H 的全部表现

**本文切入角度**：在先前工作主要使用面部、音频和文本的基础上，**额外引入场景信息**，并设计基于 Transformer 的融合模块配合**原型增强分类目标**，在模态级嵌入上进行融合而非简单拼接。

## 方法详解

### 整体框架
四阶段流程：(1) 各模态独立训练专用编码器；(2) 提取固定维度的模态嵌入；(3) 投影到共享潜在空间；(4) Transformer 融合模块建模跨模态依赖，输出最终 A/H 预测。

### 关键设计
1. **场景模型（VideoMAE）**: 使用 VideoMAE 架构（基于 ViT，Kinetics-400 预训练），对每个视频均匀采样 16 帧，通过管状嵌入（tubelet embedding）分割为 $2 \times 16 \times 16$ 的时空补丁，Transformer 编码器建模时空依赖。场景嵌入 $h_s = \frac{1}{N}\sum_{i=1}^N z_i$ 通过全局平均池化获得。训练 15 epochs，LR=2e-5，标签平滑 0.1。

2. **面部模型（EmotionEfficientNetB0）**: YOLO 人脸检测 → 最大框选择 → 裁剪至 224×224 → EmotionEfficientNetB0（AffectNet+ 微调）提取帧级情感嵌入。关键在于**统计池化**聚合：$\mu = \frac{1}{F}\sum_f e_f$，$\sigma = \sqrt{\frac{1}{F}\sum_f (e_f - \mu)^2}$，最终拼接 $[\mu; \sigma]$ 作为视频级面部表示。这保留了帧间变异性信息，对捕获 A/H 中的情感波动很有价值。

3. **音频模型（EmotionWav2Vec2.0 + Mamba）**: 音频重采样至 16kHz → 预训练 EmotionWav2Vec2.0（MSP-Podcast 情感微调）提取特征序列 $T_a \times 1024$ → **Mamba 编码器**建模时序依赖 → 均值池化获得紧凑嵌入。关键选择：使用第 10 层特征 + Mamba（优于 Transformer），隐层 256，前馈 512，Mamba 状态大小 8，卷积核 4。

4. **文本模型**: 多种策略评估——TF-IDF + 传统分类器（Logistic Regression, CatBoost）和微调 Transformer（EmotionDistilRoBERTa, EmotionTextClassifier）。最佳配置为微调 EmotionDistilRoBERTa + MLP 分类头，达 70.02% 平均 MF1。

5. **模态融合模型**: 各模态嵌入 $x_m$ 通过模态特定投影器（线性层 + LayerNorm + GELU + Dropout）映射到共享空间 $u_m = \phi_m(x_m)$。堆叠为矩阵 $U = [u_1; ...; u_M]$，加上可学习模态嵌入 $E_{\text{mod}}$，经 $L=6$ 层 Transformer 编码器处理，最后掩码均值池化得到融合表示 $z_{\text{fused}}$。支持缺失模态处理（二值模态掩码）。

6. **原型增强变体**: 为每个类别维护 $K=16$ 个可学习原型 $\{p_{c,k}\}$，计算融合表示与原型的 log-sum-exp 相似度：
   $$\hat{y}_c^{\text{proto}} = \log \sum_{k=1}^K \exp\left(\frac{\tilde{z}_{\text{fused}}^\top \tilde{p}_{c,k}}{\tau}\right)$$
   原型头作为**辅助训练损失**（不直接产生最终预测），总损失：$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$，$\lambda_{\text{proto}}=0.2$。

### 损失函数 / 训练策略
融合模型使用 RMSprop（LR=9.44e-5），余弦学习率调度，标签平滑 0.02，梯度裁剪 0.5。每个配置用 5 个固定随机种子（42, 2025, 7777, 12345, 31415）训练，选择平均 MF1 最高的配置。最终集成 5 个种子模型的类概率平均。

## 实验关键数据

### 主实验

| 模型 | 模态 | Avg MF1 | Final Test |
|------|------|---------|-----------|
| EmotionEfficientNetB0 | Face | 62.67% | - |
| VideoMAE | Scene | 61.96% | - |
| EmotionWav2Vec2.0+Mamba | Audio | 69.03% | - |
| EmotionDistilRoBERTa | Text | 70.02% | - |
| 四模态融合 (无原型) | All | 82.66% | 68.32% |
| 四模态融合 (原型增强) | All | **83.25%** | 65.21% |
| 五模型集成 (无原型) | All | 81.29% | 70.17% |
| **五模型集成 (原型增强)** | **All** | 81.89% | **71.43%** |

### 消融实验

| 模态组合 | Avg MF1 | 说明 |
|---------|---------|------|
| Scene + Text | 80.39% | 最强双模态 |
| Face + Scene + Text | 78.77% | 最强三模态 |
| Audio + Text | 69.02% | 音频+文本互补性有限 |
| Face + Audio | 67.40% | 视觉+音频不如文本 |
| Face + Text | 63.24% | 面部+文本较弱 |
| **四模态全融合** | **82.66%** | 全模态最优 |

### 关键发现
- 文本始终是最强单模态（70.02%），但场景模态虽单独较弱（61.96%）却在融合中提供最强互补（Scene+Text=80.39%）
- 原型增强在验证集上提升明显（83.25% vs 82.66%），但单模型在测试集上反而下降（65.21% vs 68.32%），说明过拟合风险
- **集成对泛化至关重要**：5 模型集成将测试集性能从 65-68% 提升到 70-71%
- 多模态融合（82.66%）远超最佳单模态（70.02%），提升 12.64 个百分点
- Mamba 时序编码器优于 Transformer 用于音频建模

## 亮点与洞察
- **场景模态的价值**：先前 A/H 工作忽略场景信息，本文证明场景是最重要的互补模态
- **原型增强正则化**：原型头不直接预测而是作为辅助损失，在保持主分类器灵活性的同时提供结构化正则
- **稳定性导向的超参搜索**：用 5 个固定随机种子评估每个配置，减少选择偏差
- **模态缺失处理**：二值模态掩码使融合模型可优雅处理部分模态缺失

## 局限与展望
- 验证集与测试集表现差距大（83.25% vs 65.21%），泛化能力有待加强
- 未建模模态间的时序交互（仅在视频级嵌入上融合）
- BAH 语料库规模较小（1,427 视频），限制了模型的训练充分度

## 相关工作与启发
- **vs González-González et al.**: 基线工作验证了文本最强，本文在此基础上引入场景并改进融合
- **vs Savchenko & Savchenko**: 使用更轻量的特征管道，本文的 Transformer 融合更充分地建模模态交互
- **vs Hallmen et al.**: 三模态融合方案，本文增加场景模态且融合策略更先进

## 评分
- 新颖性: ⭐⭐⭐ 各组件（VideoMAE、Mamba、原型学习）都不新，贡献在于系统组合和场景模态引入
- 实验充分度: ⭐⭐⭐⭐ 详尽的消融覆盖所有模态组合，多种编码器和融合策略对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设置描述细致，可复现性高
- 价值: ⭐⭐⭐ 竞赛解决方案，方法论贡献有一定局限性，但实验发现有参考价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)
- [\[CVPR 2026\] BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

<!-- RELATED:END -->
