---
title: >-
  [论文解读] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection
description: >-
  [CVPR 2026][视频理解][开放词汇时序动作检测] 提出 Phase-wise Decomposition and Alignment (PDA) 框架，利用 LLM 的 CoT 推理能力将动作标签分解为"开始-中间-结束"三个阶段描述，通过文本引导的前景过滤和自适应阶段对齐实现细粒度动作模式迁移…
tags:
  - "CVPR 2026"
  - "视频理解"
  - "开放词汇时序动作检测"
  - "链式思维提示"
  - "动作阶段分解"
  - "跨模态对齐"
  - "知识迁移"
---

# Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection

**会议**: CVPR 2026  
**arXiv**: [2603.24030](https://arxiv.org/abs/2603.24030)  
**代码**: 无  
**领域**: Video Understanding  
**关键词**: 开放词汇时序动作检测, 链式思维提示, 动作阶段分解, 跨模态对齐, 知识迁移

## 一句话总结
提出 Phase-wise Decomposition and Alignment (PDA) 框架，利用 LLM 的 CoT 推理能力将动作标签分解为"开始-中间-结束"三个阶段描述，通过文本引导的前景过滤和自适应阶段对齐实现细粒度动作模式迁移，在 THUMOS14 OV-TAD 上 Avg mAP 达 46.9（超越 SOTA Ti-FAD 的 41.2）。

## 研究背景与动机
**领域现状**: 开放词汇时序动作检测 (OV-TAD) 要求对未见过的动作类别进行定位和分类，核心是从已见类别迁移知识。

**现有痛点**: 现有方法仅进行标签级别的全局文本-视觉对齐，难以捕捉不同动作间共享的细粒度时序模式。例如 "LongJump" 和 "PoleVault" 标签相似度低，但起跑和起跳阶段视觉上高度相似。

**核心矛盾**: 标签级语义对齐无法发现跨类别的可迁移视觉模式，导致对未见类别的泛化能力有限。

**本文目标**: 如何提取和迁移不同动作之间共享的阶段级视觉先验，实现更好的开放词汇泛化。

**切入角度**: 模拟人类认知——理解一个动作是逐步展开的（启动→执行→完成），利用 LLM 的 CoT 能力自动将动作分解为多个阶段。

**核心 idea**: 将动作标签分解为阶段描述 → 每个阶段独立做文本-视觉对齐 → 自适应聚合各阶段的对齐结果。

## 方法详解

### 整体框架
三个核心模块：CSD（CoT 提示语义分解）→ TIF（文本引导前景过滤）→ APA（自适应阶段对齐）。输入视频经视觉编码器提取特征，动作标签经 GPT-4o 分解为 start/mid/end/global 四个阶段描述，每个阶段独立进行视觉-文本匹配并自适应聚合。

### 关键设计
1. **CoT-Prompting Semantic Decomposition (CSD)**: 使用 GPT-4o 的 CoT 推理将每个动作标签分解为 {start, middle, end, global} 四个阶段性描述。例如 "LongJump": start="跑道加速"，mid="蹬地起跳"，end="落地沙坑"。CLIP 文本编码器提取阶段嵌入 $t_c^p = \Phi_{txt}(s_c^p)$。设计动机：标签级语义无法表达跨类别共享的阶段模式，而阶段分解自然暴露了这些可迁移知识。

2. **Text-infused Foreground Filtering (TIF)**: 对每个阶段 $p$，用阶段文本嵌入与视频特征计算相似度，取 max 后 Softmax 得到阶段级前景置信度 $S_{fg}^p$，二值化后过滤出阶段相关的视频片段 $F_v^p = \hat{S}_{fg}^p \cdot F_v$。设计动机：仅简单均匀分割视频无法应对多动作、变时长的现实情况，需要自适应地按阶段语义选择视频片段。

3. **Adaptive Phase-wise Alignment (APA)**: 每个阶段独立做交叉注意力融合 $\bar{F}_v^p = \text{CrossAttn}(F_v^p, F_t^p)$，然后计算分类分数 $C_{cls}^p = \bar{F}_v^p \cdot F_t^{p\top}$。**自适应聚合**使用 Sigmoid 网络预测每个阶段的权重 $\omega_p$，最终分类 $C_{cls} = \sum_{p} \omega_p \cdot C_{cls}^p$。设计动机：不同动作中各阶段的判别性不同（有的动作看开头就能判断，有的需要看结尾），自适应权重优于简单平均。

### 损失函数 / 训练策略
- 总损失 $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{fg} + \mathcal{L}_{loc}$：分类（交叉熵）+ 前景感知 + DIoU 定位损失
- 推理时对测试类别同样用 LLM 做阶段分解，最后用 SoftNMS 去冗余

## 实验关键数据

### 主实验（THUMOS14, 50% Seen / 50% Unseen）

| 方法 | 0.3 | 0.5 | 0.7 | Avg mAP |
|------|-----|-----|-----|---------|
| Ti-FAD (NeurIPS'24) | 57.0 | 43.3 | 21.2 | 41.2 |
| STOV (WACV'25) | 56.3 | 34.4 | 11.3 | 34.0 |
| **PDA (Ours)** | **65.4** | **49.7** | **24.3** | **46.9** |

**ActivityNet v1.3**

| 方法 | 0.5 | 0.75 | Avg mAP |
|------|-----|------|---------|
| Ti-FAD | 50.6 | 32.2 | 32.0 |
| **PDA (Ours)** | **53.1** | **35.3** | **34.6** |

### 消融实验

| 配置 | Avg mAP | 说明 |
|------|---------|------|
| 全局对齐 baseline | ~41.2 | 仅标签级别对齐 |
| + CSD | 提升 | 阶段分解暴露可迁移模式 |
| + CSD + TIF | 进一步提升 | 自适应前景过滤替代静态时序分割 |
| + CSD + TIF + APA | **46.9** | 自适应权重优于平均聚合 |

### 关键发现
- 在 THUMOS14 50/50 划分下，与最强 baseline Ti-FAD 相比，Avg mAP 提升 5.7 个点。
- 在 LongJump→PoleVault 的跨类别迁移案例中，阶段分解后模型能识别出共享的"起跑加速"和"蹬地起跳"模式，显著提升未见类别的检测性能。
- 自适应聚合相比简单均值显示出更大的灵活性。

## 亮点与洞察
- 将 CoT 推理从 NLP 扩展到动作理解：不仅是文本增强，而是结构化的时序分解，直接关联动作的认知过程
- 阶段分解天然暴露了跨类别的可迁移知识，这是标签级方法无法做到的
- TIF 的文本引导式前景过滤优于静态时序分割，能处理多动作和变时长场景
- 在 LongJump→PoleVault 的迁移案例中，阶段分解后模型能识别共享的"起跑加速"和"蹬地起跳"模式
- 75/25 划分验证了方法在不同 seen/unseen 比例下的鲁棒性
- THUMOS14 上 IoU@0.5 的 mAP 从 43.3 提升到 49.7（+6.4%），说明细粒度对齐也提升了定位精度

## 局限与展望
- 依赖 GPT-4o 做阶段分解，成本较高且分解质量受限于 LLM 的动作知识。
- 固定三阶段分解（start/mid/end）可能对某些动作不够灵活（如周期性动作）。
- 未探索阶段数量的自适应确定。
- CLIP 文本编码器对阶段描述的编码质量可能成为瓶颈。- 未在更大规模视频数据集（如 Kinetics）上验证
- CoT 分解的质量在不同 LLM 间可能有较大差异

## 相关工作与启发
- 与 DeTAL、Ti-FAD 的区别：它们使用全局对齐或简单文本增强，本文通过结构化阶段分解实现细粒度知识迁移。
- CoT 提示在视觉任务中的应用是新兴方向，本文展示了其在时序理解中的潜力。
- 自适应阶段权重的设计可以推广到其他需要多粒度对齐的任务。
- 75/25 划分下 Avg mAP 从 Ti-FAD 的 42.9 提升到 47.3，泛化到不同比例

## 技术细节补充
- **GPT-4o Prompt**: "Decompose the action of ⟨Action⟩ into coherent three phases based on the natural temporal progression"
- **阶段文本模板**: 'a video of people's motion that [Description]'
- **交叉注意力融合**: $\bar{F}_v^p = \text{Softmax}(\frac{Q(F_v^p)K(F_t^p)^\top}{\sqrt{D}})V(F_t^p)$
- **自适应权重**: $\omega_p = \text{Sigmoid}(W_p(F_v^p))$，允许不同阶段在不同动作上有不同重要性
- **定位分支**: 拼接所有阶段视觉特征 → MLP 投影 → 前景感知头 + 回归头
- **推理**: 测试类别同样用 LLM 分解为阶段，SoftNMS 去冗余
- **75/25 划分结果**: THUMOS14 Avg mAP 47.3 (vs Ti-FAD 42.9)，ActivityNet Avg mAP 36.6 (vs DeTAL 25.5)
- **训练目标**: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{fg} + \mathcal{L}_{loc}$，分类+前景感知+DIoU 定位
- **阶段集合**: $\mathcal{P} = \{start, middle, end, glob\}$，共 4 个阶段
- **前景二值化阈值**: 取所有时间位置的平均相似度作为二值化阈值
- **适用视觉编码器**: 兼容 CLIP ViT-B/16 等标准视觉编码器

## 评分
- 新颖性: ⭐⭐⭐⭐ CoT + 阶段分解的思路新颖，从认知角度出发合理
- 实验充分度: ⭐⭐⭐⭐ THUMOS14 + ActivityNet 两个 benchmark，多种划分设置
- 写作质量: ⭐⭐⭐⭐ 动机图示直观，但公式较多
- 价值: ⭐⭐⭐⭐ 在 OV-TAD 上取得显著提升，但该任务的应用范围相对有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2025\] Context-Enhanced Memory-Refined Transformer for Online Action Detection](../../CVPR2025/video_understanding/context-enhanced_memory-refined_transformer_for_online_action_detection.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)

</div>

<!-- RELATED:END -->
