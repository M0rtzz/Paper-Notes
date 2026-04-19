---
title: >-
  [论文解读] Don't Lag, RAG: Training-Free Adversarial Detection Using RAG
description: >-
  [ICML 2025 (VecDB Workshop)][对抗样本检测] 本文提出 VRAG 框架，通过构建对抗补丁数据库 + 视觉检索增强生成（VRAG）+ VLM 推理的免训练 pipeline，实现对多种对抗补丁攻击的高效检测，Gemini-2.0 达到 98% 准确率，开源模型 UI-TARS-72B-DPO 达 95%。
tags:
  - ICML 2025 (VecDB Workshop)
  - 对抗样本检测
  - 对抗补丁攻击
  - 视觉RAG
  - VLM推理
  - 免训练防御
---

# Don't Lag, RAG: Training-Free Adversarial Detection Using RAG

**会议**: ICML 2025 (VecDB Workshop)  
**arXiv**: [2504.04858](https://arxiv.org/abs/2504.04858)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 对抗样本检测, 对抗补丁攻击, 视觉RAG, VLM推理, 免训练防御

## 一句话总结
本文提出 VRAG 框架，通过构建对抗补丁数据库 + 视觉检索增强生成（VRAG）+ VLM 推理的免训练 pipeline，实现对多种对抗补丁攻击的高效检测，Gemini-2.0 达到 98% 准确率，开源模型 UI-TARS-72B-DPO 达 95%。

## 研究背景与动机

**领域现状**: 深度学习模型（CNN、ViT）在计算机视觉任务中表现卓越，但对**对抗补丁攻击**（adversarial patch attacks）高度脆弱。对抗补丁是一种局部化的高幅度扰动，可以被打印并放置在真实场景中，在不同光照和视角下仍可导致模型误分类。

**现有痛点**:
   - **监督防御**（如训练分类器区分对抗/正常样本）依赖标注数据，泛化性差
   - **无监督防御**（如Feature Squeezing）需精细调参，易被自适应攻击绕过
   - **对抗训练**计算开销大且容易过拟合到特定攻击类型
   - **扩散模型防御**（如DIFFender）计算密集，实时性差
   - 所有以上方法都需要**某种形式的训练/微调**，部署时无法灵活适配新攻击

**核心矛盾**: 传统防御要么需要训练（不够灵活），要么是启发式的（不够准确）。如何在**完全不训练**的情况下检测多种类型的对抗补丁？

**本文目标**: 构建一个**免训练、可扩展、基于检索增强的对抗补丁检测框架**，能动态适配不断演化的攻击。

**切入角度**: 将对抗补丁检测看作**视觉检索 + VLM 推理**问题——用数据库存储已知攻击模式，用检索找到最相似的攻击，再用 VLM 进行分类判断。

**核心 idea**: 用 RAG 范式连接对抗补丁数据库和 VLM，实现免训练的上下文感知对抗检测。

## 方法详解

### 整体框架
VRAG 检测 pipeline（如 Figure 2 所示）分四步：
1. **图像预处理**: 将输入图像 $I$ 划分为 $n \times n$ 网格区域 $\{C_1, \ldots, C_{n^2}\}$
2. **特征提取**: 用预训练视觉编码器（如 CLIP）将每个区域编码为嵌入 $E_i = f(C_i)$
3. **检索步骤**: 对每个 $E_i$，在对抗补丁数据库 $\mathcal{D}$ 中做 top-$k$ 近邻搜索
4. **VLM 生成推理**: 将检索到的相似补丁/攻击图像作为 few-shot 上下文，结合结构化 prompt，让 VLM 判断"该区域是否包含对抗补丁？"

### 关键设计

1. **对抗补丁数据库构建（Database Creation, Algorithm 1）**:

    - 聚合多种攻击方法生成的补丁：SAC、BBNP、标准对抗补丁攻击
    - 将每个补丁 $P_i$ 在随机位置和尺度下放置到多张自然图像上
    - 将每张打补丁图像分为 $n \times n$ 网格，计算每个区域的 CLIP 嵌入
    - 补丁嵌入作为 **key**，区域嵌入作为 **value**，构成键值数据库
    - 数据库可**持续扩展**——遇到新攻击类型只需添加新补丁
    - **设计动机**: 以嵌入相似度而非几何假设为基础的检索，可以自然泛化到各种形状（方形、圆形、三角形、自然伪装型）的补丁。数据库方案还使得系统可增量更新

2. **VRAG 检测 Pipeline（Algorithm 2）**:

    - 对输入图像的每个网格区域，计算与数据库中补丁嵌入的**余弦相似度**
    - 设定阈值 $\tau = 0.77$（通过 ROC-AUC 分析确定最优）
    - 相似度超过阈值的区域被标记为"可疑"
    - 对可疑区域，检索 top-$k=2$ 最相似的补丁和对应攻击图像
    - 构建结构化 prompt："这是对抗补丁的示例 [Patch 1], [Patch 2]。这是包含这些补丁的图像 [Image 1], [Image 2]。基于上下文，以下图像是否包含对抗补丁？回答 'yes' 或 'no'。"
    - VLM 生成回答，据此做最终判定
    - **设计动机**: 先用高效的嵌入检索缩小候选范围，再用 VLM 的强大推理能力做精确判断——两阶段设计兼顾效率和准确性

3. **Zero-Shot 与 Few-Shot 决策机制**:

    - **Zero-Shot**: VLM 仅依赖指令 prompt 和预训练知识判断，无检索支持
    - **Few-Shot**: 检索到的 $k$ 个相似示例被注入 prompt，VLM 获得攻击模式的视觉参考
    - 实验表明 **4-shot** 在准确率和效率之间达到最佳平衡，更多 shot 收益递减
    - **设计动机**: 与标准 RAG 中检索文档增强 LLM 类似，这里检索图像增强 VLM

### 损失函数 / 训练策略
**完全免训练**。所有 VLM 和编码器均使用原始权重，无微调。这是本方法最大的部署优势。

## 实验关键数据

### 主实验（APRICOT 数据集，真实物理对抗补丁）
| 方法 | 25×25 (0S/2S/4S) | 45×45 (0S/2S/4S) | 65×65 (0S/2S/4S) |
|------|-------------------|-------------------|-------------------|
| Undefended | 34.6/–/– | 30.2/–/– | 28.6/–/– |
| JPEG Compression | 29.4/–/– | 35.3/–/– | 41.1/–/– |
| Spatial Smoothing | 33.6/–/– | 39.2/–/– | 42.3/–/– |
| SAC | 45.9/–/– | 49.1/–/– | 52.8/–/– |
| DIFFender | 65.1/–/– | 68.6/–/– | 70.9/–/– |
| **Ours (UI-TARS-72B)** | 49.4/80.2/**91.6** | 51.6/83.6/**94.5** | 55.0/85.9/**96.2** |
| **Ours (Gemini-2.0)** | 56.2/82.6/**93.9** | 58.8/86.9/**96.8** | 63.1/90.3/**97.9** |

### 消融实验
| 配置 | ImageNet-Patch 准确率 | 说明 |
|------|---------------------|------|
| 余弦相似度检索 | **98.0%** | 最优距离度量 |
| L2 距离检索 | 89.8% | 次优 |
| L1 距离检索 | 86.3% | 不如余弦 |
| Wasserstein 距离检索 | 84.3% | 效果最差 |
| Prompt: 仅指令 | 58.0% | 无上下文，效果很差 |
| Prompt: 补丁 + 攻击图像 (Combined) | **98.0%** | 提供完整上下文最有效 |
| Prompt: 仅攻击图像 | 85.5% | 缺少补丁信息 |
| Prompt: Chain-of-Thought | 91.3% | 推理增强有帮助 |
| 0-shot / 2-shot / 4-shot / 6-shot | 56/87/98/98% | 4-shot 即饱和 |

### 关键发现
1. **免训练方法首次超越需训练的传统防御**: 在 APRICOT 上，4-shot VRAG (Gemini) 达到 ~98%，远超 DIFFender 的 ~71%
2. **开源模型 UI-TARS-72B-DPO 表现出色**: 达到 95% 准确率，创开源对抗检测新 SOTA
3. **推理时间可控**: Gemini-2.0 仅需 2.25 秒/图像，与 DIFFender 的 7.98 秒 相比效率更高
4. **数据库可扩展性**: 并行化从 1 worker 的 24.6 分钟降至 6 worker 的 3.6 分钟（6.86x 加速）
5. **Prompt 设计至关重要**: 同时提供补丁和攻击图像的 combined prompt 比纯指令 prompt 提高 40 个百分点

## 亮点与洞察
- **RAG 范式在视觉安全领域的成功应用**: 将文本 RAG 的思想迁移到视觉对抗检测，是一次巧妙的跨域迁移
- **免训练 + 可扩展的防御范式**: 新攻击出现时只需向数据库添加新补丁，无需重新训练
- **设计启示**: Prompt engineering 在 VLM-based 防御中的重要性——结构化的 prompt 设计可以带来 40pp 的准确率提升
- **实用性强**: 框架简单、推理快、无需 GPU 训练，适合实际部署

## 局限与展望
- 依赖预构建的补丁数据库——对完全未见过的新型攻击可能检索不到相似补丁
- 阈值 $\tau$ 的选择比较敏感，接近 1.0 时误报率高
- 对补丁与背景高度融合（如自然伪装型补丁）时检测能力下降
- 未探讨对抗训练与 VRAG 结合的混合策略
- Gemini-2.0 虽最强但闭源，实际部署可能需依赖开源模型

## 相关工作与启发
- 将 RAG 从 NLP 推广到视觉安全领域的有趣尝试
- CLIP 嵌入作为通用视觉特征在安全检测中也表现出色
- 启发：对于快速演化的威胁，**检索增强**范式天然适合——比重训练模型快得多
- 可类比推广到其他安全检测任务（如恶意二维码、虚假图像检测）

## 评分
- 新颖性: ⭐⭐⭐⭐ 视觉RAG用于对抗检测的思路新颖，但技术框架相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、四种攻击、四种VLM、多种消融实验，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法描述详尽
- 价值: ⭐⭐⭐⭐ 免训练范式实用性强，但workshop论文影响力有限

<!-- RELATED:START -->

## 相关论文

- [SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](../../NeurIPS2025/information_retrieval/secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)
- [VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](../../ACL2025/information_retrieval/voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](../../NeurIPS2025/information_retrieval/hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)
- [RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](../../NeurIPS2025/information_retrieval/rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [Investigating Language Preference of Multilingual RAG Systems](../../ACL2025/information_retrieval/investigating_language_preference_of_multilingual_rag_systems.md)

<!-- RELATED:END -->
