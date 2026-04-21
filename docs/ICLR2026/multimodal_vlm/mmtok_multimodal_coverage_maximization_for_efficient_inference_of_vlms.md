---
title: >-
  [论文解读] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
description: >-
  [ICLR 2026][多模态][vision token selection] 提出MMTok——一种基于最大覆盖问题（Maximum Coverage Problem）的多模态视觉token选择框架，同时利用文本-视觉和视觉-视觉覆盖信息来选择最具信息量的视觉token子集，在training-free设置下显著优于单模态baseline，甚至超越需要微调的方法。
tags:
  - ICLR 2026
  - 多模态
  - vision token selection
  - coverage maximization
  - submodular optimization
  - VLM efficiency
  - 剪枝
---

# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs

**会议**: ICLR 2026  
**arXiv**: [2508.18264](https://arxiv.org/abs/2508.18264)  
**代码**: 无  
**领域**: Multimodal / VLM  
**关键词**: vision token selection, coverage maximization, submodular optimization, VLM efficiency, token pruning

## 一句话总结

提出MMTok——一种基于最大覆盖问题（Maximum Coverage Problem）的多模态视觉token选择框架，同时利用文本-视觉和视觉-视觉覆盖信息来选择最具信息量的视觉token子集，在training-free设置下显著优于单模态baseline，甚至超越需要微调的方法。

## 研究背景与动机

视觉语言模型（VLM）将图像转化为视觉token后与文本token拼接输入LLM，但视觉token数量远超文本token（如LLaVA-NeXT中单张图像可产生2,880个视觉token，而"Describe the image"仅不到10个文本token）。由于LLM中自注意力机制的计算复杂度与token总数的平方成正比，大量视觉token严重制约推理效率。

现有视觉token选择方法的核心问题在于**仅利用单模态信息**：
- **纯视觉方法**（VisionZip、FastV）：通过视觉编码器内部的attention信号（如[CLS] token attention）排序，忽略了文本查询的语义引导
- **纯文本方法**（SparseVLM）：利用文本到视觉的attention评分，但忽略了图像全局信息

关键观察：**相同图像对不同文本查询需要不同的视觉token**（如"这是什么动物"vs"背景颜色是什么"），而**相同文本指令可应用于不同图像**（如caption任务）。因此，单模态方法天然是次优的，需要同时利用视觉和文本信息。

## 方法详解

### 整体框架

MMTok将视觉token选择形式化为**最大覆盖问题**，通过子模函数优化同时覆盖文本语义和视觉全局信息：

1. 计算文本-视觉相似度矩阵 $M^{tv}$ 和视觉-视觉相似度矩阵 $M^{vv}$
2. 联合优化多模态覆盖目标
3. 利用贪心算法高效求解（保证$(1-1/e)$近似比）

### 关键设计

1. **覆盖函数定义**：
    $f(\mathcal{S}; M) = \frac{1}{m} \sum_{i=1}^{m} \max M_{i,\mathcal{S}}$
   即对每个目标token，取其与被选token子集中最大相似度的平均值。该函数被证明是**子模函数**（Proposition 1），保证贪心算法可获得$(1-1/e) \approx 63.2\%$最优解的理论保证。

2. **文本-视觉覆盖（T-V Coverage）**：

    - 相似度矩阵 $M_{i,j}^{tv} = \mathbf{t}_i^\top \mathbf{v}_j$，使用投影后的视觉token（与文本对齐）
    - 目的：选出与文本查询语义最相关的视觉token
    - 局限：文本可能模糊（如"请描述图像"），语义引导不足

3. **视觉-视觉覆盖（V-V Coverage）**：

    - 相似度矩阵 $M_{i,j}^{vv} = \mathbf{v}_i^{\prime\top} \mathbf{v}_j'$，使用投影前的视觉特征（捕获纯视觉相似性）
    - 目的：选出能代表整幅图像信息的视觉token子集
    - 与T-V覆盖互补

4. **多模态覆盖融合**：

    - 先用softmax进行校准：$M_{i,j}^{tv'} = \frac{\exp(M_{i,j}^{tv}/\tau_t)}{\sum_j \exp(M_{i,j}^{tv}/\tau_t)}$
    - 联合目标：$f(\mathcal{S}; M^{tv'}, M^{vv'}) = f(\mathcal{S}; M^{tv'}) + \alpha \cdot f(\mathcal{S}; M^{vv'})$
    - **Corollary 1**：两个子模函数之和仍为子模函数，贪心算法仍然有效
    - 默认参数：$\tau_t=0.02$, $\tau_v=0.2$, $\alpha=0.5$

5. **可选的Agent增强文本**：

    - 使用轻量级VLM（SmolVLM2-256M）生成初步回答
    - 将回答token拼接到原始文本token后增强T-V覆盖的引导
    - 适用于文本查询信息不足的场景

### 算法复杂度

贪心算法（Algorithm 1/2）仅包含简单的矩阵运算（加法、乘法、取max），每步从剩余候选中选择边际增益最大的token，总共选k个token。实验表明其运行时间与VisionZip等简单方法几乎相同。

## 实验关键数据

### 主实验（LLaVA-1.5-7B，576原始token）

| 方法 | 192 tokens保留率 | 128 tokens保留率 | 64 tokens保留率 |
|------|----------------|----------------|----------------|
| FastV | 89.6% | 84.4% | 75.6% |
| SparseVLM | 95.5% | 92.9% | 86.9% |
| VisionZip | 97.9% | 96.8% | 93.2% |
| DivPrune | 98.0% | 97.0% | 94.8% |
| VisionZip🔥(微调) | 98.4% | 97.7% | 95.0% |
| **MMTok** | **98.7%** | **97.9%** | **96.5%** |

### 跨模型泛化

| 模型 | 配置 | VisionZip | DivPrune | MMTok |
|------|------|-----------|----------|-------|
| LLaVA-1.5-13B | 64 tokens | 93.7% | 95.4% | **96.3%** |
| LLaVA-NeXT-7B | Up 160 | 90.4% | 92.4% | **95.1%** |
| LLaVA-NeXT-13B | Up 160 | 91.4% | 92.1% | **95.1%** |
| Qwen-2.5-VL-7B | 20% | 94.2% | 91.5% | **94.6%** |

### 极端压缩（LLaVA-1.5-7B，高IC数据集）

| token数 | VisionZip | DivPrune | MMTok |
|---------|-----------|----------|-------|
| 16 | 78.3% | 86.2% | **88.3%** |
| 8 | 63.2% | 76.3% | **82.9%** |
| 4 | 58.8% | 66.3% | **76.7%** |
| 2 | 57.8% | 63.5% | **70.0%** |

在POPE上仅用4个token就保留了87.7%的原始性能！

### 消融实验

| 配置 | 64 tokens保留率 | 说明 |
|------|----------------|------|
| T-V only（无softmax） | 93.7% | 仅用文本引导 |
| V-V only（无softmax） | 94.7% | 仅用视觉自覆盖 |
| T-V（softmax校准） | 93.8% | 校准不损性能 |
| V-V（softmax校准） | 95.7% | 校准有小幅提升 |
| **MMTok（T-V + V-V）** | **96.6%** | 多模态互补显著 |

### 推理效率（LLaVA-NeXT-13B，H100 GPU）

| 方法 | 总推理时间 | POPE时间 | GPU利用率 | 运行时内存 | 平均性能 |
|------|----------|---------|----------|----------|---------|
| 原始(2880) | 15204s | 1705s | 86.7% | 4.59GB | 100% |
| VisionZip(160) | 7551s | 866s | 52.4% | 1.92GB | 89.6% |
| DivPrune(160) | 8186s | 1060s | 50.9% | 1.23GB | 90.5% |
| **MMTok(160)** | **7768s** | **913s** | 58.0% | 1.78GB | **93.7%** |

实现1.87×加速同时在POPE上保持98.7%性能。

### 关键发现

- **多模态信息互补**：T-V和V-V覆盖的结合比任何单模态方法好2-3%
- **Image Contribution（IC）指标**的提出：部分数据集即使零视觉token也有很高性能（如SQA 82%、MMMU 92%），说明评估应重点关注高IC数据集
- **超参鲁棒性**：$\tau_t, \tau_v, \alpha$的选择对性能影响不大，固定默认值即可
- **Training-free优势**：无需微调即超越VisionZip🔥等微调方法
- **极端压缩潜力**：4个token时MMTok比VisionZip高18%，说明覆盖准则在极端情况下优势更大

## 亮点与洞察

- **将token选择形式化为经典组合优化问题**：子模函数+贪心算法的理论框架优雅且实用
- **投影前后特征的差异化使用**：投影后特征用于跨模态对齐（T-V），投影前特征用于纯视觉相似度（V-V），体现了对VLM架构的深入理解
- **IC指标对evaluation的反思**：指出SQA、MMMU等数据集不适合评估视觉token选择质量
- **Agent增强**：轻量级VLM的预回答作为辅助信号，思路新颖但效果因任务而异

## 局限与展望

- 当前仅在LLM输入前选择token，LLM推理过程中的token动态剪枝未探索
- Agent方法对多选题QA效果不佳（Agent回答如"A"对token选择无意义引导）
- 在Qwen-2.5-VL（已有token merging层）上的提升相对较小，说明对已优化模型的增量价值有限
- 贪心算法虽有理论保证，但可能存在更优的优化策略
- 未探索视频理解等多帧场景下的扩展

## 相关工作与启发

本文将**子模函数优化**（经典组合优化理论）引入VLM加速场景。与VisionZip（[CLS] attention排序）、FastV（层内attention剪枝）、SparseVLM（text-vision attention）、DivPrune（多样性准则）等方法相比，覆盖准则的独特之处在于其**同时优化相关性和覆盖度**。SmolVLM2-256M的agent使用暗示了小模型辅助大模型推理的有趣方向。

## 评分

- 新颖性: ⭐⭐⭐⭐ （覆盖准则+多模态融合新颖，但token pruning本身不新）
- 实验充分度: ⭐⭐⭐⭐⭐ （5个VLM×9个数据集×多压缩比+极端压缩+效率分析）
- 写作质量: ⭐⭐⭐⭐ （方法清晰，理论保证完整，实验详实）
- 价值: ⭐⭐⭐⭐ （实用性强，training-free+参数鲁棒，易于部署）

<!-- RELATED:START -->

## 相关论文

- [Multimodal Classification via Total Correlation Maximization](multimodal_classification_via_total_correlation_maximization.md)
- [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](../../ICML2025/multimodal_vlm/sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)
- [SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_infe.md)

<!-- RELATED:END -->
