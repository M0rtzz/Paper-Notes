---
title: >-
  [论文解读] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention
description: >-
  [CVPR2026][多模态][多模态幻觉] 提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。
tags:
  - CVPR2026
  - 多模态
  - 多模态幻觉
  - 视觉注意力
  - 视觉语义置信度
  - 免训练
  - 注意力机制
---

# Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention

**会议**: CVPR2026  
**arXiv**: [2511.20032](https://arxiv.org/abs/2511.20032)  
**代码**: [github.com/beta-nlp/VGA](https://github.com/beta-nlp/VGA)  
**领域**: 多模态VLM  
**关键词**: 多模态幻觉, 视觉注意力, 视觉语义置信度, 免训练, FlashAttention兼容

## 一句话总结
提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。

## 研究背景与动机
MLLM虽然在视觉理解上取得显著进展，但经常产生与实际视觉内容矛盾的幻觉输出。现有去幻觉方法主要分为训练方法和免训练方法：
- **训练方法**：构建数据集或设计损失函数，但模型架构迭代太快导致边际递减
- **免训练方法**：更具实用价值，尤其是优化视觉注意力的方向

当前视觉注意力优化方法的核心问题：
1. 过度依赖注意力本身的质量，但视觉注意力的定位能力本质上有限（受attention sink现象影响）
2. 使用外部工具或额外前向传播引入计算开销
3. 依赖attention weight的方法与FlashAttention不兼容

关键发现：模型能准确提取视觉token的语义特征并转化为条件概率（visual logits），但推理阶段未能充分利用这一优势。这意味着MLLM的视觉理解被低估了。

## 方法详解

### 整体框架
VGA分两步：(1) 通过Visual Semantic Confidence (VSC)构建视觉定位 → (2) 用定位引导视觉注意力。每个token仅需一次前向传播。

### 关键设计

1. **Visual Semantic Confidence (VSC)**：

    - 视觉token $v_i$ 对物体O的语义置信度：$c_{v_i}(O) = \text{softmax}[\text{logit}_{v_i}(O)]$
    - 用物体O的第一个token化token $o_0$ 近似
    - 物体O对整幅图像的置信度用最大池化：$c(O) = \max c_{v_i}(o_0)$
    - 视觉定位：$G_O = \text{Norm}[\{c_{v_i}(o_0)\}_{i=1}^m]$
    - 实验验证：VSC的定位能力显著优于视觉注意力机制，在大物体上尤其明显（不受attention sink影响）

2. **Visual Semantic Salience (VSS)**——面向图像描述的无目标定位：

    - 对于captioning等无特定目标的任务，用输出不确定性衡量视觉token的语义显著性
    - $c_{v_i} = -\sum_k \log c_{v_i}(w_k) / \log K$（Top-K token的熵）
    - 高VSS的token对应有意义的物体区域，低VSS对应语义不显著的背景

3. **Vision-Guided Attention (VGA)**：

    - 核心公式：$\hat{z} = z + \beta \cdot \gamma \cdot \Delta z$
    - 其中 $\Delta z$ 是引导信号，$\beta$ 是引导强度，$\gamma$ 是注意力头平衡系数
    - 关键特性：VGA不需要计算attention weight → 完全兼容FlashAttention
    - 利用加法结合律：$\hat{z} = (\alpha + \beta \cdot G)V = z + \beta \cdot \Delta z$

4. **Attention Heads Balancing**：

    - 视觉功能较强的头给予较弱引导，非视觉头给予较强引导
    - 通过z和Δz的余弦相似度近似头的视觉功能差异
    - $\gamma = \text{ReLU}(2 - H \cdot \gamma')$

5. **Programmed Visual Grounding (PVG)**——面向captioning的动态引导：

    - 随生成进行动态更新：$G_{t+1} = (1+\lambda)G_t - \lambda G_w$
    - 抑制已描述区域，引导关注待描述区域
    - 随生成内容增多，引导强度自动衰减：$\|G\|_0$ 用作衰减因子

### 损失函数 / 训练策略
完全免训练方法，仅在推理时应用。超参数包括引导强度β和衰减参数λ。

## 实验关键数据

### 主实验

| 数据集 | 指标 | VGA | 之前SOTA | 提升 |
|--------|------|-----|----------|------|
| POPE (Acc, 平均) | Accuracy | SOTA | 多个基线 | 在LLaVA-7B/13B/Next和Qwen2.5-VL上全面领先 |
| POPE (F1, 平均) | F1 | SOTA | PAI/PAICD等 | 跨模型一致性提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅PSP | 提升 | 验证位置-时间步惩罚效果 |
| VGA在不同MLLM上 | 一致性提升 | 方法的通用性强 |
| VSC定位 vs 注意力定位 | Dice系数大幅领先 | 尤其在大物体上优势明显 |

### 关键发现
- VSC的判断准确率虽低于模型本身回答，但展示了正确的偏好性（显著超过50%）
- VSC与模型回答存在一定偏好差异，证明模型的视觉理解未被充分利用
- VGA在不新增前向传播的前提下（每个token仅一次），实现了去幻觉SOTA

## 亮点与洞察
- 核心洞察极为精彩：MLLM的视觉logits蕴含丰富的语义定位信息，但推理时未被充分利用
- 方法设计优雅：利用加法结合律绕过attention weight计算，实现FlashAttention兼容
- Attention Heads Balancing是实用的设计，避免破坏模型原有的视觉功能头
- PVG为captioning场景提供了动态attention引导的有效范式

## 局限与展望
- VSC使用第一个token近似物体语义可能不够精确，尤其对多音节/多token物体
- 超参数β需要手动设置，不同模型/任务可能需要调整
- 未与训练方法结合，可能存在互补提升空间
- PVG的衰减策略较为启发式，可能对长描述不够稳定

## 相关工作与启发
- 对比解码方法(VCD, ICD等)通常需要额外前向传播来激活幻觉特征
- Attention编辑方法(PAI, OPERA等)依赖attention weight，不兼容FlashAttention
- VGA成功将视觉语义置信度作为一种新型视觉先验引入注意力引导，这一思路可推广到其他需要精确视觉定位的任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ VSC是全新的概念，FlashAttention兼容的设计非常实用
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准验证，定量+定性分析
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从观察到方法推导自然流畅
- 价值: ⭐⭐⭐⭐⭐ 免训练+FlashAttention兼容，落地价值极高

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)
- [\[CVPR 2026\] GACD: Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](gacd_gradient_self_reflection_hallucination.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)

<!-- RELATED:END -->
