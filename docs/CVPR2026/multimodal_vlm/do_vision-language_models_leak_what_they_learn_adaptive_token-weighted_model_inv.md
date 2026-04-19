---
title: >-
  [论文解读] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks
description: >-
  [CVPR2026][多模态][模型逆向攻击] 首次系统研究 VLM 的模型逆向（Model Inversion）攻击，提出基于自适应 token 注意力权重的序列级逆向方法 SMI-AW，通过动态加权视觉关联度不同的 token 梯度，从 VLM 中重建隐私训练图像，人类评估攻击准确率达 61.21%。
tags:
  - CVPR2026
  - 多模态
  - 模型逆向攻击
  - 视觉-语言模型
  - 隐私安全
  - 自适应token加权
  - 训练数据泄露
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks

**会议**: CVPR2026  
**arXiv**: [2508.04097](https://arxiv.org/abs/2508.04097)  
**代码**: [项目主页](https://ngoc-nguyen-0.github.io/SMI_AW/)  
**领域**: 多模态VLM  
**关键词**: 模型逆向攻击, 视觉-语言模型, 隐私安全, 自适应token加权, 训练数据泄露

## 一句话总结

首次系统研究 VLM 的模型逆向（Model Inversion）攻击，提出基于自适应 token 注意力权重的序列级逆向方法 SMI-AW，通过动态加权视觉关联度不同的 token 梯度，从 VLM 中重建隐私训练图像，人类评估攻击准确率达 61.21%。

## 背景与动机

1. **模型逆向攻击的隐私风险**：MI 攻击能从训练好的模型参数中重建隐私训练数据，已在单模态 DNN（如分类器）上被广泛研究，但多模态 VLM 的安全性尚未被探索。
2. **VLM 大规模部署的安全隐患**：LLaVA、Qwen2.5-VL、InternVL 等 VLM 被广泛应用于医疗、金融等敏感领域，若存在训练数据泄露风险则后果严重。
3. **VLM 与 DNN 的本质差异**：VLM 处理多模态输入、包含多个独立模块（视觉编码器、投影器、语言模型）、输出为 token 序列而非类别标签，传统 MI 不能直接套用。
4. **视觉编码器通常冻结**：许多 VLM 在微调时冻结视觉编码器，只更新语言模型和投影器，这意味着隐私信息嵌入在语言模型参数中，逆向攻击面临间接提取的挑战。
5. **token 序列的异质性**：VLM 输出的不同 token 对视觉输入的依赖程度差异很大——有的强视觉关联，有的仅由语言上下文驱动，统一处理会引入噪声。
6. **公开发布模型也有风险**：已公开的预训练 VLM（如 LLaVA-v1.6-7B）同样可能被逆向攻击重建其训练数据中的人脸等敏感信息。

## 方法详解

### 整体框架

攻击者拥有白盒访问权限（模型参数、注意力图、logits），利用公开数据集训练的生成模型 $G$（如 StyleGAN2）在低维隐空间 $w$ 中优化，通过 $x = G(w)$ 重建隐私图像。目标：找到 $w^*$ 使 VLM 对重建图像 $G(w^*)$ 的输出与目标文本答案 $\mathbf{y}$ 高度匹配。

### 四种逆向策略

1. **TMI（Token-based MI）**：逐 token 更新隐向量 $w$，每个 token $y_i$ 独立计算梯度并更新一次。
2. **TMI-C（Convergent TMI）**：对每个 token $y_i$ 进行 $K$ 次迭代收敛后再处理下一个 token，更好地对齐 token 间的生成依赖。
3. **SMI（Sequence-based MI）**：对整个序列的 token 损失取平均，单次梯度更新聚合全序列信息，提供更稳定一致的优化方向：
$$\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}_{inv}(M(\mathbf{t}, G(w), y_{<i}), y_i)$$
4. **SMI-AW（核心方法）**：在 SMI 基础上引入自适应 token 权重。

### SMI-AW 自适应加权设计

- **核心观察**：不同输出 token 的交叉注意力图差异显著——视觉关联强的 token 注意力响应高，其梯度含更丰富的视觉信息。
- **权重计算**：用 token $y_i$ 对图像的总视觉注意力值 $\alpha_i$ 计算归一化权重：
$$\beta_i = \frac{\alpha_i}{\sum_{j=1}^{m}\alpha_j}$$
- **动态更新**：$\beta_i$ 在每个逆向步骤中重新计算，因为随重建图像逐渐逼近目标，token 的视觉依赖程度会变化。
- **加权损失**：
$$\mathcal{L} = \sum_{i=1}^{m}\beta_i \mathcal{L}_{inv}(M(\mathbf{t}, G(w), y_{<i}), y_i)$$

### 逆向损失函数

扩展传统单模态 MI 的三种损失到 VLM 场景：

- $\mathcal{L}_{CE}$：交叉熵损失
- $\mathcal{L}_{MML}$：最大间隔损失
- $\mathcal{L}_{LOM}$：logit 最大化损失（效果最佳）

## 实验关键数据

### 实验设置

- **目标 VLM**：LLaVA-v1.6-7B、Qwen2.5-VL-7B、MiniGPT-v2、InternVL2.5-8B
- **数据集**：FaceScrub (530类)、CelebA (1000类)、StanfordDogs (120类)，构造 VQA 格式
- **生成模型**：StyleGAN2 (FFHQ)、AFHQ-Dogs 训练的生成器
- **评估指标**：$AttAcc_M$（MLLM评估）、$AttAcc_D$ Top1/Top5（DNN评估）、$\delta_{face}$、$\delta_{eval}$、$AttAcc_H$（人类评估）

### 主实验结果（FaceScrub + LLaVA-v1.6-7B + $\mathcal{L}_{LOM}$）

| 方法 | $AttAcc_M$↑ | Top1↑ | Top5↑ | $\delta_{face}$↓ | $\delta_{eval}$↓ |
|------|------------|-------|-------|------------------|------------------|
| TMI | 44.34% | 21.77% | 44.69% | 0.8488 | 141.87 |
| TMI-C | 31.16% | 9.32% | 24.22% | 1.0221 | 457.49 |
| SMI | 59.17% | 33.47% | 61.89% | 0.7465 | 140.83 |
| **SMI-AW** | **61.01%** | **37.62%** | **66.16%** | **0.7265** | **134.94** |

### 跨数据集结果（SMI-AW + $\mathcal{L}_{LOM}$）

| 数据集 | $AttAcc_M$↑ | Top1↑ | Top5↑ |
|--------|------------|-------|-------|
| CelebA | 67.05% | 45.25% | 69.55% |
| StanfordDogs | 78.13% | 56.15% | 84.79% |

### 跨模型泛化（FaceScrub + SMI-AW + $\mathcal{L}_{LOM}$）

| 模型 | $AttAcc_M$↑ | Top1↑ | Top5↑ |
|------|------------|-------|-------|
| InternVL2.5-8B | 55.05% | 25.05% | 52.10% |
| MiniGPT-v2 | 47.92% | 14.62% | 33.82% |
| Qwen2.5-VL-7B | 32.03% | 13.21% | 27.24% |

### 人类评估（$AttAcc_H$）

不同 VLM 在 FaceScrub 上的人类攻击准确率为 53.42%–57.22%，CelebA 上达 **61.21%**，StanfordDogs 为 55.42%。

### 消融分析：Token-based vs. Sequence-based

- 序列级方法（SMI/SMI-AW）的 match rate > 95%，远超 token 级方法（TMI 60–79%，TMI-C < 30%）。
- Token 级方法梯度方差大、不稳定，弱视觉关联 token 的梯度会误导优化方向。
- SMI-AW 的自适应权重在每步动态更新，放大视觉关联 token 的贡献，抑制语言驱动 token 的噪声。

## 亮点

- **首创性**：首次系统研究 VLM 的模型逆向攻击，填补多模态隐私安全的重要空白。
- **方法设计精妙**：利用交叉注意力图作为 token 视觉关联度的代理指标，自适应加权思路自然且有效。
- **评估全面**：4 个 VLM × 3 个数据集 × 5 种指标（含大规模人类评估），实验规模充分。
- **实际威胁验证**：在公开发布的 LLaVA-v1.6-7B 上成功重建名人面部，说明隐私风险真实存在。
- **序列 vs. token 方法的深入分析**：不仅提方法还给出了为什么 token 级方法失效的机制性解释。

## 局限与展望

- **白盒假设较强**：需要完整访问模型参数和注意力图，实际部署场景中 API 访问（黑盒）更常见。
- **生成模型依赖**：需要在与隐私数据同域的公开数据上训练 StyleGAN，对新域或稀缺域的适用性受限。
- **VQA 场景特定**：实验基于 VQA 格式构造的数据，对话式、指令式等更复杂 VLM 场景未验证。
- **防御方案缺失**：论文揭示了风险但未提出防御策略（如差分隐私、梯度裁剪等对抗手段）。
- **计算开销**：每步需计算完整序列的前向传播和注意力图提取，对长序列的扩展性需考量。

## 与相关工作的对比

- **传统 MI（PPA、PLG-MI 等）**：仅针对单模态分类器，优化单一类别标签的似然；本文针对 token 序列输出的 VLM，需全新设计。
- **VLM 安全研究**：已有越狱攻击、对抗样本等方向，但模型逆向（训练数据重建）在 VLM 上是首次系统研究。
- **注意力机制分析**：已有工作发现 VLM 后层 token 对图像的注意力衰减，本文将此观察用于指导 MI 攻击的梯度加权。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 MI 攻击拓展到 VLM，问题定义和方法设计均为开创性工作
- 实验充分度: ⭐⭐⭐⭐⭐ — 4 模型 × 3 数据集 × 多指标 × 人类评估 × 公开模型验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，从 TMI→TMI-C→SMI→SMI-AW 层层递进，但公式较多
- 价值: ⭐⭐⭐⭐⭐ — 揭示 VLM 的严重隐私风险，对安全社区有重要警示意义

<!-- RELATED:START -->

## 相关论文

- [AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)
- [Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)
- [Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](what_do_visual_tokens_really_encode_uncovering_sparsity_and_redundancy_in_multim.md)
- [Revisiting Model Stitching In the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model_era.md)

<!-- RELATED:END -->
