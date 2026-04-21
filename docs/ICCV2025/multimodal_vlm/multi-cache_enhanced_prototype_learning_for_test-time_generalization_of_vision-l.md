---
title: >-
  [论文解读] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models
description: >-
  [ICCV 2025][多模态][测试时适应] 提出 MCP/MCP++ 多缓存增强的原型学习框架，通过 entropy cache、align cache 和 negative cache 三种互补缓存机制构建紧致的类内分布，并引入跨模态残差学习进一步优化视觉和文本原型对齐，在 15 个下游任务上实现了 SOTA 的零样本泛化性能。
tags:
  - ICCV 2025
  - 多模态
  - 测试时适应
  - 视觉-语言模型
  - CLIP
  - 缓存机制
  - 原型学习
---

# Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2508.01225](https://arxiv.org/abs/2508.01225)  
**代码**: [Project Page](https://zhaihaotian.github.io/MCP-ICCV25/)  
**领域**: 多模态VLM  
**关键词**: 测试时适应, 视觉-语言模型, CLIP, 缓存机制, 原型学习

## 一句话总结

提出 MCP/MCP++ 多缓存增强的原型学习框架，通过 entropy cache、align cache 和 negative cache 三种互补缓存机制构建紧致的类内分布，并引入跨模态残差学习进一步优化视觉和文本原型对齐，在 15 个下游任务上实现了 SOTA 的零样本泛化性能。

## 研究背景与动机

视觉-语言预训练模型（如 CLIP）具有强大的零样本能力，但在实际部署时面临**分布偏移**问题——预训练数据与测试数据之间的差异导致性能下降。**测试时适应（Test-Time Adaptation, TTA）** 旨在推理阶段利用无标签测试数据快速适应新分布。

现有 TTA 方法的不足：

**Prompt-based 方法**（如 TPT）：通过一致性约束学习领域特定提示，有效但**计算代价高**

**Cache-based 方法**（如 TDA）：通过存储低熵样本构建缓存来提高适应性，效率高但**依赖关键假设**——低熵样本来自同类且类内紧致

作者通过关键实验发现：**缓存增强性能与类内紧致度（intra-class compactness）呈正相关**（Pearson $r > 0.8, p = 2.25 \times 10^{-3}$）。在类分布紧致的数据集（如 EuroSAT）上，TDA 提升显著；而在分散数据集（如 Aircraft）上，提升有限。这一发现揭示了改进缓存方法的关键方向——**提升类内紧致度**。

## 方法详解

### 整体框架

MCP 由三个互补的缓存模块组成：entropy cache 初始化原型、align cache 增强紧致度、negative cache 校准预测。MCP++ 在此基础上增加了跨模态原型残差学习。

### 关键设计

1. **Entropy Cache（熵缓存）**：
   动态存储每个类别中最低熵的样本，为类别表示提供稳定的锚点。每类维护固定容量 $M$ 的队列（$M \ll |X_{\text{test}}|$），新测试样本仅在其熵低于队列中最大熵时才替换最大熵样本。作用：防止原型漂移，锚定高置信度特征。

2. **Align Cache（对齐缓存）**：
   整合视觉和文本信息构建更紧致的类内分布。首先构建类别原型中心，融合文本原型 $\bar{t}_c$ 和视觉原型 $\bar{v}_c$：
   $$\mu_c = w \cdot \bar{v}_c + (1-w) \cdot \bar{t}_c$$
   其中 $w = 0.8$ 平衡两种模态的贡献。对齐缓存的准入条件更严格：不仅要求低熵（$H(x) < H_{\max}^{\hat{y}}$），还要求样本到原型中心的距离小于当前缓存中最远样本的距离（$d(f_{\text{test}}, \mu_{\hat{y}}) < d(f_{\max}, \mu_{\hat{y}})$）。这确保缓存始终保留最靠近原型中心的可靠样本。

3. **Negative Cache（负缓存）**：
   利用高熵样本的负伪标签来校准预测。引入**反射机制（reflecting mechanism）**——通过已有缓存中可靠样本的特征重新评估高熵样本的伪标签，确保只有校准后仍保持中等熵的样本才入库：
   - $H_{\text{low}} \le H'(x) \le H_{\text{high}}$：存入负缓存
   - $H'(x) < H_{\text{low}}$：视为可靠样本，可存入熵缓存
   - $H'(x) > H_{\text{high}}$：丢弃

4. **推理机制**：
   最终预测 logits 由三个互补信息源加权组合：
   $$p(f_{\text{test}}) = \alpha_1 \cdot f_{\text{test}} \mathcal{T}^\top + \alpha_2 \cdot P(f_{\text{test}}, \mathcal{V}, Q_n) + \alpha_3 \cdot f_{\text{test}} f_r^\top$$
   其中 $\mathcal{T}$ 为文本原型匹配，$P(\cdot)$ 为视觉原型与负缓存的对比信息，$f_r$ 为基于注意力加权的自适应缓存特征。

5. **MCP++ 残差学习**：
   引入可学习残差参数精调视觉和文本原型：
   $$\bar{t}_c' = \bar{t}_c + R_t^c, \quad \bar{v}_c' = \bar{v}_c + R_v^c$$
   联合优化损失函数包含三项：
   - **熵最小化损失** $\mathcal{L}_{\text{entro}}$：鼓励增强视图之间的预测一致性
   - **视觉-文本对齐损失** $\mathcal{L}_{\text{align}}$：基于 InfoNCE 对齐视觉和文本原型
   - **正负对比损失** $\mathcal{L}_{\text{contrast}}$：增大原型中心与负缓存样本的距离

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{entro}} + \lambda \cdot \mathcal{L}_{\text{align}} + \gamma \cdot \mathcal{L}_{\text{contrast}}$$

其中 $\lambda = 0.5$, $\gamma = 0.2$。每个测试样本生成 31 个增强视图（共 32 张），使用 AdamW 优化器更新一步，学习率 0.0001。残差参数初始化为零。

## 实验关键数据

### 主实验（跨数据集泛化，ViT-B/16）

| 方法 | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN397 | UCF101 | 平均 |
|------|----------|---------|------|-----|---------|--------|------|------|--------|--------|------|
| CLIP (零样本) | 23.67 | 93.35 | 65.48 | 44.27 | 42.01 | 67.44 | 83.65 | 88.25 | 62.59 | 65.13 | 63.58 |
| TPT | 24.78 | 94.16 | 66.87 | 47.75 | 42.44 | 68.98 | 84.67 | 87.79 | 65.50 | 68.04 | 65.10 |
| TDA | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 67.53 |
| DMN | 30.03 | 95.38 | 67.96 | 55.85 | 59.43 | 74.49 | 85.08 | 92.04 | 70.18 | 72.51 | 70.30 |
| DPE | 28.95 | 94.81 | 67.31 | 54.20 | 55.79 | 75.07 | 86.17 | 91.14 | 70.07 | 70.44 | 69.40 |
| **MCP** | **30.18** | **95.33** | **69.99** | **55.91** | **68.42** | 75.88 | 86.85 | 91.88 | 71.04 | 74.36 | 71.98 |
| **MCP++** | **31.06** | **95.50** | **70.13** | **56.97** | **68.69** | **77.55** | **87.20** | **92.40** | **71.17** | **75.44** | **72.61** |

### 消融实验（ImageNet 上各缓存组件的作用）

| Entropy | Align | Negative | ImageNet Acc. |
|---------|-------|----------|---------------|
| ✓ | ✓ | - | 72.49 |
| ✓ | - | ✓ | 71.73 |
| - | ✓ | ✓ | 72.54 |
| ✓ | - | - | 71.51 |
| - | ✓ | - | 72.47 |
| - | - | ✓ | 71.71 |
| ✓ | ✓ | ✓ | **72.64** |

### 关键发现

- MCP++ 在 10 个跨域数据集中 9 个取得最佳，平均准确率 72.61%，比 TDA 高 5.08%
- 在 OOD 基准上（ImageNet 及4个变体），MCP++ 平均 66.86%，比 DPE 高 0.93%
- 三种缓存单独使用均显著超过 CLIP 基线，组合使用最优，证明互补性
- $w = 0.8$（视觉权重）为最优模态融合比例，视觉信息比文本重要
- 缓存大小需要平衡：过小限制高置信样本数量，过大引入不确定样本

## 亮点与洞察

- **关键发现驱动设计**：从"缓存性能与类内紧致度正相关"这一实证发现出发，所有设计都围绕提升紧致度展开
- **三缓存互补架构**：entropy cache 提供锚点稳定性，align cache 增强紧致度，negative cache 校准决策边界——各司其职
- **最小化侵入式设计**：MCP 完全无训练，MCP++ 仅优化残差参数（一步梯度更新），极其轻量
- **多源信息融合推理**：文本语义匹配 + 视觉原型对比 + 缓存特征检索的三路 logits 组合

## 局限与展望

- 预测权重 $\alpha_1, \alpha_2, \alpha_3$ 需要针对每个下游任务搜索最优组合，不够自动化
- 缓存大小固定（每类10个），可能不适合类别数极多的场景（如 ImageNet 1000 类需维护 30000 个缓存条目）
- 仅在分类任务上验证，未扩展到检测、分割等其他视觉任务
- MCP++ 的残差学习需要生成增强视图并进行梯度更新，相比纯缓存方法增加了计算成本

## 相关工作与启发

- 继承 TDA 的低熵缓存思想，但通过 align cache 和负缓存从两个方向强化
- 与 DPE 类似引入了多模态原型残差调优，但 MCP++ 的三缓存设计更系统化
- 验证了"测试数据的类内紧致度"是缓存方法成败的关键因素，为后续工作指明方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 三缓存架构和紧致度分析有创新，但整体是已有思路的系统性组合
- **实验充分度**: ⭐⭐⭐⭐⭐ 15个数据集，两种骨干网络，详细消融（缓存组件、检索策略、缓存大小、模态权重）
- **写作质量**: ⭐⭐⭐⭐ 条理清晰，公式规范，但部分符号定义可更简洁
- **价值**: ⭐⭐⭐⭐ 对 TTA 领域的缓存方法有明确推进，但应用场景限于分类

<!-- RELATED:START -->

## 相关论文

- [Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](../../NeurIPS2025/multimodal_vlm/test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [Realistic Test-Time Adaptation of Vision-Language Models](../../CVPR2025/multimodal_vlm/realistic_test-time_adaptation_of_vision-language_models.md)
- [TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](../../NeurIPS2025/multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

<!-- RELATED:END -->
