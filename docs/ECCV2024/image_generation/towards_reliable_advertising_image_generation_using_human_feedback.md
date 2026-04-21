---
title: >-
  [论文解读] Towards Reliable Advertising Image Generation Using Human Feedback
description: >-
  [ECCV 2024][图像生成][广告图像生成] 构建百万级人工标注广告图像数据集 RF1M，提出多模态 RFNet 自动检测生成图像的可用性，并设计 Consistent Condition 正则化驱动的 RFFT 微调方法，将广告图像可用率从 56.4% 提升至 85.5%。
tags:
  - ECCV 2024
  - 图像生成
  - 广告图像生成
  - 人类反馈
  - 扩散模型微调
  - 多模态检测
  - 电商应用
---

# Towards Reliable Advertising Image Generation Using Human Feedback

**会议**: ECCV 2024  
**arXiv**: [2408.00418](https://arxiv.org/abs/2408.00418)  
**代码**: [有](https://github.com/ZhenbangDu/Reliable_AD)  
**领域**: 图像生成  
**关键词**: 广告图像生成, 人类反馈, 扩散模型微调, 多模态检测, 电商应用

## 一句话总结

构建百万级人工标注广告图像数据集 RF1M，提出多模态 RFNet 自动检测生成图像的可用性，并设计 Consistent Condition 正则化驱动的 RFFT 微调方法，将广告图像可用率从 56.4% 提升至 85.5%。

## 研究背景与动机

电商领域中吸引人的广告图像对提升点击率至关重要。虽然扩散模型（配合 ControlNet）可以为产品自动生成协调背景，但生成过程中频繁产生不合格图像：

- **空间不匹配**：产品与背景空间关系不当（如产品悬浮）
- **尺寸不匹配**：产品大小与背景不协调（如按摩椅比柜子小）
- **不清晰**：产品因背景复杂或颜色相似无法突出
- **形状幻觉**：背景错误延伸产品形状（如添加底座、支架）

这些不合格图像会误导消费者，需要大量人工审查。核心挑战：**如何建立可靠的广告图像生成 pipeline，生产高可用率的图像？**

两层解决思路：

**重复生成（Recurrent Generation）**：利用随机性多次生成，用自动检测代替人工审查

**模型微调（RFFT）**：用人类反馈微调扩散模型，从根本上提高可用率

## 方法详解

### 整体框架

完整 pipeline 包含三个核心组件：

1. **RF1M 数据集**：105 万张人工标注广告图像，提供五类细粒度标签
2. **RFNet（Reliable Feedback Network）**：多模态检测网络，自动评估生成图像可用性
3. **RFFT（Reliable Feedback Fine-Tuning）**：用 RFNet 反馈微调 ControlNet，配合 Consistent Condition 正则化防止崩溃

生成流程：产品图 + 文本 prompt → Stable Diffusion + ControlNet inpainting → RFNet 检测 → 可用/重新生成。

### 关键设计

#### 1. RF1M 数据集构建

基于 JD.com 商品生成，包含 1,058,230 个样本，每个样本包含：
- 生成的广告图像 + 透明背景产品图像
- 专业设计师编写的 prompt
- 深度图（DPT 生成）和显著性图（U2-Net 生成）
- 产品标题/描述
- 人工标注的五类标签（Available / Space Mismatch / Size Mismatch / Indistinctiveness / Shape Hallucination）

经京东线上 A/B 测试验证：全量曝光超 6000 万次，CTR 提升 2.2%。

#### 2. RFNet 多模态检测网络

融合五种模态信息进行判断：

**输入模态**：
- $I_o$（产品原图）：了解产品外观
- $I_g$（生成广告图）：评估整体效果
- $I_d$（深度图）：判断产品相对背景的空间位置
- $I_s$（显著性图）：检测产品轮廓是否突出
- $Cap$（产品描述文本）：提供产品属性知识

**网络结构**：
- 图像编码器：预训练 ResNet50，将四种图像编码为 $\{e_o, e_g, e_d, e_s\}$
- 文本编码器：微调 RoBERTa，将产品描述编码为 $e_c$
- **Feature Filter Module (FFM)**：N₁ 个 cross-attention + 卷积模块，$e_o$ 作为 Query、$e_c$ 作为 Key/Value，提取视觉相关属性

$$e_f = \text{Conv}(\text{Conv}(\text{CrossAttn}(e_o, e_c)) \otimes \text{Conv}(e_o)) + e_o$$

- **Self-Attention 融合**：N₂ 层 self-attention 整合所有模态特征

$$f = \text{SelfAttention}(\text{Concat}(e_f, e_g, e_d, e_s))$$

- 最终全连接分类器输出五类概率

#### 3. RFFT 微调方法

核心挑战：直接用可用性反馈微调会导致**生成崩溃**——模型学会生成简单重复背景以规避不合格案例，可用率可达 99.8% 但美观度崩溃。

**反馈信号 $F_{AC}$**：

$$F_{AC} = -\frac{1}{N} \sum_{i=1}^{N} y_d \log(\hat{o}_i)$$

其中 $y_d$ 是 "Available" 类别的 one-hot 向量，梯度回传微调 ControlNet。

**Consistent Condition (CC) 正则化**：

核心 insight：不应限制图像本身不变（KL 正则），而应保持文本条件对生成的影响方向不变。先从 classifier-free guidance 中提取文本引导方向：

$$\nabla_{x_t} \log p_\theta^t(z|x_t, c) \approx -\frac{1}{\sqrt{1-\bar{\alpha}_t}} (\epsilon_\theta(x_t, z, c) - \epsilon_\theta(x_t, c))$$

然后约束微调模型与参考模型的条件方向一致：

$$L_{CC} = \|\nabla_{x_t} \log p_\theta^t(z|x_t, c) - \nabla_{x_t} \log p_{ref}^t(z|x_t, c)\|_2$$

**KL 正则 vs CC 正则**的本质区别：
- KL 正则与 $F_{AC}$ 是**对抗关系**——一个想改变输出，一个想保持不变
- CC 正则与 $F_{AC}$ 是**协作关系**——保持条件控制方向一致的同时提升可用率

最终损失：$F_{total} = F_{AC} + \beta L_{CC}$

### 损失函数 / 训练策略

**RFNet 训练**：
- ResNet50 (ImageNet 预训练) + RoBERTa（产品描述微调），图像 resize 到 384×384
- FFM: width 384, 8 heads, N₁=1; Self-Attention: N₂=3
- 训练 10 epochs，初始 lr=1e-4，epoch 5 降 10 倍

**RFFT 微调**：
- 8×A100，local batch size 4，梯度累积 4 步，AdamW lr=1e-5
- 基础模型：MajicmixRealistic_v7 + ControlNet v1.1
- 仅训练 ControlNet，冻结其他参数
- 40 步 DDIM，最后 10 步随机选一步生成 $\hat{x}_0^t$，经过 RFNet 评估后反向传播

## 实验关键数据

### 主实验

**RFNet 检测性能对比（1000 张测试图）：**

| 模型 | Precision | Recall | F1 | AP |
|------|-----------|--------|-----|-----|
| ResNet50 | 74.87 | 73.66 | 74.26 | 77.29 |
| ResNeXt50 | 77.73 | 76.88 | 77.30 | 79.62 |
| HRNet | 72.89 | 73.12 | 73.01 | 73.07 |
| ViT | 75.59 | 78.33 | 76.93 | 79.31 |
| **RFNet** | **86.45** | **85.23** | **85.83** | **87.58** |

RFNet 在所有指标上大幅领先，F1 超越次优方法 8.5 个百分点。

**广告图像可用率对比（1000 产品，单次生成）：**

| 方法 | Ava（RFNet）↑ | Human Ava↑ |
|------|-------------|------------|
| Ori（原始模型） | 56.4% | 70.1% |
| PromptEng | 62.9% | 73.2% |
| PPO | 65.9% | 74.9% |
| DPO | 57.3% | 71.8% |
| ReFL | 84.7% | 84.9% |
| **Ours (RFFT)** | **85.5%** | **86.3%** |

RFFT 可用率比原始模型提升 29.1 个百分点（56.4→85.5%），人工审核一致性验证了 RFNet 的可靠性。

### 消融实验

**RFNet 各模态贡献消融：**

| $I_o$ | $I_g$ | $I_d$ | $I_s$ | Cap | AP |
|-------|-------|-------|-------|-----|-----|
| ✗ | ✓ | ✓ | ✓ | ✓ | 81.17 |
| ✓ | ✗ | ✓ | ✓ | ✓ | 82.06 |
| ✓ | ✓ | ✗ | ✓ | ✓ | 85.31 |
| ✓ | ✓ | ✓ | ✗ | ✓ | 83.91 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 84.53 |
| **✓** | **✓** | **✓** | **✓** | **✓** | **87.58** |
| 粗粒度标签 | | | | | 82.06 |

产品原图 $I_o$ 贡献最大（去掉后 AP 降 6.41），细粒度标签比粗粒度标签好 5.52。

**CC 正则 vs KL 正则**：随 β 增大，KL 正则可用率显著下降（对抗效应），CC 正则保持高可用率。

### 关键发现

- RFNet 的 "Ava" 和 "Human Ava" 趋势一致，证明模型忠实反映人类反馈
- 重复生成策略可进一步提升可用率，但 RFFT 微调后仅需更少次数即可达标
- 微调后的 ControlNet 可泛化到不同 LoRA 和扩散模型权重（如 Maji_v6、SD_v1.5），无需重复训练
- 200 名专业人员偏好评估显示 RFFT 图像美观度与原始模型相当，远优于 ReFL
- 美学反馈（ImageReward）+ CC 正则可与可用性反馈组合使用，互不冲突

## 亮点与洞察

1. **工业级完整方案**：从数据集→检测网络→模型微调→线上部署，提供了完整的广告图像可靠生成 solution
2. **CC 正则化解决美学-可用性 trade-off**：巧妙地将约束从"输出不变"转为"条件方向不变"，从根本上避免对抗效应
3. **百万级标注数据集**：RF1M 是广告图像生成领域首个大规模多模态带人工标注的数据集
4. **多模态融合检测**：深度图、显著性图、产品描述等辅助信息显著提升检测精度

## 局限与展望

- 数据集来源单一（JD.com），可能存在领域偏差
- RFNet 的五类分类粒度可进一步细化
- RFFT 仅微调 ControlNet，未探索对 U-Net 的微调
- CC 正则的超参数 β 需要手动调节
- 当前 pipeline 推理成本较高（多次生成 + 多模态检测），可优化效率

## 相关工作与启发

- **ReFL / DRaFT**：直接用可微奖励的梯度端到端微调扩散模型，RFFT 借鉴了类似的端到端策略
- **DDPO / DPOK**：将去噪过程建模为多步 MDP 用 Policy Gradient 更新，但训练成本更高
- **Diffusion-DPO**：用人类比较数据增强扩散模型，但无法针对特定可用性问题进行优化
- **ControlNet**：RFFT 仅微调 ControlNet 部分参数，泛化性好且训练效率高

## 评分

- **新颖性**: ⭐⭐⭐⭐ — CC 正则化解决 RLHF 崩溃问题的思路独到
- **有效性**: ⭐⭐⭐⭐⭐ — 可用率提升 29.1%，线上 CTR 提升 2.2%，效果经工业验证
- **工程价值**: ⭐⭐⭐⭐⭐ — 完整工业级方案，数据集公开，已在京东生产环境部署
- **推荐度**: ⭐⭐⭐⭐ — RLHF 应用于广告图像的开创性工作，CC 正则化可广泛借鉴

<!-- RELATED:START -->

## 相关论文

- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [Latent Guard: a Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)
- [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_text-to-image_generation_via_reinforcement_le.md)

<!-- RELATED:END -->
