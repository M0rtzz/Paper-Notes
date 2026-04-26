---
title: >-
  [论文解读] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification
description: >-
  [AAAI 2026][自动驾驶][对抗防御] 系统识别出行人ReID对抗防御的两大独特挑战（模型偏差和复合泛化需求），提出去偏双不变防御框架：数据平衡阶段用扩散模型重采样缓解偏差，双对抗自元防御阶段通过最远负样本扩展软化的度量对抗训练和对抗增强的自元学习实现对未见ID和未见攻击的双重泛化。
tags:
  - AAAI 2026
  - 自动驾驶
  - 对抗防御
  - 行人重识别
  - 元学习
  - 数据平衡
  - 度量学习
---

# Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification

**会议**: AAAI 2026  
**arXiv**: [2511.09933](https://arxiv.org/abs/2511.09933)  
**代码**: [有](https://github.com/zchuanqi/DDDefense-ReID)  
**领域**: 自动驾驶 / 行人重识别  
**关键词**: 对抗防御, 行人重识别, 元学习, 数据平衡, 度量学习

## 一句话总结

系统识别出行人ReID对抗防御的两大独特挑战（模型偏差和复合泛化需求），提出去偏双不变防御框架：数据平衡阶段用扩散模型重采样缓解偏差，双对抗自元防御阶段通过最远负样本扩展软化的度量对抗训练和对抗增强的自元学习实现对未见ID和未见攻击的双重泛化。

## 研究背景与动机

行人ReID是安防监控的核心能力，但深度学习ReID模型极易受对抗攻击影响。现有防御方法主要针对分类任务设计，迁移到ReID这一度量学习任务时面临两个独特挑战：

**挑战一：模型偏差（Model Bias）**
- **类间样本不平衡**：ReID数据集中各ID的样本数量差异巨大（取决于出现在摄像头下的频率），Market-1501和DukeMTMC的统计数据清楚展示了这一点
- **类内多样性不足**：样本通常从视频序列提取，导致同一ID的图像高度冗余，视觉多样性有限
- 偏差效应：对抗训练后模型的每ID精度方差从18.78增大至23.40，偏差加剧

**挑战二：复合泛化需求（Composite Generalization）**
- **鲁棒性分布在分类器上**：实验验证——对抗训练后仅微调分类器，clean accuracy几乎不变但robustness显著下降（ResNet50在PGD下从53.08降至50.22），说明部分鲁棒性知识被分配到了测试时不用的分类器$H$上
- **双维泛化**：ReID是开集任务（测试时出现训练中未见的ID），同时攻击类型也太多无法穷举，因此需要同时对未见ID和未见攻击类型泛化

## 方法详解

### 整体框架

分两阶段：**数据平衡阶段** → **双对抗自元防御阶段**

模型由特征编码器 $E$（参数 $\theta_E$）和分类器 $H$（参数 $\theta_H$）组成。训练时联合优化 $G = H(E(\cdot))$，测试时仅用 $E$ 提特征做检索。

### 关键设计

**（1）基于扩散模型的数据平衡**

解决类间不平衡——对样本数低于阈值 $\delta_1$ 的ID，用条件扩散模型合成伪样本补齐：

$$\mathcal{D} \leftarrow \mathcal{D} \cup \{x_i^{\text{pseudo},j} \mid i \in \mathcal{I}, n_i < \delta_1, j=1,\dots,\delta_1 - n_i\}$$

解决类内多样性不足——对某单一摄像头占比超过 $\delta_2$ 的ID，从其他摄像头视角合成补充样本：

$$\mathcal{D}_{i'} \leftarrow \mathcal{D}_{i'} \cup \{x_{i',c}^{\text{pseudo}} \mid c \in \mathcal{C} \setminus \{c_{i'}\}\}$$

采用EDM框架实现，在ID条件设定下训练扩散模型。

**（2）最远负样本扩展软化（FNES）的度量对抗训练**

现有度量PGD攻击的问题：固定迭代方向导致对抗样本间高度相似，多样性不足。FNES通过两步改进：

**线性缩放扰动**：对最终对抗扰动进行线性缩放以增加多样性

$$x^{\text{temp}} = x + \gamma \cdot (\hat{x} - x), \quad x^{\text{adv}} = \omega x + (1-\omega) x^{\text{temp}}$$

其中 $\gamma \geq 1$ 为缩放因子，$\omega \sim \mathcal{U}(a,b)$ 为混合权重。

**最远负类标签软化**：将部分标签概率从真实类重新分配到度量攻击目标的最远负类：

$$y^{\text{adv}} = \omega \phi(y, \lambda_1) + (1-\omega) \tau(\phi(y, \lambda_2), \upsilon)$$

其中 $\phi$ 为标签平滑函数，$\tau$ 将概率 $\upsilon$ 从真实类转移到最远负类。这使模型能学习关于最远负类的鲁棒性知识，并缓解硬标签过拟合。

**（3）对抗增强学习（Adversarially-enhanced Learning）**

引入特征判别器 $D$ 与编码器 $E$ 构成对抗学习框架，学习对抗不变特征（clean和adversarial样本共享的特征表示）：

$$\min_E \max_D \mathcal{L}(E,D) = \mathbb{E}_x[\log D(E(x))] + \mathbb{E}_{x^{\text{adv}}}[\log(1-D(E(x^{\text{adv}})))]$$

达到Nash均衡时，$D$ 无法区分特征来源 → 编码器提取了对抗不变特征。

**（4）自元学习（Self-Meta Learning）**

为学习在见过和未见ID间共享的泛化不变特征：
- 将每batch数据分为 $\mathcal{D}_{\text{meta-train}}$ 和 $\mathcal{D}_{\text{meta-test}}$
- 模型先在 $\mathcal{D}_{\text{meta-train}}$ 上一步梯度下降得到临时模型 $G_{\text{temp}}$：
  $$\theta_G^{\text{temp}} = \theta_G - \alpha \nabla_{\theta_G} \mathcal{L}_{\text{meta-train}}$$
- 然后在 $\mathcal{D}_{\text{meta-test}}$ 上评估 $G_{\text{temp}}$
- 最终损失 $\mathcal{L}_{\text{self-meta}} = \mathcal{L}_{\text{meta-train}} + \mathcal{L}_{\text{meta-test}}$，直接对 $\theta_G$ 做梯度下降

其中每阶段的损失包含clean和adversarial样本的 $\ell = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{tri}} + \mathcal{L}_E$。

### 损失函数 / 训练策略

总优化框架：
- **内层**（攻击）：用度量PGD最大化 $\mathcal{L}_{\text{metric}}$
- **外层**（防御）：最小化分类损失 + 三元组损失，同时对抗训练编码器-判别器 + 自元学习

## 实验关键数据

### 主实验

**Table 2: ResNet50 白盒鲁棒性（mAP/Rank-1）**

| 方法 | Market Clean | FNA 8/255 | SMA 8/255 | IFGSM 8/255 |
|------|-------------|-----------|-----------|-------------|
| Origin | 78.49/92.01 | 0.20/0.17 | 0.27/0.26 | 1.25/1.95 |
| Adv_train | 69.69/88.24 | 8.57/18.14 | 22.85/35.69 | 17.97/34.65 |
| DAS | 69.79/88.39 | 12.70/24.85 | 32.14/49.05 | 22.33/39.79 |
| **Ours** | **68.50/88.21** | **31.99/55.17** | **50.13/72.60** | **37.61/62.02** |

在Market上FNA攻击下mAP从12.70→31.99，SMA攻击下从32.14→50.13，大幅领先SOTA防御DAS。

**Table 5: 跨数据集泛化（Market→Duke, ResNet50）**

| 方法 | Clean | FNA 8/255 | SMA 8/255 | IFGSM 8/255 |
|------|-------|-----------|-----------|-------------|
| None | 15.08/27.65 | 0.15/0.13 | 0.35/0.36 | 0.29/0.36 |
| Metric AT | 16.51/29.35 | 4.47/10.89 | 10.77/22.52 | 5.78/13.14 |
| **Ours** | **19.07/34.69** | **6.17/12.88** | **13.02/24.60** | **7.92/15.35** |

### 消融实验

**Table 3: 模块逐步累加消融（ResNet50, Market）**

| 配置 | Clean | FNA 8/255 | SMA 8/255 | IFGSM 8/255 |
|------|-------|-----------|-----------|-------------|
| Metric AT（基线） | 67.20/88.00 | 28.38/52.26 | 45.38/68.74 | 33.97/58.52 |
| +Diffusion model | 66.96/86.91 | 29.85/53.36 | 45.82/68.41 | 35.32/59.68 |
| +Adversarial learning | 67.81/88.21 | 30.34/53.77 | 48.10/70.72 | 35.70/60.27 |
| +Self-meta learning | 68.24/88.03 | 29.48/52.88 | 46.91/69.30 | 35.09/59.86 |
| +FNES | 68.29/88.07 | 30.98/54.45 | 49.25/71.11 | 37.16/61.49 |
| **All modules** | **68.50/88.21** | **31.99/55.17** | **50.13/72.60** | **37.61/62.02** |

每个模块都有正贡献：扩散模型平衡数据→对抗学习提取不变特征→自元学习增强泛化→FNES提升对抗训练多样性。

### 关键发现

1. **验证了"鲁棒性分布在分类器上"的假设**（Table 4）：本方法使AT_PGD→Metric AT→Ours的鲁棒性逐步提升，有效缓解了测试时丢弃分类器导致的鲁棒性损失
2. **跨数据集泛化有效**：Market训练→Duke测试，相比Metric AT提升32-40%
3. **Grad-CAM可视化**显示本方法关注更合理的身体特征区域
4. **UMAP特征分布**展示本方法在对抗攻击下维持更好的类间可分性

## 亮点与洞察

1. **问题洞察深刻**：首次系统识别ReID对抗防御的两大独特挑战，特别是"分类器吞噬鲁棒性"的实验验证令人信服
2. **FNES设计巧妙**：线性缩放打破度量PGD的固定迭代方向 + 最远负类标签软化同时解决多样性和过拟合问题
3. **双不变特征学习**：对抗不变（clean↔adversarial）+ 泛化不变（seen↔unseen ID）相互补充

## 局限与展望

1. clean accuracy在防御后有所下降（78.49→68.50），鲁棒性-准确性权衡仍存在
2. 扩散模型生成质量影响最终效果，但论文未深入讨论生成样本的质量控制
3. 自元学习中数据划分策略（meta-train vs meta-test）对结果的敏感性未充分分析
4. 仅在ResNet18/50上验证，Vision Transformer架构的适用性未知

## 相关工作与启发

- **ReID攻击**：FNA（Bai 2020）、SMA（Bouniot 2020）为白盒度量攻击，黑盒攻击（Liu 2023, Zhang 2020）
- **ReID防御**：离线对抗训练（Bai 2020）、虚拟数据增强（Bian 2025）、动态攻击预算（Wei 2024）
- **通用对抗训练**：PGD-AT（Madry 2018）、TRADES（Zhang 2019）
- **启发**：FNES的思路可推广到其他度量学习任务的对抗训练

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ★★★★☆ | 问题洞察独到，FNES设计创新 |
| 技术深度 | ★★★★☆ | 扩散模型+对抗学习+元学习的多层设计 |
| 实验质量 | ★★★★★ | 白盒/黑盒/跨数据集/消融/可视化全面 |
| 写作质量 | ★★★★☆ | 问题分析清晰，实验证据充分 |
| 实用价值 | ★★★★☆ | 有代码开源，对安防ReID系统防御有参考价值 |

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)
- [\[CVPR 2026\] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](../../CVPR2026/autonomous_driving/fedbprompt_federated_domain_generalization_person.md)
- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](../../CVPR2025/autonomous_driving/modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)
- [\[NeurIPS 2025\] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification](../../NeurIPS2025/autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)

<!-- RELATED:END -->
