---
title: >-
  [论文解读] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning
description: >-
  [AAAI 2026][半监督学习] 提出 DiCaP（Distribution-Calibrated Pseudo-labeling），通过估计伪标签的后验正确率来校准权重、引入双阈值机制分离置信区间和模糊区间并采用不同策略，在半监督多标签学习中以最高 4.27% 的幅度超越 SOTA。
tags:
  - AAAI 2026
  - 半监督学习
  - 多标签学习
  - 伪标签
  - 校准权重
  - 对比学习
---

# DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning

**会议**: AAAI 2026  
**arXiv**: [2511.20225](https://arxiv.org/abs/2511.20225)  
**代码**: [github.com/hb-studying/DiCaP](https://github.com/hb-studying/DiCaP)  
**领域**: 其他（半监督学习 / 多标签学习）  
**关键词**: 半监督学习, 多标签学习, 伪标签, 校准权重, 对比学习

## 一句话总结

提出 DiCaP（Distribution-Calibrated Pseudo-labeling），通过估计伪标签的后验正确率来校准权重、引入双阈值机制分离置信区间和模糊区间并采用不同策略，在半监督多标签学习中以最高 4.27% 的幅度超越 SOTA。

## 研究背景与动机

### 半监督多标签学习的挑战

多标签学习（MLL）要求模型为每个样本预测多个相关标签，广泛应用于图像标注、文本分类、表情识别等。然而获取完整多标签标注成本极高，因此**半监督多标签学习（SSMLL）**——利用少量标注数据和大量无标签数据——成为研究热点。

### 现有方法的核心问题

当前主流 SSMLL 方法基于伪标签策略，但存在关键局限：

**均匀加权**：CAP、D2L 等方法对所有伪标签分配相同权重，无论其置信度高低。低质量伪标签被等同对待，放大了噪声的负面影响

**置信度校准差**：深度网络往往产生**过度自信**的预测，预测概率与实际正确率之间存在显著偏差

**标注数据与无标注数据分布差异**：有标签数据和无标签数据的正确率分布不同（因训练信号不同），不能直接用有标签数据估计无标签数据的正确率

### 核心观察

**关键发现**：在同一数据集上，即使有标签训练样本数量变化，无标签数据的伪标签正确率分布保持**稳定一致**。

这意味着可以从有标签数据中分出一小部分当作"估计集"，将其按无标签数据处理来估计正确率分布，然后将该分布应用于全部无标签数据。

## 方法详解

### 整体框架

DiCaP 包含五个阶段：

```
标注数据 D_l → 分割为 D_sup (80%) + D_est (20%)
                    ↓                    ↓
              监督训练           作为无标签数据合并到 D_unsup = D_u ∪ D_est
                    ↓                    ↓
              产生预测    →    利用 D_est 的真实标签估计正确率分布
                    ↓
              双阈值分区: 置信样本 → 加权伪标签 | 模糊样本 → 对比学习
                    ↓
              联合训练: L_sup + L_pseudo + L_uncer
                    ↓
              微调阶段: 冻结 backbone，在 D_est 上微调分类头
```

### 关键设计

#### 1. **分布校准权重估计（DCW）**：理论最优的伪标签加权

**理论推导**：通过最小化伪标签权重与正确性指示函数之间的 BCE 损失，推导出最优权重等于后验正确率：

$$w^*(p_i) = P(\hat{y} = y \mid p_i)$$

即给定置信度 $p_i$ 时，伪标签正确的概率。

**实际估计**：将置信度区间 $[0,1]$ 均匀划分为 $K=20$ 个 bin，对每个 bin $\mathcal{B}_k$ 计算：

$$r_k^{pos} = \frac{n_k^{pos}}{n_k^{pos} + n_k^{neg} + \epsilon}$$

其中 $n_k^{pos}$ 和 $n_k^{neg}$ 分别是该 bin 中真正例和真反例的数量（在估计集 $\mathcal{D}_{est}$ 上计算，因为有真实标签）。

**线性插值平滑**：对任意置信度 $p$，使用相邻 bin 的线性插值得到权重：

$$w(p, \hat{y}=1) = \left(\frac{k+1}{K} - p\right) r_k^{pos} + \left(p - \frac{k}{K}\right) r_{k+1}^{pos}$$

设计动机：比直接用预测概率作权重更可靠（实验表明直接用 confidence 甚至比均匀权重更差）；比用标注数据估计更准确（分布不同）。

#### 2. **双阈值伪标签策略（DTH）**：分离置信与模糊区域

对每个类别 $c$，基于有标签数据的正/负样本预测分数的中间值计算动态阈值：

$$\tau_c^{pos} = \frac{\max(\mathcal{P}_c^{pos}) + \min(\mathcal{P}_c^{pos})}{2}, \quad \tau_c^{neg} = \frac{\max(\mathcal{P}_c^{neg}) + \min(\mathcal{P}_c^{neg})}{2}$$

对无标签样本的预测分数进行三路划分：

$$\hat{y}_{uc} = \begin{cases} 1 & \text{if } p_{uc} > \tau_c^{pos} \quad \text{（置信正例）} \\ 0 & \text{if } p_{uc} < \tau_c^{neg} \quad \text{（置信反例）} \\ -1 & \text{if } \tau_c^{neg} \leq p_{uc} \leq \tau_c^{pos} \quad \text{（模糊样本）} \end{cases}$$

- **置信样本**：赋予伪标签并应用校准权重，使用加权 ASL 损失
- **模糊样本**：不进行硬标签，转用对比学习正则化

#### 3. **不确定样本的鲁棒表征学习（URRL）**

对模糊样本采用**类别级对比学习**，扩展标准 InfoNCE 到多标签设定：

- 每个样本 $x_i$ 生成弱增强视图 $x_i^w$ 和强增强视图 $x_i^s$
- 提取类别级特征嵌入 $\{z_{ic}^w\}, \{z_{ic}^s\}$
- 同一样本同一类别的两个视图构成正对，其余为负对

$$\mathcal{L}_{uncer} = -\frac{1}{2B}\sum_{i=1}^{2B}\log\frac{\exp(z_i \cdot z_i^+ / \tau)}{\sum_{j=1}^{2B}\mathbb{I}_{i\neq j}\exp(z_i \cdot z_j / \tau)}$$

设计动机：模糊样本不适合硬标签监督，但其特征表示仍有价值。对比学习在不引入噪声梯度的前提下利用这些样本改善表征。

### 损失函数 / 训练策略

**总体损失**：

$$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{pseudo} + \mathcal{L}_{uncer}$$

- $\mathcal{L}_{sup}$：在 $\mathcal{D}_{sup}$ 上的有监督 ASL 损失
- $\mathcal{L}_{pseudo}$：在置信样本上的加权伪标签 ASL 损失
- $\mathcal{L}_{uncer}$：在模糊样本上的类别级对比损失

**微调阶段**：冻结 backbone，仅微调分类头，在 $\mathcal{D}_{est}$ 上使用其真实标签训练 20 epochs：

$$\mathcal{L}_{ft} = \frac{1}{|\mathcal{D}_{est}|}\sum_{(x,y)\in\mathcal{D}_{est}} \ell(f_\theta(x), y)$$

**训练细节**：ResNet-50 backbone + ML-Decoder，AdamW 优化器，OneCycleLR，EMA（衰减率 0.9997），RandAugment + Cutout 数据增强。

## 实验关键数据

### 主实验

在 4 个数据集 × 4 种标注比例下全面对比 11 种方法（mAP%）：

| 方法 | VOC 5% | VOC 10% | COCO 5% | COCO 10% | NUS 5% | AWA 5% |
|------|--------|---------|---------|----------|--------|--------|
| BCE | 65.40 | 75.48 | 57.09 | 62.34 | 40.12 | 61.33 |
| ASL | 71.41 | 77.81 | 57.87 | 62.95 | 42.04 | 60.40 |
| CAP | 75.90 | 81.83 | 62.88 | 67.18 | 44.98 | 63.90 |
| PCLP | 77.25 | 82.21 | 64.43 | 69.02 | 46.39 | 64.30 |
| BBAM | 78.66 | 83.45 | 63.54 | 67.41 | 33.15 | 64.19 |
| D2L | 79.26 | 84.06 | 69.30 | 73.06 | 46.86 | 64.66 |
| **DiCaP** | **83.53** | **87.92** | **70.07** | **73.55** | **48.37** | **66.32** |
| Δ vs D2L | **+4.27** | **+3.86** | **+0.77** | **+0.49** | **+1.51** | **+1.66** |

### 消融实验

各组件逐步叠加在 COCO 和 NUS 上的效果（mAP 平均）：

| 配置 | 平均 mAP(%) | 增益 |
|------|-----------|------|
| Baseline（仅有标签数据） | 55.82 | — |
| + DCW（校准权重） | 58.73 | **+2.91** |
| + DTH（双阈值） | 59.91 | +1.18 |
| + URRL（对比学习） | 60.25 | +0.34 |
| + WCL（预热对比） | 60.40 | +0.15 |
| + FTE（微调估计集） | **60.81** | +0.41 |

**权重策略对比**（COCO，各标注比例平均）：

| 加权策略 | 平均 mAP(%) | 说明 |
|---------|-----------|------|
| Uniform | 72.04 | 均等权重 |
| Confidence | 71.84 | 直接用预测概率（比均匀**更差**） |
| Labeled | 72.42 | 基于有标签数据估计 |
| **DiCaP** | **73.01** | 基于估计集分布校准 |
| Optimal | 73.09 | 使用真实标签计算（上界） |

### 关键发现

1. **DiCaP 接近理论最优**：与 Optimal（使用真实标签）仅差 0.08%
2. **直接用 confidence 作权重反而有害**：因为深度网络过度自信，置信度校准严重偏差
3. **DCW 贡献最大**（平均 +2.91%），验证了正确率校准的核心价值
4. **效率优势明显**：相比 D2L，GPU 内存减少 ~68%（4.44 vs 14.17GB on COCO），训练速度提升 ~15%
5. **估计分布高度准确**：在 VOC 5%（仅 57 个估计样本）条件下仍能近似完美匹配真实分布

## 亮点与洞察

- **理论-实践闭环**：从理论推导最优权重 → 发现分布稳定性 → 设计实用估计策略，逻辑链条完整
- **"稳定性"这一经验发现极有价值**：无标签数据的正确率分布不随标注量变化而改变，这一观察看似简单但为整个方法提供了坚实基础
- **双阈值 + 对比学习**：优雅地处理了"模糊地带"问题，避免了非此即彼的硬边界
- **极少标签下优势更大**：5% 标注时提升 4.27%（VOC），证明方法在标签极度稀缺时最有价值

## 局限与展望

1. **估计集划分比例固定**：20% 作为估计集可能不是所有场景的最优选择
2. **Bin 数量 K=20 是固定的**：更精细或自适应的分箱策略可能进一步提升
3. **仅验证图像任务**：文本分类、视频标注等模态未覆盖
4. **对比学习增益边际递减**：URRL 仅贡献 +0.34%，可能需要更强的无监督信号
5. **依赖 ResNet-50 backbone**：在 ViT 等架构上的表现未验证

## 相关工作与启发

- **伪标签方法**：FixMatch、FlexMatch 等单标签半监督方法
- **多标签专用**：CAP（类别级阈值）、D2L（度量自适应阈值）、PCLP（因果先验）
- **校准方法**：Temperature Scaling、Mixup Calibration
- **启发**：分布稳定性观察可能推广到其他任务（如半监督目标检测中的伪框质量估计）

## 评分

- 新颖性: ⭐⭐⭐⭐（分布校准权重 + 双阈值的组合新颖，核心观察有价值）
- 实验充分度: ⭐⭐⭐⭐⭐（4数据集 × 4比例 × 11对比 + 完整消融 + 效率分析）
- 写作质量: ⭐⭐⭐⭐⭐（理论推导清晰，可视化直观）
- 价值: ⭐⭐⭐⭐（对半监督多标签社区有直接推动，但通用性待验证）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](../../NeurIPS2025/llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/llm_evaluation/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ECCV 2024\] Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](../../ECCV2024/llm_evaluation/image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)

<!-- RELATED:END -->
