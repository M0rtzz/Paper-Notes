---
title: >-
  [论文解读] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection
description: >-
  [AAAI 2026][图像生成][核心集选择] 提出利用扩散模型的部分反向去噪重建偏差作为数据似然的理论近似信号，配合信息瓶颈理论选择最优重建时间步，实现分布感知的核心集选择，在 ImageNet 上仅用 50% 数据即可逼近全量训练性能。 核心集选择（Core-set Selection）旨在从大规模数据集中选取代表性子…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "核心集选择"
  - "扩散模型"
  - "数据似然"
  - "重建偏差"
  - "信息瓶颈"
---

# Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection

**会议**: AAAI 2026  
**arXiv**: [2511.19274](https://arxiv.org/abs/2511.19274)  
**代码**: [GitHub](https://github.com/mchen725/DRD)  
**领域**: 图像生成  
**关键词**: 核心集选择, 扩散模型, 数据似然, 重建偏差, 信息瓶颈

## 一句话总结

提出利用扩散模型的部分反向去噪重建偏差作为数据似然的理论近似信号，配合信息瓶颈理论选择最优重建时间步，实现分布感知的核心集选择，在 ImageNet 上仅用 50% 数据即可逼近全量训练性能。

## 研究背景与动机

核心集选择（Core-set Selection）旨在从大规模数据集中选取代表性子集来高效训练模型。现有基于分数的方法（如 Forgetting Score、EL2N、AUM）依赖启发式代理信号——遗忘事件、模型不确定性、损失值等——但这些方法有根本性缺陷：

**缺乏对数据似然的显式建模**：启发式分数无法捕获数据分布中微妙但关键的结构特征

**区分能力不足**：如 t-SNE 可视化所示，Forgetting Score 和 EL2N 的分层效果类似随机采样，不同分数区间的样本在特征空间高度混杂

**缺乏理论基础**：分数选择更像经验性调参而非原则性设计

扩散模型是显式学习数据分布的生成模型。核心洞察：**重建误差与数据似然成反比**——高密度区域的样本重建误差小，低密度/离群样本误差大。因此，扩散重建偏差可作为天然的、理论有据的数据筛选信号。

## 方法详解

### 整体框架

方法包含两个核心步骤：

1. **扩散重建偏差评分（DRD Score）**：对每个样本 $x_0$ 执行部分加噪→反向去噪→计算重建偏差
2. **信息瓶颈时间步选择**：基于信息论方法选择最优重建时间步 $t^*$，使偏差最具区分力

然后使用 Best Window Selection（BWS）策略从排序后的分数列表中滑动窗口选取最优子集。

### 关键设计

#### 1. 重建偏差与数据似然的理论联系

**定理1**（重建偏差与对数似然的反向依赖）：给定 $x_0$，加噪到 $x_t$，用 DDPM 去噪得到 $x_{0:t}'$，期望重建偏差满足下界：

$$\mathbb{E}_\epsilon[\|\Delta x_0(t)\|^2] \geq -\kappa(t) \log q(x_0) + \mathcal{C}_{\text{noise}}(t)$$

其中 $\kappa(t) = \frac{1}{t}\sum_{s=1}^{t}\frac{1}{\sigma_s^2}$。直觉上：高密度样本重建偏差小，低密度样本偏差大。

实际使用 DDIM 进行确定性重建，以 LPIPS（感知相似度）度量偏差，比像素级 $L_2$ 更符合人类判断。

#### 2. 信息瓶颈最优时间步选择

时间步选择是关键——过小的 $t$ 导致 $x_0' \approx x_0$（偏差无信息），过大的 $t$ 导致 $x_t$ 趋向纯噪声（重建与 $x_0$ 无关）。

将问题纳入信息瓶颈（IB）框架：

$$\min_t \mathbb{E}_{x_0}[\mathcal{I}(x_0; x_t) - \beta \mathcal{I}(x_t; c)]$$

其中 $\mathcal{I}(x_0; x_t)$ 用 SNR(t) 近似，$\mathcal{I}(x_t; c)$ 为加噪数据与类标签的互信息。

最优时间步定义为互信息下降率最大的点：

$$t^* = \arg\max_t \left|\frac{\partial \mathcal{I}(x_t; c)}{\partial t}\right|, \quad \text{s.t.} \; \text{SNR}(t) \in [0.05, 1]$$

**引理1** 证明该导数等价于 $|\mathbb{E}_{x_0,c}[\frac{\partial}{\partial t}\log p(c|x_t)]|$，可通过扩散分类器近似计算。实际采用有限差分法：

$$\frac{\partial}{\partial t}\log p_\theta(c|x_t) \approx \frac{\log p_\theta(c|x_{t+1}) - \log p_\theta(c|x_{t-1})}{2}$$

对每个类别采样 $B=20$ 个样本进行蒙特卡洛估计，几分钟内即可完成时间步选择。

#### 3. 中等似然数据最优

实验揭示了一个重要发现：**中等似然度的数据对核心集构建最有效**。将数据按 DRD 分数排序并分为 5 等分，中间区间训练的模型泛化性最好。这与数据增强（如 CutMix）的效果类似——中等似然样本常包含模糊或部分语义线索，促使模型学习更鲁棒的特征。

### 损失函数 / 训练策略

- 使用预训练的 DiT（Diffusion Transformer）模型，完整调度 $T=1000$ 步
- 推理使用 DDIM 采样器，$T=50$ 步，从选定的 $t < 50$ 执行部分重建
- 扩散分类器中采样 20 组随机噪声近似 $p_\theta(c|x_t)$
- 默认使用 BWS 策略选择最优窗口，扫描步长 5%
- 所有结果可在单张 RTX 4090 上复现

## 实验关键数据

### 主实验

**ImageNet-1K 核心集选择结果（ResNet-50，测试准确率%）**

| 方法 | 10% | 30% | 50% | 75% |
|------|-----|-----|-----|-----|
| Random | ~60 | ~70 | ~74 | ~76 |
| Forgetting | ~62 | ~71 | ~74 | ~76 |
| EL2N | ~60 | ~70 | ~74 | ~76 |
| BWS (Forgetting) | ~64 | ~73 | ~75 | ~77 |
| **DRD (Ours)** | **~66** | **~75** | **~76** | **~77** |

DRD 在所有选择比率上一致超越全部 baseline，50% 数据即逼近全量训练性能。

**表1：ImageWoof 上分数导向选择策略对比**

| 策略 | 分数 | 10% | 30% | 75% |
|------|------|-----|-----|-----|
| CCS | Forgetting | 68.4 | 80.9 | 86.4 |
| CCS | DRD | 68.9 | 80.1 | 86.9 |
| BWS | Forgetting | 70.1 | 81.5 | 87.3 |
| **BWS** | **DRD** | **74.5** | **83.3** | **88.7** |

BWS+DRD 在所有比率上取得最高准确率，低比率（10%）优势尤为明显（+4.4 vs BWS+Forgetting）。

### 消融实验

**表2：时间步选择超参数敏感性（ImageWoof）**

| B | #ε | 10% | 30% | 75% |
|---|-----|-----|-----|-----|
| 20 | 5 | 67.8 | 81.1 | 86.3 |
| 20 | 20 | **74.5** | **83.3** | 88.5 |
| 40 | 20 | 74.7 | 83.3 | 88.5 |

噪声样本数 #ε 过少（5）导致显著性能下降，≥20 后趋于稳定。

### 关键发现

1. **DRD 比 Forgetting 和 EL2N 更具区分力**：t-SNE 可视化显示 DRD 的分层具有语义连贯性，而其他方法与随机采样无异
2. **中等似然数据最佳**：BWS 最优窗口起始点通常在 20%-40% 区间，极端似然度（过高或过低）的数据对训练贡献较小
3. **IB 时间步选择 vs 网格搜索**：IB 方法在几分钟内完成，而网格搜索需要对每个候选时间步执行完整训练——计算成本差距巨大
4. **跨架构泛化**：DRD 在 ResNet-18、ResNet-50、EfficientNet-B0、ViT 上均表现最优
5. 选定时间步通常聚集在 $t \approx 20$，信噪比平衡点恰好在此

## 亮点与洞察

1. **理论驱动**：不是又一个启发式分数，而是从扩散模型 ELBO 严格推导出重建偏差与数据似然的联系（定理1），并用信息瓶颈理论（引理1）指导时间步选择
2. **分布感知评分**：DRD 分数具有实际的物理意义——直接反映样本在数据流形上的位置，高分 = 低似然 = 偏离主分布
3. **中等似然偏好的解释**：与课程学习和数据增强形成统一解释框架——最有价值的训练样本既非过易也非过难
4. **计算友好**：单 GPU 可复现全部结果，时间步选择仅需几分钟

## 局限与展望

1. **依赖预训练扩散模型**：需要 DiT 在目标数据集上的预训练模型，对新领域数据可能不直接适用
2. **类别数量限制**：扩散分类器在类别极多（如 ImageNet 1000 类）时计算分类概率的成本较高
3. **仅在图像分类上验证**：未验证在检测、分割、NLP 等其他任务上的通用性
4. **BWS 需要验证集**：最优窗口搜索需要在全量验证集上评估，并非完全无监督
5. 可探索将 DRD 与主动学习、持续学习结合

## 相关工作与启发

- **BWS (Choi et al., 2024)**：最优窗口搜索策略的提出者，默认用 Forgetting Score 作为排序依据
- **EL2N / Forgetting Score**：经典的启发式筛选方法，本文证明其区分能力不足
- **扩散分类器 (Li et al., 2023)**：利用 ELBO 近似类别后验概率，本文借用其计算 $p_\theta(c|x_t)$
- **信息瓶颈理论 (Tishby et al., 2000)**：本文将其巧妙应用于时间步选择问题
- 启发：扩散模型的"重建能力"不仅可用于生成，还可以作为数据质量度量工具——这开辟了扩散模型的新应用范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（理论驱动的全新评分准则）
- 技术深度: ⭐⭐⭐⭐⭐（定理证明+信息瓶颈+系统分析）
- 实验完整性: ⭐⭐⭐⭐⭐（3数据集+4架构+策略对比+超参分析+可视化）
- 实用价值: ⭐⭐⭐⭐（单 GPU 即可运行，提升显著）
- 总体评分: 9.0/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)

</div>

<!-- RELATED:END -->
