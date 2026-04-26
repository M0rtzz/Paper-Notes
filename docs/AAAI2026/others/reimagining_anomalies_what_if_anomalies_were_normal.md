---
title: >-
  [论文解读] Reimagining Anomalies: What if Anomalies Were Normal?
description: >-
  [AAAI2026][anomaly detection] 提出首个面向无监督图像异常检测的反事实解释框架，通过训练生成器将异常样本修改为被检测器视为正常的多个解纠缠反事实，从语义层面回答“如果异常是正常的，它应该是什么样子？”，提供远超传统热力图的深层解释能力。
tags:
  - AAAI2026
  - anomaly detection
  - counterfactual explanation
  - explainable AI
  - GAN
  - 扩散模型
---

# Reimagining Anomalies: What if Anomalies Were Normal?

**会议**: AAAI2026  
**arXiv**: [2402.14469](https://arxiv.org/abs/2402.14469)  
**代码**: [liznerski/counterfactual-xad](https://github.com/liznerski/counterfactual-xad)  
**领域**: others  
**关键词**: anomaly detection, counterfactual explanation, explainable AI, GAN, diffusion model

## 一句话总结

提出首个面向无监督图像异常检测的反事实解释框架，通过训练生成器将异常样本修改为被检测器视为正常的多个解纠缠反事实，从语义层面回答“如果异常是正常的，它应该是什么样子？”，提供远超传统热力图的深层解释能力。

## 研究背景与动机

深度学习异常检测 (AD) 在图像基准上取得了巨大成功（错误率低至约 1%），但其黑盒特性导致用户难以理解检测器为何判定某个样本为异常。这在安全关键领域和需要建立信任的场景中尤为令人担忧。

**现有解释方法的局限**：
- **特征归因方法**（热力图，如 FCDD、PaDiM、PatchCore）：仅定位图像中检测器关注的区域，无法解释更高层语义
- **热力图无法捕捉多维度异常**：例如在 Colored-MNIST 中，热力图能标出数字形状差异，但完全无法解释颜色异常
- **现有反事实解释 (CE) 方法**：主要针对表格数据和时间序列；图像上的 CE 方法（如 DISSECT）仅适用于有监督分类；医学影像中的扩散模型方法需要正常/异常标注，无法用于无监督 AD

**核心想法**：生成反事实样本——对异常进行最小修改使其被检测器视为正常，从而语义层面解释检测器决策。同时生成多个“解纠缠”的反事实，捕捉异常的不同维度（如形状异常 vs 颜色异常）。

## 方法详解

### 问题定义

给定异常检测器 $\phi: \mathbb{R}^D \rightarrow [0,1]$ 和被检测为异常的样本 $\bm{x}^*$（$\phi(\bm{x}^*) \gg 0$），反事实 $\bar{\bm{x}}^*$ 需满足：
1. **正常性**：$\phi(\bar{\bm{x}}^*) \approx 0$（被检测器视为正常）
2. **最小修改**：$\|\bar{\bm{x}}^* - \bm{x}^*\|_1 \leq \epsilon$

进一步引入概念维度 $k \in \{1, \dots, K\}$，要求不同概念的反事实之间有足够差异：$\|\bar{\bm{x}}^*_k - \bar{\bm{x}}^*_{k'}\|_1 \geq \epsilon'$，实现解纠缠。

### GAN-based 反事实生成

训练生成器 $G: \mathbb{R}^D \times [0,1] \times \{1,\dots,K\} \rightarrow \mathbb{R}^D$，输入为原始图像、目标异常分数 $\alpha$ 和概念索引 $k$。整体优化目标：

$$\min_{G,R} \max_{\mathcal{D}} \mathbb{E}_{\bm{x} \sim P_X} \mathbb{E}_{\alpha,k} \left[ \lambda_{gan}(L_\mathcal{D} + L_G) - \lambda_\phi L_\phi + \lambda_{rec}(L_{rec} + L_{cyc}) + \lambda_r L_{con} \right]$$

包含五组损失：
1. **GAN 损失** ($L_\mathcal{D}, L_G$)：鉴别器与生成器的对抗损失，使用 spectral normalization + hinge loss，确保生成样本逼真
2. **异常分数损失** ($L_\phi$)：连续二值交叉熵，驱动生成样本的异常分数趋近目标 $\alpha$（推理时 $\alpha=0$）
3. **重建损失** ($L_{rec}$)：当目标分数等于真实分数时，生成器应输出原样本，鼓励最小修改
4. **循环一致性损失** ($L_{cyc}$)：从生成样本出发、以原始分数为目标，应能还原原样本，进一步约束修改幅度
5. **概念损失** ($L_{con}$)：通过概念分类器 $R$ 分类不同概念的反事实，驱动语义解纠缠

### 扩散模型扩展

针对高分辨率图像，整合 DiffEdit（基于 Stable Diffusion），重定义生成器在潜空间中操作：

$$G(\bm{x}, \alpha, k) = A_\Omega(G'(\psi(A_\mathcal{E}(\bm{x}), t), \alpha, k))$$

其中 $A_\mathcal{E}$/$A_\Omega$ 是自编码器的编码/解码器，$\psi$ 是 DiffEdit 模型，$t$ 为正常类别的文本提示。$G'$ 在潜空间中训练，使用与 GAN 相同的损失体系。

### 理论保证

- **定理 4.2**：仅用 GAN 损失时，生成器收敛到训练数据分布 $p_X$；加入重建/循环损失后，在检测器 $\phi$ 近似平坦的条件下仍近似收敛
- **定理 4.3**：异常分数损失 $L_\phi$ 是导致生成分布偏离 $p_X$ 的关键因素，正是这种偏离使得生成器学会将异常映射到正常空间

## 实验关键数据

### 实验设置
- **数据集**：MNIST、Colored-MNIST、CIFAR-10、GTSDB（交通标志）、ImageNet-Neighbors (INN)、MVTec-AD
- **检测器**：DSVDD、BCE (OE)、HSC (OE)，共 80+ 种 AD 设置
- **概念数** $K=2$，GAN 用于低分辨率数据集，扩散模型用于 INN

### Table 1: 反事实的正常性评估（AuROC，正常测试集 vs 反事实，越接近 50% 越好）

| 数据集 | BCE OE | HSC OE | DSVDD |
|---|---|---|---|
| MNIST (single) | 72.0 ± 4.0 | 80.8 ± 5.3 | 75.2 ± 9.2 |
| CIFAR-10 (single) | 47.5 ± 10.0 | 49.9 ± 4.4 | 54.6 ± 3.4 |
| INN (single) | 69.1 ± 18.1 | 67.9 ± 13.2 | × |
| C-MNIST (multi) | **55.6 ± 1.5** | **55.8 ± 4.7** | 61.5 ± 4.3 |
| CIFAR-10 (multi) | **49.0 ± 8.5** | **44.4 ± 6.7** | 50.7 ± 3.3 |
| GTSDB (multi) | **50.2 ± 8.0** | **48.6 ± 14.4** | 53.1 ± 4.8 |

在 CIFAR-10 和 GTSDB 上 AuROC 非常接近 50%，说明反事实几乎完全被检测器视为正常。

### Table 2: 反事实的逼真度（FIDN，以异常样本 FID 为 100% 基准，50-100% 为合理范围）

| 数据集 | BCE OE | HSC OE | DSVDD |
|---|---|---|---|
| MNIST (single) | 43 ± 8.1 | 68 ± 14.6 | 100 ± 8.8 |
| CIFAR-10 (single) | 116 ± 20.8 | 300 ± 90.0 | 116 ± 12.0 |
| INN (single) | 85.0 ± 28.6 | 85.4 ± 24.6 | × |
| C-MNIST (multi) | **56 ± 12.4** | 95 ± 30.5 | 83 ± 8.7 |
| MNIST (multi) | 78 ± 26.0 | 96 ± 25.0 | 100 ± 10.7 |
| GTSDB (multi) | 110 ± 101 | 95 ± 73.5 | 131 ± 118 |

BCE 和 HSC 的反事实在多数数据集上与异常样本同样逼真，在 MNIST 和 C-MNIST 上甚至更逼真。

### 定性结果亮点

- **Colored-MNIST**（正常=青色数字+数字1）：BCE 的反事实要么将数字变为 1（不改颜色），要么将颜色变为青色（不改数字），完美解纠缠
- **GTSDB**（正常=限速标志）：三角形异常标志全部变为圆形，揭示检测器依赖形状特征
- **CIFAR-10**（正常=船）：反事实保留异常物体颜色但改变背景为水面，揭示检测器主要依赖背景特征
- **ImageNet-Neighbors**（正常=斑马）：马、野猪等异常动物被转换为斑马，保持姿态和背景

### 揭示分类器偏差

训练仅使用蓝色异常的有监督分类器 vs 使用 OE 的无监督检测器：
- 无监督 BCE+OE AuROC: **98%**
- 有监督 BCE（仅蓝色异常）AuROC: **75%**
- 反事实可视化清晰揭示：有监督分类器对未见过颜色的异常生成的反事实不合理，直观展示了偏差来源

## 亮点

- **语义层面的异常解释**：首次将反事实解释系统性应用于无监督图像异常检测，从“为什么是异常”上升到“异常需要怎样改变才能正常”的语义解释，远超热力图
- **多概念解纠缠**：单个异常生成多个反事实，每个捕捉不同维度的异常性（形状 vs 颜色、前景 vs 背景），提供全面且结构化的解释
- **理论+实践双重保证**：证明了 Nash 均衡下生成器收敛到训练分布的条件，以及异常分数损失是产生有效反事实的关键驱动力
- **揭示检测器偏差**：通过反事实可以直观发现有监督分类器在异常子集上的偏差，为模型审计提供新工具
- **灵活框架**：同时支持 GAN（低分辨率）和扩散模型（高分辨率），适用于多种深度 AD 方法

## 局限性

- **生成质量受限**：在高分辨率自然图像上（如 CIFAR-10、INN），反事实的视觉质量仍有改进空间，部分反事实存在伪影
- **依赖 AD 模型质量**：DSVDD 等检测能力较弱的 AD 方法，其反事实质量也较差，框架效果与底层检测器性能耦合
- **概念数固定**：$K$ 需要预先设定，不同数据集的最优概念数可能不同，缺乏自动确定机制
- **仅适用于语义异常**：在工业缺陷检测（如 MVTec-AD）等低级异常场景中，反事实虽正确但不提供额外洞察
- **训练开销**：需要额外训练 GAN/扩散模型的生成器和鉴别器，增加了计算成本
- **解纠缠局部最优**：HSC 在某些场景下出现反事实同时改变颜色和数字的情况，表明优化可能陷入局部最优

## 相关工作

- **特征归因解释**：FCDD（全卷积热力图）、PaDiM（patch 分布建模）、PatchCore（核心集记忆库）等方法仅定位空间异常区域，无法提供语义层面的解释
- **图像反事实解释**：DISSECT（多概念解纠缠反事实）为本文方法论基础，但仅适用于有监督分类器；医学影像上的扩散模型 CE 方法（Sanchez 2022、Wolleb 2022）需要异常标注
- **AD 反事实解释**：表格/时序数据上有 Angiulli 2023、Datta 2022 等工作；图像上仅有 AR-Pro (Ji 2024) 通过掩码修复局部缺陷，无法处理语义异常
- **本文定位**：首个面向无监督语义图像 AD 的多概念解纠缠反事实解释框架，无需异常标注，支持 GAN 和扩散模型双路径

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地将反事实解释应用于无监督图像异常检测，多概念解纠缠的框架设计有明确方法论贡献
- 实验充分度: ⭐⭐⭐⭐ — 80+ AD 设置、5 数据集、3 种检测器，定性定量分析全面，但缺少与其他 CE 方法的直接定量对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论分析严谨，定性展示直观；符号较多但必要
- 价值: ⭐⭐⭐⭐ — 为 AD 的可解释性开辟了新范式，从特征归因到语义反事实是重要的方向性进步，对模型审计和偏差发现有实际意义

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](../../ECCV2024/others/learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](../../ACL2025/others/if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](../../CVPR2026/others/novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)
- [\[CVPR 2026\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](../../CVPR2026/others/integration_of_deep_generative_anomaly_detection_a.md)
- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](../../CVPR2026/others/what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)

<!-- RELATED:END -->
