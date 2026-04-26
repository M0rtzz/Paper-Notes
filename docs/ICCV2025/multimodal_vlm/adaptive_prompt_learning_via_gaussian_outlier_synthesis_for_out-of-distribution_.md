---
title: >-
  [论文解读] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection
description: >-
  [ICCV 2025][多模态][OOD检测] 提出 APLGOS 框架，利用 ChatGPT 标准化 Q&A 对来初始化可学习 ID 提示，并在类条件高斯分布的低似然区域合成虚拟 OOD 提示和图像，通过对比学习对齐文本-图像嵌入，实现更紧凑的 ID/OOD 决策边界。
tags:
  - ICCV 2025
  - 多模态
  - OOD检测
  - 提示学习
  - 高斯离群合成
  - 视觉-语言模型
  - 对比学习
---

# Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: 无  
**领域**: multimodal_vlm / OOD Detection  
**关键词**: OOD检测, 提示学习, 高斯离群合成, 视觉-语言模型, 对比学习

## 一句话总结

提出 APLGOS 框架，利用 ChatGPT 标准化 Q&A 对来初始化可学习 ID 提示，并在类条件高斯分布的低似然区域合成虚拟 OOD 提示和图像，通过对比学习对齐文本-图像嵌入，实现更紧凑的 ID/OOD 决策边界。

## 研究背景与动机

### 核心问题
深度学习模型在训练时只见过有限的已知类别（ID），在实际部署中面对大量未知类别（OOD）时，容易以高置信度做出错误预测，这在自动驾驶等安全关键领域会带来严重风险。OOD 检测的目标是让模型不仅能准确识别已知类别，还能将未知类别标记为"不确定"。

### 现有方法的局限

**基于提取的方法**：从 ID 数据中直接提取 OOD 伪样本来正则化模型，但提取质量不可控，且需要大量 ID 数据

**基于合成的方法**：直接合成 OOD RGB 图像或在低维隐空间合成虚拟离群点，部分缓解了上述问题，但合成质量仍然堪忧

**关键 Gap**：尚无工作将**提示学习**引入 OOD 检测任务，VLM 强大的预训练知识和表示能力未被充分利用

### 动机

**为什么要用提示学习？** 视觉-语言模型（VLM）具有强大的预训练知识和跨模态对齐能力，通过设计合适的文本提示（prompt），可以更有效地在隐空间中建立 ID 和 OOD 的区分边界。**为什么要在高斯分布低似然区域采样？** 真实 OOD 数据分布未知，但其大概率出现在 ID 数据分布的低密度区域，因此在类条件高斯分布的低似然区域合成虚拟 OOD 提示，能更好逼近 OOD 的真实分布。

## 方法详解

### 整体框架

APLGOS 由两个核心模块组成：
- **PLM（Prompt Learning Module）**：负责生成 ID 提示和合成 OOD 虚拟提示
- **TAM（Text-Image Alignment Module）**：计算文本-图像相似度并通过对比学习对齐多模态数据

整个训练过程分为三个阶段：
1. 第一阶段：使用 ChatGPT 标准化 Q&A 对，生成语句集
2. 第二阶段：用 ID 提示和 ID 图像进行对齐训练
3. 第三阶段：引入合成的 OOD 提示和 OOD 图像进行联合训练

**核心亮点**：只有 ID 图像来自真实数据集，ID 提示、OOD 提示和 OOD 图像全部是虚拟合成的。

### 关键设计

#### 1. ID 提示生成

传统方法使用单一不变的提示（如 "a photo of a \<CLS\>"），表达能力有限。APLGOS 设计了更丰富的提示策略：

**预定义 Q&A 对**：`Q: What is in the region with coordinates <loc1>,<loc2>,<loc3>,<loc4>? A: That's a <CLS>.`

**ChatGPT 多轮标准化**：
$$\Omega_0 = g(QA + M + G_0), \quad \Omega_i = g(\Omega_{i-1} + G_i)$$

其中 $g$ 是 ChatGPT 的标准化操作，$M$ 是预定义模板，$G_i$ 是第 $i$ 轮的引导指令。经过 $t$ 轮标准化后得到语句集 $\Omega_t$。

**可学习提示结构**：`<loc1><loc2><loc3><loc4><V1><V2>...<Vm><CLS>`
- `<loc>` 为可学习位置 token，隐式引入位置信息
- `<V>` 为可学习描述 token
- `<CLS>` 替换为当前区域的类别标签

**为什么要这样设计？** 引入位置坐标让模型进行更细粒度的区域级观察，从语句集中采样初始化使提示具有多样性，避免单一提示导致的表达瓶颈。

#### 2. OOD 提示合成（核心创新）

**类条件高斯分布假设**：假设 ID 提示嵌入服从类条件多元高斯分布：
$$p_\theta(\hat{T}|y=i) = \mathcal{N}(\hat{\mu}_i, \hat{\sigma})$$

**经验高斯均值计算**：
$$\hat{\mu}_i = \frac{1}{|Q_T|} \sum_{j=1}^{|Q_T|} \hat{T}_{i,j}$$

**绑定协方差矩阵计算**（关键公式）：
$$\hat{\sigma} = \frac{1}{K|Q_T|} \sum_{i=1}^{K} \sum_{j=1}^{|Q_T|} (\hat{T}_{i,j} + \alpha\varepsilon - \hat{\mu}_i)(\hat{T}_{i,j} + \alpha\varepsilon - \hat{\mu}_i)^T + \beta E$$

其中 $\varepsilon$ 是由随机高斯噪声初始化的可学习矩阵，$\alpha$ 控制噪声强度，$\beta E$ 提供正则化。

**在低似然区域采样虚拟 OOD 提示**：
$$V_i = \Psi(\hat{T}, \hat{\mu}_i, \hat{\sigma})$$

其中 $\Psi$ 是类条件高斯分布概率密度函数，选择概率最低的 top-k 提示作为 OOD 伪提示 $\hat{T}^\dagger$。

**为什么选择低似然区域？** 低似然区域是 ID 类别分布的边缘区域，也是最可能出现 OOD 数据的区域。在这里合成虚拟提示能有效正则化决策边界，使模型学到更紧凑的分类边界。

#### 3. OOD 虚拟图像合成

原理与 OOD 提示合成完全类似，只是输入从 ID 提示嵌入变为 ID 图像嵌入。使用图像嵌入队列 $Q_I$ 计算经验高斯均值和协方差，在低似然区域采样得到虚拟 OOD 图像 $\hat{X}^\dagger$。

#### 4. 文本-图像对齐模块（TAM）

计算归一化提示嵌入和图像嵌入之间的相似度分数：
$$S = \frac{\|\hat{X}\|_p (\|\hat{T}\|_p)^T}{e^\omega}$$

其中 $\omega$ 为缩放超参数。在第二阶段使用 ID 数据，第三阶段使用合成 OOD 数据。

### 损失函数 / 训练策略

总损失由多个部分组成：

$$\mathcal{L} = \xi_1[\gamma_1 \tau \mathcal{L}_{align}^{id} + \gamma_2(1-\tau)\mathcal{L}_{align}^{ood}] + \gamma_3 \xi_2[\kappa \mathcal{L}_{loc}^{id} + (1-\kappa)\mathcal{L}_{loc}^{ood}] + \gamma_4 \xi_3 \mathcal{L}_{cls} + \gamma_5 \xi_4 \mathcal{L}_{reg} + W$$

- **对齐损失 $\mathcal{L}_{align}$**：基于相似度分数的对比学习损失，将所有 OOD 类别视为单一"background"类
- **位置损失 $\mathcal{L}_{loc}$**：隐式引入位置信息，使提示具备区域级细粒度
- **分类损失 $\mathcal{L}_{cls}$** 和 **回归损失 $\mathcal{L}_{reg}$**：标准检测损失
- **正则化项 $W$**：进一步正则化模型

训练策略通过 $\xi, \tau, \kappa$ 控制不同训练阶段使用的损失函数组合。

## 实验关键数据

### 主实验

| ID 数据集 | 方法 | FPR95↓ (COCO/OI) | AUROC↑ (COCO/OI) | mAP↑ |
|-----------|------|-------------------|-------------------|------|
| PASCAL VOC | VOS-RegX4.0 | 50.53 / 50.27 | 88.10 / 87.08 | 49.1 |
| PASCAL VOC | **APLGOS (RegX4.0)** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |
| BDD-100k | VOS-ResNet50 | 46.97 / 31.25 | 84.97 / 89.82 | 35.7 |
| BDD-100k | **APLGOS (ResNet50)** | **41.10 / 23.30** | **87.36 / 92.87** | **35.8** |
| BDD-100k | VOS-RegX4.0 | 42.82 / 27.55 | 86.36 / 92.11 | 37.0 |
| BDD-100k | **APLGOS (RegX4.0)** | **39.48 / 19.79** | **87.47 / 93.59** | **37.6** |

在 BDD-100k + OpenImages 设置下，APLGOS 将 FPR95 从 27.55% 降至 **19.79%**，降幅达 7.76%。

### 消融实验

**提示策略消融**（PASCAL VOC + COCO/OI）：

| 策略 | FPR95↓ | AUROC↑ | mAP↑ |
|------|--------|--------|------|
| (a) VOS-RegX4.0 基线 | 50.53 / 50.27 | 88.10 / 87.08 | 49.1 |
| (b) 仅 \<CLS\> | 50.12 / 49.50 | 88.56 / 86.83 | 48.2 |
| (c) "a region of a" + \<CLS\> | 51.31 / 50.96 | 88.20 / 86.73 | 48.7 |
| (d) ChatGPT 采样提示 + \<CLS\> | 49.50 / 49.40 | 88.49 / 86.73 | 48.9 |
| (e) \<LOC\> + "a region of a" + \<CLS\> | 49.56 / 47.60 | 88.23 / 87.07 | 49.1 |
| (f) **完整 APLGOS** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |

**高斯噪声强度 $\alpha$ 消融**：

| $\alpha$ | FPR95↓ (COCO/OI) | AUROC↑ (COCO/OI) | mAP↑ |
|----------|-------------------|-------------------|------|
| 0 | 51.63 / 50.88 | 87.86 / 87.24 | 49.2 |
| 0.5 | 51.90 / 51.48 | 87.55 / 87.02 | 48.9 |
| **1.0** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |
| 1.5 | 55.88 / 53.33 | 86.29 / 86.75 | 48.9 |
| 2.0 | 55.92 / 49.54 | 86.75 / 88.00 | 48.9 |

### 关键发现

1. **语句集采样 > 固定提示**：(c) vs (d) 表明从多样化的语句集中采样初始化提示比使用固定模板更有效
2. **位置 token 至关重要**：(c) vs (e) 加入 \<LOC\> 后 FPR95 从 50.96% 降至 47.60%
3. **噪声强度存在最优值**：$\alpha=1.0$ 最佳，太小则 OOD 采样空间过窄，太大则采样空间过大难以有效正则化
4. **ID:OOD=1:1 最佳**：与基线 VOS 的 2:1 不同，APLGOS 在 1:1 比例下性能最好，说明合成 OOD 质量更高
5. **OOD 提示数量 K=10000 最优**：太少无法充分覆盖隐空间，太多则随机性过大

## 亮点与洞察

1. **全虚拟化设计**：ID 提示、OOD 提示、OOD 图像全部由合成生成，仅 ID 图像来自真实数据，这使得方法对 ID 数据量需求更少
2. **ChatGPT 多轮标准化**：利用 LLM 生成多样化的区域级描述，既保证语义一致性又增加表达多样性
3. **低似然区域采样策略**：理论上合理——OOD 数据应出现在 ID 分布的低密度区域
4. **提示中隐式编码位置信息**：通过可学习的 \<LOC\> token 引入区域坐标，使提示具备空间感知能力
5. **真实场景泛化**：用 iPhone 14 Pro Max 拍摄的真实照片也能取得良好检测效果

## 局限与展望

1. **高斯分布假设**：假设 ID 提示嵌入服从高斯分布，实际分布可能更复杂；可以考虑混合高斯或流模型
2. **ChatGPT 依赖**：依赖 ChatGPT-3.5 进行标准化，不同 LLM 的标准化质量可能不同
3. **可扩展性**：对类别数量较多的场景（如数百个类别），per-class 高斯分布估计的准确性可能受限
4. **仅限 OOD 检测**：框架专为目标检测中的 OOD 设计，是否能推广到分类、分割等任务值得探索
5. **协方差估计**：使用绑定协方差矩阵（tied covariance）简化了计算，但可能丢失类别间分布差异

## 相关工作与启发

- **VOS (ICLR 2022)**：本文的直接基线，在特征空间合成虚拟离群点
- **CoOp/CoCoOp**：经典提示学习方法，APLGOS 的提示设计受其启发但专为 OOD 检测设计
- **CLIP**：提供预训练视觉-语言对齐能力，APLGOS 利用其文本编码器
- **启发**：将 LLM（ChatGPT）与提示学习结合用于 OOD 检测是一个新颖的方向，未来可以探索更多 LLM-assisted 的检测范式

## 评分
- 新颖性: ⭐⭐⭐⭐ （首次将提示学习引入 OOD 检测，ChatGPT 标准化提示有创意）
- 实验充分度: ⭐⭐⭐⭐ （四个数据集，丰富消融实验，含真实场景测试）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式推导详细）
- 价值: ⭐⭐⭐⭐ （OOD 检测 + VLM 的结合具有实用价值，FPR95 降低 7.76% 显著）

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)

<!-- RELATED:END -->
