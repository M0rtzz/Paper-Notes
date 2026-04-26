---
title: >-
  [论文解读] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
description: >-
  [CVPR 2026][医学图像][WSOL] 提出 SFDA-DeP，受机器遗忘启发，将 SFDA 重新定义为"识别并纠正预测偏差"的迭代过程：对 dominant class 中高熵不确定样本执行"遗忘"操作迫使模型放弃偏向性预测，对可靠样本保持自训练，同时用像素级分类器锚定定位能力，在跨器官/跨中心病理基准上持续优于现有 SFDA 方法。
tags:
  - CVPR 2026
  - 医学图像
  - WSOL
  - 域适应
  - 预测偏差矫正
  - 机器遗忘
  - 病理图像
---

# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**会议**: CVPR 2026  
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)  
**代码**: [anonymous.4open.science/r/SFDA-DeP-1797/](https://anonymous.4open.science/r/SFDA-DeP-1797/)  
**作者**: Alexis Guichemerre 等 (ÉTS Montréal, Sorbonne Université, University of York, McGill University)  
**领域**: 医学图像 / 病理图像分析  
**关键词**: WSOL, Source-Free Domain Adaptation, 预测偏差矫正, 机器遗忘, 病理图像

## 一句话总结

提出 SFDA-DeP，受机器遗忘启发，将 SFDA 重新定义为"识别并纠正预测偏差"的迭代过程：对 dominant class 中高熵不确定样本执行"遗忘"操作迫使模型放弃偏向性预测，对可靠样本保持自训练，同时用像素级分类器锚定定位能力，在跨器官/跨中心病理基准上持续优于现有 SFDA 方法。

## 研究背景与动机

### 领域现状

弱监督目标定位（WSOL）模型在数字病理学中广受关注——只需图像级标签（如"有肿瘤/无肿瘤"）即可同时完成分类和 ROI 定位，大幅降低了对像素级标注的依赖。代表方法包括基于 CAM 的 PixelCAM、基于注意力的 DeepMIL、基于 Transformer 的 SAT 等。

### 现有痛点

WSOL 模型在**跨域部署**时（不同器官、不同医疗中心、不同染色/扫描协议）性能严重退化。更关键的是，这种退化不仅是低层外观变化导致的——作者实验发现，从 GlaS（结肠腺体）迁移到 CAMELYON16/17（乳腺淋巴结转移检测）时，模型几乎将所有样本都预测为 cancer，产生极端的**预测偏差**（prediction bias）。

### 核心矛盾

Source-Free Domain Adaptation（SFDA）是解决跨域部署的主流框架——不需访问源数据，仅用未标注目标数据进行适配。但现有 SFDA 方法（如 SFDA-DE、CDCL、ERL）本质上依赖自训练（pseudo-labeling + self-training），隐含假设源分类器在目标域上仍能产生合理的预测。当预测已经严重偏向 dominant class 时，自训练反而**放大偏差**——伪标签中 dominant class 占主导，模型越训越偏。Fig.1 清晰展示了这一恶性循环：SFDA-DE 在适配后偏差反而恶化，几乎坍缩到单一类别。

### 切入角度

作者从**机器遗忘（machine unlearning）**获得灵感：不是要模型遗忘某个类别或源知识，而是让模型"遗忘"错误的类别边界。具体地，如果模型对某些 dominant class 样本的预测本身就不确定（高熵），那就应该主动压制这些预测，迫使决策边界重新调整。

### 核心 idea

用"遗忘高熵 dominant 样本 + 保留可靠样本"的双集机制周期性矫正预测偏差，替代传统 SFDA 的无差别自训练。

## 方法详解

### 整体框架

SFDA-DeP 的输入是一个在源域预训练好的 WSOL 模型 $f$ 和未标注的目标数据集 $\mathbb{T}$。适配过程迭代进行：

1. 用当前模型预测所有目标样本 → 检测 dominant class（预测频率过高的类别）
2. 从 dominant class 样本中选出高熵（不确定）的子集作为 **forget set** $\mathbb{B}_f$
3. 其余样本组成 **retain set** $\mathbb{B}_r$
4. 对 forget set 施加"遗忘"损失，对 retain set 施加保持损失，联合像素级定位损失
5. 每 $m$ 个 epoch 重建 forget/retain sets，动态跟踪边界移动

### 关键设计

**1. Forget Set 构建**

- **功能**：从 dominant class 预测样本中选出最不确定的 top-$\rho$ 子集
- **核心思路**：用归一化熵 $H(x)$ 衡量预测不确定性。定义 $\mathbb{B} = \{x \in \mathbb{T}: \hat{y}(x) \in \mathcal{B}\}$（dominant class 样本集），然后 $\mathbb{B}_f = \text{top}_\rho(\mathbb{B}, H(x))$
- **设计动机**：高熵样本本来就处于决策边界附近，模型对它们"硬塞"到 dominant class 的信心不足。对这些样本执行遗忘最高效——它们最可能是被错误归入 dominant class 的样本

**2. Forget Loss（遗忘损失）**

- **功能**：迫使模型放弃对 forget set 样本的 dominant class 预测
- **核心公式**：$\mathcal{L}_{\text{forget}} = \mathbb{E}_{x_i \in \mathbb{B}_f}[-\log(1 - p_i(\hat{y}))]$
- 最小化此损失等价于最大化模型对当前伪标签 $\hat{y}$ 的交叉熵——即让模型在这些样本上"忘掉"之前的预测
- **与 retain loss 的配合**：$\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_i \in \mathbb{B}_r}[-\log(p_i(\hat{y}))]$ 是标准交叉熵，保持可靠样本的预测不变。两者协同重新定义类别边界

**3. 像素级定位损失**

- **功能**：训练一个轻量级像素级分类器 $h$，将每个像素分为前景（ROI）/背景
- **核心思路**：对每个预测类 $k$，选低熵（最可靠）样本子集 $D_{\text{loc}}$；从源模型提取 CAM 生成像素级伪标签 $\bm{Y}$；用二值交叉熵训练 $h$：
  $$\mathcal{L}_{\text{loc}} = -(1-\bm{Y}_p)\log(h(z_p)_0) - \bm{Y}_p\log(h(z_p)_1)$$
- **设计动机**：仅靠分类级别的偏差矫正不够——跨域后目标 ROI 外观可能截然不同，需要像素级监督锚定定位特征，防止适配过程中定位能力漂移

**4. 动态重采样**

- 每 $m$ 个 epoch 用当前模型重新计算预测分布和熵，重建 forget/retain sets
- 避免初期错误的遗忘决策不可逆——随着边界移动，之前的 forget 样本可能变得可靠，之前的 retain 样本可能需要遗忘

### 损失函数 / 训练策略

总损失：

$$\mathcal{L} = \lambda_{\text{retain}}\mathcal{L}_{\text{retain}} + \lambda_{\text{forget}}\mathcal{L}_{\text{forget}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}}$$

超参数搜索范围：$\lambda_{\text{retain}}, \lambda_{\text{forget}} \in \{0.2, 0.5, 1.0, 2.0\}$，$\lambda_{\text{loc}} \in \{0.5, 1.0, 5.0\}$，$\rho \in \{5\%, 15\%, 25\%\}$。

## 实验关键数据

### 主实验（GlaS → CAMELYON16/17，跨器官+跨中心）

| WSOL 模型 | 方法 | Avg PxAP | Avg CL |
|-----------|------|----------|--------|
| PixelCAM | Source only | 36.9 | 49.3 |
| PixelCAM | SFDA-DE | 28.0 | 54.6 |
| PixelCAM | ERL | 25.4 | 59.9 |
| PixelCAM | RGV | 34.7 | 52.1 |
| PixelCAM | **SFDA-DeP** | **44.1** | **67.1** |
| SAT | Source only | 21.3 | 52.1 |
| SAT | SFDA-DE | 21.6 | 68.7 |
| SAT | ERL | 22.2 | 68.9 |
| SAT | **SFDA-DeP** | **30.3** | **69.2** |
| DeepMIL | Source only | 20.9 | 49.8 |
| DeepMIL | SFDA-DE | 20.5 | 53.9 |
| DeepMIL | ERL | 16.2 | 57.8 |
| DeepMIL | **SFDA-DeP** | **40.7** | **73.4** |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无 $\mathcal{L}_{\text{loc}}$ | PxAP 下降明显 | 缺失像素级锚定，定位漂移 |
| 静态采样（不重建 forget/retain） | 性能明显劣于动态 | 初期错误遗忘不可逆 |
| 重采样频率过低 | 性能下降 | 边界变化跟踪不及时 |
| 重采样频率过高 | 性能略降 | 集合不稳定，训练抖动 |

### 关键发现

1. **SFDA 基线在强偏差下全面失效**：SFDA-DE 在多个中心 CL 停留在 50%（相当于随机猜），且 PxAP 常低于 source-only，说明自训练确实在放大偏差
2. **DeepMIL 上增益最大**（+20.2 PxAP，+19.5 CL vs SFDA-DE），说明偏差矫正对弱定位能力的模型帮助更大
3. **C17-0 中心增益最显著**（PixelCAM 从 50.0% CL 跳到 86.2%），这个中心最初偏差最严重
4. **动态重采样是关键**：静态 forget/retain 划分会导致错误不可逆累积
5. **定性分析**：SFDA-DeP 的 CAM 激活聚焦在肿瘤组织上，而 SFDA 基线在强偏移下会激活背景区域

## 亮点与洞察

1. **问题诊断精准**：清晰指出 SFDA 失败的根因不是"适配不够"而是"偏差放大"，Fig.1 的实验分析直观有力。这种先定位瓶颈再设计方案的研究范式值得学习
2. **机器遗忘的创造性借用**：不是真正"忘掉类别"，而是用遗忘机制重塑决策边界——forget loss $-\log(1-p(\hat{y}))$ 简洁优雅，实现成本极低但效果显著
3. **跨架构一致性**：在 CNN-based（PixelCAM/DeepMIL）和 Transformer-based（SAT）三种 WSOL 上均有效，说明方法与底层架构解耦，通用性好
4. **双集动态重采样**：每 $m$ epoch 重建避免了伪标签噪声的累积效应，实际上是一种隐式的课程学习策略

## 局限与展望

1. **二分类限制**：论文实验都是二分类（tumor vs normal），dominant class 偏差在多类别场景下模式更复杂，forget set 构建策略需要调整
2. **ρ 超参数敏感**：forget 比例 $\rho$ 需要手动搜索，过小则矫正力度不足，过大则可能误遗忘正确预测。自适应 ρ 值得探索
3. **CAM 伪标签质量**：像素级定位依赖源模型 CAM 的质量，如果源域 CAM 本身就不准（常见问题），$\mathcal{L}_{\text{loc}}$ 可能引入噪声
4. **计算开销**：每 $m$ epoch 需要对全量目标数据重新推理计算熵和重建集合，大规模数据下可能成为瓶颈
5. **缺少与 UDA 的对比**：只比了 SFDA 方法，未和有源数据的 UDA 方法对比，难以评估 source-free 约束带来的性能损失

## 相关工作与启发

- **vs SFDA-DE**：SFDA-DE 通过分布估计做适配，但无偏差矫正机制，在 dominant bias 下完全失效（CL 常停在 50%，PxAP 低于 source-only）。SFDA-DeP 的 forget 机制直接解决了这个根本问题
- **vs ERL/RGV**：ERL 用正则化稳定训练，RGV 用生成重放，但都未显式处理 class prediction imbalance。在强偏移下同样无力
- **vs Machine Unlearning (Basak & Yin, ECCV'24)**：传统机器遗忘是删除特定类或数据的知识，SFDA-DeP 创新性地用遗忘机制来矫正决策边界而非删除知识
- **启发**：forget/retain 双集策略可以推广到其他存在 pseudo-label bias 的自训练场景，如半监督学习中的长尾分布、自监督预训练中的类别不均

## 评分

- 新颖性: ⭐⭐⭐⭐ 机器遗忘到偏差矫正的迁移思路新颖，但整体框架相对简洁
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、三种 WSOL 模型、四个 SFDA 基线，消融和定性分析充分
- 写作质量: ⭐⭐⭐⭐ 问题动机讲述清晰，Fig.1 诊断有力，数学符号规范
- 价值: ⭐⭐⭐⭐ 指出了 SFDA 在病理中的核心失败模式并给出有效方案，实用性强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

<!-- RELATED:END -->
