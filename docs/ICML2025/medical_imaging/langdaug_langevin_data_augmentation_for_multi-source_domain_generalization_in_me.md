---
title: >-
  [论文解读] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation
description: >-
  [ICML 2025][医学图像][数据增强] LangDAug 利用基于对比散度训练的能量模型(EBM)，通过 Langevin 动力学在源域之间遍历生成中间样本，实现医学图像分割的多源域泛化，理论证明其诱导正则化效果并上界 Rademacher 复杂度。
tags:
  - ICML 2025
  - 医学图像
  - 数据增强
  - 域泛化
  - Langevin 动力学
  - 能量模型
  - 医学图像分割
---

# LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Imaging

**会议**: ICML 2025  
**arXiv**: [2505.19659](https://arxiv.org/abs/2505.19659)  
**代码**: https://github.com/backpropagator/LangDAug (有)  
**领域**: Medical Imaging  
**关键词**: 域泛化, 数据增强, Langevin 动力学, 能量模型, 医学图像分割

## 一句话总结
LangDAug 提出用基于能量模型(EBM)的 Langevin 动力学在多源域之间插值生成中间域增强样本，理论证明其诱导正则化效果并约束 Rademacher 复杂度，在眼底和前列腺 MRI 分割上超越 SOTA 域泛化方法。

## 研究背景与动机

**领域现状**：医学图像分割模型在跨域（不同设备、不同采集协议、不同机构）时泛化性差。域泛化(DG)方法试图在仅有源域训练数据的情况下提升对未见目标域的泛化。

**现有痛点**：DG 方法分两大流派：(1)表示学习方法寻求域不变特征，但依赖 ad-hoc 技术且缺乏理论保证；(2)数据增强方法通过合成样本丰富训练分布，但现有增强策略（如 RandConv、FedDG）缺乏原理性指导。

**核心矛盾**：如何系统性地生成在多个源域"之间"的有意义增强样本？简单的插值（如 MixUp）无法捕捉域间的非线性流形结构；随机增强可能生成不合理的样本。

**本文目标**：提供一种有理论保证的、原理性的域间数据增强方法。

**切入角度**：将多源域的数据分布用能量模型建模，用 Langevin 动力学在域间遍历，生成的中间态样本即为增强样本。

**核心 idea**：用 EBM 的 Langevin 采样在源域间"漫步"，经过的中间点就是有意义的域间增强数据。

## 方法详解

### 整体框架

- **输入**：多个源域的医学图像
- **EBM 训练**：用对比散度(contrastive divergence)训练域间能量模型
- **Langevin 采样**：从一个源域出发，沿能量模型的梯度场向另一源域方向采样
- **增强训练**：用原始 + Langevin 生成的增强样本训练分割模型
- **输出**：对未见目标域有更好泛化的分割模型

### 关键设计

1. **基于 EBM 的域间建模**:

    - 为每对源域训练一个 EBM $E_\phi(\mathbf{x})$
    - EBM 的特点：定义在数据空间上的能量函数，低能量区域对应数据密集区
    - 使用对比散度训练：正样本增加能量梯度，负样本（通过 MCMC 采样）减少能量梯度
    - **设计动机**：EBM 天然定义了数据流形上的势场，Langevin 动力学沿该势场运动可以生成流形上的新点

2. **Langevin 动力学域间遍历**:

    - 从源域 A 的样本 $\mathbf{x}_0$ 出发，执行 Langevin 更新：$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta}{2} \nabla_\mathbf{x} E_\phi(\mathbf{x}_t) + \sqrt{\eta} \xi_t$
    - 中间步的 $\mathbf{x}_t$ 是 A 和 B 域间的插值样本
    - 可控制步数来调节"域间距离"
    - **设计动机**：Langevin 采样在理论上收敛到 EBM 定义的分布，中间态反映了域间的渐变转移

3. **理论保证（Rademacher 复杂度界）**:

    - 证明 LangDAug 增强等价于对模型施加隐式正则化
    - 对于广义线性模型(GLM)，Langevin 增强的 Rademacher 复杂度被数据流形的本征维度上界约束
    - 这意味着：增强不会过度增加模型复杂度，而是在低维流形上提供有意义的新采样
    - **设计动机**：从理论上解释为什么这种增强有效

### 损失函数 / 训练策略

- EBM 训练：对比散度损失
- 分割模型：标准交叉熵/Dice 损失
- 增强策略：将 Langevin 样本与原始样本按比例混合
- 支持与现有域随机化方法（如 RandConv）组合使用

## 实验关键数据

### 主实验

| 数据集 | 目标域 | LangDAug | 之前 SOTA | 提升 |
|--------|--------|----------|-----------|------|
| Fundus 分割 | 域1 | 最优 | 多个 DG 方法 | 显著 |
| Fundus 分割 | 域2 | 最优 | 多个 DG 方法 | 显著 |
| Fundus 分割 | 域3 | 最优 | 多个 DG 方法 | 显著 |
| 前列腺 MRI | 域1 | 最优 | 多个 DG 方法 | 显著 |
| 前列腺 MRI | 域2 | 最优 | 多个 DG 方法 | 显著 |

### 消融实验

| 配置 | Dice↑ | 说明 |
|------|------|------|
| 无增强 (ERM) | 基线 | 标准经验风险最小化 |
| MixUp 增强 | 小幅提升 | 线性插值不够 |
| RandConv 增强 | 中等提升 | 随机卷积增强 |
| LangDAug alone | 更好 | Langevin 增强有效 |
| LangDAug + RandConv | **最优** | 互补增强 |
| 不同步数 | 有最优点 | 太少/太多步都次优 |

### 关键发现

- LangDAug 在两个标准医学图像 DG benchmark 上均达到 SOTA
- 与现有域随机化方法（RandConv）互补——组合使用效果最好
- Langevin 步数的选择影响增强质量，存在最优平衡点
- 理论预测的正则化效果在实验中得到验证

## 亮点与洞察

1. **理论与实践的统一**：不只是提出方法，还提供了 Rademacher 复杂度界
2. **与现有方法互补**：LangDAug 是增强策略，可以叠加到任何训练框架上
3. **物理直觉**：Langevin 动力学在域间"漫步"的直觉非常优雅
4. **医学应用导向**：直接在临床相关的 benchmark 上验证

## 局限与展望

1. EBM 训练本身可能不稳定，特别是在高维图像上
2. 每对源域需要训练一个 EBM，源域数量多时成本上升
3. 仅验证在 2D 医学图像分割上，3D 场景需探索
4. 理论保证限于 GLM，深度网络的理论推广需要进一步工作

## 相关工作与启发

- DRGen、SAML 等是医学图像 DG 的主要基线
- RandConv、FedDG 等提供了增强类方法的对比
- 启发：EBM + Langevin 的域间采样思路可能适用于 NLP、语音等其他跨域任务

## 评分
- 新颖性: ⭐⭐⭐⭐ EBM + Langevin 域间增强是新方法
- 实验充分度: ⭐⭐⭐⭐ 两个 benchmark、多个基线、理论验证
- 写作质量: ⭐⭐⭐⭐ 理论与实验平衡
- 价值: ⭐⭐⭐⭐ 对医学图像 DG 有实际价值
# LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation

**会议**: ICML 2025  
**arXiv**: [2505.19659](https://arxiv.org/abs/2505.19659)  
**代码**: https://github.com/backpropagator/LangDAug (有)  
**领域**: Medical Imaging  
**关键词**: 数据增强, 域泛化, Langevin 动力学, 能量模型, 医学图像分割

## 一句话总结
LangDAug 利用基于对比散度训练的能量模型(EBM)，通过 Langevin 动力学在源域之间遍历生成中间样本，实现医学图像分割的多源域泛化，理论证明其诱导正则化效果并上界 Rademacher 复杂度。

## 研究背景与动机

1. **领域现状**：医学图像分割模型在跨域场景下（如不同医院、设备、成像参数）泛化能力严重不足。域泛化(DG)方法通过表示学习或数据增强来改善这一问题。

2. **现有痛点**：表示学习方法寻求域不变特征，但常依赖 ad-hoc 技术且缺乏形式化保证。数据增强方法虽然效果接近或更好，但现有增强策略（如随机风格迁移）缺乏原理性设计，不清楚"应该增强到什么程度"。

3. **核心矛盾**：如何设计一种有理论保证的数据增强策略，能系统性地产生填充源域间隙的有效中间样本？

4. **本文目标**：提出一种基于能量模型和 Langevin 动力学的原理性数据增强方法。

5. **切入角度**：将不同源域视为能量景观中的不同低谷，用 Langevin 动力学在域之间"行走"生成中间域样本。

6. **核心 idea**：训练 EBM 捕获源域的联合能量景观，然后用 Langevin 采样器在域间遍历，生成的中间样本用于训练分割模型。

## 方法详解

### 整体框架

- **输入**：多个源域的医学图像及其分割标签
- **第一步**：训练 EBM 来建模多源域的联合分布
- **第二步**：从任一源域出发，运行 Langevin 动力学 MCMC 链，生成中间域样本
- **第三步**：用原始数据 + 增强数据联合训练分割模型
- **输出**：泛化能力更强的分割模型

### 关键设计

1. **能量模型 (EBM) 的域间建模**:
    - 用对比散度 (Contrastive Divergence) 训练 EBM：$p_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$
    - EBM 的能量景观自然编码了不同源域的分布
    - 不同域对应不同的低能量区域，域间是高能量"山丘"
    - **设计动机**：EBM 提供了一个连续的能量景观，使得域间的过渡是平滑的

2. **Langevin 动力学遍历域间**:
    - 从域 A 的样本出发，运行 Langevin MCMC：$\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\eta}{2} \nabla_\mathbf{x} E_\theta(\mathbf{x}_k) + \sqrt{\eta} \epsilon_k$
    - 随着步数增加，样本从域 A 过渡到域 A 和域 B 之间的中间区域
    - 步数控制增强样本与源域的距离
    - **设计动机**：Langevin 动力学的平稳分布就是 EBM 定义的分布，保证采样的合理性

3. **理论保证**:
    - 证明 LangDAug 诱导正则化效果
    - 对于广义线性模型(GLM)，LangDAug 将 Rademacher 复杂度上界为数据流形的内在维度
    - 这意味着增强的效果与数据的真实复杂度相关，而非模型的参数量
    - **设计动机**：提供形式化的泛化保证，不只是经验性的改善

### 损失函数 / 训练策略

- EBM 训练：对比散度损失 $\mathcal{L}_{CD} = \mathbb{E}_{p_\text{data}}[E_\theta(\mathbf{x})] - \mathbb{E}_{p_\theta}[E_\theta(\mathbf{x}')]$
- 分割模型训练：标准分割损失（交叉熵 + Dice），在原始 + 增强数据上训练
- LangDAug 可与其他域随机化(domain randomization)方法组合使用

## 实验关键数据

### 主实验

| 数据集 | 指标 | LangDAug | 之前 SOTA DG | 提升 |
|--------|------|----------|-------------|------|
| 眼底分割 (Fundus) | Dice↑ | SOTA | 次优 | 显著 |
| 前列腺 MRI | Dice↑ | SOTA | 次优 | 显著 |
| 眼底 + Domain Rand. | Dice↑ | 最优 | Domain Rand. alone | 互补提升 |

### 消融实验

| 配置 | Dice | 说明 |
|------|------|------|
| 无增强 | 基线 | 仅源域数据训练 |
| 随机增强 | 小幅提升 | 传统增强 |
| 域随机化 (DR) | 中等提升 | 现有 SOTA 增强 |
| LangDAug alone | 更好 | 优于 DR |
| **LangDAug + DR** | **最佳** | 两者互补 |
| 不同 Langevin 步数 | 步数敏感 | 过多步数可能离源域太远 |

### 关键发现

- LangDAug 在眼底分割和前列腺 MRI 分割两个 benchmark 上均超越 SOTA 域泛化方法
- LangDAug 与现有域随机化方法互补——组合使用效果最佳
- Langevin 步数是关键超参数：过少无效，过多可能生成离分布太远的样本
- 理论上的正则化效果在实验中得到验证

## 亮点与洞察

1. **理论扎实**：Rademacher 复杂度上界提供了比经验方法更强的保证
2. **物理直觉**：能量景观 + Langevin 动力学的类比直观且准确
3. **互补性**：与现有增强方法正交，可叠加使用
4. **跨模态验证**：眼底(fundus)和 MRI 两种不同模态均有效

## 局限与展望

1. EBM 训练本身不稳定，需要仔细调参
2. Langevin 采样速度较慢，增加训练时间
3. 当前仅验证 2D 分割，3D 体积分割的扩展需探索
4. 增强样本的标签需要额外处理（本文如何获得增强样本的分割标签需关注）

## 相关工作与启发

- DSU, CIRL 等域泛化方法是主要比较对象
- AdvBias, FedDG 等增强方法提供了基线
- 启发：EBM + Langevin 的增强策略可推广到其他医学影像任务（如分类、检测）

## 评分
- 新颖性: ⭐⭐⭐⭐ EBM + Langevin 用于域泛化增强是新颖的组合
- 实验充分度: ⭐⭐⭐⭐ 两个 benchmark + 互补性实验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 对医学图像域泛化有实用价值

<!-- RELATED:START -->

## 相关论文

- [Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2025/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [Controllable Latent Space Augmentation for Digital Pathology](../../ICCV2025/medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)
- [Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](../../AAAI2026/medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)
- [MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)

<!-- RELATED:END -->
