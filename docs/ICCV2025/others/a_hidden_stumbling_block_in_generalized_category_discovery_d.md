---
title: >-
  [论文解读] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention
description: >-
  [ICCV 2025][广义类别发现] 本文发现GCD任务中ViT模型在处理无标签数据时存在注意力分散（关注背景而非前景目标）的隐患，提出Attention Focusing (AF)机制，通过Token重要性评估 (TIME) + Token自适应剪枝 (TAP) 级联去除无关token，在SimGCD上取得最高15.4%的提升，且计算开销极小。
tags:
  - ICCV 2025
  - 广义类别发现
  - 注意力分散
  - Token剪枝
  - 自适应剪枝
  - 即插即用模块
---

# A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention

**会议**: ICCV 2025  
**arXiv**: [2507.14315](https://arxiv.org/abs/2507.14315)  
**代码**: [https://github.com/Afleve/AFGCD](https://github.com/Afleve/AFGCD) (有)  
**领域**: 自监督/类别发现  
**关键词**: 广义类别发现, 注意力聚焦, Token剪枝, ViT, 细粒度识别  

## 一句话总结
发现GCD中未标注数据（尤其是未知类别）的ViT注意力会分散到背景区域（distracted attention），提出Attention Focusing（AF）模块通过多尺度token重要性度量+自适应剪枝来纠正注意力，作为即插即用模块在SimGCD上最高带来15.4%的性能提升。

## 研究背景与动机

**领域现状**：广义类别发现（GCD）旨在利用已标注的已知类别知识，对同时包含已知和未知类别的未标注数据进行聚类分类。主流方法分为非参数方法（对比学习+K-means聚类）和参数方法（SimGCD等使用原型分类器联合训练）。

**现有痛点**：现有方法几乎都忽略了一个隐藏问题——distracted attention。可视化分析发现：标注数据的[CLS] token注意力一致聚焦在前景物体上，但未标注数据（尤其未知类别）的注意力严重分散到背景区域，导致特征质量下降。

**核心矛盾**：根本原因在于数据增强的不对称性——标注数据同一类别包含多样化背景，模型自然学会关注物体而非背景；未标注数据的增强仅产生轻微背景变化，模型可以利用背景中的虚假相关性作为捷径来完成自监督/无监督学习。

**本文目标**：如何在不引入外部模型的前提下，纠正GCD模型在未标注数据上的注意力分散问题？

**切入角度**：从token剪枝的角度出发——如果能自适应地识别并移除与任务无关的背景token，模型就只能基于前景区域做决策。关键挑战是如何度量token重要性，因为未标注数据没有标签。

**核心 idea**：用仅在标注数据上训练的多尺度可学习query token度量每个patch的重要性，然后自适应剪枝低重要性token，使模型被迫关注前景物体。

## 方法详解

### 整体框架
输入图像经ViT分成patch token后，在每个ViT block（最后一个除外）插入一个TIME模块来度量每个token的重要性得分。所有层的得分经多尺度聚合后，TAP模块自适应剪枝低重要性token。剩余token经过最后一个ViT block和平均池化后送入任意GCD Head。AF作为即插即用模块不改变原GCD方法的Head设计。

### 关键设计

1. **Token Importance Measurement (TIME)**:

    - 功能：在每个ViT block内度量各patch token对分类任务的重要性
    - 核心思路：引入一个可学习的query向量 $\mathbf{Q} \in \mathbb{R}^{1 \times D}$，与输入token做交叉注意力：$\mathbf{s} = \mathbf{Q}\mathbf{K}^T / \sqrt{D}$，得到重要性得分向量。然后通过Aggregator将得分加权的token表示 $\mathbf{r} = \text{Softmax}(\mathbf{s})\mathbf{V}$ 送入辅助分类器（仅预测已知类别），用交叉熵损失 $\mathcal{L}_{ce}$ 训练
    - 设计动机：**仅用标注数据训练但能泛化到未标注数据**——因为标注和未标注数据共享相似的视觉风格特征，在标注数据上学到的"什么token对分类重要"的知识可以迁移。辅助分类器通过stop-gradient与backbone隔离，避免梯度冲突。测试时丢弃辅助分类器，仅保留 $\mathbf{Q}$

2. **Token Adaptive Pruning (TAP)**:

    - 功能：基于多尺度重要性得分自适应剪枝非信息性token
    - 核心思路：将$L-1$层TIME输出的得分向量（排除[CLS]token的得分）取softmax后按层平均：$\mathbf{s}^m = \frac{1}{L-1}\sum_{l=1}^{L-1}\text{Softmax}(\hat{\mathbf{s}}_l)$。按得分从小到大排序，累积低得分token直到总和达到阈值 $\tau$，移除这些token。剩余token + [CLS]进入最后一个ViT block
    - 设计动机：多尺度聚合比单层得分更鲁棒（实验验证多尺度比仅用倒数第二层好3-5%）。自适应阈值而非固定数量剪枝能适应不同图像的背景复杂度

3. **单视图TAP策略**:

    - 功能：在两个数据增强视图中只对一个视图执行TAP
    - 设计动机：TAP本质上等价于非规则裁剪增强。如果两个视图都剪枝，会过度移除信息导致模型泛化能力下降；单视图剪枝保留了一个完整视图的信息，同时迫使模型在剪枝视图上聚焦前景

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{gcd} + \lambda \sum_{l=1}^{L-1} \mathcal{L}_{ce}^l$，其中 $\mathcal{L}_{gcd}$ 是基础GCD方法的损失，$\mathcal{L}_{ce}^l$ 是每层TIME的辅助分类损失
- TIME的辅助分类器仅在标注数据上有梯度
- 与backbone之间stop-gradient隔离

## 实验关键数据

### 主实验

| 数据集 | 方法 | All ACC | Old ACC | New ACC | 提升(All) |
|--------|------|---------|---------|---------|-----------|
| CUB | SimGCD | 60.3 | 65.6 | 57.7 | - |
| CUB | SimGCD+AF | 69.0 | 74.3 | 66.3 | **+8.7** |
| Stanford Cars | SimGCD | 53.8 | 71.9 | 45.0 | - |
| Stanford Cars | SimGCD+AF | 67.0 | 80.7 | 60.4 | **+13.2** |
| FGVC-Aircraft | SimGCD | 54.2 | 59.1 | 51.8 | - |
| FGVC-Aircraft | SimGCD+AF | 59.4 | 68.1 | 55.0 | **+5.2** |
| ImageNet-100 | SimGCD | 83.0 | 93.1 | 77.9 | - |
| ImageNet-100 | SimGCD+AF | 85.4 | 94.6 | 80.8 | **+2.4** |

### 消融实验

| 配置 | CUB All | Stanford Cars All | 说明 |
|------|---------|-------------------|------|
| SimGCD (baseline) | 60.3 | 53.8 | 基线 |
| + AF (单层TIME) | 65.8 | 61.2 | 仅用倒数第二层query |
| + AF (多尺度TIME) | 69.0 | 67.0 | 多层聚合，显著更好 |
| + AF on CMS | +0.9 | +8.7 | 在CMS方法上也有效 |
| + AF on SelEx | +5.8 | +2.3 | 在SelEx上也有效 |
| + AF on GET | +4.0 | +1.2 | 在GET上也有效 |

### 关键发现
- 细粒度数据集（复杂背景）上AF提升最大：Stanford Cars +13.2%、CUB +8.7%；通用数据集（简单背景）如CIFAR-10提升有限（+0.7%）
- 多尺度聚合比单层度量效果显著更好，低层捕获局部纹理、高层捕获语义信息的互补性很重要
- 单视图TAP优于双视图TAP——双视图会过度移除信息导致泛化下降
- AF是轻量级的，计算开销极小（额外参数主要是每层一个query向量+一个小FFN+辅助分类器，测试时辅助分类器丢弃）

## 亮点与洞察
- **distracted attention的发现和量化分析**是核心贡献：首次揭示GCD中未标注数据注意力分散的现象及其数据增强根源。这个观察具有普适性，任何以自监督学习为基础的开放世界识别任务都可能存在类似问题。
- **仅用标注数据训练却能泛化到未标注数据的TIME设计**非常巧妙——利用了标注和未标注数据视觉风格相似的先验，避免了"未标注数据没有监督信号"的鸡生蛋问题。
- **自适应阈值剪枝**优于固定比例剪枝，因为不同图像的前景/背景比例差异很大。这个思路可以迁移到任何需要token选择的ViT任务中。

## 局限与展望
- **AF在简单背景数据集上提升有限**：如CIFAR-10/100和Herbarium-19，背景本身不构成干扰时AF甚至可能在新类上轻微下降
- **不提升前景特征提取能力**：AF只解决"看哪里"的问题，不解决"怎么看"的问题；对于前景本身就很难区分的细粒度任务需要结合其他方法
- **query向量的泛化依赖**：假设标注和未标注数据风格相似，如果分布差异大可能失效
- 改进思路：可以探索在未标注数据上也引入某种自监督的token重要性信号（如重建目标），或者将AF与细粒度特征增强方法结合

## 相关工作与启发
- **vs SimGCD**：SimGCD作为简洁有效的参数化GCD方法，但完全没有处理注意力问题；AF作为插件显著提升其性能
- **vs Cropr**：Cropr在每个ViT block固定剪枝数量的token，本文用多尺度自适应剪枝更灵活，效果更好
- **vs AptGCD/MOS**：同期的竞争方法也关注背景干扰，但需要更复杂的模块设计或外部模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现独到（distracted attention），方法设计简洁但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 7个数据集、4个GCD基线方法的泛化验证、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，可视化分析直观
- 价值: ⭐⭐⭐⭐ 即插即用模块对GCD社区有实际价值，细粒度场景提升显著
**领域**: 类别发现 / 开放世界学习  
**关键词**: 广义类别发现, 注意力分散, Token剪枝, 自适应剪枝, 即插即用模块  

## 一句话总结
本文发现GCD任务中ViT模型在处理无标签数据时存在注意力分散（关注背景而非前景目标）的隐患，提出Attention Focusing (AF)机制，通过Token重要性评估 (TIME) + Token自适应剪枝 (TAP) 级联去除无关token，在SimGCD上取得最高15.4%的提升，且计算开销极小。

## 背景与动机
广义类别发现（Generalized Category Discovery, GCD）旨在利用已标注的已知类别数据的知识，对同时包含已知和未知类别的无标签数据进行分类。现有GCD方法普遍采用预训练ViT作为特征提取骨干网络，通过[CLS] token的embedding进行分类。

现有方法（如SimGCD、SPTNet、CMS等）主要关注如何利用无监督/自监督学习提升无标签数据上的性能，但忽略了一个隐性问题：**注意力分散（Distracted Attention）**。具体来说，当处理无标签数据时，模型不仅关注图像中的关键目标，还会关注与任务无关的背景区域，导致特征提取质量下降。

作者通过可视化SimGCD在CUB数据集上的自注意力图发现：有标签数据的[CLS] token始终聚焦前景目标，而无标签数据（尤其是未知类别）的[CLS] token会明显关联到背景区域。

## 核心问题
**为什么无标签数据会出现注意力分散？** 作者假设数据增强是部分原因：对于有标签数据，同一类别的不同图像往往有不同背景，模型自然学会关注前景目标。但对于无标签数据，增强通常只是对同一张图做微小变换，背景变化很小，模型容易利用背景中的虚假相关性作为"捷径"进行自监督/无监督学习。

这个问题的重要性在于：注意力分散直接降低了特征表示的质量——如果模型"看的不是关键物体"，那后续的分类/聚类自然不准。而且这个问题在细粒度数据集（背景复杂）上尤其严重。

## 方法详解

### 整体框架
AF机制插入到现有GCD模型的ViT骨干中，由两个级联模块组成：
- **输入**: 原始图像经ViT分patch后的token序列
- **TIME模块**: 插入到ViT的每个block中（最后一个block除外），在每个block产生一个token重要性分数向量
- **TAP模块**: 聚合所有TIME模块的多尺度重要性分数，自适应地剪掉不重要的token
- **输出**: 剩余token经过最后一个ViT block后做平均池化，送入GCD分类头

关键点是AF只对一个增强视图做剪枝（single-view TAP），另一个视图保持完整，这样剪枝相当于一种非规则图像裁剪增强。

### 关键设计

1. **Token重要性评估 (TIME)**:
    - 在每个ViT block中引入一个可学习的查询向量 $\mathbf{Q}$
    - 将输入token作为Key $\mathbf{K}$ 和Value $\mathbf{V}$，通过交叉注意力计算每个token的重要性分数：$\mathbf{s}(\mathbf{Q}, \mathbf{K}) = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D}}$
    - 通过Softmax加权聚合得到图像表示 $\mathbf{r} = \text{Softmax}(\mathbf{s})\mathbf{V}$，再经FFN处理
    - 用辅助分类器（仅在已标注数据上训练）指导 $\mathbf{Q}$ 学习：目标是让 $\mathbf{Q}$ 给信息量大的token分配更高分数
    - **关键**: 用stop-gradient隔离辅助分类器和骨干网络，防止梯度冲突；测试时丢弃辅助分类器，只保留 $\mathbf{Q}$

2. **Token自适应剪枝 (TAP)**:
    - 聚合所有TIME块的多尺度分数：$\mathbf{s}_m = \frac{1}{L-1}\sum_{l=1}^{L-1}\text{Softmax}(\hat{\mathbf{s}}^l)$
    - 排除[CLS] token的分数（[CLS]始终保留）
    - 按分数从低到高排序，累计分数不超过阈值 $\tau$ 的token被剪掉
    - **自适应**意味着不同图像剪掉的token数量不同——背景复杂的图像剪更多，目标占比大的图像剪更少

3. **单视图TAP vs. 多视图TAP**:
    - 只在一个增强视图上做剪枝，另一个保持完整
    - 理由：单视图TAP等价于非规则裁剪增强，帮助模型聚焦关键物体；多视图TAP虽然减少了背景噪声，但也削弱了泛化能力

### 损失函数 / 训练策略
总损失为：

$$\mathcal{L} = \mathcal{L}_{gcd} + \lambda \sum_{l=1}^{L-1}\mathcal{L}_{ce}^l$$

- $\mathcal{L}_{gcd}$: 基线GCD方法的损失（如SimGCD的表示学习 + 分类器学习损失）
- $\mathcal{L}_{ce}^l$: 每个TIME模块中辅助分类器的交叉熵损失（仅在已标注数据上计算）
- $\lambda$: 平衡系数，典型值0.05
- $\tau$: 剪枝阈值，随数据集调整（CUB: 0.2, Stanford Cars: 0.01, FGVC-Aircraft: 0.01）
- 仅微调ViT最后一个block + 各TIME模块，batch size 128，200 epochs

## 实验关键数据

**细粒度数据集（核心结果）**:

| 数据集 | 指标 | SimGCD+AF | SimGCD | 提升 | 之前最佳 |
|--------|------|-----------|--------|------|----------|
| CUB | All ACC | 69.0 | 60.3 | +8.7 | AptGCD 70.3 |
| Stanford Cars | All ACC | 67.0 | 53.8 | +13.2 | MOS 64.6 |
| Stanford Cars | New ACC | 60.4 | 45.0 | **+15.4** | MOS 56.7 |
| FGVC-Aircraft | All ACC | 59.4 | 54.2 | +5.2 | MOS/AptGCD 61.1 |

**通用数据集**:

| 数据集 | 指标 | SimGCD+AF | SimGCD | 提升 |
|--------|------|-----------|--------|------|
| CIFAR-10 | All ACC | 97.8 | 97.1 | +0.7 |
| CIFAR-100 | All ACC | 82.2 | 80.1 | +2.1 |
| ImageNet-100 | All ACC | 85.4 | 83.0 | +2.4 |
| Herbarium-19 | All ACC | 45.5 | 44.0 | +1.5 |

**AF对其他GCD方法的泛化性**:

| 方法 | CUB All | Stanford Cars All | Aircraft All |
|------|---------|------------------|-------------|
| CMS → CMS+AF | 67.3→68.2 | 53.1→61.8 (+8.7) | 54.2→57.5 |
| SelEx → SelEx+AF | 73.4→79.2 (+5.8) | 58.9→61.2 | 57.2→62.8 (+5.6) |
| GET → GET+AF | 75.2→77.3 | 78.3→81.5 (+3.2) | 57.4→59.5 |

### 消融实验要点
- **多尺度 vs. 单尺度**: 仅用倒数第二层的query做剪枝（AF-），性能远不如多尺度聚合（AF），说明不同ViT层关注的patch差异很大
- **查询仅在有标签数据上学习**: 若在全部数据（含无标签）上训练query，性能显著下降（如Stanford Cars: 67.0→63.0），因为无监督信号会引入噪声
- **TAP vs. 固定剪枝**: 固定剪掉K个patch（K=32/64/128）均不如自适应TAP——K太大会丢关键信息，K太小减不够背景噪声
- **平均池化 vs. [CLS] token**: 剪枝后用剩余token平均池化（+AF）远优于仅用[CLS] token（+AF([CLS])），因为最后一个block的[CLS]无法充分聚合来自各patch的信息
- **计算开销**: 训练参数从81.82M增至132.21M，但推理参数几乎不变（81.82M→81.83M）；训练时间增加约12%，推理时间增加约25%

## 亮点
- **问题发现精准**: 第一个系统研究GCD中注意力分散现象的工作，通过可视化清晰展示了有标签 vs. 无标签数据在注意力上的差异，非常有说服力
- **设计巧妙**:  TIME模块用独立的可学习query做交叉注意力（而非依赖质量可能不好的[CLS] token），且仅在有标签数据上训练——利用了GCD中"有标签数据是干净的"这一特性
- **即插即用**: AF可以直接加到CMS、SelEx、GET等不同GCD方法上均有提升，通用性好
- **单视图剪枝 = 非规则裁剪增强**的insight很有意思，将token pruning和数据增强联系起来

## 局限与展望
- **对简单背景数据集提升有限**: 在CIFAR-10/100等低分辨率、简单背景数据集上提升很小甚至下降（CIFAR-100 New: -1.3），说明AF本质上只解决背景干扰问题
- **无法提升前景特征的判别力**: 作者在结论中承认，AF有效抑制了背景干扰，但不能提升模型从关键目标中提取更具判别性特征的能力
- **阈值 $\tau$ 需要针对数据集手动调**: 不同数据集的最佳 $\tau$ 差异很大（0.01 ~ 0.2），没有自动确定机制
- **领域分类不当**: 这是开放世界学习/类别发现的工作，不属于自动驾驶——只是motivation提到自动驾驶和医学影像作为应用场景
- **未与外部模型结合**: 未探索用SAM/DINO等已有的前景感知模型来辅助token筛选

## 与相关工作的对比
- **vs. AptGCD/MOS**: 这两个同期工作也关注背景干扰问题。AptGCD和MOS在部分数据集上与AF性能接近，但AF的模块设计更简洁，不依赖外部模型。不过AptGCD/MOS在某些数据集上（如CUB）略优于AF
- **vs. SPTNet**: SPTNet通过空间prompt tuning优化ViT，采用交替训练策略，计算成本更高。在Stanford Cars上SPTNet（59.0）显著弱于AF（67.0），但两者在Aircraft上接近
- **vs. Token Pruning方法（EViT、ToMe、Cropr）**: 传统token pruning用[CLS] token的注意力权重做剪枝，但GCD中无标签数据的[CLS]质量不佳，容易引入误导。AF的独立query机制避免了这个问题。Cropr使用固定数量剪枝，而AF的自适应策略更灵活

## 启发与关联
- **跨域迁移潜力**: AF的"独立query + 辅助分类器"设计可以推广到其他半监督/开放世界任务（如Open-Set Detection、Novel Class Discovery），只要存在有标签数据可以训练query
- **增强 vs. 剪枝**: 单视图TAP=非规则裁剪增强这一insight，暗示token pruning和数据增强之间存在更深层的联系，值得进一步探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一个系统研究GCD注意力分散的工作，问题发现新颖，方法设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 7个数据集、4个GCD方法的泛化实验、多维度消融、计算效率分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化有说服力，动机阐述到位
- 价值: ⭐⭐⭐⭐ 揭示了GCD中的一个重要盲点，AF作为即插即用模块有实用价值，但对简单场景效果有限

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](../../CVPR2025/others/mos_modeling_object-scene_associations_in_generalized_category_discovery.md)
- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](../../ACL2025/others/the_hidden_attention_of_mamba_models.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](../../ECCV2024/others/attnzero_efficient_attention_discovery_for_vision_transformers.md)

<!-- RELATED:END -->
