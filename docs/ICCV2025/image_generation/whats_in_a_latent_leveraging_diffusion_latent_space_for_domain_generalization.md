---
title: >-
  [论文解读] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization
description: >-
  [ICCV 2025][图像生成][域泛化] 深入分析了不同预训练模型（CLIP、DiT、SD、MAE、DINOv2、ResNet）隐空间的域分离能力，发现扩散模型特征在无监督情况下最擅长分离域信息，并提出 GUIDE 框架——用扩散特征发现伪域表征并增广分类器特征，在 5 个 DomainBed 数据集上无需域标签即取得 66.3% 平均准确率（超越 ERM 基线 +2.6%，在 TerraIncognita 上 +4.3%），且优于大多数需要域标签的方法。
tags:
  - ICCV 2025
  - 图像生成
  - 域泛化
  - 扩散特征
  - 伪域发现
  - 潜在空间分析
  - 无域标签
---

# What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization

**会议**: ICCV 2025  
**arXiv**: [2503.06698](https://arxiv.org/abs/2503.06698)  
**代码**: [xthomasbu/GUIDE](https://xthomasbu.github.io/GUIDE)  
**领域**: 域泛化/扩散模型表征  
**关键词**: 域泛化, 扩散特征, 伪域发现, 潜在空间分析, 无域标签

## 一句话总结

深入分析了不同预训练模型（CLIP、DiT、SD、MAE、DINOv2、ResNet）隐空间的域分离能力，发现扩散模型特征在无监督情况下最擅长分离域信息，并提出 GUIDE 框架——用扩散特征发现伪域表征并增广分类器特征，在 5 个 DomainBed 数据集上无需域标签即取得 66.3% 平均准确率（超越 ERM 基线 +2.6%，在 TerraIncognita 上 +4.3%），且优于大多数需要域标签的方法。

## 研究背景与动机

### 问题定义

域泛化（Domain Generalization）要求模型在训练时利用多个源域数据，使其在完全未见的测试域上仍能保持良好的分类性能。更具挑战性的设定是：训练时也不知道域标签（即不知道每个样本属于哪个域）。

### 已有方法的不足

**ERM 基线过强**：Gulrajani & Lopez-Paz 已证明大多数复杂域泛化方法在严格评估下甚至不如简单的 ERM（经验风险最小化）

**需要域标签的方法（CORAL, SagNet, DANN 等）**：依赖训练时的显式域标签，在域标签不可用或有噪声时不适用

**数据增强方法**：用文生图模型做数据增强需要微调扩散模型，成本高且需要测试数据

**现有伪域方法**：DA-ERM 需要训练专门的域原型网络，AdaClust 使用的特征空间域分离能力有限

### 核心动机

**关键洞察**：不同预训练目标生成的特征空间在"编码域信息 vs 类别信息"的倾向上存在显著差异。通过 T-SNE 可视化和 NMI 定量分析，发现：
- **扩散模型**（DiT, SD-2.1）：因为生成目标完全不关心类别标签，域特定变化（风格、纹理、环境）在隐空间中自然涌现，域 NMI 分数最高
- **判别模型**（ResNet）：特征空间以类别聚合为主，域信息被类别信息压制
- **对比学习**（CLIP）：聚焦高层语义对齐，域和类别 NMI 都很低

这启发了一种互补策略：用扩散特征发现域结构，然后增广判别模型的特征。

## 方法详解

### 整体框架

GUIDE 分两步：（1）用冻结的扩散模型提取特征并通过 K-Means++ 聚类发现伪域，计算每个伪域的质心作为域表征；（2）将伪域表征通过 RBF 核岭回归变换到分类器特征空间，拼接到分类器输入后训练"域自适应分类器"。

### 关键设计

#### 1. **无监督伪域发现**

- **功能**：从冻结的预训练扩散模型中提取样本特征，通过无监督聚类发现数据中隐含的域结构
- **核心思路**：使用特征提取器 $\Psi$（扩散模型）计算每个训练样本的特征表征，然后应用 K-Means++ 聚类得到 $K$ 个簇（伪域）。每个簇的质心 $\widehat{\Psi}_k$ 作为该伪域的紧凑表征。每个训练样本被分配到最近簇，其伪域表征即为该簇质心：
  $$\widehat{\Psi}_x = \widehat{\Psi}_k, \quad k = \arg\min_j \|\Psi(x) - \widehat{\Psi}_j\|$$
  聚类的数量 $K$ 通过简单启发式确定：$K = \max(\{1,3,5\} \times n_c, 200)$，其中 $n_c$ 为类别数。
- **设计动机**：聚类有两个作用——（1）平滑样本特异性噪声，创建更稳定的域表征；（2）与直接拼接原始扩散特征相比，伪域质心提供更紧凑且更具域代表性的信息。实验证明聚类带来的增益（PACS +3.3%）远大于不聚类直接拼接（+1.3%）。

#### 2. **特征空间变换与分类器增广**

- **功能**：将伪域表征从 $\Psi$ 空间变换到分类器特征空间 $\Phi$，然后拼接用于训练
- **核心思路**：定义变换函数 $\mathcal{T}: \Psi \mapsto \Phi$，使用 RBF 核岭回归实现非线性映射。具体地，$\mathcal{T}$ 将伪域 $k$ 的质心 $\widehat{\Psi}_k$ 映射到属于该簇的所有样本的 $\Phi(x)$ 特征均值。训练输入变为：
  $$[\Phi(x); \mathcal{T}(\widehat{\Psi}_k)]$$
  采用对数调度（logarithmic schedule）定期更新 $\mathcal{T}$，以适应训练过程中 $\Phi$ 的变化。测试时，先用 $\Psi$ 提取特征，分配到最近伪域簇，再应用 $\mathcal{T}$ 后拼接。
- **设计动机**：RBF 核善于建模非线性距离关系，已在域适应中被验证有效。对数调度初期频繁更新、后期减少，既保证及时适应又控制计算开销。

#### 3. **扩散特征的域分离能力分析**

- **功能**：系统分析 6 种预训练模型在 7 个数据集上的域分离能力（域 NMI vs 类别 NMI）
- **核心思路**：使用 Normalized Mutual Information（NMI）量化聚类簇与真实域/类别标签的一致性：
  $$\text{NMI}(U,V) = \frac{2 \cdot I(U,V)}{H(U) + H(V)}$$
  理想的域增广特征空间应有**高域 NMI + 低类别 NMI**——即特征按域聚合而非按类聚合。
  
  关键发现：
  - DiT 在全局风格差异大的数据集上最优（PACS 域NMI=0.85, Synth-Artists=0.89）
  - SD-2.1 在需要细粒度空间特征的数据集上最优（TerraIncognita 域NMI=0.55）
  - 扩散模型的类别 NMI 普遍很低（DiT 在 PACS 仅 0.08），说明生成目标不鼓励类别聚合
- **设计动机**：这一分析不仅指导了 $\Psi$ 的选择，还为理解不同预训练范式捕获的信息类型提供了新视角。

### 损失函数 / 训练策略

- 标准交叉熵分类损失
- ResNet-50（AugMix 预训练）作为分类器 $\Phi$
- DomainBed 默认设置：batch 32/域，lr=5e-5，5001 步，无 dropout，weight decay=0
- Leave-one-domain-out 交叉验证，3 个种子平均
- 扩散特征提取：DiT 使用 block 14、t=50；SD-2.1 使用 up_ft:1 层、t=50

## 实验关键数据

### 主实验

**DomainBed 五数据集泛化性能（测试准确率 %）**：

| 方法 | 需域标签 | VLCS | PACS | OH | TI | DN | 平均 |
|------|---------|------|------|------|------|------|------|
| ERM | ✗ | 76.6 | 83.8 | 67.2 | 47.0 | 44.1 | 63.7 |
| CORAL | ✓ | 78.8 | 86.2 | 68.7 | 47.6 | 41.5 | 64.5 |
| SagNet | ✓ | 77.8 | 86.3 | 68.1 | 48.6 | 40.3 | 64.2 |
| MIRO | ✗ | 79.0 | 85.4 | 70.5 | 50.4 | 44.3 | 65.9 |
| AdaClust | ✗ | 78.9 | 87.0 | 67.7 | 48.1 | 43.6 | 64.9 |
| **GUIDE-BEST** | **✗** | **78.5** | **87.1** | **68.6** | **51.3** | **45.9** | **66.3** |

### 消融实验

**不同 $\Psi$ 对域泛化的影响（测试准确率 %）**：

| $\Psi$ 特征 | VLCS | PACS | OH | TI | 平均 |
|------------|------|------|------|------|------|
| DiT | 78.5 | **87.1** | 68.4 | 48.2 | 70.6 |
| SD-2.1 | 77.0 | 86.9 | **68.6** | **51.3** | **71.0** |
| CLIP | 76.8 | 84.7 | 64.6 | 47.4 | 68.4 |
| DINOv2 | 77.3 | 84.9 | 68.3 | 48.4 | 69.7 |
| MAE | 76.4 | 84.6 | 65.2 | 50.2 | 69.1 |
| ERM (无增广) | 76.6 | 83.8 | 67.2 | 47.0 | 68.7 |

**增强训练策略的效果（PACS / TerraIncognita）**：

| 方法 | PACS | TI |
|------|------|------|
| ERM | 83.8 | 47.0 |
| ERM++ | 88.0 | 50.7 |
| GUIDE + ERM++ | **89.2** | **53.6** |

### 关键发现

1. **扩散特征一致最优**：DiT 和 SD-2.1 在所有数据集上都优于非扩散特征，且互为补充
2. **DiT vs SD-2.1 互补**：DiT 擅长全局风格差异（PACS +3.3%），SD-2.1 擅长细粒度空间差异（TI +4.3%），与架构特性一致
3. **CLIP 几乎无用**：CLIP 特征的域 NMI 和类别 NMI 都很低，做域增广几乎无增益
4. **聚类是必要的**：不聚类直接拼接仅提升 +1.3%，聚类后提升 +3.3%（PACS），说明平滑噪声的作用
5. **GUIDE 与训练优化正交**：可叠加 SWAD、MIRO、ERM++ 等策略进一步提升
6. **无需域标签即超越需域标签方法**：GUIDE-BEST 平均 66.3% 超过 CORAL (64.5%)、SagNet (64.2%) 等

## 亮点与洞察

1. **深刻的特征空间分析**：系统比较 6 种预训练范式在域分离 vs 类别分离上的行为，揭示了扩散模型"类别无关、域信息丰富"的独特特性
2. **极致简单的方法**：整个框架就是"聚类+拼接"，没有复杂损失、没有对抗训练、没有元学习——证明了正确利用互补特征比设计复杂算法更重要
3. **首次利用冻结扩散特征做域泛化**：之前工作要么用扩散模型做数据增强（需要fine-tune），要么用文本条件生成；GUIDE 完全免训练地利用扩散特征
4. **Synth 数据集的构造**：用 SDXL 生成的 Synth-Artists 和 Synth-Photography 为未来研究提供了可控域偏移的基准

## 局限与展望

1. **分类器仅用 ResNet-50**：更强的骨干（如 ViT-L）可能改变扩散特征的增益幅度
2. **扩散特征提取有推理成本**：每个样本需过一次完整的扩散模型前向，相比直接训练增加了预处理时间
3. **$K$ 的选择依赖简单启发式**：更好的聚类数量选择（如基于信息论准则）可能进一步改善
4. **仅用单层扩散特征**：结合多层/多时间步特征可能提供更丰富的域信息
5. **OfficeHome 上增益有限**：域 NMI 分数低（0.25-0.28），说明当域界限模糊时方法效果减弱

## 相关工作与启发

- 与 DA-ERM 的区别：DA-ERM 使用专门训练的域原型网络+需要域标签；GUIDE 使用冻结的预训练扩散特征+无需域标签
- 与 AdaClust 的区别：AdaClust 用分类器自身的早期卷积层做聚类；GUIDE 使用互补的外部扩散特征空间
- 启发：不同预训练范式的特征可以被视为对图像不同方面的"投影"——判别模型投影到类别轴，生成模型投影到样式/域轴——巧妙组合不同投影即可增强鲁棒性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次深入分析扩散特征的域分离特性并应用于域泛化，洞察深刻但方法较简单
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 个数据集、6 种特征提取器、域 NMI + 类别 NMI 双维分析、多种训练策略消融
- **写作质量**: ⭐⭐⭐⭐⭐ — 分析极其透彻，每个数据集的域偏移特性与特征空间优劣的对应关系清晰
- **价值**: ⭐⭐⭐⭐ — 方法简单有效，但更大的价值在于对预训练特征空间的深入理解

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Latent Space Imaging](../../CVPR2025/image_generation/latent_space_imaging.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](../../CVPR2025/image_generation/probability_density_geodesics_in_image_diffusion_latent_space.md)
- [\[ICML 2025\] Hessian Geometry of Latent Space in Generative Models](../../ICML2025/image_generation/hessian_geometry_of_latent_space_in_generative_models.md)

<!-- RELATED:END -->
