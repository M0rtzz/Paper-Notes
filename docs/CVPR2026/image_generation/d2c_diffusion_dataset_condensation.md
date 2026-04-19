---
title: >-
  [论文解读] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation
description: >-
  [CVPR 2026][图像生成][扩散模型训练] 首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型训练
  - 数据集压缩
  - 难度评分
  - 区间采样
  - REPA加速
---

# D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation

**会议**: CVPR 2026  
**arXiv**: [2507.05914](https://arxiv.org/abs/2507.05914)  
**代码**: 无  
**领域**: 图像生成 / 高效训练 / 数据集压缩  
**关键词**: Dataset Condensation, Diffusion Training, Difficulty Score, Interval Sampling, REPA

## 一句话总结

首次将数据集压缩引入扩散模型训练，提出D2C两阶段框架（Select+Attach），仅用0.8% ImageNet数据在40K步达到FID 4.3，比REPA快100倍、比vanilla SiT快233倍。

## 研究背景与动机

**领域现状**：扩散模型在图像生成领域取得了卓越成果，但训练代价极其高昂。以SiT-XL/2为例，在完整ImageNet（128万图像）上需要700万步训练才能收敛，耗费数百GPU天。近年来，研究者从模型侧（架构改进、注意力优化、表示对齐如REPA）入手提升训练效率，但从数据侧直接缩减训练集规模的可能性几乎未被探索。

**现有痛点**：数据集压缩（Dataset Condensation, DC）在判别式模型上已有成熟研究（如SRe2L、RDED），但这些方法直接应用于扩散模型训练会完全崩溃——FID高达80-166。根本原因在于：判别式DC方法优化的是类别判别特征，而非真实图像的像素分布，生成的合成图像在结构和语义保真度上严重不足，无法支撑生成模型的训练需求。

**核心矛盾**：扩散模型训练对数据质量和分布覆盖的要求远高于判别式学习，简单的数据裁剪（random sampling、K-Center、Herding）在极低数据预算下无法保证训练质量，而现有的DC框架又是为判别式任务量身定制的。

**本文目标**：能否通过精心设计的数据集压缩策略，将训练数据缩减到原来的0.8%-8%，同时保持甚至超越全量数据的扩散模型生成质量？

**切入角度**：作者观察到扩散模型本身能通过去噪损失编码样本的学习难度信息，据此提出利用扩散模型自身作为数据"评分器"，从难度维度选择最有信息量的训练子集，再为选中样本附加丰富的语义和视觉先验。

**核心 idea**：用扩散难度分数做区间采样选子集（Select），再用T5文本嵌入+DINOv2视觉特征增强每个样本（Attach），实现数据极端压缩下的高效扩散训练。

## 方法详解

### 整体框架

D2C是一个两阶段框架，输入为完整训练集（如ImageNet 1.28M图像），输出为一个极小的增强子集（如10K或50K图像及其附加的语义/视觉元数据）：

- **Select阶段**：利用预训练扩散模型为每个样本计算扩散难度分数（diffusion difficulty score），在类内按难度排序后通过固定间隔$k$均匀采样，得到兼顾多样性与可学习性的紧凑子集。
- **Attach阶段**：为选中的每个样本预计算并附加两类信息——(1) DC-Embedding：T5文本编码器生成的类别语义嵌入；(2) Visual Representation：DINOv2提取的patch级视觉特征。这些附加信息存储在磁盘上，训练时直接加载。
- **训练阶段**：在压缩后的增强数据集上训练扩散模型，使用标准去噪损失加上REPA式的视觉对齐损失。

### 关键设计

1. **扩散难度分数（Diffusion Difficulty Score）**：

    - 功能：量化每个训练样本对扩散模型的学习难度，用于排序和选择
    - 核心思路：定义$s_{diff}(x) = -\mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$，即样本在预训练扩散模型上的平均去噪损失的负数。通过贝叶斯推导，$p_\theta(c|x) \propto p_\theta(x|c)$，因此去噪损失直接反映样本属于目标类别的置信度——损失越高，样本越复杂/模糊，学习难度越大
    - 设计动机：实验发现最简单（Min）和最困难（Max）的样本都不理想——Min样本（干净、背景单一）可学习性高但多样性不足，Max样本（杂乱、歧义大）难以优化。中等难度样本在分布匹配上偏差最小，因此需要一种覆盖多难度级别的采样策略

2. **区间采样（Interval Sampling）**：

    - 功能：在难度排序后的样本中进行均匀间隔采样，平衡易学样本和困难样本的覆盖
    - 核心思路：在每个类别$y$内，将样本按$s_{diff}$升序排列，每隔$k$个取一个样本：$\mathcal{D}_{IS} = \bigcup_{y=1}^{C}\{x^{(i)} \in \mathcal{D}_y \mid i \in \{0, k, 2k, \ldots\}\}$。$k$的最优值与数据预算成正比——10K子集用$k=96$，50K子集用$k=16$
    - 设计动机：与只取最简单或只取中等难度相比，区间采样自然覆盖从易到难的完整难度谱，避免极端采样的偏差。实验证明其效果优于K-Center、Herding等基于几何/特征多样性的方法，说明在扩散训练中难度维度的分布覆盖比特征空间的几何覆盖更重要

3. **双条件嵌入（DC-Embedding）与视觉信息注入**：

    - 功能：为每个选中样本附加丰富的条件信号，弥补极小数据集下语义和视觉信息的不足
    - 核心思路：DC-Embedding将T5文本编码器对类名描述（如"a photo of a cat"）生成的文本嵌入$t_c$经1D卷积处理后，通过残差MLP与可学习类别嵌入$e_c$融合：$y_{text} = \text{MLP}(\tilde{t}_c) + \tilde{t}_c + e_c$。视觉信息则由DINOv2提取patch级特征$y_{vis} = f_{vis}(x) \in \mathbb{R}^{N \times d}$，截取前$h$个token作为对齐目标
    - 设计动机：纯类别嵌入从头训练难以捕捉丰富的语义关系；T5文本嵌入天然编码了语义层级（如相似品种的狗在嵌入空间中自然聚类），与可学习嵌入融合既保留了预训练语义又保持了训练灵活性。DINOv2视觉特征则提供实例级的空间结构先验，弥补语义信息在类内变异捕捉上的不足

### 损失函数 / 训练策略

总损失为去噪损失与语义对齐损失的加权和：

$$\mathcal{L}_{total} = \mathcal{L}_{diff} + \lambda \mathcal{L}_{proj}$$

- $\mathcal{L}_{diff}$：标准去噪损失，以DC-Embedding为条件输入
- $\mathcal{L}_{proj}$：REPA式的视觉对齐损失，将扩散模型中间层特征经投影头映射后与DINOv2特征做cosine相似度对齐
- $\lambda = 0.5$为默认权重
- 优化器：Adam，学习率1e-4
- 硬件：8×A800/4090，10K子集训练仅需7.4小时

## 实验关键数据

### 主实验

**ImageNet 256² 加速对比（SiT-XL/2, CFG=1.5）：**

| 方法 | 数据量 | 训练步数 | gFID-50K↓ |
|------|--------|---------|-----------|
| Vanilla SiT-XL/2 | 1.28M | 7M | 8.3 |
| + REPA | 1.28M | 4M | 5.9 |
| + REPA-E | 1.28M | 235K | 5.9 |
| + REG | 1.28M | 200K | 5.0 |
| **D2C** | **10K (0.8%)** | **40K** | **4.3** |
| **D2C** | **50K (4%)** | **180K** | **2.78** |

**ImageNet 256² 不同DC方法对比（DiT-L/2, 0.8% 10K, 100K步）：**

| 方法 | gFID↓ | sFID↓ | IS↑ | Precision↑ |
|------|-------|-------|-----|-----------|
| RDED | 166.2 | 60.1 | 10.8 | 0.09 |
| SRe2L | 104.2 | 20.2 | 14.1 | 0.20 |
| **D2C** | **4.2** | **11.0** | **283.6** | **0.72** |

**ImageNet 512² 高分辨率（0.8% 10K, 300K步, CFG=1.5）：**

| 模型 | 方法 | gFID↓ | sFID↓ | IS↑ | Precision↑ |
|------|------|-------|-------|-----|-----------|
| DiT-L/2 | Random | 17.1 | 12.8 | 130.6 | 0.64 |
| DiT-L/2 | **D2C** | **5.8** | **15.1** | **318.9** | **0.77** |
| SiT-L/2 | Random | 5.0 | 13.6 | 316.9 | 0.76 |
| SiT-L/2 | **D2C** | **4.22** | **11.6** | **289.7** | **0.79** |

### 消融实验

**Select与Attach各组件贡献（DiT-L/2, 10K, gFID-10K）：**

| Select | DC-Embedding | Visual Embedding | gFID↓ |
|--------|-------------|-----------------|-------|
| ✗ | ✗ | ✗ | 37.07 |
| ✗ | ✓ | ✓ | 8.79 |
| ✓ | ✗ | ✗ | 14.96 |
| ✓ | ✗ | ✓ | 10.37 |
| ✓ | ✓ | ✗ | 9.01 |
| ✓ | ✓ | ✓ | **7.62** |

**Wall-Clock时间对比（SiT-XL/2）：**

| 方法 | 评分时间 | 训练时间 | 总时间 | gFID↓ |
|------|---------|---------|-------|-------|
| REPA | N/A | 750h | 750h | 5.9 |
| D2C (无Select) | N/A | 7.4h | 7.4h | 5.6 |
| D2C (从头训scorer) | 1.9h | 33.6h | 35.5h | 4.9 |
| D2C (预训练scorer) | 2.1h | 7.4h | 9.5h | 4.3 |

### 关键发现

- 判别式DC方法（SRe2L、RDED）在扩散训练中完全崩溃（FID > 80），证实生成任务需要专门设计的DC策略
- 区间采样的最优$k$值与数据预算大致成正比：10K对应$k=96$，50K对应$k=16$
- 即使使用从头训练的弱scorer（基准gFID 11.5），D2C仍能达到gFID 4.9，远超随机选择的37.07
- 仅使用Attach阶段（不做Select）即可达到gFID 5.6，已超过REPA的5.9，说明信息增强本身价值巨大

## 亮点与洞察

- **233×训练加速**是一个极其惊人的数字——原本需要几周、数百GPU天的训练压缩到不到10小时，且生成质量不降反升（FID 4.3 vs 8.3）。这意味着扩散模型训练的"资源壁垒"可能被大幅降低，学术实验室也能高效迭代大规模生成模型。
- **扩散难度分数的贝叶斯推导**非常优雅：通过$p_\theta(c|x) \propto p_\theta(x|c)$将去噪损失与样本学习难度建立理论联系，为数据选择提供了有原则的评分标准，避免了依赖外部分类器或启发式指标。
- **两阶段解耦设计**巧妙地将"选什么数据"和"如何增强数据"分离——Select和Attach各自独立有效（分别降低FID从37到15和37到8.8），组合后进一步提升，这种正交设计使得每个模块都可以独立改进。

## 局限与展望

- Select阶段依赖预训练扩散模型作为scorer，在完全冷启动场景下需要额外的预训练开销（虽然从头训练scorer也可行，但增加26小时成本）
- 仅验证了C2I（类别到图像）设定，T2I（文本到图像）仅在附录中有初步探索，大规模T2I场景（如Stable Diffusion训练数据的压缩）的适用性待验证
- 区间值$k$需要根据数据预算手动选择，虽有经验法则（与数据量成正比）但未实现自动化搜索

## 相关工作与启发

- **vs REPA（模型侧加速）**：REPA通过表示对齐加速但仍需全量1.28M数据训练4M步。D2C仅用0.8%数据+REPA的对齐loss，效果更好（FID 4.3 vs 5.9）且快100×，说明数据侧优化与模型侧优化可以互补
- **vs SRe2L/RDED（判别式DC）**：这些方法在扩散训练中FID > 80完全失败，根因在于它们的合成图像保留了判别特征但破坏了像素级分布结构
- **vs 数据剪枝（Li et al.）**：近期工作也做数据选择+类别重加权，但不做Attach阶段的信息增强，且仅在小规模实验验证
- 扩散难度分数的概念可推广到其他生成模型（自回归模型、VAE）的数据选择问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将数据集压缩系统性引入扩散模型训练，扩散难度分数+区间采样的组合是原创贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 3种数据比例(0.8%/4%/8%)、2种分辨率(256/512)、2种架构(DiT/SiT)、多种基线、详尽消融
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，Bayesian推导与实验紧密配合，附录极其详尽
- 价值: ⭐⭐⭐⭐⭐ 233×加速+极端数据压缩定义了扩散训练效率的新标杆，对学术和工业界都有重大意义
---
title: >-
  [论文解读] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation
description: >-
  [CVPR 2026][图像生成][扩散模型训练] 首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型训练
  - 数据集压缩
  - 难度评分
  - 区间采样
  - REPA加速
---

# D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation

**会议**: CVPR 2026  
**arXiv**: [2507.05914](https://arxiv.org/abs/2507.05914)  
**代码**: 无（但方法完全可复现）  
**领域**: 图像生成 / 高效训练 / 数据集蒸馏  
**关键词**: 扩散模型训练, 数据集压缩, 难度评分, 区间采样, REPA加速  

## 一句话总结
首次将数据集压缩(Dataset Condensation)应用于扩散模型训练，提出D2C两阶段框架——Select阶段用扩散难度分数+区间采样选出紧凑子集、Attach阶段为每个样本附加文本和视觉表示——仅用0.8% ImageNet(10K图像)在40K步即达FID 4.3,比REPA快100×、比vanilla SiT快233×。

## 背景与动机
扩散模型训练极其耗资源——SiT-XL/2需700万步在128万图像上训练。REPA等方法从模型侧优化(表示对齐)，但数据侧减少训练集的可能性未被探索。数据集压缩(DC)在判别式模型上研究成熟，但直接将已有DC方法(SRe2L/RDED)用于扩散训练导致崩溃——因为DC方法优化类别判别特征而非真实图像分布，生成的合成图像结构和语义保真度差。

## 核心问题
能否通过数据集压缩将训练数据减少到原来的0.8-8%,同时保持扩散模型的生成质量和大幅加快训练收敛?

## 方法详解

### 整体框架
两阶段：(1) **Select**：用预训练扩散模型计算每个样本的去噪难度分数$s_{diff}(x) = -p_\theta(x|c) \propto -\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$，按难度排序后用间隔$k$均匀采样，平衡易学样本和困难样本的多样性。(2) **Attach**：为选中样本附加两类信息——DC-Embedding（T5文本嵌入+可学习类别嵌入的残差融合）和Visual Information（DINOv2 patch特征用作REPA式对齐目标）。

### 关键设计
1. **扩散难度分数**：$s_{diff}(x)$等于样本在预训练扩散模型上的平均去噪损失。直觉：高损失=模型难预测=样本复杂/模糊。通过Bayesian推导证明，$p_\theta(c|x) \propto p_\theta(x|c)$，因此去噪损失直接反映样本属于某类的置信度。关键发现：最易(Min)和最难(Max)样本都不好——Min太简单缺多样性，Max太嘈杂无法学习。中等难度的分布discrepancy最小（U型曲线,Fig.8 Right）。

2. **区间采样(Interval Sampling)**：在类内按难度排序后每隔$k$个取一个。$k$与数据量正相关——10K子集用$k=96$，50K用$k=16$。这天然覆盖了从易到中难的样本范围，同时避免极端困难样本。比"仅取中间"(Medium)性能更好——因为完全跳过容易样本会丧失基础分布覆盖。

3. **双条件嵌入(DC-Embedding)**：T5编码器将类名("a photo of a cat")编码为文本嵌入，经1D卷积+残差MLP与可学习类别嵌入融合。比纯类别嵌入好(FID 9.01 vs 14.96)——因为文本嵌入天然编码了语义关系(相似品种的狗在T-SNE中自然聚类,Fig.9)。

### 损失函数 / 训练策略
$\mathcal{L}_{total} = \mathcal{L}_{diff} + 0.5 \mathcal{L}_{proj}$，$\mathcal{L}_{diff}$是标准去噪损失（以DC-Embedding为条件），$\mathcal{L}_{proj}$是DINOv2特征对齐（REPA式）。Adam lr=1e-4，8×A800/4090。10K子集训练仅需7.4小时（比REPA的750小时少101×）。

## 实验关键数据
**ImageNet 256² (SiT-XL/2, CFG=1.5):**

| 方法 | 数据量 | 训练步数 | gFID-50K |
|--------|------|------|------|
| Vanilla SiT | 1.28M | 7M | 8.3 |
| + REPA | 1.28M | 4M | 5.9 |
| + REPA-E | 1.28M | 235K | 5.9 |
| + REG | 1.28M | 200K | 5.0 |
| **D2C** | **10K (0.8%)** | **40K** | **4.3** |
| **D2C** | **50K (4%)** | **180K** | **2.78** |

SRe2L/RDED在扩散训练中完全崩溃(FID > 80)——证实判别式DC方法不适用于生成任务。

D2C在512²、CIFAR-10上也work：CIFAR-10 gFID 3.95 (random 9.72)。

### 消融实验要点
- **Select单独有效**：仅选择(无Attach)从37.07降至14.96 FID
- **DC-Embedding贡献最大**：Select+DC Emb=9.01，Select+Visual=10.37，Select+Both=7.62
- **所有视觉编码器都有帮助**：DINOv2-L(7.62) > CLIP-L(8.59) > MoCov3-L(8.78) > MAE-L(9.23) >> 无(37.07)
- **最优$k$与数据量成比例**：10K→k=96, 50K→k=16，约=数据量/类别数×比例
- **预训练scorer非必需**：从头训练scorer(FID 4.9)也远超random(37.07)

## 亮点
- **233×加速**是惊人的数字——意味着原本需要几周的训练可以在几小时内完成
- 首次将数据集压缩引入扩散训练——填补了一个明显的学术空白
- 扩散难度分数的信息论推导优雅——从$p(c|x) \propto p(x|c)$到去噪损失的等价性
- 区间采样比K-Center/Herding/random在扩散训练中都好——说明难度排序比几何/特征多样性更重要
- 极低的附加开销——Select仅2h，Attach预计算后存储在磁盘上

## 局限与展望
- 依赖预训练扩散模型做难度评分——冷启动场景需要额外一步
- 仅验证了C2I(类别到图像),T2I(文本到图像)仅有初步探索(Appendix G)
- 区间$k$需要手动选择——虽然有经验法则但不完全自动化
- 10K子集的类别覆盖(10/类)可能限制类别多样性
- 未与T2I的数据效率方法(如PixArt的数据策划)直接对比

## 与相关工作的对比
- **vs REPA (模型侧加速)**：REPA加速但仍用全量数据(1.28M)。D2C仅用0.8%数据(10K)+REPA的视觉对齐,效果更好(4.3 vs 5.9)且快100×
- **vs SRe2L/RDED (判别式DC)**：这些方法在扩散训练中完全失败(FID>80)——因为优化目标是判别特征而非像素分布
- **vs 数据剪枝 (Pruning then Reweighting)**：近期Li et al.的方法也做数据选择但不做Attach阶段且仅在小规模验证
- **vs HoneyBee (CVPR'26 VLM数据)**：HoneyBee研究VLM推理数据策划,D2C研究扩散训练数据策划——思路相似,领域不同

## 启发与关联
- "扩散难度分数"概念可推广到其他生成模型的数据选择——如自回归模型、VAE
- 与`ideas/image_generation/20260316_dit_compression_understanding.md`相关——D2C从数据侧减少训练成本,该idea从模型侧压缩,二者正交
- **潜在idea**: 将D2C的Select策略用于**持续学习**——当新数据到来时,仅选择最有信息量的样本来增量更新扩散模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将DC引入扩散训练,扩散难度分数+区间采样的组合是原创贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 3种数据比例、2种分辨率、2种架构(DiT/SiT)、5种基线、详尽消融
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰,理论推导与实验紧密配合,附录极其详尽
- 价值: ⭐⭐⭐⭐⭐ 233×加速+数据极端压缩——定义了扩散模型训练效率的新标杆

<!-- RELATED:START -->

## 相关论文

- [Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective](accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)
- [One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_f.md)
- [TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)

<!-- RELATED:END -->
