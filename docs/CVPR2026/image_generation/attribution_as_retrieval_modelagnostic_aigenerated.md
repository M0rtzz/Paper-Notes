---
title: >-
  [论文解读] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution
description: >-
  [CVPR 2026][图像生成][deepfake attribution] 将 AI 生成图像的归属问题从分类范式重新定义为实例检索范式，提出基于低位平面指纹的模型无关框架 LIDA，通过无监督预训练和少样本归属适应，在零样本和少样本设置下实现 SOTA 的 Deepfake 检测和图像归属性能。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "deepfake attribution"
  - "image retrieval"
  - "bit-plane"
  - "model-agnostic"
  - "few-shot"
---

# Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**会议**: CVPR 2026  
**arXiv**: [2603.10583](https://arxiv.org/abs/2603.10583)  
**代码**: [GitHub](https://github.com/hongsong-wang/LIDA)  
**领域**: Image Forensics / AI Safety  
**关键词**: Deepfake attribution, image forensics, retrieval-based, bit-plane, model-agnostic

## 一句话总结
提出 LIDA，将 AI 生成图像溯源从分类问题转化为检索问题，利用低位平面指纹捕获生成器特异性伪影，配合无监督预训练和少样本自适应，在零/少样本设置下实现 SOTA 的 Deepfake 检测和图像溯源。

## 研究背景与动机
**领域现状**：AIGC 技术快速发展，Deepfake 检测已有较多进展，但 AI 生成图像的溯源（归因到具体生成模型）仍是开放问题。现有方法分两大类：生成式水印（需访问生成模型）和基于分类的溯源。

**现有痛点**：(1) 生成式水印需要完全访问生成模型并修改其架构，缺乏灵活性和通用性；(2) 闭集溯源假设所有生成器训练时已知，无法应对新兴模型；(3) 开集溯源虽考虑未知生成器，但仍是分类范式，需要大量未标注生成图像重新训练，对新模型适应慢。

**核心矛盾**：新生成模型层出不穷（如 Midjourney、DALL-E、Stable Diffusion），分类范式每次都需要重新训练以扩展类别，且需要收集大量新模型的数据——这在实际场景中不现实。

**本文目标**：设计一个模型无关、可扩展到未见生成器的图像溯源框架，仅需少量示例即可快速适应新模型。

**切入角度**：将溯源重新定义为实例检索问题（而非分类），维护一个注册数据库，新模型只需添加几张示例图，无需重新训练。用低位平面指纹替代原始 RGB 作为输入，显式捕获生成器特异噪声。

**核心 idea**：低位平面指纹 + 检索范式 = 模型无关、可扩展、少样本友好的 AI 图像溯源。

## 方法详解

### 整体框架
LIDA 分三个模块：(1) Low-Bit Fingerprint Generation——提取 RGB 图像的低位平面指纹作为输入；(2) Unsupervised Pre-Training——在 ImageNet 真实图像指纹上进行预训练，学习通用噪声结构；(3) Few-Shot Attribution Adaptation——用少量注册数据库样本微调编码器，使用 center loss 和 real-prototype contrastive loss。推理时直接用余弦相似度在注册库中检索最近邻。

### 关键设计

1. **低位平面指纹生成（Low-Bit Fingerprint Generation）**:

    - 功能：去除图像语义内容，保留生成器特异性噪声模式作为溯源线索
    - 核心思路：对 RGB 图像每个通道做位平面分解 $\mathbf{x}_c = \sum_{k=0}^{7} 2^k \cdot \mathbf{b}_c^k$，取最低 3 个位平面并做阈值化：$\tilde{\mathbf{x}}_c = 255 \cdot \text{sgn}(\sum_{k=0}^{2} 2^k \cdot \mathbf{b}_c^k)$。生成的指纹图像几乎剥离了所有语义信息，但保留了不同生成器各自的噪声指纹
    - 设计动机：PCA 可视化显示 RGB 空间中不同生成器的图像混在一起，但低位平面指纹空间中同一生成器的图像明显聚类，且真假图像清晰分离

2. **无监督预训练**:

    - 功能：在大规模真实图像上学习指纹空间的通用表征
    - 核心思路：用 ImageNet 的低位平面指纹训练修改版 ResNet-50（去除低层下采样以保留空间信息），以 ImageNet 分类作为 pretext task，损失为标准交叉熵 $\mathcal{L}_P = -\sum_{b=1}^{B} \sum_{c=1}^{C} s_b^c \log q_b^c$
    - 设计动机：无监督预训练提供鲁棒的权重初始化，使模型学习可迁移的噪声结构特征，增强对未见生成器的泛化能力

3. **少样本溯源自适应**:

    - 功能：使用最少量（1-10张/生成器）的注册样本快速适应新生成器
    - 核心思路：不使用交叉熵（会破坏预训练特征空间结构），而使用 center loss 作为溯源损失 $\mathcal{L}_A = \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$ 促进同类聚合。检测损失使用 real-prototype contrastive loss $\mathcal{L}_D$ 将真实图像拉向真实原型、推开生成图像。最终损失 $\mathcal{L} = \mathcal{L}_A + \lambda \mathcal{L}_D$，$\lambda = 0.9$。推理采用两阶段：先检测真假，再检索归因
    - 设计动机：center loss 作为正则化保持预训练特征空间的结构，避免少样本微调导致特征漂移；contrastive loss 增强真假分离

### 损失函数 / 训练策略
检测损失 $\mathcal{L}_D$ 基于 real-prototype contrastive loss：用预训练阶段所有 ImageNet 图像的平均特征作为真实类原型 $p_r$，通过 sigmoid + cosine similarity 拉近真实图像、推远生成图像，温度参数 $\tau$ 控制分布锐利度。微调仅用 batch size 32、lr $1 \times 10^{-4}$ 训练 100 epochs。

## 实验关键数据

### 主实验
GenImage 数据集跨架构溯源（8类生成器，Rank-1 / mAP）：

| Shot | 方法 | Avg Rank-1 | Avg mAP |
|------|------|-----------|---------|
| 1-shot | ResNet | 17.4 | 37.5 |
| 1-shot | DIRE | 14.3 | 34.8 |
| 1-shot | ESSP | 17.0 | 36.0 |
| 1-shot | **LIDA** | **40.4** | **61.5** |
| 10-shot | ResNet | 21.4 | 22.4 |
| 10-shot | DIRE | 17.2 | 28.8 |
| 10-shot | ESSP | 22.4 | 23.0 |
| 10-shot | **LIDA** | **54.0** | **51.6** |

零样本 Deepfake 检测（GenImage，Accuracy）：

| 方法 | BigGAN | Mid | WuK | SDv4 | SDv5 | ADM | GLIDE | VQ | Avg |
|------|--------|-----|-----|------|------|-----|-------|-----|-----|
| RIGID | 53.0 | 94.1 | 87.8 | 87.0 | 87.2 | 51.4 | 45.9 | 52.2 | 69.8 |
| FSD | 62.1 | 75.1 | 88.0 | 88.0 | 88.0 | 74.1 | 93.9 | 69.1 | 77.1 |
| **LIDA** | **91.0** | 85.9 | 86.2 | 86.3 | 86.8 | **85.5** | 83.9 | **84.5** | **86.3** |

### 消融实验

| BF | $\mathcal{L}_P$ | $\mathcal{L}_A$ | $\mathcal{L}_D$ | Avg mAP 变化 |
|-----|------|------|------|-----------|
| ✗ | ✗ | ✗ | ✗ | baseline（RGB+ImageNet） |
| ✓ | ✗ | ✗ | ✗ | +10.6% |
| ✓ | ✓ | ✗ | ✗ | +12.1%（额外 +1.5%）|
| ✓ | ✓ | ✓ | ✗ | +15.8%（额外 +3.7%）|
| ✓ | ✓ | ✓ | ✓ | +24.0%（额外 +8.2%）|

替换损失函数对比：center loss + contrastive loss 组合比两者均替换为交叉熵高 3.9% mAP。

### 关键发现
- 低位平面指纹是溯源效果的最大贡献因子（+10.6% mAP）
- 检测损失 $\mathcal{L}_D$ 的贡献最大（+8.2%），有效拉开真假图像特征间距
- 在 BigGAN 上零样本即达 91% 准确率，验证了位平面指纹对 GAN 的极强辨别力
- JPEG 压缩下鲁棒性极强；即使高斯模糊破坏了低位平面分布，指纹特征仍优于 RGB

## 亮点与洞察
- "溯源即检索"的范式转换简洁优雅——新模型只需往注册库加几张图，零重训练开销
- 低位平面的物理直觉极好：高位承载语义，低位承载噪声/伪影，按位剥离恰好隔离了生成器指纹
- center loss 替代交叉熵的设计关键——保持预训练特征空间结构在少样本场景下的稳定性
- 方法极其轻量：ResNet-50 骨干 + 毫秒级推理

## 局限与展望
- 对高斯模糊的鲁棒性有限（直接破坏低位平面分布）
- 单一阈值的零样本检测在实际部署中可能不够灵活
- 仅在 GenImage 和 WildFake 上测试，缺乏最新视频生成模型的评估
- 位平面方法对图像后处理（如社交媒体压缩链路）的鲁棒性还需进一步验证

## 相关工作与启发
- 生成式水印（Tree-Ring、Gaussian Shading）需修改模型，本文完全模型无关
- 闭集方法 RepMix 仅限已知 GAN，本文天然支持开集
- 检索范式可扩展至视频生成器溯源，只需将 2D 指纹扩展到时序维度
- center loss 在少样本微调场景下的特征空间保持技巧值得借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ 将溯源重定义为检索是清晰的范式创新，低位平面指纹简洁有效
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多 shot 设置、检测+溯源+消融全面覆盖
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法叙述流畅
- 价值: ⭐⭐⭐⭐ 为生成图像溯源提供了高度实用的新范式，尤其适合快速演化的生成模型生态
# Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**会议**: CVPR 2026  
**arXiv**: [2603.10583](https://arxiv.org/abs/2603.10583)  
**代码**: [https://github.com/hongsong-wang/LIDA](https://github.com/hongsong-wang/LIDA)  
**领域**: 图像取证 / AI生成图像归属  
**关键词**: deepfake attribution, image retrieval, bit-plane, model-agnostic, few-shot

## 一句话总结
将 AI 生成图像的归属问题从分类范式重新定义为实例检索范式，提出基于低位平面指纹的模型无关框架 LIDA，通过无监督预训练和少样本归属适应，在零样本和少样本设置下实现 SOTA 的 Deepfake 检测和图像归属性能。

## 研究背景与动机
**领域现状**：随着 AIGC 技术快速发展，合成图像越来越逼真，检测和归属 AI 生成图像已成为关键的安全研究方向。现有方法分两类——生成式图像水印（需访问生成模型）和 AI 生成图像归属（独立于生成过程）。

**现有痛点**：现有归属方法将问题视为分类任务，存在三个核心缺陷：(1) 依赖模型——需要访问生成模型本身；(2) 缺乏通用性——对新的、未见过的生成器难以扩展；(3) 闭集假设——训练时需要所有生成器已知，open-set 场景表现差。

**核心矛盾**：AI 图像生成器快速迭代演化，而归属系统需要频繁重新训练才能适应新生成器，这种"训练-部署-重训"循环严重限制实用性。

**本文目标**：设计一个模型无关的归属框架，不需要访问任何生成模型，不需要对新生成器重训练，仅需几张样本就能将新生成器纳入归属系统。

**切入角度**：将归属问题从分类重新定义为实例检索——训练一个通用的特征编码器，对查询图像在注册数据库中检索最相似的图像来确定来源。

**核心 idea**：利用图像低位平面作为生成指纹输入，通过无监督预训练学习噪声结构表示，再用少量样本进行归属适应，实现基于检索的开放集归属。

## 方法详解

### 整体框架
LIDA 的 pipeline 包含三个模块：(1) 低位指纹生成——从 RGB 图像提取低位平面作为生成指纹；(2) 无监督预训练——在大规模真实图像上预训练归属编码器；(3) 少样本归属适应——用少量 AI 生成图像微调编码器。推理时将查询图像编码后在注册数据库中检索最近邻进行归属。

### 关键设计

1. **低位指纹生成 (Low-Bit Fingerprint Generation)**:

    - 功能：从图像中提取保留生成器特有痕迹、去除视觉内容的指纹表示
    - 核心思路：对 RGB 图像的每个通道进行位平面分解 $\mathbf{x}_c = \sum_{k=0}^{7} 2^k \cdot \mathbf{b}_c^k$，取最低 3 个位平面并二值化：$\tilde{\mathbf{x}}_c = 255 \cdot \text{sgn}(\sum_{k=0}^{2} 2^k \cdot \mathbf{b}_c^k)$。低位平面包含生成器的固有伪影（generative fingerprint），不同生成器的低位指纹具有显著差异
    - 设计动机：原始 RGB 图像中真实和 AI 生成图像在特征空间中混合无法区分，但低位指纹空间中二者清晰分离且同一生成器的图像聚类

2. **无监督预训练 (Unsupervised Pre-Training)**:

    - 功能：在大规模真实图像数据集上学习通用的噪声结构表示，增强泛化能力
    - 核心思路：采用 ResNet-50（移除低层下采样以保留空间信息）作为编码器，在 ImageNet 指纹图像上以图像分类为 pretext task 训练。预训练损失 $\mathcal{L}_P = -\sum_{b=1}^{B} \sum_{c=1}^{C} s_b^c \log q_b^c$
    - 设计动机：预训练提供鲁棒的权重初始化，使微调时收敛更快、性能更好；且在真实图像上预训练使模型学到可迁移到取证下游任务的固有噪声结构

3. **少样本归属适应 (Few-Shot Attribution Adaptation)**:

    - 功能：用每个生成器仅几张图像微调编码器，使其能区分不同生成器
    - 核心思路：采用两阶段归属范式（先检测真假，再归属到具体生成器），使用两个互补损失：(a) 中心损失 $\mathcal{L}_A = \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$ 鼓励同类聚拢；(b) 真实原型对比损失 $\mathcal{L}_D$ 将真实图像拉近真实原型、推远 AI 图像。总损失 $\mathcal{L} = \mathcal{L}_A + \lambda \mathcal{L}_D$。刻意避免使用交叉熵损失以保护预训练特征空间结构
    - 设计动机：交叉熵损失可能破坏预训练学到的特征表示结构；中心损失作为正则化鼓励类内紧凑性，约束特征漂移

### 损失函数 / 训练策略
- 预训练阶段：ImageNet 分类交叉熵（作为 pretext task）
- 微调阶段：中心损失 $\mathcal{L}_A$（归属类内聚拢）+ 真实原型对比损失 $\mathcal{L}_D$（真假检测）
- 中心更新：$c_j^{t+1} = c_j^t - \alpha \cdot \frac{\sum_{i=1}^{m} \delta(y_i = j) \cdot (c_j^t - x_i)}{1 + \sum_{i=1}^{m} \delta(y_i = j)}$

## 实验关键数据

### 主实验
在 GenImage 数据集上 1-shot 和 5-shot 归属结果 (Rank-1 / mAP %):

| 方法 | 1-shot Rank-1 | 1-shot mAP | 5-shot Rank-1 | 5-shot mAP |
|------|-------------|-----------|-------------|-----------|
| ResNet | 17.4 | 37.5 | 19.4 | 25.0 |
| DIRE | 14.3 | 34.8 | 18.7 | 24.8 |
| ESSP | 17.0 | 36.0 | 17.5 | 23.7 |
| **LIDA (Ours)** | **40.4** | **61.5** | **76.9** | **54.5** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| RGB 输入 | 特征空间中真假混合 | PCA 可视化无法区分不同生成器 |
| 低位指纹输入 | 特征空间中清晰聚类 | 真实和 AI 图像分离，同生成器图像聚拢 |
| 无预训练直接微调 | 性能显著下降 | 缺乏通用噪声结构表示 |
| 交叉熵替代中心损失 | 性能下降 | 破坏预训练特征空间结构 |

### 关键发现
- 低位平面指纹是区分生成器的关键——不同生成器在低位产生显著不同的噪声模式
- 检索范式天然支持开放集——新增生成器只需添加几张样本到数据库，无需重训练模型
- 1-shot 设置下 LIDA 的 mAP 已比 ResNet baseline 高出 24 个点
- 5-shot 设置下性能大幅跃升，Rank-1 从 40.4% 升至 76.9%

## 亮点与洞察
- 范式创新：将归属从分类转为检索，彻底解决了对新生成器的扩展性问题
- 低位平面作为指纹输入简单高效——几行代码的位操作就能提取
- 避免使用交叉熵的设计选择很精妙——保护预训练特征空间结构对少样本学习至关重要
- 提供"证据型归属"——检索到的相似图像本身就是归属决策的证据

## 局限与展望
- 低位平面对 JPEG 压缩等后处理操作的鲁棒性需要进一步验证
- 注册数据库的规模和质量直接影响归属准确率
- 当前仅在图像级别归属，未扩展到视频生成器
- 对高度相似的同族生成器（如 SD v1.4 vs v1.5）区分能力可能有限

## 相关工作与启发
- **Yu et al. (GAN fingerprint)**：首次系统研究 GAN 指纹，但限于闭集分类
- **Tree-Ring/Gaussian Shading**：生成式水印方法，需访问生成模型
- **DIRE**：使用扩散重建差异检测，但归属能力弱
- 启发：其他取证任务（如 deepfake 视频检测、AI 文本检测）也可探索"检测作为检索"范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 将归属建模为检索是很好的范式转变，低位指纹作为输入简洁有效
- 实验充分度: ⭐⭐⭐⭐ GenImage 和 WildFake 两个大规模数据集，零样本/少样本多设置评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 实用性强，模型无关+少样本适应解决了真实场景的关键需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](../../ICML2026/image_generation/barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [\[CVPR 2026\] Diffusion Probe: Generated Image Result Prediction Using CNN Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](../../NeurIPS2025/image_generation/fast_data_attribution_for_text-to-image_models.md)

</div>

<!-- RELATED:END -->
