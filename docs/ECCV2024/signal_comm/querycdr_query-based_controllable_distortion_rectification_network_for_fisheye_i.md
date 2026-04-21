---
title: >-
  [论文解读] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images
description: >-
  [ECCV 2024][鱼眼图像矫正] 提出QueryCDR网络，通过可学习查询机制（DLQM）和两种可控调制模块（CCMB/CAMB），首次实现不同畸变程度的鱼眼图像在**不重训**的情况下进行高质量可控矫正。
tags:
  - ECCV 2024
  - 鱼眼图像矫正
  - 可控畸变校正
  - 可学习查询
  - Transformer
  - 生成式矫正
---

# QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images

**会议**: ECCV 2024  
**arXiv**: [2412.13496](https://arxiv.org/abs/2412.13496)  
**代码**: https://github.com/PbGuo/QueryCDR (有)  
**领域**: 信号通信（鱼眼图像矫正/低层视觉）  
**关键词**: 鱼眼图像矫正, 可控畸变校正, 可学习查询, CNN-Transformer混合, 生成式矫正

## 一句话总结

提出QueryCDR网络，通过可学习查询机制（DLQM）和两种可控调制模块（CCMB/CAMB），首次实现不同畸变程度的鱼眼图像在**不重训**的情况下进行高质量可控矫正。

## 研究背景与动机

鱼眼相机因其超大视场角（FoV）被广泛应用于安防监控和自动驾驶领域。然而鱼眼镜头带来的图像畸变严重影响下游视觉任务的性能。现有鱼眼矫正方法分为两大类：

**回归式方法**：用网络预测畸变参数后重建图像，需要额外标注且非端到端

**生成式方法**：编码器-解码器直接生成矫正图像，端到端但泛化性差

**核心痛点**：现有方法（包括DR-GAN、PCN、SimFIR等）都只能在与训练数据相似的畸变程度上取得好效果。**当畸变程度改变而不重训时，矫正质量显著下降**。这是因为模型在训练中学习的是固定的位置映射关系，无法泛化到新的畸变分布。而鱼眼图像获取困难，为每种畸变程度重新收集数据集并重训模型代价极高。

**为何不能直接套用图像修复领域的可控机制？**

存在两个关键差距：
- **优化目标差异**：修复任务学习像素级细节恢复，而鱼眼矫正学习空间级位置映射关系。缺乏位置信息的修复控制机制无法有效控制矫正
- **控制条件差异**：鱼眼畸变从中心向边缘递增，是空间变化的。修复任务中的单标量控制条件（如CFSNet、MM-RealSR）无法处理这种空间变化的畸变

**切入角度**：设计包含空间位置信息的高维可学习查询作为控制条件，替代标量控制；同时设计专门的调制模块，在CNN和Transformer两个尺度上利用控制条件引导矫正。

**核心Idea**：将不同畸变程度的隐式空间映射关系投射到一组可学习查询的低维潜空间中，用户只需选择不同查询即可控制矫正输出；查询间插值还能实现连续平滑的畸变矫正。

## 方法详解

### 整体框架

QueryCDR由三个核心组件构成：(1) 流估计模块（沿用PCN）进行粗粒度矫正(warping)；(2) DLQM从用户给定的查询中提取控制条件并逐层输入矫正网络；(3) U型层次矫正网络由CCMB和CAMB组成，在控制条件指导下生成矫正图像。

### 关键设计

1. **Distortion-aware Learnable Query Mechanism (DLQM)**：

   功能：维护一组代表不同畸变程度的可学习查询集合，提取位置相关的控制条件。

   核心思路：构建查询集 $\mathbf{Q}_s = \{Q_i \mid Q_i \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in}}, i=1,...,N\}$，每个查询大小与输入图像相同，编码了对应畸变程度的空间位置映射关系。先用3层3×3卷积的控制提取器CE提取特征：

   $$Q_{ex} = \text{CE}(Q_i)$$

   再通过多层全连接层逐层生成控制条件：

   $$Q_c^l = \text{FC}_2^l(\text{FC}_1^l(Q_c^{l-1}))$$

   每层的输出 $Q_c^l$ 同时作为该层矫正模块的控制条件和下一层DLQM的输入。

   **关键特性**：查询间可以插值实现连续矫正，如 $Q_{1.25} = 0.75 Q_1 + 0.25 Q_2$。

   设计动机：与标量控制条件相比，和输入同尺寸的查询自然包含空间位置信息，能够表达从中心到边缘递增的畸变分布。

2. **Controllable Convolution Modulating Block (CCMB)**：

   功能：在CNN层面实现动态特征调制，保留局部纹理细节。

   核心思路：接收输入特征 $F_{in}$ 和控制条件 $Q_c$，先计算控制特征 $F_c = F_{in} \otimes Q_c$（逐元素乘），然后用系数预测器预测动态融合比率：

   $$\theta = \text{CP}(F_{in}, Q_c)$$
   $$F_{out} = \theta F_c \oplus (1-\theta) F_{in}$$

   设计动机：直接使用控制特征或固定比率融合会降低矫正质量。动态融合比率让模型自动在保留原始特征和应用控制特征之间找到最优平衡。

3. **Controllable Attention Modulating Block (CAMB)**：

   功能：利用Transformer的全局注意力机制捕获长程畸变映射关系。

   核心思路：设计控制注意力机制(CTRL-ATTN)，将控制特征 $F_c$ 投影为Query $\mathcal{Q}$，输入特征 $F_{in}$ 投影为Key $\mathcal{K}$ 和Value $\mathcal{V}$：

   $$\text{CTRL-ATTN}(\mathcal{Q},\mathcal{K},\mathcal{V}) = \text{softmax}\left(\frac{\mathcal{Q}\mathcal{K}^T}{\sqrt{m}}\right)\mathcal{V}$$

   最终通过残差连接和FFN输出：
   $$F_a = \text{CTRL-ATTN}(\text{LN}(\mathcal{Q},\mathcal{K},\mathcal{V})) \oplus F_{in}$$
   $$F_{out} = \text{Conv}_{1\times 1}(\text{FFN}(\text{LN}(F_a)) \oplus F_a)$$

   设计动机：CNN难以捕获鱼眼图像中连续、非定形的长程畸变模式。CAMB通过全局注意力感知控制条件中的空间映射关系，确保矫正的全局一致性。

**混合架构**：在特征图较大的前3层编码+后3层解码（$l=\{1,2,3,9,10,11\}$）使用CCMB保留纹理，在特征图较小的中间层（$l=\{4,5,6,7,8\}$）使用CAMB捕获全局依赖，实现6C+5A的最优配置。

### 损失函数 / 训练策略

**两阶段训练**：

- **粗粒度预训练**：仅在单一畸变程度 $d$ 上用一个查询 $Q$ 训练，损失为重建L1损失+多尺度损失：
  $$\mathcal{L}_{pre} = \|I_{out}^d - I_{gt}^d\|_1 + \sum_{j=1}^{Z-1}\|S(I_{gt}^d, j) - C(F_{out}^j)\|_1$$

- **细粒度微调**：将预训练查询权重复制到所有9个查询，然后在9种畸变程度数据上微调：
  $$\mathcal{L}_{fine}^{d_i} = \mathcal{L}_r^{d_i} + \mathcal{L}_m^{d_i}$$

预训练用40000张图像，微调仅用18000张（每种畸变2000张），测试3600张。输入256×256，Adam优化器，学习率1e-4。

## 实验关键数据

### 主实验

**COCO鱼眼数据集上9种畸变程度的PSNR(dB)比较：**

| 方法 | d1 | d5 | d9 | 平均PSNR | 平均SSIM |
|------|-----|-----|-----|---------|---------|
| SC (传统) | 10.05 | 11.50 | 9.14 | 10.73 | 0.151 |
| DR-GAN | 15.68 | 18.50 | 17.47 | 17.74 | 0.323 |
| PCN | 14.93 | 18.86 | 18.26 | 17.97 | 0.575 |
| DDA | 16.39 | 20.12 | 18.22 | 18.33 | 0.591 |
| SimFIR | 16.57 | 19.31 | 18.48 | 18.53 | 0.601 |
| **QueryCDR** | **20.01** | **20.72** | **20.53** | **20.32** | **0.676** |

QueryCDR平均PSNR超越之前最优SimFIR **1.79dB**，SSIM提升 **0.075**。即使在现有方法表现最好的d5上，仍超越DDA 0.60dB。

### 消融实验

**控制条件方式对比（PSNR）：**

| 控制方式 | d1 | d5 | d9 | 平均 | 说明 |
|---------|-----|-----|-----|------|------|
| 无控制 (PCN) | 14.93 | 18.86 | 18.26 | 17.97 | 基线 |
| 标量控制 | 19.52 | 19.89 | 20.15 | 19.78 | +1.81dB |
| 固定查询 | 20.13 | 20.16 | 19.84 | 19.93 | +1.96dB |
| 可学习查询(Ours) | 20.01 | 20.72 | 20.53 | **20.32** | **+2.35dB** |

**调制方式对比：**

| 调制方式 | PSNR | SSIM | 说明 |
|---------|------|------|------|
| 直接使用 $F_c$ | 20.14 | 0.655 | 基线 |
| 固定1:1比率融合 | 20.17 | 0.658 | +0.03dB |
| 动态机制(CCMB) | 20.20 | 0.669 | +0.06dB |
| 注意力机制(CAMB) | 20.26 | 0.671 | +0.12dB |

**网络架构对比：**

| 架构 | FLOPs(G) | 参数(M) | PSNR | SSIM |
|------|----------|---------|------|------|
| PCN | 12.305 | 35.637 | 17.97 | 0.575 |
| 11C+0A | 12.736 | 37.701 | 20.20 | 0.669 |
| 6C+5A (Ours) | 12.353 | 43.244 | 20.32 | 0.676 |
| 0C+11A | 15.190 | 51.795 | 20.26 | 0.671 |

### 关键发现

1. **可控机制至关重要**：无控制(17.97)→标量控制(19.78)→可学习查询(20.32)，三级递进提升验证了控制条件维度越高效果越好
2. **CCMB+CAMB混合架构最优**：6C+5A在性能(20.32/0.676)和计算效率(12.353G FLOPs)间取得最佳平衡
3. **仅需少量微调数据**：预训练40K+微调18K即可泛化到9种畸变程度，避免了为每种畸变重训
4. **查询插值实现连续矫正**：不在训练集中的畸变程度可通过查询线性插值覆盖，如 $Q_{1.25} = 0.75Q_1 + 0.25Q_2$
5. **在真实鱼眼图像上也有效**：尽管在合成数据上训练，QueryCDR在真实鱼眼数据集上仍展现出鲁棒的矫正能力

## 亮点与洞察

- **可学习查询作为控制条件的思路新颖**：不同于标量或one-hot编码，全分辨率的查询天然编码了空间变化的畸变分布信息
- **查询插值实现连续控制是优雅的设计**：类似StyleGAN中的潜空间插值，查询空间也具有良好的平滑性
- **两阶段训练策略务实**：先用单畸变度数据预训练（容易获取），再用少量多畸变度数据微调（昂贵但少量即可），显著降低数据需求
- **CCMB的动态融合比率预测**：自适应地在保留原始特征和应用控制特征之间权衡，比固定比率更灵活

## 局限与展望

- 当前控制需要用户手动选择查询索引，论文自己也提到未来应实现**自动控制机制**，自动估计输入图像的畸变程度
- 仅在合成鱼眼数据集上做定量评估（COCO/Places2），真实数据仅有定性结果
- 查询数量固定为9个，对应9种离散畸变程度，虽可插值但精度可能受限
- 未与最新的扩散模型based矫正方法对比
- 没有评估矫正后图像对下游任务（如目标检测、语义分割）的性能提升

## 相关工作与启发

- PCN [Yang et al.] 提出的流估计模块被本文直接沿用，QueryCDR在其基础上增加可控机制
- CFSNet [Wang et al.] 的标量可控修复思路被推广到高维查询空间
- CCMB中的动态融合思想与注意力机制中的门控机制类似，可迁移到其他可控生成任务
- CAMB中将控制特征作为Query、原始特征作为Key/Value的设计，对可控图像编辑也有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 可学习查询控制畸变矫正的思路新颖，查询插值实现连续控制巧妙
- 实验充分度: ⭐⭐⭐⭐ 9种畸变程度全面评估+详细消融，但缺少真实数据定量评估
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 解决了鱼眼矫正的泛化性痛点，对实际应用有价值

<!-- RELATED:START -->

## 相关论文

- [RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)
- [Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)
- [Unsupervised Exposure Correction](unsupervised_exposure_correction.md)

<!-- RELATED:END -->
