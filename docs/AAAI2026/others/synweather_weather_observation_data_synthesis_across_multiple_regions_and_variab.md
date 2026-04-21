---
title: >-
  [论文解读] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer
description: >-
  [AAAI 2026][气象数据合成] 构建了首个支持统一多区域多变量的气象观测数据合成数据集SynWeather（覆盖4个区域×4种变量×6颗卫星），并提出基于Diffusion Transformer的通用概率生成模型SynWeatherDiff，通过文本提示引导在多个合成任务上超越专用模型和现有通用模型。
tags:
  - AAAI 2026
  - 气象数据合成
  - Transformer
  - 多区域多变量
  - 雷达反射率
  - 降水估计
---

# SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer

**会议**: AAAI 2026  
**arXiv**: [2511.08291](https://arxiv.org/abs/2511.08291)  
**代码**: [https://github.com/Dtdtxuky/SynWeather](https://github.com/Dtdtxuky/SynWeather)  
**领域**: 气象数据合成 / 扩散模型  
**关键词**: 气象数据合成, 扩散Transformer, 多区域多变量, 雷达反射率, 降水估计

## 一句话总结

构建了首个支持统一多区域多变量的气象观测数据合成数据集SynWeather（覆盖4个区域×4种变量×6颗卫星），并提出基于Diffusion Transformer的通用概率生成模型SynWeatherDiff，通过文本提示引导在多个合成任务上超越专用模型和现有通用模型。

## 研究背景与动机

**领域现状**：随着地球静止卫星（GOES、Himawari、Meteosat）和雷达系统的发展，大量气象观测数据可用于天气预报、灾害监测和气候研究。但由于仪器固有限制——雷达在地形复杂或经济欠发达区域覆盖稀疏、可见光卫星夜间不可用、极轨卫星时间分辨率低——原始数据存在时间和空间缺口。因此需要数据合成技术来填补缺失信息。

**现有痛点**：（1）现有数据集局限于单区域或单变量：如HKO-7仅覆盖香港雷达、SEVIR仅覆盖美国本土、DigitalTyphoon仅用一个Himawari通道；（2）现有方法为每种变量合成任务定制专用网络（如SRViT合成VIL、Deep-STEP估计降水），在变量和区域之间无法统一建模；（3）确定性模型（如用MSE训练的UNet）输出过度平滑，无法捕获高强度区域（如强对流降水中心）的细粒度结构。

**核心矛盾**：不同区域的气象变量之间存在物理关联性（如降水与雷达反射率的Z-R关系、可见光与微波亮温的空间对应性），但现有方法独立建模忽略了跨变量/跨区域的互补信息。同时，确定性建模与气象事件的随机性本质矛盾。

**本文目标** 如何进行多区域多变量的通用概率气象数据合成？具体子问题：（1）如何构建统一的多区域多变量数据集？（2）如何设计一个通用模型同时处理不同卫星源、不同变量的合成？（3）如何生成具有细粒度高值区域的概率性结果？

**切入角度**：借鉴自然图像生成中的通用模型思路，利用文本提示区分不同的区域-变量任务组合，用Diffusion Transformer实现概率性生成以解决确定性模型的过度平滑问题。

**核心 idea**：构建统一数据集+文本提示驱动的扩散Transformer，在概率框架下实现多区域多变量气象合成的通用建模。

## 方法详解

### 整体框架

SynWeatherDiff由三部分组成：（1）通用自编码器：将不同气象变量压缩到共享潜在空间；（2）ViT编码器：提取卫星红外观测特征；（3）文本引导Diffusion Transformer：在潜在空间中以文本提示和卫星特征为条件进行去噪生成。输入是10通道红外卫星观测（C07-C16），输出是目标气象变量（雷达反射率/降水/可见光/微波亮温）。文本提示格式为"Synthesize the [变量] variable over the [区域] region using corresponding satellite imagery."

### 关键设计

1. **SynWeather统一数据集**:

    - 功能：提供首个标准化的多区域多变量气象数据合成基准
    - 核心思路：覆盖4个区域（美国本土CONUS、欧洲、东亚、热带气旋区域）和4种气象变量（雷达反射率CR、降水、可见光、微波亮温MWBT），整合6颗地球静止卫星（GOES-16/17/18、Meteosat-11、Himawari-8/9）的全部红外通道（10通道）作为输入。数据统一到1小时时间分辨率和4公里空间分辨率，裁切为256×256 patches（128步长滑动窗口），通过连通区域面积阈值过滤无效patches。降水做log变换（长尾分布），其他做min-max归一化
    - 设计动机：使用所有10个红外通道而非传统的3通道，因为不同光谱范围（短波红外SWIR、水汽WV、长波红外LWIR、气体吸收GAS）对不同变量合成有互补贡献——消融实验证实移除任何一组通道都会性能下降

2. **通用自编码器（General AutoEncoder）**:

    - 功能：将不同气象变量压缩到统一的低维潜在空间
    - 核心思路：目标变量 $Y_{r,b} \in \mathbb{R}^{1 \times H \times W}$ 编码为潜在表示 $z_{r,b} \in \mathbb{R}^{C_z \times H_z \times W_z}$，训练使用像素重建损失+KL散度+对抗损失。所有气象变量共享同一个自编码器
    - 设计动机：气象图像包含大量冗余（几百公里范围内同时出现降水或台风很少见），压缩后的潜在空间去除了冗余同时保留了物理信息。不同变量存在物理相似性（如降水和CR的Z-R关系），共享潜在空间有助于跨变量知识迁移

3. **文本引导Diffusion Transformer（Text-Guided DiT）**:

    - 功能：在潜在空间中以卫星输入和文本提示为条件进行概率性去噪生成
    - 核心思路：卫星观测 $X_r$ 通过ViT编码器提取特征，文本提示 $P_{r,b}$ 通过CLIP文本编码器（仅微调最后一层Transformer block）编码。采用早期融合策略：将噪声潜在 $z^t_{r,b}$ 与卫星编码器特征拼接后patchify，再与文本token拼接，通过DiT的自注意力层进行条件去噪。训练目标为标准噪声预测损失 $\mathcal{L} = \mathbb{E}_{z,\epsilon,t}[\|\epsilon_\theta(z^t_{r,b}, t, X_{r,b}, P_{r,b}) - \epsilon\|^2_2]$
    - 设计动机：相比SD3的交叉注意力融合，早期融合策略让卫星特征更直接地参与去噪过程。文本提示提供了灵活的任务控制接口——不需要修改模型结构即可扩展到新的区域-变量组合。DiT的概率框架天然避免了确定性模型的过度平滑问题

### 损失函数 / 训练策略

自编码器训练使用像素重建+KL散度+对抗损失。DiT训练使用标准的噪声预测MSE损失，AdamW优化器，余弦学习率衰减（5e-4→1e-5），600K步，batch size 16，4×A100 GPU。6个标准任务均匀采样（各1/6）训练通用模型；消融实验探索了不同任务采样比例。

## 实验关键数据

### 主实验

| 任务 | 指标 | SynWeatherDiff(通用) | WeatherGFM(通用) | UNet(专用) | ViT(专用) |
|------|------|-------------------|----------------|----------|---------|
| CONUS CR | RMSE↓ | **2.820** | 3.124 | 3.395 | 3.487 |
| CONUS CR | CSI/25↑ | **0.382** | 0.366 | 0.299 | 0.309 |
| CONUS降水 | CSI/2↑ | **0.312** | 0.288 | 0.231 | 0.250 |
| CONUS降水 | CSI/15↑ | **0.113** | 0.090 | 0.059 | 0.038 |
| 欧洲降水 | CSI/5↑ | **0.079** | 0.013 | 0.016 | 0.044 |
| 东亚可见光 | SSIM↑ | 0.868 | 0.822 | **0.917** | 0.870 |
| MWBT | LPIPS↓ | **0.254** | 0.325 | 0.329 | 0.324 |

### 消融实验

| 采样策略 | CONUS降水CSI/2 | CONUS CR CSI/25 | 欧洲可见光SSIM | MWBT SSIM |
|---------|--------------|---------------|-------------|----------|
| 均匀(1/6) | 0.312 | 0.382 | 0.864 | 0.837 |
| CR为主(1/2) | 0.320 | **0.403** | 0.857 | 0.843 |
| 降水为主(1/2) | 0.292 | 0.343 | 0.855 | 0.841 |
| 可见光为主(1/2) | 0.298 | 0.377 | **0.877** | 0.842 |
| MWBT为主(1/2) | 0.295 | 0.374 | 0.879 | 0.842 |

### 关键发现

- **通用模型超越专用模型**：SynWeatherDiff在降水合成（最具挑战性的任务）上全面超越所有专用模型，CSI高值阈值(CSI/15)取得0.113 vs UNet的0.059，提升91%
- **可见光合成是例外**：UNet在像素空间直接操作，可见光包含大量高频细节，通过自编码器会损失信息。这指向了自编码器改进的方向
- **任务间存在互补和冲突**：CR采样比例提升有助于降水（Z-R物理关系）和MWBT（相似物理域）；可见光和MWBT互相受益（强对流系统的空间对应）；但CR和可见光存在冲突
- **OOD泛化能力**：在从未见过的"东亚降水"任务上，通用模型（文本提示"东亚+降水"）超越在CONUS降水上训练的专用模型，证明跨区域知识迁移的有效性
- **输入通道消融**：10通道全用优于3通道。水汽和长波红外通道对降水/CR合成至关重要，短波红外对可见光合成更重要。移除任何通道组都导致性能下降
- **细粒度生成优势明显**：可视化显示SynWeatherDiff能恢复分散的小型气象单元（weather cells）的数量和分布，而UNet/ViT/WeatherGFM常将多个小单元合并为一大块或丢失强度中心

## 亮点与洞察

- **数据集贡献可能比模型更持久**：SynWeather是首个覆盖4区域×4变量×6卫星源的统一气象合成数据集，定义了7个标准任务（含1个OOD任务）和完整的评估协议。这种基础设施级别的贡献对推动领域发展价值很大
- **文本提示实现灵活任务控制**：不同于每个任务训练一个模型，文本提示让同一模型通过改变文字即可切换任务。这种"一个模型多个任务"的理念在气象领域的落地很有前瞻性
- **概率模型在高值区域的优势**：确定性模型倾向回归均值，导致极端天气事件的强度中心被平滑。SynWeatherDiff的概率生成能恢复CSI高阈值性能（如CSI/15），这对灾害性天气预警至关重要

## 局限与展望

- 可见光合成不如UNet，自编码器在高频细节重建上存在瓶颈，需要改进自编码器或引入skip connections
- 推理受限于训练时暴露的区域和变量组合——文本提示框架虽灵活但无法泛化到完全未见过的区域-变量对
- 数据不平衡：可见光样本最多（~500K），降水和MWBT样本较少（~10-20K），可能导致训练偏向
- 仅使用4km空间分辨率，更高分辨率（如1km）的合成尚未探索
- 缺乏与气象预报系统（如数据同化）的集成验证

## 相关工作与启发

- **vs WeatherGFM (zhao2024weathergfm)**：WeatherGFM也是通用气象模型，但基于确定性预测。SynWeatherDiff的概率框架在大多数任务上超越WeatherGFM，尤其在降水高值区域CSI提升显著
- **vs SRViT (stock2024srvit)**：SRViT是专用的VIL/CR合成模型。SynWeatherDiff作为通用模型在CONUS CR上以2.820 vs 3.561的RMSE大幅超越
- **vs Deep-STEP & TomoPE**：两者是专用降水估计模型。SynWeatherDiff的CSI/15（0.113）远超Deep-STEP（0.007）和TomoPE（0.036），证实了概率模型在极端值区域的优势

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多区域多变量气象合成数据集+通用扩散Transformer模型的组合，文本提示驱动的任务控制有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 6个标准任务+1个OOD任务，7种对比方法，完整的采样比例和输入通道消融
- 写作质量: ⭐⭐⭐⭐ 数据集构建描述详尽，实验分析深入，论文结构清晰
- 价值: ⭐⭐⭐⭐⭐ 数据集+baseline的组合对气象AI社区有重要推动作用，概率生成对极端天气预警有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [UniFormer: Unified and Efficient Transformer for Reasoning Across General and Custom Computing](../../NeurIPS2025/others/uniformer_unified_and_efficient_transformer_for_reasoning_across_general_and_cus.md)
- [Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [A Mind Cannot Be Smeared Across Time](a_mind_cannot_be_smeared_across_time.md)
- [Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)

<!-- RELATED:END -->
