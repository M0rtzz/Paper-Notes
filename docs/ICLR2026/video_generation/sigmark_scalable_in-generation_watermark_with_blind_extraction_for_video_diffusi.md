---
title: >-
  [论文解读] SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion
description: >-
  [ICLR 2026][视频扩散模型] SIGMark提出首个针对现代视频扩散模型的盲提取生成内水印框架，通过全局帧级伪随机编码(GF-PRC)实现常数级提取开销的盲水印，并设计分段分组排序(SGO)模块增强因果3D VAE下的时序鲁棒性，在HunyuanVideo和Wan-2.2上以512×16位容量达到90%+比特精度。
tags:
  - ICLR 2026
  - 视频扩散模型
  - 水印
  - 盲提取
  - 伪随机编码
  - 因果3D VAE
  - 可扩展性
---

# SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion

**会议**: ICLR 2026  
**arXiv**: [2603.02882](https://arxiv.org/abs/2603.02882)  
**代码**: https://github.com/JeremyZhao1998/SIGMark-release  
**领域**: 视频水印 / AI安全  
**关键词**: 视频扩散模型, 水印, 盲提取, 伪随机编码, 因果3D VAE, 时序鲁棒性

## 一句话总结
SIGMark提出首个面向现代视频扩散模型的盲水印框架，通过全局帧级伪随机编码(GF-PRC)实现恒定提取成本的可扩展盲水印，并设计分段组排序(SGO)模块应对因果3D VAE下的时序扰动，在HunyuanVideo和Wan-2.2上实现高bit精度与强鲁棒性。

## 研究背景与动机

**领域现状**：视频扩散模型（如HunyuanVideo、Wan-2.2）快速发展，隐式水印是保护AI生成视频版权和追踪有害内容的关键技术。现有方法分为后处理水印（降低视频质量）和生成中水印（理论无损但存在局限）。

**现有痛点**：现有生成中水印方法（如VideoShield、VideoMark）是**非盲方法**——提取时需要维护所有message-key对并进行模板匹配，计算代价随用户/请求数线性增长，无法扩展到大规模平台。

**核心矛盾**：现代视频扩散模型采用因果3D VAE，将一组$d_t$帧从一个时间维度的latent特征解码。时序扰动（帧删除、裁剪）会破坏帧分组，导致VAE编码产生错误的latent特征，使水印提取**时序鲁棒性极差**。

**本文目标**：(1) 如何实现恒定提取代价的盲水印？(2) 如何在因果3D VAE下保持时序鲁棒性？

**切入角度**：利用全局共享的帧级PRC密钥替代per-request密钥存储，并设计光流分割+滑动窗口检测恢复正确帧分组。

**核心 idea**：用全局帧级伪随机编码实现盲提取+分段组排序模块恢复时序信息，使水印提取成本从线性降为常数。

## 方法详解

### 整体框架
SIGMark分为嵌入和提取两部分。嵌入时，水印消息通过全局帧级PRC密钥编码到初始latent噪声中，经扩散模型生成无损水印视频。提取时，对可能被扰动的视频先用SGO模块恢复帧分组，再通过扩散反演获得latent噪声，最后用GF-PRC密钥解码消息。

### 关键设计

1. **全局帧级伪随机编码 (GF-PRC)**:

    - 功能：将水印消息编码到初始latent噪声中，实现盲提取
    - 核心思路：维护一组全局帧级PRC密钥 $K[i]$，每个密钥对应latent空间的一个时间维度。嵌入时：$\text{TP}[i] = \text{PRC.Encode}(m[i]; K[i])$，然后通过符号调制映射到噪声：$z_0(m) = (\text{TP} \times 2 - 1) \times |z_0|$
    - 设计动机：PRC编码即使对相同消息也能产生不同的随机模板bit，在全局密钥下保持噪声多样性。传统流密码（如ChaCha20）使用固定密钥材料无法做到这点。提取时仅需全局密钥，复杂度从 $O(N)$ 降为 $O(1)$

2. **分段组排序模块 (SGO)**:

    - 功能：恢复被时序扰动破坏的因果帧分组信息
    - 核心思路：两步走——(1) 光流分割：计算双向Farnebäck光流，得到运动一致的连续帧段；(2) 滑动窗口检测：在每个段内，通过PRC.Detect确定帧索引 $\hat{\text{Idx}[j]} = \text{argmax}(\text{PRC.Detect}(z_0'[j]; K[0,1,...,f_l]))$，找到连续检测结果以确定正确分组
    - 设计动机：因果3D VAE要求正确的帧分组才能产生一致的latent特征，SGO通过两阶段方法鲁棒地恢复分组信息

3. **水印提取与消息恢复**:

    - 对重组帧进行3D VAE编码和扩散反演得到 $z_0'$
    - 用PRC解码：$\hat{m[i]} = \text{PRC.Decode}(\frac{\text{Sgn}(z_0'[i])+1}{2}; K[i])$
    - 无需存储原始消息或模板匹配

### 损失函数 / 训练策略
SIGMark是**无训练方法**，直接在推理阶段嵌入水印。嵌入的噪声保持高斯分布 $z_0(m) \sim \mathcal{N}(0, 1)$，因此理论上不影响生成质量。反演使用flow-matching Euler离散反演。

## 实验关键数据

### 主实验（HunyuanVideo T2V，512 bits）

| 方法 | 类别 | Bit acc↑ | V-score↑ |
|------|------|----------|----------|
| No-mark | – | – | 0.490 |
| DCT | 后处理 | 0.889 | 0.424 |
| VideoMark | 非盲 | 0.873 | 0.507 |
| VideoShield | 非盲 | 1.000 | 0.497 |
| **SIGMark** | **盲** | **0.958** | **0.506** |

高容量模式（512×16 bits）下SIGMark达到0.885 bit acc，超越VideoMark（0.758）。

### 鲁棒性实验（HunyuanVideo I2V）

| 方法 | 无扰动 | 高斯噪声 | 压缩 | 时序drop | 时序insert |
|------|--------|----------|------|----------|-----------|
| VideoMark | 0.85 | 0.64↓0.21 | 0.63↓0.22 | 0.52↓0.19 | 0.51↓0.20 |
| VideoShield | 1.00 | 1.00↓0.00 | 0.99↓0.01 | 0.89↓0.10 | 0.84↓0.15 |
| **SIGMark** | **0.98** | **0.89↓0.09** | **0.84↓0.14** | **0.81↓0.10** | **0.87↓0.04** |

### 消融实验

| 配置 | Bit acc | 说明 |
|------|---------|------|
| Single PRC (非GF) | 0.707 | 去掉全局帧级编码 |
| GF-PRC (完整) | 0.905 | 帧级编码提升显著 |
| w/o SGO | 0.534 | 无分段排序，时序鲁棒性崩溃 |
| w/o OF-seg | 0.762 | 去掉光流分割 |
| w/o SW-det | 0.823 | 去掉滑动窗口检测 |
| SGO (完整) | 0.869 | 两个组件互补 |

### 关键发现
- 后处理水印会显著降低视频质量（V-score从0.490降至~0.42），而生成中方法几乎无损
- GF-PRC不仅实现盲提取，还通过帧间冗余提供额外纠错能力
- SGO模块的两个组件（光流分割和滑动窗口检测）都不可或缺
- 提取时间：VideoShield随视频数量线性增长，SIGMark恒定不变

## 亮点与洞察
- **从非盲到盲的范式转换**：首次在视频扩散水印中实现盲提取，对大规模平台部署至关重要。核心洞察是PRC编码的伪随机性可以兼容全局密钥
- **时序鲁棒性的系统解决**：SGO模块针对因果3D VAE的特性设计，是通用的帧分组恢复方案，可迁移到其他需要正确帧分组的任务
- **无训练即插即用**：整个框架不需要微调任何模型参数，可直接应用于任意视频扩散模型

## 局限与展望
- 未达到100%的bit精度（受PRC编码容错特性与扩散反演精度的制约）
- 空间扰动下的精度下降（如高斯噪声下0.89）相比VideoShield（1.00）仍有差距
- SGO模块引入额外的光流计算开销
- 可以探索与后处理水印方法的混合策略，进一步增强鲁棒性

## 相关工作与启发
- **vs VideoShield**: 非盲方法，需存储所有message-key对，提取成本线性增长。SIGMark盲提取，恒定成本
- **vs VideoMark**: 同样非盲且使用PRC，但未解决全局密钥和时序鲁棒性问题
- **vs Gaussian Shading**: 图像水印方法的视频扩展，未考虑因果3D VAE的时序特性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个盲提取视频水印框架，GF-PRC和SGO设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 两个模型、T2V/I2V、多种扰动、消融完整
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 解决了视频水印的实际部署瓶颈，对AI安全有重要价值
# SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion

**会议**: ICLR 2026  
**arXiv**: [2603.02882](https://arxiv.org/abs/2603.02882)  
**代码**: https://github.com/JeremyZhao1998/SIGMark-release  
**领域**: 视频生成/水印  
**关键词**: 视频扩散模型, 水印, 盲提取, 伪随机编码, 因果3D VAE, 可扩展性

## 一句话总结
SIGMark提出首个针对现代视频扩散模型的盲提取生成内水印框架，通过全局帧级伪随机编码(GF-PRC)实现常数级提取开销的盲水印，并设计分段分组排序(SGO)模块增强因果3D VAE下的时序鲁棒性，在HunyuanVideo和Wan-2.2上以512×16位容量达到90%+比特精度。

## 研究背景与动机

1. **领域现状**：视频扩散模型（如HunyuanVideo、Wan-2.2）快速发展，AI生成内容的版权保护和溯源需求日益迫切。不可见水印是关键技术，分为后处理水印和生成内水印两类。

2. **现有痛点**：
    - 后处理水印（如DCT、DT-CWT）不可避免地降低视频质量
    - 现有生成内方法（如VideoShield、VideoMark）是非盲的：提取时需维护所有消息-密钥对进行模板匹配，开销随生成视频数量线性增长
    - 现代视频扩散模型采用因果3D VAE，时序扰动（如帧丢失）会破坏因果分组，导致水印反演极不准确

3. **核心矛盾**：可扩展性（盲提取 vs 非盲模板匹配）与时序鲁棒性（因果3D VAE的帧分组敏感性）是两个未被同时解决的关键挑战。

4. **本文目标**：(1) 如何实现常数复杂度的盲水印提取？(2) 如何在时序扰动下恢复正确的因果帧分组？

5. **核心 idea**：使用全局共享的帧级PRC密钥编码水印消息到初始噪声实现盲提取，并通过光流分割+滑动窗口检测恢复因果帧分组保证时序鲁棒性。

## 方法详解

### 整体框架
SIGMark遵循生成内水印范式：嵌入阶段将水印消息通过GF-PRC编码到初始噪声中，扩散模型去噪生成携带水印的视频；提取阶段通过SGO模块恢复帧分组，反演得到噪声后用GF-PRC解码消息。

### 关键设计

1. **全局帧级伪随机编码 (GF-PRC)**:
    - 功能：用全局共享的PRC密钥集编码水印消息到初始噪声
    - 核心思路：为每个时序维度的潜在帧分配一个PRC密钥$K[i]$。水印消息$m[i] \in \{0,1\}^M$通过PRC编码为模板比特$\mathrm{TP}[i] = \mathrm{PRC.Encode}(m[i]; K[i])$，再通过元素级调制映射到噪声：$z_0(m) = (\mathrm{TP} \times 2 - 1) \times |z_0|$
    - 设计动机：PRC的伪随机映射即使在全局密钥下也能将相同消息编码为不同随机模板比特，保持噪声的高斯分布和多样性。传统流密码（如ChaCha20）在固定密钥材料下无法提供此特性
    - 提取时仅需全局密钥即可直接解码，无需模板匹配，复杂度从$O(N)$降为$O(1)$

2. **分段分组排序模块 (SGO)**:
    - 功能：在时序扰动后恢复正确的因果帧分组和排序
    - 核心思路：两步走——
     - (1) **光流分割**：计算相邻帧的Farnebäck双向光流，通过中值流幅值、前后一致性、运动补偿残差三个指标检测时序切点，将视频分割为运动一致的片段
     - (2) **滑动窗口检测**：在每个片段中，填充$d_t-1$帧后滑动窗口，对每个窗口位置$j$反演得到潜在帧，通过PRC检测确定帧索引$\hat{\mathrm{Idx}[j]} = \mathrm{argmax}(\mathrm{PRC.Detect}(z_0'[j]; K[0,...,f_l]))$，连续检测结果一致时停止
    - 设计动机：因果3D VAE将$d_t$个帧解码为一个时序维度的潜在特征，错误分组产生不一致的潜在特征。SGO利用全局PRC密钥的帧索引检测能力恢复正确分组

3. **消息提取**:
    - 重分组后的帧经因果3D VAE编码和扩散反演得到$z_0'$
    - 通过符号函数和PRC解码恢复消息：$\hat{m[i]} = \mathrm{PRC.Decode}(\frac{\mathrm{Sgn}(z_0'[i])+1}{2}; K[i])$

### 损失函数 / 训练策略
SIGMark是training-free方法，无需训练。嵌入阶段通过数学变换保持噪声的高斯分布$z_0(m) \sim \mathcal{N}(0, \mathbf{I})$，理论上不影响生成质量。提取阶段使用DDIM反演（HunyuanVideo和Wan使用流匹配Euler离散反演），以空提示作为条件。

## 实验关键数据

### 主实验（HunyuanVideo T2V/I2V，VBench-2.0评测）

| 方法 | 类别 | 512位Bit Acc | V-score | 512×16位Bit Acc | V-score |
|------|------|-------------|---------|----------------|---------|
| No-mark | - | - | 0.490 | - | 0.490 |
| DCT | 后处理 | 0.889 | 0.424 | 0.862 | 0.423 |
| VideoMark | 非盲 | 0.873 | 0.507 | 0.758 | 0.502 |
| VideoShield | 非盲 | 1.000 | 0.497 | 0.991 | 0.506 |
| **SIGMark** | **盲** | **0.958** | **0.506** | **0.885** | **0.499** |

### 鲁棒性实验（HunyuanVideo I2V，512位/512×16位）

| 方法 | 空间(无扰动/高斯噪声/压缩/模糊) | 时序(无扰动/丢帧/插帧/裁剪) |
|------|------|------|
| VideoMark | 0.85/0.64/0.63/0.64 | 0.71/0.52/0.51/0.51 |
| VideoShield | 1.00/1.00/0.99/1.00 | 0.99/0.89/0.84/0.83 |
| **SIGMark** | **0.98/0.89/0.84/0.95** | **0.91/0.81/0.87/0.85** |

### 消融实验

| 配置 | Bit Acc | 说明 |
|------|---------|------|
| Single PRC (非盲) | 0.707 | 去掉GF-PRC后退化到VideoMark策略 |
| **GF-PRC (Ours)** | **0.905** | 完整嵌入方案 |
| w/o SGO | 0.534 | 去掉分组排序后时序扰动下大幅下降 |
| w/o OF-seg | 0.762 | 去掉光流分割 |
| w/o SW-det | 0.823 | 去掉滑动窗口检测 |
| **SGO (Ours)** | **0.869** | 完整提取方案 |

### 关键发现
- SIGMark的提取时间为常数级，而VideoShield随视频数量线性增长（百万级视频不可行）
- GF-PRC不仅实现盲提取，还通过帧间冗余纠错提升了比特精度
- SGO的两个子模块（光流分割和滑动窗口检测）缺一不可
- 后处理水印（DCT）的V-score显著低于生成内方法，验证了生成内水印的质量无损特性

## 亮点与洞察
- **盲提取的范式突破**：首次在视频扩散水印中实现真正的盲提取，提取复杂度从$O(N)$降到$O(1)$，对大规模视频平台至关重要
- **PRC的精妙应用**：利用PRC的伪随机特性在全局密钥下保持噪声多样性，这是传统流密码无法实现的
- **因果3D VAE的专用设计**：SGO模块是针对现代视频扩散模型的专用设计，对该架构的时序特性有深刻理解

## 局限与展望
- 比特精度未达100%，这与PRC编码的纠错能力和扩散反演精度有关
- 仅在HunyuanVideo和Wan-2.2上评测，对其他视频模型的泛化性需进一步验证
- 高压缩率（如极低比特率视频压缩）下的鲁棒性有待探索
- 可以尝试结合多帧投票等策略进一步提升时序鲁棒性

## 相关工作与启发
- **vs VideoShield/VideoMark**: 这些非盲方法提取时需全量匹配，SIGMark通过全局PRC实现常数级开销
- **vs Gaussian Shading**: 图像水印方法，SIGMark将其扩展到视频并解决了因果3D VAE的特有挑战
- **vs DCT/DT-CWT**: 后处理方法不可避免降低质量，SIGMark保持生成质量不变

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个盲提取视频扩散水印，GF-PRC和SGO设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 两个主流模型、多种扰动、消融实验、可扩展性分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法展开逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 对AI视频安全领域具有重要实用价值

<!-- RELATED:START -->

## 相关论文

- [DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes](../../CVPR2025/video_generation/dynamicscaler_seamless_and_scalable_video_generation_for_panoramic_scenes.md)
- [STiV: Scalable Text and Image Conditioned Video Generation](../../ICCV2025/video_generation/stiv_scalable_text_and_image_conditioned_video_generation.md)
- [VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](../../ECCV2024/video_generation/vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](../../CVPR2025/video_generation/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)
- [BlazeBVD: Make Scale-Time Equalization Great Again for Blind Video Deflickering](../../ECCV2024/video_generation/blazebvd_make_scale-time_equalization_great_again_for_blind_video_deflickering.md)

<!-- RELATED:END -->
