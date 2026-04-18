---
title: >-
  [论文解读] TrajTok: Learning Trajectory Tokens Enhances Video Understanding
description: >-
  [CVPR 2026][视频理解][视频Token化] 提出TrajTok——首个端到端可微的轨迹视频Tokenizer，通过隐式时空聚类将视频编码为物体轨迹Token，无需外部分割/跟踪管线，在K400上+4.8%、SSv2上+4.1%，长视频QA上+8.8%，且推理效率与最高效基线持平。
tags:
  - CVPR 2026
  - 视频理解
  - 视频Token化
  - 轨迹Token
  - 端到端可微
  - Token压缩
  - 视频LLM
---

# TrajTok: Learning Trajectory Tokens Enhances Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2602.22779](https://arxiv.org/abs/2602.22779)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频Token化, 轨迹Token, 端到端可微, Token压缩, 视频LLM

## 一句话总结

提出TrajTok——首个端到端可微的轨迹视频Tokenizer，通过隐式时空聚类将视频编码为物体轨迹Token，无需外部分割/跟踪管线，在K400上+4.8%、SSv2上+4.1%，长视频QA上+8.8%，且推理效率与最高效基线持平。

## 研究背景与动机

**领域现状**：视频Transformer主流做法是将视频切成时空patch生成Token，但Token数随视频长度线性甚至二次增长，冗余严重。TrajViT首次证明按物体轨迹分组Token比patch Token更优。

**现有痛点**：TrajViT依赖外部SAM+SAM2分割跟踪管线，存在三个根本限制：(1) 管线速度慢、不可微，是独立的预处理步骤；(2) 分割粒度由通用分割模型固定，无法适配下游任务需求（如舞蹈理解需要身体部位级分割vs队形识别需要人物级分割）；(3) 数据规模增大后性能增益递减——扩展性差。

**核心矛盾**：轨迹Token范式的优越性已被证实，但生成轨迹的方式（外部管线）成为性能和效率的瓶颈。

**本文目标** 设计一个端到端可微、轻量高效的轨迹Tokenizer，使Token数量与视频时长解耦、分割粒度由下游任务反向驱动。

**切入角度**：将轨迹生成重新定义为隐式时空聚类问题——不追求像素级分割精度，而是优化语义级分组能力。

**核心 idea**：用可学习query做隐式时空聚类生成轨迹掩码，端到端与下游目标联合训练，让分割粒度被任务目标"反向塑造"。

## 方法详解

### 整体框架

TrajTok由通用分割器（Universal Segmenter）和轨迹编码器（Trajectory Encoder）两部分组成，联合训练。输入视频 $\mathbf{V}\in\mathbb{R}^{T\times H\times W\times 3}$ → 分割器生成软/硬分割掩码 → 编码器将掩码区域聚合为 $N$ 个轨迹Token $\mathbf{Z}\in\mathbb{R}^{N\times d}$（$N$ 随场景复杂度动态变化）→ 送入下游Transformer/LLM。

### 关键设计

1. **通用分割器（Universal Segmenter）**:
    - 功能：在单次前向传播中将视频划分为语义一致的轨迹区域
    - 核心思路：ConvNeXt-tiny逐帧提取1/4分辨率的多尺度特征 $\mathbf{F}\in\mathbb{R}^{T\times h\times w\times d}$；128个可学习query $\mathbf{Q}$ 通过Perceiver层对特征做cross-attention（对 $\mathbf{F}$ 施加1D RoPE编码时空位置）；输出softmax软分割图 $\mathbf{M}^{\text{soft}}_{k,t,i,j}=\text{softmax}_k(\hat{\mathbf{q}}_k\cdot\mathbf{F}_{t,i,j})$；空掩码query自动丢弃，长视频分chunk并行处理。关键trick：patch特征梯度detach后再进Perceiver，防止不稳定共适应
    - 设计动机："不需要像素完美的分割"——下游理解任务只需语义分组能力，Dice+Focal loss（不用交叉熵）强调发现所有物体区域而非像素级精度

2. **轨迹编码器（Trajectory Encoder）**:
    - 功能：将分割区域聚合为紧凑的轨迹Token表示
    - 核心思路：初始嵌入通过软掩码加权聚合保持可微 $\mathbf{z}_k^{\text{init}}=\sum_{t,i,j}\mathbf{M}^{\text{soft}}_{k,t,i,j}\cdot\mathbf{F}_{t,i,j}$；精细化阶段用第二个Perceiver做masked cross-attention（硬掩码），每个query只关注对应区域特征保证解耦；自适应Matryoshka机制——每个轨迹可输出 $n\in\{1,2,4\}$ 个sub-token（用Fourier位置嵌入初始化保证多样性），训练时随机采样 $n$，推理时按计算预算调整
    - 设计动机：软聚合保证梯度回传到分割器；硬掩码保证轨迹间解耦不混淆；自适应token数平衡效率与表达力（运动复杂的轨迹用4个token，简单的用1个）

3. **三种应用场景**:
    - 功能：验证TrajTok作为通用模块的跨场景适用性
    - 核心思路：**TrajViT2**（从头训练CLIP视频编码器）、**TrajAdapter**（冻结预训练ViT后插入TrajTok做特征适配器）、**TrajVLM**（LLaVA架构中TrajTok替代patch pooling做视觉-语言连接器，处理128帧）
    - 设计动机：证明轨迹Token不仅是Tokenizer，更是通用的特征重组模块

### 损失函数 / 训练策略

分割器：Dice + Focal loss（伪标签来自TrajViT管线注释的8M视频+15M图像）。下游目标：CLIP对比损失（TrajViT2）/ 分类损失（TrajAdapter）/ LM损失（TrajVLM）。分割器可与下游任务联合训练（TrajViT2）或预训练后冻结复用（TrajAdapter/TrajVLM）。全局batch 1024图像+128视频，8×A100训练20 epoch。

## 实验关键数据

### 主实验

| 模型 | K400 Top-1↑ | SSv2 Top-1↑ | ActivityNet vid2txt R@5↑ | VATEX vid2txt R@5↑ |
|------|------------|------------|-------------------------|-------------------|
| ViT3D | 54.2 | 46.3 | 35.6 | 60.2 |
| TokenLearner | 52.9 | 42.4 | 36.2 | 58.8 |
| TrajViT | 55.3 | 45.7 | 38.1 | 61.1 |
| **TrajViT2** | **59.1 (+4.8)** | **48.7 (+4.1)** | **42.2 (+4.1)** | **65.0 (+3.9)** |

| VLM连接器 | LongVideoBench | LVBench |
|-----------|---------------|---------|
| PatchVLM (pool=3, 32帧) | 基线 | 基线 |
| **TrajVLM (128帧)** | **+8.8%** | **+5.4%** |

| Probing方法 | K400 (VideoMAE-v2) | SSv2 (V-JEPA2) |
|------------|-------------------|----------------|
| Linear probing | 79.4 | 73.7 |
| Attentive probing | 80.2 | 74.2 |
| **TrajAdapter (4 tok/traj)** | **82.5** | **75.1** |

### 消融实验

| 模块 | 变化 | VEQ(%) | STQ(%) | R@5 |
|------|------|--------|--------|-----|
| 默认架构 | — | 42.3 | 70.1 | 22.1 |
| Perceiver | 不detach梯度 | 34.1 (↓8.2) | 59.3 (↓10.8) | 18.3 (↓3.8) |
| 分割损失 | 去掉Dice loss | 39.0 (↓3.3) | 68.9 (↓1.2) | 16.7 (↓5.4) |
| Backbone | 无层级特征 | 39.3 (↓3.0) | 66.2 (↓3.9) | 19.2 (↓2.9) |

### 关键发现

- 梯度detach是最关键设计（去掉后VEQ暴跌8.2%）——防止patch特征和query之间的不稳定共适应
- 端到端训练使分割粒度自适应下游任务：CLIP目标驱动更细的前景分割+更粗的背景合并（Figure 3可视化验证）
- TrajViT2的数据扩展性远好于TrajViT——从1M到8M训练数据持续保持对ViT3D的大幅领先
- Tokenizer仅46M参数，比ViT-Large backbone（304M）小一个量级
- 在ImageNet上TrajViT2略低于ViT3D，因为单物体简单场景下分割器产生token太少

## 亮点与洞察

- "不需要像素完美的分割"是核心insight——为理解任务做分割时，语义分组能力远比边界精度重要
- Matryoshka思路用在轨迹Token上很巧妙：运动复杂的轨迹用多Token，简单的用单Token，推理时可灵活调整
- 端到端训练让分割粒度被下游任务"反向塑造"，比固定管线灵活得多
- TrajTok作为通用模块的三种应用场景（编码器/适配器/连接器）验证了其versatility

## 局限与展望

- ImageNet上略低于ViT3D——单物体简单场景下分割器产生Token太少，需要自适应策略
- TrajVLM目前是小规模验证（Qwen3-4B），扩展到更大LLM+更多数据是未来方向
- 分割器预训练依赖TrajViT管线生成伪标签，完全自监督的轨迹发现值得探索
- temporal chunking处理长视频时可能丢失跨段轨迹连续性信息

## 相关工作与启发

- **vs TrajViT**: 端到端可微替代外部管线，效率提升一个量级，数据扩展性更好，但分割精度略低（不影响理解任务）
- **vs TokenLearner/ToMe/RLT等Token压缩方法**: 推理FLOPs相当但准确率显著更高，说明轨迹级分组比简单合并更有效
- **vs patch pooling VLM连接器（Molmo/LLaVA）**: 长视频优势巨大（+8.8% LongVideoBench），因为轨迹Token数与帧数解耦

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个端到端可微轨迹Tokenizer，范式级贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三种场景验证+全面消融+数据扩展性实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 轨迹Token思路对视觉Token压缩和视频理解有直接启发
