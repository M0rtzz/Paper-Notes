---
title: >-
  [论文解读] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model
description: >-
  [AAAI 2026][图像恢复][遥感图像] 提出MFmamba多功能网络，基于UNet++骨架结合Mamba上采样模块（MUB）、双池化注意力（DPA）和多尺度混合交叉块（MHCB），仅使用全色（PAN）图像输入即可同时实现超分辨率、光谱恢复及联合SR与着色三种任务。
tags:
  - AAAI 2026
  - 图像恢复
  - 遥感图像
  - 全色图像
  - 超分辨率
  - 光谱恢复
  - Mamba
  - 状态空间模型
  - UNet++
---

# MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model

**会议**: AAAI 2026  
**arXiv**: [2511.18888](https://arxiv.org/abs/2511.18888)  
**作者**: Qian Jiang, Qianqian Wang, Xin Jin, Michal Wozniak, Shaowen Yao, Wei Zhou  
**代码**: [GitHub](https://github.com/QianqianWang1325/MFmamba)  
**领域**: image_restoration  
**关键词**: 遥感图像, 全色图像, 超分辨率, 光谱恢复, Mamba, 状态空间模型, UNet++  

## 一句话总结

提出MFmamba多功能网络，基于UNet++骨架结合Mamba上采样模块（MUB）、双池化注意力（DPA）和多尺度混合交叉块（MHCB），仅使用全色（PAN）图像输入即可同时实现超分辨率、光谱恢复及联合SR与着色三种任务。

## 研究背景与动机

### 问题背景
遥感图像在目标检测、城市规划、环境监测等领域广泛应用。受传感器物理限制，全色（PAN）图像具有高空间分辨率但为灰度图，多光谱（MS）图像具有高光谱分辨率但空间分辨率低。实际需求是在仅有PAN图像输入的情况下，同时提升空间和光谱分辨率，获取高分辨率彩色遥感图像。

### 已有工作的不足
- **超分辨率（SR）方法**：只能提升空间分辨率，无法恢复光谱信息
- **着色（Colorization）方法**：只能恢复光谱信息，无法提升空间分辨率
- **Pansharpening方法**：需要PAN和MS两种配准输入，且无法实现超分辨率
- **串联方案**（如先SR再着色，或先着色再SR）：存在严重的误差累积，实验中PSNR大幅下降（如CIR+HAT在Potsdam上仅27.777 dB）
- 现有方法在复杂细节区域容易生成错误纹理，颜色失真问题尚未有效解决

### 核心动机
设计一个统一的多功能网络，通过不同输入模式在单一框架内同时实现SR、光谱恢复及联合恢复三种任务，避免多阶段串联带来的误差积累。

## 方法详解

### 整体架构
MFmamba以UNet++为骨架网络（深度为4），整合三个核心模块：
1. **MHCB**：用于初始特征提取（使用2个MHCB串联）
2. **DPA**：替换UNet++原始跳跃连接，用于同层特征传递
3. **MUB**：结合Mamba状态空间模型，用于上采样与分辨率恢复

三种任务通过不同输入实现：SR任务输入低分辨率PAN图；光谱恢复任务输入原始PAN灰度图；联合任务输入低分辨率PAN图。

### 多尺度混合交叉块（MHCB）
MHCB通过并行卷积路径提取多尺度特征：
- $3\times3$卷积分支：关注局部细节特征
- $5\times5$卷积分支：捕获更大范围的全局特征
- 两分支输出与输入分别进行残差连接后通过$1\times1$卷积融合
- 第二阶段重复上述操作，最终输出再次融合并添加残差连接
- 密集残差分组策略保持梯度流动，强化关键特征在网络层级间的持久性

### 双池化注意力（DPA）
DPA采用双流架构进行通道级特征校准：
- **自适应全局平均池化**：压缩全局空间信息为通道描述符 $A_{sq1} = \frac{1}{H \times W}\sum_{i,j} X_{i,j,c}$
- **最大池化**：捕获显著特征 $A_{sq2} = \max_{p,q} X_{p,q,c}$
- 两路分别通过Sigmoid函数生成$1\times1\times C$权重图
- 最终输出为两路加权特征的元素级求和：$DPA_{out} = (X \odot A_{ex1}) \oplus (X \odot A_{ex2})$
- 结合两种池化帮助模型同时关注关键细节和全局语义信息

### Mamba上采样模块（MUB）
MUB基于选择性状态空间模型（S6），核心是2D选择性扫描机制（2D-SSM）：
- 将2D特征图展平为1D序列，通过离散化的状态空间方程计算长程依赖
- 状态空间方程：$h_\tau = \bar{\mathbf{A}} h_{\tau-1} + \bar{\mathbf{B}} x_\tau$，$y_\tau = \mathbf{C} h_\tau + \mathbf{D} x_\tau$
- **关键改进**：将原始4个水平扫描方向扩展为**6个扫描方向**，额外增加2个对角线方向，更全面地捕获空间信息
- 使用$2\times2$ patch配置（优于$3\times3$ patch）
- 相比Transformer模块：运行更快、参数更少、显存占用更低

### 损失函数
使用L1损失：$L_1(y, \hat{y}) = \sum_{i=1}^{n} |y_i - \hat{y}_i|$

## 实验关键数据

### 实验设置
- 数据集：Potsdam、NWPU、QuickBird、GF2、RSSCN7
- 评价指标：PSNR、SSIM、MSE、MAE、SAM、LPIPS
- UNet++深度4，growth rate 32，batch size 1，训练32 epochs
- Adam优化器，学习率$1\times10^{-4}$，StepLR每10轮衰减

### 表1：联合SR与光谱恢复对比实验（Potsdam & NWPU）

| 方法 | Potsdam PSNR↑ | Potsdam SSIM↑ | Potsdam MSE↓ | NWPU PSNR↑ | NWPU SSIM↑ | NWPU MSE↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MBPRR | 34.953 | 0.943 | 28.653 | 28.504 | 0.913 | 101.887 |
| CASR | 32.685 | 0.946 | 51.054 | 28.200 | 0.906 | 109.594 |
| RSI | 34.857 | 0.910 | 25.760 | 32.354 | 0.913 | 59.534 |
| HAT+CIR | 32.853 | 0.958 | 34.774 | 22.760 | 0.533 | 393.884 |
| CIR+HAT | 27.777 | 0.720 | 375.405 | 24.519 | 0.672 | 257.197 |
| SwinIR+CIR | 32.838 | 0.957 | 34.907 | 22.789 | 0.536 | 390.958 |
| **MFmamba** | **40.148** | **0.966** | **7.096** | **33.183** | **0.927** | **46.606** |

MFmamba在Potsdam上PSNR达40.148 dB，较最佳baseline MBPRR高出**+5.2 dB**；MSE从25.760降至7.096，降幅达72%。串联方案（CIR+HAT）因误差累积严重，PSNR仅27.777 dB。

### 表2：光谱恢复（着色）对比实验（Potsdam）

| 方法 | PSNR↑ | SSIM↑ | MSE↓ | MAE↓ | SAM↓ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| RSI | 33.135 | 0.983 | 47.805 | 103.962 | 0.068 | 0.052 |
| CIR | 31.990 | 0.957 | 42.428 | 95.134 | 0.069 | 0.077 |
| SEGAN | 32.513 | 0.632 | 396.840 | 122.054 | 0.083 | 0.640 |
| Huang | 32.025 | 0.968 | 72.866 | 105.335 | 0.066 | 0.175 |
| **MFmamba** | **35.569** | **0.984** | **28.477** | **88.036** | **0.052** | **0.048** |

MFmamba在着色任务上PSNR达35.569 dB，较RSI高出**+2.4 dB**；SAM从0.066降至0.052，光谱角度误差最低。

### 消融实验关键发现
- 移除DPA：PSNR从40.148降至39.980（-0.17 dB）
- 移除MUB：PSNR降至39.794（-0.35 dB），影响最大
- 移除MHCB：PSNR降至40.061（-0.09 dB）
- MHCB数量：2个最优（40.148），1个（40.072）和3个（40.080）均略差
- UNet++深度：4层最优（40.148），2层仅39.548
- 6方向扫描优于单方向5或6扫描

## 亮点

- **多功能统一框架**：单一模型通过不同输入同时完成SR、着色、联合恢复三种任务，避免多阶段串联的误差累积问题
- **Mamba引入遥感**：首次将Mamba状态空间模型应用于PAN图像分辨率恢复，6方向2D-SSM扫描增强全局感受野，且运行速度和参数量均优于Transformer
- **DPA注意力设计**：结合全局平均池化和最大池化的双流通道注意力，兼顾全局语义和显著特征捕获
- **优势显著**：联合任务上PSNR领先次优方法5+ dB，且在三个独立任务上均取得最优或接近最优结果
- **代码开源**：提供完整代码和实验复现支持

## 局限与展望

- **训练效率偏低**：batch size仅为1，训练32 epochs，80GB GPU需求较高，对资源受限场景不友好
- **数据集规模有限**：5个遥感数据集均为中小规模，缺乏大规模高分辨率卫星数据验证
- **仅支持固定倍率**：SR任务仅支持x2和x4固定尺度，不支持任意尺度超分辨率
- **损失函数单一**：仅使用L1损失，未引入感知损失、对抗损失或光谱一致性约束
- **缺乏下游任务评估**：未验证恢复后图像在目标检测、语义分割等下游任务中的实际效果
- **仅针对遥感场景**：未在自然图像SR或着色基准上验证泛化能力

## 与相关工作的对比

- **MBPRR (Jin et al. 2024)**：同为PAN图像联合恢复方法，但PSNR仅34.953，MFmamba高出5.2 dB，主要受益于Mamba的长程建模能力
- **HAT (Chen et al. 2023)**：基于Transformer的SR方法，单独做SR时表现中等（34.963 dB），但与着色级联后性能严重退化
- **SwinIR (Liang et al. 2021)**：类似HAT的问题，SR性能34.921 dB，级联着色后退化
- **MambaIR (Guo et al. 2024)**：将Mamba引入图像恢复的代表工作，MFmamba在此基础上扩展到多功能统一框架，并增加6方向扫描
- **RSI (Feng et al. 2022)**：端到端CNN方法，着色和SR均有一定能力，但各项指标均不及MFmamba
- **CSRDNN (Feng et al. 2021)**：CNN方法，MAE上有优势但PSNR和MSE远逊于MFmamba

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多功能统一框架思路新颖，Mamba引入PAN图像恢复有创新性，但各模块（注意力、多尺度卷积）本身较为常规
- 实验充分度: ⭐⭐⭐⭐ — 5个数据集、3种任务、详细消融实验，但缺乏大规模数据和下游任务验证
- 写作质量: ⭐⭐⭐ — 结构完整，但部分公式有小错误（如DPA输出公式中$A_{ex1}$重复），写作细节有待打磨
- 价值: ⭐⭐⭐⭐ — 对遥感PAN图像处理有实际意义，统一框架思路可推广到其他多任务场景

<!-- RELATED:START -->

## 相关论文

- [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](../../ECCV2024/image_restoration/mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [QMambaBSR: Burst Image Super-Resolution with Query State Space Model](../../CVPR2025/image_restoration/qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)
- [MambaIRv2: Attentive State Space Restoration](../../CVPR2025/image_restoration/mambairv2_attentive_state_space_restoration.md)

<!-- RELATED:END -->
