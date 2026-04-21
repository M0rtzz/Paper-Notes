---
title: >-
  [论文解读] Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors
description: >-
  [ICCV 2025][非视线成像] 提出Learnable Path Compensation (LPC)和Adaptive Phasor Field (APF)两个模块，分别解决NLOS成像中辐射强度衰减的材质依赖性问题和不同信噪比条件下的频域去噪问题，仅在合成数据上训练即可在多种真实数据集上实现SOTA泛化性能。
tags:
  - ICCV 2025
  - 非视线成像
  - 可学习物理先验
  - 路径补偿
  - 自适应相位场
  - SPAD
---

# Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors

**会议**: ICCV 2025  
**arXiv**: [2409.14011](https://arxiv.org/abs/2409.14011)  
**代码**: 无  
**领域**: 信号与通信  
**关键词**: 非视线成像, 可学习物理先验, 路径补偿, 自适应相位场, SPAD

## 一句话总结

提出Learnable Path Compensation (LPC)和Adaptive Phasor Field (APF)两个模块，分别解决NLOS成像中辐射强度衰减的材质依赖性问题和不同信噪比条件下的频域去噪问题，仅在合成数据上训练即可在多种真实数据集上实现SOTA泛化性能。

## 研究背景与动机

### 领域现状

非视线(NLOS)成像通过时间飞行(ToF)系统——脉冲激光和SPAD探测器——捕捉从隐藏物体间接反射的光信号，实现"看见"视线之外的物体。该技术在自动驾驶、遥感、医疗诊断等领域具有重要应用前景。

### 现有痛点

**辐射强度衰减(RIF)的经验先验不适用**：反射光子强度随距离衰减，且衰减程度与物体表面材质相关。现有方法使用单一固定系数（如$1/r^2$用于漫反射、$1/r^4$用于回反射）补偿整个场景，但同一场景可能包含多种材质

**泛化能力不足**：低信噪比(SNR)条件下（如短采集时间），泊松分布噪声导致高频混叠，传统方法产生大量伪影，学习方法泛化能力崩溃

**噪声源多样**：SPAD暗计数和环境光是两个主要噪声来源，采集时间减少时SNR急剧下降

### 核心矛盾

使用单一材质假设的路径补偿在混合材质场景中必然失败——增强某种材质物体重建的同时会显著降低其他材质物体的SNR。同时，固定的频域滤波窗口无法适应不同SNR条件。

### 切入角度

基于虚拟波相位场(phasor field)框架，设计两个可学习模块：LPC学习每个扫描点自适应的补偿系数，APF学习自适应的高斯窗口带宽来选择有效频谱。

## 方法详解

### 整体框架

基于LFE框架，输入瞬态测量信号后：(1) 特征提取模块下采样并提取特征$F_E$；(2) LPC模块学习自适应补偿系数，输出$F_C$；(3) APF模块学习最优频域窗口宽度，生成$F_A$；(4) 波传播和渲染模块将$F_A$从时空域转换到空间域，输出强度图像和深度图。

### 关键设计

#### 1. **可学习路径补偿 (LPC)**

- **功能**：为瞬态测量中每个扫描点预测自适应的路径补偿系数，替代全局固定补偿
- **核心思路**：
  1. 预定义三种物理先验补偿权重 $\{(G_Z)^r, r=1,2,4\}$，对应不同材质的衰减幅度
  2. 将补偿权重与增强特征逐元素相乘得到初始补偿特征：$F_C^{ini} = \{(G_Z)^1, (G_Z)^2, (G_Z)^4\} \otimes F_E'$
  3. 用CNN预测各补偿权重的Softmax概率，通过加权求和得到最终补偿特征
- **设计动机**：不直接预测连续的补偿系数（难以约束），而是预测三种已知物理补偿方式的概率分布，既保持了物理约束又实现了自适应。对远距离区域尤其有效

#### 2. **自适应相位场 (APF)**

- **功能**：自适应学习照明函数的高斯窗口标准差$\sigma_{pred}$，动态选择瞬态测量的有效频谱带
- **核心思路**：
  1. 将补偿特征$F_C$变换到频域
  2. 在频域上做空间和频谱卷积增强特征
  3. 用全连接层预测标准差$\sigma_{pred}$
  4. 生成自适应高斯函数：$K_G(\sigma) = \sigma\sqrt{2\pi} \exp(-\sigma^2\Omega^2/2)$
  5. 带宽关系：$\Delta\Omega = \frac{1}{2\pi\sigma}$
- **关键公式**：$F_A = \mathcal{F}^{-1}(\mathcal{F}(F_C) \cdot \mathcal{F}(\mathcal{P}(x_p, t)))$
- **设计动机**：高SNR时需要宽频带保留细节，低SNR时需要窄频带抑制噪声。固定的经验标准差无法兼顾，自适应学习可以根据输入信号的质量动态调整

#### 3. **损失函数**

端到端训练，总损失为强度重建损失和深度估计损失的加权和：

$$\mathcal{L} = \mathcal{L_I}(I, \hat{I}) + \lambda \mathcal{L_D}(D, \hat{D})$$

其中$\mathcal{L_I}$和$\mathcal{L_D}$均为MSE损失，$\lambda=1$。

## 实验关键数据

### 主实验（合成数据Seen测试集）

| 方法 | 骨干 | 内存 | 时间 | PSNR↑ | SSIM↑ | RMSE↓ | MAD↓ |
|------|------|------|------|-------|-------|-------|------|
| LCT | 物理 | 18GB | 0.11s | 19.51 | 0.3615 | 0.4886 | 0.4639 |
| FK | 物理 | 26GB | 0.16s | 21.69 | 0.6283 | 0.6072 | 0.5801 |
| RSD | 物理 | 33GB | 0.23s | 21.74 | 0.1817 | 0.5677 | 0.5320 |
| LFE | CNN | 13GB | 0.05s | 23.27 | 0.8118 | 0.1037 | 0.0488 |
| I-K | CNN | 14GB | 0.08s | 23.44 | 0.8514 | 0.1041 | 0.0476 |
| NLOST | Trans. | 38GB | 0.38s | 23.74 | 0.8398 | 0.0902 | 0.0342 |
| **Ours** | CNN | 17GB | 0.24s | **23.99** | **0.8703** | **0.0874** | **0.0312** |

### 消融实验（泛化性测试，Unseen测试集，不同SNR）

| 方法 | 10dB PSNR | 5dB PSNR | 3dB PSNR | 10dB RMSE | 3dB RMSE |
|------|-----------|----------|----------|-----------|----------|
| LFE | 23.22 | 23.15 | 23.10 | 0.1036 | 0.1044 |
| I-K | 23.45 | 23.38 | 23.32 | 0.1045 | 0.1099 |
| NLOST | 23.63 | 23.74 | 23.71 | 0.0939 | 0.0918 |
| **Ours** | **23.91** | **23.83** | **23.80** | **0.0893** | **0.0902** |

**消融各模块贡献**（真实数据定性结果）：

| 配置 | 效果 |
|------|------|
| Baseline (w/o LPC, w/o APF) | 丢失细节，背景噪声显著 |
| + LPC only | 物体细节增强（如鹿的腿部），但背景仍有伪影 |
| + APF only | 背景伪影抑制，但细节不足 |
| + LPC + APF (完整方法) | 细节完整 + 背景干净 |

### 关键发现

- **PSNR超越SOTA** 0.25dB（vs NLOST），同时**内存仅为NLOST的45%**（17GB vs 38GB）
- **SSIM最高**（0.8703），说明结构保持能力强
- **深度估计RMSE降低3.1%、MAD降低8.8%**，尤其在远距离和混合材质区域表现突出
- **跨SNR泛化稳定**：即使在极低SNR（3dB）下，性能降幅最小
- **真实数据泛化优异**：仅在合成数据训练，在FK数据集（10min采集）和NLOST数据集上均显示最佳重建质量
- **自采数据验证**：在自建的3个复杂场景（含不同材质）上表现最优

## 亮点与洞察

1. **物理先验与数据驱动的优雅结合**：LPC不是黑盒预测补偿系数，而是预测三种已知物理补偿方式的概率分布，兼具可解释性和自适应性
2. **频域自适应去噪**：APF直接在频域上学习滤波窗口，比空间域去噪更高效且物理意义明确
3. **合成到真实的零样本迁移**：仅合成训练即可泛化到多种真实成像系统，实用价值高
4. **效率突出**：相比Transformer-based NLOST，使用CNN骨干在性能更优的同时减半内存和推理时间
5. **自采数据补充**：构建了3个新的真实场景增加NLOS数据多样性

## 局限与展望

1. **仅限共焦成像系统**：扩展到非共焦系统是未来方向
2. **SPAD采集过程建模与真实传感器仍有差距**：合成数据的噪声模型可能不完全匹配真实场景
3. **未处理遮挡和多次反射**：与所有phasor field方法一样，假设场景内无遮挡和互反射
4. **定量评估仅在合成数据上**：真实数据缺少ground truth，只能做定性比较
5. **三种补偿系数是否足够**：r=1,2,4 覆盖了已知材质类型，但对新颖材质可能不够

## 相关工作与启发

- **Phasor Field (RSD)** 提供了波传播框架的理论基础，本文在此基础上实现了可学习的物理先验
- **LFE** 是第一个物理引导的学习框架，本文在其上嵌入LPC和APF模块
- **NLOST** 使用Transformer捕捉全局相关性，但代价是巨大的内存和计算开销
- NLOS成像的关键瓶颈正在从算法转向传感器和数据质量

## 评分

- **新颖性**: ⭐⭐⭐⭐ — LPC和APF的设计将物理先验与学习巧妙结合，但整体框架基于已有LFE
- **实验充分度**: ⭐⭐⭐⭐⭐ — 合成+公开真实+自采真实数据全面验证，多种SNR条件下的泛化测试
- **写作质量**: ⭐⭐⭐⭐ — 物理公式推导清晰，方法动机论述充分
- **价值**: ⭐⭐⭐⭐ — 显著提升NLOS成像的泛化能力和实用性，对低SNR场景尤其有价值

<!-- RELATED:START -->

## 相关论文

- [Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](../../ECCV2024/signal_comm/optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)
- [Boosting Multimodal Learning via Disentangled Gradient Learning](boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [ConTextTab: 语义感知的表格上下文学习器](../../NeurIPS2025/signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [Continuous Space-Time Video Resampling with Invertible Motion Steganography](../../CVPR2025/signal_comm/continuous_space-time_video_resampling_with_invertible_motion_steganography.md)
- [Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)

<!-- RELATED:END -->
