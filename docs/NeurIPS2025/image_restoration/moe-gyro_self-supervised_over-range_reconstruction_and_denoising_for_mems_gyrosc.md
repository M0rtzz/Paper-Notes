---
title: >-
  [论文解读] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes
description: >-
  [NeurIPS 2025][图像恢复][MEMS陀螺仪] 提出MoE-Gyro自监督专家混合框架，通过超量程重建专家(ORE，含高斯衰减注意力和物理信息约束)和降噪专家(DE，含双分支互补掩码和FFT引导增强)同时解决MEMS陀螺仪量程-噪声的根本权衡，将可测量范围从±450°/s扩展到±1500°/s…
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "MEMS陀螺仪"
  - "自监督学习"
  - "Mixture of Experts"
  - "超量程重建"
  - "信号降噪"
---

# MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes

**会议**: NeurIPS 2025  
**arXiv**: [2506.06318](https://arxiv.org/abs/2506.06318)  
**代码**: [GitHub](https://github.com/2002-Pan/Moe-Gyro)  
**领域**: 信号处理/惯性导航/自监督学习  
**关键词**: MEMS陀螺仪, 自监督学习, Mixture of Experts, 超量程重建, 信号降噪

## 一句话总结

提出MoE-Gyro自监督专家混合框架，通过超量程重建专家(ORE，含高斯衰减注意力和物理信息约束)和降噪专家(DE，含双分支互补掩码和FFT引导增强)同时解决MEMS陀螺仪量程-噪声的根本权衡，将可测量范围从±450°/s扩展到±1500°/s，偏置不稳定性降低98.4%。

## 研究背景与动机

MEMS陀螺仪在惯性导航和运动控制中扮演关键角色，但存在一个**根本性权衡**：
- 提高测量范围 → 噪声特性（ARW和BI）恶化
- 优化低噪声 → 角速度测量能力受限

传统方案的局限：

**硬件方案**（谐振频率调谐、闭环力反馈、多量程读出电路）：增加制造复杂度和成本

**深度学习降噪方法**（CNN、LSTM-GRU）：需要精确对齐的真值信号，仅关注噪声而不解决量程问题

**自监督方法**（LIMU-BERT、IMUDB）：缺乏同时处理降噪和超量程重建的统一框架

**现有评估缺乏统一标准**来评估多维信号增强

## 方法详解

### 整体框架

MoE-Gyro包含三个核心组件：
- **轻量级门控模块**：根据简单启发式规则将输入信号段路由到合适的专家
- **超量程重建专家(ORE)**：处理饱和/截幅信号段
- **降噪专家(DE)**：处理正常范围内的噪声信号段

两个专家共享MAE (Masked Autoencoder) backbone，端到端以纯自监督方式训练。

### 关键设计

1. **门控路由机制（Algorithm 1）**：

    - 通过检测连续3个截幅点判断是否为peak段
    - 通过检测低于阈值τ的连续样本判断是否为噪声段
    - 支持仅调用单个专家（节省约50%显存），也支持两者同时处理
    - 最终将专家输出拼接为完整增强信号

2. **超量程重建专家(ORE)的Gaussian-Decay Attention (GD-Attn)**：

    - **动机**：恢复截幅峰值所需信息集中在峰值附近短时间窗内，固定窗口忽略传感器特异性且不可微
    - **核心思路**：用可学习的连续高斯偏置替代二值窗口掩码
    - **公式**：$B_{ij} = -\frac{d_{ij}^2}{2\sigma^2}$，σ为可训练参数
    - 输出 = $\text{softmax}(QK^\top/\sqrt{d_k} + B) \cdot V$
    - σ→∞时退化为标准全局注意力，有限σ平滑下调远距离token权重
    - **最优放置位置**：仅在Decoder中（对比Encoder-only、Enc.+Dec.等配置）

3. **相关性损失(Correlation Loss)**：

    - 纯L2重建会平滑峰值，忽略局部动态
    - 第一项：一阶差分（局部斜率）对齐
    - 第二项：符号变化位置（峰/谷）的幅值保留
    - $\mathcal{L}_{\text{corr}} = \frac{1}{|\mathcal{M}|}\sum_{t\in\mathcal{M}}(\Delta x_t - \Delta \hat{x}_t)^2 + \lambda_{\text{sign}}\frac{1}{|\mathcal{E}|}\sum_{t\in\mathcal{E}}(x_t - \hat{x}_t)^2$

4. **物理信息能量损失(PINN)**：

    - 基于IMU证明质量的位移-功率关系推导
    - 计算一、二阶离散导数，定义瞬时比功率
    - 使用sigmoid归一化后加barrier项，惩罚过高或过低能量
    - 确保重建波形在物理上合理，提升跨传感器泛化能力

5. **降噪专家(DE)的双分支互补掩码策略**：

    - 两个分支共享所有encoder和decoder权重
    - 构建互补的50%掩码（cross pattern）：偶数patch给分支A可见、对分支B掩码，反之亦然
    - 保证$\mathcal{M}_A \cup \mathcal{M}_B = \{1,...,L\}$且$\mathcal{M}_A \cap \mathcal{M}_B = \varnothing$
    - 融合方式：$y_{\text{final}} = y_A \cdot \mathcal{M}_A + y_B \cdot \mathcal{M}_B$
    - 权重共享正则化提取通用特征，有效抑制高频随机噪声

6. **FFT引导训练增强**：

    - 噪声底估计→弱信号注入（从真实IMU运动片段中采样并缩放）→频谱匹配噪声合成
    - 基于IEEE标准和Allan方差分析的物理合理噪声特征
    - 针对QN、ARW、BI主导频带合成噪声
    - 训练目标：混合信号为输入，原始噪声+缩放运动为clean参考

### 损失函数 / 训练策略

- 总损失 = L2重建损失 + 相关性损失 + PINN能量损失（ORE）
- 降噪专家使用L2重建损失 + FFT引导增强
- 纯自监督训练，无需时间同步的真值标签
- 在单块NVIDIA RTX-4060 GPU上训练

## 实验关键数据

### 主实验：ISEBench全指标对比

| 模型 | PSNR↑ | P_MSE↓ | Corr↑ | SNR↑(dB) | QN↓ | ARW↓ | BI↓ | 平均排名 |
|------|-------|--------|-------|---------|-----|------|-----|---------|
| RAW | 2.67 | 1 | - | 10.18 | 0 | 0 | 0 | - |
| Matlab 2023 | 6.03 | 0.515 | 0.86 | 10.02 | -25.8% | -3.1% | +5.4% | 7.0 |
| EMD | 5.44 | 0.655 | 0.77 | 13.85 | -91.1% | -85.9% | -96.8% | 5.3 |
| SG_filter | 4.35 | 0.767 | 0.79 | 12.03 | -85.0% | -86.3% | -90.0% | 6.6 |
| HEROS_GAN | 7.70 | 0.354 | 0.89 | 16.86 | -92.8% | -51.6% | -58.3% | 3.6 |
| IMUDB | 6.59 | 0.442 | 0.82 | 17.76 | -85.8% | -87.8% | -93.7% | 3.4 |
| **MoE-Gyro** | **8.29** | **0.325** | **0.92** | **24.19** | **-98.0%** | **-94.1%** | **-98.4%** | **1** |

### 消融实验

**MoE架构消融：**

| 模型 | PSNR↑ | SNR↑ | 显存↓ |
|------|-------|------|------|
| ORE+DE（无路由） | 8.19 | 24.58 | 129 MB |
| SingleNet（同等大小） | 7.23 | 12.9 | 64.7 MB |
| **MoE-Gyro** | **8.29** | **24.19** | **71.3 MB** |

**GD-Attn放置位置消融：**

| 配置 | PSNR↑ | P_MSE↓ | Corr↑ |
|------|-------|--------|-------|
| 无GD-Attn | 8.08 | 0.345 | 0.87 |
| 仅Encoder | 8.03 | 0.345 | 0.88 |
| Enc.+Dec. | 8.27 | 0.335 | 0.90 |
| **仅Decoder** | **8.29** | **0.324** | **0.92** |

**ORE组件消融：**

| 组件组合 | PSNR↑ | P_MSE↓ | Corr↑ |
|---------|-------|--------|-------|
| 无组件 | 7.68 | 0.369 | 0.88 |
| 仅GD-Attn | 7.96 | 0.364 | 0.90 |
| 仅Corr | 7.83 | 0.354 | 0.91 |
| 仅PINN | 7.95 | 0.345 | 0.90 |
| 全部 | **8.29** | **0.324** | **0.92** |

**DE权重共享消融：**

| 权重共享方式 | SNR↑ | GPU显存↓ | 参数量↓ |
|------------|------|---------|---------|
| 不共享 | 24.51 | 116.8 MB | 27.8M |
| 仅Encoder共享 | 24.35 | 76.2 MB | 17.15M |
| **Enc.+Dec.共享** | **24.19** | **64.4 MB** | **13.9M** |

### 关键发现

1. **量程扩展**：实际角速度-1731.8°/s时，MoE-Gyro从±450°/s截幅信号重建到-1453.7°/s
2. **跨设备泛化**：在iPhone 14上训练，零样本迁移到华为P70和小米14效果良好（PSNR分别为8.28和7.95）
3. **跨任务泛化**：在周期摆动、跳跃撞击、高频扭转三类运动中均保持稳定
4. **实时性能**：完整模型117ms处理2.56秒数据段；压缩模型41ms，参数量仅1.85M
5. **MoE架构优势**：与同等大小SingleNet相比，PSNR提升1.06 dB，SNR提升11.29 dB

## 亮点与洞察

- **问题定义独到**：首次用统一自监督框架同时解决超量程重建和降噪，打破长期存在的量程-噪声权衡
- **GD-Attn设计精巧**：可学习的高斯衰减替代硬窗口，自适应聚焦峰值区域
- **PINN物理约束**：将MEMS弹簧-质量块动力学嵌入损失函数，提升泛化能力
- **FFT引导增强实用**：基于Allan方差标准的物理合理噪声生成
- **ISEBench贡献**：首个开源IMU信号增强评估基准，统一7个评估指标

## 局限与展望

- 架构相对较大，对资源受限的嵌入式设备（如MCU）部署仍有挑战
- 训练数据来自iPhone 14单一设备（虽然零样本泛化效果不错）
- 仅在陀螺仪上验证，未扩展到加速度计等其他IMU传感器
- 100 Hz采样率限制了某些高频应用
- 门控路由基于简单启发式规则，可能不够灵活

## 相关工作与启发

- 与HEROS-GAN（全监督生成方法）不同，MoE-Gyro是纯自监督的
- MAE (Masked Autoencoder) backbone为时序信号处理提供了有效框架
- 专家混合(MoE)架构成功解决了多任务学习中的梯度冲突问题
- ISEBench可作为未来惯性信号增强研究的标准评估平台

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ （首个统一自监督IMU信号增强框架，多项首创设计）
- **实验充分度**: ⭐⭐⭐⭐⭐ （全面消融、跨设备泛化、实时性能、真实场景验证）
- **写作质量**: ⭐⭐⭐⭐ （结构清晰，算法描述详细）
- **价值**: ⭐⭐⭐⭐ （解决实际工程问题，代码和benchmark开源）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](../../ICCV2025/image_restoration/blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)
- [\[ICML 2025\] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](../../ICML2025/image_restoration/timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)

</div>

<!-- RELATED:END -->
