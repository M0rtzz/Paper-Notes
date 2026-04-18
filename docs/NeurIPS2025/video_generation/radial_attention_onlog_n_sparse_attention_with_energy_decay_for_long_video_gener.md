---
title: >-
  [论文解读] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation
description: >-
  [NeurIPS 2025][视频理解][稀疏注意力] Radial Attention 发现了视频扩散模型中注意力分数随时空距离指数衰减的"时空能量衰减"现象，据此设计了一种 O(n log n) 复杂度的静态稀疏注意力掩码，在 HunyuanVideo/Wan2.1 等模型上实现最高 3.7× 推理加速，并通过 LoRA 微调支持 4× 更长视频生成。
tags:
  - NeurIPS 2025
  - 视频理解
  - 稀疏注意力
  - 时空能量衰减
  - O(n log n)
  - 长视频生成
  - LoRA 微调
---

# Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation

**会议**: NeurIPS 2025  
**arXiv**: [2506.19852](https://arxiv.org/abs/2506.19852)  
**代码**: https://github.com/mit-han-lab/radial-attention  
**领域**: 高效视频生成 / 稀疏注意力  
**关键词**: 稀疏注意力, 时空能量衰减, O(n log n), 长视频生成, LoRA 微调

## 一句话总结
Radial Attention 发现了视频扩散模型中注意力分数随时空距离指数衰减的"时空能量衰减"现象，据此设计了一种 O(n log n) 复杂度的静态稀疏注意力掩码，在 HunyuanVideo/Wan2.1 等模型上实现最高 3.7× 推理加速，并通过 LoRA 微调支持 4× 更长视频生成。

## 研究背景与动机
视频扩散模型的注意力计算是核心瓶颈：

**二次复杂度不可扩展**：3D 密集注意力的 O(n²) 复杂度使得长视频训练和推理极其昂贵。HunyuanVideo 生成 5 秒 720p 视频需要约 110K token

**现有稀疏方法的局限**：
   - SVG 动态选择空间/时间注意力头，但推理时剖析可能误分类注意力头，且不适用于训练
   - STA 使用固定 3D 滑窗，但固定感受野限制了远程依赖
   - 线性注意力需要大幅架构修改，轻量微调难以恢复质量

**长视频需求**：预训练模型只能生成短视频（5秒），直接在长视频上全参数训练成本过高

核心洞察：注意力分数的衰减类似物理中信号/波随距离传播的能量衰减——时空距离越远，注意力分数越低。这启发了将能量衰减转化为计算密度衰减的稀疏策略。

## 方法详解

### 整体框架
Radial Attention 是一种静态稀疏注意力机制，使用预定义的掩码替代密集注意力。掩码设计核心原则：每个 token 关注空间位置相近的 token，注意力窗口随时间距离指数缩小。该掩码可直接用于 FlashAttention 的 block-sparse 实现。

### 关键设计

1. **时空能量衰减（Spatiotemporal Energy Decay）现象**

    - 实验观察：在 HunyuanVideo 中，post-softmax 注意力分数随 token 间的空间和时间距离增大而衰减
    - 衰减模型：$p_{js+l} \leq C_{rel} e^{-\alpha|j-i_0| - \beta|l-k_0|} p_{i_0 s+k_0}$
    - 参数 α 控制时间衰减，β 控制空间衰减
    - "空间注意力"头（SVG 定义）表现为高时间衰减 + 低空间衰减
    - "时间注意力"头表现为高空间衰减 + 低时间衰减
    - 回归分析确认衰减近似指数分布

2. **Radial Attention 掩码设计**

    - **时间维度密度衰减**：帧 i 和帧 j 之间的计算密度为 $(1/2)^{\lfloor \log_2(\max(|i-j|, 1)) \rfloor}$
    - 形成 $2\lceil \log_2 f \rceil - 1$ 个对角带（band），中心带保留 100% 密度，每个外层带密度减半
    - **空间维度密度衰减**：帧 i 到帧 j 的对角宽度为 $\lfloor s / 2^{\lfloor \log_2 \max(|i-j|,1) \rfloor} \rfloor$
    - 当对角宽度低于 1 时，降低对角频率而非继续缩小
    - **Attention sink**：第一帧保留完整注意力，因为第一帧的注意力对生成至关重要
    - 最终掩码是静态的 4D 张量 $\tilde{M} \in \{-\infty, 0\}^{f \times f \times s \times s}$

3. **复杂度分析**

    - 掩码中零元素数量上界：$4s^2 f \log_2 f = 4sn(\log_2 n - \log_2 s)$
    - 对于长视频（固定分辨率 s，大 f），复杂度为 O(n log n)
    - 4× 长视频（509帧 720p HunyuanVideo）：注意力计算减少 9×

4. **误差分析**

    - L1 注意力误差界：$O(C_{rel} e^{-\min(\beta/2, \alpha)s})$
    - 误差随衰减率 α、β 增大和空间分辨率 s 增大而指数衰减
    - 实证表明 Radial Attention 的误差比 SVG 更小

5. **LoRA 长视频适配**

    - 关键洞察：Radial Attention 仅裁剪不重要的 token 关系，保持了 softmax 注意力机制，预训练权重基本可用
    - 只需在 Q/K/V/O 投影上加 LoRA 进行轻量微调
    - 经验发现 LoRA + Radial Attention 甚至优于全参数微调，因为 LoRA 聚焦更新最关键的参数
    - 长度扩展 LoRA 兼容现有风格 LoRA

### 损失函数 / 训练策略
- 直接使用原模型的扩散训练目标
- LoRA 微调：rank 32，应用于所有注意力层的 Q/K/V/O
- 硬件友好：使用 128×128 block sparse 实现，适配 FlashAttention

## 实验关键数据

### 主实验（默认视频长度）
| 模型 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Vision Reward↑ | 加速比 |
|------|------|-------|-------|--------|----------------|--------|
| HunyuanVideo | Original | - | - | - | 0.141 | 1.0× |
| HunyuanVideo | STA | 26.7 | 0.866 | 0.167 | 0.132 | 2.29× |
| HunyuanVideo | SVG | 27.2 | 0.895 | 0.114 | 0.144 | 1.90× |
| HunyuanVideo | **Radial** | **27.3** | 0.886 | **0.114** | 0.139 | **1.88×** |
| Wan2.1-14B | Original | - | - | - | 0.136 | 1.0× |
| Wan2.1-14B | STA | 22.9 | 0.830 | 0.171 | 0.132 | 2.01× |
| Wan2.1-14B | **Radial** | **23.9** | **0.842** | **0.163** | 0.128 | 1.77× |

### 长视频扩展实验（HunyuanVideo）
| 倍数 | 方法 | 稀疏率 | 训练加速 | 推理加速 | Vision Reward↑ |
|------|------|--------|----------|----------|----------------|
| 2× (253帧) | Original | 0% | - | 1.0× | 0.122 |
| 2× | RIFLEx | 0% | - | 1.0× | 0.128 |
| 2× | Full 微调 | 0% | 1.0× | 1.0× | 0.124 |
| 2× | **Radial+LoRA** | **80.8%** | **2.78×** | **2.35×** | **0.126** |
| 4× (509帧) | Original | 0% | - | 1.0× | 0.054 |
| 4× | Full 微调 | 0% | 1.0× | 1.0× | 0.133 |
| 4× | **Radial+LoRA** | **88.3%** | **4.37×** | **3.71×** | **0.134** |

### 关键发现
- 80-88% 的注意力计算可以安全跳过，几乎不损失视频质量
- Radial+LoRA 在 4× 长视频上的 Vision Reward（0.134）甚至超过了全参数密集微调（0.133），同时训练成本降低 4.37×
- RIFLEx（training-free 方法）在 4× 长度时 Vision Reward 仅 0.037，质量急剧下降
- 将 SVG 的空间/时间注意力统一为单一 Radial 模式，消除了动态分类的不稳定性
- VBench 各维度（主体一致性、美学质量、图像质量）均保持稳定

## 亮点与洞察
- "时空能量衰减"是一个有物理直觉启发的优美发现，将物理世界中的信号衰减类比到注意力机制
- 静态掩码设计简洁且有效：无需运行时剖析，无需架构修改，可同时用于训练和推理
- O(n log n) 复杂度介于 O(n²) 密集注意力和 O(n) 线性注意力之间，在质量和效率间取得良好平衡
- LoRA 微调比全参数微调效果更好的发现令人意外，说明轻量适配可以更精准地更新关键参数
- 跨三个模型（Mochi 1/HunyuanVideo/Wan2.1）的一致表现证明了方法的通用性

## 局限与展望
- 衰减参数 α、β 在不同模型间可能不同，目前假设统一的衰减模式
- Block-sparse 的 128×128 粒度可能对小分辨率视频不够精细
- 长视频的叙事一致性和运动连贯性仍是挑战
- 与 flash attention 4 等更新硬件加速方案的结合尚待探索

## 相关工作与启发
- 统一了 SVG 中分离的空间/时间注意力模式，消除了模式分类的人工决策
- O(n log n) 复杂度族（Reformer、H-Transformer-1D、PowerAttention）中，Radial Attention 的物理直觉最强且最硬件友好
- LoRA 长视频适配为视频生成模型的长度扩展提供了一种低成本路径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
