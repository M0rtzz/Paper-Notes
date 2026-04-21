---
title: >-
  [论文解读] Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication
description: >-
  [CVPR 2026][图像生成][扩散模型水印] 将扩散模型的水印嵌入和提取过程建模为噪声信道通信，提出 Gaussian Shannon 框架，通过级联的多数投票和 LDPC 纠错码实现水印的比特精确恢复（而非仅阈值检测），在三种 Stable Diffusion 版本和七种扰动下达到 SOTA 的比特精度和检测率。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型水印
  - 通信理论
  - 纠错码
  - 比特精确恢复
  - 版权保护
---

# Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication

**会议**: CVPR 2026  
**arXiv**: [2603.26167](https://arxiv.org/abs/2603.26167)  
**代码**: https://github.com/Rambo-Yi/Gaussian-Shannon (有)  
**领域**: 图像生成 / AI安全  
**关键词**: 扩散模型水印, 通信理论, 纠错码, 比特精确恢复, 版权保护

## 一句话总结
将扩散模型的水印嵌入和提取过程建模为噪声信道通信，提出 Gaussian Shannon 框架，通过级联的多数投票和 LDPC 纠错码实现水印的比特精确恢复（而非仅阈值检测），在三种 Stable Diffusion 版本和七种扰动下达到 SOTA 的比特精度和检测率。

## 研究背景与动机
1. **领域现状**：扩散模型生成的高质量图像带来版权侵犯和虚假信息传播风险，水印技术是追踪和认证 AI 生成内容的关键防线。现有方法如 Tree-Ring、GaussianShading、PRCW 已能实现较好的水印检测。
2. **现有痛点**：现有方法依赖**阈值匹配**进行检测——即只判断"是否含水印"，而不能精确恢复水印中的每一位信息。当水印需要承载结构化数据（如许可证信息、创作者、时间戳、使用权限、加密验证标记）时，主流的的模糊匹配方案远远不够。
3. **核心矛盾**：扩散模型的生成过程本身引入预测误差，加上图像在传播过程中遭受各种攻击（JPEG压缩、高斯噪声等），导致从 DDIM Inversion 恢复的初始噪声与嵌入时的噪声存在偏差。这些偏差表现为两种错误模式：**局部比特翻转**（潜空间局部区域出现大面积错误）和**全局随机扰动**（散布在整个潜空间的随机错误）。
4. **本文目标** 如何在保持鲁棒检测的同时实现水印的**无损恢复**（100% 比特精度）？
5. **切入角度**：将水印嵌入-提取过程类比为经典通信系统中消息经噪声信道的传输与接收，用通信理论中的纠错和冗余机制来保证传输可靠性。
6. **核心 idea**：级联使用多数投票（对抗局部错误）和 LDPC 纠错码（对抗全局随机噪声）来实现扩散模型水印的比特精确恢复。

## 方法详解

### 整体框架
嵌入阶段：二进制水印 $\mathbf{w}$ → LDPC 编码得到码字 $\mathbf{c}$ → 冗余扩展为 $\mathbf{c}_R$（匹配潜空间维度）→ 伪随机调制生成信号 $\mathbf{s}$（保持标准高斯分布）→ 根据 $\mathbf{s}$ 采样初始噪声 $\mathbf{z}_T$ → 扩散模型去噪生成水印图像。提取阶段：图像 → DDIM Inversion 恢复 $\mathbf{z}_T$ → 解调制得到 $\mathbf{c}'_R$ → 尝试直接 LDPC 解码各个码字副本 → 若失败则多数投票聚合后再 LDPC 解码 → 恢复 $\mathbf{w}$。

### 关键设计

1. **基于通信理论的水印框架建模**

    - 功能：将扩散模型水印问题转化为噪声信道的可靠通信问题
    - 核心思路：扩散模型的采样→DDIM Inversion 过程等价于消息经噪声信道的输入→输出过程。"信道噪声"来自两方面：神经网络的预测误差（intrinsic noise）和图像传播中的对抗攻击（extrinsic noise）。整个系统形成一个 Binary Input AWGN Channel (BIAWGN)。基于香农理论，只要信道容量允许，就可以通过适当的编码方案实现可靠传输。
    - 设计动机：之前的方法没有从通信理论角度系统分析水印系统的可靠性，导致无法保证比特级别的精度。通信理论提供了成熟的工具来分析和对抗信道噪声。

2. **冗余扩展 + 伪随机调制的嵌入方案**

    - 功能：在保持生成质量不变的前提下将水印编码嵌入初始噪声
    - 核心思路：LDPC 编码后的码字 $\mathbf{c}$（长度 $n$）被重复 $R = P/n$ 次（$P$ 为潜空间维度），得到 $\mathbf{c}_R$。然后用密钥 $K$ 进行伪随机调制得到信号 $\mathbf{s}$。对每个位置 $j$，采样 $\epsilon_j \sim \mathcal{N}(0,1)$，初始噪声定义为 $z_T^j = (-1)^{1-s_j} \cdot |\epsilon_j|$。由于正半轴和负半轴各占 50%，所以 $z_T$ 仍然服从标准高斯分布——生成质量完全不受影响。
    - 设计动机：冗余扩展提供了多数投票所需的多个副本；伪随机调制保证了分布一致性，避免了 Tree-Ring 等方法因修改噪声分布而带来的质量损失。

3. **级联纠错系统：多数投票 + LDPC 解码**

    - 功能：对抗两种不同类型的信道错误，实现比特精确恢复
    - 核心思路：提取阶段分两层纠错。**第一层**：如果某个码字副本直接通过 LDPC 校验方程 $H \cdot c_r^T = 0 \pmod{2}$，直接取其信息位作为水印。**第二层**：若所有副本都不满足校验，对 $R$ 个副本逐位进行多数投票 $\tilde{c}_i = \text{mode}\{c_{1i}, c_{2i}, ..., c_{ri}\}$ 得到聚合码字 $\tilde{c}$，再次尝试 LDPC 解码。多数投票的错误率以指数速率衰减：$P_{error}^{\text{maj}} \leq \exp(-m \cdot D(1/2 \| p))$，当原始错误率 $p < 0.5$ 时，增加冗余数 $m$ 可快速降低错误率；LDPC 则在投票改善信道质量后处理剩余的随机错误。两者互补——多数投票擅长处理局部集中错误，LDPC 擅长处理分散的随机错误。
    - 设计动机：单一纠错机制无法同时处理两种错误模式。从图 4 的可视化可以清晰看到，局部错误（潜空间的大块黑色区域）需要多数投票来补偿，全局随机错误（分散的黑点）需要 LDPC 来纠正。

### 损失函数 / 训练策略
这个方法不需要训练或微调——它是 training-free 的。使用 DDIM 50 步采样，DDIM Inversion 50 步恢复（空提示，guidance scale=1）。默认参数：冗余度 $m=16$，LDPC 码率 $R=0.25$，信道 SNR 估计 0 dB，水印容量 256 bits。

## 实验关键数据

### 主实验（三个 SD 版本平均性能，TPR@10⁻⁶FPR / BitAcc / TPR@BitAcc.100%）

| 方法 | TPR@FPR (无噪/有噪) | BitAcc (无噪/有噪) | TPR@100%Acc (无噪/有噪) |
|------|---------------------|--------------------|-----------------------|
| GaussianShading | 1.000 / 0.999 | 0.9999 / 0.9703 | 0.989 / 0.389 |
| PRCW (ICLR2025) | 1.000 / 0.845 | 1.0000 / 0.9176 | 1.000 / 0.836 |
| **Ours** | **1.000 / 1.000** | **1.0000 / 0.9928** | **1.000 / 0.966** |

### 消融实验

| 噪声条件 | 码率 1/6 | 1/5 | **1/4** | 1/3 | 1/2 |
|----------|---------|-----|---------|-----|-----|
| 无噪声 TPR@100% | 1.000 | 0.999 | **1.000** | 1.000 | 1.000 |
| 有噪声 TPR@100% | 0.781 | 0.873 | **0.965** | 0.852 | 0.795 |

| 噪声条件 | 冗余度 16 | 8 | 4 | 2 | 1 |
|----------|----------|---|---|---|---|
| 无噪声 TPR@100% | **1.000** | 1.000 | 1.000 | 1.000 | 0.929 |
| 有噪声 TPR@100% | **0.965** | 0.739 | 0.592 | 0.314 | 0.187 |

### 关键发现
- **TPR@BitAcc.100% 是最核心的指标**：在有噪声环境下，GaussianShading 只有 38.9% 的图像能做到所有 256 位完全恢复正确，PRCW 为 83.6%，本文达到 96.6%——这在实际版权认证场景中差距巨大。
- 码率 $R=1/4$ 是甜区：更高码率冗余不足，更低码率 LDPC 校验矩阵结构缺陷导致解码失败。
- 冗余度的影响很明显：$m=16$ 时多数投票率极低（0.028），说明大部分码字可以直接 LDPC 解码成功；$m=1$ 时（无冗余无法投票）TPR@100% 直接降到 18.7%。
- 图像质量方面（FID、CLIP Score），所有语义水印方法几乎无差异，证明该方法是 quality-free 的。
- 在高级攻击（VAE 压缩、扩散重生、嵌入攻击）下仍保持强鲁棒性。

## 亮点与洞察
- **通信理论视角的深度整合**：不是简单借用纠错码，而是从信道模型出发系统分析了两种错误模式及其互补纠错策略，理论分析和实验验证高度一致。这种跨学科的方法论值得学习。
- **比特精确恢复的实用价值**：之前的水印只能回答"这是AI生成的吗？"，而 Gaussian Shannon 能回答"这张图的版权属于谁、使用条款是什么？"——从检测提升到了信息解析。
- **Zero-cost 嵌入**：通过伪随机调制保持噪声分布不变，生成质量零损失，也不需要任何微调——这是对 GaussianShading 思路的继承和发展。

## 局限与展望
- 当前使用规则 LDPC 码，码率低于 1/4 时结构缺陷导致性能下降。作者提到使用不规则 LDPC 码可以改善，但留作 future work。
- 依赖 DDIM Inversion 的准确性——不同采样器（如 DPM-Solver、Euler）恢复精度不同，影响信道质量。
- 256 bits 的水印容量在结构化数据场景下可能不够（如嵌入完整的 JSON 许可证信息）。
- 信道 SNR 估计固定为 0 dB，虽然实验显示这是合理的默认值，但自适应 SNR 估计可能在极端条件下表现更好。

## 相关工作与启发
- **vs GaussianShading (CVPR 2024)**：Gaussian Shannon 在其基础上增加了 LDPC 编码和多数投票的级联纠错机制。GaussianShading 的 BitAcc 虽然很高（0.97），但无法保证逐位精确，TPR@100% 只有 39%。
- **vs PRCW (ICLR 2025)**：PRCW 也使用了纠错码，但没有级联多数投票机制来处理局部错误。在有噪声场景下，Gaussian Shannon 的 TPR@100% 比 PRCW 高出 13 个百分点。
- **vs Tree-Ring**：Tree-Ring 在频域嵌入水印，约束了采样随机性，无法实现比特精确恢复。
- 这篇工作的启示是：**信息论和通信理论在 AI 安全领域有丰富的应用潜力**——信道编码、速率失真理论等工具可以系统性地提升水印、溯源等任务的可靠性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 通信理论视角有创意，级联纠错方案设计合理，但核心思路（冗余+纠错）在通信领域是经典方法
- 实验充分度: ⭐⭐⭐⭐⭐ 三个SD版本×七种扰动×多种消融，覆盖非常全面，高级攻击实验也做了
- 写作质量: ⭐⭐⭐⭐ 通信理论的类比解释清晰，图4的错误可视化很直观
- 价值: ⭐⭐⭐⭐⭐ 比特精确恢复对版权保护的实际部署至关重要，填补了重要空白

<!-- RELATED:START -->

## 相关论文

- [Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)

<!-- RELATED:END -->
