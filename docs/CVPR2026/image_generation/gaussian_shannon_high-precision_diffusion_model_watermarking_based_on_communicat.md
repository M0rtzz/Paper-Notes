---
title: >-
  [论文解读] Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication
description: >-
  [CVPR 2026][图像生成][扩散模型水印] 将扩散模型的水印嵌入和提取过程建模为噪声信道通信，提出 Gaussian Shannon 框架，通过级联的多数投票和 LDPC 纠错码实现水印的比特精确恢复（而非仅阈值检测）…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型水印"
  - "通信理论"
  - "纠错码"
  - "比特精确恢复"
  - "版权保护"
---

# Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication

**会议**: CVPR 2026  
**arXiv**: [2603.26167](https://arxiv.org/abs/2603.26167)  
**代码**: [https://github.com/Rambo-Yi/Gaussian-Shannon](https://github.com/Rambo-Yi/Gaussian-Shannon)  
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

**1. 把水印问题翻译成噪声信道通信：先有信道模型，才谈得上可靠纠错**

以往的水印方法各自堆鲁棒性技巧，却没人把"水印到底在和什么噪声搏斗"讲清楚，于是只能做到阈值检测、保证不了逐位精度。本文的第一步是给整个系统找一个数学框架：扩散模型的"采样 → 生成图像 → DDIM Inversion 还原噪声"这一来一回，恰好对应通信里"消息 → 经过信道 → 接收端解出消息"的过程。沿着这个类比，水印遭遇的噪声被拆成两类——神经网络去噪/反演本身的预测误差（intrinsic noise），以及图像在传播中被 JPEG、高斯噪声等攻击叠加的扰动（extrinsic noise）。两类合在一起，整个系统就被刻画成一个二进制输入加性高斯白噪声信道（BIAWGN）。一旦落到这个经典模型上，香农信道编码定理就直接可用：只要信道容量够，就存在编码方案让传输错误率趋于零。这一步本身不动任何算法，但它把"能不能做到 100% 比特恢复"从玄学变成了有成熟工具可解的工程问题——后面两个设计就是在这张信道图上分别对付收发两端。

**2. 冗余扩展 + 伪随机调制：把水印塞进初始噪声，又不动它的高斯分布**

水印要藏进扩散模型的初始噪声 $\mathbf{z}_T$，但只要改了 $\mathbf{z}_T$ 的分布，生成图像质量就会塌（Tree-Ring 改频域结构就吃了这个亏）。本文的做法是把"加冗余"和"保分布"两件事一次办了。先把 LDPC 编出的码字 $\mathbf{c}$（长度 $n$）整段重复 $R = P/n$ 次（$P$ 是潜空间维度），得到铺满整个潜空间的 $\mathbf{c}_R$——这 $R$ 份副本就是后面多数投票的弹药。再用密钥 $K$ 对 $\mathbf{c}_R$ 做伪随机调制得到信号 $\mathbf{s}$，然后按下式采样初始噪声：

$$z_T^j = (-1)^{1-s_j}\cdot|\epsilon_j|,\qquad \epsilon_j\sim\mathcal{N}(0,1)$$

关键在于符号位由 $s_j$ 决定、幅值取 $|\epsilon_j|$。由于伪随机调制让 $s_j$ 取 0/1 各约一半，正负半轴恰好各占 50%，拼回来的 $z_T$ 仍然是标准高斯分布——扩散模型完全察觉不到噪声被动过手脚，生成质量零损失。冗余负责"有得投票"，调制负责"分布不变"，一个嵌入方案同时满足了纠错和无损两个看似冲突的需求。

**3. 级联纠错：多数投票先收拾局部错误，LDPC 再扫掉残余随机错误**

提取端真正决定能否做到比特精确的，是这套两层纠错。难点在于 DDIM Inversion 还原出的噪声会犯两种性质完全不同的错：一种是**局部比特翻转**，潜空间某块区域整片错掉（对应图 4 里的大块黑斑）；另一种是**全局随机扰动**，错误零星散布在各处。任何单一纠错码都顾此失彼，所以本文让两种机制各管一摊。第一层先碰运气直接解码：把每个码字副本送进 LDPC 校验，只要某个副本满足 $H\cdot c_r^{\top}=0\pmod 2$，说明它本身就没坏，直接取信息位当水印输出。第二层兜底：如果 $R$ 个副本全都过不了校验，就对它们逐位做多数投票

$$\tilde{c}_i=\operatorname{mode}\{c_{1i},c_{2i},\dots,c_{Ri}\}$$

把 $R$ 份带噪副本融成一份更干净的聚合码字 $\tilde{c}$，再喂回 LDPC 解码。多数投票的妙处是它的错误率随冗余指数衰减——$P_{error}^{\text{maj}}\le\exp(-m\cdot D(1/2\|p))$，只要单副本错误率 $p<0.5$，加大冗余 $m$ 就能把投票后的错误率压垮，正好治那种"成片但不过半"的局部错误；而投票拉高了整体信噪比之后，剩下零散的随机错误就交给 LDPC 的稀疏校验去精修。举个直观的走法：默认 $m=16$ 时绝大多数码字副本第一层就直接解码成功，只有约 2.8% 的码字需要走到第二层投票——两层一前一后，把两类错误分而治之，最终把比特精度顶到 100%。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)

</div>

<!-- RELATED:END -->
