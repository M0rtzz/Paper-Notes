---
title: >-
  [论文解读] Tiny Inference-Time Scaling with Latent Verifiers
description: >-
  [CVPR 2026][图像生成][推理时扩展] 提出VHS（Verifier on Hidden States）——一种直接在DiT生成器中间层隐状态上工作的验证器，跳过解码-重编码开销，在单步图像生成的推理时扩展（inference-time scaling）场景下将联合生成-验证时间减少63.3%、FLOPs降低51%，同时在GenEval上相同时间预算下提升2.7%的性能。
tags:
  - CVPR 2026
  - 图像生成
  - 推理时扩展
  - 隐空间验证器
  - 单步生成
  - DiT
  - MLLM
---

# Tiny Inference-Time Scaling with Latent Verifiers

**会议**: CVPR 2026  
**arXiv**: [2603.22492](https://arxiv.org/abs/2603.22492)  
**代码**: [https://aimagelab.github.io/VHS](https://aimagelab.github.io/VHS)  
**领域**: 扩散模型 / 图像生成 / LLM效率  
**关键词**: 推理时扩展, 隐空间验证器, 单步生成, DiT, MLLM

## 一句话总结
提出VHS（Verifier on Hidden States）——一种直接在DiT生成器中间层隐状态上工作的验证器，跳过解码-重编码开销，在单步图像生成的推理时扩展（inference-time scaling）场景下将联合生成-验证时间减少63.3%、FLOPs降低51%，同时在GenEval上相同时间预算下提升2.7%的性能。

## 研究背景与动机

1. **领域现状**：推理时扩展（inference-time scaling）已成为提升生成模型质量的有效方式——通过生成多个候选样本并用验证器（verifier）评分选出最佳结果。常见的Best-of-N策略在文本到图像生成中广泛使用。

2. **现有痛点**：当前验证器通常基于MLLM（多模态大语言模型），流程是：生成器在latent空间生成→解码到像素空间→MLLM的视觉编码器（如CLIP）重新编码→LLM评分。存在两个问题：(a) 解码-重编码是冗余操作——latent空间已隐式包含语义信息，却被解码后又重新编码；(b) 文献中通常只计算生成步数（function evaluations）而忽视验证器开销，但对于**单步生成器**（如SANA-Sprint），解码器和验证器的开销与生成本身可比拟。

3. **核心矛盾**：实际部署场景（如商业图像生成服务）通常只返回2-4张候选图片，是"tiny budget"设定。在如此紧的预算下，MLLM验证器的开销不可忽视。扩散模型在压缩latent空间操作以减少计算，但验证时又退回到像素空间，形成了计算上的矛盾。

4. **本文目标**：设计一种更高效的验证器，能直接在生成器的latent空间评估生成质量，消除解码-重编码开销。

5. **切入角度**：DiT生成器的中间隐层已经编码了丰富的语义信息（可以被LLM理解），不需要先解码再编码。直接用中间层特征替代CLIP视觉编码器的输出作为LLM的视觉输入。

6. **核心 idea**：验证器直接消费DiT生成器的中间隐状态作为视觉输入，跳过后续DiT层、自编码器解码和CLIP重编码，实现隐空间内的高效验证。

## 方法详解

### 整体框架
标准流程：$z_T \rightarrow$ DiT全部L层 $\rightarrow z_0 \rightarrow$ 自编码器解码 $\rightarrow x_0 \rightarrow$ CLIP编码 $\rightarrow$ LLM评分。VHS流程：$z_T \rightarrow$ DiT前 $\ell^*$ 层 $\rightarrow h_{\ell^*} \rightarrow$ MLP连接器 $\rightarrow$ LLM评分。跳过了后续DiT层、自编码器解码和CLIP重编码三个步骤。

### 关键设计

1. **隐状态验证器（VHS）**:

    - 功能：直接从DiT中间层特征进行语义评估，替代传统的解码-重编码-MLLM pipeline
    - 核心思路：标准MLLM验证器的评分为 $s = \text{LLM}(\mathcal{C}(\mathcal{V}(\mathcal{D}(z_0))), p)$，其中 $\mathcal{D}$ 是解码器，$\mathcal{V}$ 是视觉编码器，$\mathcal{C}$ 是连接器。VHS将其简化为 $s = \text{LLM}(\mathcal{C}(h_{\ell^*}), p)$，直接取DiT第 $\ell^*$ 层的隐状态 $h_{\ell^*}$ 通过MLP连接器送入LLM。这不仅跳过了 $\mathcal{D}$ 和 $\mathcal{V}$，还可以截断生成器在第 $\ell^*$ 层之后的 $L - (\ell^* + 1)$ 层，进一步减少计算。
    - 设计动机：生成式latent空间已经隐含了图像的语义信息（这是扩散模型能生成图像的前提），额外的编码步骤是冗余的。论文通过消融验证了AE latent特征虽然感知上丰富但语义较弱（重建预训练目标导致），而DiT中间层特征受生成prompt条件化，语义对齐更强。

2. **DiT层的选择（Layer Selection）**:

    - 功能：找到延迟-性能的最优平衡点
    - 核心思路：在20层DiT中测试了 $h_1, h_5, h_7, h_9, h_{19}$ 五个层。发现极浅层（$h_1$）靠近噪声输入，表示不稳定；极深层（$h_{19}$）接近AE重建空间，主导感知重建而非语义；中间层 $h_7$（约35%深度）最优——GenEval overall比 $h_5$ 高2.8%，比 $h_9$ 高2.2%，同时延迟更低（因为可以截断后续13层）。
    - 设计动机：这形成了一个非单调的trade-off：太浅语义弱，太深偏向感知重建（类似AE特征），中间层恰好保留了足够的语义信息且计算成本最低。

3. **训练流程（两阶段）**:

    - 功能：将生成器的隐状态空间与LLM的输入空间对齐，并训练为验证器
    - 核心思路：**对齐阶段**：类似LLaVA的第一阶段，用图像-文本对训练MLP连接器。由于输入是生成模型的latent而非真实图像，先用生成器从caption生成图像并记录 $h_{\ell^*}$，再用Gemma-3-4B对生成图像重标注描述（避免生成偏差）。仅训练连接器。**验证器微调阶段**：从Reflect-DiT的prompt生成20张候选图/prompt共118K样本，用GenEval自动评估得到二值标签（Yes/No）。由于正样本过多（~63%），使用**加权交叉熵**重新平衡类别权重。训练连接器+LLM全部参数。推理时用LLM输出"yes"/"no"的token概率作为连续分数。
    - 设计动机：标准交叉熵因类别不平衡偏向正类，导致验证器无法有效拒绝低质量生成。加权交叉熵和focal loss都能缓解此问题（focal loss +3.7%，加权XE +4.2%）。

### 损失函数 / 训练策略
对齐阶段用标准LLaVA训练方式，仅训连接器。验证器微调阶段用加权交叉熵（weighted cross-entropy loss），训练连接器和完整LLM。LLM使用Qwen2.5-0.5B，生成器为SANA-Sprint（单步）。

## 实验关键数据

### 主实验
SANA-Sprint + Qwen2.5-0.5B在GenEval上（匹配时间预算内的Best-of-N）：

| 时间预算 | 验证器 | Best-of-N | GenEval Overall |
|----------|--------|-----------|-----------------|
| 550ms | MLLM w/ CLIP | Bo2 | 75.4% |
| 550ms | **VHS** | **Bo4** | **78.1%** (+2.7%) |
| 1100ms | MLLM w/ CLIP | Bo4 | 78.8% |
| 1100ms | **VHS** | **Bo9** | **80.5%** (+1.7%) |
| 1650ms | MLLM w/ CLIP | Bo6 | 80.4% |
| 1650ms | **VHS** | **Bo15** | **80.9%** (+0.5%) |

延迟与资源对比（Bo1基准）：

| 验证器 | 时间 | 节省 | FLOPs节省 | VRAM节省 |
|--------|------|------|-----------|----------|
| MLLM w/ CLIP | 277ms | - | - | - |
| MLLM w/ AE | 138ms | 50.2% | 51.0% | 14.5% |
| **VHS on $h_7$** | **102ms** | **63.3%** | **62.9%** | **14.5%** |

### 消融实验

| 配置 | GenEval Overall (1100ms) | 说明 |
|------|--------------------------|------|
| VHS $h_7$ + Weighted XE | **80.5%** | 最优配置 |
| VHS $h_1$ + Weighted XE | 71.3% | 太浅，语义不足 |
| VHS $h_{19}$ + Weighted XE | 76.5% | 太深，偏向感知重建 |
| VHS $h_7$ + XE | 76.3% | 标准XE，类别不平衡 |
| VHS $h_7$ + Focal | 80.0% | Focal loss也有效 |
| MLLM w/ AE + Weighted XE | 74.7% | AE latent语义弱 |
| VHS $h_7$ + Qwen2-1.5B | 78.4% | 更大LLM无帮助，瓶颈在视觉而非推理 |

### 关键发现
- VHS的核心优势在"tiny budget"场景：相同时间MLLM w/ CLIP评2个，VHS能评4个，翻倍的候选池带来显著质量提升
- 层选择非单调：太浅语义弱，太深偏重建，$h_7$（~35%深度）最优。AE latent效果差证实了"感知特征≠语义特征"
- 增大LLM（0.5B→1.5B）几乎无帮助，瓶颈在视觉表示质量而非语言推理——这是一个重要洞察
- 加权XE > Focal loss > XE，类别不平衡处理对验证器训练至关重要
- 在PixArt-α-DMD上也有效（48%加速），证明泛化性

## 亮点与洞察
- **"少即是多"的验证器设计**：移除视觉编码器反而更好——因为DiT latent已经是条件化的语义表示，比CLIP的通用视觉特征更适合判断生成质量。这挑战了"MLLM需要强视觉编码器"的常识
- **延迟→候选数的转化**：VHS的真正价值不只是"更快"，而是在同样时间内能评估更多候选，将"效率优势"转化为"质量优势"
- **生成器中间层特征的语义分析**：DiT不同深度层的特征从噪声→语义→感知的渐变规律，对理解生成模型的内部表示有理论价值

## 局限与展望
- 仅适用于单步生成器——多步生成器的latent在每步不同，VHS需要适配
- 仅在GenEval上评估，缺少更主流的benchmark（如T2I-CompBench、DrawBench）
- 依赖特定的DiT架构——对于非DiT的生成器（如U-Net扩散模型）需要重新设计
- 当前用fixed layer $\ell^*$，未探索自适应层选择的可能性
- VHS本身需要训练（对齐+微调），不是完全training-free的方案

## 相关工作与启发
- **vs VQA-Score**: VQA-Score用VQA模型打分，需要完整的像素图像。VHS在latent空间直接评估，适合低延迟场景
- **vs Vision-Reward**: Vision-Reward用MLLM做细粒度二值QA然后加权，也需要像素图像。VHS跳过了这一步
- **vs SANA-Sprint多步**: 8步SANA-Sprint（74.0%）不如VHS的Bo4（78.1%），进一步证实Best-of-N比增加步数更高效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 直接在隐空间做验证的想法简洁有力，DiT层特征的语义分析有洞察
- 实验充分度: ⭐⭐⭐⭐ 延迟/性能/消融分析全面，但benchmark单一（仅GenEval）
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机阐述精当，效率分析细致
- 价值: ⭐⭐⭐⭐ 对实际部署图像生成服务有直接价值，隐空间验证的思路可推广到视频生成

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](../../NeurIPS2025/image_generation/remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[CVPR 2026\] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring](score2instruct_scaling_up_video_quality-centric_instructions_via_automated_dimen.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)

<!-- RELATED:END -->
