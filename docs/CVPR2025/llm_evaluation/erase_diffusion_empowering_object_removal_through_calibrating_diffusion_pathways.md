---
title: >-
  [论文解读] Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)
description: >-
  [CVPR 2025][LLM评测][目标移除] 本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799）…
tags:
  - "CVPR 2025"
  - "LLM评测"
  - "目标移除"
  - "图像修复"
  - "扩散路径校准"
  - "自注意力修正"
  - "链式优化"
---

# Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)

**会议**: CVPR 2025  
**arXiv**: [2503.07026](https://arxiv.org/abs/2503.07026)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 目标移除, 图像修复, 扩散路径校准, 自注意力修正, 链式优化

## 一句话总结
本文提出EraDiff，通过链式校正优化范式（CRO）建立从"含物体"到"纯背景"的渐进扩散路径，并用自校正注意力机制（SRA）在采样时抑制伪影，使扩散模型真正理解"擦除意图"，在OpenImages V5上取得SOTA的Local FID（3.799），在复杂真实场景中显著优于SD2-Inpaint和LaMa。

## 研究背景与动机

1. **领域现状**：基于LDM的图像修复方法在生成自然内容方面取得了很大进展，SD2-Inpaint等模型能生成高质量修复结果。

2. **现有痛点**：现有方法在用于目标移除（erase inpainting）时效果不佳——模型会在mask区域重新生成物体而不是移除它们。例如，用户想擦掉一块披萨，模型可能会生成另一块披萨而不是干净的盘子。

3. **核心矛盾**：标准扩散训练只建立了"噪声→清晰图像"的去噪路径，训练目标是还原被mask遮挡的原始内容（包括物体），而不是理解"移除物体"的意图。这导致模型的扩散路径本身就是从噪声走向"有物体的图像"。此外，去噪早期mask形状和噪声级别产生的伪影会被自注意力机制放大。

4. **本文目标**：(1) 如何让扩散模型学会"从物体到背景"的擦除路径？(2) 如何在采样过程中纠正伪影导致的路径偏差？

5. **切入角度**：观察到擦除任务需要专用的扩散路径——模型应沿着"物体逐渐消失"的轨迹去噪，而非标准的"噪声→原图"路径。同时，自注意力的注意力分配——mask区域不应关注自身（可能包含伪影），而应关注背景区域。

6. **核心 idea**：用动态mixup图像构建渐进擦除的扩散路径（CRO）+ 用注意力掩码让mask区域只看背景（SRA）。

## 方法详解

### 整体框架
基于SD2-Inpaint架构，EraDiff包含两个核心改进：训练时用CRO建立擦除专用扩散路径，推理时用SRA替换标准自注意力纠正采样偏差。输入为含噪图像、二值mask和mask后的图像，输出为物体被移除的干净背景。

### 关键设计

1. **链式校正优化范式 (Chain-Rectifying Optimization, CRO)**:

    - 功能：建立从"含物体噪声图"到"干净背景"的渐进扩散转换路径
    - 核心思路：首先用matting模型分割原图中的主体物体，对其做旋转、缩放等变换后paste到背景上得到合成图 $x_0^{obj}$。然后对每个时间步 $t$ 构造动态混合图像 $\tilde{x}_t^{mix} = (1-\lambda_t) x_0^{ori} + \lambda_t x_0^{obj}$，其中 $\lambda_t$ 随时间步递减（大 $t$ 时物体更明显，小 $t$ 时物体逐渐消失）。$\lambda_t$ 与噪声调度 $1-\bar{\alpha}_t$ 同步变化。对混合图加噪得到新路径的隐状态 $x_t^{mix}$。新的优化目标变为最小化模型预测的前一状态与真实前一混合状态的距离：$\min_\theta \|x_{t-\gamma}^{mix} - p_\theta(\hat{x}_{t-\gamma}^{mix} | x_t^{mix})\|^2$，其中 $\gamma$ 随机采样控制跳步幅度。
    - 设计动机：标准训练的扩散路径终点是原图（含物体），而CRO的路径终点是纯背景。模型在各时间步都看到"物体逐渐消失"的过程，自然学会了擦除意图。mix-up策略的渐进变化也稳定了训练过程，避免低噪声步数时难以一步预测整个mask区域。

2. **自校正注意力机制 (Self-Rectifying Attention, SRA)**:

    - 功能：在采样过程中阻止伪影信息扩散，引导mask区域从背景而非自身提取特征
    - 核心思路：将图像mask下采样扁平化为向量 $m$，构造扩展注意力掩码矩阵 $m'_{i,j}$——当 $m_i=0$ 或 $m_j=0$（即Query或Key任一来自背景）时值为1，否则为 $-\inf$。将此掩码乘到注意力权重上：$\text{SRA}(Q,K,V) = \text{Softmax}(\frac{QK^T}{\sqrt{d}} \cdot m') V$。这样mask区域的Query只能attend到背景的Key/Value（忽略自身），而背景区域不受mask区域影响。
    - 设计动机：标准自注意力中mask区域会关注自身，但早期去噪步骤中mask区域充斥着噪声/伪影。这些伪影特征被误认为重要信息并在后续步骤中放大，最终导致物体重生。SRA强制mask区域只看背景，从根本上切断了伪影传播路径。

3. **动态图像合成策略 (Dynamic Image Synthesis)**:

    - 功能：无需额外配对数据就能构建"含物体vs纯背景"的训练对
    - 核心思路：从原图中用matting模型提取前景物体，对其做随机变换（旋转、缩放），再paste到原图的背景区域，生成新的含物体图 $x_0^{obj}$。原图 $x_0^{ori}$ 作为背景目标。通过时间步相关的mix-up将两者渐进混合，模拟物体淡出过程。这种自合成策略成本极低，不需要真实的"有物体/无物体"配对。
    - 设计动机：擦除任务的理想训练数据是同一场景的有/无物体版本，但这种配对数据几乎不存在。自合成策略巧妙地利用图像自身的前景物体来创建近似配对，虽然合成物体位置和姿态不同于原物体，但足以让模型学习"物体消失"的概念。

### 损失函数 / 训练策略
CRO优化目标：$\min_\theta \mathbb{E}_{\gamma, t} \|x_{t-\gamma}^{mix} - p_\theta(\hat{x}_{t-\gamma}^{mix} | x_t^{mix})\|_2^2$，其中 $\gamma \in (0, \gamma_m)$，$\gamma_m = 100$。$\lambda_t$ 与 $1 - \bar{\alpha}_t$ 同调度。基于SD2-Inpaint微调，使用Adam优化器，学习率 $3 \times 10^{-6}$，A100 GPU训练。

## 实验关键数据

### 主实验

| 方法 | FID↓ | LPIPS↓ | Local FID↓ |
|------|------|--------|-----------|
| SD2-Inpaint | 3.805 | 0.301 | 8.852 |
| SD2-Inpaint* (with prompt) | 4.019 | 0.308 | 7.194 |
| PowerPaint | 6.027 | 0.289 | 10.021 |
| Inst-Inpaint | 11.423 | 0.410 | 43.472 |
| LaMa | 7.533 | 0.219 | 6.091 |
| **EraDiff (ours)** | 6.540 | **0.192** | **3.799** |

注：EraDiff在Local FID和LPIPS上大幅领先，Local FID反映擦除区域的生成质量，是擦除任务最核心的指标。

### 消融实验

| 配置 | Local FID↓ | GPT-4o擦除成功率↑ |
|------|-----------|------------------|
| Full EraDiff | **3.799** | **83.43%** |
| w/o CRO | 5.713 | 72.96% |
| w/o SRA | 4.950 | 78.54% |
| w/o CRO + SRA | 8.852 | 27.80% |
| w/o mix-up | NaN | NaN |

### 关键发现
- CRO贡献最大——去掉CRO后Local FID从3.799升至5.713，GPT评分从83.43%降至72.96%
- SRA也有显著作用——去掉SRA后Local FID升至4.950，确认了伪影传播确实是重要问题
- w/o CRO+SRA等价于标准SD2-Inpaint，GPT评分仅27.80%，说明标准扩散路径完全不适合擦除任务
- 去掉mix-up后训练不收敛（NaN），验证了渐进mixup对稳定CRO训练的必要性
- 在GPT-4o对比评估中，EraDiff vs SD2-Inpaint在80.90%的案例中更优，vs LaMa在51.54%更优
- 用户研究中擦除效果和视觉连贯性两方面得分均最高

## 亮点与洞察
- **扩散路径重新定义**：CRO的核心洞察是擦除任务需要专属扩散路径而非通用去噪路径。通过mix-up构造"物体渐消"的中间状态序列，优雅地将擦除意图编码到扩散路径中。这个思路可迁移到任何需要"概念变化方向"的扩散任务（如风格渐变、季节转换）。
- **SRA零额外参数的注意力修正**：不需要添加任何参数，只需修改注意力掩码就能有效抑制伪影传播。这个trick简单但强效，可直接应用于任何基于注意力的修复模型。
- **GPT-4o作为擦除质量评估工具**：传统FID/LPIPS无法评估"是否成功移除物体"，本文用GPT-4o进行配对比较，为擦除任务建立了更合理的评估方式。

## 局限与展望
- 合成训练对中物体位置和姿态与原图不一致，可能引入域差距
- SRA可能导致mask区域内容过度依赖远距离背景，对大面积mask的语义合理性可能不足
- 仅基于SD2-Inpaint微调，未验证在更新的扩散架构（如SDXL、Flux）上的效果
- Local FID虽好但FID中等，说明全局图像质量可能有轻微下降
- 未与最新的instruction-based editing方法（如InstructPix2Pix的后续工作）进行比较

## 相关工作与启发
- **vs SD2-Inpaint**: SD2-Inpaint用标准去噪路径，终点是"还原原图"而非"背景"；EraDiff通过CRO重新定义路径终点
- **vs LaMa**: LaMa基于FFCs+GAN更擅长纹理补全，视觉连贯性好但不理解擦除意图；EraDiff在擦除效果上更优但全局FID略差
- **vs Inst-Inpaint/PowerPaint**: 这些方法依赖文本prompt来定位擦除目标，但文本指令跟随不稳定；EraDiff只用mask无需文本，更适合大规模应用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从扩散路径角度重新定义擦除任务，CRO和SRA设计都有深刻的方法论思考
- 实验充分度: ⭐⭐⭐⭐ GPT-4o评估+用户研究+消融+注意力可视化+去噪过程可视化
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，方法动机清晰，但公式较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ 解决了目标移除中"物体重生"这个实际痛点，对修图/广告/social media有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](../../ICCV2025/llm_evaluation/degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](../../ICCV2025/llm_evaluation/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](../../ICLR2026/llm_evaluation/adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](../../ACL2026/llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2025\] CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges](../../ACL2025/llm_evaluation/calibraeval_calibrating_prediction_distribution_to_mitigate_selection_bias_in_ll.md)

</div>

<!-- RELATED:END -->
