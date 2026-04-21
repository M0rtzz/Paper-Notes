---
title: >-
  [论文解读] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks
description: >-
  [ICCV 2025][多模态][VLM安全] 发现主流VLM普遍缺乏高斯噪声鲁棒性，提出Robust-VLGuard安全数据集（含图文对齐/不对齐场景）配合噪声增强微调提升高斯噪声鲁棒性，再结合DiffPure将对抗噪声转化为高斯噪声，构建DiffPure-VLM通用防御框架，有效抵御多种强度的对抗攻击。
tags:
  - ICCV 2025
  - 多模态
  - VLM安全
  - 高斯噪声
  - 对抗攻击防御
  - 扩散模型净化
  - 安全微调
---

# Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks

**会议**: ICCV 2025  
**arXiv**: [2504.01308](https://arxiv.org/abs/2504.01308)  
**代码**: [https://github.com/JarvisUSTC/DiffPure-RobustVLM](https://github.com/JarvisUSTC/DiffPure-RobustVLM)  
**领域**: 多模态VLM  
**关键词**: VLM安全, 高斯噪声, 对抗攻击防御, 扩散模型净化, 安全微调

## 一句话总结
发现主流VLM普遍缺乏高斯噪声鲁棒性，提出Robust-VLGuard安全数据集（含图文对齐/不对齐场景）配合噪声增强微调提升高斯噪声鲁棒性，再结合DiffPure将对抗噪声转化为高斯噪声，构建DiffPure-VLM通用防御框架，有效抵御多种强度的对抗攻击。

## 研究背景与动机
VLM通过整合视觉和文本信息扩展了LLM的能力，但相比纯文本LLM面临更严峻的安全挑战：

**视觉模态引入新攻击面**：传统基于优化的对抗攻击（PGD等）可以注入不可感知的扰动来越狱VLM

**关键盲点——高斯噪声脆弱性**：作者发现一个被忽视的根本问题——**主流VLM（InternVL2、LLaVA、MiniGPT-4）在训练中缺少噪声增强**，导致即使简单的高斯噪声就能同时降低模型的有用性和安全性

具体表现：
- **有用性退化**：InternVL2在MM-Vet上的表现从59.9%降到54.4%（加高斯噪声后）
- **安全对齐崩塌**：InternVL2在RealToxicityPrompts上的攻击成功率从50.5%升到57.2%（仅加σ=0.1的高斯噪声）

核心矛盾：现有防御方法（如VLGuard）仅关注内容层面的安全数据，忽略了**噪声扰动本身就能破坏安全对齐**的事实。DiffPure虽能净化对抗噪声但不完全消除，而是将其转化为高斯分布——这恰好可以与噪声增强微调形成互补。

核心idea：**两步防御**——先通过噪声增强安全微调让VLM对高斯噪声免疫，再利用DiffPure将任意对抗扰动转化为高斯类噪声，从而实现对广谱攻击的防御。

## 方法详解

### 整体框架
DiffPure-VLM由两个组件串联构成：
1. 前端：DiffPure扩散模型预处理（将对抗噪声→高斯类噪声）
2. 后端：经Robust-VLGuard噪声增强微调的鲁棒VLM（对高斯噪声免疫）

### 关键设计
1. **Robust-VLGuard数据集**:

    - 功能：构建包含三类数据的多模态安全微调数据集
    - 数据构成：
        - **通用指令数据**（4,467条）：涵盖QA、知识、数学、OCR、空间推理等，使用GPT-4V优化标注答案（原始标注过于简短影响学习效果）
        - **图文对齐安全数据**（1,000条）：来自VLGuard，图像内容与安全相关文本对齐
        - **图文不对齐安全数据**（1,000条）：图像与安全文本无关的场景——这是关键创新点
    - 设计动机：微调VLM进行视觉任务本身就会破坏预训练LLM的安全对齐，且对抗攻击注入的噪声与文本提示无关，因此必须训练模型处理图文不匹配的场景

2. **噪声增强安全微调**:

    - 功能：在Robust-VLGuard上用LoRA微调视觉编码器，训练时以70%概率对图像添加随机标准差（σ ∈ [0.01, 0.15]）的高斯噪声
    - 核心思路：仅微调视觉编码器的LoRA参数，训练3个epoch，单卡A100约3小时
    - 设计动机：通过让模型在带噪图像上学习正确的安全响应，使其对噪声扰动具备内在鲁棒性
    - 关键消融：通用/安全数据比例4:2最优，epoch增加对有用性影响很小但持续降低攻击成功率

3. **DiffPure的分布转移特性**:

    - 功能：利用扩散模型的前向-逆向过程净化对抗图像
    - 核心发现：DiffPure在适当时间步 $t^* \in [50, 150]$ 下，不是完全去噪，而是将对抗噪声的分布从非高斯转移为高斯类分布
    - 量化验证：
        - **峰度（Kurtosis）**：高斯分布为3，DiffPure处理后的残差噪声峰度落入[3,6]区间
        - **Q-Q偏差**：DiffPure处理后的残差与理论高斯分布的RMSE ≤ 0.01
    - 设计动机：直接对VLM应用DiffPure无法降低攻击成功率（因模型本身对高斯噪声也不鲁棒），但配合噪声增强微调后的VLM则效果显著

### 损失函数 / 训练策略
- LoRA微调视觉编码器，冻结其他参数
- 标准指令微调损失（下一token预测）
- DiffPure使用无条件扩散模型，时间步 $t^*$ 选择50或150

## 实验关键数据

### 主实验

| 模型 | 配置 | MM-Vet (↑) | 攻击成功率 (↓) | 说明 |
|------|------|-----------|--------------|------|
| InternVL2-8B | 原始 | 59.9% | 50.5% | 基线 |
| +VLGuard | 微调 | 42.9% (-17.0) | 27.7% | 有用性严重退化 |
| +RobustVLGuard | 微调 | **56.2%** (-3.7) | **29.9%** | 安全性相当，有用性远优 |
| LLaVA-v1.5-7B | 原始 | 33.0% | 57.7% | — |
| +RobustVLGuard | 微调 | 30.3% (-2.7) | **43.6%** | 安全性大幅改善 |
| MiniGPT-4-13B | 原始 | 26.7% | 34.8% | — |
| +RobustVLGuard | 微调 | **26.9%** (+0.2) | **16.0%** | 有用性不降反升 |

### 消融实验 / DiffPure-VLM防御效果

| 图像类型 (ε=32/255) | InternVL2 攻击率 | LLaVA 攻击率 | MiniGPT-4 攻击率 |
|---------------------|-----------------|-------------|-----------------|
| 干净图像 | 29.9% | 43.6% | 16.0% |
| 高斯噪声图像 | 34.5% | 42.3% | 16.5% |
| 对抗图像 | **70.6%** | **62.5%** | **53.7%** |
| +DiffPure-VLM (t*=50) | 33.4% | 43.9% | **13.6%** |
| +DiffPure-VLM (t*=150) | **32.8%** | **42.5%** | 11.9% |

| 防御方法 (LLaVA, ε=32/255) | 攻击成功率 | 说明 |
|---------------------------|-----------|------|
| 无防御 (VLGuard) | 70.4% | 基线 |
| JailGuard + VLGuard | 52.1% | 检测型防御 |
| DiffPure + VLGuard | 51.1% | 净化型防御 |
| JailGuard + RobustVLGuard | 48.9% | — |
| DiffPure + RobustVLGuard | **43.9%** | 最优组合 |

### 关键发现
- **高斯噪声是被严重低估的威胁**：仅σ=0.1的随机噪声就能让InternVL2的攻击成功率提升6.7个百分点
- **VLGuard的"过度防御"问题**：InternVL2-VLGuard的MM-Vet从59.9%骤降至42.9%，说明盲目增加安全数据会大幅牺牲有用性
- **图文不对齐数据的重要性**：VLGuard仅含对齐安全数据，在RealToxicityPrompts（图文不对齐的攻击）上改善有限
- **DiffPure-VLM的防御几乎完美恢复基线**：对抗攻击后的攻击率可降至接近干净图像水平（如MiniGPT-4的13.6% vs 16.0%）
- DiffPure的关键不是"去噪"而是"转噪"——将非高斯对抗噪声转为高斯噪声

## 亮点与洞察
1. **发现+解决并举**：首次系统证明主流VLM对高斯噪声的脆弱性，然后提出完整的防御方案
2. **分布转移的巧妙利用**：DiffPure不完全消除噪声反而是优势——将未知分布的对抗噪声统一转化为已知的高斯分布，配合针对性微调实现"攻防闭环"
3. **数据设计的匠心**：图文不对齐安全数据填补了VLGuard的盲区，少量高质量数据即可显著提升鲁棒性
4. **实用性强**：单卡A100训练3小时，DiffPure推理开销低于JailGuard（无需多次模型推理）

## 局限与展望
1. 仅在三个VLM上验证，未测试更新的模型（如Qwen2-VL、LLaMA-3.2-Vision）——附录有初步结果但不够深入
2. 噪声增强仅应用于微调阶段，若能在预训练中集成效果可能更好
3. DiffPure的时间步 $t^*$ 需要根据攻击强度选择，是否可以设计自适应的时间步选择机制？
4. 安全数据集规模较小（总计6,467条），扩展到更多任务和更多安全类别可能进一步提升效果

## 相关工作与启发
- **与VLGuard的关系**：Robust-VLGuard在其基础上增加了图文不对齐场景和噪声增强训练，本质上是将鲁棒性维度纳入安全考量
- **与DiffPure原始工作的关系**：原始DiffPure针对图像分类器，本文将其适配到VLM场景，并发现其"分布转移"而非"去噪"的独特性质
- **启发**：VLM的安全问题不仅是内容层面的，**信号层面的噪声也可以破坏安全对齐**——这启示我们安全训练应考虑更广泛的攻击面

## 评分
- 新颖性: ⭐⭐⭐⭐ 高斯噪声脆弱性的发现新颖，DiffPure-VLM的组合设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个基础模型、多攻击强度、详细消融，但VLM种类可更多
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从发现问题到解决问题层层推进
- 价值: ⭐⭐⭐⭐ 揭示了VLM安全的重要盲区，防御框架实用且可推广

<!-- RELATED:START -->

## 相关论文

- [One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models](one_perturbation_is_enough_on_generating_universal_adversarial_perturbations_aga.md)
- [NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](../../CVPR2026/multimodal_vlm/llavashield_safeguarding_multimodal_multi-turn_dialogues_in_vision-language_mode.md)
- [Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](../../CVPR2025/multimodal_vlm/4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)

<!-- RELATED:END -->
