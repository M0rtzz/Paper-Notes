---
title: >-
  [论文解读] Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization
description: >-
  [CVPR 2026][图像生成][模型验证] BPO 提出一种无需参考模型的白盒 T2I 模型验证方法，通过三阶段流程（对抗锚点识别→二分搜索边界探索→目标优化）找到模型特有的语义边界区域，生成的验证 prompt 在 5 个 T2I 模型上达到平均 96% 准确率和 0.93 F1，比 TVN 方法快 2 倍。
tags:
  - CVPR 2026
  - 图像生成
  - 模型验证
  - 语义边界
  - 提示学习
  - T2I模型指纹
  - 知识产权
---

# Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization

**会议**: CVPR 2026  
**arXiv**: [2603.26328](https://arxiv.org/abs/2603.26328)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 模型验证、语义边界、对抗prompt优化、T2I模型指纹、知识产权

## 一句话总结

BPO 提出一种无需参考模型的白盒 T2I 模型验证方法，通过三阶段流程（对抗锚点识别→二分搜索边界探索→目标优化）找到模型特有的语义边界区域，生成的验证 prompt 在 5 个 T2I 模型上达到平均 96% 准确率和 0.93 F1，比 TVN 方法快 2 倍。

## 研究背景与动机

1. **领域现状**：T2I 模型（如 Stable Diffusion 系列）的商业价值使模型归属认证成为重要需求。需要验证一个公开部署的 T2I 模型是否确实是声称的模型（如防止换皮或盗用）。
2. **现有痛点**：(1) TVN 方法依赖多个参考模型对比，需要维护参考模型集合且难以扩展；(2) 随机/贪心 prompt 方法准确率仅 17-23%，因为通用 prompt 无法区分相似模型；(3) 现有方法计算效率低。
3. **核心矛盾**：不同 T2I 模型的文本编码器和生成器虽然相似（多基于同一架构微调），但它们的语义边界（嵌入空间中输出语义发生跳变的区域）是模型特有的。
4. **本文目标**：直接利用目标模型自身的语义边界特性生成验证 prompt，无需任何参考模型。
5. **切入角度**：类比分类器的决策边界——每个模型的语义边界位置不同，通过精确定位边界后生成贴近边界的 prompt，可以区分不同模型。
6. **核心 idea**：三阶段流程——对抗攻击找到语义翻转点→二分搜索精确定位边界→GCG 优化生成朝向边界的验证 prompt。

## 方法详解

### 整体框架

输入原始 prompt $I$ → Stage 1: GCG 对抗攻击添加后缀 $s$ 使生成语义翻转，获得边界两侧的锚点 $(P_{pis}, P_{adv})$ → Stage 2: 在嵌入空间中线性插值 $e_\alpha = (1-\alpha)e_{pis} + \alpha e_{adv}$ + 二分搜索找到精确边界 $e_{\alpha^*}$ → Stage 3: 在 $P_{adv}$ 上优化新后缀 $s'$ 使嵌入趋近 $e_{\alpha^*}$ → 输出验证 prompt $P_v$。

### 关键设计

1. **对抗锚点识别（Stage 1）**

    - 功能：找到嵌入空间中语义翻转的两个锚点
    - 核心思路：用 GCG 优化一个 8-token 后缀 $s$，目标函数 $\min_s \cos(E_t(I+s), E_t(I))$，在迭代过程中找到语义首次翻转的步 $k^*$（由 VLM 判断），取 $P_{adv} = P_{k^*}$ 和 $P_{pis} = P_{k^*-1}$ 作为边界两侧锚点
    - 设计动机：直接搜索边界不可行（嵌入空间太大），但对抗攻击天然沿着远离原始语义的方向走，路径上必然穿过语义边界

2. **二分搜索边界探索（Stage 2）**

    - 功能：从粗略锚点精确定位语义边界
    - 核心思路：在 $e_{pis}$ 和 $e_{adv}$ 之间线性插值，用二分搜索找到 $\alpha^*$ 使得 $S(G_t(e_{\alpha^*})) \neq S(M_t(I))$，精度阈值 $\epsilon = 0.001$
    - 设计动机：线性插值假设嵌入空间局部线性（事实证明在边界附近成立），二分搜索的 $O(\log(1/\epsilon))$ 复杂度远优于网格搜索

3. **目标优化（Stage 3）**

    - 功能：生成可用于验证的高鉴别力 prompt
    - 核心思路：在 $P_{adv}$ 上优化新后缀 $s'$，目标 $\max_{s'} \cos(E_t(I+s'), e_{\alpha^*})$，100 次 GCG 迭代，batch size 256。结果 $P_v$ 的嵌入恰好在目标模型的语义边界附近
    - 设计动机：$P_v$ 在目标模型上处于语义边界，但在其他模型上大概率不在边界——因此相同 $P_v$ 在不同模型上生成的语义不同，实现鉴别

### 损失函数 / 训练策略

无训练过程，纯推理时优化。GCG 攻击用于 suffix 优化，VLM（qwen-vl-max）用于语义判断。每个验证任务生成 10 张图像评估一致性分数 $C = |2r - 1|$。

## 实验关键数据

### 主实验

| 方法 | SD v1.4 | SD v2.1 | SDXL | Dreamlike | Openjourney | 平均 Acc |
|------|---------|---------|------|-----------|-------------|----------|
| Normal | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 |
| Random | 0.33 | 0.20 | 0.17 | 0.33 | 0.17 | 0.23 |
| TVN | 0.50 | 1.00 | 0.83 | 0.50 | 0.17 | 0.60 |
| **BPO** | **1.00** | **0.80** | **1.00** | **1.00** | **1.00** | **0.96** |

### 消融实验

| Prompt 变体 | 平均 Acc | 平均 F1 | 说明 |
|------------|---------|---------|------|
| $P_{pis}$（边界前） | 0.80 | 0.78 | 不够接近边界 |
| $P_{adv}$（边界后） | 0.84 | 0.80 | 已越过边界 |
| **$P_v$（优化后）** | **0.96** | **0.93** | 精确定位边界 |

### 关键发现

- BPO 平均准确率 96%，比 TVN 的 60% 高出 36 个百分点，且不需要任何参考模型
- 效率提升 2 倍：BPO 平均 159s vs TVN 321s（SD v1.4 上 5 倍加速：108s vs 553s）
- 10 张生成图像即可达到平台精度（0.96），更多图像无显著增益
- 后缀长度 8-9 token 为最优，过短信息不足，过长可能过拟合
- VLM 选择影响不大：qwen-vl-max=0.96, gemini-2.5-flash=0.92, gpt-5=0.92

## 亮点与洞察

- **语义边界作为模型指纹**：巧妙类比分类器决策边界的思路——语义边界是模型内在的、不可复制的特征，比模型水印更难伪造
- **三阶段的渐进精炼设计**：对抗攻击→二分搜索→目标优化，每步都有明确的数学基础和实验验证
- **完全无参考模型**：消除了参考模型集合的维护成本，使方法可扩展到任意新模型

## 局限与展望

- 需要白盒访问目标模型的文本编码器（梯度计算），不适用于纯 API 服务
- 仅测试了 5 个开源模型，对最新的私有模型（如 DALL-E 3、Midjourney）泛化性未知
- 对抗鲁棒性强的模型可能语义边界更模糊，更难定位
- 正则化技术可能使边界更不典型，降低验证准确率
- 后续可探索黑盒版本（通过查询 API 做边界探测）

## 相关工作与启发

- **vs TVN**: TVN 需要参考模型集合比较不一致率，BPO 直接利用模型自身特性，概念更简洁且效果更好
- **vs 模型水印**: 水印需要训练时植入，BPO 是事后验证方法，适用于已部署的模型
- **vs GCG 对抗攻击**: BPO 将 GCG 从"攻击"工具转化为"诊断"工具，用途完全不同

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 语义边界作为模型指纹的概念极具创新性
- 实验充分度: ⭐⭐⭐⭐ 5个模型+消融+效率分析，但测试规模偏小
- 写作质量: ⭐⭐⭐⭐ 三阶段描述清晰，形式化严谨
- 价值: ⭐⭐⭐⭐ 模型知识产权保护的实际需求+新视角方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](../../CVPR2025/image_generation/minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_the_score_function_for_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)

</div>

<!-- RELATED:END -->
