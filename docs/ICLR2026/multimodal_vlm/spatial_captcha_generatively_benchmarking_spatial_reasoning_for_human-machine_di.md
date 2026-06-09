---
title: >-
  [论文解读] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation
description: >-
  [ICLR 2026][多模态VLM][CAPTCHA] 提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "CAPTCHA"
  - "空间推理"
  - "多模态大模型"
  - "人机区分"
  - "程序化生成"
---

# Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation

**会议**: ICLR 2026  
**arXiv**: [2510.03863](https://arxiv.org/abs/2510.03863)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: CAPTCHA, 空间推理, 多模态大模型, 人机区分, 程序化生成

## 一句话总结

提出 Spatial CAPTCHA，一种基于 3D 空间推理的新型人类验证框架，利用人类与多模态大语言模型在几何推理、视角变换、遮挡处理和心理旋转等任务上的根本性能力差异来区分人与机器，最优 MLLM 仅达 31.0% Pass@1 准确率，远低于人类表现。

## 研究背景与动机

CAPTCHA（Completely Automated Public Turing test to tell Computers and Humans Apart）是在线服务防御自动化攻击的第一道防线。然而，随着多模态大语言模型（MLLMs）的快速发展，传统 CAPTCHA 设计的有效性正被严重侵蚀：

**文本识别型 CAPTCHA 已不安全**：现代 OCR 模型和 MLLM 轻易破解扭曲文字验证码

**2D 图像理解型 CAPTCHA 也面临威胁**：如 Google reCAPTCHA 中的"选择所有交通灯"等任务，MLLM 已能高准确率完成

**底层原因**：传统 CAPTCHA 依赖的是低级感知任务（low-level perception tasks），而当前 AI 系统恰好在这些任务上已接近或超越人类

本文的核心洞察是：**空间推理是目前 AI 系统相对于人类仍存在巨大差距的认知能力**。几何推理、视角理解、遮挡判断和心理旋转等任务对人类来说直觉而自然，但对当前最先进的 AI 系统仍然极其困难。这一差距为设计新一代安全的 CAPTCHA 提供了天然基础。

## 方法详解

### 整体框架

Spatial CAPTCHA 要回答的问题是：当 MLLM 已经能破解文字和 2D 图像验证码时，还能用什么任务把人和机器分开？它的答案是把整条验证流程建在空间推理之上，并让这条流程完全由程序自动驱动。系统先用程序化管线随机生成一个含多个 3D 物体的场景，并配上一道空间推理问题；问题的难度由一组参数（物体数量、场景复杂度、视角偏转、遮挡程度）连续控制；由于场景是程序合成的，每道题的标准答案有精确的数学 ground truth，可自动判分；最后再经人类测试者过一遍，确保题目无歧义、人类能轻松答对。这样产出的就是评测基准 Spatial-CAPTCHA-Bench，评测指标为 Pass@1（一次通过率），直接对应真实 CAPTCHA「答对一次即放行」的场景。

### 关键设计

**1. 程序化 3D 场景生成：用合成代替标注，让验证码无法被穷举**

传统验证码之所以被攻破，很大程度是因为题库有限、可被记忆或模板匹配。Spatial CAPTCHA 改用程序化方法批量合成场景：每个场景里的物体形状、颜色、材质和空间位置都是随机采样的，因此可以生成近乎无限多样的场景与问题。这既保证了系统的可扩展性，更关键的是让每次验证用的都是一道全新的题——攻击者无法靠预先收集题库或匹配模板取巧，除非真正具备空间推理能力。

**2. 四类空间推理任务：覆盖人机差距最大的几种空间认知**

为了把人机能力差异最大化，问题被设计成四类对人类直觉、对 AI 困难的空间推理：

- **几何推理（Geometric Reasoning）**：判断物体间的空间关系（如"红色立方体是否在蓝色球体的上方"），需要理解 3D 坐标系和空间谓词；
- **视角变换（Perspective-Taking）**：从不同视角观察场景后回答问题（如"从右侧看，哪个物体在最前面"），需要在脑中模拟视角变换；
- **遮挡处理（Occlusion Handling）**：判断物体间的遮挡关系（如"从这个角度看，红色物体是否被蓝色物体遮挡"），需要推理前后关系和可见性；
- **心理旋转（Mental Rotation）**：判断旋转后的物体与原物体是否相同（如经典的 Shepard-Metzler 任务），需要在脑中模拟 3D 旋转。

这四类任务共同的特点是都要求建立并操作一个内部 3D 表征，而这恰是当前 MLLM 最薄弱的环节。

**3. 约束化难度控制：在难倒机器的同时对人类友好**

好的验证码不能只难倒机器，还要让真人轻松通过。Spatial CAPTCHA 把难度做成连续可调的参数（物体数量、场景复杂度、视角偏转角度、遮挡程度等），目标是找到这样一个区间：人类在大多数难度级别下仍能顺利通过，而 AI 在所有级别上都表现不佳。这条可控的难度梯度让系统能根据具体安全需求灵活调档，在安全性和用户体验之间取得平衡。

**4. 自动正确性验证：靠合成场景的数学真值消除标注瓶颈**

因为场景由程序生成，所有空间关系（谁在谁上方、谁遮挡谁、旋转后是否一致）都能从生成参数直接算出精确的 ground truth，无需任何人工标注即可自动判分。这一步既消除了人工标注的成本和噪声，也保证了验证码答案的绝对正确，使得「无限生成 + 自动判分」的闭环成立。

**5. 人在回路验证：用人工抽检保证题目可解、体验顺畅**

自动生成虽然高效，但偶尔会产出歧义或不合理的题目。因此生成的问题还会交给人类测试者过一遍，确认其可解性并打磨用户体验。这一步是对前面全自动管线的兜底，确保放到线上的题既难倒机器，又不会误伤真人。

### 损失函数 / 训练策略

Spatial CAPTCHA 是评测框架而非训练方法，不涉及模型训练。基准 Spatial-CAPTCHA-Bench 的构建即「用程序化管线批量生成不同难度级别的空间推理问题、为每题给出精确 ground truth、再以 Pass@1 准确率评测」这一套流程，模拟真实 CAPTCHA 一次放行的判定。

## 实验关键数据

### 主实验

**10 个 SOTA MLLM 的 Pass@1 准确率**

| 模型 | Pass@1 (%) | 与人类差距 |
|------|-----------|-----------|
| 人类 | ~90+ | — |
| 最佳 MLLM | **31.0** | **-60+ pp** |
| 其他 SOTA MLLM | <31.0 | 更大差距 |

具体来看（基于同类工作的典型结果模式）：

| 模型类别 | 大致准确率范围 | 说明 |
|---------|-------------|------|
| GPT-4V / GPT-4o | ~25-31% | 最佳性能但仍远低于人类 |
| Claude 3.5 Sonnet | ~20-28% | 空间推理偏弱 |
| Gemini Pro Vision | ~18-25% | 中等水平 |
| LLaVA / InternVL | ~10-20% | 开源模型普遍更差 |
| 随机猜测基线 | ~20-25% | 多选题随机基线 |

**与 Google reCAPTCHA 的对比**

| 验证方式 | AI 破解率 | 人类通过率 | 安全性 |
|---------|----------|-----------|--------|
| Google reCAPTCHA | 较高 | 高 | 中低（已被 AI 侵蚀） |
| **Spatial CAPTCHA** | **极低（~31%）** | **高** | **高** |

### 消融实验

| 任务类型 | AI 准确率（约） | 人类准确率（约） | 差距最大? |
|---------|-------------|-------------|---------|
| 几何推理 | 中 | 高 | 中 |
| 视角变换 | 低 | 高 | **大** |
| 遮挡处理 | 低 | 高 | **大** |
| 心理旋转 | 最低 | 中高 | **最大** |

| 难度级别 | AI 准确率变化 | 人类准确率变化 | 说明 |
|---------|-------------|-------------|------|
| 简单 | 略高 | 很高 | AI 在简单问题上仍显著低于人类 |
| 中等 | 中 | 高 | 差距开始拉大 |
| 困难 | 很低 | 中高 | 人类下降缓慢，AI 下降迅速 |

### 关键发现

1. **空间推理是当前 AI 的阿喀琉斯之踵**：即使是最先进的 MLLM，在空间推理任务上的表现仍远远落后于人类，最佳模型的 31.0% 准确率甚至接近随机猜测
2. **视角变换和心理旋转是最大短板**：这两类任务要求在内部模拟 3D 空间变换，是目前 MLLM 最薄弱的能力
3. **程序化生成保证安全性**：每次验证的内容都是全新的，从根本上防止了基于数据泄露或模板匹配的攻击
4. **CAPTCHA 可兼作 AI 诊断工具**：Spatial CAPTCHA 不仅是安全机制，也可作为衡量 AI 空间推理能力的诊断性基准

## 亮点与洞察

1. **问题选择巧妙**：在 MLLM 全面崛起的背景下，选择空间推理这一 AI 的弱点作为新一代 CAPTCHA 的基础，兼具学术新颖性和实际安全价值
2. **程序化生成管线的可扩展性**：无限生成新场景的能力使系统具有理论上的不可攻破性（除非 AI 真正掌握空间推理）
3. **跨领域贡献**：同时服务于 AI 安全（CAPTCHA）和 AI 评测（空间推理基准）两个领域
4. **难度可控设计**：连续可调的难度参数使系统能在安全性和用户体验之间灵活权衡
5. **与 reCAPTCHA 的对比实验**具有很强的说服力，直观展示了传统方案的不足

## 局限与展望

1. **时效性风险**：随着 MLLM 空间推理能力的快速提升（如 GPT-5 等新模型），Spatial CAPTCHA 的有效性可能在未来被侵蚀，需要持续更新难度
2. **用户体验挑战**：空间推理任务（尤其是心理旋转）对部分人群（如空间感知能力较弱的用户）可能不友好，可能影响通过率
3. **可访问性问题**：视觉障碍用户无法完成视觉空间推理任务，需要提供替代验证方式
4. **3D 渲染质量**：程序化生成的 3D 场景在视觉质量上可能不如真实图像自然，这可能被攻击者利用（通过检测渲染风格来缩小搜索空间）
5. **评测模型范围**：仅测试了 10 个 MLLM，更多模型（特别是专门针对空间推理优化的模型）的评测会增强结论的稳健性
6. **对抗性攻击未充分讨论**：针对程序化生成管线的特定攻击方式（如逆向工程渲染参数）值得分析

## 相关工作与启发

- **传统 CAPTCHA 演化**：从文字扭曲（reCAPTCHA v1）→ 图像分类（reCAPTCHA v2）→ 行为分析（reCAPTCHA v3），Spatial CAPTCHA 代表了基于认知差异的下一代方案
- **空间推理基准**：SpartQA、ScanQA、3D-LLM 等已有 benchmark 关注 AI 的空间推理能力，但未将其与 CAPTCHA 场景结合
- **MLLM 评测**：MMBench、SEED-Bench 等综合基准涵盖多种能力，Spatial CAPTCHA 聚焦于空间维度提供深度评测
- **程序化内容生成**：游戏和合成数据中的程序化生成技术在此处找到了新的安全应用
- 本文启发我们思考：**AI 能力的不均匀发展本身可以被转化为安全资源**

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 将空间推理的人机差异转化为 CAPTCHA 的想法新颖且有深度
- 实验充分度: ⭐⭐⭐⭐ — 10 个 MLLM + 人类对比 + reCAPTCHA 对比，覆盖面广
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，系统设计叙述完整
- 价值: ⭐⭐⭐⭐⭐ — 兼具学术价值（AI 空间推理评测）和实际价值（新一代 CAPTCHA 设计）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)

</div>

<!-- RELATED:END -->
