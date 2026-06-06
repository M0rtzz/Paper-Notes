---
title: >-
  [论文解读] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection
description: >-
  [CVPR 2026][LLM安全][深度伪造检测] 提出TriDF——首个从感知 (Perception)、检测 (Detection) 和幻觉 (Hallucination) 三个维度综合评估可解释深度伪造检测的基准，包含55K高质量样本覆盖16种DeepFake类型和3种模态…
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "深度伪造检测"
  - "可解释检测"
  - "多模态大模型"
  - "幻觉评估"
  - "伪影分类体系"
---

# TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection

**会议**: CVPR 2026  
**arXiv**: [2512.10652](https://arxiv.org/abs/2512.10652)  
**代码**: [https://j1anglin.github.io/TriDF/](https://j1anglin.github.io/TriDF/)  
**领域**: 扩散模型 / 安全检测  
**关键词**: 深度伪造检测, 可解释检测, 多模态大模型, 幻觉评估, 伪影分类体系

## 一句话总结
提出TriDF——首个从感知 (Perception)、检测 (Detection) 和幻觉 (Hallucination) 三个维度综合评估可解释深度伪造检测的基准，包含55K高质量样本覆盖16种DeepFake类型和3种模态，揭示了准确感知是可靠检测的基础但幻觉会严重破坏决策的三方耦合关系。

## 研究背景与动机

1. **领域现状**：随着生成模型的快速发展，DeepFake检测已从单纯的二分类发展到需要可解释性——不仅要判断真假，还要给出为什么判定为假的理由。多模态大语言模型 (MLLM) 越来越多地被用于可解释DeepFake检测。

2. **现有痛点**：
    - **现有数据集标注粗粒度**：FF++、DFDC等只有二分类标签，无法评估可解释性。
    - **现有基准覆盖不全**：DD-VQA只覆盖4种伪造类型、FakeBench只1种、LOKI仅3种；多数只支持图像模态，缺乏跨模态覆盖。
    - **缺乏幻觉评估**：MLLM生成解释时可能出现"幻觉"——给出不存在的伪影的理由。这在DeepFake检测中尤为危险，因为虚假的解释可能误导判断。现有benchmark完全没有评估这一方面。
    - **依赖MLLM判评MLLM**：许多benchmark用GPT-4o来评判其他模型的输出，引入自我偏好偏差。

3. **核心矛盾**：可解释DeepFake检测需要模型同时具备三种能力——感知伪影、正确检测、可靠解释——但目前没有统一框架来评估这三者及其相互依赖关系。

4. **本文目标** 构建一个全面的可解释DeepFake检测基准，统一评估感知、检测和幻觉三个维度，揭示它们之间的耦合关系。

5. **切入角度**：从人类标注的细粒度伪影分类体系出发，建立可量化的感知评估；将真假样本配对以支持幻觉检测；覆盖图像/视频/音频三种模态和16种DeepFake类型。

6. **核心 idea**：Perception-Detection-Hallucination构成可解释DeepFake检测的不可分割三元组，TriDF是第一个同时评估这三者的统一基准。

## 方法详解

### 整体框架
TriDF的构建流程分为两部分：(1) 数据生成与标注——从公开数据集收集人脸相关数据，用16种DeepFake技术生成真假对，进行质量控制和人工标注细粒度伪影；(2) 评估——设计三类问题（True-False、Multiple-Choice、Open-Ended），输入MLLM后用提出的指标体系评估感知、检测和幻觉。

### 关键设计

1. **细粒度伪影分类体系 (Artifact Taxonomy)**:

    - 功能：建立标准化的DeepFake伪影分类框架，为感知评估提供可靠的人工标注ground truth。
    - 核心思路：将伪影分为两大类——**质量伪影**（模糊、噪声、闪烁等，可用传统图像处理检测）和**语义伪影**（解剖学不一致、物体完整性缺陷、不自然韵律等，需要常识推理）。质量伪影进一步定位到具体位置（如鼻部、四肢、背景），系统性评估MLLM的定位能力。这套分类体系由人工标注，避免了MLLM自评估的偏差。
    - 设计动机：之前benchmark缺乏标准化的伪影标注框架，依赖MLLM生成的解释作为ground truth不可靠。人工标注的细粒度伪影提供了客观的感知评估基准。

2. **三维度评估框架 (Perception / Detection / Hallucination)**:

    - 功能：从三个互补角度全面评估MLLM的可解释DeepFake检测能力。
    - 核心思路：
        - **感知**：仅使用伪造样本，通过TFQ/MCQ/OEQ-A测试模型对伪影的识别能力（分为伪影识别和位置识别）。MCQ包含"以上都不是"选项和多选以增加难度。
        - **检测**：使用真假样本，通过OEQ-B让模型先给出真假判断再列伪影，用Accuracy和Cover评估。
        - **幻觉**：从OEQ-A和OEQ-B的回答中识别模型虚构的不存在的伪影，用CHAIR、Hal、F0.5评估。当映射伪影列表长度为0或模型将假样本判为真时，CHAIR设为1作为惩罚。
    - 设计动机：三维度不可分割——准确感知是检测的基础，但即使感知正确，幻觉也会破坏最终决策。只评估其中一两个维度无法全面理解模型能力。

3. **数据生成与质量控制**:

    - 功能：生成55K高质量真假配对样本，覆盖16种DeepFake类型和3种模态。
    - 核心思路：从30+公开数据集收集真实人脸数据，用50+专门的生成模型（GAN、SD、DiT、商业API等）生成伪造样本。16种DeepFake类型分为部分篡改（换脸、属性修改、唇形同步、面部重演、全身操纵、主体驱动编辑、语音转换）和完全合成（音频驱动说话人头、身份保持生成、文本到人类图/视频等）。每种至少用3个不同模型生成，确保生成器多样性。使用真实性和一致性指标进行自动质量筛选。
    - 设计动机：一一对应的真假配对使得伪影标注更精确（可以对比真实样本），也支持幻觉评估（对真实样本检查模型是否虚构伪影）。

### 评估指标
- 感知/检测：Accuracy（TFQ）、加减分制（MCQ，正确+1/K，错误-1/(M-K)）、Cover（OEQ中正确识别伪影的覆盖率）
- 幻觉：CHAIR（虚构伪影比例）、Hal（包含幻觉的回答比例）、F0.5（精确率加权的综合分数）
- 使用外部轻量LLM（Gemini 2.5 Flash-Lite）做伪影映射，避免"用MLLM评MLLM"的偏差。

## 实验关键数据

### 主实验 - 感知评估（TFQ）

| MLLM | 图像TFQ Avg | 视频TFQ Avg | 总Avg | 排名 |
|------|-------------|-------------|-------|------|
| GPT-5 | 63.36% | 57.02% | 60.19% | 1 |
| Gemini 2.5-Pro | 61.69% | 57.58% | 59.63% | 3 |
| Qwen3-VL-30B | 61.04% | 58.65% | 59.85% | 2 |
| Claude Sonnet 4.5 | 53.57% | 51.05% | 52.31% | 14 |
| InternVL3_5-8B | 53.69% | 54.03% | 53.86% | 7 |

### 主实验 - 检测+幻觉评估（Type-B OEQ）

| MLLM | 图像Acc | 图像Cover↑ | 图像CHAIR↓ | 图像F0.5↑ |
|------|---------|-----------|-----------|-----------|
| Qwen3-Omni-30B | 0.6942 | 0.4143 | 0.6701 | 0.3381 |
| Qwen3-VL-30B | 0.6894 | 0.3661 | 0.7137 | 0.2388 |
| InternVL2_5-38B | 0.5747 | 0.2306 | 0.8066 | 0.1971 |
| GPT-5 | - | - | - | - |

### 关键发现
- **感知是检测的基础**：感知能力（TFQ/MCQ排名）高的模型通常检测性能也更好，但不是充分条件。
- **幻觉严重破坏决策**：即使感知能力强，如果幻觉率高，检测性能也会不稳定。大多数MLLM的CHAIR > 0.5，即超过一半的回答包含虚构伪影。
- **开源vs闭源差距**：GPT-5在感知上排名第一，但在开源模型中Qwen3-VL-30B表现最好。Claude Sonnet 4.5虽然感知排名低（14），但在MCQ上得分最高（0.21），说明它的推理"精度"较高。
- **视频比图像更难**：几乎所有模型在视频模态上表现更差，说明时序伪影识别仍是挑战。
- **幻觉普遍严重**：大部分模型Type-A OEQ的Hal > 0.9，即90%以上的回答都包含至少一个幻觉伪影。这对可解释检测的可信度构成严重威胁。

## 亮点与洞察
- **三元组框架的完整性**：之前的benchmark只看检测准确率或只看解释质量，TriDF首次将感知-检测-幻觉统一到一个框架中，揭示了三者不可分割的关系。这个框架设计可以迁移到其他需要可解释AI的领域。
- **人工标注的伪影分类体系**：避免了MLLM自评估的循环偏差，建立了客观的感知评估基准。分为质量伪影和语义伪影两层的设计很有条理。
- **幻觉评估的引入非常及时**：MLLM在DeepFake检测中的幻觉问题之前被完全忽视，但实际上Hal > 0.9的结果说明目前的可解释检测远不可靠。

## 局限与展望
- 伪影标注依赖人工，成本高且可能存在标注者间一致性问题。
- 55K样本虽然规模不小，但每种DeepFake类型的分配可能不均匀，某些罕见类型样本不足。
- 评估框架主要面向静态检测，未考虑交互式检测场景（如追问细节）。
- Cover指标只评估覆盖率，不评估描述的精确度和详细程度。
- 未提供针对幻觉的改善建议或方法。

## 相关工作与启发
- **vs FakeBench**：FakeBench只覆盖1种DeepFake类型且无幻觉评估，TriDF覆盖16种且有完整的幻觉评估。
- **vs LOKI**：LOKI支持多模态但只有3种DeepFake类型，且无人工标注的伪影分类体系。
- **vs Forensics-Bench**：覆盖了10种DeepFake类型但63K样本无感知和幻觉评估。
- **vs DD-VQA**：开创了VQA形式的检测评估但只有4种类型，TriDF更全面。

## 评分
- 新颖性: ⭐⭐⭐⭐ 三维度评估框架和幻觉评估是新贡献，但核心是benchmark而非方法
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖16种类型51个生成器，评估18个MLLM
- 写作质量: ⭐⭐⭐⭐ 结构清晰但表格过多，核心洞察可以更突出
- 价值: ⭐⭐⭐⭐⭐ 填补了可解释DeepFake检测评估的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[CVPR 2026\] Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)

</div>

<!-- RELATED:END -->
