---
title: >-
  [论文解读] Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues
description: >-
  [ACL2026][多模态VLM][视线目标推断] 这篇论文用 1,360 张受控真实照片和预注册统计检验发现，当前 VLM 在判断人看向哪个物体时远弱于人类，主要会把头部朝向误当成视线方向；对专门 gaze 模型微调后可缓解但不能完全消除这种偏差。 领域现状：视线是人类交流中非常重要的非语言线索。对话机器人、具身智能和多模…
tags:
  - "ACL2026"
  - "多模态VLM"
  - "视线目标推断"
  - "头部朝向偏差"
  - "非语言线索"
  - "VLM行为评估"
  - "受控实验"
---

# Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues

**会议**: ACL2026  
**arXiv**: [2506.05412](https://arxiv.org/abs/2506.05412)  
**代码**: https://zoryzhang.github.io/gaze  
**领域**: 多模态VLM  
**关键词**: 视线目标推断, 头部朝向偏差, 非语言线索, VLM行为评估, 受控实验

## 一句话总结
这篇论文用 1,360 张受控真实照片和预注册统计检验发现，当前 VLM 在判断人看向哪个物体时远弱于人类，主要会把头部朝向误当成视线方向；对专门 gaze 模型微调后可缓解但不能完全消除这种偏差。

## 研究背景与动机
**领域现状**：视线是人类交流中非常重要的非语言线索。对话机器人、具身智能和多模态助手如果能判断用户看向哪个物体，就可以更好地消解指代、理解意图和协作行动。专门 gaze estimation 模型在一些互联网图像 benchmark 上接近人类水平，而 VLM 具备跨任务通用性，看起来是整合 gaze 技能和语言推理的理想载体。

**现有痛点**：已有 VLM gaze 评测多停留在“能不能答对”层面，不能解释模型为什么错。互联网上的 gaze 图像常常存在头部方向和眼睛方向一致的捷径，所以模型即使答对，也可能只是用了头部朝向或身体方向，而没有真正读取眼睛外观。

**核心矛盾**：应用需要模型理解“眼睛实际看向哪里”，但训练数据和常规 benchmark 可能鼓励模型学习“头朝哪里就看哪里”的捷径。只报告准确率无法区分真正的 gaze inference 和 head-orientation heuristic。

**本文目标**：作者要构建一个受控实验，隔离 gaze target inference 的核心能力，比较 VLM 与人类差距，判定模型是否存在头部朝向偏差，并通过微调实验判断这种偏差更可能来自训练数据还是模型架构。

**切入角度**：论文采用认知科学式实验设计：预注册实验方案，控制物体数量、物体距离、拍摄视角、视线目标和头部朝向，并用一组统计测试把模型行为分类为 head-only、head-dominant、eye-head ambivalent 或更可靠的 gaze inference。

**核心 idea**：不要只给 VLM 做自然图片 benchmark，而要通过眼睛方向和头部方向的系统性错位来压力测试模型是否真的会看眼睛。

## 方法详解
这篇论文的核心贡献是实验设计而不是模型算法。作者先拍摄真实场景：一个人坐在桌前，桌上有 2 到 4 个物体，他的眼睛看向其中一个物体。关键是作者控制头部朝向，使其有时与视线方向一致，有时指向干扰物体，有时自然不约束。这样一来，如果模型只靠头部朝向，它会在 incongruent condition 中系统性出错。

### 整体框架
评测输入是一张真实照片和一道多选题，问题大意是“这个人正在看哪个物体？”，选项是桌上物体名称。主实验包含 1,360 张照片，每个 VLM 对每张照片看 16 个 prompt 模板之一，因此每个 VLM 有 21,760 次试验。主测模型包括 GPT-5.2、GPT-4o-2024-08-06、Qwen3-VL-30B-A3B-Instruct、InternLM-XComposer2-vl-7B、GLM-4.6V，同时加入 Moondream2 的专门 gaze function 和 GazeLLE 专门视觉模型作为对照。

实验变量包括物体数量、物体组合、拍摄视角、物体之间距离、prompt 模板、刺激 ID、真实 GazeTarget 和 HeadTarget。因变量包括模型选择的物体 Choice，以及更细粒度的 Wrongness。Wrongness 是选择物体与真实 gaze target 的相对距离，范围从 0 到 1，比单纯 accuracy 更能反映错得有多远。

### 关键设计

**1. 真实照片中的受控 gaze/head 分离：让 incongruent 条件成为"模型是否真看眼睛"的因果探针**

互联网 gaze 图里头部方向和眼睛方向几乎总是一致，模型即使答对也分不清是读了眼睛还是抄了头部捷径。本文用真实拍摄而非合成图来制造系统性错位：Natural condition 让被拍摄者自然转头看向目标，Congruent condition 要求头和眼睛都朝同一物体，Incongruent condition 则固定头部朝向、只移动眼睛去看别处。这样从 congruent 切到 incongruent 时，场景中唯一改变的就是眼睛外观——如果模型只靠头部朝向，它必然在 incongruent 条件下系统性出错。之所以坚持真实照片而非合成图，是因为合成图可能缺少真实光照、阴影和人脸的统计规律，互联网图又无法控制头眼错位；受控实拍同时满足了现实性和因果可解释性。

**2. 从整体差距到机制分类的测试链：用一串统计检验把"准确率低"翻译成具体偏差类型**

只报告准确率无法区分"真做 gaze 推断"和"只用头部启发式"，单个统计结果也容易被其他因素解释掉。论文因此设计了层层递进的检验链：Test 1 先比较 VLM 与人类的总体差距；Head-Bias Test 2 含四个子测试，分别查模型是否更常选 HeadTarget、是否在只改眼睛时保持原选择、是否在 head/gaze 一致时表现更好、错误距离是否随 HeadTarget 与 GazeTarget 的距离增大；Test 3 检查模型对眼睛目标变化是否有响应；Test 4 进一步区分模型是在 gaze 和 head 之间随机摇摆，还是只能粗略锁定两者中间的区域。当多个独立检验共同指向头部朝向偏差时，结论才足够可信，也才能据此给出"补充 head-gaze 不一致训练样本"这类具体改进方向，而非泛泛地说模型不行。

**3. 严格的响应解析与 GLMM 分析：堵住 prompt、选项顺序和个别刺激带来的混淆**

行为评估极易被 prompt 模板、选项匹配方式和单张图片偶然带偏，平均准确率下结论很危险。响应解析上，模型既可直接输出选项字母也可自由作答，系统先用规则匹配，再用 Llama-3.1-70B 做语义匹配，仍解析不出时人工复核，确保统计的是模型的真实选择而非匹配失败。分析上则采用广义线性混合效应模型（GLMM），把 StimulusID、PromptID、Objects 等当作随机效应吸收掉刺激和模板的个体差异，把 View、Proximity、物体数量和 Condition 当作固定效应或交互项来估计真正关心的因果变量。这样得到的头部偏差结论是在剔除了这些噪声来源之后成立的，而不是某个 prompt 写法或选项顺序的副产品。

### 损失函数 / 训练策略
主实验中的 VLM 不训练，只做零样本多选评测。论文额外做了一个 proof-of-concept 微调实验：把 pilot 和 main study 共 2,260 张刺激按 7:1:2 划分训练、验证、测试集，微调 GazeLLE 50 个 epoch，用预测 gaze point 与目标物体位置的欧氏距离作为训练目标，学习率从 $1e-3$ 以 cosine schedule 衰减到 $1e-6$。

## 实验关键数据

### 主实验
主结果显示，人类在同样任务上明显强于所有 VLM 和专门模型。随着物体数量从 2 增到 4，所有模型准确率下降，但 VLM 多数接近随机基线。

| 方法 | 2 个物体 Acc | 3 个物体 Acc | 4 个物体 Acc | 说明 |
|--------|------|------|----------|------|
| Humans (n=59) | 94% | 88% | 76% | 人类同样看低分辨率刺激，仍远高于模型 |
| GazeLLE-DinoV2-ViTL14 | 78% | 67% | 47% | 专门 gaze 模型优于 VLM，但仍有差距 |
| Moondream2 (Hybrid) | 78% | 58% | 41% | 调用专门 gaze function |
| GPT-5.2 | 64% | 46% | 31% | 明显低于人类，4 物体接近随机 |
| GPT-4o-20240806 | 65% | 41% | 30% | 与 GPT-5.2 相近 |
| Qwen3-VL-30B | 59% | 39% | 28% | 明显受物体数量影响 |
| GLM-4.6V-Flash | 62% | 43% | 30% | 仍低于专门模型 |
| InternLM-XComposer2-VL-7B | 64% | 43% | 29% | 4 物体接近随机 |
| Guessing baseline | 50% | 33% | 25% | 多选随机基线 |

头部偏差测试进一步表明，模型不是单纯“看不懂物体”或“不会答多选题”，而是系统性把头朝向当作 gaze 线索。论文还排除了若干替代解释：提高分辨率到 896 或 1024 没有改善 GPT-5.2；提供从左到右物体名称仅给 GPT-5.2 和 GPT-4o 带来 2% 与 1.5% 小幅提升；GPT-5.2 完全按字母输出，说明结果不是选项解析流程导致的。

### 消融实验

| 方法 | Test 2.1 | Test 3 | Test 4 | 行为类型 | 说明 |
|------|---------|------|------|------|------|
| GazeLLE | - | + | - | Head-only | 专门模型也强依赖头部朝向 |
| Moondream2 | - | + | + | Head-dominant | 有 gaze 响应，但头部线索占主导 |
| GPT-5.2 | + | + | - | Head-only | 头部偏差最强，几乎忽略眼睛变化 |
| GPT-4o | - | - | + | Head-dominant | 更像在头部和眼睛间摇摆，但头部主导 |
| Qwen3-VL-30B | - | + | - | Head-only | 对眼睛变化响应不足 |
| GLM-4.6V-Flash | - | - | + | Unclear | 偏差存在但分类不稳定 |
| InternLM | - | - | + | Head-dominant | 头部线索主导 |

| GazeLLE 设置 | Pilot Acc | Congruent Acc | Incongruent Acc | Natural Acc | 说明 |
|------|---------|------|------|------|------|
| 原始 GazeLLE (ViT-B/14) | 63.80 | 62.82 | 15.65 | 65.22 | Incongruent 低于 chance，说明强头部捷径 |
| 微调后 | 85.77 ± 1.40 | 70.00 ± 5.01 | 34.15 ± 1.76 | 76.81 ± 2.90 | 对反头部捷径样本明显改善，但仍低于其他条件 |

### 关键发现
- 94/111 个 pre-pilot VLM 没有显著超过随机选择，说明 gaze target inference 不是当前 VLM 的稳定涌现能力。
- 所有主测 VLM 都表现出 Proximity effect：物体越靠近，Wrongness 越高，说明模型确实在使用视觉信息，而不是完全随机或只靠语言先验。
- 除 GPT-5.2 之外，多数主测 VLM 没有人类那样稳定的 View effect，说明它们对侧脸情况下眼睛外观变化不够敏感。
- GazeLLE 微调后 Incongruent accuracy 从 15.65% 提到 34.15%，支持“训练数据缺少 head-gaze 不一致案例”是主要原因之一。

## 亮点与洞察
- 这篇论文最有价值的地方是从 benchmark 走向机制诊断。它不满足于说明 VLM gaze 很差，而是通过一串实验把错误定位到 head orientation bias。
- 受控真实照片的选择很聪明。合成图可能引入生成伪影，互联网图又不能控制头眼错位；真实拍摄让结论更接近视觉系统问题本身。
- Wrongness 比 accuracy 更适合这个任务。选错相邻物体和选错最远物体不是同一种错误，Wrongness 可以反映 gaze 推断的空间精度。
- 微调实验给了很实际的改进方向：收集或构造足够多 head、body、eye 不一致的训练样本，可能比单纯增大模型规模更有效。

## 局限与展望
- Incongruent condition 是必要的因果探针，但现实中强烈头眼错位的频率有限；轻微错位更自然，极端错位更像压力测试。
- 主实验人物多样性有限，pilot 有两位女性，main study 有一位男性。虽然有利于控制变量，但无法覆盖不同脸型、眼妆、肤色、眼镜等因素。
- 任务是孤立的多选 gaze target inference，不包括上下文、对话历史和共同注意等真实交互线索。
- 专门模型微调使用的数据规模不大，只能说明数据可能缓解偏差，不能证明所有架构都能靠这种数据完全解决问题。
- 后续应测试更多开放环境、更多人物、多轮交互和 downstream reference resolution，看 gaze 能力是否能迁移到真实人机协作。

## 相关工作与启发
- **vs GazeFollow / in-the-wild gaze benchmarks**: 互联网图像中头部方向经常足够预测答案，模型可能靠捷径拿高分；本文用 controlled incongruent cases 暴露真实眼睛读取能力。
- **vs 专门 gaze estimation 模型**: GazeLLE 这类模型有监督训练目标，但也出现 head bias，说明问题不只是 VLM 架构，而与训练分布有关。
- **vs 常规 VLM 能力评测**: 常规评测多看平均准确率；本文展示了 hypothesis-driven behavioral probing 对理解模型机制更有帮助。
- **对后续工作的启发**: 多模态评测可以借鉴认知科学实验，把关键视觉线索系统性解耦，而不是只堆更大的自然图数据集。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用预注册受控实验定位 VLM gaze 错误机制，非常有辨识度。
- 实验充分度: ⭐⭐⭐⭐⭐ 有 pre-pilot、人类对照、主实验、多模型、多统计测试和微调验证。
- 写作质量: ⭐⭐⭐⭐ 实验逻辑清晰，但统计测试链较长，需要耐心跟随。
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 视觉理解评测、机器人指代理解和非语言社交线索建模都有很高价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](../../ACL2025/multimodal_vlm/speaking_beyond_language.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](../../ICCV2025/multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ICLR 2026\] Procedural Mistake Detection via Action Effect Modeling](../../ICLR2026/multimodal_vlm/procedural_mistake_detection_via_action_effect_modeling.md)
- [\[ACL 2026\] PROGRESSLM: Towards Progress Reasoning in Vision-Language Models](progresslm_towards_progress_reasoning_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
