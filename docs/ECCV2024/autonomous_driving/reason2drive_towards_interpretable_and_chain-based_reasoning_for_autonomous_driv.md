---
title: >-
  [论文解读] Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][视觉语言模型] Reason2Drive 构建了一个包含 60 万+视频-文本对的大规模自动驾驶推理数据集，将驾驶决策拆解为感知→预测→推理的链式过程，并提出 ADRScore 评估指标和带 prior tokenizer + instructed vision decoder 的 VLM 框架，显著提升了驾驶场景的链式推理准确性。
tags:
  - ECCV 2024
  - 自动驾驶
  - 视觉语言模型
  - 可解释自动驾驶
  - Chain-of-Thought
  - 数据集
  - 推理评估
---

# Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2312.03661](https://arxiv.org/abs/2312.03661)  
**代码**: https://github.com/fudan-zvg/reason2drive  
**领域**: 自动驾驶  
**关键词**: 视觉语言模型, 可解释自动驾驶, Chain-of-Thought, 数据集, 推理评估

## 一句话总结
Reason2Drive 构建了一个包含 60 万+视频-文本对的大规模自动驾驶推理数据集，将驾驶决策拆解为感知→预测→推理的链式过程，并提出 ADRScore 评估指标和带 prior tokenizer + instructed vision decoder 的 VLM 框架，显著提升了驾驶场景的链式推理准确性。

## 研究背景与动机

**领域现状**：端到端自动驾驶方法通常将系统视为黑盒，直接从传感器输入到控制信号，难以解释决策逻辑。而大型视觉语言模型（VLM）凭借广泛的世界知识和推理能力，有望实现可解释的自动驾驶决策。

**现有痛点**：现有驾驶相关语言数据集（Talk2Car, NuScenesQA 等）大多是简单 QA 任务，只涉及感知层面的对错判断（yes/no 或 multiple-choice），缺少描述完整决策链的推理标注。同时，BLEU、CIDEr 等文本生成指标无法衡量推理步骤与最终结论之间的因果关系。

**核心矛盾**：自动驾驶决策是"感知→预测→推理"的多步过程，但既缺乏标注这一链条的数据集，也缺乏评估这种链式推理质量的指标。

**本文目标**  
   - 构建一个包含感知、预测、推理三步链式标注的大规模驾驶数据集  
   - 设计能准确评估链式推理质量的指标（而非仅评估文本生成质量）  
   - 增强 VLM 利用目标级感知先验的能力，提升驾驶推理精度

**切入角度**：观察到现有 VLM 难以有效利用目标级感知信息（位置、运动等），原因是缺少专用的 tokenizer 和解码器只有语言模型，无法输出精确感知结果。

**核心 idea**：用自动化标注构建大规模感知-预测-推理链数据集，配合 ADRScore 评估指标和感知增强 VLM 框架，实现可解释的自动驾驶推理。

## 方法详解

### 整体框架
系统由三部分组成：(1) 数据集构建管线：从 nuScenes/Waymo/ONCE 的标注中自动提取目标级信息，通过模板 + GPT-4 生成感知/预测/推理的 QA 对；(2) ADRScore 评估指标：通过推理对齐、冗余度和缺失步骤三个维度衡量链式推理质量；(3) 增强 VLM 框架：在 InstructBLIP 基础上加入 prior tokenizer 编码感知先验和 instructed vision decoder 输出精确感知结果。

### 关键设计

1. **Reason2Drive 数据集构建**:

    - 功能：从多源驾驶数据集自动生成 63 万+ QA 对，覆盖感知/预测/推理三级任务
    - 核心思路：先将各数据集的标注（bbox, tracklet, map 等）解析为按关键帧组织的目标级数据库，包含位置、类别、属性、运动等。然后用手工模板将结构化数据转为对话格式，分为目标级（单个物体属性/运动）和场景级（全局风险/决策）。最后用 GPT-4 进行验证和丰富，增加文本多样性
    - 设计动机：自动化标注管线使得数据集可扩展到新的源数据集。三级任务分解（感知24.6万+预测21.6万+推理17.1万）确保模型必须掌握完整决策链

2. **ADRScore 评估指标**:

    - 功能：衡量生成的推理链与 ground-truth 推理链的对齐质量
    - 核心思路：将生成的推理步骤 $\vec{h}$ 和参考步骤 $\vec{r}$ 逐步计算语义相似度，定义三个子指标：(1) 推理对齐 $RA = \frac{1}{N}\sum_i \max_j \cos(h_i, r_j)$；(2) 冗余度 $RD = \min_i \max_j \cos(h_i, r_j)$，惩罚不必要的步骤；(3) 缺失步骤 $MS = \min_i \max_j \cos(r_i, h_j)$，检测遗漏。最终 $ADRScore = \frac{1}{3}(RA + RD + MS)$。还有 ADRScore-S 变体，将文本语义相似度替换为几何真实误差，更严格评估空间推理
    - 设计动机：BLEU/CIDEr 只评估文本整体相似度，无法区分"推理正确但表述不同"和"推理错误但表述流畅"的情况

3. **Prior Tokenizer**:

    - 功能：将感知先验（检测框、运动轨迹）编码为 VLM 可理解的 token
    - 核心思路：用 RoIAlign 从视觉特征图中提取目标区域特征 $f_r$（7×7），加上位置编码 $E(P)$，过两层 MLP 得到先验 token：$f_p = F_p(f_r + E(P))$。该 token 与视觉 token 一起送入 Q-former 对齐到文本空间
    - 设计动机：直接将坐标作为文本输入 LLM 会导致信息损失（文本无法完整表达复杂的空间关系），用专门的 tokenizer 编码视觉先验更精确

4. **Instructed Vision Decoder**:

    - 功能：使 VLM 能输出精确的感知预测（位置坐标、运动轨迹）
    - 核心思路：在 LLM 词表中新增 `<LOC>` 和 `<MOT>` 特殊 token。当 LLM 需要生成感知预测时输出这些 token，提取对应位置的 last-layer 特征经 MLP 投影为 $f_h$，再送入 transformer decoder 解码器结合视觉特征 $f_v$ 预测坐标和轨迹：$\hat{P} = D(f_v, f_h)$
    - 设计动机：纯语言模型输出坐标不精确且缺乏视觉上下文，用 vision decoder 将视觉特征与文本指令结合可生成更准确的感知结果

### 损失函数 / 训练策略
- 总损失：$\mathcal{L} = \mathcal{L}_{txt} + \lambda_{per}\mathcal{L}_{per}$
- $\mathcal{L}_{txt}$：自回归交叉熵（文本生成），$\mathcal{L}_{per}$：BCE + MSE（位置和轨迹预测）
- 两阶段训练：Stage 1 预训练 prior tokenizer + Q-former + vision decoder（冻结 LLM 和 ViT）；Stage 2 用 LoRA 微调 LLM 获得推理能力
- 输入为 5 帧裁剪图像（224×224），batch=8，8× V100

## 实验关键数据

### 主实验：Reason2Drive 验证集推理性能

| 方法 | LLM | ADRScore | ADRScore-S | BLEU@4 | CIDEr |
|------|-----|----------|------------|--------|-------|
| BLIP-2 | OPT-2.7B | 0.296 | 0.162 | 0.361 | 0.174 |
| InstructBLIP | FlanT5-XL | 0.329 | 0.187 | 0.376 | 0.196 |
| InstructBLIP | Vicuna-7B | 0.351 | 0.214 | 0.408 | 0.211 |
| MiniGPT-4 | Vicuna-7B | 0.338 | 0.203 | 0.396 | 0.219 |
| **Ours** | Vicuna-7B | **0.463** | **0.432** | **0.457** | **0.298** |

### 消融实验：训练任务组合

| 感知 | 预测 | 推理 | ADRScore | ADRScore-S |
|------|------|------|----------|------------|
| ✓ | | | 0.282 | 0.253 |
| ✓ | ✓ | | 0.297 | 0.264 |
| | | ✓ | 0.351 | 0.323 |
| ✓ | | ✓ | 0.407 | 0.364 |
| ✓ | ✓ | ✓ | **0.463** | **0.432** |

### 消融实验：视觉输入与感知先验

| 图像级 | 视频级 | 区域级 | 位置先验 | ADRScore | ADRScore-S |
|--------|--------|--------|----------|----------|------------|
| ✓ | | | | 0.414 | 0.379 |
| | ✓ | | | 0.431 | 0.394 |
| | ✓ | ✓ | | 0.447 | 0.418 |
| | ✓ | ✓ | ✓ | **0.463** | **0.432** |

### 关键发现
- 三级任务联合训练（感知+预测+推理）比单独训练推理任务提升 31.9% ADRScore（0.351→0.463），说明感知和预测是推理的重要基础
- Prior tokenizer 引入区域级特征和位置先验贡献了 +3.2% ADRScore 提升（从 0.431 到 0.463）
- 本文方法的感知预测质量显著优于 baseline：bbox 准确率 0.806 vs Kosmos-2 的 0.745，轨迹 ADE 1.875 vs MiniGPT-4 的 2.334
- ADRScore 与 CIDEr 等传统指标的排序不完全一致，说明传统指标确实不能反映推理链质量

## 亮点与洞察
- **链式推理标注**：将自动驾驶决策显式拆解为感知→预测→推理三步，每步都有标注，是该方向首个完整链式推理数据集。这种分解思路可以迁移到机器人决策、医疗诊断等需要可解释推理的领域
- **ADRScore 指标**：同时考虑对齐度、冗余和缺失，比传统文本指标更能反映推理质量。特别是 ADRScore-S 用几何误差替代语义相似度，对空间推理有更严格的评估
- **Prior Tokenizer + Vision Decoder 双向增强**：输入端用 prior tokenizer 引入感知先验，输出端用 vision decoder 输出精确预测，形成闭环

## 局限与展望
- 数据集的推理标注是模板+GPT-4 生成的，可能存在模式单一和噪声问题
- 只使用单目前视图像（224×224 裁剪），丢失了自动驾驶中重要的多视角和 3D 信息
- 当前推理仅限语言层面，未与实际控制信号（规划轨迹）deep integration
- 可以考虑引入多帧更长时序信息，提升预测步骤的准确性

## 相关工作与启发
- **vs DriveLM**：DriveLM 也做驾驶 VQA 但主要关注感知信息，Reason2Drive 更强调完整的感知→预测→推理链，数据规模更大（60万 vs ~30万），且提出了专门的推理评估指标
- **vs DriveGPT4**：DriveGPT4 直接用 GPT-4 做驾驶解释，但没有提供大规模微调数据集和定量评估框架。Reason2Drive 提供了系统性的数据集+指标+模型改进方案
- **vs NuScenesQA**：NuScenesQA 是封闭式 QA（yes/no 或选择题），Reason2Drive 是开放式链式推理，复杂度高得多

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个大规模链式推理驾驶数据集，ADRScore 指标设计新颖
- 实验充分度: ⭐⭐⭐⭐ 多维度消融验证了数据集、指标和模型的有效性
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，数据集构建流程和指标定义描述详尽
- 价值: ⭐⭐⭐⭐ 为 VLM 在自动驾驶中的可解释推理提供了重要基础设施

<!-- RELATED:START -->

## 相关论文

- [Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)
- [PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation](panovos_bridging_non-panoramic_and_panoramic_views_with_transformer_for_video_se.md)
- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)

<!-- RELATED:END -->
