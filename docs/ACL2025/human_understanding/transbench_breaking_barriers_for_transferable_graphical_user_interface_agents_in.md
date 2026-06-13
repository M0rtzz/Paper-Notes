---
title: >-
  [论文解读] TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments
description: >-
  [ACL 2025][人体理解][GUI grounding] 提出首个系统评估 GUI Agent **迁移性**（跨版本/跨平台/跨应用）的 benchmark TransBench，涵盖 81 个中文 App、1459 张截图、22K+ 标注指令，实验表明在旧版本上微调可有效迁移到新版本和其他平台，而跨平台迁移中 Android 数据的泛化性最强。
tags:
  - "ACL 2025"
  - "人体理解"
  - "GUI grounding"
  - "迁移性"
  - "跨版本"
  - "跨平台"
  - "跨应用"
  - "benchmark"
---

# TransBench: Breaking Barriers for Transferable Graphical User Interface Agents in Dynamic Digital Environments

**会议**: ACL 2025  
**arXiv**: [2505.17629](https://arxiv.org/abs/2505.17629)  
**代码**: TransBench (将公开)  
**领域**: 人机交互 / GUI Agent  
**关键词**: GUI grounding, 迁移性, 跨版本, 跨平台, 跨应用, benchmark

## 一句话总结
提出首个系统评估 GUI Agent **迁移性**（跨版本/跨平台/跨应用）的 benchmark TransBench，涵盖 81 个中文 App、1459 张截图、22K+ 标注指令，实验表明在旧版本上微调可有效迁移到新版本和其他平台，而跨平台迁移中 Android 数据的泛化性最强。

## 研究背景与动机

**领域现状**：GUI Agent 通过自然语言指令自主操作数字界面，其核心能力是 grounding——将语言意图映射到视觉界面元素。现有 benchmark 如 Mind2Web（Web）、GUI-Odyssey（移动）等只关注单一平台。

**现有痛点**：真实环境中 App 不断更新版本（界面布局变化）、用户跨平台使用（iOS/Android/Web）、任务跨应用（如购物+看评测视频）。现有 Agent 在这些动态场景下表现脆弱，缺少系统化评估迁移性的 benchmark。

**核心矛盾**：没有数据集同时涵盖版本差异、平台差异和应用差异，导致无法评估和提升 GUI Agent 的泛化能力。

**本文目标**：(1) 形式化定义 GUI Agent 迁移性的三个层次；(2) 构建全面的迁移性评估 benchmark；(3) 通过微调实验验证迁移性的规律。

**切入角度**：从实际使用场景出发，收集同一 App 不同版本、不同平台的截图，并标注统一的 grounding 指令。

**核心 idea**：在旧版本/单一平台数据上微调的模型，可以有效迁移到新版本和其他平台，多样化数据混合训练是提升全面迁移性的最佳策略。

## 方法详解

### 整体框架
TransBench 是一个 **benchmark 数据集**，不提出新模型。核心贡献在数据构建和系统化评估。

### 关键设计

1. **三层迁移性定义**:

    - **跨版本迁移**：同一 App 的旧版 Android → 新版 Android（74.6% 截图有显著界面变化）
    - **跨平台迁移**：Android ↔ iOS ↔ Web（同一 App 不同平台的界面差异）
    - **跨应用迁移**：相似功能 App（如京东→拼多多）和不同功能 App（如购物→金融）之间的泛化
    - 设计动机：这三个维度覆盖了真实使用中 GUI Agent 面临的主要泛化挑战

2. **数据收集三步流水线**:

    - **截图采集**：81 个中文 App，覆盖 15 个类别（购物/视频/社交/旅行/金融等），采集基础页面（首页/消息/个人中心）和领域特定页面，共 1459 张截图（Android 旧版 393 + 新版 432 + iOS 429 + Web 205）
    - **Bounding Box 标注**：先用 OmniParser 自动检测 GUI 元素（>65K 框），再由 4 名标注员用自研 GUILabeller 工具手动验证和修正，处理语义等价元素（如图标+文字触发相同操作）
    - **Grounding 指令生成**：用 Qwen2VL 提取框属性和页面摘要，再用 Qwen-plus 生成自然语言指令，最后人工验证达到 95.5% 准确率，共 22K+ 高质量指令

3. **评估指标**:

    - **准确率 (acc)**：预测点落在 GT bounding box 内即正确
    - **平均距离 (dis)**：预测点与 GT 中心的归一化欧氏距离（坐标归一化到 0-100），补充衡量精细度
    - 设计动机：距离指标可区分"刚好在框内"和"精准点中"，弥补准确率的粗粒度

### 评估模型
测试了 CogAgent、SeeClick、Aria-UI、OS-Atlas、UGround、Qwen2.5VL 六个模型，覆盖不同基座和训练策略。

### 迁移性微调实验
以 Aria-UI 为基础模型，设计多组数据划分进行微调：
- **跨版本**：旧 Android 训练 → 新 Android 测试
- **跨平台**：单平台训练 → 其他平台测试，及混合训练
- **跨应用**：前 7 个类别 App 训练 → 同类别其他 App 和不同类别 App 测试

## 实验关键数据

### 主实验（Standard Set，无微调）

| 模型 | Overall acc↑ | Android Old | Android New | iOS | Web |
|------|-------------|-------------|-------------|-----|-----|
| SeeClick | 39.90% | 46.86% | 46.42% | 43.57% | 15.37% |
| CogAgent | 72.16% | 76.04% | 75.70% | 68.61% | 66.69% |
| Aria-UI | 77.51% | 80.97% | 81.38% | 77.61% | 66.86% |
| OS-Atlas | 81.37% | 84.52% | 84.60% | 79.64% | 74.76% |
| UGround | 84.18% | 86.94% | 87.71% | 82.43% | 77.62% |
| **Qwen2.5VL** | **86.43%** | 88.87% | **90.29%** | **84.72%** | **79.79%** |

### 消融实验（Aria-UI 微调迁移性）

| 微调数据 | Android New | iOS | Web |
|---------|-------------|-----|-----|
| 无（Base） | 81.38% | 77.61% | 66.86% |
| Android Old (5K) | **88.36%** | 82.57% | 73.61% |
| iOS (5K) | 87.06% | 82.03% | 73.66% |
| Web (4K) | 84.87% | 80.62% | 66.49% |
| **General Mix (5K)** | 88.15% | **83.15%** | **76.54%** |

### 关键发现
- **跨版本迁移有效**：旧版 Android 微调后，新版准确率从 81.38%→88.36%（+6.98%），甚至对新版新增 UI 元素也从 80.15%→87.50%。这说明旧版数据中的 grounding 知识可迁移
- **Android 数据泛化性最强**：Android 数据微调后对 iOS 提升 +4.96%，超过 iOS 自身数据微调的 +4.42%；但 Android/iOS 数据难以迁移到 Web
- **混合数据最优**：General Mix 在所有平台均取得最佳或接近最佳表现，尤其 Web 从 66.86%→76.54%（+9.68%）
- **性能排序 Android > iOS > Web**：所有模型一致表现为 Android 最好、Web 最差，反映平台异质性
- **跨应用迁移约束较小**：同类别和不同类别 App 的迁移提升差距不大，说明版本/平台差异比应用差异更关键

## 亮点与洞察
- **首个迁移性 benchmark**：系统化定义并评估 GUI Agent 的三层迁移性，填补了重要空白。数据构建流水线（自动标注→人工验证→LLM 生成指令）可复用
- **距离指标的引入**：准确率之外增加归一化距离指标，发现 UGround 归一化坐标输出比 Qwen2.5VL 绝对坐标更鲁棒——这对 GUI Agent 设计有实际参考价值
- **"旧数据不过时"的发现**：在旧版本上微调不仅改善旧版表现，还提升新版和其他平台，历史数据的价值被低估

## 局限与展望
- 仅覆盖中文 App，英文/多语言 App 的迁移性未评估
- Grounding 只评估单步点击，未涉及多步交互和滚动/拖拽等复杂操作
- 微调实验只用 Aria-UI 一个模型，不同基座模型的迁移性规律可能不同
- Web 平台数据量偏少（205 截图 vs Android 825），可能影响 Web 相关结论的可靠性
- 未评估端到端任务完成率，只评估 grounding 准确率

## 相关工作与启发
- **vs Mind2Web**: 只评估 Web 环境，无跨版本/跨平台维度。TransBench 更全面
- **vs GUI-Odyssey**: 支持跨应用但不支持跨版本/跨平台
- **vs ScreenSpot**: 关注 grounding 但不关注迁移性
- **vs VisualAgentBench/WebHybrid**: 有跨平台但无跨版本评估

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统定义 GUI Agent 迁移性三维度，问题定义有价值
- 实验充分度: ⭐⭐⭐⭐ 多模型多维度评估，但微调只用一个模型
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设计系统，图表丰富
- 价值: ⭐⭐⭐⭐ 对 GUI Agent 实际部署有直接参考价值，benchmark 可推动社区研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](../../CVPR2025/human_understanding/d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](../../CVPR2025/human_understanding/x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2026\] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](../../CVPR2026/human_understanding/regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](../../CVPR2025/human_understanding/motionrefit_motion_editing.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)

</div>

<!-- RELATED:END -->
