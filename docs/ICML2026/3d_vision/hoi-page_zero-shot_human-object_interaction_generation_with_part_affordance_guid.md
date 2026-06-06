---
title: >-
  [论文解读] HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance
description: >-
  [ICML 2026][3D视觉][4D HOI] HOI-PAGE 让 LLM 先"想清楚"身体哪个部位该接触物体哪个部件，把推理结果写成一张「部件 affordance 图」(PAG)，再用它去驱动 3D 部件分割、视频扩散和优化求解…
tags:
  - "ICML 2026"
  - "3D视觉"
  - "4D HOI"
  - "部件级 affordance"
  - "affordance 图"
  - "视频扩散蒸馏"
  - "零样本生成"
---

# HOI-PAGE: Zero-Shot Human-Object Interaction Generation with Part Affordance Guidance

**会议**: ICML 2026  
**arXiv**: [2506.07209](https://arxiv.org/abs/2506.07209)  
**代码**: https://craigleili.github.io/projects/hoipage (项目主页)  
**领域**: 3D视觉 / 人-物交互生成 / 视频扩散  
**关键词**: 4D HOI、部件级 affordance、affordance 图、视频扩散蒸馏、零样本生成

## 一句话总结
HOI-PAGE 让 LLM 先"想清楚"身体哪个部位该接触物体哪个部件，把推理结果写成一张「部件 affordance 图」(PAG)，再用它去驱动 3D 部件分割、视频扩散和优化求解，从而在零样本、零 4D 训练数据的条件下生成可处理"多人单物 / 单人多物"等复杂场景的 4D 人-物交互序列。

## 研究背景与动机
**领域现状**：4D 人-物交互 (HOI) 生成的主流路线是扩散模型（HOI-Diff、CHOIS 等），把人体和物体的整体运动作为联合 token 一起去噪。这些方法依赖 BEHAVE、GRAB 等真值 4D 抓取/搬运数据训练，物体词表很窄，几乎只覆盖"单人单物"场景。

**现有痛点**：训练数据采集昂贵且稀缺，泛化到新物体（如吉他、割草机）时人体往往"飘在物体附近"，出现明显穿模、接触不上、动作和文本不匹配；多人或多物体场景几乎无法处理，因为接触关系数量随对数增长。

**核心矛盾**：HOI 的本质不是"人的质心靠近物体质心"，而是"特定身体部位 ↔ 特定物体功能部件"之间的精细接触（手握把柄、脚踩踏板）。整体姿态级建模丢掉了这层部件级语义，缺乏数据时学不出来，有数据时也只能记住分布而非推理。

**本文目标**：在不依赖任何 4D HOI 训练数据的前提下，从一句文本和若干 3D 物体出发，生成对"哪个手接哪个把手"这种部件级 affordance 显式建模、可扩展到多人多物的 4D 序列。

**切入角度**：LLM 已经具备关于日常交互的常识知识（迭代衣服时哪只手握底板、哪只手按底面）。如果把这种语言空间的"接触剧本"显式落地成图结构，再分别落到 3D 几何 / 视频 / 优化，就能把视觉-动作生成的负担分摊到"已有的强先验组件"上。

**核心 idea**：用 LLM 推理出**部件 affordance 图 (PAG)** 作为整套 pipeline 的脚本——节点是部件、边是接触约束，让 PAG 统一指挥 3D 物体部件分割、视频扩散提示、4D 优化的接触/穿模/平滑各项 loss。

## 方法详解
HOI-PAGE 是一个四段式 pipeline：**(1) LLM 把文本+物体清单翻译成 PAG；(2) 把 PAG 中的抽象物体部件锚定到 3D 几何上（部件分割）；(3) 用扩写后的 prompt 调视频扩散生成参考交互视频，并从视频里恢复 2D/3D 物体点云、深度、人体 SMPL-X 序列；(4) 用 PAG 当作约束做 part-affordance-guided 优化，把视频"提升"成可控的 4D 物体姿态序列**。最终输出每个物体的 $\{(R_t, t_t)\}_{t=1}^T$ 和每个人体的 SMPL-X 参数 $\{\Theta_t\}_{t=1}^T$，全程不需要任何 4D HOI 真值。

### 整体框架
输入：一组 3D 物体网格 $\{O\}$ + 一句文本 $\Gamma$（如"一个人在熨衣板上熨衣服"）。输出：$T=49$ 帧的人体 SMPL-X 序列 + 每个物体的 6DoF 位姿轨迹。中间四个阶段——LLM 出 PAG、SAM-2 / 开放词表检测做 3D 部件分割、CogVideoX 出参考视频 + 单目深度/人体恢复抽约束、最后做 600 步梯度下降优化物体轨迹——其中**只有最后一步是可学习/可调的，其余全部冻结**，所以叫"zero-shot"。

### 关键设计

1. **部件 affordance 图 (PAG)——用 LLM 写出来的"接触剧本"**：

    - 功能：把 HOI 的全部关键约束（哪个部件存在、哪只手接它、是不是持续接触、是不是相对静止、整体物体是否平移/转动）压成一张图 $G=(V,E)$，作为后续所有阶段的统一控制信号。
    - 核心思路：节点 $V=V_o \cup V_h$ 包含物体部件和 12 类人体部件（左右手、左右脚、髋等），并为每个物体/人加一个虚拟父节点 $v$ 携带两个运动状态 $(a_r, a_\tau)$ 表示是否旋转/平移。每条边 $e=(v_1,v_2)$ 表示一次部件级接触，带两个属性 $(a_c, a_s)$：$a_c$ 表示接触是否在整段视频中**持续**，$a_s$ 表示接触是否**相对静止**（手握住把手 vs. 熨斗在板上滑）。整张图由 LLM (DeepSeek 系) 用 in-context 推理一次性给出。作者实验过 VLM 直接看物体图，但发现 VLM 经常忽略视觉输入或幻觉，反而不如 LLM 稳。
    - 设计动机：HOI 不是端到端能学好的"姿态-物体联合分布"，而是一组组离散的部件级约束。把约束抽出来交给语言模型，相当于把"常识推理"和"几何执行"解耦，让没有 4D 训练数据的问题变成"几何优化怎么满足图约束"的问题。同时图结构天然可扩展，多人/多物只需要加节点和边，不需要改 pipeline。

2. **PAG 引导的视频扩散+约束抽取**：

    - 功能：把 PAG 翻译成扩写 prompt $\Gamma^+$ 喂给 CogVideoX 生成一段参考交互视频，再把视频"切片"成可被 3D 优化使用的几何/运动约束。
    - 核心思路：用 LLM 把 $\Gamma$ 结合每条边的接触类型扩写成更细致的 video prompt（如"右手始终紧握把手，左手按住底面"），同时用 FLUX 生成 5 张首帧候选并让 GPT-4.1 挑解剖最合理的一张作锚定；扩散出 49 帧视频后，按 PAG 中的部件名做开放词表检测+SAM-2 部件级分割，得到每帧每部件的 2D mask；再用单目深度估计 (Wang et al. 2024) 把 mask 反投影成 3D 部件点云序列；同时用 Shen et al. 2024 的方法从视频里抽出人体 SMPL-X 序列 $\{\Theta_t\}$。注意这里得到的人体动作是"孤立"的，物体姿态还没解出来。
    - 设计动机：让视频扩散承担"真实的人和物大致怎么动"这件最难的事，但只把视频当作"软参考"——它的几何精度不够，所以需要后面优化阶段用 PAG 的硬约束把物体姿态"校正"过来。这种分工绕开了"视频模型几何不准"和"几何模型语义不够"两难。

3. **PAG 引导的 4D 优化——把视频"提升"成 4D**：

    - 功能：求解每个物体的 $\{(R_t, t_t)\}_{t=1}^T$，使其同时拟合视频中的 2D/3D 观察、满足 PAG 中的每条接触/运动状态约束、不与人体穿模、并保持时序平滑。
    - 核心思路：四项 loss 加权求和 $L_{\text{total}} = \lambda_{\text{fit}} L_{\text{fit}} + \lambda_{\text{con}} L_{\text{con}} + \lambda_{\text{pen}} L_{\text{pen}} + \lambda_{\text{smo}} L_{\text{smo}}$。其中 $L_{\text{fit}}$ 是物体级 + 部件级、2D + 3D 的 Chamfer 距离；$L_{\text{con}} = L_{cc} + L_{cd}$ 中接触项 $L_{cc}$ 在 $a_c=\text{true}$ 时取所有帧最近邻平均，$a_c=\text{false}$ 时只取最小帧（一次性接触）；接触动力学项 $L_{cd}$ 在 $a_s=\text{true}$ 时惩罚相邻帧相对位移、$a_s=\text{false}$ 时用 $L_2(P_t^{v_2 \to v_1}, \tfrac{1}{2}(P_{t-1}^{v_2 \to v_1}+P_{t+1}^{v_2 \to v_1}))$ 鼓励平滑变化；$L_{\text{pen}}$ 用预计算 SDF 惩罚人体顶点穿入物体；$L_{\text{smo}}=L_r+L_\tau$ 按 PAG 中 $(a_r, a_\tau)$ 在"球面线性平滑"和"惩罚一切变化"两种模式间切换。优化跑 600 步梯度下降，单物体约 6 分钟、双物体约 10 分钟（A100），并用 4 个绕重力轴的初始旋转重复求解以避开 Chamfer 的局部极小。
    - 设计动机：所有 loss 都由 PAG 的属性"按边/按节点条件化"——这意味着同一套损失结构既能处理"持续紧握"也能处理"短暂触碰"，既能处理"物体静止"也能处理"物体随人移动"，无须为不同交互改实现。这是 PAG 设计真正发挥威力的地方。

### 损失函数 / 训练策略
全程**无任何模型训练**，只有最后一步对物体位姿做优化。LLM、视频扩散、深度估计、人体恢复、SAM-2 全部冻结。优化使用 4 次随机初始化中最优解；CogVideoX 50 去噪步、视频 49 帧；$\lambda$ 权重由经验设定（论文附录给出）。

## 实验关键数据

### 主实验
作者自建 Sketchfab 数据集（24 个日常 3D 物体 + 16 单人单物 prompt + 5 个多人/多物 prompt），与依赖 4D 真值训练的 HOI-Diff 和 CHOIS 比较。

| 指标 | HOI-Diff | CHOIS | HOI-PAGE |
|------|---------|-------|----------|
| VideoCLIP ↑ (语义) | 0.233 | 0.239 | **0.250** |
| 物体平滑度 ↓ | 0.035 | 0.009 | **0.006** |
| 物体动作多样性 ↑ | 0.72 | 0.49 | **0.80** |
| Non-collision ↑ | 0.98 | 0.98 | **0.99** |
| Contact ↑ | 0.76 | 0.64 | **0.92** |

感知评测中，HOI-PAGE 在二元偏好上以 91%–99% 击败两个 baseline；1-5 分制单评中 HOI-PAGE 拿到 ~4.0 分（真实感 3.97、文本匹配 4.07），而 baseline 普遍 ≤ 1.9。

### 消融实验
| 配置 | VideoCLIP ↑ | Smoothness ↓ | Diversity ↑ | Contact ↑ | 说明 |
|------|-----|-----|-----|-----|-----|
| Full | 0.290 | 0.004 | 0.83 | 0.76 | 完整 PAG 三项约束 |
| w/o 部件级拟合 (PF) | 0.290 | 0.004 | 0.81 | 0.76 | 物体姿态略糙 |
| w/o 部件级接触 (PC) | 0.289 | 0.011 | 0.71 | **0.26** | 接触崩塌、运动抖 |
| w/o 物体运动状态 (OMS) | 0.290 | 0.006 | 0.78 | 0.73 | 物体动作不该动时也动 |

### 关键发现
- **去掉 PC 后 Contact 从 0.76 暴跌到 0.26**：说明 LLM 推理出的接触图就是整个 pipeline 的命门，几何 loss 本身根本压不住"手要握住把手"这种语义约束。
- **HOI-Diff 的人体局部更平滑 (0.007)，但多样性最低 (0.35)**：暴露了纯监督模型的过拟合——记忆训练分布而非生成真实多样动作。
- **零数据反超有数据**：HOI-PAGE 在所有维度全面胜过两个用真值 4D 训练的 baseline，是这类工作里第一篇做到这一点的；对未见物体（割草机、吉他）尤其明显，因为 baseline 训练集根本没见过。
- **多场景扩展几乎免费**：单人单物 4.0 分、多人单物 4.17 分、单人多物 4.46 分——只增加 PAG 节点边，性能反而更稳。

## 亮点与洞察
- **用 LLM 当"导演"而不是"编剧"是个值得迁移的设计**：让 LLM 出结构化的约束图（节点+边+属性），而不是出长 prompt，能让视觉/几何模块严格执行约束，绕开了 LLM/VLM 的幻觉问题。可推广到机器人任务分解、场景生成、动作编辑。
- **PAG 的"按边条件化 loss"是优雅的统一**：同一份代码用 $a_c/a_s/a_r/a_\tau$ 四个布尔属性切换八种 loss 模式，避免了为每种交互写专用 pipeline，工程性很强。
- **视频扩散的几何弱、几何先验的语义弱，互相弥补**：HOI-PAGE 没有试图用一个模型解决所有事，而是把"语义剧本-视觉运动-几何精度"分给 LLM、视频扩散、SDF 优化各自最擅长的领域，是组合式零样本 pipeline 的范例。
- **多人/多物只改图不改算法**这件事在 HOI 领域非常稀有，传统监督模型几乎无法做到（要重训）。

## 局限与展望
- 优化依赖视频扩散质量，长视频 (>49 帧)、复杂背景、相机大幅运动时几何抽取会崩，约束跟着崩。
- 单目深度 + 视频反投影得到的点云本身误差很大，物体姿态最终由 Chamfer 拟合主导，对薄/小物体（叉子、笔）效果可疑（数据集只选了 24 个偏大的日常物体，回避了这点）。
- LLM 是否能正确给出"接触是否持续"等属性，强依赖 prompt 工程；论文用了 DeepSeek，作者也承认 VLM 仍不稳定。
- 单次优化 6-10 分钟、还要重复 4 次初始化，不适合实时；对长序列、剧烈动作（跳跃、翻滚）的扩展尚未验证。
- 评测全部在自建 Sketchfab 数据集上做，缺少在 BEHAVE/GRAB 等公开 benchmark 的对照实验。

## 相关工作与启发
- **vs HOI-Diff / CHOIS**：他们端到端学"人+物联合姿态分布"，依赖 4D 真值；HOI-PAGE 把语义剥到 LLM、几何留给优化，零样本反超。本质区别是"学习 vs. 组合"。
- **vs ZeroHSI / ZeroHOI / DAViD**：同样零样本、同样借助视频扩散，但都把人和物当作整体处理；HOI-PAGE 引入显式部件级图结构，是第一个能扩展到多人/多物的方案。
- **vs PiGraphs / iMapper**：PiGraphs 早期就用过"原型交互图"做静态人-场景合成；HOI-PAGE 把这个思想搬到 4D + 视频扩散时代，并让 LLM 来推理图结构，避免了对训练数据的依赖。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 LLM 推理变成显式图结构再驱动几何优化，思想清晰、和并发工作有明显差异化（部件级 vs. 整体级）。
- 实验充分度: ⭐⭐⭐ 对比和消融都做了，但数据集是作者自建，缺公开 benchmark；多人/多物只有定性 + 小规模表格。
- 写作质量: ⭐⭐⭐⭐ 方法分阶段清晰，PAG 的图示和 loss 条件化写得明白，公式编号干净。
- 价值: ⭐⭐⭐⭐ "零 4D 数据 + 可扩展多人多物" 这两点对 HOI 生成社区有真实推动；PAG 的设计模式可迁移到机器人和场景生成。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](../../AAAI2026/3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)
- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](../../CVPR2026/3d_vision/cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](../../ECCV2024/3d_vision/zero-shot_multi-object_scene_completion.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)

</div>

<!-- RELATED:END -->
