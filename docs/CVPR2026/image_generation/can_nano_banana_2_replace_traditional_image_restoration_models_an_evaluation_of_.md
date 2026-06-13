---
title: >-
  [论文解读] Can Nano Banana 2 Replace Traditional Image Restoration Models? An Evaluation of Its Performance on Image Restoration Tasks
description: >-
  [CVPR 2026][图像生成][通用图像编辑模型] 这是一篇评测研究：作者系统测试了通用图像编辑模型 Nano Banana 2 能否充当统一的图像恢复器，发现它在感知质量和用户偏好上很强、在 FR 指标上也有竞争力，但存在「感知好看 ≠ 保真还原」的根本缺口——容易过度生成、幻觉纹理、语义漂移，而这一缺口恰恰被现有 IQA 指标和用户研究所掩盖。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "通用图像编辑模型"
  - "图像恢复(IR)"
  - "提示学习"
  - "感知-保真权衡"
  - "IQA 评测"
---

# Can Nano Banana 2 Replace Traditional Image Restoration Models? An Evaluation of Its Performance on Image Restoration Tasks

**会议**: CVPR 2026  
**arXiv**: [2604.03061](https://arxiv.org/abs/2604.03061)  
**代码**: https://github.com/yxyuanxiao/NanoBanana2TestOnIR (有)  
**领域**: 图像生成 / 图像恢复 / 评测研究  
**关键词**: 通用图像编辑模型, 图像恢复(IR), Prompt 设计, 感知-保真权衡, IQA 评测

## 一句话总结
这是一篇评测研究：作者系统测试了通用图像编辑模型 Nano Banana 2 能否充当统一的图像恢复器，发现它在感知质量和用户偏好上很强、在 FR 指标上也有竞争力，但存在「感知好看 ≠ 保真还原」的根本缺口——容易过度生成、幻觉纹理、语义漂移，而这一缺口恰恰被现有 IQA 指标和用户研究所掩盖。

## 研究背景与动机
**领域现状**：图像恢复（去噪、去模糊、超分、伪影去除）长期是一个「专用范式」——每类退化训一个专门模型，针对预定义退化类型、在相对封闭的分布上优化。近年扩散式恢复模型（DiffBIR、TSD-SR 等）借助强生成先验把感知质量推得很高，但仍困在「为某一族退化定制 pipeline」的框架里。

**现有痛点**：专用模型在真实世界里失效——真实退化是混合的、空间非均匀的、且事先未知（mixed / non-uniform / unknown a priori）。每遇到一种新退化就要重新设计/重训，缺乏统一性。

**核心矛盾**：能不能用一个通用图像编辑模型当「万能恢复器」？它的吸引力在于单框架处理多退化、复用大规模语义/结构先验补全严重 ill-posed 的区域、支持指令驱动控制。但通用生成先验带来一个根本张力——**感知合理性（perceptual plausibility）与像素级保真（pixel-level fidelity）之间的 trade-off**：生成模型能产出更锐利、更干净、更讨喜的图，却可能严重偏离 ground truth（幻觉纹理、语义漂移、结构不一致、颜色偏移）。

**本文目标**：以新发布的高关注度通用编辑模型 Nano Banana 2 为代表，系统回答——它能否作为可靠的统一恢复器？怎样的 prompt 影响恢复表现、它在 prompt 扰动下稳不稳定、在多样场景/退化上表现如何、与 SOTA IR 模型相比处于什么位置。

**切入角度**：不提新方法，而是设计一套**评测协议**去丈量「通用图像编辑」与「可靠图像恢复」之间的真实边界，为未来「生成模型的感知力 + 专用方法的保真保证」的混合框架提供经验证据。

**核心 idea**：用「Prompt 分类法 × FR/NR 双指标 × 稳定性检验 × 用户研究」四件套，把 Nano Banana 2 在 IR 上的能力边界量化出来，并揭示现有评测体系无法捕捉保真缺口这一盲点。

## 方法详解

### 整体框架
这是一篇评测研究，没有提出新模型，"方法"即一套针对 Nano Banana 2 的图像恢复能力评测协议。整体逻辑是：先在一个覆盖 13 类场景、7 类退化的子集数据上，用 12 种结构化 prompt 探究「prompt 怎么写最有效」；再用 FR（PSNR/SSIM/LPIPS）+ NR（MUSIQ/MANIQA/CLIP-IQA）双类指标 + 重复推理稳定性检验 + 人类用户研究，从多个互补维度刻画 Nano Banana 2 与 4 个 SOTA IR 模型（HYPIR、TSD-SR、PiSA-SR、DiffBIR）的差距；最终落到一个核心诊断——感知质量和保真度之间存在系统性缺口，且被现有评测掩盖。

数据集取自前作 [42]，选了 1024×1024 的子集，覆盖 13 个场景类别（航拍、动物毛发、建筑、卡通、大/中/小人脸、人群、文字、手脚、树叶、手绘、织物纹理）和 7 种退化（散焦模糊、运动模糊、数字变焦、老电影、黑白老照片、彩色老照片、监控），刻意挑了头发/文字/小目标这类细粒度难恢复的内容；infidelity 案例分析用的是 35 张测试图。

### 关键设计

**1. Prompt 分类法：把"提示词怎么写"做成可控变量**

通用编辑模型靠文本指令驱动，恢复结果对 prompt 极度敏感，但"prompt 怎么影响恢复"此前没人系统量化。作者把 prompt 拆成两个正交因子做受控实验：**长度**（短 26–30 词 vs 长 34–47 词）和**是否显式加保真约束**（带 Fidelity 指令 vs 不带），交叉出 4 组（L / LF / S / SF），每组 3 条共 12 条 prompt（见原文 Table 1）。保真指令的典型措辞是"preserve the original scene structure / keep their identity unchanged"。这种设计的价值在于：它把"prompt engineering"这个玄学变量变成了可对照分析的实验轴，从而能干净地回答"长 prompt 和保真约束各自带来什么"。结论是长 prompt 在全部 6 个 IQA 指标上一致优于短 prompt（更全面的引导），而保真约束对短 prompt 提升最明显——SF 组拿到了所有 prompt 类型里最好的 FR 结果；全数据集最终评测统一采用表现最好的 LF3

**2. FR/NR 双指标 + 案例级 infidelity 度量：刻画"感知 vs 保真"的双面**

单看任一类指标都会被误导：NR 指标（MUSIQ/MANIQA/CLIP-IQA）只奖励"看起来好看"，FR 指标（PSNR/SSIM/LPIPS）只衡量"和 GT 像不像"。作者同时报告两类，正是要把生成模型"感知好看但偏离原图"的双面性逼出来。最尖锐的证据是 prompt S3：它在全部 NR 指标上最高、却在 FR 指标上最低——说明不带保真约束时模型会生成讨喜纹理和增强细节，视觉上诱人但偏离 GT 结构，拉低 PSNR/SSIM。为补 IQA 指标的盲区，作者额外定义了 **infidelity（保真失效）的案例级度量**：把"与输入的严重语义偏离"（插入不存在的物体、显著形状扭曲、图像语义不一致）人工计数；在 12 条 prompt × 35 张图上，不带保真约束平均每条 prompt 出现 2 个 infidelity 案例，加了保真约束降到 0.5 个——直接量化了保真指令对语义篡改的抑制作用

**3. 稳定性检验（Friedman test）：量化生成随机性带来的不一致**

生成模型每次采样都有随机性，恢复结果"跑一次和跑一次不一样"是部署隐患。作者选 L3/LF1/S1/SF2 四条代表性 prompt，对每张输入在相同条件下独立跑 3 次，对每次的 6 个 IQA 指标做 **Friedman 非参数检验**（零假设：同一输入+prompt 的重复运行之间无显著差异）。在 4 组 prompt × 6 指标 = 24 个统计检验里，只有 2 个出现显著差异（$p<0.05$），其余 $p$ 值远大于 0.05——总体上模型在重复推理下相当稳定，FR 和 NR 两侧都稳。但作者诚实指出存在离群：在结构复杂或退化严重的难样本上，重复运行会产生主物体颜色偏移、尺度变化、结构改变等大幅波动（Fig. 4），说明在困难场景下控制生成随机性仍是可靠性的关键

**4. 人类用户研究：揭示"用户偏好"本身也测不准保真**

IQA 指标在感知评估上有局限，作者用人类视角补一刀：招募 20 名参与者，对 5 个模型（Nano Banana 2 + 4 个 baseline）在 20 张跨场景/退化图上的恢复结果按 0（最差）到 5（最好）打分。结果 Nano Banana 2 平均分最高、分布集中，说明人类普遍偏好它的输出。但这条设计的真正洞见是它的**反讽**：仔细看会发现用户偏好强烈偏向"视觉丰富、锐利"的结果，经常偏爱被增强/合成出来的细节，而几乎不惩罚与退化输入的不一致（颜色偏移、背景改动、过度生成纹理）。也就是说用户研究主要反映的是"感知吸引力"而非"恢复保真度"——这与 IQA 指标的盲区一起，共同构成了"现有评测体系系统性掩盖保真缺口"的核心论断

## 实验关键数据

### Prompt 设计的影响（Table 2，群均值）

| Prompt 组 | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | CLIP-IQA |
|-----------|------|------|--------|-------|--------|----------|
| L（长，无保真） | 22.950 | 0.668 | 0.222 | 69.352 | 0.396 | 0.676 |
| LF（长+保真） | 22.856 | 0.666 | 0.221 | 67.974 | 0.379 | 0.664 |
| S（短，无保真） | 22.516 | 0.662 | 0.228 | 69.340 | 0.394 | 0.675 |
| SF（短+保真） | **23.151** | 0.669 | **0.216** | 67.346 | 0.374 | 0.657 |
| 长合计 (L+LF) | 22.903 | 0.667 | 0.221 | 68.663 | 0.387 | 0.670 |
| 短合计 (S+SF) | 22.834 | 0.665 | 0.222 | 68.343 | 0.384 | 0.666 |

- 长 prompt 在全部 6 个 IQA 指标上一致优于短 prompt；SF（短+保真）拿到最高 PSNR 23.151 与最低 LPIPS 0.216（最好 FR）。
- 反例 prompt S3：NR 全部最高（MUSIQ 70.444 / MANIQA 0.408 / CLIP-IQA 0.686），FR 却最低（PSNR 21.975），是"感知-保真权衡"的最直接证据。

### 与 SOTA 在难场景上的对比（Table 4，节选）

| 场景 | 方法 | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | CLIP-IQA |
|------|------|------|------|--------|-------|--------|----------|
| Small Faces | **Nano Banana 2** | **23.809** | **0.735** | **0.146** | 72.989 | 0.427 | 0.662 |
| Small Faces | TSD-SR | 21.375 | 0.654 | 0.183 | **76.590** | **0.593** | **0.737** |
| Hands/Feet | **Nano Banana 2** | **26.456** | **0.763** | **0.148** | 70.032 | 0.431 | 0.683 |
| Text | Nano Banana 2 | 20.347 | 0.659 | 0.230 | 69.136 | 0.455 | 0.716 |
| Text | PiSA-SR | **21.551** | 0.642 | 0.247 | 70.474 | 0.482 | 0.597 |
| Surveillance | **Nano Banana 2** | / | / | / | 65.514 | 0.346 | 0.555 |

- Nano Banana 2 在 Small Faces / Hands/Feet 等场景拿下最好的 FR 指标（PSNR/SSIM/LPIPS），但 NR 指标常被专用扩散模型（TSD-SR、DiffBIR）反超——印证它"保真侧不错、但极致感知锐度不及专用生成模型"。

### 整体对比与稳定性（Table 5 + Table 3）

| 方法 | PSNR | SSIM | LPIPS↓ | MUSIQ | MANIQA | CLIP-IQA |
|------|------|------|--------|-------|--------|----------|
| HYPIR | 21.307 | 0.622 | 0.240 | 67.103 | 0.407 | 0.582 |
| PiSA-SR | 22.744 | 0.633 | 0.237 | 71.276 | 0.468 | 0.681 |
| TSD-SR | 21.439 | 0.599 | 0.232 | **72.144** | **0.479** | **0.708** |
| DiffBIR | 21.856 | 0.602 | 0.273 | 69.974 | 0.478 | 0.685 |
| **Nano Banana 2** | 22.541 | **0.649** | **0.222** | 68.841 | 0.394 | 0.676 |

- 整体上 Nano Banana 2 拿到最好的 SSIM(0.649) 与 LPIPS(0.222)、PSNR 第二（22.541），FR 侧有竞争力；但 NR 指标全面落后专用方法。
- 稳定性（Table 3 Friedman 检验）：24 个检验仅 2 个显著（S1 的 CLIP-IQA $p=0.019$、LF1 的 MUSIQ $p=0.044$），多数 $p$ 远大于 0.05，重复推理总体稳定。

### 关键发现
- **感知-保真权衡贯穿始终**：无保真约束 → NR 高、FR 低；有保真约束 → FR 高、NR/感知锐度降。最有效的引导是"简洁 prompt + 显式保真指令"。
- **保真约束能压住但压不死语义篡改**：infidelity 案例从平均 2 个/prompt 降到 0.5 个，但即便加了约束仍偶发失效（Fig. 3），说明纯靠 prompt 不足以完全约束生成过程。
- **现有评测系统性失真**：FR 指标对幻觉细节/局部不一致/颜色偏移不敏感；用户研究偏向"视觉丰富锐利"、几乎不惩罚与输入的不一致——两者共同掩盖了保真缺口。
- **难样本上随机性放大**：小脸/人群/重退化输入下，重复运行出现颜色与结构的大幅漂移。

## 亮点与洞察
- **把"prompt engineering"做成受控变量**：长度 × 保真约束的 2×2 正交设计，干净拆解出两个因子各自的贡献，方法学上可直接迁移到任何"指令驱动 + 生成"的评测（如指令式编辑、可控生成的可靠性研究）。
- **案例级 infidelity 度量补 IQA 盲区**：当标准指标测不出"语义篡改"时，人工定义并计数严重语义偏离（插物体/形变/语义不一致），是一个简单但有效的"指标补丁"思路——2 → 0.5 的对比比任何 IQA 数字都更有说服力。
- **最"啊哈"的点是对评测体系的反身批判**：论文不只测模型，还顺手证明了"用户研究本身也测不准保真"——人偏爱锐利好看的图、不惩罚不一致，这把"感知-保真缺口被掩盖"从猜想坐实成了证据，指向 fidelity-aware evaluation 这一未来方向。
- **结论是"有条件肯定"而非站队**：通用模型可作统一 IR 求解器，但当前强在感知增强、弱在保真还原，需要更强可控性 + 保真感知的评测协议——这是给社区的诚实定位。

## 局限与展望
- **作者承认的局限**：① 即便有保真 prompt，infidelity 与 over-generation（过密头发、放大纹理、不真实细节）在复杂场景仍会发生；② 模型对 prompt 高度敏感，常需 prompt engineering 甚至多轮迭代细化，恢复过程尚非完全确定性；③ 难样本上生成随机性放大，可靠性不足。
- **自己发现的局限**：评测规模偏小——数据集是前作子集、infidelity 分析仅 35 张图、用户研究 20 人 × 20 图；被测的"通用模型"只有 Nano Banana 2 一个，缺乏与其他通用编辑模型（如 GPT-Image、Qwen-Image-Edit 等）的横向对比，结论的普适性有待扩展。Nano Banana 2 本身是闭源/不可控的黑盒，无法做内部机制分析。
- **改进思路**：① 设计 **fidelity-aware 的 IQA 指标**，专门惩罚幻觉纹理/语义漂移/颜色偏移；② 把生成模型的感知力与专用方法的保真保证做混合框架（如保真正则、参考引导）；③ 控制生成随机性（固定种子/一致性约束）提升难场景可靠性。

## 相关工作与启发
- **vs 专用扩散恢复模型（DiffBIR / TSD-SR / PiSA-SR / HYPIR）**：它们是"恢复专用范式"，针对预定义退化定制 pipeline、保守但保真度好（NR 指标常更高、输出更贴合输入低层细节）；Nano Banana 2 是"通用编辑范式"，单框架处理多退化、FR 指标（SSIM/LPIPS）有竞争力，但倾向过度增强、保真度不稳。本文证明二者各有所长，未来应融合。
- **vs 通用图像编辑模型评测（一般创意/指令场景）**：以往评测通用编辑模型多看创意合成与指令遵循；本文把镜头对准 IR 这一对保真要求极高的低层视觉任务，揭示通用模型在"语义重建 vs 信号恢复"上的边界。
- **vs 感知-失真权衡理论 [Blau & Michaeli, 2018]**：本文是该理论在"通用生成模型做 IR"场景下的一次实证落地，并进一步指出现有评测协议（IQA + 用户研究）无法可靠地坐标化这条权衡曲线。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不提新模型，但"把 prompt 做成受控变量 + 案例级 infidelity 度量 + 揭示评测体系盲区"的评测视角有价值，问题切得准且及时。
- 实验充分度: ⭐⭐⭐ 多维度（FR/NR/稳定性/用户研究/分场景）覆盖到位，但规模偏小（35 图 infidelity、20 人用户研究）、被测通用模型仅 1 个，横向对比不足。
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、对自身结论保持克制（"有条件肯定"），对指标/用户研究的局限诚实交代。
- 价值: ⭐⭐⭐⭐ 给"通用模型能否替代专用 IR"提供了经验证据，并把"fidelity-aware evaluation"明确为后续方向，对社区有定位意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro](banana100_breaking_nr-iqa_metrics_by_100_iterative_image_replications_with_nano_.md)
- [\[CVPR 2026\] 4KLSDB: A Large-Scale Dataset for 4K Image Restoration and Generation](4klsdb_a_large-scale_dataset_for_4k_image_restoration_and_generation.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](../../ICML2026/image_generation/image_restoration_via_diffusion_models_with_dynamic_resolution.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] FAGER: Factually Grounded Evaluation and Refinement of Text-to-Image Models](fager_factually_grounded_evaluation_and_refinement_of_text-to-image_models.md)

</div>

<!-- RELATED:END -->
