---
title: >-
  [论文解读] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos
description: >-
  [NeurIPS 2025][3D视觉][动态新视角合成] 提出 CogNVS，将动态场景新视角合成分解为三阶段管线——3D 重建（获取可见像素）→ 视频扩散修复（生成遮挡区域）→ 测试时微调（适应目标视频的分布），用纯 2D 视频自监督训练修复模型，实现零样本泛化到新测试视频。 领域现状：单目视频的动态新视角合成是极具挑战…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "动态新视角合成"
  - "单目视频"
  - "视频修复"
  - "测试时微调"
  - "扩散模型"
---

# Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos

**会议**: NeurIPS 2025  
**arXiv**: [2507.12646](https://arxiv.org/abs/2507.12646)  
**代码**: [https://cog-nvs.github.io/](https://cog-nvs.github.io/)  
**领域**: 3D视觉  
**关键词**: 动态新视角合成, 单目视频, 视频修复, 测试时微调, 扩散模型

## 一句话总结
提出 CogNVS，将动态场景新视角合成分解为三阶段管线——3D 重建（获取可见像素）→ 视频扩散修复（生成遮挡区域）→ 测试时微调（适应目标视频的分布），用纯 2D 视频自监督训练修复模型，实现零样本泛化到新测试视频。

## 研究背景与动机

**领域现状**：单目视频的动态新视角合成是极具挑战的问题。两条路线：(i) 测试时优化 4D 表示（如 4DGS、Shape-of-Motion）——几何准确但耗时数小时，且新视角远离训练视角时失败；(ii) 大规模前馈视频模型（GCD、TrajectoryCrafter）——速度快但 3D 一致性差

**现有痛点**：
   - 4D 优化方法无法处理遮挡区域——只能合成"共可见"像素
   - 数据驱动方法过度幻想——物体突然出现/消失
   - 缺乏动态场景的**多视角**训练数据——最佳数据来源是单目 2D 视频

**核心矛盾**：3D 重建给出几何精确但不完整的渲染；扩散生成给出完整但几何不一致的结果

**切入角度**：解耦——共可见像素由 3D 重建渲染（精确），遮挡像素由视频扩散修复（创造性）

**核心 idea**：CogNVS 是一个条件视频修复扩散模型，只生成遮挡区域而保留已知区域。训练时用 2D 视频自监督（随机相机轨迹构造共可见掩码作为训练对），测试时在目标视频上 TTF 适应

## 方法详解

### 整体框架
三阶段管线：(1) **重建**：用 MegaSAM 从单目视频得到动态 3D 重建 $\mathcal{G}_{src}$ 和相机里程 $\mathbf{c}_{src}$。渲染到新视角得到部分可见像素 $\mathbf{V}_{nvs}^{cov}$；(2) **修复**：CogNVS (基于 CogVideoX-5B) 以部分可见渲染为条件，生成完整新视角视频 $\mathbf{V}_{nvs}$——修复遮挡区域同时允许更新可见区域的外观（如视角相关光照）；(3) **测试时微调**：用目标视频的自监督 pair 微调 CogNVS 200-400 步，减少训练-测试域差异。

### 关键设计

1. **自监督数据构造**：

    - 功能：从**任意** 2D 单目视频构造修复训练对，无需多视角 GT
    - 核心思路：给定源视频 → 重建 → 随机采样 N 条新相机轨迹 → 找到源视角和新视角之间的共可见 3D 点 $\mathcal{G}_{src,n}^{cov}$ → 渲染到源视角得到"部分可见的源视频" $\mathbf{V}_{src,n}^{cov}$。训练对 = (部分可见源视频, 完整源视频)
    - 设计动机：**3D 感知的掩码**（而非随机 2D 掩码）更好地模拟了真实的 3D 可见性——遮挡模式与视角变化直接相关

2. **CogNVS 架构**：

    - 基于 CogVideoX-5B（Transformer 视频扩散模型），自注意力+3D-RoPE
    - 原本是 image-to-video 模型，改为 video-to-video——条件视频和目标视频形状对齐，无需填充
    - 条件输入：VAE 编码的部分可见新视角渲染 $\mathbf{z}_{cond}$
    - 目标输出：完整新视角视频
    - 训练目标：标准 score matching $\|\epsilon_\theta(\mathbf{z}_k, k, \mathbf{z}_{cond}) - \epsilon\|_2^2$

3. **测试时微调 (TTF)**：

    - 功能：减少预训练数据和目标测试视频之间的域差异
    - 核心思路：用目标测试视频自身构造自监督 pair（同样的重建+随机轨迹方法），用 200-400 步 AdamW 微调 CogNVS
    - 设计动机：不同视频的光照、外观、运动模式差异大——预训练模型的通用先验不够。TTF 让模型"记住"目标视频的特征同时保留通用修复能力
    - 关键发现：**TTF 是从"还行"到"最优"的关键**——无 TTF 时性能与竞品持平，有 TTF 后超越所有先前方法

### 损失函数 / 训练策略
- 预训练：10K 个 2D 视频（SA-V, TAO, YouTube-VOS, DAVIS），全量微调 42 层 Transformer，12K 步，3 天 × 8 A6000
- TTF：每个测试视频 200-400 步 AdamW，学习率 2e-5

## 实验关键数据

### 主实验 — Kubric-4D (合成动态场景，零样本)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | 训练数据 |
|------|-------|-------|--------|------|---------|
| GCD | 20.1 | 0.734 | 0.186 | 85.3 | Kubric (in-domain) |
| TrajectoryCrafter | 18.5 | 0.701 | 0.228 | 102.4 | 大规模 |
| Gen3C | 19.3 | 0.715 | 0.210 | 93.5 | 大规模 |
| **CogNVS (zero-shot+TTF)** | **20.8** | **0.745** | **0.172** | **78.6** | 2D视频 (OOD) |

### 主实验 — DyCheck (真实动态场景)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 类型 |
|------|-------|-------|--------|------|
| Shape-of-Motion | 21.2 | 0.665 | 0.341 | 测试时优化 (小时级) |
| MoSca | 20.8 | 0.651 | 0.356 | 测试时优化 |
| **CogNVS (TTF)** | **21.9** | **0.683** | **0.312** | 前馈+TTF (分钟级) |

### 消融实验 — TTF 的影响

| 配置 | Kubric PSNR↑ | DyCheck PSNR↑ | 说明 |
|------|-------------|-------------|------|
| CogNVS (无TTF) | 19.2 | 20.4 | 仅预训练模型 |
| CogNVS (**有TTF**) | **20.8** | **21.9** | **+1.6/+1.5 PSNR** |
| 随机2D掩码训练 (无TTF) | 18.1 | 19.2 | 3D掩码>2D掩码 |

### 关键发现
- **TTF 是最关键的组件**——带来 1.5+ PSNR 提升，将性能从"竞品水平"推到"超越所有先前方法"
- **零样本超越 in-domain 训练的 GCD**——CogNVS 没见过 Kubric 数据但表现更好，说明 2D 视频自监督+TTF 的范式比特定数据集训练更强
- 3D 感知掩码比随机 2D 掩码好 1+ PSNR——遮挡模式需要与 3D 可见性对齐
- CogNVS 能产出**清晰的动态物体**（其他方法在动态区域模糊）——因为修复模型专门处理遮挡，不需要像 4D 表示那样在有限视角中拟合动态

## 亮点与洞察
- **"重建+修复+微调"的三阶段解耦**思路清晰且各阶段可独立改进——重建用最好的 SLAM，修复用最好的扩散模型，微调用标准 TTF
- **自监督训练数据构造是核心创新**——将"需要多视角 GT 的 NVS 训练"转化为"用任意 2D 视频即可训练"。这解锁了海量的互联网视频数据
- TTF 是 test-time optimization 的"最佳两全方案"——保留数据驱动的鲁棒性（来自大规模预训练），获得优化的精度（来自测试时适应）
- 方法**首次**在动态场景 NVS 上实现了"前馈速度+优化精度"的组合

## 局限与展望
- 推理仍需 ~5 分钟/视频（CogVideoX 较大）——更小更快的扩散模型可以加速
- 依赖 MegaSAM 重建质量——重建失败的区域会传递错误的可见性掩码
- 真实场景测试（DyCheck）仅 5 个视频——更大规模的真实 benchmark 验证待开展
- 当新视角与源视角差异极大时（如 180° 旋转），重建的共可见区域太少，退化为几乎纯修复

## 相关工作与启发
- **vs GCD (ECCV'24)**：GCD 在 Kubric 上训练，domain-specific；CogNVS 在 2D 视频上训练，零样本泛化更强
- **vs TrajectoryCrafter (concurrent)**：TrajectoryCrafter 过度幻想遮挡区域（不保留几何），CogNVS 的重建先验确保了共可见区域的精确
- **vs CAT4D (concurrent)**：CAT4D 是 score distillation 方法，泛化能力受限；CogNVS 通过 TTF 实现了更好的泛化
- 这种"修复而非重建"的范式可以推广到其他 3D 任务——如 3D 补全、场景编辑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将动态 NVS 重新定义为视频修复问题，自监督训练+TTF 的组合深思熟虑
- 实验充分度: ⭐⭐⭐⭐⭐ 3 数据集零样本评估 + 详细消融（TTF/掩码/训练数据/步数）
- 写作质量: ⭐⭐⭐⭐⭐ 分阶段叙述清晰，自监督数据构造的解释直观
- 价值: ⭐⭐⭐⭐⭐ 推进了单目动态 NVS 的 SOTA，且方法思路对更广泛的 3D 视觉任务有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [\[ECCV 2024\] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis](../../ECCV2024/3d_vision/generative_camera_dolly_extreme_monocular_dynamic_novel_view_synthesis.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)

</div>

<!-- RELATED:END -->
