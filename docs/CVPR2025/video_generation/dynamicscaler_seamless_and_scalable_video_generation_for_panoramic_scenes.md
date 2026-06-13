---
title: >-
  [论文解读] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
description: >-
  [视频生成] DynamicScaler 提出了一个无需微调的统一框架，通过偏移移位去噪器（OSD）和全局运动引导（GMG）实现任意分辨率/宽高比的全景动态场景合成，支持常规全景和 360° 视野视频生成，同时保持恒定 VRAM 消耗。 沉浸式 AR/VR 应用对场景级和 360° 全景视频的需求日益增长…
tags:
  - "视频生成"
---

# DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes

## 一句话总结

DynamicScaler 提出了一个无需微调的统一框架，通过偏移移位去噪器（OSD）和全局运动引导（GMG）实现任意分辨率/宽高比的全景动态场景合成，支持常规全景和 360° 视野视频生成，同时保持恒定 VRAM 消耗。

## 研究背景与动机

沉浸式 AR/VR 应用对场景级和 360° 全景视频的需求日益增长，但现有视频扩散模型受限于固定分辨率和宽高比：

1. **分辨率限制**：大多数视频扩散模型只能生成固定分辨率（如 512×512）的短视频，无法直接生成超宽或超高分辨率全景
2. **运动一致性难题**：拼接式方法（如 MultiDiffusion、SyncDiffusion）使用重叠窗口，计算开销大且运动不一致
3. **360° 全景特殊挑战**：等距投影（ERP）的变形、曲线运动模式、以及左右边界需要无缝衔接
4. **内存限制**：高分辨率视频生成的 VRAM 消耗随分辨率增长，限制了实际应用
5. **现有 360° 方法限制**：360DVD 需要微调且分辨率低；4K4DGen 依赖优化过程且运动范围受限

核心问题：如何用固定分辨率的预训练视频扩散模型，无需微调地生成任意分辨率/宽高比的全景视频，同时保证运动一致性和空间连贯性？

## 方法详解

### 整体框架

DynamicScaler 采用两阶段生成策略：(1) 低分辨率阶段建立粗略运动结构（对 360° 场景使用全景投影去噪初始化）；(2) 上采样阶段通过 GMG 从低分辨率引导生成高分辨率全景。核心是 OSD 机制在每个去噪步中移位窗口位置，创建"跨步重叠"实现全局同步。

### 关键设计

#### 1. 偏移移位去噪器（Offset Shifting Denoiser, OSD）

- **功能**：将全景视频 latent 分割为多个窗口，每步移位窗口位置实现无缝去噪
- **核心思路**：在每个去噪步中，将 $W_p \times H_p$ 的全景 latent 分为 $n_W \times n_H$ 个窗口送入固定分辨率的扩散模型去噪。关键创新是**每步在水平和垂直方向偏移窗口位置**，使得某一步的窗口边界在下一步被移位覆盖。水平方向上将全景视为环形——左右边界相连，窗口可跨越边界
- **设计动机**：传统分块去噪在窗口边界产生接缝和不一致。显式重叠（如 MultiDiffusion）需要更多窗口导致计算翻倍。OSD 通过"跨步重叠"（不同步骤间的窗口偏移）实现隐式同步，不增加每步计算量，边界伪影在下一步被消除

#### 2. 全局运动引导（Global Motion Guidance, GMG）

- **功能**：确保高分辨率生成中的全局运动一致性
- **核心思路**：将生成分解为全局布局和局部内容两个阶段——先在低分辨率生成视频捕获高层运动结构，再上采样+重新加噪作为高分辨率生成的初始化，引导内容布局和运动模式
- **设计动机**：OSD 的同步效果需要足够的去噪步积累，在早期去噪步（决定整体布局的关键阶段）影响不足，导致不同区域可能产生分离的运动模式。GMG 通过层级化方法先确定全局运动，再用高分辨率细化局部细节

#### 3. 全景投影去噪器 + 时序扩展

- **功能**：将 OSD 扩展到 360° 球面全景和时间维度
- **核心思路**：对 360° 全景，将等距投影映射回多个透视视口窗口去噪，再反投影回 ERP。视口的视角每步偏移，实现球面空间的 OSD。对长视频，在时间维度类似地分割帧窗口并偏移，帧序列视为环形可实现无缝循环视频。通过 mask $M_d$ 追踪已去噪区域，对重叠区域进行噪声重平衡
- **设计动机**：ERP 的变形使常规扩散模型直接在 ERP 空间去噪效果差。投影到透视视口后使用常规模型避免变形问题。时序扩展突破短视频限制（16帧→任意长度），环形机制支持循环播放

### 损失函数

DynamicScaler 是无训练（training-free）方法，不涉及训练损失。核心公式是 OSD 去噪过程：

$$Z_t = Con|_{1:n_W, 1:n_H}\left(\Phi_\theta(t, c, Split(Z_{t-1}, i, j, t, n_W, n_H))\right)$$

GMG 层级生成：

$$Z_{HR^0} = \Phi_\theta^{OSD}(noise(inter(\Phi_\theta^{OSD}(Z_{LR^T}))))$$

## 实验关键数据

### 主实验表

**与 360DVD 定量比较（Tab. 1）**：

| 指标 | 360DVD | DynamicScaler |
|------|--------|---------------|
| CLIP-Score↑ | 0.293 | **0.302** |
| Image Quality↑ | 0.436 | **0.583** |
| Dynamic Degree↑ | 0.412 | **0.783** |
| Motion Smoothness↑ | 0.917 | **0.963** |
| Temporal Flickering↑ | 0.964 | **0.982** |
| Scene↑ | 0.417 | **0.499** |
| Q-Align(I)↑ | 0.485 | **0.632** |
| Q-Align(V)↑ | 0.532 | **0.613** |

### 功能对比

| 特性 | 360DVD | 4K4DGen | ScaleCrafter | VividDream | **DynamicScaler** |
|------|--------|---------|-------------|------------|-------------------|
| 无需微调 | ✗ | ✗ | ✓ | ✗ | **✓** |
| 任意尺寸 | ✓ | ✗ | ✓ | ✓ | **✓** |
| 360° FoV | ✓ | ✓ | ✗ | ✗ | **✓** |
| 文本条件 | ✓ | ✗ | ✓ | ✗ | **✓** |
| 图像条件 | ✗ | ✓ | ✗ | ✓ | **✓** |
| 无限视频 | ✗ | ✗ | ✗ | ✗ | **✓** |
| 循环生成 | ✗ | ✗ | ✗ | ✗ | **✓** |

### 关键发现

1. **全面超越 360DVD**：在所有 8 个指标上优于 360DVD，尤其动态程度（0.783 vs 0.412）和图像质量（0.583 vs 0.436）差距显著
2. **恒定 VRAM**：不论输出分辨率如何，VRAM 消耗保持恒定（每次只处理一个固定窗口）
3. **功能最全面**：是唯一同时支持无训练、任意尺寸、360°、文本/图像条件、长视频和循环生成的方法
4. 视频长度从 16 帧扩展到 80+ 帧，质量保持一致

## 亮点与洞察

1. **偏移移位的核心洞察**：与其用重叠窗口增加计算量来消除接缝，不如在不同去噪步之间移位窗口位置，让"接缝"在下一步被覆盖。这是一个简洁却极其有效的设计
2. **环形连接实现无缝**：将全景 latent 水平视为环形使窗口可跨越左右边界，自然支持 360° 全景的连续性要求，也优雅地支持时序循环
3. **无训练方案的优势**：完全基于预训练视频扩散模型，可直接受益于模型升级（如从 SVD 升级到更好的基座模型），无需重新训练
4. **层级化 GMG 设计合理**：先低分辨率确定运动结构，再高分辨率细化，符合扩散模型"从粗到细"的生成机制

## 局限性与可改进方向

1. **运动复杂度受限**：依赖基座模型的运动生成能力，对复杂场景级运动（多物体交互）可能力不从心
2. **高极角区域变形**：360° 投影在极点附近窗口大量重叠，需要噪声重平衡可能引入额外伪影
3. **缺乏更多定量评估**：仅与 360DVD 做了定量比较，与更多 SOTA 方法的对比不足
4. **文本语义控制精度**：大宽幅全景的不同区域可能需要不同的文本控制，单一文本描述可能不够
5. 没有评估用户交互需求，如对全景特定区域的局部编辑

## 相关工作与启发

- **MultiDiffusion / SyncDiffusion**：全景图像拼接方法，使用重叠窗口，DynamicScaler 用偏移移位替代
- **360DVD**：首个 360° 视频扩散模型，但需微调且分辨率低
- **ScaleCrafter**：空间可扩展扩散模型，但不支持 360°
- 启发：偏移移位的思路可推广到其他需要空间/时间可扩展性的扩散任务，如超分辨率和视频外推

## 评分：⭐⭐⭐⭐

核心 OSD 机制设计简洁高效，功能覆盖全面（同时支持 7 种能力），无训练方案实用性强。扣一星因为定量评估对比不够充分，且运动复杂度受基座模型限制。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
