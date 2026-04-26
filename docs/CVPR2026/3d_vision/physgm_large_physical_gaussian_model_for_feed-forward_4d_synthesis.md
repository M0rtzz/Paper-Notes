---
title: >-
  [论文解读] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
description: >-
  [CVPR2026][3D视觉][3D高斯溅射] PhysGM 提出首个前馈式框架，从单张图像一次推理即可同时预测 3D 高斯表示和物理属性（刚度、质量等），结合 MPM 仿真在一分钟内生成高保真的物理合理 4D 动画，无需任何逐场景优化。
tags:
  - CVPR2026
  - 3D视觉
  - 3D高斯溅射
  - 物理仿真
  - 前馈式4D合成
  - MPM
  - DPO
  - 物理属性预测
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis

**会议**: CVPR2026  
**arXiv**: [2508.13911](https://arxiv.org/abs/2508.13911)  
**代码**: [项目主页](https://hihixiaolv.github.io/PhysGM.github.io/)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 物理仿真, 前馈式4D合成, MPM, DPO, 物理属性预测

## 一句话总结

PhysGM 提出首个前馈式框架，从单张图像一次推理即可同时预测 3D 高斯表示和物理属性（刚度、质量等），结合 MPM 仿真在一分钟内生成高保真的物理合理 4D 动画，无需任何逐场景优化。

## 研究背景与动机

**3DGS 重建瓶颈**：现有物理驱动 4D 合成依赖从稠密多视角图像重建 3DGS，需要耗时的逐场景优化，无法规模化部署。

**物理参数手动指定**：PhysGaussian 等方法需通过配置文件手动设定刚度、质量等物理属性，缺乏自动化和泛化能力。

**SDS 优化代价高**：OmniPhysGS、DreamPhysics 等用 SDS 从视频模型蒸馏物理先验，需梯度回传穿过可微物理仿真器，单场景优化耗时 0.5–12 小时。

**外观与物理脱节**：现有方法将预建的 3DGS 与物理模块简单拼接，忽略了外观中蕴含的物理线索（如金属光泽暗示材料刚度），导致性能次优。

**缺乏训练数据**：此前不存在将 3D 资产与物理属性标注和参考仿真视频配对的大规模数据集。

**核心问题**：能否完全绕过逐场景优化，通过单次前馈推理生成完整的物理驱动 4D 仿真？

## 方法详解

### 整体框架

PhysGM 为 Transformer 架构的物理感知重建模型，输入带位姿的 RGB 图像（1 或 4 视角），单次前馈输出：(1) 3D 高斯表示参数 $\psi$；(2) 物理属性向量 $\theta$。这些参数直接初始化 MPM 仿真器生成动态序列。单视角输入时用 MVAdapter 合成后、左、右三个辅助视角。

**多模态分词**：

- 图像编码器采用 DINOv3 (ViT-L/16)，对输入图像进行 patch 化并投影为 1024 维特征
- 相机几何用 Plücker 射线坐标表示，经密集表示编码器处理后与图像 token 拼接
- 额外引入 3 个可学习全局 token $\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3$ 用于聚合全局场景信息

**Transformer 骨架**：24 层 Transformer，收集中间层输出 token 提供多尺度表示。

### 关键设计一：双头预测

**DPT 头（3DGS 参数）**：采用 Dense Prediction Transformer 头，从多尺度特征逐步上采样，输出逐像素 3DGS 参数图（位置 $\mu$、旋转 $\mathbf{q}$、缩放 $\mathbf{s}$、不透明度 $\alpha$、颜色 $\mathbf{c}$），各视角预测的高斯聚合为完整 3D 场景。

**物理头**：从 3 个全局 token 预测三类物理属性：

- 分类头 $f_{\text{material}}$：预测材料类别 $C$（决定本构模型选择）
- 回归头 $f_{\text{phys}}$：输出 Young 模量 $E$ 和泊松比 $\nu$ 的均值与对数方差，定义条件概率分布 $P(\theta|I) = \mathcal{N}(\theta|\mu_\theta, \text{diag}(\sigma_\theta^2))$

概率式建模允许采样多样化物理参数，为后续 DPO 偏好对的构建提供基础。

### 关键设计二：两阶段训练

**Stage 1 — 监督预训练**：在 PhysAssets 数据集上联合优化 3DGS 重建（MSE + Alpha + LPIPS 损失）和物理属性预测，建立强泛化先验。联合优化的关键优势是让几何与物理互相增强。

**Stage 2 — DPO 微调**：冻结骨架，仅微调物理预测头。

- 对每个场景采样 $K$ 组物理参数候选 $\phi_k \sim \pi_\omega(\cdot|\mathbf{z})$
- 各候选运行 MPM 仿真并渲染视频 $V_k$
- 用 SAM-2 分割 + CoTracker-3 轨迹提取计算与 GT 视频的感知距离
- 距 GT 最近者为 "winner" $\phi_w$，最远者为 "loser" $\phi_l$
- DPO 损失：$L_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\omega(\phi_w|\mathbf{z})}{\pi_{\text{ref}}(\phi_w|\mathbf{z})} - \beta\log\frac{\pi_\omega(\phi_l|\mathbf{z})}{\pi_{\text{ref}}(\phi_l|\mathbf{z})})]$

DPO 的关键优势：完全绕过可微物理引擎和 SDS，仅需 ranking 信号即可对齐物理合理性。

### 关键设计三：MPM 物理仿真

采用 Material Point Method（MPM）驱动动态仿真，将 3D 高斯与 MPM 粒子一一对应：

- 粒子位置 $\mathbf{x}_p$ 直接定义高斯均值 $\mu$
- 变形梯度 $\mathbf{F}_p$ 经极分解得旋转 $\mathbf{R}_p$ 和拉伸张量 $\mathbf{S}_p$，分别更新高斯旋转和缩放
- 根据预测材料类别选择本构模型：Neo-Hookean（果冻/橡胶）、Fixed Corotational（金属）、Drucker-Prager（沙/雪/橡皮泥）

仿真参数：网格 200³，子步长 $2\times10^{-5}$s，帧时间 $4\times10^{-2}$s，每序列 50 帧。

### 损失函数

- 预训练阶段：$L = L_{\text{MSE}} + L_{\alpha} + L_{\text{LPIPS}}$（渲染图与 GT 对比）+ 物理属性监督损失
- 微调阶段：$L_{\text{DPO}}$（偏好对排序损失）

## 实验

### PhysAssets 数据集

自建的大规模数据集，包含 50K+ 3D 资产（来源于 Objaverse、OmniObject3D、ABO、HSSD），每个资产标注材料类别、Young 模量、泊松比及参考仿真视频。材料分布涵盖塑料（27.3%）、木材（16.8%）、金属（14.6%）、织物（14.5%）等 46 类。

### 主实验结果

| 方法 | 训练方式 | 泛化性 | 推理时间 | CLIP$_{\text{sim}}$ |
|------|---------|--------|---------|-----------|
| OmniPhysGS | SDS | ✗ | >12h | 0.2091 |
| DreamPhysics | SDS | ✗ | >0.5h | 0.2291 |
| **PhysGM (w/o DPO)** | 监督 | ✓ | <1min | 0.2693 |
| **PhysGM (w/ DPO)** | DPO | ✓ | <1min | **0.2748** |

- PhysGM 在 5 种材料类别（金属/果冻/橡皮泥/雪/沙）上全面领先
- 用户偏好率 UPR：PhysGM w/ DPO 达 42.8%（4 选 1，随机 25%），远超 OmniPhysGS 10% 和 DreamPhysics 17.2%

### 多视角重建质量（GSO 数据集）

| 方法 | 分辨率 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|-------|-------|--------|
| LGM | 256 | 21.44 | 0.832 | 0.122 |
| **PhysGM** | 256 | **25.47** | **0.916** | **0.071** |
| GS-LRM | 512 | 30.52 | 0.952 | 0.050 |
| **PhysGM** | 512 | 28.95 | **0.953** | **0.039** |

PhysGM 在 LPIPS 上超越 GS-LRM，仅用 10% 的训练数据。

### 消融实验

DPO 微调阶段的关键性验证：

- w/o DPO → w/ DPO：CLIP$_{\text{sim}}$ 从 0.2693 → 0.2748，UPR 从 30% → 42.8%
- DPO 在所有 5 种材料上均带来一致提升，尤其在金属（UPR 30%→49%）和雪（26%→47%）上增益显著
- 表明 DPO 有效地将统计意义上合理的物理先验转化为感知质量更优的生成器

### 关键发现

1. **前馈推理完全可行**：单次前向传播 <30s + MPM 仿真即可完成端到端 4D 合成，比 SDS 方法快 720× 以上
2. **联合训练优于拼接**：3DGS 与物理属性的联合预测互相增强，避免了信息割裂
3. **DPO 替代 SDS**：无需可微物理引擎，仅需 ranking 反馈即可有效对齐物理真实性
4. **泛化能力强**：能处理拉伸、扭转、多物体多材料交互、真实世界图像等多样场景

## 亮点

- **首个前馈式物理 4D 合成框架**：从单图到完整物理仿真动画 <1 分钟，实现了从 "逐场景优化" 到 "摊销推理" 的范式转变
- **创新性 DPO 训练**：将 RLHF 中的偏好对齐思想引入物理仿真领域，用 SAM-2 + CoTracker-3 自动构建偏好对，完全绕过可微仿真器的限制
- **概率式物理预测**：物理头输出分布而非点估计，既能量化不确定性又为 DPO 采样提供基础
- **PhysAssets 大规模数据集**：50K+ 带物理标注的 3D 资产填补了领域空白
- **速度-质量双赢**：不以牺牲质量换取速度，在各指标上全面超越需要数小时优化的 SDS 方法

## 局限性

- **MPM 仿真计算代价高**：MPM 本身仍是主要计算瓶颈，限制了大规模实时应用
- **Sim-to-Real Gap**：训练数据为合成数据，简化的本构模型与真实物理存在固有差距，影响真实场景泛化
- **GT 视频生成依赖 FramePack**：参考视频由 FramePack 生成而非真实物理录制，可能引入偏差
- **材料类别有限**：当前支持 5 大类本构模型，对流体、断裂等复杂物理现象尚未覆盖
- **物理标注质量**：依赖 MLLM（Qwen3VL）自动标注物理属性，存在标注噪声

## 相关工作

| 方法 | 需预建 3DGS | 自动物理参数 | 可泛化 | 推理时间 |
|------|-----------|------------|--------|---------|
| PhysGaussian | ✗ | 手动指定 | ✓ | - |
| DreamPhysics | ✗ | 仅 Young 模量 | ✓ | >0.5h |
| PhysDreamer | ✗ | 仅 Young 模量 | ✓ | >1h |
| OmniPhysGS | ✗ | 仅材料类别 | ✓ | >12h |
| PhysSplat | ✗ | ✓(需 LLM) | ✗ | <2min |
| **PhysGM** | **✓(无需)** | **✓(全自动)** | **✓** | **<30s** |

PhysGM 是唯一同时满足"无需预建 3DGS""全自动物理参数""强泛化""快速推理"的方法。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个前馈式物理4D合成框架 + DPO用于物理仿真对齐均为全新贡献
- 实验充分度: ⭐⭐⭐⭐ — 5种材料对比、消融、用户研究、多视角重建均有覆盖，但缺少真实场景定量评估
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、方法描述完整、补充材料详尽
- 价值: ⭐⭐⭐⭐⭐ — 将物理4D合成从小时级优化推进到分钟级推理，附带大规模数据集，对后续工作有重大推动

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation](let_it_snow_animating_3d_gaussian_scenes_with_dynamic_weather_effects_via_physic.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation](physgs_bayesian-inferred_gaussian_splatting_for_physical_property_estimation.md)

<!-- RELATED:END -->
