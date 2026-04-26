---
title: >-
  [论文解读] Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion
description: >-
  [CVPR 2025][图像生成][渲染方程] Uni-Renderer 提出了一种基于双流扩散模型的统一框架，将渲染（从固有属性到 RGB 图像）和逆渲染（从 RGB 图像分解固有属性）建模为两个条件生成任务，通过循环一致性约束缓解逆渲染中的固有歧义问题，在材质分解和渲染编辑上取得了优于现有方法的效果。
tags:
  - CVPR 2025
  - 图像生成
  - 渲染方程
  - 逆渲染
  - 双流扩散
  - 循环一致性
  - 材质编辑
---

# Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion

**会议**: CVPR 2025  
**arXiv**: [2412.15050](https://arxiv.org/abs/2412.15050)  
**代码**: 即将开源（论文承诺）  
**领域**: 扩散模型 / 渲染  
**关键词**: 渲染方程, 逆渲染, 双流扩散, 循环一致性, 材质编辑

## 一句话总结

Uni-Renderer 提出了一种基于双流扩散模型的统一框架，将渲染（从固有属性到 RGB 图像）和逆渲染（从 RGB 图像分解固有属性）建模为两个条件生成任务，通过循环一致性约束缓解逆渲染中的固有歧义问题，在材质分解和渲染编辑上取得了优于现有方法的效果。

## 研究背景与动机

**领域现状**：基于物理的渲染（PBR）通过渲染方程模拟光与材质的交互，生成逼真的 2D 图像。传统方法依赖蒙特卡罗光传输模拟和路径追踪来求解渲染方程，但递归追踪计算成本极高。逆渲染则从图像中反推几何、材质和光照等固有属性，是一个严重的 ill-posed 问题。近年来，扩散模型在材质估计和渲染中展现了潜力。

**现有痛点**：（1）传统渲染方法计算昂贵，难以实时运行；（2）逆渲染存在固有歧义——同一张图像可对应多种几何-材质-光照的组合，从图像到固有属性的映射不是一一对应的；（3）现有基于扩散模型的方法（如 RGB2X）将渲染和逆渲染作为两个独立任务分别训练两个模型，无法利用两个任务的互补关系。

**核心矛盾**：逆渲染的歧义问题本质上是因为缺乏约束——给定一张图片，有无数种材质-光照组合可以产生同样的外观。如果能用渲染过程来验证逆渲染的结果（分解出的属性重新渲染应该恢复原图），就能大大缩小解空间。

**本文目标**：构建一个统一的框架同时处理渲染和逆渲染，使两个任务互相促进——渲染为逆渲染提供一致性约束，逆渲染为渲染提供条件输入。

**切入角度**：受 UniDiffuser 启发，通过两个独立的时间步调度来同时建模两个条件分布（属性→RGB 和 RGB→属性），而不是像 UniDiffuser 那样建模所有分布。这种有针对性的多任务学习减少了任务冲突。

**核心 idea**：用一个双流扩散模型同时学习渲染方程的正向和逆向条件分布，并通过循环一致性约束（逆渲染→渲染→与原图比较）显式减少歧义。

## 方法详解

### 整体框架

Uni-Renderer 的输入输出取决于任务模式：渲染模式下输入固有属性（metallic、roughness、albedo、normal、specular lighting、diffuse lighting），输出 RGB 图像；逆渲染模式下输入 RGB 图像，输出所有固有属性。框架核心是一个双流扩散网络：上层分支处理属性，下层分支处理 RGB，通过交叉条件进行信息交互。训练时使用 timestep 选择器在两个任务之间交替：一个分支 timestep 为 0（作为条件），另一个分支选择 $t \in [0, T]$（需要去噪）。

### 关键设计

1. **双流扩散架构（Dual Stream Diffusion）**:

    - 功能：在单一模型中同时支持渲染和逆渲染两个条件生成任务
    - 核心思路：模型包含两个预训练的扩散模型分支——一个处理 RGB 图像，一个处理 PBR 属性图。两个分支通过"双流模块"进行信息交叉条件化。时间步选择遵循 $(t_{\text{attributes}}, t_{\text{RGB}}) = (0, \tilde{t})$（渲染模式）或 $(\tilde{t}, 0)$（逆渲染模式），其中 $\tilde{t}$ 以概率 $p$ 取 $t$、否则取 $T$。与 UniDiffuser 建模所有分布不同，Uni-Renderer 只建模两个条件分布 $q(\mathbf{x}_0 | \mathbf{y}_0)$ 和 $q(\mathbf{y}_0 | \mathbf{x}_0)$，减少了任务冲突。
    - 设计动机：从多任务学习角度看，网络容量有限，同时拟合太多分布会导致任务竞争和次优表现。限制为两个互补的条件分布既足以覆盖渲染/逆渲染需求，又避免了不必要的任务冲突。双流架构允许两个分支共享中间层信息，实现互益学习。

2. **循环一致性约束（Cycle-Consistent Constraint）**:

    - 功能：通过"逆渲染→渲染→比较"的循环验证减少逆渲染歧义
    - 核心思路：在训练中，对逆渲染的预测结果执行额外的正向渲染：用模型预测的属性 $\hat{\mathbf{C}}$ 重新渲染得到 $\hat{\mathbf{x}}_{\text{rgb}}$，然后将这个重新渲染的图像与原始 RGB 图像比较。损失函数为 $\mathcal{L} = \mathbb{E}[||\mathbf{x}_0 - \hat{\mathbf{x}}_0(\hat{\mathbf{x}}_{\text{rgb}}, t, \mathbf{C})||^2]$。这本质上是一个自监督约束——分解出的属性如果正确，重新渲染后应该能恢复原图。
    - 设计动机：逆渲染的歧义源于从 RGB 到属性的映射不是单射。循环一致性约束强制模型预测的分解结果在渲染方程下是"可还原的"，大大缩小了解空间。这个约束在统一框架中非常自然地实现——同一个模型既能做逆渲染又能做渲染，循环过程无需额外模型。

3. **潜在空间属性编码（Latent Preparation）**:

    - 功能：将不同类型和尺度的固有属性统一编码到扩散模型的潜在空间
    - 核心思路：使用独立的预训练 VAE 分别编码 albedo $\mathbf{a}$、surface normal $\mathbf{n}$、环境光照 $\mathbf{s}, \mathbf{d}$。对于标量属性 metallic $m$ 和 roughness $r$，先将其展开为灰度图，与二值掩码 $\mathbb{m}$ 组成三通道组 $[m, r, \mathbb{m}]$，用 RGB VAE 编码。所有编码后的潜在特征沿通道维度拼接后输入模型。
    - 设计动机：metallic 和 roughness 是标量，无法直接用 VAE 编码，将它们转为图像格式配合掩码使用现有的 RGB VAE 是一个工程上的巧妙处理。分离各属性的 VAE 避免了不同属性之间的信息混淆。

### 损失函数 / 训练策略

- 采用 $\mathbf{x}_0$-prediction 形式的扩散训练
- 训练数据使用 Objaverse 中的 200K 3D 资产，每个资产通过改变 metallic（0-1，步长 0.1）和 roughness（0-1，步长 0.1）生成 121 对渲染对，随机选择 LHQ-1024 中的 20K 环境光照图
- 渲染分辨率 1024×1024，相机位置固定在物体正面
- 预留 100 个训练中未见的物体用于测试

## 实验关键数据

### 主实验

渲染性能对比（metallic/roughness 编辑，PSNR↑ / LPIPS↓）：

| 方法 | Metallic PSNR↑ | Metallic LPIPS↓ | Roughness PSNR↑ | Roughness LPIPS↓ |
|------|---------------|----------------|----------------|-----------------|
| InstructPix2Pix* | 24.25 | 0.1032 | 24.43 | 0.1056 |
| Subias et al. | 28.09 | 0.0954 | 28.13 | 0.0817 |
| **Ours** | **30.72** | **0.0763** | **31.68** | **0.0695** |

逆渲染性能对比（albedo / metallic / roughness / normal）：

| 方法 | Albedo PSNR↑ | Albedo LPIPS↓ | Metallic MSE↓ | Roughness MSE↓ | Normal cos↑ |
|------|-------------|--------------|--------------|----------------|------------|
| IntrinsicAnything | 22.67 | 0.0633 | - | - | - |
| RGB2X | 18.15 | 0.0851 | - | - | 0.871 |
| GaussianShader | 16.55 | 0.0906 | 0.3421 | 0.3714 | 0.908 |
| Intrinsic Image Diff | 21.83 | 0.0632 | 0.1920 | 0.1315 | - |
| **Ours** | **23.20** | **0.0532** | **0.1182** | **0.1037** | **0.928** |

### 消融实验

| 配置 | Albedo PSNR↑ | Metallic MSE↓ | Roughness MSE↓ | Relighting PSNR↑ |
|------|-------------|--------------|----------------|-----------------|
| **Full model** | **23.20** | **0.1182** | **0.1037** | **30.84** |
| w/o unified（分离两个模型） | 18.62 | 0.1632 | 0.1391 | 26.95 |
| w/o cycle constrain | 21.20 | 0.1391 | 0.1304 | 28.12 |

### 关键发现

- 统一框架 vs 分离两个模型：albedo PSNR 从 18.62 提升到 23.20（+4.58），relighting PSNR 从 26.95 提升到 30.84（+3.89），证明联合训练的两个任务互相促进
- 循环一致性约束额外贡献了约 2 个 PSNR 的提升（逆渲染）和 2.72 个 PSNR（relighting），验证了通过渲染循环验证分解结果的有效性
- 渲染质量方面比 InstructPix2Pix（即使在同数据集上微调）高出 6+ PSNR，因为物理基础的属性条件化比纯文本提示提供了更精确的控制
- 模型在真实世界图像上也展现了有效的逆渲染能力（如金属手机架、水壶等），尽管仅在合成数据上训练

## 亮点与洞察

- **将渲染方程建模为条件生成问题**：用数据驱动的扩散模型替代物理模拟来近似渲染方程，避免了递归光线追踪的高计算成本，这个思路为"用生成模型替代/辅助物理模拟"开辟了新方向
- **循环一致性约束的巧妙实现**：在统一框架中，同一个模型既做渲染又做逆渲染，循环验证过程不需要任何额外模型或额外训练成本，实现非常自然
- **多任务学习的取舍**：与 UniDiffuser 建模所有分布不同，Uni-Renderer 只建模两个互补的条件分布，减少了任务冲突。这个"少即是多"的设计哲学值得在其他多任务生成场景中借鉴

## 局限与展望

- 训练数据全部为合成数据（Objaverse），存在合成-真实域间隙。虽然在部分真实图像上展示了效果，但复杂的真实场景（如多物体、复杂光照）可能表现不佳
- 相机位置固定在物体正面，不支持多视角渲染/逆渲染
- 每个物体仅有 121 对（11×11 metallic-roughness 组合），材质变化的细粒度有限
- 环境光照使用 2D 环境图表示，无法处理 3D 空间中的局部光源、阴影等复杂光照现象
- 当前只能处理单物体场景，不支持多物体场景的分层渲染
- 未来需要引入更多真实世界数据来弥补域间隙

## 相关工作与启发

- **vs RGB2X**: RGB2X 也使用扩散模型进行正向和逆向渲染，但使用两个独立模型，无法利用两者的互补关系。Uni-Renderer 通过统一框架和循环一致性约束取得了更好的逆渲染效果（albedo PSNR: 23.20 vs 18.15）
- **vs MaterialGAN/SIC**: 这些方法使用 GAN 或编码器-解码器架构，主要处理平面表面材质，且仍需要渲染器集成预测属性；Uni-Renderer 直接端到端生成
- **vs UniDiffuser**: 技术灵感来源，但 Uni-Renderer 选择性地只建模两个条件分布而非所有分布，更适合渲染/逆渲染的特定需求
- **vs NvDiffRec/GaussianShader**: 这些是基于优化的方法，需要多视角输入和逐物体优化；Uni-Renderer 是前馈式的，单图即可完成逆渲染

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一渲染/逆渲染到一个扩散框架的想法有新意，循环一致性约束设计巧妙，但双流扩散的技术基础来自 UniDiffuser
- 实验充分度: ⭐⭐⭐⭐ 渲染和逆渲染双方向的定量定性对比充分，消融设计合理，但缺少更多真实场景的验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，figure 可视化效果好，但部分细节（如双流模块的具体交互方式）描述不够详细
- 价值: ⭐⭐⭐⭐ 为数据驱动的渲染/逆渲染提供了有效的统一范式，对游戏制作、建筑可视化等下游应用有潜在价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[ICCV 2025\] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](../../ICCV2025/image_generation/ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)
- [\[CVPR 2025\] PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [\[CVPR 2025\] AMO Sampler: Enhancing Text Rendering with Overshooting](amo_sampler_enhancing_text_rendering_with_overshooting.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)

<!-- RELATED:END -->
