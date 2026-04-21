---
title: >-
  [论文解读] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion
description: >-
  [ICCV 2025][3D视觉][城市级3D生成] 提出 Sat2City，首个从单张卫星图像同时生成城市级几何和外观的3D生成框架，通过将稀疏体素与级联潜扩散模型结合，引入 Re-Hash 多尺度特征网格和逆采样策略，在自建3D城市数据集上实现了优于现有方法的高保真生成。
tags:
  - ICCV 2025
  - 3D视觉
  - 城市级3D生成
  - 卫星图像
  - 稀疏体素网格
  - 级联潜扩散
  - 外观建模
---

# Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion

**会议**: ICCV 2025  
**arXiv**: [2507.04403](https://arxiv.org/abs/2507.04403)  
**代码**: [ai4city-hkust/Sat2City](https://ai4city-hkust.github.io/Sat2City/)  
**领域**: 3d_vision  
**关键词**: 城市级3D生成, 卫星图像, 稀疏体素网格, 级联潜扩散, 外观建模

## 一句话总结

提出 Sat2City，首个从单张卫星图像同时生成城市级几何和外观的3D生成框架，通过将稀疏体素与级联潜扩散模型结合，引入 Re-Hash 多尺度特征网格和逆采样策略，在自建3D城市数据集上实现了优于现有方法的高保真生成。

## 研究背景与动机

**城市级3D场景生成** 在游戏、城市规划和数字孪生等领域有广泛应用需求。现有方法存在三个核心瓶颈：

**2D渲染方法缺乏真实3D结构**：CityDreamer 等基于神经渲染的方法通过GAN或扩散模型生成街景图像，但只能从有限视角渲染，无法显式重建3D结构。由于缺乏3D纹理坐标的直接监督，存在严重的3D几何歧义问题。

**3D方法难以扩展到城市尺度**：Sat2Scene 虽然首次将扩散模型用于3D点云颜色生成，但依赖于预定义的密集点云（约400点/平方米），计算代价极高且无法精化几何，限于街道级别场景。

**高质量城市级3D数据的稀缺**：缺乏同时具备高质量几何和外观的城市尺度3D训练数据，制约了方法的训练与评估。

**核心洞察**：XCube 等工作证明稀疏体素网格与潜扩散模型的结合在大规模户外场景生成中有巨大潜力，但现有方法仅处理几何而忽略外观。Sat2City 的目标是将外观编码为体素颜色属性，通过三个关键技术突破同时实现几何和外观的联合生成。

## 方法详解

### 整体框架

Sat2City 的架构包含 **两阶段训练范式**（先VAE、后扩散模型）和 **三级级联推理**（密集几何→稀疏几何→外观）：

- **输入**：卫星高度图（height map），被提升为点云 $P_h$
- **3D训练数据**：彩色点云 $P_C \in \mathbb{R}^{N \times 6}$，体素化为稀疏体素网格 $G$
- **输出**：具有几何结构和纹理外观的3D城市模型

### 关键设计一：三叉瓶颈 VAE (Triplet Bottleneck VAE)

传统VAE使用单一稀疏潜变量，但这对于同时编码几何和外观是不够的。Sat2City 引入三种不同的瓶颈结构：

1. **密集瓶颈（Dense Neck）**：将稀疏编码膨胀为密集体积，使扩散模型能显式区分占据和非占据区域。这对修正含噪高度图引入的伪体素至关重要。密集化在瓶颈层进行而非输入层，保持了可承受的计算代价。

2. **稀疏瓶颈（Sparse Neck）**：标准的稀疏编码-解码结构，直接解码重建几何和法线属性。稀疏潜变量 $X_S$ 同时用于：(a) 在训练中为外观网格提供结构化剪枝指导；(b) 消除密集解码未能抑制的多余体素。

3. **Re-Hash瓶颈（Re-Hash Neck）** — 本文最核心的创新之一。通过层次化粗化机制，对稀疏特征网格进行迭代重采样，构建多层次表示：

$$v_n = 2^n v_0, \quad o_n = \frac{v_n}{2}$$

在每一层级，通过三线性插值从前一层级采样特征：

$$X_{Cn} = \text{Tri}(G_{Cn}, X_{Cn-1})$$

多层次结构既保留了精细的外观细节，又提供了全局上下文信息，对平滑外观过渡至关重要。

**双阶段外观训练**：先训练几何（稀疏VAE的编解码），到第 $E$ 轮后冻结几何编码器，用 $X_S$ 初始化最精细层外观潜变量 $X_{C0}$，再训练外观解码器 $\mathcal{D}_c$。

### 关键设计二：逆采样 (Inverse Sampling)

直接将点云颜色分配给体素网格顶点存在两难困境：

- **最近邻赋值**：缺乏平滑性，导致颜色跳变
- **三线性泼溅**：多个点重叠贡献产生颜色混合冲突

Sat2City 采用 **逆采样** 策略：不直接学习每顶点颜色属性 $A_C$，而是在输入点云位置上进行隐式监督。训练时，对外观潜变量层次中的每一层 $X_{Cn}$，解码为每顶点颜色特征后，在彩色点云 $P_C$ 位置进行三线性采样，拼接所有层级特征后由MLP生成估计颜色：

$$\tilde{P}_C = \text{MLP}\{\oplus_{k=0}^n \text{Tri}(P_C, \mathcal{D}_c(X_{Cn}))\}$$

推理时，在预测的网格顶点 $\tilde{G}$ 上同样采样获取颜色：

$$\tilde{A}_C = \text{MLP}\{\oplus_{k=0}^n \text{Tri}(\tilde{G}, \mathcal{D}_c(X_{Cn}))\}$$

### 关键设计三：条件级联3D潜扩散

单阶段潜扩散无法处理大规模3D场景。Sat2City 采用三级顺序条件扩散管线，后一阶段以前一阶段输出为条件：

1. **密集几何潜扩散**：以高度图 $P_h$ 为条件，生成密集潜变量 $X_D$，编码整体空间布局：
$$p(X_D, G, A_N) = p_{\mathcal{D}_d}(G, A_N | X_D) \cdot p_{\Psi_D}(X_D | c(P_h))$$

2. **稀疏几何潜扩散**：以第一阶段解码的 $\{G, A_N\}$ 为条件，拟合精细表面结构，并记录体素剪枝决策 $\textit{struct}$：
$$p(X_S, \textit{struct}) = p_{\mathcal{D}_s}(G, A_N | X_S) \cdot p_{\Psi_S}(X_S | G, A_N)$$

3. **外观潜扩散**：以 $\textit{struct}$ 为结构指导进行多层级外观生成：
$$p(G, A_N, A_C) = p_{\mathcal{D}_c}(G, A_N, A_C | X_C) \cdot \prod_{n=0}^N p_{\Psi_{Cn}}(X_{Cn} | \textit{struct})$$

### 数据集

由艺术家在 Blender 中创建的城市网格模型，采样1亿个点形成彩色点云。通过正交相机渲染高度图（2268×3423像素），覆盖 2090×3449.4 m² 区域。裁剪为 300×300 像素的训练样本，共3110个实例（90%训练、10%测试验证）。

## 实验

### 主实验：几何质量对比

| 方法 | MMD↓ (CD) | MMD↓ (EMD) | COV↑ (CD) | COV↑ (EMD) |
|------|-----------|------------|-----------|------------|
| NFD (无条件) | 0.0445 | 0.2363 | 22.66% | 29.66% |
| BlockFusion (无条件) | 0.0326 | 0.1865 | 50.49% | 55.66% |
| **Sat2City (条件)** | **0.0165** | **0.1157** | **100.00%** | **60.00%** |

Sat2City 在所有几何指标上全面超越现有方法：COV(CD)达到100%且MMD(CD)降低49.4%，表明生成质量稳定、模式坍缩概率极低。

### 用户感知质量评估

| 方法 | TPQ↑ | TSC↑ | GPQ↑ | GSC↑ |
|------|------|------|------|------|
| Sat2Scene (2D) | 6.17 | 5.90 | - | - |
| CityDreamer (2D) | 6.40 | 6.63 | - | - |
| CityDreamer (3D)* | 4.48 | 4.48 | 3.60 | 3.38 |
| Sat2Scene* (重训练) | 3.18 | 3.30 | 3.03 | 3.02 |
| **Sat2City** | **7.35** | **8.03** | **6.27** | **7.02** |

60名参与者的主观评估中，Sat2City 在纹理感知质量（TPQ/TSC）和几何感知质量（GPQ/GSC）上均取得最高分。值得注意的是，直接3D生成的外观质量（TPQ=7.35）甚至超越了2D渲染方法。

### 消融实验

| 消融项 | 关键发现 |
|--------|----------|
| 瓶颈设计 | Single-dense 和 Dual-sparse 变体在训练中颜色损失始终为零（梯度冲突）；Re-Hash 相比 Dual-dense 收敛更快、更稳定、无伪影 |
| 逆采样 | 无逆采样时颜色泼溅产生严重渲染伪影，逆采样确保顶点颜色被体素内点约束 |
| 级联扩散层级 | 仅稀疏结构无法捕获未占据区域→混乱生成；去掉稀疏而保留Re-Hash→缺少结构化剪枝导致外观异常 |

## 亮点与洞察

1. **首个同时建模城市级几何+外观的3D生成框架**（三项能力集于一体的唯一方法）
2. **Re-Hash 操作的精巧设计**：通过层次粗化解决了在稀疏体素上编码平滑外观的难题，避免了密集体积的计算代价
3. **逆采样的巧妙规避**：绕开了点云体素化中颜色分配的两难困境，通过在点云级别的隐式监督实现平滑过渡
4. 从单张卫星图快速生成3D城市（~1分钟），无需辅助输入（如分割图）

## 局限性

1. 训练数据为合成城市（Blender创建），与真实卫星图+3D Tiles配对数据尚有域差距
2. 数据集规模相对有限（3110个实例），可能限制了生成多样性
3. 仅在自建数据集上验证，缺少与真实世界城市数据的对比

## 相关工作

- **物体级3D生成**：XCube/SCube（稀疏体素+扩散）、3D latent set diffusion
- **城市级神经渲染**：InfiniCity（GAN + NeRF）、CityDreamer（语义体素划分）、Sat2Scene（3D点云扩散）
- **资产检索方法**：Blender资产库拼装城市，受限于已有资产多样性

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 5 |
| 实验充分性 | 3 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.0 |

<!-- RELATED:START -->

## 相关论文

- [Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_a.md)
- [WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)
- [AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [UniVG: A Generalist Diffusion Model for Unified Image Generation and Editing](univg_a_generalist_diffusion_model_for_unified_image_generation_and_editing.md)

<!-- RELATED:END -->
