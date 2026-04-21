---
title: >-
  [论文解读] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity
description: >-
  [ICLR 2026][医学图像][3D医学重建] 提出 Inter-Slice Consistent Stochasticity (ISCS)，通过球面线性插值(Slerp)在扩散采样的 re-noising 步骤中生成层间相关噪声，从根源消除 2D 扩散先验做 3D 医学重建时的层间不连续伪影——零额外计算/超参数/训练开销，即插即用到任何 2D 扩散逆问题求解器，在稀疏视角 CT、限角 CT 和 MRI 超分辨率上均持续提升。
tags:
  - ICLR 2026
  - 医学图像
  - 3D医学重建
  - 2D扩散模型
  - 层间一致性
  - 球面线性插值
  - 即插即用
---

# Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity

**会议**: ICLR 2026  
**arXiv**: [2602.04162](https://arxiv.org/abs/2602.04162)  
**代码**: [GitHub](https://github.com/duchenhe/ISCS)  
**领域**: 医学图像/扩散模型  
**关键词**: 3D医学重建, 2D扩散模型, 层间一致性, 球面线性插值, 即插即用

## 一句话总结

提出 Inter-Slice Consistent Stochasticity (ISCS)，通过球面线性插值(Slerp)在扩散采样的 re-noising 步骤中生成层间相关噪声，从根源消除 2D 扩散先验做 3D 医学重建时的层间不连续伪影——零额外计算/超参数/训练开销，即插即用到任何 2D 扩散逆问题求解器，在稀疏视角 CT、限角 CT 和 MRI 超分辨率上均持续提升。

## 研究背景与动机

**3D 医学成像的临床需求**：临床诊断（如肿瘤体积评估、手术规划、疾病进展追踪）依赖完整准确的 3D 体数据重建，而非单独的 2D 切片。

**3D 扩散模型的不可行性**：直接在高维体数据上训练扩散模型面临"维度灾难"——内存、计算和数据需求远超大多数实验室和工业界所能承受（Pinaya et al., 2022; Guo et al., 2025; Wang et al., 2025）。

**2D 先验的实用妥协**：主流做法是在 2D 切片上训练扩散模型，然后逐层重建 3D 体——计算上可行，但引入了新问题。

**层间不连续的根源**：每个 2D 切片在反向扩散过程中独立采样，固有的随机噪声注入导致相邻切片的采样轨迹完全不相关，堆叠后沿 z 轴产生严重的结构不连续和伪影。

**现有方法的局限**：(a) TV 正则化——引入敏感超参数，过度平滑抹去细节；(b) 3D patch 训练 / 双平面先验——增加训练/推理复杂度，对数据有额外约束（如要求立方体）；(c) 这些方法本质上是"后处理修补"，未触及根因。

**来自视频领域的启发**：Kwon & Ye (2025) 指出视频恢复中的时间闪烁同样源于不协调的扩散采样随机性，提出 Batch-Consistent Sampling (BCS) 缓解——本文将此洞察系统性地迁移到 3D 医学重建场景，并提出更优的解决方案。

## 方法详解

### 总体框架

基于 2D 扩散模型的 3D 医学逆问题求解框架包含三步迭代：(1) 去噪预测 $\hat{x}_{0|t}$，(2) 数据保真更新，(3) re-noising 到时间步 $t-1$。ISCS 仅修改第(3)步中的随机噪声注入方式，其余步骤完全不变。

### 关键设计 1：层间不一致性的根因分析

- **功能**：系统分析 2D 扩散先验做 3D 重建时层间不连续的根本原因。
- **核心思路**：DDIM 采样器的 re-noising 步可分解为确定性分量和随机分量：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_{0|t} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}-\eta^2\tilde{\beta}_t^2}\epsilon_{\theta^*}^{(t)}(x_t)}_{\text{确定性噪声}} + \underbrace{\eta\tilde{\beta}_t\epsilon}_{\text{随机噪声}}$$

其中确定性部分由网络预测决定（对相似输入给出相似输出），而随机噪声 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 对每层独立采样——这正是层间不一致的根源。当逆问题严重欠约束（如稀疏视角CT）时，数据保真项约束力弱，独立噪声获得过度自由度，驱使相邻层走向完全不同的轨迹。

- **设计动机**：只有找准根因（不协调的随机性），才能设计出原理性的解决方案，而非对症状做后处理。

### 关键设计 2：基于 Slerp 的层间相关噪声生成

- **功能**：用球面线性插值（Spherical Linear Interpolation, Slerp）替代独立噪声采样，生成层间平滑相关的噪声体。
- **核心思路**：对 $S$ 层的 3D 体，先采样两个锚点噪声向量 $\mathbf{z}_1, \mathbf{z}_S \sim \mathcal{N}(0, \mathbf{I})$，然后在高维超球面上沿测地线插值生成中间层的噪声：

$$\epsilon_i^{\text{ISCS}} = \text{slerp}(\mathbf{z}_1, \mathbf{z}_S; \alpha_i) = \frac{\sin((1-\alpha_i)\Omega)}{\sin(\Omega)}\mathbf{z}_1 + \frac{\sin(\alpha_i\Omega)}{\sin(\Omega)}\mathbf{z}_S$$

其中 $\alpha_i = (i-1)/(S-1)$ 是归一化位置，$\Omega = \arccos(\langle \mathbf{z}_1, \mathbf{z}_S \rangle / (\|\mathbf{z}_1\| \cdot \|\mathbf{z}_S\|))$ 是锚点间的夹角。

- **设计动机**：为何用 Slerp 而非线性插值？根据高维高斯环形定理（Gaussian Annulus Theorem），高维各向同性高斯分布的概率质量集中在半径为 $\sqrt{d}$ 的薄球壳上。线性插值走弦——内插点 $\|z\| < \sqrt{d}$，偏离典型集；Slerp 走测地线——保持向量范数和分布统计量，确保每层噪声仍服从 $\mathcal{N}(0, \mathbf{I})$。

### 关键设计 3：为何 Slerp 优于 BCS（全同噪声）

- **功能**：设计具有"近处强关联、远处弱关联"特性的噪声结构，替代 BCS 的全同噪声方案。
- **核心思路**：BCS 对所有层施加完全相同的噪声，这在视频修复（<16帧、帧间变化小）中可行，但在医学体数据（>300层、层间解剖结构显著变异）中过于刚性——抑制解剖变化，导致"复制伪影"（特征在解剖不同的层间不当复制）。ISCS 的 Slerp 噪声天然满足理想特性：(i) 相邻层噪声高度相关→局部一致性，(ii) 距离增大时相关性衰减→允许全局结构变化。
- **设计动机**：医学体数据的本质特征是"局部连续但全局变化"，噪声的相关结构应当与此匹配，而非一刀切地全同或全独立。

### 关键设计 4：即插即用的集成方式

- **功能**：ISCS 噪声体直接替换任意扩散采样器 re-noising 步中的独立噪声。
- **核心思路**：修改后的更新规则为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_{0|t} + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t) + \sigma_t \cdot \epsilon^{\text{ISCS}}$$

无需改变网络架构、损失函数、训练流程或推理优化步骤。

- **设计动机**：最小化集成成本，使方法可广泛适用于任何已有的 2D 扩散逆问题求解器（DDNM、DDS 等），对资源有限的医学场景尤其有价值。

## 实验关键数据

### 实验设置

- **CT 数据集**：AAPM 2016 低剂量CT（10名患者，5936切片，256×256），评估体积 256×256×300
- **MRI 数据集**：IXI T1 脑部扫描，评估体积 256×256×150，z 轴 5× 降采样模拟各向异性
- **基线方法**：FDK、ADMM-TV（传统）；DDNM、DDS（2D 扩散求解器）；DDS+TV 正则化
- **评估指标**：PSNR、SSIM、LPIPS；三视图（轴向/冠状/矢状）独立评估

### 表 1：稀疏视角 CT (30 views) 主要结果

| 方法 | Axial PSNR | Coronal PSNR | Sagittal PSNR | Coronal SSIM | Sagittal LPIPS |
|------|-----------|-------------|-------------|-------------|---------------|
| FDK | 23.91 | 23.92 | 23.79 | 0.414 | 0.310 |
| ADMM-TV | 32.94 | 33.67 | 33.72 | 0.895 | 0.107 |
| DDS | 34.76 | 35.12 | 35.33 | 0.906 | 0.141 |
| DDS+TV | 36.26 | 37.08 | 37.50 | 0.938 | 0.088 |
| **DDS+ISCS** | **36.97** | **37.75** | **38.16** | **0.944** | **0.065** |

### 表 2：限角 CT ([0°, 100°]) 主要结果

| 方法 | Axial PSNR | Coronal PSNR | Sagittal PSNR | Coronal SSIM | |Δ| |
|------|-----------|-------------|-------------|-------------|-----|
| DDNM | 28.40 | 28.75 | 28.22 | 0.774 | 0.016443 |
| DDNM+ISCS | 30.89 | 31.88 | 31.59 | 0.906 | 0.001899 |
| DDS+TV | 31.40 | 33.33 | 32.83 | 0.906 | 0.002566 |
| **DDS+ISCS** | **31.65** | **32.90** | **32.49** | **0.917** | **0.001966** |

### 表 3：BCS vs ISCS 消融 (SVCT)

| 噪声类型 | Coronal PSNR | Coronal SSIM | Sagittal PSNR | Sagittal LPIPS |
|---------|-------------|-------------|-------------|---------------|
| BCS (全同噪声) | 38.00 | 0.937 | 38.24 | 0.081 |
| **ISCS (Slerp噪声)** | **38.16** | **0.941** | **38.78** | **0.073** |

## 关键发现

1. **ISCS 在所有任务和基线上一致提升**：无论 DDNM 还是 DDS，无论 CT 还是 MRI，加入 ISCS 均有提升，且在多数指标上超越需要额外优化的 TV 正则化方案。
2. **冠状面和矢状面提升尤为显著**：这两个面直接反映 z 轴方向的层间一致性——ISCS 的 Sagittal LPIPS 在 SVCT 上从 DDS 的 0.141 降至 0.065（降幅54%），在 LACT 上从 0.193 降至 0.077。
3. **Slerp 优于 BCS**：全同噪声在医学体数据中产生明显的条纹"复制伪影"，Slerp 的渐变相关结构更适合层间解剖变化。
4. **层间差异指标 |Δ| 早期收敛**：ISCS 使层间差异在采样过程早期即接近 GT 参考值，而基线到后期仍维持较大间隙——ISCS 缩小有效搜索空间，帮助采样器更可靠地收敛。
5. **DDNM 受益更大**：在 LACT 任务中，DDNM+ISCS 相对 DDNM 提升 +2.49/+3.13/+3.37 dB（三视图），说明约束越弱的逆问题越受益于噪声协调。
6. **TV 正则化的代价**：虽然 TV 也能提升量化指标，但视觉上产生过平滑/"卡通化"伪影，擦除精细解剖细节——ISCS 无此副作用。

## 亮点与洞察

- **根因治疗 vs 症状治疗**：TV 正则化是"对不连续的结果做后处理平滑"，ISCS 是"在源头控制产生不连续的原因"——前者掩盖症状，后者消除根因，更优雅且彻底。
- **高维几何的原理性利用**：Slerp 的选择不是ad hoc的，而是基于 Gaussian Annulus Theorem 的严格数学推导——高维高斯噪声集中在超球壳上，因此插值必须沿测地线走才能保持分布不变性。
- **零代价的改善**：不增加任何计算开销、不引入超参数、不需要重新训练——对计算资源有限的医学影像场景极具实用价值。
- **从视频到医学的跨域迁移**：帧间不连续(视频)与层间不连续(3D医学)的根源相同（不协调的采样随机性），但直接搬用 BCS 效果不佳，需要针对医学体数据特性（长序列+大解剖变化）设计更精细的相关结构——体现了"借鉴思想但因地制宜"的研究范式。

## 局限性

1. **仅验证了两类逆问题求解器**：实验仅在 DDNM 和 DDS 上验证，未覆盖更多近年的 DIS 方法（如 MCG、DiffPIR 等），泛化性有待进一步确认。
2. **噪声相关结构固定**：两端锚点+线性分配的 Slerp 结构是固定的，未学习也未适应数据——对于解剖结构变化剧烈的区域（如颈胸交界），可能需要空间自适应的相关场。
3. **仅使用 VE 扩散模型**：所有实验基于 VE-SDE 框架，未验证 VP-SDE 或基于 Flow Matching 的更新框架下的效果。
4. **评估数据规模有限**：CT 仅用一名患者的体数据评估，MRI 也仅一个体积——统计显著性有限。
5. **未探索多锚点或分段 Slerp**：对于超长序列（>300层），两端锚点可能不足以精细控制中间区域的相关性，分段插值或多锚点策略值得探索。

## 与相关工作对比

### vs DDS+TV (Chung et al., 2024)
DDS+TV 在 re-noising 后额外执行 TV 正则化优化步来平滑 z 轴——需要调敏感的正则化权重 $\lambda$，且过平滑会擦除细节。ISCS 从 re-noising 噪声本身入手，不需要额外优化步和超参数，在 SVCT 上 Sagittal LPIPS 0.065 vs TV 的 0.088（↓26%），同时避免了卡通化伪影。

### vs BCS (Kwon & Ye, 2025)
BCS 为视频修复设计，对所有帧/层施加完全相同的噪声——在短视频(<16帧)中可行，但在医学体数据(>300层)中产生"复制伪影"。ISCS 的 Slerp 噪声允许层间相关性随距离衰减，更好地适应医学数据的局部连续+全局变化特性。消融实验（Table 2）显示 ISCS 在 Sagittal 上优于 BCS 0.54 dB PSNR、0.008 LPIPS。

### vs DiffusionBlend (Song et al., 2024)
DiffusionBlend 通过 3D patch 训练混合 diffusion score 来增强 3D 一致性——需要额外的 3D 训练成本和特殊的数据处理。ISCS 完全不需要任何训练，仅在推理时修改噪声采样方式，更简洁且通用。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 从根因出发的简洁分析 + 利用高维几何的原理性解法，Slerp 在扩散采样中的应用新颖
- **实验充分度**: ⭐⭐⭐⭐ 三任务(SVCT+LACT+MRI SR) × 两求解器(DDNM+DDS)交叉验证，含消融和轨迹分析
- **写作质量**: ⭐⭐⭐⭐⭐ 从问题定义→根因分析→方案推导→实验验证的逻辑链清晰流畅
- **实用价值**: ⭐⭐⭐⭐⭐ 零额外计算+即插即用+开源代码，对3D医学成像社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](../../CVPR2026/medical_imaging/markerbased_3d_reconstruction_of_aggregates_with_a.md)
- [Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](../../NeurIPS2025/medical_imaging/consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](../../ICML2025/medical_imaging/raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)
- [Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)

<!-- RELATED:END -->
