---
title: >-
  [论文解读] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals
description: >-
  [CVPR 2026][自动驾驶][毫米波通信] 提出 Rascene，一种利用毫米波 OFDM 通信信号（5G/Wi-Fi）进行高保真 3D 场景成像的集成感知与通信（ISAC）框架，通过置信度加权的多帧融合实现对稀疏、多径干扰的射频观测的几何一致性恢复。 3D 环境感知对自动驾驶和机器人导航至关重要…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "毫米波通信"
  - "3D场景成像"
  - "OFDM信号"
  - "多帧融合"
  - "ISAC"
---

# Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals

**会议**: CVPR 2026  
**arXiv**: [2604.02603](https://arxiv.org/abs/2604.02603)  
**代码**: 无  
**领域**: 自动驾驶 / 3D感知 / 集成感知与通信  
**关键词**: 毫米波通信, 3D场景成像, OFDM信号, 多帧融合, ISAC

## 一句话总结

提出 Rascene，一种利用毫米波 OFDM 通信信号（5G/Wi-Fi）进行高保真 3D 场景成像的集成感知与通信（ISAC）框架，通过置信度加权的多帧融合实现对稀疏、多径干扰的射频观测的几何一致性恢复。

## 研究背景与动机

3D 环境感知对自动驾驶和机器人导航至关重要。现有主流方案存在明显局限：
- **相机**：受光照条件严格约束，在烟雾、雾天等恶劣环境下失效
- **LiDAR**：昂贵、体积大、功耗高，同样受恶劣天气影响
- **专用雷达**：虽可穿透障碍物，但需要超宽带硬件（多 GHz 带宽）和专用频谱许可，成本高、可扩展性差

核心洞察：毫米波通信设备（如 5G 和 Wi-Fi）已广泛部署，其 OFDM 波形天然包含距离和角度信息。如果能复用这些已有的通信信号进行感知，就可以在不增加专用传感硬件和频谱许可的前提下实现低成本、可扩展的 3D 感知。

关键发现：商用毫米波设备可以全双工模式进行单站感知——由于相控阵天线的高方向性和短载波波长，发射 / 接收路径之间具有足够的射频隔离。

## 方法详解

### 整体框架

Rascene 想做的事很反直觉：不加任何专用雷达，只复用已经满街都是的毫米波通信设备（5G/Wi-Fi），就还原出高保真的 3D 场景。整条链路分两段——先用通信设备的全双工单站能力，从 OFDM 波形里提取 CIR 和角度信息，把每帧射频观测变成稀疏的 3D 射频点云；再用一个多帧成像网络，对 N 帧带已知位姿的点云 $\mathcal{S} = \{\mathbf{S}_i\}_{i=1}^N$、$\mathcal{G} = \{\mathbf{G}_i\}_{i=1}^N$ 做置信度加权的前向投影融合，学习映射 $\mathcal{F}$ 输出密集体素网格 $\hat{\mathbf{V}}_r$ 和深度图 $\hat{\mathbf{D}}_r$。网络内部先由共享编码器把每帧点云编成特征体积与置信度，再做变形融合，最后由粗到细解码器密化。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["N 帧 OFDM 通信信号 + 已知位姿"] --> B["全双工单站感知<br/>CIR 测距 + 相控阵测角 → 每帧稀疏射频点云"]
    B --> C["共享编码器<br/>每帧 → 特征体积 + 置信度 logits"]
    C --> D["空间自适应变形与融合<br/>刚性变换 warp 到参考帧 + 源驱动前向投影"]
    D -->|各向同性高斯核 + 置信度加权| E["统一融合表示 Z"]
    E --> F["粗到细 3D 解码器<br/>逐级密化成稠密特征体积"]
    F --> G["体素占用网格"]
    F --> H["深度图"]
```

### 关键设计

**1. 全双工单站感知：让一台通信设备同时当发射机和接收机来测距**

专用雷达要超宽带硬件和专用频谱，成本和可扩展性都差；Rascene 转而让商用毫米波设备一边发一边收 OFDM 信号自己感知自己。能这么做的物理前提是相控阵天线方向性强、载波波长短，收发路径之间有足够射频隔离，加上收发天线共位的时钟同步，CIR 就能直接用来测距——距离 $r = nc/(2B)$。再配上相控阵的角度估计（波束赋形权重 $w_{i,j}(\theta,\phi)$），每帧射频数据就被转成球坐标系下的 3D 点云 $\mathbf{S}$。

**2. 空间自适应变形与融合：用源驱动前向投影把稀疏的多帧射频拼成稠密几何**

单帧射频观测稀疏又被多径干扰，传统按目标体素去"查询"源数据的方式会在大片空白区域反复采样、浪费在没信息的地方。Rascene 反过来做源驱动前向投影：每个源体素经刚性变换映射到参考帧坐标系，再用各向同性高斯核 $K_\sigma$ 在局部支撑区域内分配贡献；融合权重同时考虑几何邻近度和学到的置信度（经 softplus 映射后取 $\eta$ 次方来控制锐度），最终统一表示 $\mathbf{Z}_r$ 由归一化加权平均得到。

这样稀疏但信息丰富的射频响应被保留下来，$\eta$ 调高时高置信度的几何信号会主导融合，多径产生的幻觉结构则被压低——这正是多帧能补漏检、抑幻觉的来源。

**3. 粗到细 3D 解码器：把稀疏融合表示逐级密化成场景体积**

编码器和解码器都用 4 层卷积（通道倍数 1,2,4,8），并在每个编码器阶段后插入阶段级的变形和融合，让融合不只发生在一个尺度上。解码器把稀疏的融合表示逐步密化成稠密特征体积，最后两个任务头分别预测体素占用和深度图。

### 损失函数 / 训练策略

总损失是体素损失和深度损失的加权和，且窗口内每一帧都轮流当参考帧、累加所有参考帧的损失：

$$\mathcal{L} = \sum_{r=1}^N (\lambda_v \mathcal{L}_{\text{voxel}}^{(r)} + \lambda_d \mathcal{L}_{\text{depth}}^{(r)})$$

其中体素损失为预测网格与 GT 的二元交叉熵（BCE），深度损失为预测深度图与 GT 的 L1 损失。硬件原型：60 GHz 频段，1.2288 GHz 带宽，16 Tx + 16 Rx 天线，有效感知距离 7 米，FoV 120°×60°，体素网格 64×64×32（12 cm 分辨率）。

## 实验关键数据

### 主实验

数据集：20 个室内环境，12 个训练 / 8 个测试（跨场景泛化评估）。

| 方法 | 帧数 | AbsRel | MAE(cm) | CD(cm) | CD_Diag(%) |
|------|------|--------|---------|--------|------------|
| PanoRadar | 1 | 14.7% | 34.1 | 32.2 | 3.8% |
| CartoRadar | 5 | — | — | 26.8 | 3.1% |
| **Rascene** | 1 | 14.1% | 32.9 | 31.6 | 3.6% |
| **Rascene** | 5 | **9.4%** | **20.2** | **19.7** | **2.3%** |

跨场景泛化平均：AbsRel 9.4%，MAE 20.2cm，RMSE 38.0cm，CD 19.7cm，CD_Diag 2.3%。

### 消融实验

| 融合帧数 | AbsRel | MAE(cm) | CD(cm) | CD_Diag(%) |
|----------|--------|---------|--------|------------|
| 1 | 14.1% | 32.9 | 31.6 | 3.6% |
| 2 | 11.1% | 24.6 | 26.0 | 3.0% |
| 3 | 9.8% | 21.8 | 21.9 | 2.5% |
| 5 | 9.4% | 20.2 | 19.7 | 2.3% |

位姿鲁棒性测试：对平移扰动高度稳定（15cm 扰动几乎无影响），对旋转更敏感（5°-10° 旋转误差导致显著退化，10° 时 CD_Diag 从 2.3% 升至 3.6%）。

### 关键发现

- 从 1 帧到 2 帧的提升最为显著，表明即使一个额外视角也能提供强几何约束
- 中值绝对深度误差仅 6.1cm，90% 像素误差低于 37.6cm
- 多帧融合有效抑制幻觉结构并补充漏检区域
- 即使在 LiDAR 因吸收/镜面反射而失败的区域（如深色地毯、玻璃），Rascene 仍能恢复连贯的场景几何

## 亮点与洞察

- **范式创新**：首次证明 OFDM 通信信号可支持高保真 3D 成像，无需专用感知硬件或频谱许可
- **源驱动融合**优于目标驱动融合——避免重复采样空白区域，更好保留稀疏但信息丰富的射频响应
- **互补性**：RF 感知对光学材料失效模式（低反照率表面吸收、光滑材料镜面反射）具有天然鲁棒性，与 LiDAR 形成互补
- 置信度锐度参数 $\eta$ 的设计允许融合过程被高置信度几何信号主导

## 局限与展望

- 感知范围仅 7 米，适用于室内场景，室外大范围场景有待验证
- 需要已知的 6-DoF 位姿信息（目前依赖外部 IMU）
- 角度估计分辨率受天线阵列规模限制（当前 16×16）
- 仅在室内环境评估，真实户外自动驾驶场景的泛化能力未知
- 多径干扰虽被融合抑制，但极端多径场景（如高度复杂遮挡）可能仍具挑战

## 相关工作与启发

- **与 PanoRadar/CartoRadar 对比**：这些方法依赖 FMCW 雷达专用硬件，而 Rascene 复用通信设备
- **与 NeRF/多视图重建对比**：视觉方法依赖纹理丰富的 RGB 图像，而 RF 观测既不纹理丰富也非几何显式
- **ISAC 趋势**：集成感知与通信是 6G 研究热点，Rascene 为此提供了 3D 成像的具体范例
- **启发**：前向投影 + 置信度加权的融合策略可推广到其他稀疏多视图重建任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次将毫米波通信信号用于高保真 3D 场景成像，提出全双工单站感知 + 源驱动融合的完整系统
- **实验充分度**: ⭐⭐⭐⭐ — 20 个室内环境的跨场景评估充分，消融详细；但缺少室外和更大规模场景验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，物理原理和系统设计的阐述专业完整
- **价值**: ⭐⭐⭐⭐⭐ — 为低成本、可扩展的 3D 感知开辟了全新路径，对 ISAC 和自动驾驶领域均有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[CVPR 2026\] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation](hg-lane_high-fidelity_generation_of_lane_scenes_under_adverse_weather_and_lighti.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[ICCV 2025\] Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging](../../ICCV2025/autonomous_driving/free-running_vs_synchronous_single-photon_lidar_for_high-flux_3d_imaging.md)

</div>

<!-- RELATED:END -->
