# 受电弓 4 关键点倾斜角度几何解算与异常判定技术规范 (V2.0)
*基于下横梁局部浮动基准系与双层解耦诊断架构*

本文档详述基于 HRNet 预测的 4 个刚性关键点，进行受电弓 2D 几何解算、消除相机安装滚转角误差、双层解耦诊断（内部结构性平行度失效 vs 整弓刚体倾斜）的完整数学原理与工程实现。

---

## 一、 核心痛点与局部参考系设计理念

### 1. 传统图像绝对角算法的工业痛点
在车载高速巡检环境中，如果直接计算连线相对于**图像水平 X 轴**的绝对斜率并求平均（旧版 $\theta_{\text{fused}}$），系统在工程应用中会产生大量误报警：
1. **相机安装滚转偏置（Camera Roll Bias）**：巡检相机底座受机械螺栓紧固公差影响，通常存在 $1.0^\circ \sim 3.0^\circ$ 的安装倾角偏置；
2. **线路过弯超高角（Superelevation / Cant）**：高速列车驶入曲线区段时，轨道外轨抬高，车体、相机和受电弓整体随之产生 $2^\circ \sim 6^\circ$ 的刚体侧滚；
3. **车体横向振颤**：动态受流时车顶横向颠簸导致整弓绝对斜率频繁抖动。

这些因素都会导致上滑板角 $\theta_{\text{horn}}$ 和下横梁角 $\theta_{\text{support}}$ 的**绝对值**同时发生偏移。**这种整体偏移并不代表受电弓发生了结构故障。**

### 2. 解决方案：下横梁浮动参考系（Local Reference Frame）
受电弓弓头机械结构设计为平行四边形/刚性支撑架，**下横梁（$P_2-P_3$）直接与上下臂杆铰接，是弓头装配的刚性基座**。
因此：
- 将**下横梁（$P_2 \rightarrow P_3$）作为局部浮动坐标系的基准 $X_{\text{local}}$ 轴**；
- 测量**上滑板（$P_1 \rightarrow P_4$）在该局部基准中的相对夹角 $\theta_{\text{rel}}$**；
- $\theta_{\text{rel}}$ 具备严格的 **$SO(2)$ 刚体旋转不变性**，彻底剥离了相机滚转角、车体侧滚角和弯道超高角的影响！

---

## 二、 4 关键点拓扑结构与坐标系定义

```
         P1 (左羊角/滑板端) ================================ P4 (右羊角/滑板端)
             \                                               /
              \--- P2 (左支撑铰点) ======= P3 (右支撑铰点) ---/
```

| 序号 | 关键点标识 | 物理结构解剖位置 | 角色与几何特性 |
| :---: | :--- | :--- | :--- |
| **P1** | `p1_left_horn` | 碳滑板左端外沿（左减震立柱上方） | 上滑板向量起点，避开滑磨磨损区，外侧基线跨度大 |
| **P2** | `p2_left_support` | 下横梁与左 U 形支架 T 型交汇铰点 | **局部基准系起点**，角点对比度强，刚性受力铰接点 |
| **P3** | `p3_right_support`| 下横梁与右 U 形支架 T 型交汇铰点 | **局部基准系方向点**，角点对比度强，刚性受力铰接点 |
| **P4** | `p4_right_horn` | 碳滑板右端外沿（右减震立柱上方） | 上滑板向量终点，避开滑磨磨损区，外侧基线跨度大 |

---

## 三、 几何解算严密数学推导

### 1. 向量定义
在图像平面（$X$ 向右递增，$Y$ 向下递增）中构建两组二维几何向量：
- **下横梁局部基准向量（Local Base Vector）**：
  $$\vec{v}_{\text{base}} = \vec{P_2 P_3} = \begin{bmatrix} x_3 - x_2 \\ y_3 - y_2 \end{bmatrix} = \begin{bmatrix} \Delta x_{\text{base}} \\ \Delta y_{\text{base}} \end{bmatrix}$$

- **上滑板目标向量（Horn Working Vector）**：
  $$\vec{v}_{\text{horn}} = \vec{P_1 P_4} = \begin{bmatrix} x_4 - x_1 \\ y_4 - y_1 \end{bmatrix} = \begin{bmatrix} \Delta x_{\text{horn}} \\ \Delta y_{\text{horn}} \end{bmatrix}$$

---

### 2. 第一层指标：弓头内部相对平行度失效角（$\theta_{\text{rel}}$）
通过二维向量的点积（内积）与叉积（外积，即伪标量二维外积）解算上滑板相对下横梁的有符号旋转夹角：

#### (1) 二维点积（反映共线投影与模长乘积）：
$$\text{Dot} = \vec{v}_{\text{base}} \cdot \vec{v}_{\text{horn}} = \Delta x_{\text{base}} \cdot \Delta x_{\text{horn}} + \Delta y_{\text{base}} \cdot \Delta y_{\text{horn}}$$

#### (2) 二维叉积（反映相对旋转方向与正弦分量）：
$$\text{Cross} = \vec{v}_{\text{base}} \times \vec{v}_{\text{horn}} = \Delta x_{\text{base}} \cdot \Delta y_{\text{horn}} - \Delta y_{\text{base}} \cdot \Delta x_{\text{horn}}$$

#### (3) 相对平行度夹角公式：
$$\theta_{\text{rel}} = \operatorname{atan2}(\text{Cross}, \ \text{Dot}) \times \frac{180^\circ}{\pi}$$

#### (4) 物理意义与符号准则：
- **$\theta_{\text{rel}} = 0^\circ$**：上滑板与下横梁绝对平行（理想平行四边形悬挂几何态）；
- **$\theta_{\text{rel}} > 0^\circ$**：上滑板相对下梁**顺时针偏折（右端相对下塌，RIGHT_DROP）**；
- **$\theta_{\text{rel}} < 0^\circ$**：上滑板相对下梁**逆时针偏折（左端相对下塌，LEFT_DROP）**。
- **数学性质**：当全图刚体旋转任意角度 $\alpha$ 时，$\text{Dot}$ 与 $\text{Cross}$ 的值严格不变，故 $\theta_{\text{rel}}$ 具有**旋转不变量性质**。

---

### 3. 第二层指标：整弓刚体偏角与基线姿态（$\Delta\theta_{\text{rigid}}$）
为了监测“受电弓整体底座歪斜 / 升弓倾覆”这类全局故障，同时保留下横梁对相机参考系的观测，并去除静态安装偏置：

#### (1) 下横梁绝对倾角：
$$\theta_{\text{support}} = \operatorname{arctan2}(\Delta y_{\text{base}}, \Delta x_{\text{base}}) \times \frac{180^\circ}{\pi}$$

#### (2) 整弓刚体倾角增量（去相机安装偏置）：
$$\Delta\theta_{\text{rigid}} = \theta_{\text{support}} - \theta_{\text{install\_bias}}$$
> 注：$\theta_{\text{install\_bias}}$ 为相机静态安装滚转角偏置（通过列车静止标定或大样本中位数获取，在本项目测试环境中约为 $+2.08^\circ$）。

---

## 四、 纯条件 A 故障诊断体系（免疫摄像头偏移）

在工业车载运行中，由于摄像头底座机械振动、支架松动或车体在曲线轨道的超高侧滚，会导致整弓基线绝对角（$\theta_{\text{support}}$）产生假性大幅漂移。如果将基线角作为受电弓故障判据，会导致频繁的虚警。

因此，**受电弓倾斜故障判定（ALARM / WARNING）严格仅以【条件 A：弓头相对平行度偏角 $\theta_{\text{rel}}$】为唯一判据**，下横梁基线角仅作为相机健康度辅助记录：

| 诊断层级 | 核心指标 | 计算公式 | 预警门限 | 告警门限 | 角色与工程作用 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **【唯一故障判据】条件 A: 弓头平行度失效** | **相对偏角 $\theta_{\text{rel}}$** | $\operatorname{atan2}(\text{Cross}, \text{Dot})$ | **$\pm 1.2^\circ$** | **$\pm 1.8^\circ$** | **触发 ALARM_TILT 告警！针对滑板扭折、弹簧盒断裂、羊角挂扯（完全免疫摄像头偏移）** |
| **【辅助观测指标】条件 B: 下梁基线动态角度** | **基线角度 $\theta_{\text{support}}$** | $\operatorname{atan2}(\Delta y_{\text{base}}, \Delta x_{\text{base}})$ | — | — | **仅作巡检相机位姿健康度参考日志（排查摄像头是否松动偏转），不触发受电弓故障报警** |
| **【抗干扰门控】光电眩光过滤** | **最低置信度 $\text{Conf}_{\min}$** | $\min(c_1, c_2, c_3, c_4)$ | — | **$< 0.35$** | **离线强电弧眩光过曝（触发自动剔除，防止误报）** |

---

## 五、 实测数据推导对比（`/home/ibd/jyb/pantograph_tilt.png`）

使用训练完成的 HRNet-W32 模型（`weights/best_hrnet_w32_panto.pth`）在 2048×1536 分辨率测试图上的实测推理关键点：

- $P_1$ (`p1_left_horn`): $(592.0, 432.0)$
- $P_2$ (`p2_left_support`): $(773.3, 496.0)$
- $P_3$ (`p3_right_support`): $(1168.0, 506.7)$
- $P_4$ (`p4_right_horn`): $(1349.3, 464.0)$

### 1. 向量构建：
- $\vec{v}_{\text{base}} = (1168.0 - 773.3, \; 506.7 - 496.0) = (\mathbf{394.7\text{ px}}, \; \mathbf{10.7\text{ px}})$
- $\vec{v}_{\text{horn}} = (1349.3 - 592.0, \; 464.0 - 432.0) = (\mathbf{757.3\text{ px}}, \; \mathbf{32.0\text{ px}})$

### 2. 内积与外积数值计算：
- $\text{Dot} = 394.7 \times 757.3 + 10.7 \times 32.0 = 298904.5 + 342.4 = \mathbf{299246.9}$
- $\text{Cross} = 394.7 \times 32.0 - 10.7 \times 757.3 = 12630.4 - 8103.1 = \mathbf{+4527.3}$

### 3. 局部相对偏角求解（条件 A）：
$$\theta_{\text{rel}} = \operatorname{atan2}(4527.3, \ 299246.9) \times \frac{180^\circ}{\pi} = \mathbf{+0.87^\circ}$$

### 4. 新旧算法诊断对比：
- **旧版算法（绝对角平均）**：$\theta_{\text{fused}} = \frac{2.42^\circ + 1.55^\circ}{2} = +1.98^\circ \ge 1.8^\circ \implies$ **误报 `ALARM_TILT`（虚警！）**；
- **新版算法（纯条件 A 判据）**：$\theta_{\text{rel}} = +0.87^\circ < 1.2^\circ \implies$ **精准判定为 `NORMAL`（上横板与下横梁平行健康！）**。

---

## 六、 生产环境 Python 完整参考实现代码

```python
import numpy as np

def compute_pantograph_tilt_v2(
    kpts,
    structural_warning_thresh=1.2,
    structural_alarm_thresh=1.8,
    install_bias_deg=2.0,
):
    """
    基于下横梁局部基准系的受电弓倾斜解算算法（严格以条件 A 为唯一告警依据）
    
    参数:
      kpts: shape (4, 2) 或 (4, 3) 的关键点数组 [P1, P2, P3, P4]
            P1: 左羊角, P2: 左支撑, P3: 右支撑, P4: 右羊角
      structural_warning_thresh: 相对平行度预警阈值 (度)
      structural_alarm_thresh:   相对平行度告警阈值 (度)
      install_bias_deg:          相机静态安装滚转偏置角 (度，仅作日志参考)
      
    返回:
      dict 包含相对平行度倾角、整弓基线角、诊断等级与状态描述
    """
    coords = np.array(kpts, dtype=np.float64)[:, :2]
    p1, p2, p3, p4 = coords[0], coords[1], coords[2], coords[3]

    # 1. 构造局部基准向量与工作滑板向量
    v_base = p3 - p2  # 下横梁 P2 -> P3 (局部 X 基准轴)
    v_horn = p4 - p1  # 上滑板 P1 -> P4

    # 2. 向量点积与二维叉积 (消除相机与车体倾斜)
    dot = float(v_base[0] * v_horn[0] + v_base[1] * v_horn[1])
    cross = float(v_base[0] * v_horn[1] - v_base[1] * v_horn[0])
    theta_rel = float(np.degrees(np.arctan2(cross, dot)))

    # 3. 计算辅助观测角度 (仅供参考日志，不作为报警依据)
    theta_support = float(np.degrees(np.arctan2(v_base[1], v_base[0])))
    theta_horn = float(np.degrees(np.arctan2(v_horn[1], v_horn[0])))
    theta_rigid = theta_support - install_bias_deg

    # 4. 纯条件 A 判定受电弓结构状态 (完全免疫摄像头偏移与振动)
    abs_rel = abs(theta_rel)

    direction = "RIGHT_DROP" if theta_rel > 0 else "LEFT_DROP"
    if abs_rel < structural_warning_thresh:
        direction = "PARALLEL"

    is_alarm = (abs_rel >= structural_alarm_thresh)
    is_warning = (abs_rel >= structural_warning_thresh) and not is_alarm

    if is_alarm:
        status = "ALARM_TILT"
        severity = "CRITICAL"
    elif is_warning:
        status = "WARNING_TILT"
        severity = "WARNING"
    else:
        status = "NORMAL"
        severity = "NORMAL"

    return {
        "status": status,
        "severity": severity,
        "is_alarm": is_alarm,
        "tilt_direction": direction,
        "relative_tilt_deg": round(theta_rel, 2),       # Layer 1: 弓头内部平行度偏角
        "rigid_tilt_deg": round(theta_rigid, 2),         # Layer 2: 整弓刚体偏角增量
        "support_angle_deg": round(theta_support, 2),   # 下横梁相对图像绝对角
        "horn_angle_deg": round(theta_horn, 2),         # 上滑板相对图像绝对角
        "horn_drop_px": round(float(v_horn[1]), 1),     # 上滑板垂直落差 (px)
        "support_drop_px": round(float(v_base[1]), 1),  # 下横梁垂直落差 (px)
    }
```
