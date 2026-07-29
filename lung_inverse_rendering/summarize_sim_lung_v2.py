"""Create an auditable Markdown summary from frozen sim_lung_v2 result files."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sim_lung_v2"


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    oracle = read("metrics_oracle.json")
    noisy = read("metrics_noisy.json")
    tracks0 = read("metrics_image_tracks_px_0.00.json")
    tracks005 = read("metrics_image_tracks_px_0.05.json")
    tracks025 = read("metrics_image_tracks.json")
    force = read("metrics_noisy_force_p0.10.json")
    single = read("metrics_noisy_loads_1.json")
    pose = read("metrics_image_tracks_pose_p1.0_px_0.00.json")
    unknown = read("metrics_unknown_region.json")
    visco = read("metrics_viscoelastic.json")
    lines = [
        "# sim_lung_v2 全部扩展实验",
        "",
        "## 核心材料辨识",
        "",
        "| 协议 | 背景 E 中位误差 | 模量比中位误差 |",
        "|---|---:|---:|",
        f"| 3D 表面运动，无噪声 oracle | {percent(oracle['E_background_median_relative_error'])} | {percent(oracle['inclusion_ratio_median_relative_error'])} |",
        f"| 3D 表面运动，噪声 σ=0.00025 | {percent(noisy['E_background_median_relative_error'])} | {percent(noisy['inclusion_ratio_median_relative_error'])} |",
        f"| 图像平面对应轨迹，无噪声 | {percent(tracks0['E_background_median_relative_error'])} | {percent(tracks0['inclusion_ratio_median_relative_error'])} |",
        f"| 图像平面对应轨迹，0.05 px 噪声 | {percent(tracks005['E_background_median_relative_error'])} | {percent(tracks005['inclusion_ratio_median_relative_error'])} |",
        f"| 图像平面对应轨迹，0.25 px 噪声 | {percent(tracks025['E_background_median_relative_error'])} | {percent(tracks025['inclusion_ratio_median_relative_error'])} |",
        "",
        "图像平面轨迹使用已知 FEM 节点对应关系，不是从 RGB 自动计算的光流。结果证明无噪投影保留了信息，但亚像素误差会迅速破坏异质参数辨识。",
        "",
        "## 敏感性",
        "",
        "| 扰动 | 背景 E 中位误差 | 模量比中位误差 | 结论 |",
        "|---|---:|---:|---|",
        f"| +10% 系统力尺度误差 | {percent(force['E_background_median_relative_error'])} | {percent(force['inclusion_ratio_median_relative_error'])} | 对力标定高度敏感 |",
        f"| 仅 1 个压入实验 | {percent(single['E_background_median_relative_error'])} | {percent(single['inclusion_ratio_median_relative_error'])} | 患者间不稳定，多方向载荷必要 |",
        f"| 无噪轨迹 + 1° 相机 yaw 误差 | {percent(pose['E_background_median_relative_error'])} | {percent(pose['inclusion_ratio_median_relative_error'])} | 当前相对轨迹定义下影响很小 |",
        "",
        "## 未知区域与黏弹性",
        "",
        f"- 未知区域中心误差：几何尺度的 {percent(unknown['center_error_normalized_median'])}；",
        f"- 未知区域半径误差：{percent(unknown['radius_relative_error_median'])}；",
        f"- 区域未知后背景 E/模量比误差：{percent(unknown['E_background_relative_error_median'])}/{percent(unknown['inclusion_ratio_relative_error_median'])}；",
        f"- 一阶加载—保持—卸载时间常数 τ 中位误差：{percent(visco['tau_median_relative_error'])}。",
        "",
        "黏弹结果以已知 FEM 平衡响应为条件，尚未与弹性参数联合反演；当前模型是 `τ·du/dt + u = u_equilibrium` 的降阶模型，不是完整有限应变纤维黏弹本构。",
        "",
        "## 结论与证据边界",
        "",
        "1. 在几何、力、接触、边界和区域已知且运动足够精确时，患者特异 E 与区域模量比在合成数据中可辨识。",
        "2. 最大现实瓶颈不是 FEM 求解，而是力标定和运动测量精度；0.05 px 轨迹噪声已使结果明显退化。",
        "3. 多位置、多方向载荷比单次压入稳定；未知区域可以粗定位，但当前仅有 2 名合成测试患者。",
        "4. 尚未验证真实 CT、自动 RGB 光流、未知边界、各向异性纤维或真实临床绝对模量。",
        "",
        "所有指标均为描述性结果，不应在仅 2 名测试患者上给出总体统计推断。",
    ]
    (RESULTS / "FULL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(RESULTS / "FULL_REPORT.md")


if __name__ == "__main__":
    main()
