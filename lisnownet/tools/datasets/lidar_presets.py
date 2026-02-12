"""
Fixed inclination (elevation) angles for common lidar setups, in radians.
Use with CustomBin via --lidar <name> in eval; no interpolation.
"""
import numpy as np

# CADC / VLP-32C factory angles (deg) -> rad
_CADC_DEG = [
    -25.010, -15.639, -11.311, -8.843, -7.255, -6.148, -5.334, -4.667,
    -4.000, -3.667, -3.334, -3.000, -2.667, -2.333, -2.001, -1.667,
    -1.333, -1.000, -0.667, -0.333, -0.000, 0.332, 0.667, 1.000,
    1.332, 1.667, 2.333, 3.333, 4.667, 7.000, 10.334, 15.000,
]
# WADS / HDL-64 style factory angles (deg) -> rad
_WADS_DEG = [
    -24.937, -18.929, -13.970, -13.014, -12.046, -11.072, -10.085, -9.100,
    -8.099, -7.103, -6.101, -5.938, -5.766, -5.605, -5.431, -5.269,
    -5.097, -4.932, -4.760, -4.598, -4.425, -4.261, -4.090, -3.924,
    -3.752, -3.588, -3.415, -3.250, -3.080, -2.913, -2.740, -2.576,
    -2.405, -2.238, -2.068, -1.900, -1.728, -1.562, -1.391, -1.224,
    -1.053, -0.885, -0.715, -0.548, -0.377, -0.209, -0.040, 0.129,
    0.297, 0.468, 0.635, 0.806, 0.973, 1.144, 1.311, 1.482,
    1.648, 1.820, 1.988, 3.000, 5.017, 8.019, 10.992, 14.842,
]

# (name, inc_rad, description)
LIDAR_PRESETS = {
    # 32-beam (VLP-32C / CADC)
    "cadc32": (np.deg2rad(np.array(_CADC_DEG)), "CADC / VLP-32C, 32 beams, -25° to +15° (factory)"),
    "vlp32_linear": (np.deg2rad(np.linspace(-25, 15, 32)), "32 beams, linear -25° to +15°"),
    "linear32_24_3": (np.deg2rad(np.linspace(-24, 3, 32)), "32 beams, linear -24° to +3°"),

    # 64-beam
    "wads64": (np.deg2rad(np.array(_WADS_DEG)), "WADS / HDL-64 style, 64 beams (factory)"),
    "linear64_24_3": (np.deg2rad(np.linspace(-24, 3, 64)), "64 beams, linear -24° to +3°"),
    "ouster64": (np.deg2rad(np.linspace(-22.5, 22.5, 64)), "64 beams, linear ±22.5° (OS1-64 style)"),
    "linear64_17_17": (np.deg2rad(np.linspace(-17, 17, 64)), "64 beams, linear ±17°"),

    # 16-beam (VLP-16)
    "vlp16": (np.deg2rad(np.linspace(-15, 15, 16)), "VLP-16, 16 beams, linear ±15°"),

    # 32-beam other spans
    "linear32_15_15": (np.deg2rad(np.linspace(-15, 15, 32)), "32 beams, linear ±15°"),
    "ouster32": (np.deg2rad(np.linspace(-22.5, 22.5, 32)), "32 beams, linear ±22.5° (OS1-32 style)"),
}


def get_inc(name: str):
    if name not in LIDAR_PRESETS:
        raise ValueError(
            f"Unknown --lidar '{name}'. Choices: {list(LIDAR_PRESETS.keys())}"
        )
    return LIDAR_PRESETS[name][0]


def list_presets():
    for name, (inc, desc) in LIDAR_PRESETS.items():
        deg = np.rad2deg(inc)
        print(f"  {name:20s}  {len(inc)} beams  [{deg.min():.1f}°, {deg.max():.1f}°]  {desc}")
