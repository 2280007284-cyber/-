# 基于 PINN-SR 的高分辨率海流重构（CMEMS + GDP）

你当前的目标是：先把 GDP 做 **ERA5 风滑移矫正 + Stokes 矫正**，再把修正后的 GDP 与 CMEMS 一起输入 PINN-SR，输出高分辨率流场。

## 1) 第一步：GDP 矫正（已提供脚本）

新增脚本：`gdp_correction.py`

### 输入
- GDP 轨迹文件（CSV/Parquet/NetCDF）：至少包含 `lon, lat, time, u, v`
- ERA5 风场（NetCDF）：默认变量名 `u10, v10`
- Stokes 漂移（NetCDF，可选）：默认变量名 `ust, vst`
  - 如果你没有 Stokes 文件，脚本会自动用经验公式估算：`U_stokes ≈ 0.015 × U10`

### 矫正公式

```text
u_corr = u_obs - alpha * u10 - u_stokes
v_corr = v_obs - alpha * v10 - v_stokes
```

其中：
- `alpha`：风滑移系数，默认 `0.007`
- `beta`：Stokes 缩放系数，默认 `1.0`（仅在提供 `--stokes` 数据时生效）
- 若不提供 `--stokes`，则自动使用 `u_stokes=gamma*u10, v_stokes=gamma*v10`，默认 `gamma=0.015`

### 运行示例

```bash
python gdp_correction.py \
  --gdp data/gdp.csv \
  --era5 data/era5_u10v10.nc \
  --out data/gdp_corrected.csv \
  --alpha-wind 0.007 \
  --stokes-from-wind-coeff 0.015
```

### 你给的路径可直接这样运行（Windows）

```bash
python gdp_correction.py
```

> 代码已内置默认路径：  
> `D:\download\uswc_drifter_6hour_2023.nc`  
> `D:\data\wind_2023_uswc.nc`  
> 输出到 `D:\download\uswc_drifter_6hour_2023_corrected.nc`

如果你想手动覆盖默认路径，再用下面命令：

```bash
python gdp_correction.py ^
  --gdp "D:\\download\\uswc_drifter_6hour_2023.nc" ^
  --era5 "D:\\data\\wind_2023_uswc.nc" ^
  --out "D:\\download\\uswc_drifter_6hour_2023_corrected.nc" ^
  --alpha-wind 0.007 ^
  --stokes-from-wind-coeff 0.015
```

如果你的 GDP 或风场变量名不是默认值（`lon/lat/time/u/v` 与 `u10/v10`），请额外传参：
`--gdp-lon-col --gdp-lat-col --gdp-time-col --gdp-u-col --gdp-v-col --era5-u-col --era5-v-col`。

脚本现在也会自动尝试常见 GDP 变量名（例如 `longitude/latitude/ve/vn`）。  
如果仍匹配失败，会在报错里打印可用列名，你把那行报错贴出来我可以直接给你最终参数。

另外脚本会自动把 GDP 时间统一成 **UTC 无时区 datetime64**，避免 `xarray.interp` 对时区对象报错。

输出文件会新增：`u_corr, v_corr` 以及中间项（`u_wind_slip, v_wind_slip, u_stokes, v_stokes`）。

---

## 2) 第二步：PINN-SR 重构

脚本：`pinn_sr_workflow.py`

- 把 `load_gdp_corrected()` 接到第一步输出数据（`u_corr, v_corr`）。
- 把 `load_cmems_background()` 接到 CMEMS 背景流场。
- 训练后调用高分辨率网格推理，输出 `u_hr, v_hr, psi_hr`。

### 当前模板包含
- Fourier 特征 + MLP PINN
- 三项损失：`L_data + L_bg + L_phy`
- 可扩展物理约束（当前内置不可压缩约束）

### 运行

```bash
python pinn_sr_workflow.py
```

> 注意：`pinn_sr_workflow.py` 的数据加载函数仍需你按实际文件格式补齐（xarray/netCDF/CSV）。

## 3) 单独训练脚本（你当前这个 u/v CMEMS 场景）

你这个 `D:\data\uswc_2023_cmems.nc` 是 **u 和 v 都有**，我给了两个独立训练脚本：

- `train_pinn_sr_mlp.py`：PINN-SR（MLP骨干）
- `train_pinn_sr_kan.py`：PINN-SR（KAN风格骨干）

两个脚本都默认读取：  
- CMEMS：`D:\data\uswc_2023_cmems_cleaned.nc`（优先读 `u_clean/v_clean`）  
- GDP：`D:\download\uswc_drifter_6hour_2023_corrected.nc`（优先读 `u_corr/v_corr`）  
并做联合训练 `(u,v)(lon,lat,t)`。
默认会把 **GDP 最后 30 天** 预留为验证集（可用 `--val-days` 调整）。

### 训练 MLP 版本

```bash
python train_pinn_sr_mlp.py --epochs 3000 --batch-size 4096 --val-days 30
# 推荐：开启残差学习（预测相对 CMEMS 的修正量）
python train_pinn_sr_mlp.py --residual-mode --epochs 3000 --batch-size 4096 --val-days 30
```

### 训练 KAN 版本

```bash
python train_pinn_sr_kan.py --epochs 3000 --batch-size 4096 --num-basis 16 --val-days 30
# 推荐：开启残差学习（预测相对 CMEMS 的修正量）
python train_pinn_sr_kan.py --residual-mode --epochs 3000 --batch-size 4096 --num-basis 16 --val-days 30
```

已内置早停（Early Stopping）：`--early-stop-patience 8 --early-stop-min-delta 1e-4 --eval-every 200`。  
会额外保存最佳模型：`pinn_sr_mlp_uv_best.pt` / `pinn_sr_kan_uv_best.pt`。

如果你临时不想用 GDP，可显式关闭：`--gdp ""`。
默认采用“GDP 权重上升 + 物理权重下降”的调度：  
`--lambda-gdp-start 0.2 --lambda-gdp-end 1.0`，  
`--lambda-phy-start 0.05 --lambda-phy-end 0.005`，  
并会自动检查 GDP/CMEMS 速度量级（优先读 units，或按分位数判定；疑似 cm/s 会自动转 m/s）。
残差学习 checkpoint 会被导出脚本自动识别（`residual_mode=True`），并在导出时加回 CMEMS 背景场。
并自动对齐 GDP 经度到 CMEMS 坐标体系（-180~180 或 0~360）。
训练启动时会打印空间匹配度：GDP 点落在 CMEMS 边界框内的比例（`[Match] ... GDP in-box=...`）。
训练内部会对输入和速度标签做标准化，并额外打印 `bg_rmse_mps`（物理单位下的背景场 RMSE）。
其中 `gdp/gdp_val` 误差是“模型在 GDP 轨迹点上的预测”与“矫正后的 GDP 速度（u_corr/v_corr）”之间的误差。

输出模型：
- `pinn_sr_mlp_uv.pt`
- `pinn_sr_kan_uv.pt`
- `pinn_sr_mlp_uv_best.pt`
- `pinn_sr_kan_uv_best.pt`

## 4.5) 把 best 模型导出成“重构流场文件”

新增脚本：`export_reconstruction_from_checkpoint.py`
以及拆分版：
- `export_reconstruction_kan.py`（只导出 KAN）
- `export_reconstruction_mlp.py`（只导出 MLP）
> 两个拆分脚本已做“同目录动态加载”，即使你不在脚本目录下执行，也能找到 `export_reconstruction_from_checkpoint.py`。

现在已内置默认路径，可直接运行（默认同时导出 KAN+MLP 的 best）：  

```bash
python export_reconstruction_from_checkpoint.py
```

默认输出：
- `D:\data\recon_kan.nc`
- `D:\data\recon_mlp.nc`
- 当 `--model-type both` 时，请使用：
  - `--ckpt-kan / --ckpt-mlp`
  - `--out-kan / --out-mlp`
  不能再用单模型参数 `--ckpt / --out / --recon`。

> 注意：`--recon` 是 **compare_reconstruction_vs_cmems.py** 的参数；  
> 在导出脚本里它只是 `--out` 的兼容别名。
> 若 checkpoint 是旧版（仅 `state_dict`，无归一化统计），导出脚本会自动从 `--cmems` 估计归一化参数后继续导出。

示例（KAN）：

```bash
python export_reconstruction_kan.py \
  --ckpt pinn_sr_kan_uv_best.pt \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --out D:\\data\\recon_kan.nc
```

如果显存不够可加：`--batch-size 20000`（脚本也会在 CUDA OOM 时自动回退到 CPU 分批推理）。
如果你感觉“卡住很久”，通常不是死掉，而是在跑大网格推理；可以：
- 先只导出单模型：`--model-type kan`（或 `mlp`）
- 减小 `--batch-size`（显存紧张）或调大（显存充足时更快）
- 利用新参数 `--log-every-batches` 看实时进度
- 旧 checkpoint 触发归一化兜底时，可用 `--max-norm-samples` 控制采样规模加速
- 先做冒烟验证可加 `--max-time-steps 10`，只导出前 10 个时刻
- 若你是 PyTorch 2.6+ 且 checkpoint 来自可信来源，脚本会自动兼容 `weights_only` 变更；若不信任来源可加 `--untrusted-checkpoint` 禁用该回退
- 若你在 IDE 里看不到实时日志，建议用 `python -u ...` 运行（无缓冲输出）
- 若你要输出更高分辨率网格，可加 `--upscale-factor 2`（或更大），在 CMEMS 经纬度范围内细化网格再导出

示例（MLP）：

```bash
python export_reconstruction_mlp.py \
  --ckpt pinn_sr_mlp_uv_best.pt \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --out D:\\data\\recon_mlp.nc
```

## 5) 重构场 vs CMEMS 基线：对验证浮标误差对比

新增脚本：`compare_reconstruction_vs_cmems.py`
以及三方对比脚本：`compare_kan_mlp_vs_cmems.py`（KAN + MLP + CMEMS 同时评估）
> `compare_kan_mlp_vs_cmems.py` 已内嵌全部计算逻辑，不再依赖 `compare_reconstruction_vs_cmems.py`。

用途：
1. 重构流场 vs 浮标验证集误差
2. 原始/清洗后 CMEMS vs 浮标验证集误差
3. 输出两者对比和提升百分比

示例：

```bash
# 使用默认路径（recon_kan/recon_mlp/cmems/gdp）直接评估
python compare_kan_mlp_vs_cmems.py

# 自定义路径
python compare_reconstruction_vs_cmems.py \
  --recon D:\\data\\your_reconstructed_flow.nc \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --gdp D:\\download\\uswc_drifter_6hour_2023_corrected.nc \
  --val-days 30

# 一次性对比 KAN/MLP/CMEMS（推荐）
python compare_kan_mlp_vs_cmems.py \
  --recon-kan D:\\data\\recon_kan.nc \
  --recon-mlp D:\\data\\recon_mlp.nc \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --gdp D:\\download\\uswc_drifter_6hour_2023_corrected.nc \
  --val-days 30 \
  --plot-out D:\\data\\compare_kan_mlp_vs_cmems.png
```

若读取 `.nc` 报 xarray backend 依赖缺失，请先安装：
`pip install netCDF4`（或安装 `h5netcdf/scipy`），
或者把 GDP `.nc` 转成 `csv/parquet` 再传给 `--gdp`。
对比脚本会自动把 GDP / KAN / MLP / CMEMS 速度统一到 m/s（优先读 units，缺失时按量级启发式推断），并打印缩放系数，避免单位不一致导致误差虚高。
如果 GDP 列名不标准，可手动指定：`--gdp-u-col --gdp-v-col --gdp-lon-col --gdp-lat-col --gdp-time-col`。
若数据中含异常填充值/坏点，可用 `--max-abs-speed`（默认 5 m/s）进行物理范围过滤。
脚本会同时汇报 `RMSE_vec`（u/v 向量误差，推荐主指标）和 `RMSE_speed`（仅速度大小误差）。
并自动输出柱状图（默认 `compare_kan_mlp_vs_cmems.png`）。

### 我是不是要先跑 KAN 和 MLP？

- **如果你已经有“重构流场文件”**（`--recon`），**不用先跑** KAN/MLP，直接做第 5 步对比评估。  
- **如果你还没有重构流场文件**，就需要先训练（第 3 步）：  
  - 跑 `train_pinn_sr_mlp.py` 得到 `pinn_sr_mlp_uv_best.pt`  
  - 跑 `train_pinn_sr_kan.py` 得到 `pinn_sr_kan_uv_best.pt`  
  然后用你导出的重构场去做第 5 步评估。  
- 一般建议：**两个都跑**，再用第 5 步统一和 CMEMS 基线对比，选误差更小的模型。

> 当前物理约束采用简化 2D 粘性 NS 形式（u、v 双分量）+ 无散度约束，适合先跑通对比两种骨干网络。

## 5.1) 用 HF 雷达流场做“参考真值”评估（KAN / MLP / CMEMS）

新增脚本：`compare_kan_mlp_cmems_vs_hf.py`

示例（你的目录可直接跑）：

```bash
python compare_kan_mlp_cmems_vs_hf.py \
  --hf-dir D:\\生产力\\py程序\\pythonProject\\hf\\202307_uswc_1km_rtv_sio \
  --recon-kan D:\\data\\recon_kan.nc \
  --recon-mlp D:\\data\\recon_mlp.nc \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --max-abs-speed 5.0 \
  --snapshot-count 3 \
  --snapshot-out-dir D:\\data\\hf_snapshots
```

输出包括：
- KAN vs HF
- MLP vs HF
- CMEMS vs HF
- KAN/MLP 相对 CMEMS 的 `RMSE_vec` 提升百分比
- 小尺度梯度能量（`Small-scale gradient energy`，越接近 HF 越好）
- 流场快照对比图（HF/KAN/MLP/CMEMS，默认首个 HF 文件里自动挑 3 个时刻）

注：该脚本按 HF 文件逐个流式处理，避免一次性拼接整月导致内存爆掉。

## 5.2) 按 SR 四宫格结构导出流场快照图

新增脚本：`plot_sr_snapshots_auto.py`  
会自动选观测点最多的时刻，输出你需要的 2x2 结构图：  
CMEMS(展示到HR网格) / Bilinear / PINN-SR(HR) / 差值图。

示例：

```bash
python plot_sr_snapshots_auto.py \
  --data D:/data/output/train_data_2023.nc \
  --ckpt D:/data/output/checkpoints_sr_strict_weighted/best_sr.pt \
  --out-dir D:/data/output/snapshots_sr_auto \
  --scale 2 --channels 64 --top-n 6 --min-gap-hours 24
```

## 4) 先做 CMEMS 数据清洗（你刚提到的这一步）

新增脚本：`clean_cmems_u.py`（针对你当前 **u/v** CMEMS 文件）

默认输入输出：
- 输入：`D:\data\uswc_2023_cmems.nc`
- 输出：`D:\data\uswc_2023_cmems_cleaned.nc`

直接运行：

```bash
python clean_cmems_u.py
```

可调参数示例：

```bash
python clean_cmems_u.py --max-abs-speed 3.0 --interp-limit 8 --smooth-window 3
```

清洗内容：时间排序、异常值剔除（|u|/|v|>阈值）、缺测插值（time→lat→lon）、可选时间平滑，输出 `u_clean,v_clean`。
