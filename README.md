# 海上搜救漂流轨迹预测（PINN）项目模板

本仓库用于开展基于 **Physics-Informed Neural Networks (PINN)** 的海上搜救漂流轨迹预测研究，融合 ERA5 风场与 HYCOM 海流数据，支持从数据处理、模型训练、实验评估到专利材料沉淀的完整流程。

## 1. 项目目标
- 预测海上目标在给定时间窗口内的漂流轨迹。
- 对比传统基线方法与 PINN 方法的效果差异。
- 输出可复现实验结果与可用于专利申请的技术文档。

## 2. 推荐研究设置（当前建议）
- 目标区域：**台湾海峡**（经度 117.0E–122.5E，纬度 22.0N–26.5N）
- 推荐时间范围：**2021-01-01 至 2025-12-31**
- 时间分辨率：**1h**

> 详细约束、切分方案与风险应对见 `project_scope.md`。

## 3. 目录结构
```text
.
├── data/
│   ├── raw/                  # 原始数据（ERA5, HYCOM, 轨迹）
│   └── processed/            # 清洗/重采样后数据
├── src/
│   └── models/               # 模型代码（PINN等）
├── scripts/                  # 数据构建、训练、评估脚本
├── experiments/              # 实验配置和日志
├── figures/                  # 可视化图表
├── notebooks/                # 探索性分析
├── patent/                   # 专利草稿与权利要求
├── project_scope.md          # 研究范围定义
├── weekly_plan.md            # 8周执行计划
├── requirements.txt          # Python依赖
└── environment.yml           # Conda环境
```

## 4. 快速开始
### 4.1 Conda 环境（推荐）
```bash
conda env create -f environment.yml
conda activate sar-pinn
```

### 4.2 Pip 环境
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. 建议工作流
1. 确认 `project_scope.md` 中的区域、时间范围和切分方案。
2. 先运行 `scripts/fetch_real_data.py` 下载 ERA5/HYCOM，再运行 `scripts/build_dataset.py` 生成训练/验证/测试集。
3. 完成 `src/models/pinn.py` 并运行训练。
4. 运行 `scripts/evaluate.py` 与基线对比。
5. 在 `patent/` 中沉淀技术方案和权利要求。

## 6. 版本管理建议
- `main`: 稳定版本。
- `feat/*`: 功能开发分支（如 `feat/data-pipeline`）。
- 提交信息建议使用：
  - `feat:` 新功能
  - `fix:` 缺陷修复
  - `docs:` 文档更新
  - `chore:` 维护工作

## 7. 下一步（真实数据接入版）
- [ ] 下载真实数据（ERA5 + HYCOM）：
  ```bash
  python scripts/fetch_real_data.py --start-date 2024-01-01 --end-date 2024-03-31
  ```
- [ ] 使用真实数据构建训练集：
  ```bash
  python scripts/build_dataset.py --data-source real --start-date 2024-01-01 --end-date 2024-03-31
  ```
- [ ] 若仅需离线联调，可回退到合成模式：
  ```bash
  python scripts/build_dataset.py --data-source synthetic
  ```
- [ ] 在 `weekly_plan.md` 按实际日期落地每周里程碑。
