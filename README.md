# 电商用户增长与补贴策略评估

面向抖音电商用户增长数据分析岗位的独立作品集：从增长指标、人货场分层，到补贴 A/B 实验、Uplift 定向和增量 ROI。

**默认使用合成数据。没有抖音内部数据、真实线上实验或真实业务收益。**

## 运行

Python 3.11–3.13。先用 `python --version` 检查版本；固定依赖不支持 Python 3.14，需选用3.11–3.13解释器。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m growth.run --users 60000 --seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux 使用 `.venv/bin/python`。无需 Hive 集群即可跑默认实验。

`requirements-lock.txt` 记录验证环境的完整依赖版本，可用它替代 `requirements.txt` 安装。

可选真实行为日志（五列、无表头的天池UserBehavior CSV）：

```powershell
.\.venv\Scripts\python.exe -m growth.run --userbehavior data/UserBehavior.csv
```

数据需从[原始来源](https://tianchi.aliyun.com/dataset/649)自行取得，原始文件不进版本库。流式聚合以每日去重用户集合保存，仍需与用户规模匹配的内存。请勿把真实日志指标与模拟实验作用户关联。

## 内容

| 文件 | 作用 |
|---|---|
| `growth/data.py` | 可复现随机补贴实验及公开行为日志适配 |
| `growth/analysis.py` | SRM、ITT、CUPED、样本量、T-Learner、IPW经济评估 |
| `growth/run.py` | 一键运行、报告、图表、人货场分层与评分 |
| `sql/hive_user_growth.sql` | Hive清洗、增长指标、有序漏斗、成熟cohort留存模板 |
| `reports/REPORT.md` | 默认运行的实际计算结果 |
| `docs/PROJECT_PLAN.md` | 业务指标、JD映射、实验设计、参考及改进 |
| `docs/RESUME.md` | 简历项目经历与面试要点 |
| `tests/` | 因果加权、成本口径、可复现与输入边界校验 |

## 方法与解读

用户按60/20/20分割；训练T-Learner，验证集确定定向比例，独立测试集用已知随机分组概率估计相对不发券的收益。比较随机、转化评分、Uplift和全量发券。

净ROI =（增量毛利－补贴成本）/补贴成本。相同触达比例不等于相同实际预算。模型和策略bootstrap区间只覆盖测试集抽样，不覆盖重新训练。分层探索不等于发现真实业务规律。

Hive脚本需要单独的Hive/HDFS环境，未在本地Python验证中运行。不得将SQL模板称为已部署的生产数仓。方法和数据限制详见[项目设计](docs/PROJECT_PLAN.md)。

参考GitHub项目与数据源见项目设计；本仓库代码为原创实现。MIT许可仅覆盖本仓库原创代码，不覆盖第三方数据。
