# 验证记录

- 日期：2026-10-10（Asia/Shanghai）。
- 环境：Windows，Python 3.12.14，完整版本见requirements-lock.txt。
- `python -m unittest discover -s tests -v`：7项测试通过。
- `python -m growth.run --users 60000 --seed 42`：完整运行成功，独立测试集12,000人。
- 验证内容：IPW全体用户分母、核销成本、零方差统计、零效应随机实验、固定种子复现、特征排除、定向人数及不发券选项、公开日志跨批次用户去重及时区转换、Qini曲线终点。
- 输出图表已检查：策略名称、金额单位、零基线及95%bootstrap区间可见。
- 结果：购买转化差3.98个百分点；Uplift策略净利润每名合格用户0.215元，区间[-0.014,0.423]，不能声称稳定盈利。
- Hive SQL未在Hive集群执行；未下载或分析完整真实UserBehavior；未运行Criteo实验；GitHub Actions另行在远端执行。

这些结果基于合成实验，仅证明项目流程可复现，不证明真实业务策略有效。
