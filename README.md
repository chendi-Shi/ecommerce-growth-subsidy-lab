# 电商增长诊断与营销投入决策

**当买家增长、购买金额下降时，应该扩大补贴吗？当触达平均有效时，应该增加定向模型投入吗？** 本仓库用两个独立公开案例，把问题定义、竞争解释、数值核对和决策边界放在分析中心。两个案例没有共同客户，不拼接为同一商家的策略效果。

## 先看改变决策的发现

**经营诊断：原先的下降信号只在部分账簿成立。** 2011年11月有客户ID的买家同比增加3.36%、购买金额减少1.68%，但全体商品正向金额增长1.58%，退货抵减后增长2.34%。按之前的2010年1—10月消费冻结前1%客户，38名高历史购买额客户贡献−£38,225，其余已识别客户反而增加£18,781（+2.04%）。单位金额下滑更多对应商品组合变化，不能直接解释为普遍降价。建议优先核查身份覆盖、大客户订单和商品结构；海外固定名单外客户另行调查。

→ **[阅读经营诊断决策备忘录](reports/retail_diagnosis/DECISION_MEMO.md)**：竞争解释、11个可比月份、同客户/同商品检查、具体SKU与行动条件。

**营销决策：不显著的定向优势，还不足以决定停止研究。** 64,000人真实三臂邮件实验有正向平均购买率效果，但冻结定向相对同触达随机的收入优势为−$0.006/合格用户，95% CI [−$0.279, $0.267]。该区间仍包含有意义的正向收益和损失。补充人群选择/内容选择的配对拆解，并把额外实施成本转成收入门槛；在相同三臂采集的未来评估情景中，检出$0.05/人的优势约需78.1万名新的独立评估用户，而非继续解释当前点估计。

→ **[阅读营销投入决策备忘录](reports/campaign_decision/DECISION_MEMO.md)**：哪些结论成立、哪些仍不确定、成本门槛和下一轮研究设计。

![经营诊断：先核对范围，再解释结构](reports/retail_diagnosis/diagnosis.png)

## 业务研究怎样推进

|问题|实际分析|交付给下一环节的内容|
|---|---|---|
|增长问题是否在同一范围成立？|已识别/匿名/退货账簿桥接，三种ID覆盖口径|可核对的问题定义、身份记录排查条件|
|是谁、哪些交易改变了结果？|客户集合桥接、前期冻结大户、国家分层、同SKU和商品组合分解|需要优先核查的客户与商品范围；未识别的机制|
|结果稳定还是单月现象？|1—11月同比、季度/累计窗口、去重与退款敏感性|结论适用窗口及会改变判断的证据|
|模型是否覆盖额外投入？|同触达配对比较、指定顺序的人群/内容拆解、成本门槛|明确的支持/排除/证据不足状态|
|还值得补多少证据？|独立评估样本量与方差情景；区分OPE和策略A/B设计|下一轮实验前需要的成本、方差和策略合同|

[研究协议与竞争假设](docs/RESEARCH_PROTOCOL.md)记录研究逻辑；[技术分解说明](reports/retail_diagnosis/TECHNICAL.md)记录恒等式与局限。机械分解用于对账和定位，不被称为因果归因、价格弹性或可恢复的补贴收益。新增研究属于已查看数据后的回顾分析，没有补写成预注册。

## 数据与实现

- **UCI Online Retail II：**1,067,371条真实发票商品明细，GBP。逐批清洗与DuckDB SQL生成订单、用户月、复购和诊断表；缺少曝光、真实注册、挂牌价、库存、毛利与补贴分配。
- **Hillstrom：**64,000名存量客户、三臂随机邮件、14天结果，USD。完成ITT、SRM/平衡、多重检验和冻结策略评估。没有真实券额、成本与退款，不把邮件结果写成补贴ROI。
- **实际Apache Hive：**完整64,000行实验数据执行ODS→ORC分区DWD→ADS，完成21项独立人数/金额核验，保存schema、分区与EXPLAIN。全量交易SQL由DuckDB执行；Hive不是生产Hadoop集群。

代码包含原始指标与模型实现、业务诊断模块、测试及聚合结果。数据源、许可和下载哈希见[DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md)，实际执行证据见[VALIDATION.md](docs/VALIDATION.md)。原始数据、客户级记录、行级预测和本地数据库不提交到仓库。

## 复现

需要Python3.12。基础数据首次下载约44MB工作簿与0.44MB实验文件；完整处理逐批读取，DuckDB内存上限256MB并允许磁盘溢写。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py
python -m unittest discover -s tests -v
```

已有原分析结果和交易数据库，只重跑业务诊断：

```bash
python run_project.py --diagnostics-only
# 或分别运行
python -m retail.diagnose
python -m campaign.decision
```

只研究营销实验：`python run_project.py --campaign-only`。新增决策模块会核对原冻结策略的点估计，避免悄悄重新选模型。

已有Docker daemon时：

```bash
python run_project.py --hive
# 或单独验证真实Hive
python -m campaign.download
docker pull apache/hive:4.2.1
python infra/run_hive.py --input data/processed/hillstrom.csv --output reports/hive
```

[Hive执行说明](infra/README.md)明确官方镜像、local Tez与对账方法。可选50,000条交易Hive样本只验证字段和聚合合同；完整交易结论来自全量DuckDB。基础复购与人货分布见[交易指标报告](reports/retail/README.md)，原随机实验指标见[实验报告](reports/campaign/REPORT.md)。

## 投递材料

- [简历项目描述](docs/RESUME.md)：以业务发现、排除的解释和决策条件组织。
- [项目方案与JD映射](docs/PROJECT_PLAN.md)：明确各项岗位能力的实际证据。
- [面试追问指南](docs/INTERVIEW_GUIDE.md)：如何讲清口径反转、结构分解与负结果。
- [补贴实验迁移方案](docs/EXPERIMENT_DESIGN.md)：尚待真实业务数据实施的指标、成本、功效与协作合同。

代码采用MIT许可；UCI数据需CC BY4.0署名。Hillstrom原始再分发许可未明确，使用运行时下载与聚合结果，不再分发行级记录。
