# 数据来源、字段契约与证据边界

这是独立公开数据研究。交易分析与营销实验来自不同商家，分别回答增长诊断和增量触达问题；不按用户关联、不合并 GBP 与 USD，也不称为抖音数据或真实补贴上线。

## Online Retail II：交易侧

[UCI 原始页面](https://archive.ics.uci.edu/dataset/502/online+retail+ii) / [官方下载](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip) / [数据 DOI](https://doi.org/10.24432/C5CG6D)。Daqing Chen 提供英国礼品电商 2009-12-01 至 2011-12-09 的两年交易，官方记录 1,067,371 行，部分客户是批发商。数据为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，代码许可不替代来源署名。

本次官方下载传输不稳定，实际获取 [KristerMartinez GitHub 镜像](https://github.com/KristerMartinez/online_retail_ii) 的工作簿；[固定版本下载](https://raw.githubusercontent.com/KristerMartinez/online_retail_ii/3a5eb9132055218af623a5550fc3f8d124b0e10c/online_retail_II.xlsx) 对应提交 `3a5eb9132055218af623a5550fc3f8d124b0e10c`。文件为 **45,622,278 字节**，已验证 SHA256 `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` 和 ZIP 内部 CRC；加载校验两张年度表分别为 525,461 / 541,910 行，总计 1,067,371 行。字段及年度行数符合数据发布说明；**没有获得官方文件的独立 checksum，不能声称镜像与官方字节一致**。下载来源、固定提交、本地哈希和转换校验写入运行清单，原始文件仍不进入仓库。

一行是发票商品明细，**不是一单或一个用户**。`Invoice` 是发票号，C 前缀表示取消；`StockCode` 为商品代码；`Quantity` 为数量；`InvoiceDate` 为交易时间；`Price` 为单价 GBP；`Customer ID` 为可缺失的客户号；`Country` 为客户国家；`Description` 为商品名。原始工作簿命名与 UCI 页面通用字段名可能不同，转换器显式映射并校验两张年度表。

退货/贷项单与正向购买分开保留：正向商品收入、退货金额、净商品收入分别对账；负数量或 C 前缀不能简单删除后声称净收入。没有原始唯一行 ID，完全相同行去重是一项可审计假设，原始行数、重复数和金额差须披露。客户缺失记录可进入金额统计，但不能补造客户加入复购 cohort。

仅观察购买者，无法推出曝光、DAU、平台购买转化率、注册拉新或真实场景分布。`first_observed` 是首次观测买家；数据起点之前的购买未知，即左截断。留存是后续月份再次购买率，尚未完整观察的月份标记缺失，不能当作零。RFM 使用预测时点之前的交易；退款未与原订单完整链接，不能宣称成熟订单退款率或利润。

## Hillstrom：实验侧

[作者原始发布](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) / [原始 CSV](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv) / [实际使用的 gzip 镜像](https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz) / [镜像维护代码](https://github.com/maks-sh/scikit-uplift/blob/master/sklift/datasets/datasets.py)。作者描述 64,000 名过去十二个月内购买过的客户，随机三等分至男装邮件、女装邮件和不发邮件组，跟踪之后两周。原文随机化证据：“1/3 were randomly chosen to not receive an e-mail campaign.”

已核对 gzip 长度 443,070 字节；MD5 `a68a81291f53a14f4e29002629803ba3`，SHA256 `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f`。加载时验证哈希、64,000 行及三臂计数；实际值写入 `data/processed/hillstrom_manifest.json`。本地 `customer_id` 是稳定行号，**不是原始客户 ID**。

| 时点 | 字段与含义 |
|---|---|
| 处理前 | `recency`：距上次购买月数；`history/history_segment`：过去一年消费美元/分档；`mens/womens`：对应商品购买标记，不是性别；`newbie`：十二个月内新客；`zip_code`：城乡类别，不是真实邮编；`channel`：历史购买渠道 |
| 分配 | `segment`：No E-Mail/Mens E-Mail/Womens E-Mail；项目映射 0/1/2，各臂设计概率 1/3 |
| 处理后 | `visit/conversion`：两周是否访问/购买；`spend`：两周消费 USD。均不能作为分组前特征 |

这是存量客户邮件激活实验，没有券额、毛利率、触达成本、退款或长期留存。主分析为全体分配用户的 ITT；访问后购买率仅作漏斗诊断。利润与盈亏平衡接触成本是明确参数下的情景，不是真实补贴 ROI。原始页面未找到明确再分发许可；仓库只发布下载器、聚合结果和方法，原始记录及行级预测不上传。

## 为什么没有选其他数据

淘宝 UserBehavior 没有随机处理或金额，不能评估补贴因果 ROI。[Criteo](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) 是广告 RCT，但特征匿名且没有收入/成本，不补足本项目业务语义。[MT-LIFT](https://github.com/MTDJDSP/MT-LIFT) 更接近发券；2026-10-10 核查时官方下载需登录/有过期反馈，公开特征分布质疑未澄清，[arXiv 当前版本](https://arxiv.org/abs/2402.03379) 已撤回，故未作为结果证据。公开宣传规模不能替代可下载、可校验、实际跑通的数据。
