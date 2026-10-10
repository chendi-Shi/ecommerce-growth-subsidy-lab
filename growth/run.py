"""python -m growth.run --users 60000 --seed 42"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import generate, behavior_metrics
from .analysis import experiment, split_and_fit, top_mask, evaluate_policy, select_fraction, required_sample_per_arm, uplift_curve, segment_effects


def main():
    parser = argparse.ArgumentParser(description='Ecommerce growth portfolio (synthetic coupon RCT)')
    parser.add_argument('--users',type=int,default=60000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('reports'))
    parser.add_argument('--userbehavior',type=Path,help='Optional genuine headerless Tianchi CSV, separate analysis')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    frame, truth = generate(args.users,args.seed)
    ab = experiment(frame)
    if ab['srm_p_value'] < .001:
        raise ValueError('SRM check failed (p < .001): stop policy evaluation and investigate assignment.')
    validation, test = split_and_fit(frame,args.seed)
    fraction = select_fraction(validation)
    curve_x, curve_y, qini_area = uplift_curve(test)
    indices = np.linspace(0,len(curve_x)-1,101).astype(int)
    pd.DataFrame(dict(target_share=curve_x[indices], incremental_conversion=curve_y[indices])).to_csv(args.output/'uplift_curve.csv',index=False)
    rng = np.random.default_rng(args.seed)
    masks = {'uplift':top_mask(test.uplift,fraction),
             'conversion_score':top_mask(test.natural_conversion,fraction),
             'random':top_mask(rng.random(len(test)),fraction),
             'all':np.ones(len(test),dtype=bool),
             'none':np.zeros(len(test),dtype=bool)}
    policies = pd.DataFrame([dict(policy=name,**evaluate_policy(test,mask,args.seed)) for name,mask in masks.items()])
    policies.to_csv(args.output/'policies.csv',index=False)
    segments = frame.groupby(['new_user','category','scene','treatment']).agg(
        users=('user_id','size'),conversion=('converted','mean'),
        margin=('margin','mean'),subsidy=('subsidy','mean')).reset_index()
    segments.to_csv(args.output/'segments.csv',index=False)
    pd.DataFrame(segment_effects(frame)).to_csv(args.output/'segment_effects.csv',index=False)
    test[['user_id','uplift','natural_conversion']].to_csv(args.output/'test_scores.csv',index=False)
    truth_summary = truth.true_uplift.describe().to_dict()
    summary = dict(data_source='SYNTHETIC: not Douyin data, not a live experiment',
        seed=args.seed, experiment=ab, selected_fraction=fraction,
        normalized_ipw_qini_area=qini_area,
        train_validation_test='60/20/20, unique users, stratified by assignment',
        power_example_per_arm=required_sample_per_arm(), truth_diagnostics=truth_summary,
        test_n=len(test), policies=policies.astype(object).where(pd.notnull(policies),None).to_dict(orient='records'))
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(policies.policy,policies.net_profit_per_user,color='#2864ae')
    ax.errorbar(policies.policy,policies.net_profit_per_user,
        yerr=np.array([np.maximum(0,policies.net_profit_per_user-policies.net_ci_low),
                       np.maximum(0,policies.net_ci_high-policies.net_profit_per_user)]),fmt='none',color='black',capsize=4)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set_ylabel('Incremental net profit / eligible user (CNY)')
    ax.set_title('Synthetic coupon RCT: held-out policy estimates (95% bootstrap CI)')
    fig.tight_layout(); fig.savefig(args.output/'policy_comparison.png',dpi=160); plt.close(fig)
    ab_delta = ab['metrics']['converted']
    best = policies[policies.policy=='uplift'].iloc[0]
    report = f'''# 电商用户增长与补贴策略评估 — 运行报告

**数据来源：合成数据。没有使用抖音内部数据，没有真实上线或业务收益。**

- 随机种子：{args.seed}；用户数：{len(frame):,}；独立测试集：{len(test):,}。
- 模拟场景：短视频、直播、商城；控制组不发券，实验组发8元券，购买时核销。
- SRM p-value：{ab['srm_p_value']:.4f}。
- ITT 购买转化差：{ab_delta['delta']*100:.2f} 个百分点，95% CI [{ab_delta['ci_low']*100:.2f}, {ab_delta['ci_high']*100:.2f}]。
- 验证集选择的定向比例：{fraction:.0%}；随后冻结策略，在测试集估计收益。
- 归一化 IPW Qini 面积：{qini_area:.5f}（累计增量转化曲线减去随机基线，仅作测试集诊断）。
- Uplift 策略增量净利润/全体合格用户：{best.net_profit_per_user:.3f} 元，95% bootstrap CI [{best.net_ci_low:.3f}, {best.net_ci_high:.3f}]。
- 盈利证据：{'净利润区间下限大于零，在此模拟测试中有正收益证据。' if best.net_ci_low>0 else '净利润区间包含或低于零，当前不能认定策略稳定盈利，应扩大验证或调整券额。'}
- 样本量示例：基线15%、绝对MDE 2个百分点、双侧alpha=.05、power=.8，每组约{summary['power_example_per_arm']:,}人。

![策略评估](policy_comparison.png)

## 解释

所有定向策略与“不发券”比较。随机策略与转化评分策略使用相同触达比例；购买核销率不同，因此实际支出不同，不能称为等预算实验。全量策略用于参照。ROI 定义为（增量毛利－补贴成本）/补贴成本，不把 GMV 当作利润。

模型为 T-Learner（分别拟合实验组与控制组随机森林），只使用分组前特征；验证集调比例，测试集不参与调参。测试集利用已知随机分组概率进行 IPW 估计，区间来自对固定测试策略逐用户 bootstrap，未覆盖重新训练与调参的不确定性。分层结果仅作探索，不直接声称显著。

人货场分层位于 segments.csv，A/B、CUPED、ROI明细位于 summary.json。公开行为日志如有输入，则单独输出描述性指标，不能与此实验连接成同一批用户。
'''
    (args.output/'REPORT.md').write_text(report,encoding='utf-8')
    if args.userbehavior:
        behavior_metrics(args.userbehavior).to_csv(args.output/'public_behavior_daily.csv',index=False)
    print(json.dumps(dict(users=len(frame), test_users=len(test), selected_fraction=fraction,
        conversion_delta=ab_delta['delta'], uplift_net_profit=float(best.net_profit_per_user)),indent=2))


if __name__ == '__main__':
    main()
