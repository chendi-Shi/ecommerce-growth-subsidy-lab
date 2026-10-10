"""Real three-arm marketing RCT: python -m campaign.run [--quick]."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .download import prepare
from .stats import audit,balance,itt,estimate,dr_scores,policy_scores,targeting,fixed_random,sample_size
from .models import split,fit_all,rankers,predictions,feature_frame

ARM_NAMES={0:'No email',1:'Mens merchandise',2:'Womens merchandise'}
BUDGET_SHARE=10000/64000  # explicit 10,000-contact business case, not an observed cost budget


def probabilities(actions):
    return np.eye(3)[np.asarray(actions,dtype=int)]


def policy_evaluation(name,p,psi,ht,margin,cost):
    revenue=policy_scores(p,psi); contact=1-p[:,0]
    net=margin*revenue-cost*contact
    rv,profit=estimate(revenue),estimate(net)
    return dict(policy=name,contact_share=float(contact.mean()),mens_share=float(p[:,1].mean()),
        womens_share=float(p[:,2].mean()),incremental_revenue=rv['estimate'],
        revenue_ci_low=rv['ci_low'],revenue_ci_high=rv['ci_high'],revenue_se=rv['se'],
        ht_revenue=estimate(policy_scores(p,ht))['estimate'],
        scenario_net_profit=profit['estimate'],profit_ci_low=profit['ci_low'],profit_ci_high=profit['ci_high'],
        breakeven_cost=margin*rv['estimate']/contact.mean() if contact.mean()>0 else None)


def adjusted_effects(train,test):
    rows=[]
    for outcome in ['visit','conversion','spend']:
        # Fit the adjustment only on training, residualized by randomized arm.
        x=np.log1p(train.history.to_numpy()); y=train[outcome].to_numpy()
        centered_x=x-train.groupby('treatment').history.transform(lambda v:np.log1p(v).mean()).to_numpy()
        centered_y=y-train.groupby('treatment')[outcome].transform('mean').to_numpy()
        theta=float(np.dot(centered_x,centered_y)/np.dot(centered_x,centered_x))
        adjusted=test[outcome]-theta*(np.log1p(test.history)-x.mean())
        from .stats import difference
        for arm in [1,2]:
            result=difference(adjusted[test.treatment==arm],adjusted[test.treatment==0])
            raw=difference(test.loc[test.treatment==arm,outcome],test.loc[test.treatment==0,outcome])
            rows.append(dict(outcome=outcome,arm=arm,theta=theta,raw_se=raw['se'],
                variance_reduction=1-result['se']**2/raw['se']**2,**result))
    return pd.DataFrame(rows)


def exploratory_segments(frame):
    from .stats import difference,holm
    rows=[]
    for dimension in ['mens','womens','newbie','channel','zip_code']:
        for value,subset in frame.groupby(dimension):
            for arm in [1,2]:
                result=difference(subset.loc[subset.treatment==arm,'spend'],subset.loc[subset.treatment==0,'spend'])
                rows.append(dict(dimension=dimension,value=str(value),arm=arm,n=len(subset),**result))
    result=pd.DataFrame(rows)
    # These are subgroup effects, NOT tests that one subgroup differs from another.
    result['holm_p_all_exploratory']=holm(result.p_value)
    return result


def write_plot(effects,policies,output):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    conversion=effects[effects.outcome=='conversion']
    x=np.arange(2)
    axes[0].bar(x,conversion.estimate*100,color=['#2563eb','#0891b2'])
    axes[0].errorbar(x,conversion.estimate*100,yerr=np.vstack([
        (conversion.estimate-conversion.ci_low)*100,(conversion.ci_high-conversion.estimate)*100]),fmt='none',color='black',capsize=4)
    axes[0].set_xticks(x,['Mens merchandise','Womens merchandise']);axes[0].axhline(0,color='gray')
    axes[0].set_title('Real RCT: 14-day purchase ITT'); axes[0].set_ylabel('Percentage points vs no email (95% CI)')
    y=np.arange(len(policies))
    axes[1].errorbar(policies.incremental_revenue,y,xerr=np.vstack([
        policies.incremental_revenue-policies.revenue_ci_low,policies.revenue_ci_high-policies.incremental_revenue]),fmt='o',color='#2563eb',capsize=3)
    axes[1].set_yticks(y,policies.policy);axes[1].axvline(0,color='gray')
    axes[1].set_title('Frozen policies: independent test users');axes[1].set_xlabel('Incremental spend USD / eligible recipient (95% CI)')
    fig.tight_layout();fig.savefig(output/'experiment_policy.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('reports/campaign'))
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--margin',type=float,default=.30)
    parser.add_argument('--contact-cost',type=float,default=.05)
    parser.add_argument('--quick',action='store_true',help='Audit + ITT only, no policy claim; for CI smoke checks')
    args=parser.parse_args()
    if not 0<args.margin<=1 or args.contact_cost<0: raise ValueError('Invalid economic scenario.')
    args.output.mkdir(parents=True,exist_ok=True)
    path,manifest=prepare(args.data_root)
    frame=pd.read_csv(path,dtype={'customer_id':'string'})
    quality=audit(frame); balance_table=balance(frame); effects=itt(frame)
    balance_table.to_csv(args.output/'balance.csv',index=False)
    effects.to_csv(args.output/'itt.csv',index=False)
    exploratory_segments(frame).to_csv(args.output/'exploratory_segments.csv',index=False)
    if quality['srm_p']<.001: raise ValueError('SRM failed; halt analysis for assignment audit.')
    summary=dict(source=manifest,audit=quality,primary_endpoint='14-day purchase conversion; two comparisons vs control, Holm correction',
        seed=args.seed,economics=dict(margin=args.margin,contact_cost_usd=args.contact_cost,
        type='ASSUMED scenario, not observed profit/coupon effect'),max_abs_smd=float(balance_table.smd.abs().max()))
    if args.quick:
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(quality,indent=2));return
    train,val,test=split(frame,args.seed)
    all_ids=[set(part.customer_id) for part in [train,val,test]]
    if any(all_ids[i]&all_ids[j] for i,j in [(0,1),(0,2),(1,2)]): raise ValueError('User split leakage.')
    splits=pd.concat([part.groupby('treatment').agg(n=('customer_id','size'),buyers=('conversion','sum')).reset_index().assign(split=name)
        for name,part in [('train',train),('validation',val),('test',test)]])
    splits.to_csv(args.output/'splits.csv',index=False)
    models=fit_all(train,args.seed)
    # Same train-fitted revenue nuisance model for validation and final evaluation.
    val_mu=predictions(models['ridge'],val)
    val_psi=dr_scores(val.spend,val.treatment,val_mu)
    val_rankers=rankers(models,val)
    candidates=[]
    for name,cates in val_rankers.items():
        for depth in [BUDGET_SHARE,.3,.5,1.0]:
            p=targeting(cates,depth,args.margin,args.contact_cost)
            score=args.margin*policy_scores(p,val_psi)-args.contact_cost*(1-p[:,0])
            candidates.append(dict(model=name,depth=depth,score=float(score.mean()),contact_share=float((1-p[:,0]).mean())))
    # No-contact participates; negative candidate values cannot force contact.
    candidates.append(dict(model='none',depth=0,score=0,contact_share=0))
    selection=pd.DataFrame(candidates)
    selection.to_csv(args.output/'validation_selection.csv',index=False)
    selected=selection.loc[selection.score.idxmax()].to_dict()
    budget_candidates=selection[(selection.depth==BUDGET_SHARE)|(selection.model=='none')]
    budget_selected=budget_candidates.loc[budget_candidates.score.idxmax()].to_dict()
    # Best homogeneous arm is chosen on validation, never on test.
    arm_values=[policy_scores(fixed_random(len(val),arm,1),val_psi).mean() for arm in [1,2]]
    best_arm=int(np.argmax(arm_values))+1
    test_mu=predictions(models['ridge'],test)
    psi=dr_scores(test.spend,test.treatment,test_mu)
    ht=dr_scores(test.spend,test.treatment,np.zeros_like(test_mu))
    test_rankers=rankers(models,test)
    def chosen(row):
        return fixed_random(len(test),best_arm,0) if row['model']=='none' else targeting(test_rankers[row['model']],row['depth'],args.margin,args.contact_cost)
    optimized=chosen(selected); budget=chosen(budget_selected)
    budget_share=float((1-budget[:,0]).mean())
    # Purchase-propensity baseline: same maximum contacts, same best content.
    probability=models['conversion'].predict_proba(feature_frame(test))[:,1]
    natural_actions=np.zeros(len(test),dtype=int)
    natural_actions[np.argsort(-probability,kind='stable')[:int(np.floor(budget_share*len(test)))]]=best_arm
    policies=dict(no_contact=fixed_random(len(test),best_arm,0),
        all_best_arm=fixed_random(len(test),best_arm,1),
        random_same_contacts=fixed_random(len(test),best_arm,budget_share),
        purchase_propensity=probabilities(natural_actions),budget_targeted=budget,
        validation_optimized=optimized)
    table=pd.DataFrame([policy_evaluation(name,p,psi,ht,args.margin,args.contact_cost) for name,p in policies.items()])
    table.to_csv(args.output/'policies.csv',index=False)
    paired=[]
    for comparator in ['no_contact','random_same_contacts','purchase_propensity','all_best_arm']:
        delta=policy_scores(budget,psi)-policy_scores(policies[comparator],psi)
        paired.append(dict(comparison=f'budget_targeted vs {comparator}',**estimate(delta)))
    pd.DataFrame(paired).to_csv(args.output/'paired_comparisons.csv',index=False)
    # Economic thresholds over a fixed policy, not post-test reoptimized ROI.
    sensitivity=[]
    for margin in [.1,.2,.3,.4,.5]:
        for cost in [0,.01,.05,.1,.25,.5,1]:
            for name,p in policies.items():
                net=margin*policy_scores(p,psi)-cost*(1-p[:,0])
                sensitivity.append(dict(policy=name,assumed_margin=margin,assumed_contact_cost=cost,**estimate(net)))
    pd.DataFrame(sensitivity).to_csv(args.output/'economic_sensitivity.csv',index=False)
    adjusted=adjusted_effects(train,test);adjusted.to_csv(args.output/'covariate_adjustment.csv',index=False)
    # Honest budget-policy calibration on untouched test bins; no extra tuning.
    if budget_selected['model']!='none':
        rank=test_rankers[budget_selected['model']].max(axis=1)
        order=np.argsort(-rank);bins=np.array_split(order,5);calibration=[]
        for i,idx in enumerate(bins):
            predicted=test_rankers[budget_selected['model']][idx]
            arm=np.argmax(predicted,axis=1)+1
            score=psi[idx,arm]-psi[idx,0]
            calibration.append(dict(quintile=i+1,n=len(idx),predicted_mean=float(predicted.max(axis=1).mean()),**estimate(score)))
        pd.DataFrame(calibration).to_csv(args.output/'cate_calibration.csv',index=False)
    summary.update(train_n=len(train),validation_n=len(val),test_n=len(test),
        validation_selected=selected,budget_selected=budget_selected,budget_max_share=BUDGET_SHARE,
        best_homogeneous_arm=best_arm,policy_nuisance='Train-fitted Ridge; randomized design e=(1/3,1/3,1/3)',
        uncertainty='Paired per-user DR influence-score t interval conditional on fixed train/validation selection.',
        controls={'no_test_tuning':True,'train_validation_test_disjoint':True,'feature_post_treatment_excluded':True},
        test_revenue_outlier_check={'buyers':int(test.conversion.sum()),'max_spend':float(test.spend.max())})
    baseline=float(frame.loc[frame.treatment==0,'conversion'].mean())
    summary['design_power_example']=dict(baseline=baseline,absolute_mde=.003,
        per_arm=sample_size(baseline,.003,alpha=.025),alpha_per_comparison=.025,power=.8,
        note='Bonferroni conservative planning for 2 primary vs-control comparisons; Holm used in analysis.')
    summary['policies']=table.astype(object).where(pd.notnull(table),None).to_dict('records')
    summary['paired']=paired
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_plot(effects,table,args.output)
    budget_row=table[table.policy=='budget_targeted'].iloc[0]
    target_comparison=next(row for row in paired if 'random_same_contacts' in row['comparison'])
    conversion=effects[effects.outcome=='conversion']
    report=f'''# 存量用户激活：真实三臂随机营销实验

数据：Hillstrom原始64,000名客户，前一年内购买过；随机分配男装邮件、女装邮件、不发邮件，结果观察14天。消费金额为USD。不是抖音数据，也不是发券实验。

## 确认性结果

|营销内容|购买转化增量(pp)|95% CI(pp)|Holm p-value|
|---|---:|---:|---:|
'''
    for row in conversion.itertuples():
        report+=f'|{ARM_NAMES[row.arm]}|{row.estimate*100:.3f}|[{row.ci_low*100:.3f}, {row.ci_high*100:.3f}]|{row.holm_p:.5g}|\n'
    report+=f'''
SRM p={quality['srm_p']:.4f}，最大前置特征|SMD|={summary['max_abs_smd']:.4f}。通过审计不等于证明任何数据都随机。全部分配用户进入ITT；访问后购买率仅用于描述路径，不能把访问者当作重新随机的实验人群。

## 有限触达策略

预设最多触达10,000/64,000={BUDGET_SHARE:.3%}用户。60/20/20分割，训练集38,400人、验证和测试各12,800人；验证集选择模型与内容、冻结后测试。收入T-Learner（Ridge/RF）、训练集5折交叉拟合DR-Learner与自然购买概率基线均实际计算。

受约束候选：{budget_selected['model']}；实际测试触达{budget_row.contact_share:.3%}。
相对不触达的增量消费/合格用户：${budget_row.incremental_revenue:.4f}，95% CI [{budget_row.revenue_ci_low:.4f}, {budget_row.revenue_ci_high:.4f}]。
相对等触达随机基线的**配对差值**：${target_comparison['estimate']:.4f}，95% CI [{target_comparison['ci_low']:.4f}, {target_comparison['ci_high']:.4f}]。
{'区间下限大于零，当前固定策略在测试人群中的收入定向优势有统计证据。' if target_comparison['ci_low']>0 else '区间跨零或低于零，不能认定定向优于等触达随机；采用更复杂模型没有自动形成业务收益。'}

等触达随机策略使用概率混合，报告期望效果；不是多次随机抽样挑最差基线。DR与HT对照见policies.csv，测试分位校准见cate_calibration.csv。区间不包含重新训练和选择不确定性；稀疏重尾收入、单次历史实验与无新时段外部验证限制结论外推。

## 决策与成本

实测收入与假设成本分开。利润率{args.margin:.0%}、每触达${args.contact_cost:.2f}是预设情景，并非实际财务数据。优先查看增量收入、盈亏平衡触达成本和economic_sensitivity.csv中的参数区间。未观察真实邮件成本、退款或毛利；不能声称真实补贴ROI提升，也不能据此断言某个券额有效。

后续真实业务：在有成本/核销/退款数据后，预注册按用户稳定分组的内容或券额实验，先验样本量来自真实控制购买率{baseline:.3%}；MDE 0.3pp、80% power、每比较alpha=.025需每组约{summary['design_power_example']['per_arm']:,}人。既有用户激活不能包装为新客拉新。

## 可验证产物

ITT/协变量调整/前置平衡/实验划分/验证选择/策略配对比较/预算与利润率敏感性表均包含实际数值。历史消费是前一年数据，训练集固定系数调整报告实际精度改善；不会称其为实验前14天消费，也不会因改善不显著而删除结果。所有模型输入字段明确排除处理标志及处理后visit/conversion/spend。

![真实实验与策略](experiment_policy.png)
'''
    (args.output/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(users=len(frame),srm=quality['srm_p'],budget_model=budget_selected['model'],
        paired_vs_random=target_comparison,output=str(args.output)),indent=2))


if __name__=='__main__': main()
