"""User-level ITT and fixed-policy inference with known randomized probabilities."""
import numpy as np
import pandas as pd
from scipy import stats


def estimate(scores):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite user scores.')
    mean=float(values.mean()); se=float(values.std(ddof=1)/np.sqrt(len(values)))
    radius=float(stats.t.ppf(.975,len(values)-1)*se)
    p=2*stats.t.sf(abs(mean/se),len(values)-1) if se else (1.0 if mean==0 else 0.0)
    return dict(estimate=mean,se=se,ci_low=mean-radius,ci_high=mean+radius,p_value=float(p))


def difference(treatment,control):
    a,b=np.asarray(treatment,dtype=float),np.asarray(control,dtype=float)
    if min(len(a),len(b))<2: raise ValueError('Each arm needs at least two units.')
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
    se=np.sqrt(va+vb); mean=float(a.mean()-b.mean())
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if se else len(a)+len(b)-2
    radius=stats.t.ppf(.975,df)*se
    p=2*stats.t.sf(abs(mean/se),df) if se else (1.0 if mean==0 else 0.0)
    return dict(estimate=mean,se=float(se),ci_low=float(mean-radius),ci_high=float(mean+radius),p_value=float(p))


def holm(p_values):
    p=np.asarray(p_values,dtype=float); order=np.argsort(p); result=np.empty(len(p))
    adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result[order]=np.minimum(1,adjusted)
    return result


def audit(frame):
    required={'customer_id','recency','history','mens','womens','newbie','zip_code','channel','treatment','visit','conversion','spend'}
    if not required.issubset(frame.columns): raise ValueError('Missing contract fields.')
    if frame.isna().any().any(): raise ValueError('Missing fields in released experiment.')
    numeric=['recency','history','mens','womens','newbie','treatment','visit','conversion','spend']
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all(): raise ValueError('Nonfinite numeric fields.')
    if not frame.customer_id.is_unique: raise ValueError('Duplicate row keys.')
    if not set(frame.treatment.unique())=={0,1,2}: raise ValueError('Expected all 3 arms.')
    for col in ['mens','womens','newbie','visit','conversion']:
        if not frame[col].isin([0,1]).all(): raise ValueError(f'{col} must be binary.')
    if (frame.conversion>frame.visit).any(): raise ValueError('Purchase without visit needs investigation.')
    if ((frame.spend>0)!=(frame.conversion==1)).any(): raise ValueError('Spend/purchase contract inconsistent.')
    if (frame[['history','spend','recency']]<0).any().any(): raise ValueError('Negative monetary/history fields.')
    counts=frame.treatment.value_counts().sort_index()
    srm=stats.chisquare(counts,np.full(3,len(frame)/3))
    outcomes=frame.groupby('treatment').agg(users=('customer_id','size'),buyers=('conversion','sum'),
        visit_rate=('visit','mean'),conversion_rate=('conversion','mean'),spend_per_user=('spend','mean'))
    return dict(n=len(frame),srm_p=float(srm.pvalue),arms=outcomes.reset_index().to_dict('records'),
        spend_quantiles=frame.spend.quantile([0,.5,.9,.99,.999,1]).to_dict(),
        zero_spend_share=float((frame.spend==0).mean()))


def balance(frame):
    features=frame[['recency','history','mens','womens','newbie']].copy()
    features['history']=np.log1p(features.history)
    features=pd.concat([features,pd.get_dummies(frame[['zip_code','channel']],dtype=float)],axis=1)
    rows=[]
    for arm in [1,2]:
        a=features[frame.treatment==arm]; b=features[frame.treatment==0]
        for name in features:
            denominator=np.sqrt((a[name].var(ddof=1)+b[name].var(ddof=1))/2)
            smd=(a[name].mean()-b[name].mean())/denominator if denominator else 0
            rows.append(dict(arm=arm,feature=name,smd=float(smd)))
    return pd.DataFrame(rows)


def itt(frame):
    rows=[]
    for outcome in ['visit','conversion','spend']:
        group=[]
        for arm in [1,2]:
            a,b=frame[frame.treatment==arm],frame[frame.treatment==0]
            group.append(dict(outcome=outcome,arm=arm,control_mean=float(b[outcome].mean()),
                treatment_mean=float(a[outcome].mean()),**difference(a[outcome],b[outcome])))
        adjusted=holm([row['p_value'] for row in group])
        for row,p in zip(group,adjusted): row['holm_p']=float(p)
        rows+=group
    return pd.DataFrame(rows)


def dr_scores(y,actions,mu,propensity=None):
    y=np.asarray(y); actions=np.asarray(actions); mu=np.asarray(mu)
    e=np.full(mu.shape[1],1/mu.shape[1]) if propensity is None else np.asarray(propensity)
    if mu.shape!=(len(y),len(e)) or not np.isclose(e.sum(),1) or (e<=0).any():
        raise ValueError('Invalid score/probability shape.')
    residual=np.eye(len(e))[actions]*(y[:,None]-mu)/e
    return mu+residual


def policy_scores(probabilities,scores):
    p=np.asarray(probabilities); s=np.asarray(scores)
    if p.shape!=s.shape or not np.allclose(p.sum(axis=1),1) or (p<0).any():
        raise ValueError('Policy must give nonnegative action probabilities summing to 1.')
    return np.sum(p*s,axis=1)-s[:,0]


def targeting(cates,limit,margin=.3,cost=.05):
    """At most limit share contacts; individual action chosen from predicted dollar CATE."""
    cates=np.asarray(cates)
    if not 0<=limit<=1 or cates.ndim!=2 or cates.shape[1]!=2: raise ValueError('Invalid contact budget.')
    benefit=margin*cates-cost
    arm=np.argmax(benefit,axis=1)+1
    best=benefit.max(axis=1)
    order=np.argsort(-best,kind='stable')[:int(np.floor(len(best)*limit))]
    action=np.zeros(len(best),dtype=int)
    selected=order[best[order]>0]; action[selected]=arm[selected]
    return np.eye(3)[action]


def fixed_random(n,arm,share):
    p=np.zeros((n,3)); p[:,0]=1-share; p[:,arm]=share
    return p


def sample_size(p0,mde,alpha=.05,power=.8):
    if not 0<p0<p0+mde<1: raise ValueError('Invalid baseline/MDE.')
    p1=p0+mde; pooled=(p0+p1)/2
    return int(np.ceil((stats.norm.ppf(1-alpha/2)*np.sqrt(2*pooled*(1-pooled))+
        stats.norm.ppf(power)*np.sqrt(p0*(1-p0)+p1*(1-p1)))**2/mde**2))
