import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .data import FEATURES


def difference_ci(treatment, control):
    a, b = np.asarray(treatment), np.asarray(control)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('Both arms require at least two observations.')
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = np.sqrt(va+vb)
    delta = float(a.mean()-b.mean())
    if se == 0:
        return dict(delta=delta, ci_low=delta, ci_high=delta, p_value=1.0 if delta == 0 else 0.0)
    df = (va+vb)**2 / (va**2/(len(a)-1)+vb**2/(len(b)-1))
    radius = stats.t.ppf(.975, df) * se
    return dict(delta=delta, ci_low=float(delta-radius), ci_high=float(delta+radius),
                p_value=float(2*stats.t.sf(abs(delta/se), df)))


def experiment(df):
    t, c = df[df.treatment == 1], df[df.treatment == 0]
    srm = stats.chisquare([len(t), len(c)], [len(df)/2]*2)
    results = {name:difference_ci(t[name], c[name])
               for name in ['converted','revenue','margin','subsidy']}
    results['net_profit'] = difference_ci(t.margin-t.subsidy, c.margin-c.subsidy)
    covariate = df.pre_orders.to_numpy()
    y = df.converted.to_numpy()
    variance = covariate.var(ddof=1)
    theta = np.cov(y, covariate, ddof=1)[0,1]/variance if variance > 0 else 0.0
    adjusted = y-theta*(covariate-covariate.mean())
    results['cuped_conversion'] = difference_ci(adjusted[df.treatment==1], adjusted[df.treatment==0])
    return dict(n=len(df), treatment_n=len(t), control_n=len(c),
                srm_p_value=float(srm.pvalue), metrics=results)


def segment_effects(df):
    """Predefined exploratory cuts; no multiple-testing significance claims."""
    rows = []
    for dimension in ['new_user','category','scene']:
        for value, subset in df.groupby(dimension):
            t, c = subset[subset.treatment==1], subset[subset.treatment==0]
            conversion = difference_ci(t.converted,c.converted)
            profit = difference_ci(t.margin-t.subsidy,c.margin-c.subsidy)
            rows.append(dict(dimension=dimension,value=int(value),users=len(subset),
                conversion_delta=conversion['delta'],conversion_ci_low=conversion['ci_low'],
                conversion_ci_high=conversion['ci_high'],net_profit_delta=profit['delta'],
                net_profit_ci_low=profit['ci_low'],net_profit_ci_high=profit['ci_high']))
    return rows


def required_sample_per_arm(baseline=.15, mde=.02, alpha=.05, power=.8):
    """Normal approximation, absolute MDE, equal allocation, two-sided test."""
    if not (0 < baseline < 1 and 0 < mde < 1-baseline and 0 < alpha < 1 and 0 < power < 1):
        raise ValueError('Invalid power analysis parameters.')
    p1, p2 = baseline, baseline+mde
    pooled = (p1+p2)/2
    z1, z2 = stats.norm.ppf(1-alpha/2), stats.norm.ppf(power)
    return int(np.ceil((z1*np.sqrt(2*pooled*(1-pooled))+
                        z2*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/mde**2))


def split_and_fit(df, seed=42):
    train, remainder = train_test_split(df, test_size=.4, random_state=seed, stratify=df.treatment)
    validation, test = train_test_split(remainder, test_size=.5, random_state=seed, stratify=remainder.treatment)
    models = []
    for arm in [0,1]:
        subset = train[train.treatment==arm]
        if subset.converted.nunique() != 2:
            raise ValueError('Each training arm requires both outcomes.')
        model = RandomForestClassifier(n_estimators=100, max_depth=8,
                    min_samples_leaf=80, random_state=seed+arm, n_jobs=2)
        model.fit(subset[FEATURES], subset.converted)
        models.append(model)
    def score(frame):
        result = frame.copy()
        result['natural_conversion'] = models[0].predict_proba(frame[FEATURES])[:,1]
        result['uplift'] = models[1].predict_proba(frame[FEATURES])[:,1] - result.natural_conversion
        return result
    return score(validation), score(test)


def top_mask(scores, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('Targeting fraction must be in [0,1].')
    n = len(scores)
    selected = np.zeros(n, dtype=bool)
    selected[np.argsort(-np.asarray(scores), kind='stable')[:int(np.ceil(n*fraction))]] = True
    return selected


def policy_rows(frame, mask):
    """HT/IPW contrasts vs treat-none; denominator is whole eligible population.

    Assignment probability is known to be 0.5 in this synthetic RCT.
    Coupon cost occurs only for treated purchasers. Treatment is assignment,
    not redemption. Features and policy selection must be independent of test outcomes.
    """
    t, e = frame.treatment.to_numpy(), frame.propensity.to_numpy()
    if np.any((e <= 0) | (e >= 1)):
        raise ValueError('Propensities must be strictly between zero and one.')
    weight = t/e - (1-t)/(1-e)
    conversion = mask * weight * frame.converted.to_numpy()
    margin = mask * weight * frame.margin.to_numpy()
    cost = mask * t/e * frame.subsidy.to_numpy()
    return conversion, margin, cost


def evaluate_policy(frame, mask, seed=42, bootstraps=300):
    conversion, margin, cost = policy_rows(frame, mask)
    net = margin-cost
    rng = np.random.default_rng(seed)
    samples = np.array([net[rng.integers(0,len(net),len(net))].mean() for _ in range(bootstraps)])
    c = float(cost.mean())
    return dict(target_share=float(mask.mean()), incremental_conversion=float(conversion.mean()),
        incremental_margin_per_user=float(margin.mean()), subsidy_per_user=c,
        net_profit_per_user=float(net.mean()), net_ci_low=float(np.quantile(samples,.025)),
        net_ci_high=float(np.quantile(samples,.975)),
        net_roi=float(net.mean()/c) if c>0 else None)


def select_fraction(validation):
    # Freeze tuning on validation; test remains untouched.
    candidates = [0,.1,.2,.3,.4,.5]
    gains = [np.mean(policy_rows(validation,top_mask(validation.uplift,k))[1] -
                     policy_rows(validation,top_mask(validation.uplift,k))[2]) for k in candidates]
    return candidates[int(np.argmax(gains))]


def uplift_curve(frame):
    """Normalized cumulative IPW incremental conversions; Qini subtracts random line.

    Diagnostic only: test curve must not be used to tune policy thresholds.
    """
    order = np.argsort(-frame.uplift.to_numpy(),kind='stable')
    t, e, y = (frame[name].to_numpy()[order] for name in ['treatment','propensity','converted'])
    gains = np.r_[0,np.cumsum((t/e-(1-t)/(1-e))*y)/len(frame)]
    fractions = np.linspace(0,1,len(frame)+1)
    random_line = fractions*gains[-1]
    return fractions,gains,float(np.trapezoid(gains-random_line,fractions))
