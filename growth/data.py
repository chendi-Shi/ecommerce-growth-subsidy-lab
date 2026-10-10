"""Synthetic randomized coupon experiment; never presented as Douyin data."""
import numpy as np
import pandas as pd

FEATURES = ['pre_visits', 'pre_cart', 'pre_orders', 'days_since_visit',
            'category', 'scene', 'new_user']


def generate(n=60000, seed=42):
    if n < 1000:
        raise ValueError('Use at least 1000 users for the demonstration.')
    rng = np.random.default_rng(seed)
    visits = rng.poisson(6, n)
    cart = rng.binomial(visits, .18)
    orders = rng.binomial(cart, .25)
    recency = rng.integers(0, 15, n)
    category = rng.integers(0, 5, n)
    scene = rng.integers(0, 3, n)  # synthetic short video / live / mall
    new = rng.binomial(1, .35, n)
    treatment = rng.binomial(1, .5, n)
    base = 1 / (1 + np.exp(-(-3 + .11 * visits + .32 * orders - .07 * recency)))
    # High cart intent with no purchase responds more than natural buyers.
    effect = (.10 * ((cart >= 2) & (orders == 0)) + .04 * new
              + .025 * (scene == 1) - .04 * (orders >= 2))
    p1 = np.clip(base + effect, .005, .9)
    y = rng.binomial(1, np.where(treatment == 1, p1, base))
    order_value = rng.uniform(60, 180, n)
    coupon = 8.0
    df = pd.DataFrame(dict(user_id=np.arange(n), pre_visits=visits, pre_cart=cart,
        pre_orders=orders, days_since_visit=recency, category=category, scene=scene,
        new_user=new, treatment=treatment, propensity=.5, converted=y,
        revenue=y * order_value, margin=y * order_value * .25,
        subsidy=treatment * y * coupon))
    # Ground truth is a separate diagnostics table, never a model feature.
    truth = pd.DataFrame(dict(user_id=np.arange(n), true_uplift=p1-base))
    return df, truth


def behavior_metrics(path, chunksize=500000):
    """Stream the public five-column, headerless Tianchi UserBehavior CSV.

    Aggregate distinct users per behavior, not an ordered conversion funnel.
    No revenue, subsidy, scene, or causal conclusions are inferred.
    """
    daily = {}
    for chunk in pd.read_csv(path, header=None,
            names=['user_id','item_id','category_id','behavior','timestamp'],
            chunksize=chunksize):
        if not chunk.behavior.isin(['pv','cart','fav','buy']).all():
            raise ValueError('Expected headerless UserBehavior CSV with pv/cart/fav/buy.')
        chunk['date'] = pd.to_datetime(chunk.timestamp, unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
        for (date, behavior), group in chunk.groupby(['date', 'behavior']):
            entry = daily.setdefault((date, behavior), {'events':0, 'users':set()})
            entry['events'] += len(group)
            entry['users'].update(group.user_id)
    return pd.DataFrame([dict(date=d, behavior=b, events=v['events'], users=len(v['users']))
                         for (d,b),v in sorted(daily.items())])
