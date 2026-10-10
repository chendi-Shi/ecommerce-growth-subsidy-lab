import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold,train_test_split
from .stats import dr_scores

NUMERIC=['recency','log_history','mens','womens','newbie']
CATEGORICAL=['zip_code','channel']
FEATURES=NUMERIC+CATEGORICAL


def feature_frame(frame):
    x=frame.copy(); x['log_history']=np.log1p(x.history)
    return x[FEATURES]


def preprocessing():
    return ColumnTransformer([('numeric',StandardScaler(),NUMERIC),
        ('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)])


def split(frame,seed=2026):
    # Split independently of realized outcomes; report rare-buyer imbalance honestly.
    train,rest=train_test_split(frame,test_size=.4,random_state=seed,stratify=frame.treatment)
    val,test=train_test_split(rest,test_size=.5,random_state=seed,stratify=rest.treatment)
    return train.copy(),val.copy(),test.copy()


def revenue_models(train,kind='ridge',seed=2026):
    models=[]
    for arm in [0,1,2]:
        subset=train[train.treatment==arm]
        estimator=Ridge(alpha=100) if kind=='ridge' else RandomForestRegressor(
            n_estimators=100,max_depth=5,min_samples_leaf=250,n_jobs=2,random_state=seed+arm)
        model=make_pipeline(preprocessing(),estimator)
        model.fit(feature_frame(subset),subset.spend)
        models.append(model)
    return models


def predictions(models,frame):
    return np.column_stack([model.predict(feature_frame(frame)) for model in models])


def fit_all(train,seed=2026):
    ridge=revenue_models(train,'ridge',seed)
    forest=revenue_models(train,'forest',seed)
    # Training-only cross-fitted residuals for DR-Learner; never test labels.
    oof=np.zeros((len(train),3))
    for fit_idx,score_idx in StratifiedKFold(5,shuffle=True,random_state=seed).split(train,train.treatment):
        models=revenue_models(train.iloc[fit_idx],'ridge',seed)
        oof[score_idx]=predictions(models,train.iloc[score_idx])
    pseudo=dr_scores(train.spend,train.treatment,oof)
    dr=[]
    for arm in [1,2]:
        model=make_pipeline(preprocessing(),RandomForestRegressor(n_estimators=100,
            max_depth=4,min_samples_leaf=300,n_jobs=2,random_state=seed+10+arm))
        model.fit(feature_frame(train),pseudo[:,arm]-pseudo[:,0]); dr.append(model)
    conversion=make_pipeline(preprocessing(),LogisticRegression(C=.5,max_iter=1000))
    control=train[train.treatment==0]
    conversion.fit(feature_frame(control),control.conversion)
    return dict(ridge=ridge,forest=forest,dr=dr,conversion=conversion)


def rankers(models,frame):
    ridge=predictions(models['ridge'],frame); forest=predictions(models['forest'],frame)
    return dict(t_ridge=ridge[:,1:]-ridge[:,[0]],
        t_forest=forest[:,1:]-forest[:,[0]],
        dr_learner=np.column_stack([model.predict(feature_frame(frame)) for model in models['dr']]))
