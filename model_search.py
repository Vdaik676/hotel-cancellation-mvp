"""Compare fixed-feature models. Selection uses validation only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import time
import numpy as np,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder,PolynomialFeatures,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from src.data import prepare
from src.contract import ROOT,FEATURES,NUMERIC,CATEGORICAL,TARGET,save_json,load_json
from src.train import metrics,threshold,pipeline

def binned_logit(c):
    prep=ColumnTransformer([
        ('bins',KBinsDiscretizer(n_bins=8,encode='onehot',strategy='quantile',quantile_method='averaged_inverted_cdf',subsample=None),NUMERIC),
        ('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)
    ],sparse_threshold=1)
    return Pipeline([('preprocess',prep),('pairs',PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),
                     ('classifier',LogisticRegression(C=c,max_iter=500,solver='liblinear',random_state=42))])

def native_hgb(leaves,lr,n):
    prep=ColumnTransformer([('category',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CATEGORICAL),('numeric','passthrough',NUMERIC)])
    est=HistGradientBoostingClassifier(max_iter=n,max_leaf_nodes=leaves,categorical_features=list(range(len(CATEGORICAL))),
                   learning_rate=lr,l2_regularization=3,early_stopping=False,random_state=42)
    return Pipeline([('preprocess',prep),('classifier',est)])

def main():
    df,audit=prepare()
    tr,va=(df[df.split==s] for s in ('train','validation'))
    fixed=load_json(ROOT/'reports/metrics.json')
    candidates={
        'E5_logit_interactions_C025':binned_logit(.25),
        'E6_logit_interactions_C1':binned_logit(1),
        'E7_random_forest':pipeline(RandomForestClassifier(n_estimators=100,max_depth=16,min_samples_leaf=14,n_jobs=2,random_state=42,max_features=.65)),
        'E8_extra_trees':pipeline(ExtraTreesClassifier(n_estimators=100,max_depth=18,min_samples_leaf=12,n_jobs=2,random_state=42,max_features=.65)),
        'E9_hgb_categories':native_hgb(31,.07,190),
        'E10_hgb_regularized':native_hgb(63,.05,210),
    }
    scores={}; rows=[]; fitted={}
    with threadpool_limits(limits=2):
        for original in fixed['experiments']:
            name=original['id']
            fitted[name]=joblib.load(ROOT/f'artifacts/{name}.joblib')
            pred=fitted[name].predict_proba(va[FEATURES])[:,1]
            scores[name]=pred
            rows.append({'id':name,'validation':metrics(va[TARGET],pred,original['validation']['threshold'])})
        for name,model in candidates.items():
            start=time.perf_counter()
            model.fit(tr[FEATURES],tr[TARGET])
            fitted[name]=model
            pred=model.predict_proba(va[FEATURES])[:,1]
            scores[name]=pred
            row={'id':name,'validation':metrics(va[TARGET],pred,threshold(va[TARGET],pred)),
                 'fit_seconds':time.perf_counter()-start,'parameters':model[-1].get_params()}
            rows.append(row)
            print(name,row['validation'],flush=True)
        ensemble_ids=('E2_hist_gradient_boosting','E6_logit_interactions_C1','E7_random_forest','E9_hgb_categories')
        name='E11_mean_ensemble'
        pred=np.mean([scores[k] for k in ensemble_ids],axis=0)
        rows.append({'id':name,'components':list(ensemble_ids),'validation':metrics(va[TARGET],pred,threshold(va[TARGET],pred))})
        # Prioritize all original Should targets on validation, then F1 with recall >= 0.70.
        def rank(row):
            m=row['validation']
            return (m['roc_auc']>=.8 and m['f1']>=.7 and m['recall']>=.7,m['recall']>=.7,m['f1'],m['roc_auc'])
        winner=max(rows[1:],key=rank)
        save_json(ROOT/'reports/model_search_selection.json',{'selected':winner['id'],'threshold':winner['validation']['threshold'],
            'basis':'validation only, all goals then recall floor then F1 then AUC','train_rows':len(tr),'validation_rows':len(va)})
        te=df[df.split=='test']
        test_scores={k:m.predict_proba(te[FEATURES])[:,1] for k,m in fitted.items()}
        test_scores[name]=np.mean([test_scores[k] for k in ensemble_ids],axis=0)
        for row in rows:
            row['test']=metrics(te[TARGET],test_scores[row['id']],row['validation']['threshold'])
        save_json(ROOT/'reports/model_search.json',{'release_model_unchanged':True,'selected':winner['id'],
                 'source_sha256':audit['sha256'],'features':FEATURES,
                 'limits':'Historical split already benchmarked; not a new independent evaluation. E3/E4 with extra fields reported separately.',
                 'experiments':rows})
        print('VALIDATION WINNER',winner,flush=True)

if __name__=='__main__':main()
