"""Offline feature audit. The public interface and release model stay unchanged."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import joblib,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits
from src.train import metrics,threshold,prepare
from src.contract import ROOT,FEATURES,CATEGORICAL,TARGET,save_json

EARLY=['previous_cancellations','previous_bookings_not_canceled','is_repeated_guest',
       'distribution_channel','customer_type','arrival_date_day_of_month']
SNAPSHOT=['total_of_special_requests','required_car_parking_spaces']
CAT=['distribution_channel','customer_type']

def main():
    df,audit=prepare()
    raw=pd.read_csv(ROOT/'data/hotels.csv',usecols=EARLY+SNAPSHOT)
    df=df.join(raw,on='row_id',validate='one_to_one')
    tr,va=(df[df.split==s] for s in ('train','validation'))
    runs=[]
    models={}
    with threadpool_limits(limits=2):
        for name,extra in [('E3_booking_history',EARLY),('E4_snapshot_requests',EARLY+SNAPSHOT)]:
            f=FEATURES+extra
            cats=CATEGORICAL+CAT
            nums=[k for k in f if k not in cats]
            pre=ColumnTransformer([('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cats),('num','passthrough',nums)])
            model=make_pipeline(pre,HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=31,
                       categorical_features=list(range(len(cats))),learning_rate=.07,l2_regularization=2,
                       early_stopping=False,random_state=42))
            model.fit(tr[f],tr[TARGET])
            p=model.predict_proba(va[f])[:,1]
            t=threshold(va[TARGET],p)
            runs.append({'id':name,'features':f,'validation':metrics(va[TARGET],p,t)})
            models[name]=model
        # Freeze validation decision before computing held-out scores. Prior E2 metrics are historical.
        best=max(runs,key=lambda r:r['validation']['f1'])
        save_json(ROOT/'reports/research_selection.json',{'candidate':best['id'],'threshold':best['validation']['threshold'],
            'basis':'maximum validation F1 with recall >= 0.70, experimental only; no change to production input contract'})
        te=df[df.split=='test']
        for r in runs:
            p=models[r['id']].predict_proba(te[r['features']])[:,1]
            r['test']=metrics(te[TARGET],p,r['validation']['threshold'])
        save_json(ROOT/'reports/research_metrics.json',{'status':'offline_only','runs':runs,'source_sha256':audit['sha256'],
            'limits':['Reused historical E2 benchmark, not a fresh external test.',
            'Historical data is a final snapshot. Availability of requests and parking at booking is unverified.',
            'No assigned room, status, status date, deposit, agent/company IDs or nationality in these candidates.',
            'Release stays E2 because expanded feature contract and point-in-time availability need validation.']})
        print([(r['id'],r['test']) for r in runs],flush=True)

if __name__=='__main__':main()
