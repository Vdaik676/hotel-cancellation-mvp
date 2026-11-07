"""Describe validation trade-offs without changing the model, split, or threshold."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import joblib, numpy as np
from sklearn.metrics import precision_recall_curve,roc_auc_score
from threadpoolctl import threadpool_limits
from src.data import prepare
from src.contract import ROOT,FEATURES,TARGET,save_json,load_json
from src.train import metrics

def main():
    data,audit=prepare()
    va=data[data.split=='validation']
    bundle=joblib.load(ROOT/'artifacts/model.joblib')
    with threadpool_limits(limits=2):
        pred=bundle['pipeline'].predict_proba(va[FEATURES])[:,1]
    prec,rec,t=precision_recall_curve(va[TARGET],pred)
    f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-15)
    groups=va.groupby('group_id')[TARGET].agg(['size','sum','mean'])
    mixed=(groups['sum']>0)&(groups['sum']<groups['size'])
    # Descriptive oracle: estimates what a classifier knowing empirical group
    # outcomes could achieve on these rows. Not a deployable model or fresh test.
    groups=groups.sort_values('mean',ascending=False)
    possible_f1=2*groups['sum'].cumsum()/(groups['size'].cumsum()+groups['sum'].sum())
    oracle_scores=va['group_id'].map(groups['mean'])
    report={'basis':'validation only; no model or threshold changed',
            'validation_rows':len(va),'validation_cancellation_rate':float(va[TARGET].mean()),
            'mixed_input_groups':int(mixed.sum()),'rows_in_mixed_input_groups':int(groups['size'].where((groups['sum']>0)&(groups['sum']<groups['size']),0).sum()),
            'e2_frozen_validation':metrics(va[TARGET],pred,bundle['threshold']),
            'e2_best_f1_any_validation_threshold':float(f.max()),
            'e2_best_f1_validation_recall_at_least_070':float(f[rec[:-1]>=.7].max()),
            'precision_needed_for_f1_070_at_recall_070':.7,
            'validation_descriptive_group_oracle_f1':float(possible_f1.max()),
            'validation_descriptive_group_oracle_auc':float(roc_auc_score(va[TARGET],oracle_scores)),
            'oracle_limit':'Uses validation labels; descriptive only, not a generalization bound. A stronger model or new known-at-booking features could improve performance.',
            'test_frozen':next(r['test'] for r in load_json(ROOT/'reports/metrics.json')['experiments'] if r['id']==bundle['model_id'])}
    save_json(ROOT/'reports/metrics_diagnosis.json',report)
    print(report)
if __name__=='__main__':main()
