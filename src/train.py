"""Train a prior baseline and two candidates, lock selection before opening test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from datetime import datetime,timezone
import hashlib,time,platform,sys,importlib.metadata
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix,precision_recall_curve,brier_score_loss
from threadpoolctl import threadpool_limits
from .data import prepare
from .contract import ROOT,FEATURES,NUMERIC,CATEGORICAL,TARGET,MONTHS,LIMITS,save_json,predict_one

def metrics(y,p,t):
 pred=p>=t
 return {'roc_auc':float(roc_auc_score(y,p)),'f1':float(f1_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'precision':float(precision_score(y,pred,zero_division=0)),'brier':float(brier_score_loss(y,p)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'threshold':float(t),'n':len(y)}

def threshold(y,p):
 precision,recall,ths=precision_recall_curve(y,p)
 f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
 choices=np.flatnonzero((recall[:-1]>=.70)&(ths>0)&(ths<=1))
 if not len(choices):choices=np.flatnonzero((ths>0)&(ths<=1))
 best=max(choices,key=lambda i:(f1[i],recall[i],ths[i]))
 return float(ths[best])

def pipeline(estimator):
 pre=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],remainder='drop')
 return Pipeline([('preprocess',pre),('classifier',estimator)])

def main():
 for name in ['artifacts','reports','docs','tests','demo']: (ROOT/name).mkdir(exist_ok=True)
 df,audit=prepare();df[['row_id','group_id','split']].to_csv(ROOT/'artifacts/split_manifest.csv',index=False)
 save_json(ROOT/'reports/data_audit.json',audit)
 train=df[df.split=='train'];val=df[df.split=='validation']
 schema={'version':'1.0','fields':FEATURES,'numeric_limits':LIMITS,'categories':{k:([m for m in MONTHS if m in set(train[k])] if k=='arrival_date_month' else sorted(train[k].unique().tolist())) for k in CATEGORICAL},'unknown_policy':'UI rejects unseen categories; pipeline encoder ignores unforeseen categories for offline evaluation','prediction_moment':'registration of booking; point-in-time validity not verified by original snapshot'}
 save_json(ROOT/'artifacts/schema.json',schema)
 models=[('B0_prior','Reference prevalence, no input features',DummyClassifier(strategy='prior')),('E1_logistic','Linear relationships on the conservative input set',LogisticRegression(max_iter=1000,C=1.0,random_state=42)),('E2_hist_gradient_boosting','Nonlinear interactions on exactly the same inputs',HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=15,learning_rate=.08,l2_regularization=2.0,early_stopping=False,random_state=42))]
 records=[];fitted={}
 with threadpool_limits(limits=2):
  for model_id,hypothesis,est in models:
   start=time.perf_counter();model=pipeline(est);model.fit(train[FEATURES],train[TARGET]);elapsed=time.perf_counter()-start
   prob=model.predict_proba(val[FEATURES])[:,1];t=.5 if model_id=='B0_prior' else threshold(val[TARGET],prob)
   record={'id':model_id,'hypothesis':hypothesis,'validation':metrics(val[TARGET],prob,t),'fit_seconds':elapsed,'parameters':est.get_params(),'seed':42}
   records.append(record);fitted[model_id]=model
   print(model_id,record['validation'],flush=True)
  candidates=records[1:];selected=max(candidates,key=lambda r:(r['validation']['recall']>=.7,r['validation']['f1'],r['validation']['roc_auc']))
  decision={'selected_model':selected['id'],'threshold':selected['validation']['threshold'],'basis':'Highest validation F1 among recall>=0.70; tie by ROC-AUC. No test metric used for selection.','recorded_utc':datetime.now(timezone.utc).isoformat(),'train_refit':False}
  save_json(ROOT/'artifacts/selection_before_test.json',decision)
  test=df[df.split=='test'];results=[]
  for rec in records:
   model_id=rec['id'];p=fitted[model_id].predict_proba(test[FEATURES])[:,1]
   rec['test']=metrics(test[TARGET],p,rec['validation']['threshold']);results.append(rec)
   np.savez_compressed(ROOT/f'artifacts/{model_id}_test.npz',row_id=test['row_id'].to_numpy(),y=test[TARGET].to_numpy(),p=p)
   joblib.dump(fitted[model_id],ROOT/f'artifacts/{model_id}.joblib')
  bundle={'pipeline':fitted[selected['id']],'threshold':decision['threshold'],'model_id':selected['id'],'data_sha256':audit['sha256']}
  joblib.dump(bundle,ROOT/'artifacts/model.joblib')
  examples=[]
  valid=[]
  for _,row in test.iterrows():
   r={k:int(row[k]) if k in NUMERIC else row[k] for k in FEATURES}
   if all(r[k] in schema['categories'][k] for k in CATEGORICAL):valid.append(r)
  probabilities=bundle['pipeline'].predict_proba(pd.DataFrame(valid,columns=FEATURES))[:,1]
  for idx in [int(np.argmin(probabilities)),int(np.argsort(probabilities)[len(probabilities)//2]),int(np.argmax(probabilities))]:
   examples.append({'name':['Пример 1','Пример 2','Пример 3'][len(examples)],'input':valid[idx],'expected':predict_one(valid[idx],bundle,schema)})
  save_json(ROOT/'demo/examples.json',examples)
  save_json(ROOT/'demo/acceptance_inputs.json',valid[:20])
  versions={k:importlib.metadata.version(k) for k in ['scikit-learn','streamlit','numpy','pandas','scipy','joblib','pytest','matplotlib']}
  sm=next(r['test'] for r in results if r['id']==selected['id']);bm=results[0]['test']
  report={'date':'2026-10-23','selection':decision,'experiments':results,'goals':{'REQ-04_baseline_beaten':sm['roc_auc']>bm['roc_auc'] and sm['f1']>bm['f1'],'REQ-05_recall_070':sm['recall']>=.7,'REQ-06_auc_080':sm['roc_auc']>=.8,'REQ-06_f1_070':sm['f1']>=.7},'versions':versions,'python':platform.python_version(),'split':audit['split'],'limitations':['Historical random grouped split; no prospective hotel validation.','No PMS/CRM and no causal or revenue benefit proved.','Final snapshot cannot prove registration-time availability.','Probability estimates not calibrated on a separate calibration set.','Should targets are reported, never silently weakened.']}
  save_json(ROOT/'reports/metrics.json',report)
  print('SELECTED',decision,'TEST',sm,'GOALS',report['goals'],flush=True)
 # Record the exact execution environment, not globally installed unrelated packages.
 with (ROOT/'requirements.txt').open('w',encoding='utf-8') as f:
  for name,version in versions.items():f.write(f'{name}=={version}\n')
 # The release lock is maintained independently. Training must not overwrite it.
if __name__=='__main__':main()
