"""Re-evaluate frozen models on the saved test manifest, never retrain."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,time,platform
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from src.contract import ROOT,FEATURES,TARGET,load_json,save_json,predict_one
from src.data import prepare
from src.train import metrics

def run():
 df,audit=prepare();manifest=pd.read_csv(ROOT/'artifacts/split_manifest.csv',dtype={'group_id':str})
 expected=df[['row_id','group_id','split']].reset_index(drop=True)
 if not expected.equals(manifest):raise ValueError('Сохранённое разбиение не соответствует исходным данным.')
 test=df[df.split=='test'];report=load_json(ROOT/'reports/metrics.json');res={};max_error=0.
 with threadpool_limits(limits=2):
  for r in report['experiments']:
   model=joblib.load(ROOT/f'artifacts/{r["id"]}.joblib');p=model.predict_proba(test[FEATURES])[:,1]
   current=metrics(test[TARGET],p,r['validation']['threshold']);res[r['id']]=current
   max_error=max(max_error,max(abs(current[k]-r['test'][k]) for k in ['roc_auc','f1','recall','precision','brier']))
  bundle=joblib.load(ROOT/'artifacts/model.joblib');schema=load_json(ROOT/'artifacts/schema.json');inputs=load_json(ROOT/'demo/acceptance_inputs.json')
  predict_one(inputs[0],bundle,schema);times=[]
  for row in inputs:
   start=time.perf_counter();predict_one(row,bundle,schema);times.append(time.perf_counter()-start)
 result={'max_metric_error':max_error,'repeatability_pass':max_error<=1e-6,'test_rows':len(test),'metrics':res,'latency':{'n':len(times),'unit':'seconds','p95':float(np.percentile(times,95)),'maximum':max(times),'measurements':times,'pass':float(np.percentile(times,95))<=3,'includes':'validation + DataFrame + model inference; excludes page rendering and cold load','environment':{'processor':platform.processor(),'platform':platform.platform(),'python':platform.python_version()}},'data_sha256':audit['sha256']}
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/evaluation_repeat.json');args=parser.parse_args()
 result=run();save_json(ROOT/args.output,result);print(result)
