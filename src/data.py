"""Deterministic cleaning and split with identical-input groups kept together."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .contract import ROOT,FEATURES,NUMERIC,CATEGORICAL,TARGET,LIMITS,MONTHS,load_json

def prepare():
 path=ROOT/'data/hotels.csv'
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 if actual!=load_json(ROOT/'data/source.json')['sha256']:raise ValueError('SHA-256 исходного CSV не совпадает. Проверьте data/source.json.')
 raw=pd.read_csv(path)
 if set(FEATURES+[TARGET])-set(raw):raise ValueError('В CSV отсутствуют обязательные столбцы.')
 raw.insert(0,'row_id',np.arange(len(raw),dtype=int))
 duplicates=raw.drop(columns='row_id').duplicated()
 df=raw.loc[~duplicates].copy()
 valid=df[TARGET].isin([0,1])
 for key in NUMERIC:
  value=pd.to_numeric(df[key],errors='coerce');lo,hi=LIMITS[key]
  valid&=value.notna()&np.isfinite(value)&value.between(lo,hi)&value.eq(value.round())
 for key in CATEGORICAL:valid&=df[key].notna()&df[key].astype(str).str.strip().ne('')
 valid&=df['arrival_date_month'].isin(MONTHS)
 valid&=(df['stays_in_weekend_nights']+df['stays_in_week_nights']).between(1,365)
 valid&=(df['adults']+df['children']+df['babies']).between(1,20)
 kept=df.loc[valid,['row_id']+FEATURES+[TARGET]].copy()
 for key in NUMERIC+[TARGET]:kept[key]=kept[key].astype(int)
 kept['group_id']=pd.util.hash_pandas_object(kept[FEATURES],index=False).astype(str)
 # Group-only stratification is determined before experiments. No outcome is used as a feature.
 groups=kept.groupby('group_id')[TARGET].mean().reset_index()
 strata=np.where(groups[TARGET].eq(0),'no_cancel',np.where(groups[TARGET].eq(1),'cancel','mixed'))
 tr,rest=train_test_split(groups['group_id'].to_numpy(),test_size=.4,random_state=42,stratify=strata)
 rg=groups.set_index('group_id').loc[rest,TARGET]
 rs=np.where(rg.eq(0),'no_cancel',np.where(rg.eq(1),'cancel','mixed'))
 va,te=train_test_split(rest,test_size=.5,random_state=42,stratify=rs)
 mapping={g:split for split,ids in [('train',tr),('validation',va),('test',te)] for g in ids}
 kept['split']=kept['group_id'].map(mapping)
 audit={'raw_rows':len(raw),'raw_columns':len(raw.columns)-1,'raw_positive_rate':float(raw[TARGET].mean()),'exact_duplicates_removed':int(duplicates.sum()),'invalid_rows_removed':int((~valid).sum()),'usable_rows':len(kept),'input_groups':len(groups),'sha256':actual,'split_seed':42,'split_method':'60/20/20 of input groups; stratified by group outcome profile; row shares can differ','split':{s:{'rows':int((kept['split']==s).sum()),'positive_rate':float(kept.loc[kept['split']==s,TARGET].mean()),'groups':int(kept.loc[kept['split']==s,'group_id'].nunique())} for s in ['train','validation','test']},'missing_raw':{k:int(v) for k,v in raw.isna().sum().items() if v},'features':FEATURES,'excluded_columns':[c for c in raw.columns if c not in FEATURES+['row_id']], 'availability_limit':'CSV is a final historical snapshot; fields are conservatively chosen candidates, not verified point-in-time snapshots.'}
 return kept,audit
