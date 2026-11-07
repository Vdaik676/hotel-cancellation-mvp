import copy,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,pytest
from src.contract import ROOT,FEATURES,NUMERIC,LIMITS,load_json,validate_record,predict_one,risk_category

@pytest.fixture(scope='session')
def resources():return joblib.load(ROOT/'artifacts/model.joblib'),load_json(ROOT/'artifacts/schema.json'),load_json(ROOT/'demo/examples.json')

def test_twenty_valid_records(resources):
 bundle,schema,_=resources
 for row in load_json(ROOT/'demo/acceptance_inputs.json'):
  result=predict_one(row,bundle,schema)
  assert 0<=result['probability']<=1 and result['risk'] in ['Низкий','Средний','Высокий']

@pytest.mark.parametrize('field,value',[('adults',-1),('children',1.5),('lead_time',float('nan')),('lead_time',float('inf')),('adults',True),('lead_time','30'),('babies',None),('stays_in_week_nights',366),('lead_time',1001)])
def test_invalid_numeric(resources,field,value):
 _,schema,examples=resources;r=copy.deepcopy(examples[0]['input']);r[field]=value
 with pytest.raises(ValueError):validate_record(r,schema)

@pytest.mark.parametrize('change',['missing','extra','zero_guests','zero_nights','too_many_guests','too_many_nights','unknown_category','empty_category'])
def test_contract_rejects_invalid_requests(resources,change):
 _,schema,examples=resources;r=copy.deepcopy(examples[0]['input'])
 if change=='missing':del r['hotel']
 if change=='extra':r['reservation_status']='Canceled'
 if change=='zero_guests':r.update(adults=0,children=0,babies=0)
 if change=='zero_nights':r.update(stays_in_week_nights=0,stays_in_weekend_nights=0)
 if change=='too_many_guests':r.update(adults=20,children=1)
 if change=='too_many_nights':r.update(stays_in_week_nights=365,stays_in_weekend_nights=1)
 if change=='unknown_category':r['hotel']='Unseen Hotel'
 if change=='empty_category':r['meal']=''
 with pytest.raises(ValueError):validate_record(r,schema)

def test_risk_boundaries():
 t=.6
 assert risk_category(0,t)=='Низкий'
 assert risk_category(t/2-1e-8,t)=='Низкий'
 assert risk_category(t/2,t)=='Средний'
 assert risk_category(t-1e-8,t)=='Средний'
 assert risk_category(t,t)=='Высокий'
 assert risk_category(1,t)=='Высокий'

def test_train_only_preprocessing_and_excluded_fields(resources):
 bundle,_,_=resources
 assert list(bundle['pipeline'].feature_names_in_)==FEATURES
 assert not set(FEATURES)&{'is_canceled','reservation_status','reservation_status_date','assigned_room_type','booking_changes','adr','deposit_type'}
 assert bundle['model_id']==load_json(ROOT/'artifacts/selection_before_test.json')['selected_model']

def test_no_group_overlap():
 df=pd.read_csv(ROOT/'artifacts/split_manifest.csv',dtype={'group_id':str})
 assert df.row_id.is_unique
 assert df.groupby('group_id').split.nunique().max()==1
 assert set(df.split)=={'train','validation','test'}

def test_model_artifact_matches_saved_predictions(resources):
 from src.data import prepare
 bundle,_,_=resources;df,_=prepare();test=df[df.split=='test']
 stored=np.load(ROOT/f'artifacts/{bundle["model_id"]}_test.npz')
 assert np.array_equal(stored['row_id'],test.row_id.to_numpy())
 np.testing.assert_allclose(bundle['pipeline'].predict_proba(test[FEATURES])[:,1],stored['p'],rtol=0,atol=1e-12)

def test_baseline_comparison_is_factual():
 report=load_json(ROOT/'reports/metrics.json');models=report['experiments']
 assert len(models)==3 and len({m['test']['n'] for m in models})==1
 assert models[0]['test']['roc_auc']==.5
 selected=next(r for r in models if r['id']==report['selection']['selected_model'])
 assert report['goals']['REQ-04_baseline_beaten']==(selected['test']['roc_auc']>models[0]['test']['roc_auc'] and selected['test']['f1']>models[0]['test']['f1'])

def test_ui_happy_path_and_invalid_form():
 from streamlit.testing.v1 import AppTest
 at=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
 assert not at.exception
 at.button[1].click().run()
 assert not at.exception and len(at.metric)==2
 at.number_input(key='adults').set_value(0)
 at.number_input(key='children').set_value(0)
 at.number_input(key='babies').set_value(0)
 at.button[1].click().run()
 assert not at.exception and len(at.error)==1 and len(at.metric)==0
 assert 'гостей' in at.error[0].value

def test_ui_demo_examples():
 from streamlit.testing.v1 import AppTest
 at=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
 for i in range(3):
  at.selectbox(key='example_index').set_value(i).run()
  at.button[0].click().run();at.button[1].click().run()
  assert not at.exception and len(at.metric)==2
