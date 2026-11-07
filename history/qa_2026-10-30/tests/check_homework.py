"""Run with the MVP's Python: python check_homework.py --app PATH --out PATH."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
p=argparse.ArgumentParser(); p.add_argument('--app',required=True);p.add_argument('--out',required=True)
a=p.parse_args(); root=Path(a.app).resolve(); out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root))
from streamlit.testing.v1 import AppTest
import streamlit, joblib
from src.contract import load_json,predict_one,validate_record,risk_category
bundle=joblib.load(root/'artifacts/model.joblib');schema=load_json(root/'artifacts/schema.json');examples=load_json(root/'demo/examples.json')
results=[]
def record(id,title,ok,details):
 results.append(dict(id=id,title=title,status='PASS' if ok else 'FAIL',details=details))
def app():return AppTest.from_file(str(root/'streamlit_app.py'),default_timeout=60).run()
def snapshot(at):return {'exceptions':[x.message for x in at.exception],'errors':[x.value for x in at.error],'metrics':[{ 'label':x.label,'value':x.value} for x in at.metric]}
at=app();record('TC-01','Запуск интерфейса',not at.exception and len(at.number_input)==6 and len(at.selectbox)==6,snapshot(at))
for i,e in enumerate(examples):
 at.selectbox(key='example_index').set_value(i).run();at.button[0].click().run();at.button[1].click().run()
 actual=at.session_state['prediction'];expected=e['expected']
 record(f'TC-0{i+2}',e['name'],not at.exception and abs(actual['probability']-expected['probability'])<1e-12 and actual['risk']==expected['risk'],dict(input=e['input'],actual=actual,expected=expected,ui=snapshot(at)))
for id,title,changes,word in [('TC-05','Ноль гостей',dict(adults=0,children=0,babies=0),'гостей'),('TC-06','Ноль ночей',dict(stays_in_weekend_nights=0,stays_in_week_nights=0),'ночей'),('TC-07','Суммарные ограничения',dict(adults=20,children=1,babies=0),'гостей')]:
 at.button[0].click().run();at.button[1].click().run()
 for k,v in changes.items():at.number_input(key=k).set_value(v)
 at.button[1].click().run()
 record(id,title,not at.exception and len(at.error)==1 and word in at.error[0].value and len(at.metric)==0,snapshot(at))
at.button[0].click().run()
for k,v in dict(lead_time=1000,stays_in_weekend_nights=0,stays_in_week_nights=365,adults=20,children=0,babies=0).items():at.number_input(key=k).set_value(v)
at.button[1].click().run();record('TC-08','Допустимая верхняя граница',not at.exception and len(at.metric)==2,snapshot(at))
at.button[0].click().run();at.button[1].click().run();at.button[0].click().run()
record('TC-09','Очистка результата при заполнении примера',not at.exception and len(at.metric)==0,snapshot(at))
bad=[]
for change in [dict(adults=-1),dict(children=1.5),dict(lead_time=None),dict(hotel='Unknown')]:
 r={**examples[0]['input'],**change}
 try:validate_record(r,schema);bad.append({'input_change':change,'rejected':False})
 except ValueError as e:bad.append({'input_change':change,'rejected':True,'message':str(e)})
record('TC-10','Некорректные значения на уровне функции',all(x['rejected'] for x in bad),bad)
# Fault injection is confined to a dedicated copy; restore file even if test fails.
example_path=root/'demo/examples.json';original=example_path.read_bytes()
try:
 example_path.rename(root/'demo/examples.test-backup.json')
 broken=app()
 record('TC-11','Отсутствует файл демонстрационных примеров',not broken.exception and len(broken.error)>0,snapshot(broken))
finally:
 (root/'demo/examples.test-backup.json').rename(example_path)
at=app()
fields=[{'label':at.selectbox(key=k).label,'options':at.selectbox(key=k).options,'help':at.selectbox(key=k).proto.help} for k in ['meal','market_segment','reserved_room_type']]
record('TC-12','Пояснения к кодам в форме',all(bool(f['help']) for f in fields),{'fields':fields,'note':'Дополнительная проверка удобства: в форме нет расшифровки кодов и подсказок; это замечание UX, не численный критерий ТЗ.'})
manifest={str(f.relative_to(root)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'streamlit_app.py',root/'src/contract.py',root/'artifacts/model.joblib',root/'demo/examples.json',root/'requirements-lock.txt']}
payload={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'python':platform.python_version(),'streamlit':streamlit.__version__,'method':'Streamlit AppTest and direct contract calls','source_sha256':manifest,'results':results,'pass':sum(x['status']=='PASS' for x in results),'fail':sum(x['status']=='FAIL' for x in results)}
(out/'test_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pass':payload['pass'],'fail':payload['fail'],'cases':[(r['id'],r['status']) for r in results]},ensure_ascii=False))
