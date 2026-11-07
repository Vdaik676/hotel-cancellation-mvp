"""Local single-booking demo. Model loading is cached; no network APIs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,sqlite3,joblib,pandas as pd,streamlit as st
from threadpoolctl import threadpool_limits
from src.contract import ROOT,NUMERIC,CATEGORICAL,LABELS,LIMITS,MONTHS,load_json,predict_one,validate_record
from src.history import MAX_RESULTS,save_result,read_results,clear_results

st.set_page_config(page_title='Риск отмены бронирования',page_icon='🏨',layout='wide')
@st.cache_resource
def resources():
 return joblib.load(ROOT/'artifacts/model.joblib'),load_json(ROOT/'artifacts/schema.json'),load_json(ROOT/'reports/metrics.json')
try:bundle,schema,report=resources()
except (FileNotFoundError,OSError,ValueError,ImportError):
 st.error('Не удалось загрузить модель. Проверьте наличие папки artifacts и версии зависимостей по README.');st.stop()

st.title('Риск отмены бронирования')
st.write('Введите параметры одной брони и получите оценку риска для ручного рассмотрения менеджером.')
st.caption('Версия 1.2. Учебный прогноз на исторических бронированиях двух отелей.')
history_path=ROOT/'runtime'/'risk_history.sqlite3'
try:
 examples=load_json(ROOT/'demo/examples.json')
 if not isinstance(examples,list) or len(examples)<2: raise ValueError('Неполный набор примеров')
 for item in examples:
  if not isinstance(item['name'],str): raise ValueError('Некорректное название примера')
  validate_record(item['input'],schema)
except (OSError,ValueError,KeyError,TypeError):
 examples=[]
 st.warning('Демонстрационные примеры недоступны. Можно заполнить форму вручную. Для восстановления примеров распакуйте demo/examples.json из архива этой версии.')
MONTH_RU=dict(zip(MONTHS,['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь']))
translation={'City Hotel':'Городской отель','Resort Hotel':'Курортный отель'}
HELP={
 'meal':'BB — завтрак, HB — полупансион, FB — полный пансион, SC — без включённого питания. Undefined — неизвестный тип.',
 'market_segment':'Direct — напрямую; Online TA — онлайн-агентство; Offline TA/TO — офлайн-агентство или туроператор; Groups — группы; Corporate — корпоративный сегмент; Complementary — бесплатное размещение; Aviation — авиационный сегмент.',
 'reserved_room_type':'Буквы — условные коды исходного набора. Они не обозначают конкретные категории вроде «стандарт» или «люкс».',
}
tab,quality,history=st.tabs(['Оценить бронирование','Качество и ограничения','История проверок'])
with tab:
 def fill_example():
  example=examples[int(st.session_state['example_index'])]
  for k,v in example['input'].items():st.session_state[k]=v
  st.session_state.pop('prediction',None)
 if examples:
  with st.container(horizontal=True):
   st.selectbox('Демонстрационный пример',range(len(examples)),index=1,format_func=lambda i:examples[i]['name'],key='example_index')
   st.button('Заполнить пример',on_click=fill_example)
 defaults={k:schema['categories'][k][0] for k in CATEGORICAL}
 defaults.update({k:(1 if k in ('adults','stays_in_week_nights') else LIMITS[k][0]) for k in NUMERIC})
 for k,v in (examples[1]['input'] if examples else defaults).items():
  if k not in st.session_state:st.session_state[k]=v
 with st.expander('Подсказки к полям бронирования'):
  for key,help_text in HELP.items(): st.markdown(f'**{LABELS[key]}:** {help_text}')
 with st.form('booking'):
  left,right=st.columns(2)
  record={}
  with left:
   for k in CATEGORICAL:
    fmt=(lambda v:MONTH_RU.get(v,v)) if k=='arrival_date_month' else (lambda v:translation.get(v,v))
    record[k]=st.selectbox(LABELS[k],schema['categories'][k],format_func=fmt,key=k,help=HELP.get(k))
  with right:
   for k in NUMERIC:
    low,high=LIMITS[k]
    record[k]=st.number_input(LABELS[k],min_value=low,max_value=high,step=1,key=k)
  keep=st.checkbox('Сохранить результат в локальную историю',key='keep_result',help='Сохраняются время UTC, оценка, категория, модель и порог. Параметры брони не сохраняются.')
  submitted=st.form_submit_button('Оценить риск',type='primary')
 if submitted:
  st.session_state.pop('prediction',None)
  try:
   with threadpool_limits(limits=2):st.session_state['prediction']=predict_one(record,bundle,schema)
  except ValueError as error:st.error(str(error))
  else:
   if keep:
    try:
     save_result(history_path,st.session_state['prediction'])
     st.success('Результат сохранён в локальной истории.')
    except (OSError,sqlite3.Error):
     st.warning('Прогноз рассчитан, но история недоступна. Проверьте доступ на запись в папку приложения.')
 result=st.session_state.get('prediction')
 if result:
  probability=result['probability']
  threshold=result['threshold']
  near_boundary=min(abs(probability-threshold/2),abs(probability-threshold))<.001
  a,b=st.columns(2);a.metric('Оценка вероятности отмены',f'{probability:.4%}' if near_boundary else f'{probability:.1%}');b.metric('Категория риска',result['risk'])
  st.progress(result['probability'])
  st.write({'Низкий':'Риск ниже установленной границы. Прогноз не гарантирует сохранение брони.','Средний':'Бронирование можно включить в список для ручного наблюдения.','Высокий':'Рассмотрите бронь вручную в рамках правил отеля. Автоматические действия не выполняются.'}[result['risk']])
  st.caption(f'Границы риска: средний от {threshold/2:.4%}, высокий от {threshold:.4%}. Значения на экране округлены; категория рассчитана без округления.')
  st.download_button('Сохранить результат JSON',json.dumps(result,ensure_ascii=False,indent=2),'prediction.json','application/json')
 st.info('Не вводите персональные данные. Оценка обучена на историческом наборе и не прошла проверку на будущих бронированиях вашего отеля.')
with quality:
 st.subheader('Качество прогноза')
 active=next(r['test'] for r in report['experiments'] if r['id']==bundle['model_id'])
 st.write('Рабочая модель — E2. Результаты получены на отложенной исторической выборке.')
 st.dataframe(pd.DataFrame([{'ROC-AUC':round(active['roc_auc'],4),'F1':round(active['f1'],4),'Recall':round(active['recall'],4)}]),hide_index=True,width='stretch')
 st.caption('ROC-AUC — различение отменённых и сохранённых броней; F1 — баланс precision и recall; recall — доля найденных отмен. Эти показатели не являются вероятностью для конкретной брони.')
 st.markdown('**Ограничения:** модель проверена на исторических данных двух отелей за 2015–2017 годы. Качество на будущих бронированиях вашего отеля не проверено. Исходный набор не позволяет полностью подтвердить значения полей на момент регистрации. Вероятности отдельно не калиброваны. Используйте оценку для ручного рассмотрения, а не для автоматической отмены брони.')
with history:
 st.subheader('Локальная история проверок')
 st.write(f'Хранятся до {MAX_RESULTS} последних сохранённых прогнозов на этом ПК. Параметры брони и идентификаторы гостей не записываются. История общая для пользователей этой копии приложения.')
 if st.checkbox('Подтверждаю очистку всей истории',key='confirm_history_clear'):
  if st.button('Очистить историю'):
   try:
    clear_results(history_path)
    st.success('История очищена.')
   except (OSError,sqlite3.Error):st.warning('Не удалось очистить историю. Проверьте доступ к папке приложения.')
 try:
  saved=read_results(history_path)
 except (OSError,sqlite3.Error):
  st.warning('История недоступна. Расчёт риска остаётся доступен.')
 else:
  if saved:
   frame=pd.DataFrame(saved).rename(columns={'checked_at':'Время UTC','probability':'Вероятность','risk':'Риск','model_id':'Модель','threshold':'Порог'})
   st.dataframe(frame,hide_index=True)
  else: st.info('Сохранённых результатов пока нет.')

