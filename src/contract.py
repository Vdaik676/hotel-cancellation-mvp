"""One input contract shared by training, inference, UI and tests."""
from pathlib import Path
import json,math,numbers
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NUMERIC=['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies']
CATEGORICAL=['hotel','arrival_date_month','meal','market_segment','reserved_room_type']
FEATURES=NUMERIC+CATEGORICAL
TARGET='is_canceled'
MONTHS=['January','February','March','April','May','June','July','August','September','October','November','December']
LABELS={'lead_time':'Дней до заезда','stays_in_weekend_nights':'Ночей в выходные','stays_in_week_nights':'Ночей в будни','adults':'Взрослых','children':'Детей','babies':'Младенцев','hotel':'Тип отеля','arrival_date_month':'Месяц заезда','meal':'Питание','market_segment':'Сегмент бронирования','reserved_room_type':'Забронированный тип номера'}
LIMITS={'lead_time':(0,1000),'stays_in_weekend_nights':(0,365),'stays_in_week_nights':(0,365),'adults':(0,20),'children':(0,20),'babies':(0,20)}

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_record(record,schema):
 if not isinstance(record,dict):raise ValueError('Параметры бронирования должны быть словарём.')
 missing=set(FEATURES)-set(record);extra=set(record)-set(FEATURES)
 if missing:raise ValueError('Заполните поля: '+', '.join(LABELS[k] for k in sorted(missing)))
 if extra:raise ValueError('Неизвестные поля: '+', '.join(sorted(extra)))
 clean={}
 for key in NUMERIC:
  value=record[key]
  if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or not float(value).is_integer():
   raise ValueError(LABELS[key]+': требуется целое число.')
  low,high=LIMITS[key]
  if not low<=value<=high:raise ValueError(f'{LABELS[key]}: допустимо от {low} до {high}.')
  clean[key]=int(value)
 nights=clean['stays_in_weekend_nights']+clean['stays_in_week_nights']
 guests=clean['adults']+clean['children']+clean['babies']
 if not 1<=nights<=365:raise ValueError('Общая длительность проживания должна быть от 1 до 365 ночей.')
 if not 1<=guests<=20:raise ValueError('Общее число гостей должно быть от 1 до 20.')
 for key in CATEGORICAL:
  value=record[key]
  if not isinstance(value,str) or value not in schema['categories'][key]:raise ValueError(LABELS[key]+': выберите значение из списка.')
  clean[key]=value
 return clean

def risk_category(p,threshold):
 if not 0<=p<=1 or not 0<threshold<=1:raise ValueError('Неверная вероятность или порог.')
 return 'Низкий' if p<threshold/2 else ('Средний' if p<threshold else 'Высокий')

def predict_one(record,bundle,schema):
 clean=validate_record(record,schema)
 probability=float(bundle['pipeline'].predict_proba(pd.DataFrame([clean],columns=FEATURES))[0,1])
 if not math.isfinite(probability) or not 0<=probability<=1:raise ValueError('Модель вернула некорректный результат.')
 t=float(bundle['threshold'])
 return {'probability':probability,'risk':risk_category(probability,t),'threshold':t,'model_id':bundle['model_id']}
