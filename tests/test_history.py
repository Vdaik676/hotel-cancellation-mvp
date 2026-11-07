from streamlit.testing.v1 import AppTest
import src.contract as contract
from src.history import save_result, read_results, clear_results

def test_result_storage_is_durable_minimal_and_clearable(tmp_path):
    file=tmp_path/'runtime/history.sqlite3'
    save_result(file,{'probability':.42,'risk':'Высокий','model_id':'E2','threshold':.29,'guest':'not retained'})
    rows=read_results(file)
    assert len(rows)==1
    assert set(rows[0])=={'checked_at','probability','risk','model_id','threshold'}
    assert rows[0]['probability']==.42
    clear_results(file)
    assert read_results(file)==[]

def test_history_only_saves_opted_in_success(monkeypatch,tmp_path):
    root=contract.ROOT
    for name in ('artifacts','reports','data','demo'):
        (tmp_path/name).mkdir()
    for name in ('artifacts/model.joblib','artifacts/schema.json','reports/metrics.json','data/source.json','demo/examples.json'):
        (tmp_path/name).write_bytes((root/name).read_bytes())
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    def submit(app):
        return next(b for b in app.button if b.label=='Оценить риск').click().run()
    app=AppTest.from_file(str(root/'streamlit_app.py'),default_timeout=30).run()
    file=tmp_path/'runtime/risk_history.sqlite3'
    submit(app)
    assert not file.exists()
    app.checkbox(key='keep_result').set_value(True)
    submit(app)
    assert not app.exception and len(read_results(file))==1
    app.run()
    assert len(read_results(file))==1
    for key in ('adults','children','babies'):
        app.number_input(key=key).set_value(0)
    submit(app)
    assert app.error and len(read_results(file))==1
