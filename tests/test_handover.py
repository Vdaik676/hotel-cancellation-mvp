import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
import src.contract as contract

@pytest.mark.parametrize('content',[None,'{ broken','[{},{}]'])
def test_optional_examples_fail_safely(monkeypatch,tmp_path,content):
    root=contract.ROOT
    for name in ('artifacts','reports','data','demo'):
        (tmp_path/name).mkdir()
    for name in ('artifacts/model.joblib','artifacts/schema.json','reports/metrics.json','data/source.json'):
        (tmp_path/name).write_bytes((root/name).read_bytes())
    if content is not None:
        (tmp_path/'demo/examples.json').write_text(content,encoding='utf-8')
    monkeypatch.setattr(contract,'ROOT',tmp_path)
    app=AppTest.from_file(str(root/'streamlit_app.py'),default_timeout=30).run()
    assert not app.exception
    assert any('заполнить форму вручную' in x.value for x in app.warning)
    next(b for b in app.button if b.label=='Оценить риск').click().run()
    assert not app.exception and len(app.metric)==2

def test_field_codes_have_help():
    app=AppTest.from_file(str(contract.ROOT/'streamlit_app.py'),default_timeout=30).run()
    assert not app.exception
    for key,word in [('meal','завтрак'),('market_segment','туроператор'),('reserved_room_type','условные коды')]:
        assert word in app.selectbox(key=key).proto.help
        assert any(word in item.value for item in app.markdown)
