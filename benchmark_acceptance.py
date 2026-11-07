"""Record all NFR-02 conditions and 20 measurements after explicit warm-up."""
import ctypes, importlib.metadata, json, math, os, platform, time
import joblib
from threadpoolctl import threadpool_limits
from src.contract import ROOT, load_json, save_json, predict_one

def ram_bytes():
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(k,ctypes.c_uint64) for k in ('total','free','page_total','page_free','virtual_total','virtual_free','extended_free')]
        value=Memory(); value.length=ctypes.sizeof(value)
        return int(value.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) else None
    try:
        return os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (AttributeError,OSError,ValueError):
        return None

def main():
    model=joblib.load(ROOT/'artifacts/model.joblib')
    schema=load_json(ROOT/'artifacts/schema.json')
    inputs=load_json(ROOT/'demo/acceptance_inputs.json')
    assert len(inputs)==20
    elapsed=[]
    with threadpool_limits(limits=2):
        start=time.perf_counter()
        for _ in range(5): predict_one(inputs[0],model,schema)
        warmup=time.perf_counter()-start
        for entry in inputs:
            start=time.perf_counter()
            predict_one(entry,model,schema)
            elapsed.append(time.perf_counter()-start)
    p95=sorted(elapsed)[math.ceil(len(elapsed)*.95)-1]
    report={'python':platform.python_version(),'os':platform.platform(),
            'processor':os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
            'ram_bytes':ram_bytes(),'versions':{k:importlib.metadata.version(k) for k in ('scikit-learn','numpy','scipy','pandas','joblib','streamlit')},
            'model_id':model['model_id'],'warmup_calls':5,'warmup_seconds':warmup,'samples_seconds':elapsed,
            'p95_method':'nearest rank','p95_seconds':p95,'limit_seconds':3,
            'result':'pass' if p95<=3 else 'fail','scope':'validation of inputs and local model computation only; excludes UI and history writes'}
    save_json(ROOT/'reports/latency_AT05_v1_2.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if p95<=3 else 1

if __name__=='__main__': raise SystemExit(main())
