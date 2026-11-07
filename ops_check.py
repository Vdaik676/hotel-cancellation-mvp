"""Verify the installed release without starting a server or using the network."""
import argparse, time, statistics, sys, json
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from src.contract import ROOT, load_json, predict_one

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,help='Optional evidence output path')
    args=parser.parse_args()
    bundle=joblib.load(ROOT/'artifacts/model.joblib')
    schema=load_json(ROOT/'artifacts/schema.json')
    examples=load_json(ROOT/'demo/examples.json')
    durations=[]
    rows=[]
    with threadpool_limits(limits=2):
        predict_one(examples[0]['input'],bundle,schema)
        for entry in examples:
            result=predict_one(entry['input'],bundle,schema)
            expected=entry['expected']
            if result['risk']!=expected['risk'] or abs(result['probability']-expected['probability'])>1e-10:
                raise RuntimeError(f"Unexpected output for {entry['name']}")
            rows.append({'name':entry['name'],**result})
        cases=load_json(ROOT/'demo/acceptance_inputs.json')
        if len(cases)!=20:
            raise RuntimeError('Expected 20 acceptance inputs')
        for entry in cases:
            start=time.perf_counter()
            predict_one(entry,bundle,schema)
            durations.append((time.perf_counter()-start)*1000)
    p95=statistics.quantiles(durations,n=100,method='inclusive')[94]
    report={'status':'ok' if p95<=3000 else 'fail','python':sys.version.split()[0],
            'examples':rows,'valid_inputs':len(cases),'warm_inference_p95_ms':p95,
            'measurement':'local inference only, 20 calls, excludes browser and startup'}
    output=json.dumps(report,ensure_ascii=False,indent=2)
    print(output)
    if args.json: args.json.write_text(output+'\n',encoding='utf-8')
    return 0 if report['status']=='ok' else 1

if __name__=='__main__': raise SystemExit(main())
