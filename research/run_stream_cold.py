"""One unchanged-harness solution cold screen with installed streaming candidate."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'harness'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    from common import OFFLINE_ENV,run_logged
    from run_task import make_config,soup_outputs
    assert not out.exists();out.mkdir()
    config=make_config(out,ROOT/'data/train.jsonl',1)
    command=[sys.executable,str(ROOT/'research/census_stream_child.py'),'--config',str(config),'--metrics',str(out/'metrics.json'),'--solution']
    r=run_logged(command,out/'train.log',600,dict(OFFLINE_ENV,SOUP_DB_PATH=str(out/'experiments.db'),
        TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),STREAM_CENSUS_FILE=str(out/'CENSUS.json')))
    report={'source':'candidate-streaming','external':dict(zip(('exit_code','wall_s','peak_mib','gpu_start_mib'),r)),
        'metrics':json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else {},
        **soup_outputs(out/'adapter'),'limits':'ONE solution-only Brevscreen, notABBA/check_rules/Kaggle/admission.'}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'external':report['external'],'steps':report['metrics'].get('steps')},indent=2));raise SystemExit(r[0])
