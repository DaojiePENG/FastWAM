"""Run technical repetitions of serial, pipelined, and isolated-stage capacity.

Launch with the target host's configured Python environment; use local client,
image and output paths during measurement, then copy only completed small JSON
and log files to --results-dir. One process amortizes dependency imports.
Every condition has its own warmup and raw per-request records. No robot is
actuated. Example: python benchmark_matrix.py --primary primary.png --wrist
wrist.png --local-output /tmp/bench --results-dir results --cloud http://cloud:8600
"""
import argparse, contextlib, hashlib, importlib, io, json, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--client-dir',default=str(Path(__file__).resolve().parent))
p.add_argument('--primary',required=True)
p.add_argument('--wrist',required=True)
p.add_argument('--local-output',required=True)
p.add_argument('--results-dir',required=True)
p.add_argument('--cloud',default='http://127.0.0.1:8600')
p.add_argument('--edge',default='http://127.0.0.1:8601')
p.add_argument('--n',type=int,default=50)
p.add_argument('--warmup',type=int,default=5)
p.add_argument('--raw-n',type=int,default=20)
p.add_argument('--raw-warmup',type=int,default=2)
p.add_argument('--edge-n',type=int,default=20)
p.add_argument('--repetitions',type=int,default=3)
p.add_argument('--conditions',default='serial_codec,serial_raw,pipeline_codec,edge_codec,cloud_codec')
a=p.parse_args()
client_dir=Path(a.client_dir); local=Path(a.local_output); output=Path(a.results_dir)
local.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(client_dir))
import bench_client
conditions=a.conditions.split(',')
known={'serial_codec':('full','serial',False),'serial_raw':('full','serial',True),
       'pipeline_codec':('full','pipeline',False),'edge_codec':('edge','serial',False),'cloud_codec':('cloud','serial',False)}
assert all(c in known for c in conditions)
manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
          'n_default':a.n,'warmup_default':a.warmup,'raw_n':a.raw_n,'raw_warmup':a.raw_warmup,'edge_n':a.edge_n,'repetitions':a.repetitions,'conditions':conditions,
          'replication_unit':'Technical repeated inference requests on the same hosts with fixed replayed 224-pixel observations; no physical actuation',
          'client_software':{'python':sys.version,'bench_client_sha256':hashlib.sha256((client_dir/'bench_client.py').read_bytes()).hexdigest(),
                             'protocol_sha256':hashlib.sha256((client_dir/'protocol.py').read_bytes()).hexdigest()},
          'cloud':a.cloud,'edge':a.edge,'runs':[]}
manifest_path=output/'model_benchmark_manifest.json'
def write_manifest():
    tmp=manifest_path.with_suffix('.partial')
    tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(manifest_path)
class Tee:
    def __init__(self, target, log): self.target=target;self.log=log
    def write(self,text): self.target.write(text);self.log.write(text);return len(text)
    def flush(self): self.target.flush();self.log.flush()
failed=False
for repetition in range(1,a.repetitions+1):
    # Rotate the five conditions between repetitions to spread temporal drift.
    shift=(2*(repetition-1))%len(conditions)
    ordered=conditions[shift:]+conditions[:shift]
    for name in ordered:
        stage,schedule,raw=known[name]
        run_n=a.raw_n if raw else a.edge_n if stage=='edge' else a.n
        run_warmup=a.raw_warmup if raw else a.warmup
        mode='raw' if raw else 'codec'
        label=schedule if stage=='full' else stage
        tag=f'hardware_{mode}_{label}_r{repetition}'
        path=local/f'{tag}.json';logpath=local/f'{tag}.log'
        command=['--cloud',a.cloud,'--edge',a.edge,'--stage',stage,'--schedule',schedule,
                 '--n',str(run_n),'--warmup',str(run_warmup),'--ping-rounds','10','--timeout','600',
                 '--primary',a.primary,'--wrist',a.wrist,'--seed','0','--action-horizon','32',
                 '--executed-steps','10','--inference-steps','10','--tag',tag,'--out',str(path)]
        if raw:command.append('--raw')
        record={'name':tag,'n':run_n,'warmup':run_warmup,'args':command,'started_utc':datetime.now(timezone.utc).isoformat()}
        manifest['runs'].append(record);write_manifest()
        print('MODEL_START',tag,record['started_utc'],flush=True)
        start=time.perf_counter()
        with logpath.open('w') as logfile, contextlib.redirect_stdout(Tee(sys.stdout,logfile)):
            code=bench_client.main(command)
        record.update(returncode=code,elapsed_s=time.perf_counter()-start,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        for source in (path,logpath):
            target=output/source.name
            temporary=target.with_suffix(target.suffix+'.partial')
            shutil.copy2(source,temporary);temporary.replace(target)
        if path.exists():
            data=json.loads(path.read_text());record['throughput']=data.get('throughput');record['status']=data.get('status')
        write_manifest()
        print('MODEL_RUN_COMPLETE',tag,json.dumps(record.get('throughput')),flush=True)
        if code:
            failed=True;break
    if failed:break
manifest.update(status='failed' if failed else 'ok',completed_utc=datetime.now(timezone.utc).isoformat())
write_manifest()
print('MODEL_MATRIX_COMPLETE',manifest['status'],flush=True)
raise SystemExit(int(failed))
