"""Read-only 1 Hz inference resource monitor with host-local output.

Use --kind gpu for nvidia-smi (per-GPU metrics each second, process memory
roughly each 10 s), or --kind tegra for tegrastats. Each sample has UTC time.
Stop this Python process with SIGTERM; it terminates only the tegrastats child
it created. Copy the completed small logs after the measurement window.
"""
import argparse, json, os, select, signal, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['gpu','tegra'],required=True);p.add_argument('--out-dir',required=True)
a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
stopped=False

def stop(signum,frame):
    global stopped
    stopped=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
meta={'kind':a.kind,'pid':os.getpid(),'host':socket.gethostname(),'started_utc':datetime.now(timezone.utc).isoformat(),
      'period_ms':1000,'read_only':True,'storage':str(out),'status':'running'}
(out/'monitor_metadata.json').write_text(json.dumps(meta,indent=2))
(out/'monitor.pid').write_text(str(os.getpid())+'\n')
child=None
try:
    with (out/'samples.jsonl').open('a',buffering=1) as log:
        if a.kind=='gpu':
            index=0
            while not stopped:
                start=time.monotonic()
                cmd=['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,pstate,temperature.gpu',
                     '--format=csv,noheader,nounits']
                record={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':cmd}
                data=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
                record.update(returncode=data.returncode,stdout=data.stdout,stderr=data.stderr)
                if index%10==0:
                    apps=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']
                    proc=subprocess.run(apps,capture_output=True,text=True,timeout=10)
                    record['compute_apps']={'command':apps,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
                log.write(json.dumps(record)+'\n');index+=1
                while not stopped and time.monotonic()-start<1:
                    time.sleep(min(.2,max(.001,1-(time.monotonic()-start))))
        else:
            child=subprocess.Popen(['tegrastats','--interval','1000'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            meta['tegrastats_pid']=child.pid
            (out/'monitor_metadata.json').write_text(json.dumps(meta,indent=2))
            while not stopped and child.poll() is None:
                readable,_,_=select.select([child.stdout],[],[],.5)
                if readable:
                    line=child.stdout.readline()
                    if line:
                        log.write(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'tegrastats':line.rstrip()})+'\n')
except Exception as exc:
    meta['error']=f'{type(exc).__name__}: {exc}'
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired: child.kill();child.wait()
    meta.update(status='failed' if 'error' in meta else 'stopped',completed_utc=datetime.now(timezone.utc).isoformat())
    (out/'monitor_metadata.json').write_text(json.dumps(meta,indent=2))
