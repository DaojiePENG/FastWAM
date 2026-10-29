"""Summarize finalized read-only resource monitors within recorded benchmark windows."""
from __future__ import annotations
import argparse,csv,json,re,statistics
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path


def moment(s):
    return datetime.fromisoformat(s.replace('Z','+00:00'))


def describe(values):
    if not values:return None
    values=sorted(values)
    def q(p):
        pos=(len(values)-1)*p;i=int(pos)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(pos-i)
    return {'n':len(values),'min':values[0],'max':values[-1],'mean':statistics.mean(values),'p50':q(.5),'p95':q(.95)}


def parse_tegra(line):
    fields={}
    for name,pat in [('ram_used_mb',r'RAM (\d+)/'),('ram_total_mb',r'RAM \d+/(\d+)MB'),('swap_used_mb',r'SWAP (\d+)/'),('swap_total_mb',r'SWAP \d+/(\d+)MB'),('gr3d_util_percent',r'GR3D_FREQ (\d+)%')]:
        m=re.search(pat,line)
        if m:fields[name]=float(m[1])
    m=re.search(r'GR3D_FREQ \d+%@(?:\[([^]]+)\]|(\d+))',line)
    if m:
        freq=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',m[1] or m[2])]
        if freq:fields['gr3d_clock_mhz_max']=max(freq)
    for name,value in re.findall(r'([A-Za-z0-9_]+)@([\d.]+)C',line):
        fields['temperature_'+name+'_c']=float(value)
    for name,now,average in re.findall(r'([A-Z][A-Z0-9_]+) (\d+)mW/(\d+)mW',line):
        fields['power_'+name+'_mw']=float(now)
    cpu=re.search(r'CPU \[([^]]+)\]',line)
    if cpu:
        freqs=[float(x) for x in re.findall(r'%@(\d+)',cpu[1])]
        util=[float(x) for x in re.findall(r'(\d+)%@',cpu[1])]
        if freqs:fields['cpu_active_clock_mhz_max']=max(freqs)
        if util:fields['cpu_core_util_percent_mean']=statistics.mean(util)
    return fields


def parse_gpu(stdout):
    out=[]
    for row in csv.reader(stdout.splitlines(),skipinitialspace=True):
        if len(row)!=8:continue
        index,uuid,util,used,total,power,pstate,temp=[x.strip() for x in row]
        metrics={}
        for key,value in [('gpu_util_percent',util),('memory_used_mib',used),('memory_total_mib',total),('power_w',power),('temperature_c',temp)]:
            try:metrics[key]=float(value)
            except ValueError:pass
        out.append((index,uuid,pstate,metrics))
    return out


def accumulate(items):
    metrics=defaultdict(list);pstates=defaultdict(int)
    for item in items:
        for k,v in item['metrics'].items():metrics[k].append(v)
        if 'pstate' in item:pstates[item['pstate']]+=1
    return {'sample_count':len(items),'metrics':{k:describe(v) for k,v in sorted(metrics.items())},'pstates':dict(pstates)}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--tegra',type=Path,required=True);p.add_argument('--gpu',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    windows=[]
    for f in sorted(a.directory.glob('hardware_*_r[123].json')):
        d=json.loads(f.read_text())
        if d.get('schema_version')!=2 or d.get('status') not in ('ok','measured_with_failures'):continue
        records=[r for r in d['records'] if not r['warmup']]
        if not records:continue
        starts=[moment(r['started_utc']) for r in records]
        ends=[moment(r['started_utc'])+timedelta(seconds=r['completed_perf']-r['started_perf']) for r in records]
        windows.append({'file':f.name,'stage':d['stage'],'mode':d['mode'],'schedule':d['schedule'],'start':min(starts),'end':max(ends),'attempted':len(records),'failed':sum(not r['ok'] for r in records)})
    sources={};warnings=[]
    for kind,path in [('jetson',a.tegra),('aihub',a.gpu)]:
        if not path.is_file():raise SystemExit(f'missing monitor {path}')
        data=[];bad=0
        for line in path.read_text().splitlines():
            try:
                r=json.loads(line);ts=moment(r['timestamp_utc'])
                if kind=='jetson':data.append({'ts':ts,'metrics':parse_tegra(r['tegrastats'])})
                else:
                    if r.get('returncode')!=0:bad+=1;continue
                    for index,uuid,pstate,metrics in parse_gpu(r['stdout']):data.append({'ts':ts,'metrics':metrics,'index':index,'uuid':uuid,'pstate':pstate,'compute_apps':r.get('compute_apps')})
            except (ValueError,KeyError,TypeError):bad+=1
        sources[kind]={'file':str(path),'parse_or_query_errors':bad,'samples':data}
    runs=[]
    for w in windows:
        run={k:(v.isoformat() if isinstance(v,datetime) else v) for k,v in w.items()}
        t=[r for r in sources['jetson']['samples'] if w['start']<=r['ts']<=w['end']]
        g=[r for r in sources['aihub']['samples'] if w['start']<=r['ts']<=w['end']]
        run['jetson']=accumulate(t);run['aihub_by_gpu']={idx:accumulate([r for r in g if r['index']==idx]) for idx in sorted({r['index'] for r in g})};runs.append(run)
    pooled={}
    for kind in ['jetson','aihub']:
        selected=[r for r in sources[kind]['samples'] if any(w['start']<=r['ts']<=w['end'] for w in windows)]
        pooled[kind]=accumulate(selected) if kind=='jetson' else {idx:accumulate([r for r in selected if r['index']==idx]) for idx in sorted({r['index'] for r in selected})}
    process_samples=defaultdict(list)
    process_seen=set()
    for r in sources['aihub']['samples']:
        if not any(w['start']<=r['ts']<=w['end'] for w in windows):continue
        apps=r.get('compute_apps')
        if not apps or apps.get('returncode')!=0 or r['ts'] in process_seen:continue
        process_seen.add(r['ts'])
        for row in csv.reader(apps.get('stdout','').splitlines(),skipinitialspace=True):
            if len(row)!=4:continue
            uuid,pid,process,used=[x.strip() for x in row]
            try:process_samples[(uuid,pid,process)].append(float(used))
            except ValueError:pass
    process_rows=[{'gpu_uuid':key[0],'pid':key[1],'process_name':key[2],'sampled_memory_mib':describe(values)} for key,values in sorted(process_samples.items())]
    metadata={}
    for kind,path in [('jetson',a.tegra),('aihub',a.gpu)]:
        mp=path.with_name('monitor_metadata.json')
        if mp.exists():metadata[kind]=json.loads(mp.read_text())
    result={'measurement':'1 Hz sampled resources in measured-operation windows only; warmup/setup excluded; descriptive sampling, not continuous tracing','utc_window_mapping':'Record started_utc + (completed_perf-started_perf) on Jetson client; 1Hz monitor host UTC; NTP synchronized but subsecond attribution approximate','run_count':len(runs),'sources':{k:{x:y for x,y in v.items() if x!='samples'} for k,v in sources.items()},'monitor_metadata':metadata,'pooled':pooled,'compute_process_snapshots':process_rows,'runs':runs,'interpretation_limits':['Temperatures and utilization do not establish throttling. No throttle-reason counters were sampled.','GR3D clock MHz is unavailable when tegrastats reports utilization only.','Jetson RAM is unified system memory, not isolated model GPU memory.','Cloud utilization/memory include existing processes; compute-app snapshots were sampled every 10s.','Power rails are as reported by tegrastats; do not sum overlapping rails or equate with wall power.','GPU temperature is absent when the gpu@ temperature token is not present; tj@ is a separate reported sensor.']}
    hardware=a.directory/'hardware_jetson.json'
    if hardware.exists():
        h=json.loads(hardware.read_text());result['jetson_configuration_before']={'nvpmodel':h.get('power_mode'),'cpu_governor':h.get('cpu_governor'),'source':hardware.name,'note':'Preflight configuration only; no per-sample NVPModel polling'}
    result['monitoring_unavailable_fields']=['throttle reasons/counters not sampled']
    jm=pooled.get('jetson',{}).get('metrics',{})
    if 'gr3d_clock_mhz_max' not in jm:result['monitoring_unavailable_fields'].append('GR3D clock MHz absent: GR3D_FREQ provided percentage only')
    if 'temperature_gpu_c' not in jm:result['monitoring_unavailable_fields'].append('GPU temperature gpu@ absent from all retained Jetson monitor lines')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'runs':len(runs),'jetson_samples':pooled.get('jetson',{}).get('sample_count'),'aihub_gpu_indices':list(pooled.get('aihub',{}))}))

if __name__=='__main__':main()
