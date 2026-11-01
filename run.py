#!/usr/bin/env python3
"""Verify or reanalyze the minimal report package. No command starts a model job."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tarfile
ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def safe_rel(s):
    p=Path(s)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe package path')
    return p
def manifest():return json.loads((ROOT/'manifest.json').read_text())
def verify():
    x=manifest();seen=set();count=0
    for r in x['files']+x['containers']+x.get('package_files',[]):
        if 'member' in r or r['file'] in seen:continue
        seen.add(r['file']);p=ROOT/safe_rel(r['file'])
        if not p.is_file() or p.is_symlink() or sha(p)!=r['sha256']:raise ValueError('Integrity failure: '+r['file'])
        count+=1
    for c in x['containers']:
        records={r['member']:r for r in x['files'] if r['file']==c['file'] and 'member' in r}
        with tarfile.open(ROOT/c['file']) as t:
            if set(t.getnames())!=set(records):raise ValueError('Container member grid differs')
            for m in t:
                if not m.isfile():raise ValueError('Unexpected non-file member')
                if hashlib.sha256(t.extractfile(m).read()).hexdigest()!=records[m.name]['sha256']:raise ValueError('Observation hash differs')
    print(json.dumps({'status':'PASS','packaged_files':count,'scientific_artifacts':sum('restore_path' in r for r in x['files']),
                      'raw_graphs':sum(r.get('restore_path','').endswith('.pt') for r in x['files'])}),flush=True)

def prepare(work,graphs=False):
    work=work.resolve()
    if work==ROOT or ROOT in work.parents:raise ValueError('Use a working directory outside the delivered package')
    marker=work/'.minimal-package-work.json';digest=sha(ROOT/'manifest.json')
    if work.exists() and any(work.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text())['manifest_sha256']!=digest:raise ValueError('Work directory must be empty or prepared from this exact package')
        state=json.loads(marker.read_text())
        if graphs and not state['raw_graphs_materialized']:
            for r in manifest()['files']:
                if not r.get('restore_path','').endswith('.pt'):continue
                origin=ROOT/r['file']
                if sha(origin)!=r['sha256']:raise ValueError('Graph hash differs')
                target=work/safe_rel(r['restore_path']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
            state['raw_graphs_materialized']=True;marker.write_text(json.dumps(state,indent=2)+'\n')
        return work
    work.mkdir(parents=True,exist_ok=True);x=manifest()
    tar_records={}
    for r in x['files']:
        if 'restore_path' not in r:continue
        rel=safe_rel(r['restore_path'])
        if str(rel).endswith('.pt') and not graphs:continue
        dest=work/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if 'member' in r:
            tar_records.setdefault(r['file'],[]).append(r);continue
        origin=ROOT/r['file']
        if sha(origin)!=r['sha256']:raise ValueError('Source hash differs: '+r['file'])
        shutil.copy2(origin,dest)
    for name,records in tar_records.items():
        with tarfile.open(ROOT/name) as t:
            for r in records:
                data=t.extractfile(r['member']).read()
                if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Observation hash differs')
                dest=work/safe_rel(r['restore_path']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                if '/private/' in r['restore_path']:os.chmod(dest,0o600)
    # Private file mode is part of the frozen numerical contracts, even for text-free tensors.
    for p in work.rglob('*'):
        if 'private' in p.relative_to(work).parts:os.chmod(p,0o700 if p.is_dir() else 0o600)
    (work/'reproduction').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'setup/reproduce_summary.py',work/'reproduction/reproduce_summary.py')
    shutil.copy2(ROOT/'setup/crosscheck_record_order.json',work/'reproduction/crosscheck_record_order.json')
    marker.write_text(json.dumps({'manifest_sha256':digest,'raw_graphs_materialized':graphs},indent=2)+'\n')
    print('Prepared selected files in '+str(work),flush=True)
    return work

def call(command,work):
    subprocess.run(command,cwd=work,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
def figures(work):
    import importlib.util
    call([sys.executable, '-B', str(work/'reproduction/build_route_figure.py')], work)
    spec = importlib.util.spec_from_file_location('selected_figures', work/'figure-sources/build_figures.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dest = work/'submission/figures'; dest.mkdir(parents=True, exist_ok=True)
    for fn in (module.figure4, module.figure6, module.figure7): fn(dest)
    print('Rebuilt selected figures 3, 4, 6, 7 from the saved inputs.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','prepare','cpu','figures','severity'])
    p.add_argument('--work',type=Path);p.add_argument('--include-graphs',action='store_true')
    a=p.parse_args()
    if a.command=='verify':return verify()
    if a.work is None:p.error('--work is required')
    work=prepare(a.work,a.include_graphs)
    if a.command=='cpu':call([sys.executable,'-B',str(work/'reproduction/reproduce_summary.py'),'cpu','--output',str(work/'results')],work)
    elif a.command=='figures':figures(work)
    elif a.command=='severity':call([sys.executable,'-B',str(work/'reproduction/reproduce_summary.py'),'severity','--output',str(work/'severity-results')],work)
if __name__=='__main__':main()
