#!/usr/bin/env python3
"""Selected saved-data replay, adapted from the original bundle's driver. No model jobs."""
from pathlib import Path
import argparse, contextlib, glob, importlib, io, json, math, runpy, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = ROOT / 'ALL WORK/steps'
def step(n): return next(STEPS.glob(f'{n:02d}-*'))
def read(p): return json.loads(Path(p).read_text())
def module(n, name):
    sys.path.insert(0, str(step(n) / 'src'))
    return importlib.import_module(name)
def stripped(x):
    if isinstance(x, dict): return {k: stripped(v) for k, v in x.items() if k not in {'created_at_utc','inputs','input_file_sha256','payload_sha256','output_sha256','command'}}
    if isinstance(x, list): return [stripped(v) for v in x]
    return x
def compare(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f'keys differ: {path}'
        for k in a: compare(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f'length differs: {path}'
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + f'/{i}')
    elif isinstance(a, float) and isinstance(b, (int, float)):
        assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10), f'numerical mismatch: {path}'
    else: assert a == b, f'value mismatch: {path}'

def cpu(out):
    out.mkdir(parents=True, exist_ok=True); results = {}
    def record(name, **data):
        results[name] = {'status': 'PASS', **data}
        (out / 'cpu_reproduction.json').write_text(json.dumps(results, indent=2) + '\n')
        print(name + ': PASS', flush=True)
    s9 = step(9) / 'outputs'; s16 = step(16)
    # This development plan/result pair has no redacted fields and retains its original authentication.
    m = module(9, 'analyze_s9_e7c')
    new = m.analyze(read(s9/'s9_e7b_plan.json'), read(s9/'s9_e7c_result.json'))
    compare(stripped(new), stripped(read(s9/'s9_e7c_analysis.json')))
    record('route_development', scientific_fields_match=True)
    bc = s16/'outputs/phase_c_local'
    old = read(bc/'s16_phase_c_tier1_result.json'); m = module(16, 'analyze_s16_phase_c')
    with contextlib.redirect_stdout(io.StringIO()): m.main()
    new = read(bc/'s16_phase_c_tier1_result.json'); compare(new, old)
    assert new['payload_sha256'] == 'cda74c73ea5840f9ead48da64f074ff8dc3e3e96e4b181f102289c533c423ae6'
    record('behavioral_primary', generation_records=1020, payload_sha256=new['payload_sha256'])
    old = read(bc/'s16_phase_c_tier1_crosscheck.json')
    order = read(ROOT/'reproduction/crosscheck_record_order.json')['record_filenames']; original_glob = glob.glob
    def ordered_glob(pattern, *args, **kwargs):
        if str(pattern) == str(bc/'records/*.json'): return [str(bc/'records'/name) for name in order]
        return original_glob(pattern, *args, **kwargs)
    glob.glob = ordered_glob
    try:
        with contextlib.redirect_stdout(io.StringIO()): runpy.run_path(str(s16/'src/crosscheck_tier1_estimates.py'), run_name='__main__')
    finally: glob.glob = original_glob
    compare(read(bc/'s16_phase_c_tier1_crosscheck.json'), old)
    record('independent_behavioral_crosscheck', complete_analysis_matches=True)

def severity(out):
    s15, s9 = step(15), step(9)/'outputs'; out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(s15/'src/w3_gate_severity.py'), '--e9-analysis', str(s9/'s9_e9c_analysis.json'), '--e4c-analysis', str(s9/'s9_e4c_analysis.json'), '--e8b-analysis', str(s9/'s9_e8b_analysis.json'), '--config', str(s15/'inputs/s15_frozen_config.json'), '--output', str(out/'w3_analysis.json'), '--curves-csv-output', str(out/'w3_severity_curves.csv'), '--future-csv-output', str(out/'w3_future_gate_candidates.csv'), '--report-output', str(out/'w3_report.md'), '--batch-size', '2048'], check=True)
    compare(stripped(read(out/'w3_analysis.json')), stripped(read(s15/'outputs/w3_analysis.json')))
    print('Severity scientific fields match the frozen record.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['cpu','severity']); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    (cpu if a.command == 'cpu' else severity)(a.output.resolve())
