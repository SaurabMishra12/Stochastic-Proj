"""Execute the validated full local workflow and regenerate both reports."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run(args,cwd=ROOT,check=True,env={**os.environ,'MPLCONFIGDIR':'/tmp/stochastic-pll-mpl'})

def finalize():
    meta_path=ROOT/'results/run_metadata/full_run.json';meta=json.loads(meta_path.read_text());syn=json.loads((ROOT/'results/run_metadata/synthetic_run.json').read_text())
    if not syn.get('complete') or syn.get('completed_cells')!=144:
        raise RuntimeError('All144 synthetic cells must finish before reporting completion')
    meta.update(status='complete',synthetic_conditions=144,synthetic_trajectories=72000,synthetic_network_condition_runs=72000,synthetic_independent_matched_blocks=1500,synthetic_status='complete',synthetic_wall_seconds=syn.get('wall_seconds',syn.get('elapsed_seconds')))
    # The complete suite is run again after every experiment family is present.
    check=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml=results/run_metadata/final_tests.xml'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'results/run_metadata/final_tests.txt').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('Final tests failed; inspect final_tests.txt')
    import xml.etree.ElementTree as ET
    suites=ET.parse(ROOT/'results/run_metadata/final_tests.xml').getroot();meta['tests_passed']=sum(int(x.attrib.get('tests',0))-int(x.attrib.get('failures',0))-int(x.attrib.get('errors',0)) for x in suites.findall('testsuite'))
    meta_path.write_text(json.dumps(meta,indent=2))
    captions=ROOT/'results/figures/captions.json';merged=json.loads(captions.read_text());other=ROOT/'results/figures/synthetic_captions.json'
    if other.exists():merged.update(json.loads(other.read_text()))
    captions.write_text(json.dumps(merged,indent=2))
    manifest={}
    for p in sorted((ROOT/'results/raw_data').rglob('*')):
        if p.is_file():manifest[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT/'results/run_metadata/pll_raw_integrity.json').write_text(json.dumps(manifest,indent=2))
    return meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports-only',action='store_true');args=parser.parse_args()
    if not args.reports_only:
        run(sys.executable,'-m','pytest','-q');run(sys.executable,'-m','src.experiment_runner','--workers','2');run(sys.executable,'-m','src.analysis_supplement');run(sys.executable,'-m','src.visualization');run(sys.executable,'-m','src.synthetic_agents','--full')
    finalize();run(sys.executable,'-m','src.verify_outputs');run(sys.executable,'-m','src.deterministic_validation');run(sys.executable,'-m','src.latex_report');run(sys.executable,'-m','src.report_generator')
    with (ROOT/'results/run_metadata/latex_build.log').open('w') as log:
        subprocess.run(['tectonic','--keep-logs','report/main.tex'],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    print('Reports regenerated from completed full local experiments.')
if __name__=='__main__':main()
