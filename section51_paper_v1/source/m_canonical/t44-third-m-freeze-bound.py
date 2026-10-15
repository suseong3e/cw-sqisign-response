#!/usr/bin/env python3
"""Freeze the target-only third-key M branch before CRT analysis or KAT."""
import hashlib,json,os
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np
root=Path(os.environ['SECTION51_SOURCE_ROOT'])
e=Path(os.environ['SECTION51_M_OUT'])
p=SourceFileLoader('product',str(root/'m_canonical/t05-m-product-groups.py')).load_module()
screen_path=Path(os.environ['SECTION51_M_OUT'])/'t41-third-m-branch-screen.json'
screen=json.loads(screen_path.read_text())
branch_path=Path(os.environ['SECTION51_M_OUT'])/'t40-third-m-full-product-branch.json'
branch=json.loads(branch_path.read_text())
admissible=[r for r in screen['results'] if r['determinant_odd'] and r['m01_product_min_validation_abs_r']>0.5 and r['m00_add_min_validation_abs_r']>0.5]
assert len(admissible)==1
selected=admissible[0]
selected_branch=next(r for r in branch['results'] if r['m01_source']==selected['branch'])
assert selected_branch['split_agreement']
assert selected_branch['m01_candidate']==selected['matrix']['m01']
assert selected_branch['m00_candidates']['discovery']==selected['matrix']['m00']
assert selected_branch['m00_candidates']['validation']==selected['matrix']['m00']
m={k:int(v,16) for k,v in selected['matrix'].items()}
assert ((m['m00']*m['m11']-m['m01']*m['m10'])&1)==1
data=Path(os.environ['SECTION51_M_DATA']);manifest=json.loads((data/'manifest.json').read_text())
assert not manifest['target_truth_loaded'] and not manifest['known_key_trace_count']
assert p.rank.digest(data/'traces.npy')==manifest['target_trace_sha256']
assert p.rank.digest(data/'public_schedule.jsonl')==manifest['target_public_schedule_sha256']
rows=[json.loads(line) for line in (data/'public_schedule.jsonl').read_text().splitlines()]
traces=np.load(data/'traces.npy',mmap_mode='r',allow_pickle=False)
challenges=p.rank.words(rows)
ix=np.array([i for i,r in enumerate(rows) if r['block']%2==1])
subset=[rows[i] for i in ix]
points={'m01':[10440+52*j for j in range(6)],'m00':[11432+84*j for j in range(6)],
        'm11':[22652+52*j for j in range(6)],'m10':json.loads((e/'t08-m10-groups.json').read_text())['groups'][0]['points']}
def event_labels(name):
    if name=='m01': return p.rank.product_labels(m['m01'],challenges[ix],p.limb)[0].T
    if name=='m11': return p.rank.product_labels(m['m11'],challenges[ix],p.limb)[0].T
    if name=='m00':
        _,state=p.rank.product_labels(m['m01'],challenges[ix],p.limb)
        return p.rank.add_labels(m['m00'],state,p.limb)[0]
    _,state=p.rank.product_labels(m['m11'],challenges[ix],p.limb)
    return p.rank.add_labels(m['m10'],state,p.limb)[0]
def fixed(label,samples):
    wave=np.asarray(traces[np.ix_(ix,samples)],float)
    return np.diag(p.CONTROL.grid(p.residual(label,subset),p.residual(wave,subset)))
def artifact_name(path):
    if path.is_relative_to(root): return f"source/{path.relative_to(root)}"
    if path.is_relative_to(e): return f"m-output/{path.relative_to(e)}"
    if path.is_relative_to(data): return f"m-input/{path.relative_to(data)}"
    raise ValueError(f"unbound artifact path: {path}")
rng=np.random.default_rng(2026100701);controls=[]
for name,poi in points.items():
    label=event_labels(name)
    corr=fixed(label,poi)
    wrong=fixed(label,[36864+84*j for j in range(6)])
    wave=p.residual(np.asarray(traces[np.ix_(ix,poi)],float),subset)
    observed=min(abs(v) for v in corr)
    exceed=0;blocks=np.array([r['block'] for r in subset])
    for _ in range(99):
        perm=np.arange(len(ix))
        for block in np.unique(blocks):
            within=np.flatnonzero(blocks==block)
            perm[within]=rng.permutation(within)
        shuffled=p.residual(label[perm],subset)
        value=np.diag(p.CONTROL.grid(shuffled,wave))
        exceed+=min(abs(v) for v in value)>=observed
    controls.append({'event':name,'validation_correlations':corr.tolist(),
                     'min_validation_abs_r':observed,
                     'wrong_window_max_abs_r':float(max(abs(v) for v in wrong)),
                     'block_shuffle_p':(exceed+1)/100})
    print(name,round(observed,3),controls[-1]['wrong_window_max_abs_r'],controls[-1]['block_shuffle_p'],flush=True)
report={'schema':'r3-target-only-m-candidates-v1','status':'one_candidate',
        'matrix':{'m00':hex(m['m00']),'m01':hex(m['m01']),'m11':hex(m['m11']),'m10_candidates':[hex(m['m10'])]},
        'source_order_pois':points,'controls':controls,
        'selection':{'branch_screen_sha256':hashlib.sha256(screen_path.read_bytes()).hexdigest(),
                     'rule':'predefined odd-determinant condition and both full-width M01 product and M00 addition validation minima above 0.5',
                     'selected_branch':selected['branch'],
                     'm01_discovery_validation_candidates_disagree':branch['results'][0]['m01_candidate']!=branch['results'][1]['m01_candidate'],
                     'm00_selected_branch_split_agreement':selected_branch['split_agreement'],
                     'validation_reused_for_branch_selection':True},
        'inputs':{'target_trace_sha256':manifest['target_trace_sha256'],
                  'public_schedule_sha256':manifest['target_public_schedule_sha256'],
                  'source_artifacts_sha256':{artifact_name(path):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (root/'m_canonical/t05-m-product-groups.py',e/'t05-m-product-groups.json',e/'t08-m10-groups.json',
                                   branch_path,screen_path,data/'manifest.json')},
                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'evidence_contract':{'evidence_origin':'measured','evidence_purpose':'diagnostic',
                             'blinding':'unblinded','measurement_scope':'isolated_primitive',
                             'target_truth_loaded':False,'known_key_traces_loaded':False,
                             'claim_scope':'r3_single_split:target_only_M_candidate_screen_for_CRT_method_development',
                             'claim_not_allowed':['full_key_recovery','scheme_break']}}
out=Path(os.environ['SECTION51_M_OUT'])/'t44-third-m-candidates-bound.json'
with out.open('x') as output:json.dump(report,output,indent=2);output.write('\n')
os.chmod(out,0o600)
print('frozen',out)
