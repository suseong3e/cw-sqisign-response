#!/usr/bin/env python3
"""Re-evaluate M00 with both full-width M01 product branches and fixed source stores."""
import hashlib,json
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np
root=Path('/home/tntjd/cw-sqisign-response')
e=root/'runs/20261007T025549776051Z-r3-onekey-m-only/evidence'
product=SourceFileLoader('product',str(e/'t05-m-product-groups.py')).load_module()
add=SourceFileLoader('add',str(e/'t07-m00-groups.py')).load_module()
data=e/'inputs/m';manifest=json.load(open(data/'manifest.json'))
assert not manifest['target_truth_loaded'] and not manifest['known_key_trace_count']
assert product.rank.digest(data/'traces.npy')==manifest['target_trace_sha256']
assert product.rank.digest(data/'public_schedule.jsonl')==manifest['target_public_schedule_sha256']
rows=[json.loads(v) for v in (data/'public_schedule.jsonl').read_text().splitlines()]
traces=np.load(data/'traces.npy',mmap_mode='r',allow_pickle=False)
challenges=product.rank.words(rows)
group=next(g for g in json.load(open(e/'t05-m-product-groups.json'))['groups'] if g['anchor']==10440)
points=[11432+84*j for j in range(6)]
ix={name:np.array([i for i,r in enumerate(rows) if r['block']%2==parity]) for name,parity in (('discovery',0),('validation',1))}
results=[]
for source_name,m01_hex in group['split_candidates'].items():
    m01=int(m01_hex,16)
    _,state=product.rank.product_labels(m01,challenges,product.limb)
    guessed={}
    for name,indices in ix.items():
        wave=np.asarray(traces[np.ix_(indices,points)],float)
        guessed[name]=add.recover(state[indices],wave)
    selected=guessed['discovery']
    correlations={}
    for name,indices in ix.items():
        label,_=product.rank.add_labels(selected,state[indices],product.limb)
        subset=[rows[i] for i in indices]
        wave=np.asarray(traces[np.ix_(indices,points)],float)
        correlations[name]=np.diag(product.CONTROL.grid(product.residual(label,subset),product.residual(wave,subset))).tolist()
    results.append({'m01_source':source_name,'m01_candidate':m01_hex,'m00_candidates':{k:hex(v) for k,v in guessed.items()},'split_agreement':guessed['discovery']==guessed['validation'],'correlations':correlations,'min_validation_abs_r':min(abs(v) for v in correlations['validation'])})
    print(source_name,'agree',results[-1]['split_agreement'],'min',results[-1]['min_validation_abs_r'],flush=True)
report={'status':'KAT_free_full_M01_branch_diagnostic','target_truth_loaded':False,'points':points,'results':results,'input_trace_sha256':manifest['target_trace_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out=root/'.plans/2026-10-07-E-r3-single-split-cpa/t40-third-m-full-product-branch.json'
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
