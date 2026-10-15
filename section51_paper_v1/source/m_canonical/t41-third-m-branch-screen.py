import hashlib,json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np
root=Path(os.environ['SECTION51_SOURCE_ROOT']);e=Path(os.environ['SECTION51_M_OUT'])
p=SourceFileLoader('product',str(root/'m_canonical/t05-m-product-groups.py')).load_module()
data=Path(os.environ['SECTION51_M_DATA']);manifest=json.load(open(data/'manifest.json'))
assert not manifest['target_truth_loaded'] and not manifest['known_key_trace_count']
rows=[json.loads(v) for v in (data/'public_schedule.jsonl').read_text().splitlines()]
traces=np.load(data/'traces.npy',mmap_mode='r',allow_pickle=False);challenge=p.rank.words(rows)
ix={name:np.array([i for i,r in enumerate(rows) if r['block']%2==parity]) for name,parity in (('discovery',0),('validation',1))}
branches=json.load(open(Path(os.environ['SECTION51_M_OUT'])/'t40-third-m-full-product-branch.json'))['results']
m10group=json.load(open(e/'t08-m10-groups.json'))['groups'][0]
assert m10group['split_agreement'] and m10group['min_validation_abs_r']>0.5
m10=int(m10group['split_candidates']['discovery'],16);m11=int(m10group['product_candidate'],16)
points=[10440+52*j for j in range(6)]
results=[]
for branch in branches:
 m01=int(branch['m01_candidate'],16);m00=int(branch['m00_candidates']['discovery'],16)
 score={}
 for name,ids in ix.items():
  values,_=p.rank.product_labels(m01,challenge[ids],p.limb)
  subset=[rows[i] for i in ids];wave=np.asarray(traces[np.ix_(ids,points)],float)
  corr=np.diag(p.CONTROL.grid(p.residual(values.T,subset),p.residual(wave,subset)))
  score[name]=corr.tolist()
 results.append({'branch':branch['m01_source'],'matrix':{'m00':hex(m00),'m01':hex(m01),'m10':hex(m10),'m11':hex(m11)},'determinant_odd':bool((m00*m11-m01*m10)&1),'m01_product_correlations':score,'m01_product_min_validation_abs_r':min(abs(v) for v in score['validation']),'m00_add_min_validation_abs_r':branch['min_validation_abs_r']})
 print(branch['m01_source'],'product_min',results[-1]['m01_product_min_validation_abs_r'],'det_odd',results[-1]['determinant_odd'])
out=Path(os.environ['SECTION51_M_OUT'])/'t41-third-m-branch-screen.json'
with out.open('x') as f:json.dump({'status':'KAT_free_M_branch_screen','target_truth_loaded':False,'target_trace_sha256':manifest['target_trace_sha256'],'points':points,'results':results,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f,indent=2);f.write('\n')
