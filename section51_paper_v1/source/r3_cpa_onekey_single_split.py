#!/usr/bin/env python3
"""Freeze single-key split-zero CRT low/u candidates before heldout audit."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ORACLE = ROOT / '.plans/2026-09-27-D-r3-full-matrix-heldout/evidence/t03-ideal-chall-oracle'
sys.path.insert(0, str(ROOT))
import r3_cpa_crt_analyze as crt

MASK = np.uint64(0xffffffff)
LOW_POINTS = []
MUL_POINTS = []
PREFIX_BEAM, WORD_KEEP, COORD_BEAM = 32, 16, 32
PAIR_BEAM, U_PREFIX_BEAM, U_WORD_KEEP = 8, 16, 2


def corr(pred, wave):
    pred = pred.astype(float); wave = wave.astype(float)
    pred -= pred.mean(); wave -= wave.mean()
    den = np.linalg.norm(pred)*np.linalg.norm(wave)
    return 0.0 if den == 0 else float(pred @ wave / den)


def score_candidates(prefixes, known, carry, wave, bits, multiply):
    mask = np.uint64((1 << bits) - 1)
    out = np.empty(len(prefixes))
    wc = wave.astype(float) - float(np.mean(wave))
    we = float(wc @ wc)
    for start in range(0, len(prefixes), 512):
        c = prefixes[start:start+512]
        stored = ((c[:,None]*known[None,:] + carry[None,:]) if multiply else
                  (c[:,None] - known[None,:] - carry[None,:])) & mask
        pred = crt.low.matrix_attack.limb.low.hw32(stored).astype(float)
        pred -= pred.mean(axis=1, keepdims=True)
        den = np.sqrt(np.einsum('ij,ij->i', pred, pred)*we)
        out[start:start+len(c)] = np.divide(np.abs(pred @ wc), den,
                                            out=np.zeros(len(c)), where=den>0)
    return out


def rank_word(known, carry, wave, multiply, keep=WORD_KEEP):
    prefixes = np.arange(256, dtype=np.uint64)
    scores = None
    for stage in range(4):
        if stage:
            prefixes = (prefixes[:,None] | (np.arange(256,dtype=np.uint64)[None,:] << np.uint64(8*stage))).ravel()
        scores = score_candidates(prefixes, known, carry, wave, 8*(stage+1), multiply)
        order = np.lexsort((prefixes, -scores))
        cut = min(PREFIX_BEAM, len(order)); threshold = scores[order[cut-1]]
        prefixes = prefixes[order[scores[order] >= threshold-1e-12][:PREFIX_BEAM]]
    scores = score_candidates(prefixes, known, carry, wave, 32, multiply)
    order = np.lexsort((prefixes, -scores))
    cut = min(keep, len(order)); threshold = scores[order[cut-1]]
    order = order[scores[order] >= threshold-1e-12][:max(keep, 32)]
    return [(int(prefixes[i]), float(scores[i])) for i in order]


def split_indices(rows, public, coord):
    main = [i for i,r in enumerate(rows) if r['coord']==coord and not r['heldout'] and public['split'][i]==0]
    train = np.array([i for i in main if hashlib.sha256(rows[i]['challenge_hex'].encode()).digest()[0]&1==0], int)
    valid = np.array([i for i in main if hashlib.sha256(rows[i]['challenge_hex'].encode()).digest()[0]&1==1], int)
    held = np.array([i for i,r in enumerate(rows) if r['coord']==coord and r['heldout'] and public['split'][i]==0], int)
    return train, valid, held


def coordinate_beam(public_words, traces, indices):
    train, valid, _ = indices
    states = [(0, np.zeros(len(train),dtype=np.uint64), np.zeros(len(valid),dtype=np.uint64), 0.0, 0.0)]
    stages=[]
    for limb, point in enumerate(LOW_POINTS):
        expanded=[]
        wave_t=np.asarray(traces[train,point],float); wave_v=np.asarray(traces[valid,point],float)
        for prefix,bt,bv,st,sv in states:
            ranked=rank_word(public_words[train,limb],bt,wave_t,False)
            for word,score_t in ranked:
                pred=(np.uint64(word)-public_words[valid,limb]-bv)&MASK
                score_v=abs(corr(crt.low.matrix_attack.limb.low.hw32(pred),wave_v))
                nbt=(np.uint64(word)<public_words[train,limb]+bt).astype(np.uint64)
                nbv=(np.uint64(word)<public_words[valid,limb]+bv).astype(np.uint64)
                expanded.append((prefix | word<<(32*limb),nbt,nbv,st+score_t,sv+score_v))
        expanded.sort(key=lambda z:(-(z[3]+z[4]),z[0]))
        states=expanded[:COORD_BEAM]
        stages.append({'limb':limb,'expanded':len(expanded),'kept':len(states)})
    return states,stages


def pair_byte_score(xword,yword,public,traces,index,wave_point):
    pieces=[]; waves=[]
    for coord,word in (('x',xword),('y',yword)):
        ix=index[coord]
        pieces.append((np.uint64(word)-public[coord][ix,0])&np.uint64(255))
        waves.append(np.asarray(traces[ix,wave_point],float))
    factor=np.concatenate(pieces); wave=np.concatenate(waves)
    candidates=np.arange(256,dtype=np.uint64)
    score=score_candidates(candidates,factor,np.zeros(len(factor),dtype=np.uint64),wave,8,True)
    best=int(np.lexsort((candidates,-score))[0])
    return best,float(score[best])


def recover_u(xword,yword,public,traces,index,points):
    word_count=len(points[0])
    factor={}; wave={}
    for part in ('train','valid'):
        pieces=[]
        for coord,word in (('x',xword),('y',yword)):
            ix=index[part][coord]
            pieces.append((np.uint64(word)-public[coord][ix,0])&MASK)
        factor[part]=np.concatenate(pieces)
        wave[part]=[np.concatenate([np.asarray(traces[index[part]['x'],points[0][j]],float),
                                     np.asarray(traces[index[part]['y'],points[0][j]],float)]) for j in range(word_count)]
    states=[(0,np.zeros(len(factor['train']),dtype=np.uint64),np.zeros(len(factor['valid']),dtype=np.uint64),0.0,0.0)]
    for j in range(word_count):
        expanded=[]
        for prefix,ct,cv,st,sv in states:
            for word,score_t in rank_word(factor['train'],ct,wave['train'][j],True,U_WORD_KEEP):
                stored=(factor['valid']*np.uint64(word)+cv)&MASK
                score_v=abs(corr(crt.low.matrix_attack.limb.low.hw32(stored),wave['valid'][j]))
                expanded.append((prefix|word<<(32*j),
                                 (factor['train']*np.uint64(word)+ct)>>np.uint64(32),
                                 (factor['valid']*np.uint64(word)+cv)>>np.uint64(32),
                                 st+score_t,sv+score_v))
        expanded.sort(key=lambda z:(-(z[3]+z[4]),z[0]))
        states=expanded[:U_PREFIX_BEAM]
    return states


def product_score(secret, u, public_words, traces, ix, points):
    word_count=len(points[0])
    sw=np.array([(secret>>(32*i))&0xffffffff for i in range(6)],dtype=np.uint64)
    borrow=np.zeros(len(ix),dtype=np.uint64); diff=np.empty((len(ix),6),dtype=np.uint64)
    for i in range(6):
        sub=public_words[ix,i]+borrow; diff[:,i]=(sw[i]-sub)&MASK; borrow=(sw[i]<sub).astype(np.uint64)
    uw=np.array([(u>>(32*j))&0xffffffff for j in range(word_count)],dtype=np.uint64)
    state=np.zeros((len(ix),6+word_count),dtype=np.uint64); total_score=0.0; per=[]
    for i in range(6):
        carry=np.zeros(len(ix),dtype=np.uint64)
        for j in range(word_count):
            total=diff[:,i]*uw[j]+state[:,i+j]+carry
            state[:,i+j]=total&MASK; carry=total>>np.uint64(32)
            r=abs(corr(crt.low.matrix_attack.limb.low.hw32(state[:,i+j]),np.asarray(traces[ix,points[i][j]],float)))
            total_score+=r; per.append(r)
        state[:,i+word_count]=carry
    return total_score,per


def low_score(secret,public_words,traces,ix,points):
    label,_=crt.low.subtraction_leakage(secret,public_words[ix])
    vals=[abs(corr(label[i],np.asarray(traces[ix,points[i]],float))) for i in range(6)]
    return sum(vals),vals


def normalized(matrix, axis):
    matrix=np.asarray(matrix,dtype=np.float32)
    matrix-=matrix.mean(axis=axis,keepdims=True)
    scale=np.sqrt(np.sum(matrix*matrix,axis=axis,keepdims=True))
    return np.divide(matrix,scale,out=np.zeros_like(matrix),where=scale>1e-6)


def inverse_labels(xword,yword,public,index):
    labels={}
    candidate=np.arange(256,dtype=np.uint64)[:,None]
    for coord,word in (('x',xword),('y',yword)):
        ix=index[coord]
        factor=(np.uint64(word)-public[coord][ix,0])&np.uint64(255)
        labels[coord]=normalized(crt.low.matrix_attack.limb.low.hw32((candidate*factor[None,:])&np.uint64(255)),axis=1)
    return labels


def multiply_points(anchor,word_count):
    return [[anchor+(13*word_count+42)*i+13*j for j in range(word_count)] for i in range(6)]


def inverse_row0_score(xword,yword,u,public,traces,index,anchor,word_count):
    u_words=[(u>>(32*j))&0xffffffff for j in range(word_count)]
    values=[]
    for coord,word in (('x',xword),('y',yword)):
        ix=index[coord]
        factor=(np.uint64(word)-public[coord][ix,0])&MASK
        carry=np.zeros(len(ix),dtype=np.uint64)
        for j,u_word in enumerate(u_words):
            total=factor*np.uint64(u_word)+carry
            stored=total&MASK
            carry=total>>np.uint64(32)
            predicted=crt.low.matrix_attack.limb.low.hw32(stored)
            values.append(abs(corr(predicted,np.asarray(traces[ix,anchor+13*j],float))))
    return sum(values),min(values)


def inverse_anchor(low_pairs,public,traces,train_ix,valid_ix,last_low,failure_path):
    left,right=last_low+1,min(last_low+2049,traces.shape[1])
    if right-left<1024:
        raise RuntimeError('inverse search window is truncated')
    best_score=np.zeros(right-left,dtype=np.float32)
    best_pair=np.zeros(right-left,dtype=np.int16)
    best_byte=np.zeros(right-left,dtype=np.uint8)
    for pair_index,(xword,yword) in enumerate(low_pairs):
        model=inverse_labels(xword,yword,public,train_ix)
        for start in range(left,right,512):
            stop=min(start+512,right)
            grids=[np.abs(model[coord]@normalized(np.asarray(traces[train_ix[coord],start:stop]),axis=0))
                   for coord in ('x','y')]
            grid=np.minimum(*grids)
            choice=np.argmax(grid,axis=0)
            score=grid[choice,np.arange(stop-start)]
            positions=np.arange(start-left,stop-left)
            improved=score>best_score[positions]
            selected=positions[improved]
            best_score[selected]=score[improved]
            best_pair[selected]=pair_index
            best_byte[selected]=choice[improved].astype(np.uint8)
    validation_score=np.zeros_like(best_score)
    for pair_index,(xword,yword) in enumerate(low_pairs):
        offsets=np.flatnonzero(best_pair==pair_index)
        if not len(offsets):
            continue
        model=inverse_labels(xword,yword,public,valid_ix)
        grids=[]
        for coord in ('x','y'):
            wave=normalized(np.asarray(traces[np.ix_(valid_ix[coord],left+offsets)]),axis=0)
            grids.append(np.abs(np.sum(model[coord][best_byte[offsets]]*wave.T,axis=1)))
        validation_score[offsets]=np.minimum(*grids)
    peaks=[]
    for offset in np.argsort(np.minimum(best_score,validation_score))[::-1]:
        sample=left+int(offset)
        if any(abs(sample-row['sample'])<13 for row in peaks):
            continue
        xword,yword=low_pairs[int(best_pair[offset])]
        byte=int(best_byte[offset])
        validation=inverse_labels(xword,yword,public,valid_ix)
        score_by_coord={coord:np.abs(validation[coord]@normalized(
            np.asarray(traces[valid_ix[coord],sample:sample+1]),axis=0))[:,0]
            for coord in ('x','y')}
        values=np.minimum(score_by_coord['x'],score_by_coord['y'])
        if not np.isclose(values[byte],validation_score[offset],atol=1e-4):
            raise RuntimeError('inverse validation score changed between scan and peak audit')
        peaks.append({'sample':sample,'low_pair':[hex(xword),hex(yword)],
                      'u_first_byte':byte,'train_abs_r':float(best_score[offset]),
                      'validation_abs_r':float(values[byte]),
                      'validation_abs_r_by_coord':{coord:float(score_by_coord[coord][byte])
                                                    for coord in ('x','y')},
                      'validation_rank':1+int(np.count_nonzero(values>values[byte]+1e-6))})
        if len(peaks)==32:
            break
    accepted=[]
    attempts=[]
    for row in peaks:
        if row['validation_rank']!=1 or row['validation_abs_r']<=0.2:
            continue
        anchor=row['sample']
        xword,yword=(int(value,16) for value in row['low_pair'])
        for word_count in (8,9):
            points=multiply_points(anchor,word_count)
            if points[-1][-1]>=traces.shape[1]:
                attempts.append({'sample':anchor,'word_count':word_count,'window_fit':False})
                continue
            candidates=recover_u(xword,yword,public,traces,
                                 {'train':train_ix,'valid':valid_ix},points)
            length_valid=0
            for state in candidates:
                u=state[0]
                if (u.bit_length()+31)//32!=word_count:
                    continue
                length_valid+=1
                train_sum,_=inverse_row0_score(xword,yword,u,public,traces,train_ix,anchor,word_count)
                valid_sum,valid_min=inverse_row0_score(xword,yword,u,public,traces,valid_ix,anchor,word_count)
                accepted.append({**row,'u_candidate':hex(u),'u_word_count':word_count,
                                 'row0_train_mean':train_sum/(2*word_count),
                                 'row0_validation_mean':valid_sum/(2*word_count),
                                 'row0_validation_min':valid_min})
            attempts.append({'sample':anchor,'word_count':word_count,'window_fit':True,
                             'raw_candidates':len(candidates),'length_valid_candidates':length_valid})
    scored=accepted
    accepted=[row for row in accepted if row['row0_validation_min']>0.2]
    if not accepted:
        with failure_path.open('x') as output:
            json.dump({'status':'no_inverse_source_order_path_validates','window':[left,right],
                       'peaks':peaks,'first_byte_gate_count':len({row['sample'] for row in attempts}),
                       'attempts':attempts,'scored_candidate_count':len(scored),
                       'best_row0_validation_min':max((row['row0_validation_min'] for row in scored),default=None)},
                      output,indent=2,sort_keys=True)
            output.write('\n')
        raise RuntimeError('no inverse source-order path validates')
    chosen=max(accepted,key=lambda row:(row['row0_train_mean']+row['row0_validation_mean'],
                                        row['row0_validation_min']))
    return chosen['sample'],{'window':[left,right],'peaks':peaks,'selected':chosen}


def main():
    global LOW_POINTS
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--m-candidates',type=Path,required=True)
    parser.add_argument('--subtraction-poi',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    args=parser.parse_args()
    if args.frozen.exists():
        raise FileExistsError('frozen output already exists')
    started=time.perf_counter()
    manifest=json.loads((args.data/'manifest.json').read_text())
    if manifest['target_truth_loaded'] or manifest['known_key_trace_count']:
        raise RuntimeError('analysis input is not target-only')
    if (crt.digest(args.data/'traces.npy')!=manifest['target_trace_sha256'] or
            crt.digest(args.data/'public_schedule.jsonl')!=manifest['target_public_schedule_sha256']):
        raise RuntimeError('analysis input hash mismatch')
    poi=json.loads(args.subtraction_poi.read_text())
    if (poi['status']!='selected' or poi['target_trace_sha256']!=manifest['target_trace_sha256'] or
            poi['target_public_schedule_sha256']!=manifest['target_public_schedule_sha256'] or
            poi['m_candidates_sha256']!=crt.digest(args.m_candidates)):
        raise RuntimeError('subtraction POI is not bound to these inputs')
    LOW_POINTS=poi['low_points']
    if len(LOW_POINTS)!=6 or max(LOW_POINTS)>=manifest['sample_count']:
        raise RuntimeError('subtraction POI points are outside trace window')
    rows=[json.loads(s) for s in (args.data/'public_schedule.jsonl').read_text().splitlines()]
    traces=np.load(args.data/'traces.npy',mmap_mode='r',allow_pickle=False)
    m=json.loads(args.m_candidates.read_text())['matrix']
    if poi['m10'] not in m['m10_candidates']:
        raise RuntimeError('selected M branch is not among frozen M candidates')
    all_results=[]
    inverse_report=None
    for m10s in (poi['m10'],):
        matrix=tuple(int(m[k],16) for k in ('m00','m01'))+(int(m10s,16),int(m['m11'],16))
        public=crt.oracle_public(rows,matrix,ORACLE)
        idx={c:split_indices(rows,public,c) for c in ('x','y')}
        xb,xstage=coordinate_beam(public['x'],traces,idx['x'])
        yb,ystage=coordinate_beam(public['y'],traces,idx['y'])
        train_ix={c:idx[c][0] for c in ('x','y')}; valid_ix={c:idx[c][1] for c in ('x','y')}
        low_pairs=sorted({(x[0]&0xffffffff,y[0]&0xffffffff) for x in xb for y in yb})
        inverse_base,inverse_report=inverse_anchor(low_pairs,public,traces,train_ix,valid_ix,
                                                  LOW_POINTS[-1],args.frozen.with_name('inverse_failure.json'))
        layouts={word_count:multiply_points(inverse_base,word_count) for word_count in (8,9)}
        if any(points[-1][-1]>=traces.shape[1] for points in layouts.values()):
            raise RuntimeError('inverse/product points exceed trace window')
        screened=[]
        for xw,yw in low_pairs:
            ubyte,ts=pair_byte_score(xw,yw,public,traces,train_ix,inverse_base)
            pred=[]; wav=[]
            for c,w in (('x',xw),('y',yw)):
                ix=valid_ix[c]; d=(np.uint64(w)-public[c][ix,0])&np.uint64(255)
                pred.append(crt.low.matrix_attack.limb.low.hw32(d*np.uint64(ubyte)&np.uint64(255)))
                wav.append(np.asarray(traces[ix,inverse_base],float))
            vs=abs(corr(np.concatenate(pred),np.concatenate(wav)))
            screened.append((-(ts+vs),xw,yw,ubyte,ts,vs))
        screened.sort(); screened=screened[:PAIR_BEAM]
        u_by_pair={}
        uv_index={'train':train_ix,'valid':valid_ix}
        for _,xw,yw,*_ in screened:
            candidates=[]
            for word_count,points in layouts.items():
                candidates.extend((word_count,state) for state in
                                  recover_u(xw,yw,public,traces,uv_index,points)
                                  if (state[0].bit_length()+31)//32==word_count)
            u_by_pair[(xw,yw)]=candidates
        tuples=[]
        for x in xb:
            for y in yb:
                pair=(x[0]&0xffffffff,y[0]&0xffffffff)
                for word_count,u in u_by_pair.get(pair,[]):
                    points=layouts[word_count]
                    psx,_=product_score(x[0],u[0],public['x'],traces,idx['x'][1],points)
                    psy,_=product_score(y[0],u[0],public['y'],traces,idx['y'][1],points)
                    ls=x[4]+y[4]
                    tuples.append({'m10':m10s,'x':x[0],'y':y[0],'u':u[0],
                                   'u_word_count':word_count,'multiply_points':points,
                                   'selection_score':ls/12+(psx+psy)/(12*word_count),
                                   'low_validation_mean':ls/12,
                                   'product_validation_mean':(psx+psy)/(12*word_count)})
        tuples.sort(key=lambda z:(-z['selection_score'],z['x'],z['y'],z['u']))
        all_results.extend(tuples[:64])
        print('branch',m10s[-4:],'xbeam',len(xb),'ybeam',len(yb),'pairs',len(low_pairs),'screened',len(screened),'tuples',len(tuples),flush=True)
    all_results.sort(key=lambda z:(-z['selection_score'],z['m10'],z['x'],z['y'],z['u']))
    frozen=all_results[:64]
    selection_elapsed=time.perf_counter()-started
    if len(frozen)<2:
        raise RuntimeError('not enough CRT candidates to assess selection margin')
    frozen_report={'schema':'r3-single-split-low-joint-frozen-v2',
                   'status':'frozen_before_heldout_and_KAT',
                   'selection_elapsed_s':selection_elapsed,
                   'beam':{'prefix':PREFIX_BEAM,'word_keep':WORD_KEEP,'coord':COORD_BEAM,
                           'pair':PAIR_BEAM,'u':U_PREFIX_BEAM,'u_word_keep':U_WORD_KEEP,'final':64},
                   'points':{'low':LOW_POINTS,'inverse_base':inverse_base,
                             'layouts':{str(count):points for count,points in layouts.items()}},
                   'inverse_anchor':inverse_report,
                   'target_trace_sha256':manifest['target_trace_sha256'],
                   'target_public_schedule_sha256':manifest['target_public_schedule_sha256'],
                   'subtraction_poi_sha256':crt.digest(args.subtraction_poi),
                   'm_candidates_sha256':crt.digest(args.m_candidates),
                   'oracle_sha256':crt.digest(ORACLE),
                   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'selection_margin':frozen[0]['selection_score']-frozen[1]['selection_score'],
                   'evidence_contract':{'evidence_origin':'measured','evidence_purpose':'diagnostic',
                                        'blinding':'unblinded','measurement_scope':'isolated_primitive',
                                        'claim_scope':'r3_single_split:one_key_coordinate_candidate',
                                        'target_truth_loaded':False,
                                        'claim_not_allowed':['blind_success_rate','scheme_break']},
                   'frozen':frozen}
    with args.frozen.open('x') as output:
        json.dump(frozen_report,output,indent=2)
        output.write('\n')
    os.chmod(args.frozen,0o600)
    print(json.dumps({'frozen':str(args.frozen),'count':len(frozen),
                      'selection_margin':frozen_report['selection_margin'],
                      'elapsed_s':time.perf_counter()-started},indent=2))

if __name__=='__main__': main()
