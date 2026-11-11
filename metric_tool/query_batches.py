"""Full constant references, query-only non-replacement batches (version 1)."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from .aligned_dcr import min_l1_distances

PROTOCOL='query_batches_full_train_v1'


def aggregate(dtrain,dreference,batch_size,repeats=2,seed=20261027):
    dt=np.asarray(dtrain);dr=np.asarray(dreference)
    if dt.shape!=dr.shape or dt.ndim!=1 or not np.all(np.isfinite(dt)) or not np.all(np.isfinite(dr)):raise ValueError('Invalid distances')
    if not 1<=repeats<=4 or batch_size<=0 or len(dt)<batch_size:raise ValueError('Invalid query plan')
    rng=np.random.default_rng(seed);plans=[];batches=[];rounds=[]
    for repeat in range(repeats):
        order=rng.permutation(len(dt));plans.append(order);shares=[]
        for batch in range(len(dt)//batch_size):
            ids=order[batch*batch_size:(batch+1)*batch_size];value=float(np.mean(dt[ids]<dr[ids]));shares.append(value)
            batches.append(dict(round=repeat+1,batch=batch+1,Share=value,rows=len(ids)))
        rounds.append(float(np.mean(shares)))
    plans=np.asarray(plans)
    return dict(Share=float(np.mean(rounds)),round_shares=rounds,batch_records=batches,query_plans=plans,
        all_query_Share=float(np.mean(dt<dr)),query_remainder=len(dt)%batch_size,
        plan_sha256=hashlib.sha256(plans.tobytes()).hexdigest(),spec=dict(protocol_id=PROTOCOL,encoding='existing_range_scaled_L1',
            train_pool='complete_constant',reference_pool='complete_constant',train_unique_matching=False,tie_weight=0,
            seed=seed,rounds=repeats,batch_size=batch_size,remainder='audit_and_exclude_main_report'))


def evaluate(runner,df,encoding):
    import time
    started=time.perf_counter()
    if runner.dcr_cap:raise ValueError(PROTOCOL+' forbids dcr-cap')
    side='hold' if encoding is runner._dcr_hold_encoding else 'test'
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(df,index=False).to_numpy(np.uint64).tobytes()+repr(list(zip(df.columns,map(str,df.dtypes)))).encode()).hexdigest()
    if getattr(runner,'_query_table_fingerprint',None)!=fingerprint:
        runner._query_table_fingerprint=fingerprint
        runner._query_train_backbone=None
        runner._query_batch_calls={}
    x=encoding.transform(df.copy())
    if runner.dataset_name=='diabetes':
        from post_selection_tool.hold_generalization.compact_l1 import CompactL1
        from .aligned_dcr import fit_tabdiff_l1_encoding
        real=pd.read_csv(runner.real_data_path);hold=pd.read_csv(runner.val_data_path)
        if getattr(runner,'_query_train_backbone',None) is None:
            fixed=CompactL1(fit_tabdiff_l1_encoding(real,hold,runner.info));runner._query_train_backbone=fixed.minimum(df,real).astype(np.float32)
        dt=runner._query_train_backbone.copy()
        ref=pd.read_csv(runner.val_data_path if side=='hold' else runner.test_data_path);dr=CompactL1(encoding).minimum(df,ref).astype(np.float32)
    else:
        distance = runner.dcr_min_l1_fn or min_l1_distances
        dt=np.asarray(distance(x,encoding.real_x),np.float32)
        dr=np.asarray(distance(x,encoding.test_x),np.float32)
    result=aggregate(dt,dr,len(encoding.test_x),int(runner.dcr_repeats),int(runner.dcr_seed))
    calls=getattr(runner,'_query_batch_calls',{});calls[side]=dict(result=result,dtrain=dt,dreference=dr,train_rows=len(encoding.real_x),reference_rows=len(encoding.test_x))
    runner._query_batch_calls=calls;rates=result['round_shares']
    return {'dcr':result['Share']},dict(dcr_real=dt,dcr_test=dr,dcr_std=float(np.std(rates,ddof=1)) if len(rates)>1 else 0.,
        dcr_n=len(encoding.test_x),dcr_repeats=len(rates),dcr_requested_repeats=int(runner.dcr_repeats),dcr_aligned=True,
        dcr_align_mode=PROTOCOL,dcr_cap=0,dcr_repeat_rates=rates,aligned_dcr_seconds=time.perf_counter()-started,
        aligned_dcr_timed_repeats=1,dcr_protocol=result['spec'])


def save_details(runner,summary,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);calls=runner._query_batch_calls
    if set(calls)!={'hold','test'}:raise ValueError('This protocol requires --enable-hold-dcr for paired auditing')
    h,t=calls['hold'],calls['test'];hr,tr=h['result'],t['result']
    if h['reference_rows']!=t['reference_rows']:raise ValueError(PROTOCOL+' requires equal hold/test reference row counts')
    if len(hr['batch_records'])!=len(tr['batch_records']):raise ValueError('Paired batch record counts differ')
    np.testing.assert_array_equal(hr['query_plans'],tr['query_plans']);np.testing.assert_array_equal(h['dtrain'],t['dtrain'])
    batches=[dict(round=a['round'],batch=a['batch'],Share_hold=a['Share'],Share_test=b['Share'],query_rows=a['rows']) for a,b in zip(hr['batch_records'],tr['batch_records'])]
    rounds=[dict(round=i+1,Share_hold=a,Share_test=b,difference_abs=abs(a-b)) for i,(a,b) in enumerate(zip(hr['round_shares'],tr['round_shares']))]
    for name,value in [('all_batches.json',batches),('all_rounds.json',rounds)]: (folder/name).write_text(json.dumps(value,indent=2))
    np.savez_compressed(folder/'dcr_points_and_plans.npz',plans=tr['query_plans'],dtrain=t['dtrain'],dhold=h['dreference'],dtest=t['dreference'])
    summary.update(protocol_id=PROTOCOL,dcr_align_mode=PROTOCOL,DCR_test=tr['Share'],DCR_hold=hr['Share'],
        full_query_DCR_test=tr['all_query_Share'],full_query_DCR_hold=hr['all_query_Share'],
        dcr_test_round_shares=tr['round_shares'],dcr_hold_round_shares=hr['round_shares'],dcr_difference_abs=abs(tr['Share']-hr['Share']),
        dcr_protocol=dict(**tr['spec'],train_reference_rows=t['train_rows'],hold_reference_rows=h['reference_rows'],test_reference_rows=t['reference_rows'],
            query_rows=len(t['dtrain']),query_remainder_per_round=tr['query_remainder'],query_plan_sha256=tr['plan_sha256'],
            query_grouping_rounds=len(tr['round_shares']),target_interpretation='empirical_0.5',train_backbone='shared train+hold encoding' if runner.dataset_name=='diabetes' else 'original per-reference encoding'),
        dcr_semantics='Complete constant train/reference L1 NN Share; query-only batches; 0.5 empirical target')
    summary['dcr_hold_semantics']=summary['dcr_semantics']
