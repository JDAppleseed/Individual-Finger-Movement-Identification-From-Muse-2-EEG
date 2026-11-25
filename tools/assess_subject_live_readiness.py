#!/usr/bin/env python3
"""Compare frozen CNN/ensemble candidates through shared Step 7 gates, without serial IO."""
from pathlib import Path
import sys,json,hashlib,time
from dataclasses import asdict,replace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import compare_subject_models as bench
from utils.subject_ensemble import SubjectEnsemblePredictor
from utils.live_infer_common import replay_ordered_windows,compute_replay_metrics
from utils.step7_config import build_step7_postprocess_settings,build_step7_replay_runtime_config

def main():
 torch.set_num_threads(1)
 out=ROOT/'logs/live_readiness_20260928';out.mkdir(exist_ok=True)
 ref=ROOT/'Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json'
 reference=json.loads(ref.read_text());settings=reference['settings']
 post=build_step7_postprocess_settings(settings)
 runtime=build_step7_replay_runtime_config(settings,latency_mode='compute',fixed_latency_ms=None)
 runtime=replace(runtime,reset_on_trial_change=True,use_inference_engine=False,deterministic=True)
 summary=[]
 with threadpool_limits(limits=1):
  for subject in ['1-M16','2-M16','3-M16']:
   folder=ROOT/'logs/all_subjects_20260928'/subject;pr=json.loads((folder/'protocol.json').read_text())
   X,a,f,m=bench.load_sequence_npz(ROOT/pr['dataset'])
   with np.load(folder/'splits.npz') as z:test=z['test']
   order,seg=bench.ev._session_temporal_segments(m,test);idx=test[order]
   predictor=SubjectEnsemblePredictor(folder/'model_manifest.json')
   # Parity against cached raw outputs, before quality processing and gates.
   with np.load(folder/'test_predictions.npz') as z:
    check=np.linspace(0,len(test)-1,32,dtype=int);errors=[]
    for k in check:
     r=predictor.predict_window(X[test[k]])
     errors.append(max(np.max(abs(r['action_probs']-z['action_probs'][k])),np.max(abs(r['finger_probs']-z['finger_probs'][k]))))
   assert max(errors)<2e-5,(subject,max(errors))
   for variant in ['cnn','ensemble']:
    print('REPLAY',subject,variant,flush=True)
    dest=out/subject/variant;dest.mkdir(parents=True,exist_ok=True)
    meta=lambda key:np.asarray(m[key])[idx] if key in m else None
    kwargs=dict(X=X[idx],window_start_s=meta('window_start'),window_end_s=meta('window_end'),y_action_true=a[idx],y_finger_true=f[idx],trial_ids=seg[order],session_ids=meta('session_id'),event_ids=meta('event_id'),event_onset_s=meta('event_onset_s'),scaler=predictor.scaler,model=predictor.model,device=torch.device('cpu'),postprocess_enabled=True,postprocess_settings=post,runtime_config=runtime,temperature_state=predictor.temperature,window_predictor=predictor if variant=='ensemble' else None)
    started=time.perf_counter();replay=replay_ordered_windows(**kwargs);records=replay['records']
    mk={k:kwargs[k] for k in ['y_action_true','y_finger_true','window_start_s','window_end_s','trial_ids','session_ids','event_ids','event_onset_s']}
    metrics=compute_replay_metrics(records=records,**mk,applicability_threshold=post.threshold_applicability)
    assert metrics['deployment_pair_invariant_ok']
    compute=np.array([r['offline_compute_ms'] for r in records]);fallback=sum(r['inference_backend'].endswith('fallback') for r in records)
    result=dict(subject=subject,variant=variant,scope='fixed held-out labeled segments; not unseen continuous live recording',test_hash=bench.digest(test),test_n=len(test),metrics=metrics,compute_ms_median=float(np.median(compute)),compute_ms_p95=float(np.quantile(compute,.95)),compute_ms_max=float(compute.max()),over_50ms_windows=int(sum(compute>50)),quality_cnn_fallback_windows=fallback,raw_probability_parity_max_abs_error=float(max(errors)),elapsed_s=time.perf_counter()-started,postprocess=asdict(post),runtime=asdict(runtime),reference_config_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),reset_policy='session or observable time gap >1.5 hops; no event labels')
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    with (dest/'predictions.jsonl').open('w') as fp:
     for row in records:fp.write(json.dumps(row,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
    np.savez_compressed(dest/'evaluated_indices.npz',indices=idx,segments=seg[order])
    summary.append(result);(out/'results.json').write_text(json.dumps(summary,indent=2))
    print(subject,variant,'precision',metrics['would_send_window_precision_non_rest'],'event',metrics['would_send_first_onset_latency_s'].get('hit_rate'),'REST false',metrics['false_actuation_rate_rest'],'compute_p95',result['compute_ms_p95'],flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
