#!/usr/bin/env python3
"""Retain initial 60-epoch CNNs as comparators; do not assume newer is better."""
from pathlib import Path
import sys,json,hashlib
from dataclasses import replace,asdict
import torch,numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import compare_subject_models as bench
from utils.live_infer_common import load_model_artifacts,replay_ordered_windows,compute_replay_metrics
from utils.step7_config import build_step7_postprocess_settings,build_step7_replay_runtime_config

def main():
 torch.set_num_threads(1);out=ROOT/'logs/live_readiness_20260928'
 initial=json.loads((ROOT/'logs/full_training_20260927/results.json').read_text());summary=json.loads((out/'results.json').read_text())
 summary=[r for r in summary if r['variant']!='initial_60epoch_cnn']
 settings=json.loads((ROOT/'Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json').read_text())['settings'];post=build_step7_postprocess_settings(settings);runtime=replace(build_step7_replay_runtime_config(settings,latency_mode='compute',fixed_latency_ms=None),reset_on_trial_change=True,deterministic=True,use_inference_engine=False)
 with threadpool_limits(limits=1):
  for row in initial:
   subject=row['subject'];folder=ROOT/'logs/all_subjects_20260928'/subject;protocol=json.loads((folder/'protocol.json').read_text());X,a,f,m=bench.load_sequence_npz(ROOT/protocol['dataset']);run=ROOT/row['run_dir']
   with np.load(folder/'splits.npz') as z:test=z['test']
   with np.load(run/'test_predictions.npz') as z:assert np.array_equal(test,z['test_indices_global'] if 'test_indices_global' in z else z['test_indices'])
   model,scaler,temp=load_model_artifacts(run_dir=run,device=torch.device('cpu'),n_channels=4)
   order,seg=bench.ev._session_temporal_segments(m,test);idx=test[order];meta=lambda k:np.asarray(m[k])[idx] if k in m else None
   mk=dict(window_start_s=meta('window_start'),window_end_s=meta('window_end'),y_action_true=a[idx],y_finger_true=f[idx],trial_ids=seg[order],session_ids=meta('session_id'),event_ids=meta('event_id'),event_onset_s=meta('event_onset_s'))
   replay=replay_ordered_windows(X=X[idx],**mk,scaler=scaler,model=model,device=torch.device('cpu'),postprocess_enabled=True,postprocess_settings=post,runtime_config=runtime,temperature_state=temp)
   met=compute_replay_metrics(records=replay['records'],**mk,applicability_threshold=post.threshold_applicability);times=np.array([r['offline_compute_ms'] for r in replay['records']])
   result=dict(subject=subject,variant='initial_60epoch_cnn',scope='fixed held-out labeled segments; no new-session claim',test_n=len(test),test_hash=bench.digest(test),metrics=met,compute_ms_p95=float(np.quantile(times,.95)),cnn_run=str(run.relative_to(ROOT)),postprocess=asdict(post),runtime=asdict(runtime))
   dest=out/subject/'initial_60epoch_cnn';dest.mkdir(exist_ok=True);(dest/'result.json').write_text(json.dumps(result,indent=2))
   with (dest/'predictions.jsonl').open('w') as fp:
    for rec in replay['records']:fp.write(json.dumps(rec,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
   summary.append(result);print(subject,met['would_send_window_precision_non_rest'],met['would_send_first_onset_latency_s']['hit_rate'],met['false_actuation_rate_rest'],flush=True)
 (out/'results.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
