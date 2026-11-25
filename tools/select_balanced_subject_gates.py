#!/usr/bin/env python3
"""Validation-only gate/model selection; frozen held-out command-policy evaluation."""
from pathlib import Path
import sys,json
from dataclasses import replace,asdict
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import compare_subject_models as bench
from utils.subject_ensemble import SubjectEnsemblePredictor
from utils.live_infer_common import predict_window,sanitize_live_window,replay_ordered_windows,compute_replay_metrics
from utils.step7_config import build_step7_postprocess_settings,build_step7_replay_runtime_config
OUT=ROOT/'logs/live_readiness_20260928/balanced_selection'

class CachedWindows:
 def __init__(self,rows):self.rows=rows;self.position=0
 def predict_window(self,window,*,prepared_window=None):
  result=self.rows[self.position];self.position+=1;return result

def setup(p,X):
 cache={k:[] for k in ['cnn','ensemble']}
 for x in X:
  q=sanitize_live_window(x,scaler=p.scaler,enabled=True,input_clip_abs_z=6,bad_channel_rms_z=4,bad_channel_abs_p95_z=6,bad_channel_clipped_frac=.05,bad_window_clipped_frac=.1,bad_window_max_masked_channels=1)
  cache['cnn'].append(predict_window(x,scaler=p.scaler,model=p.model,device=p.device,inference_engine=None,temperature_state=p.temperature,prepared_window=q.prepared_window))
  cache['ensemble'].append(p.predict_window(x,prepared_window=q.prepared_window))
 return cache

def evaluate(p,X,a,f,m,idx,segments,cache,post,runtime):
 meta=lambda k:np.asarray(m[k])[idx] if k in m else None
 mk=dict(y_action_true=a[idx],y_finger_true=f[idx],window_start_s=meta('window_start'),window_end_s=meta('window_end'),trial_ids=segments,session_ids=meta('session_id'),event_ids=meta('event_id'),event_onset_s=meta('event_onset_s'))
 replay=replay_ordered_windows(X=X[idx],**mk,scaler=p.scaler,model=p.model,device=p.device,postprocess_enabled=True,postprocess_settings=post,runtime_config=runtime,temperature_state=p.temperature,window_predictor=CachedWindows(cache))
 metrics=compute_replay_metrics(records=replay['records'],**mk,applicability_threshold=post.threshold_applicability)
 return metrics,replay['records']

def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 ref=json.loads((ROOT/'Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json').read_text())['settings']
 post=build_step7_postprocess_settings(ref)
 runtime=build_step7_replay_runtime_config(ref,latency_mode='fixed',fixed_latency_ms=125.)
 runtime=replace(runtime,reset_on_trial_change=True,use_inference_engine=False,deterministic=True)
 recipes=json.loads((ROOT/'logs/improve_20260927/selected_recipes_before_test.json').read_text());results=[]
 with threadpool_limits(limits=1):
  for subject in ['1-M16','2-M16','3-M16']:
   folder=ROOT/'logs/all_subjects_20260928'/subject;dest=OUT/subject;dest.mkdir(exist_ok=True)
   result=json.loads((folder/'result.json').read_text());manifest=json.loads((folder/'model_manifest.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
   X,a,f,m=bench.load_sequence_npz(ROOT/protocol['dataset'])
   with np.load(folder/'splits.npz') as z:val=z['validation'];test=z['test']
   valrun=folder/'cnn_validation' if subject=='2-M16' else ROOT/next(r['run_dir'] for r in recipes if r['subject']==subject)
   vmanifest=dict(manifest,cnn_run=str(valrun.relative_to(ROOT)),feature_model=str(folder/(result['selected']['name']+'_validation.joblib')))
   vpath=dest/'validation_model_manifest.json';vpath.write_text(json.dumps(vmanifest,indent=2));p=SubjectEnsemblePredictor(vpath)
   order,seg=bench.ev._session_temporal_segments(m,val);idx=val[order];cache=setup(p,X[idx]);choices=[]
   for variant in ['cnn','ensemble']:
    for stability in [2,3,4]:
     for threshold in [.2,.4,.6,.8]:
      policy=replace(runtime,actuation_stability=stability,actuation_min_prob=threshold)
      met,_=evaluate(p,X,a,f,m,idx,seg[order],cache[variant],post,policy)
      precision=met['would_send_window_precision_non_rest'] or 0.;hit=met['would_send_first_onset_latency_s']['hit_rate'] or 0.
      choices.append(dict(variant=variant,stability=stability,min_prob=threshold,utility=.5*precision+.5*hit,metrics=met))
   # Equal importance to command precision and movement-event hit rate. This is
   # an explicit utility, not statistical F1 (the denominators differ).
   best=max(choices,key=lambda r:(r['utility'],-r['metrics']['false_actuation_rate_rest'],r['min_prob'],r['stability']))
   frozen=dict(subject=subject,objective='0.5 command precision + 0.5 movement-event hit rate; descriptive utility, not F1',selection_split='inner validation only',validation_index_sha256=bench.digest(val),test_index_sha256=bench.digest(test),selected=best,all_choices=choices)
   (dest/'selected_before_test.json').write_text(json.dumps(frozen,indent=2));print('SELECTED',subject,best['variant'],best['stability'],best['min_prob'],best['utility'],flush=True)
   p=SubjectEnsemblePredictor(folder/'model_manifest.json');order,seg=bench.ev._session_temporal_segments(m,test);idx=test[order];cache=setup(p,X[idx]);policy=replace(runtime,actuation_stability=best['stability'],actuation_min_prob=best['min_prob'])
   met,records=evaluate(p,X,a,f,m,idx,seg[order],cache[best['variant']],post,policy)
   final=dict(subject=subject,variant=best['variant'],stability=best['stability'],min_prob=best['min_prob'],validation_utility=best['utility'],test_metrics=met,postprocess=asdict(post),runtime=asdict(policy),scope='Exploratory repeated fixed-test comparison; cached inference gate simulation. Timing metrics not hardware latency.')
   (dest/'test_result.json').write_text(json.dumps(final,indent=2))
   with (dest/'test_predictions.jsonl').open('w') as fp:
    for row in records:fp.write(json.dumps(row,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
   results.append(final);(OUT/'results.json').write_text(json.dumps(results,indent=2));print('TEST',subject,met['would_send_window_precision_non_rest'],met['would_send_first_onset_latency_s']['hit_rate'],met['false_actuation_rate_rest'],flush=True)
if __name__=='__main__':main()
