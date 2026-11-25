#!/usr/bin/env python3
"""Check previously investigated April movement windows; no new fitting or label edits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.subject_ensemble import SubjectEnsemblePredictor
from utils.sequence_data import load_sequence_npz
from utils.live_infer_common import predict_window

def main():
 torch.set_num_threads(1)
 path=ROOT/'Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01/processed/eeg_windows.npz'
 X,a,f,m=load_sequence_npz(path);idx=np.flatnonzero(a!=0)
 p=SubjectEnsemblePredictor(ROOT/'logs/all_subjects_20260928/2-M16/model_manifest.json');rows=[]
 with threadpool_limits(limits=1):
  for name in ['cnn','ensemble']:
   acts=[];fingers=[]
   for i in idx:
    r=p.predict_window(X[i]) if name=='ensemble' else predict_window(X[i],scaler=p.scaler,model=p.model,device=p.device,inference_engine=None,temperature_state=p.temperature)
    act=int(np.argmax(r['action_probs']));finger=int(np.argmax(r['finger_probs']))+1 if act else 0
    acts.append(act);fingers.append(finger)
   acts=np.array(acts);fingers=np.array(fingers)
   rows.append(dict(model=name,movement_windows=len(idx),predicted_rest_count=int(sum(acts==0)),action_accuracy=float(np.mean(acts==a[idx])),active_finger_accuracy=float(np.mean(fingers==f[idx])),joint_accuracy=float(np.mean((acts==a[idx])&(fingers==f[idx])))))
 out=dict(scope='Previously investigated April 28 session, not a new untouched test; active labeled windows only. No claims from retrospective background REST labels.',dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=rows)
 dest=ROOT/'logs/live_readiness_20260928/session_transfer_2m16.json';dest.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
