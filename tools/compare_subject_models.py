#!/usr/bin/env python3
"""Validation-selected signal-feature models and uniform offline subject comparison."""
from pathlib import Path
import sys,os,json,hashlib,time,importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from utils.signal_features import extract_signal_features,joint_probabilities_to_heads
from utils.sequence_data import load_sequence_npz
from utils.splitting import compose_split_indices,infer_groups
spec=importlib.util.spec_from_file_location('step3_comparison',ROOT/'3_evaluate_model.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
OUT=ROOT/'logs/all_subjects_20260928';OUT.mkdir(exist_ok=True)

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def digest(indices):return hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest()
def subset(m,idx,n):return {k:(v[idx] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v) for k,v in m.items()}
def metrics(a,f,ya,yf):
 active=ya!=0;rest=~active
 return {'action_acc':float(np.mean(a==ya)),'finger_acc_non_rest':float(np.mean(f[active]==yf[active])),'joint_acc':float(np.mean((a==ya)&(f==yf))),'rest_tpr':float(np.mean(a[rest]==0)) if rest.any() else None,'action_balanced_acc':float(balanced_accuracy_score(ya,a)),'finger_balanced_acc':float(np.mean([np.mean(f[active&(yf==c)]==c) for c in np.unique(yf[active])]))}
def score(m):return .5*m['joint_acc']+.25*m['action_balanced_acc']+.25*m['finger_balanced_acc']
def evaluate(ap,fp,a,f,m,idx):
 pa=ap.argmax(1);pf=fp.argmax(1)+1;pf[pa==0]=0
 raw=metrics(pa,pf,a[idx],f[idx])
 order,segments=ev._session_temporal_segments(m,idx)
 settings=ev.PostprocessSettings(smoothing_enabled=True,smoothing_method='ema',smoothing_window=5,hysteresis_enabled=False,threshold_action=.05,threshold_finger=.2,threshold_applicability=.4,adjacency_enabled=False,finger_mode='smooth')
 sa,sf=ev.apply_postprocess_sequence(ap,fp,order,settings,trial_ids=segments)
 smoothed=metrics(sa,sf,a[idx][order],f[idx][order])
 # Restore original index order to allow paired comparisons and trial bootstrap.
 fulla=np.empty_like(sa);fullf=np.empty_like(sf);fulla[order]=sa;fullf[order]=sf
 return {'raw':raw,'ema5':smoothed},fulla,fullf

def estimator(name):
 if name=='extra_trees':return ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.7,class_weight='balanced',n_jobs=4,random_state=43)
 if name=='hist_gradient':return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=1.,learning_rate=.08,random_state=43)
 if name=='rbf_svm':return make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',class_weight='balanced',probability=True,random_state=43,cache_size=1024))
 raise ValueError(name)

def contexts():
 previous=json.loads((ROOT/'logs/improve_20260927/results.json').read_text())
 selected=json.loads((ROOT/'logs/improve_20260927/selected_recipes_before_test.json').read_text())
 out=[]
 for subject in ['1-M16','2-M16','3-M16']:
  if subject=='2-M16':
   s=ROOT/'Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2';cnn=s/'processed/models/20260319_075520';valcnn=None
  else:
   old=next(r for r in previous if r['subject']==subject);cnn=ROOT/old['run_dir'];s=cnn.parents[2];valcnn=ROOT/next(r['run_dir'] for r in selected if r['subject']==subject)
  out.append((subject,s,cnn,valcnn))
 return out

def main():
 summary=[]
 with threadpool_limits(limits=4):
  for subject,s,cnn,valcnn in contexts():
   folder=OUT/subject;folder.mkdir(exist_ok=True)
   if (folder/'result.json').exists():
    summary.append(json.loads((folder/'result.json').read_text()));save(OUT/'results.json',summary);continue
   X,a,f,m=load_sequence_npz(s/'processed/eeg_windows.npz');n=len(a)
   plan=compose_split_indices(a,f,m,test_size=.2,random_state=43,split_mode='group_trial',purge_seconds=.25,hop_seconds=None,allow_fallback=False,aux_rest_session_policy='auto_train_only')
   train,test=plan['train_idx'],plan['test_idx']
   with np.load(cnn/'test_predictions.npz') as z:ca=z['action_probs'];cf=z['finger_probs'];ct=z['test_indices_global'] if 'test_indices_global' in z else z['test_indices']
   assert np.array_equal(ct,test)
   if valcnn:
    with np.load(s/'processed/tuning_2m16_reference/validation_split.npz') as z:fit=z['fit_global'];val=z['validation_global']
    with np.load(valcnn/'test_predictions.npz') as z:va=z['action_probs'];vf=z['finger_probs'];vya=z['y_action'];vyf=z['y_finger']
    assert np.array_equal(a[val],vya) and np.array_equal(f[val],vyf)
   else:
    inner=compose_split_indices(a[train],f[train],subset(m,train,n),test_size=.2,random_state=43,split_mode='group_trial',purge_seconds=.25,hop_seconds=None,allow_fallback=False,aux_rest_session_policy='auto_train_only')
    fit,val=train[inner['train_idx']],train[inner['test_idx']];va=vf=None
   groups=np.asarray([f'{sid}:{gid}' for sid,gid in zip(m['session_id'],infer_groups(m,n))])
   for left,right in [(fit,val),(fit,test),(val,test)]:assert not np.intersect1d(groups[left],groups[right]).size
   np.savez_compressed(folder/'splits.npz',fit=fit,validation=val,train=train,test=test)
   save(folder/'protocol.json',{'dataset':str((s/'processed/eeg_windows.npz').relative_to(ROOT)),'dataset_sha256':hashlib.sha256((s/'processed/eeg_windows.npz').read_bytes()).hexdigest(),'test_sha256':digest(test),'n_total':n,'fit_n':len(fit),'validation_n':len(val),'test_n':len(test),'seed':43,'purge_seconds':.25,'smoothing':'EMA5, session/gap reset, both heads; no trial labels','selection_score':'.5 joint + .25 balanced action + .25 balanced active-finger accuracy on validation after fixed EMA5','selection_models':6,'published_2m16_unchanged':True})
   y=np.where(a==0,0,(a-1)*5+f)
   rows=[];features={}
   for mode in ['center_detrend','center']:
    features[mode]=extract_signal_features(X,fs=float(m['fs']),mode=mode)
    for family in ['extra_trees','hist_gradient','rbf_svm']:
     name=f'{mode}_{family}';cache=folder/f'{name}_validation.npz';meta_path=folder/f'{name}.json'
     print('FIT',subject,name,flush=True)
     if cache.exists() and meta_path.exists():
      with np.load(cache) as z:pa=z['action_probs'];pf=z['finger_probs']
     else:
      model=estimator(family);started=time.monotonic();model.fit(features[mode][fit],y[fit]);prob=model.predict_proba(features[mode][val]);pa,pf=joint_probabilities_to_heads(prob,model.classes_)
      joblib.dump(model,folder/f'{name}_validation.joblib',compress=3)
      np.savez_compressed(cache,action_probs=pa,finger_probs=pf,indices=val)
      save(meta_path,{'fit_seconds':time.monotonic()-started,'mode':mode,'family':family})
     met,_,_=evaluate(pa,pf,a,f,m,val)
     row={'name':name,'mode':mode,'family':family,'cnn_weight':0.,'validation':met,'score':score(met['ema5'])};rows.append(row)
     print('VALIDATION',subject,name,json.dumps(met['ema5']),flush=True)
     if va is not None:
      # Small predeclared blend grid; only inner validation labels select weights.
      for weight in [.25,.5,.75]:
       mix,_,_=evaluate(weight*va+(1-weight)*pa,weight*vf+(1-weight)*pf,a,f,m,val)
       rows.append({'name':name,'mode':mode,'family':family,'cnn_weight':weight,'validation':mix,'score':score(mix['ema5'])})
     save(folder/'validation_results.json',rows)
   if va is not None:
    met,_,_=evaluate(va,vf,a,f,m,val)
    rows.append({'name':'cnn_only','mode':None,'family':None,'cnn_weight':1.,'validation':met,'score':score(met['ema5'])})
   # Freeze decision before any candidate predicts final-test windows.
   best=max(rows,key=lambda r:(r['score'],r['validation']['ema5']['rest_tpr'] or 0))
   save(folder/'validation_results.json',rows);save(folder/'selected_before_test.json',best)
   print('SELECTED',subject,json.dumps(best),flush=True)
   if best['family']:
    model=estimator(best['family']);model.fit(features[best['mode']][train],y[train]);joblib.dump(model,folder/'selected_feature_model.joblib',compress=3)
    prob=model.predict_proba(features[best['mode']][test]);pa,pf=joint_probabilities_to_heads(prob,model.classes_)
   else:pa,pf=ca,cf
   w=best['cnn_weight'];pa=w*ca+(1-w)*pa;pf=w*cf+(1-w)*pf
   baseline,ba,bf=evaluate(ca,cf,a,f,m,test)
   candidate,na,nf=evaluate(pa,pf,a,f,m,test)
   np.savez_compressed(folder/'test_predictions.npz',action_probs=pa,finger_probs=pf,y_action=a[test],y_finger=f[test],test_indices=test,baseline_action_probs=ca,baseline_finger_probs=cf,baseline_smoothed_action=ba,baseline_smoothed_finger=bf,smoothed_action=na,smoothed_finger=nf)
   save(folder/'model_manifest.json',{'feature_model':'selected_feature_model.joblib' if best['family'] else None,'feature_mode':best['mode'],'cnn_weight':w,'cnn_run':str(cnn.relative_to(ROOT)),'feature_extractor':'utils.signal_features.extract_signal_features','prediction_decoder':'utils.signal_features.joint_probabilities_to_heads','selected_on':'inner validation only','smoothing_reset_policy':'session_or_time_gap_1.5_hops','fs':float(m['fs']),'input_shape':list(X.shape[1:]),'signal_feature_version':1})
   result={'subject':subject,'selected':best,'baseline_common_protocol':baseline,'candidate_common_protocol':candidate,'test_n':len(test),'test_sha256':digest(test),'artifact_dir':str(folder.relative_to(ROOT))};summary.append(result);save(folder/'result.json',result);save(OUT/'results.json',summary)
   print('TEST',subject,json.dumps(result),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
