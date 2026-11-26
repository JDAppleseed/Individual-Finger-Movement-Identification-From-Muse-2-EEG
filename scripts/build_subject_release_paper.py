"""Generate cohort summaries with explicit macro versus pooled denominators."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'docs/research/finalization_20261125'
def main():
 rows=json.loads((REPORT/'comparison.json').read_text())['subjects']
 keys=['action_accuracy','active_finger_accuracy','joint_accuracy','rest_recall']
 macro={k:float(np.mean([r[k] for r in rows])) for k in keys}
 counts={k:[0,0] for k in keys}
 for r in rows:
  with np.load(ROOT/r['run']/'test_predictions.npz') as z:
   a=z['y_action'];f=z['y_finger'];pa=z['action_probs'].argmax(1);pf=z['finger_probs'].argmax(1)+1;pf[pa==0]=0
   active=a!=0; rest=~active
   for k,num,den in [('action_accuracy',np.sum(pa==a),len(a)),('active_finger_accuracy',np.sum(pf[active]==f[active]),active.sum()),('joint_accuracy',np.sum((pa==a)&(pf==f)),len(a)),('rest_recall',np.sum(pa[rest]==0),rest.sum())]:
    counts[k][0]+=int(num);counts[k][1]+=int(den)
 pooled={k:v[0]/v[1] for k,v in counts.items()}
 summary=dict(subject_macro=macro,pooled=pooled,pooled_counts=counts,subjects=3,total_corpus_windows=37343,total_holdout_windows=7321)
 (REPORT/'cohort_averages.json').write_text(json.dumps(summary,indent=2)+'\n')
 tex=[r'\subsection{Three-Subject Aggregation and Full-Corpus Diagnostics}',
 r'The three November subject-specific models cover 37,343 processed windows and 7,321 held-out windows in total. Subject-macro means give each subject equal weight. Pooled rates sum correct predictions and the corresponding denominator across subjects: all held-out windows for action/joint accuracy, true movement windows for finger accuracy, and true REST windows for REST recall. These are descriptive summaries of three separately trained models, not cross-subject transfer estimates. No independent-window confidence interval is asserted because overlapping windows are correlated.',
 r'\begin{table}[t]\centering\small\caption{November descriptive cohort averages (\%).}\begin{tabular}{lrrrr}\toprule Average & Action & Finger & Joint & REST \\\midrule']
 for name,vals in [('Subject macro',macro),('Pooled windows',pooled)]:tex.append(name+' & '+' & '.join(f'{vals[k]*100:.2f}' for k in keys)+r' \\')
 tex += [r'\bottomrule\end{tabular}\end{table}',r'Full-corpus diagnostics apply the saved scaler, preprocessing, CNN+LSTM weights, and available temperatures to every finite window of the selected subject corpus. They do not require a new split or retraining and include fitting/calibration windows. They are unsmoothed classifier metrics, distinct from held-out testing and gated command replay.',r'\begin{table}[t]\centering\scriptsize\caption{Full-corpus classifier diagnostics (\%); includes development data.}\begin{tabular}{lrrrrr}\toprule Model & Windows & Action & Finger & Joint & REST \\\midrule']
 for name in ['1-M16_march5','2-M16_march19','1-M16_november','2-M16_november','3-M16_november']:
  r=json.loads((REPORT/'full_corpus'/f'{name}.json').read_text());m=r['metrics']
  tex.append(name.replace('_',' ')+' & '+str(r['evaluated_windows'])+' & '+' & '.join(f'{m[k]*100:.2f}' for k in ['action_acc','finger_acc_non_rest','joint_acc','rest_tpr'])+r' \\')
 tex += [r'\bottomrule\end{tabular}\end{table}',r'The published March 2-M16 saved-test headlines remain 89.79\% action, 85.96\% finger, and 84.66\% joint accuracy; the March 1-M16 report retains 83.94\% action, 80.61\% finger, and 92.11\% REST recall on its original holdout. Historical May command replay retains 95.37\% would-send precision, 91.11\% event hit rate, 86.42\% committed joint accuracy, and 0.25\% false REST actuation. The latter are preserved historical summaries, not newly reproduced by classifier-only full-corpus evaluation. Different corpora, holdout populations, and prediction policies prevent controlled ranking of these historical and current values.',r'The current audit reproduces all 7,321 held-out action/finger decisions and the fitting-only scaler statistics. Evaluation uses original saved holdout membership and fresh inference, never new random splits of an already fitted model. Completed fit/calibration/test folds are checked for raw-signal overlap after auxiliary REST policies. Label arrays must be one-dimensional integers; original source indices and dataset hashes identify new fold artifacts. Performance-based event flags request independent signal/label review rather than automatic pruning or relabeling.']
 (ROOT/'paper/subject_release_20261125.tex').write_text('\n\n'.join(tex)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
