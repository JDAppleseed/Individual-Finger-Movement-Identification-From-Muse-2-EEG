"""Compact, descriptive checks against class imbalance on frozen holdouts."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/research/finalization_20261125'
def main():
 rows=json.loads((D/'comparison.json').read_text())['subjects']; results=[]
 for r in rows:
  run=ROOT/r['run'];cfg=json.loads((run/'train_config.json').read_text())
  # Resolve the pinned dataset using the audited registry, not machine-specific config paths.
  registry=json.loads((D/'datasets.json').read_text()); dataset=next(x['dataset'] for x in registry if x['subject']==r['subject'])
  with np.load(ROOT/dataset,allow_pickle=False) as z, np.load(run/'split_indices.npz') as folds, np.load(run/'test_predictions.npz') as pred:
   fit=folds['fit'];a=z['y_action'];f=z['y_finger'];train_pair=a[fit]*6+f[fit]
   constant=int(np.bincount(train_pair,minlength=18).argmax())
   truth=pred['y_action']*6+pred['y_finger'];pa=pred['action_probs'].argmax(1);pf=pred['finger_probs'].argmax(1)+1;pf[pa==0]=0;guess=pa*6+pf
   recalls=[float(np.mean(guess[truth==k]==k)) for k in np.unique(truth)]
   results.append(dict(subject=r['subject'],fit_selected_constant_pair=constant,majority_joint_accuracy=float(np.mean(truth==constant)),joint_balanced_accuracy=float(np.mean(recalls)),joint_accuracy=float(np.mean(truth==guess)),represented_joint_classes=len(recalls),test_events=r['test_event_count'],test_rest_events=r['test_rest_event_count'],event_majority_joint_accuracy=r['event_majority_joint_accuracy']))
 out=ROOT/'paper_artifacts/concise_support_analysis.json';out.write_text(json.dumps(results,indent=2)+'\n')
 tex=[r'\begin{table}[t]\centering\scriptsize',r'\caption{Class-imbalance and event-level checks (\%). Constant predicts the most frequent fitting-set joint label; balanced accuracy averages recall over represented joint classes. Events are source-session/event groups.}',r'\label{tab:support}',r'\begin{tabular}{lrrrrr}\toprule Subject & Constant & Balanced & Event joint & Events & REST events \\\midrule']
 for r in results:tex.append(r['subject']+' & '+' & '.join(f'{r[k]*100:.2f}' for k in ['majority_joint_accuracy','joint_balanced_accuracy','event_majority_joint_accuracy'])+f" & {r['test_events']} & {r['test_rest_events']}"+r' \\')
 tex.append(r'\bottomrule\end{tabular}\end{table}')
 (ROOT/'paper/support_analysis.tex').write_text('\n'.join(tex)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__': main()
