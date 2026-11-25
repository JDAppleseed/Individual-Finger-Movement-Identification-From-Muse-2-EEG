"""Build paper tables from the archived 2026-09-28 comparison; no model rerun."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'logs/all_subjects_20260928/results.json').read_text())
out = ROOT / 'paper_artifacts'
def table(name, caption, label, cols, header, body, wide=False):
    env='table*' if wide else 'table'
    text='\n'.join([r'% Generated from logs/all_subjects_20260928/results.json',rf'\begin{{{env}}}[t]',r'\centering\footnotesize',rf'\caption{{{caption}}}',rf'\label{{{label}}}',r'\setlength{\tabcolsep}{4pt}',rf'\begin{{tabular}}{{{cols}}}',r'\toprule',header+r' \\',r'\midrule',*body,r'\bottomrule',r'\end{tabular}',rf'\end{{{env}}}',''])
    (out/name).write_text(text)
body=[]
for r in rows:
 a=r['audit']; body.append(f"{r['subject']} & {a['source_sessions']} & {a['total_windows']:,} & {a['training_windows']:,} & {r['test_n']:,}"+r' \\')
table('tables_subject_inventory.tex','Combined subject datasets. Training pool includes inner validation/calibration. Purged windows explain differences between total and train plus test.','tab:subject-inventory','lrrrr',r'Subject & Sources & Total & Train pool & Test',body)
body=[]
for r in rows:
 for key, name in [('baseline_common_protocol','CNN'),('candidate_common_protocol','Ensemble')]:
  values=[]
  for protocol in ['raw','ema5']:
   m=r[key][protocol]
   values += [f'{100*m[k]:.2f}' for k in ['action_acc','finger_acc_non_rest','joint_acc','rest_tpr']]
  body.append(' & '.join([r['subject'],name]+values)+r' \\')
 body.append(r'\addlinespace')
table('tables_subject_comparison.tex',r'Fixed-test offline comparison (\%). Raw columns use no temporal smoothing; EMA columns use the same span-5 gap-reset smoothing for both models. A: action accuracy; F: active-finger accuracy on true non-REST windows; J: joint accuracy; R: REST recall. These are classification metrics, not actuator replay results.','tab:subject-comparison','llrrrrrrrr',r'Subject & Model & Raw A & Raw F & Raw J & Raw R & EMA A & EMA F & EMA J & EMA R',body,True)
body=[]
for r in rows:
 a=r['audit']; lo,hi=a['paired_event_bootstrap_95_ci_pp']; e=a['event_majority_metrics']
 body.append(f"{r['subject']} & {a['test_event_groups']} ({a['test_rest_event_groups']}) & {100*e['baseline']['joint_acc']:.2f} & {100*e['candidate']['joint_acc']:.2f} & {a['joint_improvement_pp']:+.2f} [{lo:.2f}, {hi:.2f}]"+r' \\')
table('tables_subject_events.tex',r'Event-level outcomes and exploratory uncertainty under common EMA. Event columns report majority joint accuracy (\%); gain is the window-level joint difference in percentage points with its paired event-bootstrap 95\% interval. REST event counts appear in parentheses.','tab:subject-events','llrrl',r'Subject & Events (REST) & Event CNN & Event ensemble & Window gain [95\% CI]',body,True)
body=[]
for r in rows:
 vals=[f'{100*v:.2f}' for v in r['audit']['active_finger_recall'].values()]
 body.append(' & '.join([r['subject']]+vals)+r' \\')
table('tables_subject_fingers.tex',r'Ensemble per-finger recall on true active windows after common EMA (\%).','tab:subject-fingers','lrrrrr',r'Subject & Thumb & Index & Middle & Ring & Pinky',body)
print('Generated four comparison tables.')
