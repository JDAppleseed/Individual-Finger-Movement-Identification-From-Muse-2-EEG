#!/usr/bin/env python3
"""Build public aggregate tables, a chart, and private manuscript input."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/research/finalization_20261125'


def main():
    payload = json.loads((REPORT / 'comparison.json').read_text())
    rows = payload['subjects']
    headers = ['Subject', 'Fit', 'Calibration', 'Test', 'Action', 'Active finger', 'Joint', 'REST recall', 'Event joint']
    lines = ['# CNN-LSTM comparison', '',
        'Fixed 60-epoch recipe, seed 43, separate subject models, unsmoothed calibrated predictions.',
        'Retrospective event holdouts; no unseen-session or live-actuation claim.', '',
        '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    metrics = ['action_accuracy', 'active_finger_accuracy', 'joint_accuracy', 'rest_recall', 'event_majority_joint_accuracy']
    for row in rows:
        cells = [row['subject']] + [str(row['fold_counts'][k]) for k in ['fit', 'calibration', 'test']]
        cells += [f'{row[k] * 100:.2f}%' for k in metrics]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Finger accuracy conditions on true movement; predicted REST counts as an error.',
        'Event joint uses the majority joint label within each held-out source event.', '',
        '## Per-finger recall', '', '| Subject | Thumb | Index | Middle | Ring | Pinky |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + row['subject'] + ' | ' + ' | '.join(f'{row["per_finger_recall"][str(k)] * 100:.2f}%' for k in range(1, 6)) + ' |')
    lines += ['', '## Historical context', '',
        '| Historical CNN-LSTM | Action | Active finger | Joint |',
        '| --- | ---: | ---: | ---: |',
        '| 1-M16 September initial 60 epochs | 76.23% | 81.75% | 67.71% |',
        '| 2-M16 March deployment reference | 89.79% | 85.96% | 84.66% |',
        '| 3-M16 September initial 60 epochs | 67.77% | 59.47% | 53.62% |', '',
        'The 1-M16/3-M16 datasets and outer test indices match the September initial comparison,',
        'but fitting/calibration allocation and normalizer fitting changed. The 2-M16 November',
        'dataset includes April and uses a different test set. These are not controlled architecture deltas.',
        'Historical values are archived summaries; November values are recomputed from saved predictions.', '',
        '![Subject comparison](comparison.png)', '']
    (REPORT / 'COMPARISON.md').write_text('\n'.join(lines))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(rows))
    for i, (key, name, color) in enumerate(zip(metrics[:4], ['Action', 'Active finger', 'Joint', 'REST recall'], ['#3465a4', '#5b9a8b', '#8c6bb1', '#c98739'])):
        bars = ax.bar(x + (i - 1.5) * .18, [100 * row[key] for row in rows], .17, label=name, color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=[r['subject'] for r in rows], ylim=(0, 112), ylabel='Percent',
        title='CNN-LSTM: fixed-recipe within-subject comparison')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=4, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .015, 'Single seed; retrospective event holdouts; no new live validation', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(REPORT / 'comparison.png', dpi=180)
    plt.close(fig)
    paper = ROOT / 'paper'
    if paper.exists():
        tex = [r'\subsection{November CNN-LSTM Baseline Comparison}',
            r'On November 25, 2026, the live workflow returned to the CNN+LSTM-only path. A fixed 60-epoch, seed-43 recipe was trained separately on the latest combined datasets: 14,672 windows for 1-M16, 14,673 for 2-M16 (including the corrected April sitting), and 7,998 for 3-M16. The historical March deployment snapshot was preserved.',
            r'The outer split uses event groups with a 250 ms purge and a nominal 20\% test allocation. A nominal 10\% of the remaining eligible training pool is allocated to calibration. Calibration preserves groups and purges by maximum window duration; auxiliary REST-only sessions stay in fitting. Saved fitting/calibration/test indices passed pairwise event and raw-signal interval separation checks. No random-window fallback or individual-window calibration rebalancing is used.',
            r'\begin{table}[t]\centering\small',
            r'\caption{November CNN+LSTM results (\%). Unsmoothed calibrated predictions; each subject has its own event holdout.}',
            r'\label{tab:november-cnn}',
            r'\begin{tabular}{lrrrr}\toprule Subject & Action & Finger & Joint & REST \\\midrule']
        for row in rows:
            tex.append(row['subject'] + ' & ' + ' & '.join(f'{row[k] * 100:.2f}' for k in metrics[:4]) + r' \\')
        tex += [r'\bottomrule\end{tabular}\end{table}',
            r'Finger accuracy conditions on true movement and counts predicted REST as an error. These are classification metrics, not gated-command or hardware results. The dataset and outer test indices for 1-M16/3-M16 match their earlier initial comparison, but calibration and fitting allocation changed. The 2-M16 expanded corpus uses a different holdout and cannot be compared as a controlled delta to the March checkpoint.',
            r'This single-seed retrospective comparison does not establish unseen-session transfer. Historical 2-M16 REST exclusions remain, 1-M16 has sparse movement-session REST evidence, and 3-M16 finger coverage is confounded with source session. Structural data vetting does not establish physiological specificity. Ensemble results below are retained as archived exploratory work, not the current runtime method.']
        (paper / 'finalization_20261125.tex').write_text('\n\n'.join(tex) + '\n')


if __name__ == '__main__':
    main()
