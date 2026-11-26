"""Export de-identified processed EEG windows with explicit metadata allowlisting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'docs/research/finalization_20261125/datasets.json').read_text())
    manifest=[]
    for number,row in enumerate(rows,1):
        with np.load(root/row['dataset'],allow_pickle=False) as z:
            keys=['X','y_action','y_finger','fs','channel_names','window_sec','step_sec','event_id','trial_id','block_id','artifact_flag','gap_flag','gap_fraction']
            data={k:z[k] for k in keys if k in z}
            sessions=z['session_id']; unique=np.unique(sessions)
            data['subject_id']=np.full(len(z['X']),f'S{number:02d}')
            data['session_id']=np.empty(len(sessions),dtype='U16')
            data['window_start']=np.asarray(z['window_start'],dtype=float).copy()
            data['window_end']=np.asarray(z['window_end'],dtype=float).copy()
            for j,sid in enumerate(unique,1):
                mask=sessions==sid; offset=float(np.min(data['window_start'][mask]))
                data['session_id'][mask]=f'S{number:02d}-R{j:02d}'
                data['window_start'][mask]-=offset;data['window_end'][mask]-=offset
            path=out/f'S{number:02d}_windows.npz'
            if path.exists(): raise FileExistsError(path)
            np.savez_compressed(path,**data)
            with np.load(path,allow_pickle=False) as check:
                assert np.array_equal(check['X'],z['X'])
                assert np.array_equal(check['y_action'],z['y_action'])
                assert np.array_equal(check['y_finger'],z['y_finger'])
            manifest.append(dict(subject=f'S{number:02d}',file=path.name,windows=len(data['X']),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(description='De-identified processed EEG windows, not raw continuous recordings. Paths, recording dates, free-text configuration and original subject/session strings removed. Relative recording time retained. EEG signals are not guaranteed anonymous.',subjects=manifest),indent=2)+'\n')
if __name__=='__main__': main()
