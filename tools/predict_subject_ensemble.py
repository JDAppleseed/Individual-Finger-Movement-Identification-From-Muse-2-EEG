#!/usr/bin/env python3
"""Run a saved offline subject ensemble on extracted EEG windows."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.subject_ensemble import predict_subject_ensemble


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--npz',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=parser.parse_args()
    with np.load(args.npz,allow_pickle=False) as data:X=data['X']
    action,finger=predict_subject_ensemble(X,args.manifest,device=args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,action_probs=action,finger_probs=finger)
    print(f'Saved {len(X)} predictions to {args.out}')
if __name__=='__main__':main()
