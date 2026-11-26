"""Generate manuscript figures directly from frozen comparison artifacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper_figures'; DATA=ROOT/'docs/research/finalization_20261125'

def main():
 rows=json.loads((DATA/'comparison.json').read_text())['subjects']
 plt.rcParams.update({'font.size':11,'savefig.dpi':220})
 fig,ax=plt.subplots(figsize=(12,4.6));ax.set(xlim=(0,12),ylim=(0,4.6));ax.axis('off')
 boxes=[(.15,1.65,1.35,1.2,'EEG window\n64 samples × 4\n256 Hz'),(1.85,1.65,1.55,1.2,'Preprocessing\nCenter + detrend\nFitting-only z-score'),(3.75,1.65,1.5,1.2,'Conv1D block\n4 → 16, kernel 5\n64 time steps'),(5.6,1.65,1.5,1.2,'Conv1D block\n16 → 32, kernel 3\n64 time steps'),(7.45,1.65,1.55,1.2,'LSTM\n32 → 64 hidden\nLast time step')]
 for x,y,w,h,label in boxes:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor='#e6eef7',edgecolor='#3465a4'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
 for a,b in zip(boxes,boxes[1:]):ax.annotate('',(b[0]-.08,2.25),(a[0]+a[2]+.08,2.25),arrowprops={'arrowstyle':'->','lw':1.5})
 for y,label in [(3.1,'Action: 64 → 3\nREST / OPEN / CLOSE'),(1.7,'Finger: 64 → 5\nMovement windows only'),(.3,'Applicability: 64 → 1\nMovement vs. REST')]:
  ax.add_patch(FancyBboxPatch((9.65,y),2.1,.9,boxstyle='round,pad=0.05',facecolor='#e6f1ec',edgecolor='#39785d'));ax.text(10.7,y+.45,label,ha='center',va='center',fontsize=10)
  ax.annotate('',(9.55,y+.45),(9.1,2.25),arrowprops={'arrowstyle':'->','lw':1.3})
 ax.text(5.4,.65,'Each convolution: same padding → GroupNorm (4 groups) → ReLU → dropout (0.3)',ha='center',fontsize=10)
 ax.text(5.4,.15,'Shared last-step features use dropout (0.3); dropout is disabled for deterministic evaluation.',ha='center',fontsize=10)
 fig.tight_layout();fig.savefig(OUT/'cnn_lstm_architecture.png',bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4.1))
 for ax,key,fullkey,title in zip(axes,['joint_accuracy','rest_recall'],['joint_acc','rest_tpr'],['Joint action–finger accuracy','REST recall']):
  x=np.arange(3); full=[json.loads((DATA/'full_corpus'/f'{r["subject"]}_november.json').read_text())['metrics'][fullkey]*100 for r in rows]
  for offset,values,label,color in [(-.18,[r[key]*100 for r in rows],'Saved holdout','#3465a4'),(.18,full,'Full corpus (includes fitting)','#be7c3b')]:
   bars=ax.bar(x+offset,values,.34,label=label,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
  ax.set(xticks=x,xticklabels=[r['subject'] for r in rows],ylim=(0,110),ylabel='Percent',title=title);ax.spines[['top','right']].set_visible(False)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=(0,.1,1,1));fig.savefig(OUT/'holdout_full_corpus_comparison.png');plt.close(fig)
 matrix=np.array([[r['per_finger_recall'][str(k)]*100 for k in range(1,6)] for r in rows])
 fig,ax=plt.subplots(figsize=(8,3));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
 ax.set(xticks=np.arange(5),xticklabels=['Thumb','Index','Middle','Ring','Pinky'],yticks=np.arange(3),yticklabels=[r['subject'] for r in rows])
 for i in range(3):
  for j in range(5):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>65 else 'black')
 fig.colorbar(im,ax=ax,label='Recall (%)');fig.tight_layout();fig.savefig(OUT/'subject_finger_comparison.png');plt.close(fig)
if __name__=='__main__':main()
