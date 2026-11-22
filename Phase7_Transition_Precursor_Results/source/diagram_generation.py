"""Precise architecture diagrams built from vector primitives, not generated imagery."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def diagram(root,name,title,nodes,edges):
    fig,ax=plt.subplots(figsize=(12,8));ax.set(xlim=(0,12),ylim=(0,8));ax.axis('off');ax.set_title(title,fontsize=17,pad=15)
    for key,(x,y,label) in nodes.items():
        ax.add_patch(FancyBboxPatch((x-1.55,y-.45),3.1,.9,boxstyle='round,pad=.12',facecolor='#edf5f8',edgecolor='#24566c',zorder=2));ax.text(x,y,label,ha='center',va='center',fontsize=10,zorder=3)
    for a,b,label in edges:
        x,y,_=nodes[a];u,v,_=nodes[b];ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=15,shrinkA=38,shrinkB=38,color='#426878',connectionstyle='arc3,rad=0.03'))
        if label:ax.text((x+u)/2,(y+v)/2,label,fontsize=9,ha='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
    fig.savefig(root/f'diagrams/{name}.png',dpi=200,bbox_inches='tight');fig.savefig(root/f'diagrams/{name}.pdf',bbox_inches='tight');plt.close(fig)
    (root/f'diagrams/{name}.json').write_text(json.dumps({'nodes':nodes,'edges':edges},indent=2),encoding='utf8')

def generate(root,repo):
    diagram(root,'01_project','Complete research architecture',{'a':(2,7,'12-station hourly data'),'b':(6,7,'Training-only preprocessing'),'c':(10,7,'24 x 12 x 11 histories'),'d':(3,4.8,'PCA: flatten and project'),'e':(9,4.8,'HoSVD: mode projections'),'f':(6,3,'Linear / RBF / RFF SVM'),'g':(2,1.2,'Markov probabilities'),'h':(7,1.2,'F3 + precursor change risk')},[('a','b',''),('b','c',''),('c','d',''),('c','e',''),('d','f',''),('e','f',''),('f','h',''),('g','h','')])
    diagram(root,'02_phase7','Implemented precursor fusion',{'a':(6,7,'Physical PM2.5 histories'),'b':(2,5,'Threshold distances + state'),'c':(6,5,'Target trends: 3 h and 6 h'),'d':(10,5,'11 other-station summaries'),'e':(6,3.1,'Scaled logistic change risk'),'f':(2,1,'Original F3 q and destination'),'g':(8,1,'Convex q fusion + probabilities')},[('a','b',''),('a','c',''),('a','d',''),('b','e',''),('c','e',''),('d','e','E5/E6/E7'),('e','g',''),('f','g','')])
    P=pd.read_csv(repo/'Air_Pollution_Phase3_Results/phase3_output/markov_transition_matrix.csv').filter(regex='^next_').to_numpy()
    fig,ax=plt.subplots(figsize=(9,7));ax.axis('off');pos=[(.2,.25),(.5,.8),(.8,.25)];names=['Low','Moderate','High']
    for i,(x,y) in enumerate(pos):ax.text(x,y,f'{names[i]}\nP(stay)={P[i,i]:.6f}',ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#e5f2f5',ec='#24566c'))
    for i in range(3):
        for j in range(3):
            if i!=j:
                x,y=pos[i];u,v=pos[j];ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.22',shrinkA=55,shrinkB=55));dx,dy=u-x,v-y
                ax.text((x+u)/2+dy*.13,(y+v)/2-dx*.13,f'{P[i,j]:.6f}',ha='center',bbox=dict(fc='white',ec='none'))
    ax.set_title('Verified training Markov transition matrix',fontsize=16);fig.savefig(root/'diagrams/03_markov.png',dpi=200,bbox_inches='tight');fig.savefig(root/'diagrams/03_markov.pdf',bbox_inches='tight');plt.close(fig);pd.DataFrame(P).to_csv(root/'diagrams/03_markov_probabilities.csv',index=False)
    diagram(root,'04_compression','PCA and tensor-mode compression',{'a':(6,7,'History tensor: 24 x 12 x 11'),'b':(3,5,'Flatten to 3,168 coordinates'),'c':(9,5,'Unfold time / station / channel'),'d':(3,3,'Project on k principal axes'),'e':(9,3,'Three mode matrices'),'f':(3,1,'k-dimensional feature vector'),'g':(9,1,'Core: r_time x r_station x r_feature')},[('a','b',''),('a','c',''),('b','d',''),('c','e',''),('d','f',''),('e','g','')])
    diagram(root,'05_decision','Transition prediction and missing-state fallback',{'a':(6,7,'Observed current state available?'),'b':(2,5,'No: feature-only SVM'),'c':(9,5,'Yes: q fusion + F3 destination'),'d':(6,3,'Argmax probabilities'),'e':(10,3,'E6: compare q with theta'),'f':(6,1,'Next-hour state + probabilities')},[('a','b','No'),('a','c','Yes'),('c','d','E3/E4/E5/E7'),('c','e','E6'),('b','f',''),('d','f',''),('e','f','')])
    diagram(root,'06_contribution','Research contribution and evidence',{'a':(2,6.5,'Markov persistence misses changes'),'b':(7,6.5,'Phase 6 SVM + Markov fusion'),'c':(3,4,'Interpretable precursor risk'),'d':(9,4,'Controlled E0-E7 experiments'),'e':(3,1.5,'Validation lock + independent audit'),'f':(9,1.5,'Recall gains and stability losses')},[('a','b',''),('b','c',''),('c','d',''),('d','e',''),('e','f','')])
