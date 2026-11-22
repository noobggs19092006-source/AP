"""Independent scalar-count implementation; no implementation-side imports."""
import numpy as np

def calculate(df,scope):
    if scope=='A':df=df[df.current_state>=0]
    y=df.y.to_numpy(int);pred=df.pred.to_numpy(int);s=df.current_state.to_numpy(int)
    div=lambda a,b:a/b if b else 0.
    f=[]
    for k in range(3):
        t=sum((y==k)&(pred==k));fp=sum((y!=k)&(pred==k));fn=sum((y==k)&(pred!=k));f.append(div(2*t,2*t+fp+fn))
    v=s>=0;a=y!=s;b=pred!=s
    tp=sum(v&a&b);fp=sum(v&~a&b);fn=sum(v&a&~b);tn=sum(v&~a&~b)
    out=dict(N=len(df),accuracy=float(np.mean(y==pred)),macro_f1=float(np.mean(f)),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),change_precision=div(tp,tp+fp),change_recall=div(tp,tp+fn),change_f1=div(2*tp,2*tp+fp+fn),exact_destination=div(sum(v&a&(y==pred)),sum(v&a)),stable_accuracy=div(sum(v&~a&(y==pred)),sum(v&~a)))
    a=v&(s!=2)&(y==2);b=v&(s!=2)&(pred==2);t=sum(a&b);fp=sum(~a&b);fn=sum(a&~b)
    out.update(high_precision=div(t,t+fp),high_recall=div(t,t+fn),high_f1=div(2*t,2*t+fp+fn))
    p=df[['p0','p1','p2']].to_numpy();out['brier']=float(np.mean([sum((row-np.eye(3)[int(k)])**2) for row,k in zip(p,y)]))
    clipped=np.maximum(p,1e-12);clipped/=clipped.sum(1)[:,None];out['log_loss']=float(np.mean([-np.log(row[int(k)]) for row,k in zip(clipped,y)]))
    confidence=p.max(1);correct=p.argmax(1)==y;ece=0
    for k in range(10):
        m=(confidence>=k/10)&((confidence<(k+1)/10) if k<9 else (confidence<=1))
        if sum(m):ece+=sum(m)/len(y)*abs(np.mean(confidence[m])-np.mean(correct[m]))
    out['ece']=ece;return out
