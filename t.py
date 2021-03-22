def so(a,b):
    X=sorted(zip(a,b),reverse=True)
    i,j=[],[]
    for x,y in X:
        i.append(x)
        j.append(y)
        
    return i,j
def minimumTime(w,c,a,n,m):
    n=len(w)
    w,c=so(w,c)
    a=sorted(a,reverse=True)
    t=0
    if w[0]>a[0]:
        return -1
    while max(c)>0:
        for i in a:
            for j in range(n):
                if i>=w[j]:
                    if c[j]>0:
                        c[j]-=1
                        break;
        t+=1
                
                
    return t
    
T=int(input())
for i in range(T):
    N,M=list(map(int,input().split()))
    A=list(map(int,input().split()))
    W=list(map(int,input().split()))
    C=list(map(int,input().split()))
    print(minimumTime(W,C,A,M,N))

