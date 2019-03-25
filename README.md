# Genetic-Algorithm
A simple model of GA written in python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
start=time.clock()
G=[]
T=[]



T1=[]
groupnum=200


data=pd.read_csv('C://data/location.csv',names=['X','Y'])
data.plot.scatter('X','Y',label='City')
plt.show()
X=data.iloc[:,0]
Y=data.iloc[:,-1]
def CreatGroup():
    Init=[0,1,2,3,4,5]
    random.shuffle(Init)
    T=Init
    Init=[0,1,2,3,4,5]
    return T
for i in range(0,groupnum):
    G.append(CreatGroup())

def distance(arr):
    D=np.zeros(groupnum)
    for i in range(0,len(arr)):
        for j in range(0,5):
            D[i]=D[i]+((X[arr[i][j]]-X[arr[i][j+1]])**2+(Y[arr[i][j]]-Y[arr[i][j+1]])**2)**0.5
        
        D[i]=D[i]+((X[arr[i][5]]-X[arr[i][0]])**2+(Y[arr[i][5]]-Y[arr[i][0]])**2)**0.5
        
    return D
D=distance(G)

def fitness(arr,t):
    F=[]
    for i in range(0,t):
        F.append(1/(arr[i]*arr[i]))
    return F
F=fitness(D,len(G))

def prob(arr,t):
    P=[]
    P.append(0)
    m=0
    for i in range(0,t):
        m=arr[i]+m
    for i in range(0,t):
        P.append(arr[i]/m)
    return P
P=prob(F,len(F))

def sum1(arr):
    
    arr2=np.zeros(len(arr))
    
    arr2[0]=arr[0]+arr[1]
    for i in range(1,len(arr)-1):
        arr2[i]=arr[i+1]+arr2[i-1]
    i=len(arr2)-1
    while(i>=0):
        arr2[i]=arr2[i-1]
        i=i-1
    arr2[0]=0
    return arr2
P=sum1(P)


def chose(arr):
    c=np.zeros(groupnum)
    a=np.zeros(groupnum)
    for i in range(0,len(arr)-1):
        a[i]=random.uniform(0,1)
        
    for i in range(0,len(arr)-1):
        for j in range(0,len(arr)-1):
            if(a[i]>=arr[j] and a[i]<arr[j+1]):
                c[i]=j
    c=[int(i) for i in c]
    return c
c=chose(P)




def newgroup(arr,arr2):
    q=np.zeros(len(arr))
    q=[int(i) for i in q]
    
    for i in range(0,len(arr)):
        q[i]=arr[arr2[i]]
    for i in range(0,len(arr)):
        arr[i]=q[i]
    
    return arr
G=newgroup(G,c)

def vir(arr,l):
    m=0
    while(m<l-1):
        
        r1=random.randint(0,5)
        r2=random.randint(0,5)
   
        l1=[]
        l2=[]
    
        tag1=np.zeros(6)
        tag2=np.zeros(6)
        while(r2<=r1):
        
            r2=random.randint(0,5)
            if(r1==5):
                r1=random.randint(0,5)
        
        for j in range(r1,r2+1):
            l1.append(arr[m][j])
        for j in range(r1,r2+1):
            l2.append(arr[m+1][j])
       
        for j in range(r1,r2+1):
            arr[m+1][j]=l1[j-r1]
        for j in range(r1,r2+1):
            arr[m][j]=l2[j-r1]
        
        i=0
        
   
        while(i!=6):
            j=i+1
            while(j!=6):
                if(arr[m][i]==arr[m][j]):
                    tag1[j]=1
                j=j+1
            i=i+1
        i=0
    
        while(i!=6):
            j=i+1
            while(j!=6):
                if(arr[m+1][i]==arr[m+1][j]):
                    tag2[j]=1
                j=j+1
            i=i+1
        
        tag1=[int(i) for i in  tag1]
        tag2=[int(i) for i in  tag2]
      
        
        for i in range(0,6):
            p=0
            j=0
            
            if(tag1[i]==1):
                while(j!=6 and p!=1):
                    if(tag2[j]==1):
                        t1=arr[m][i]
                        
                        t2=arr[m+1][j]
                        
                        arr[m][i]=t2
                        arr[m+1][j]=t1
                        tag2[j]=0
                        p=1
                    
                    j=j+1
       
        m=m+2
    return arr
G=vir(G,len(G))


'''def bian(arr):
    r1=random.randint(0,5)
    r2=random.randint(0,5)
    while(r1==r2):
        r2=random.randint(0,5)
    t=arr[0][r1]
    arr[0][r1]=arr[0][r2]
    arr[0][r2]=t
    
    r1=random.randint(0,5)
    r2=random.randint(0,5)
    while(r1==r2):
        r2=random.randint(0,5)
    t=arr[2][r1]
    arr[2][r1]=arr[2][r2]
    arr[2][r2]=t
    return arr
  '''
bestnow=max(D)
lbest=[]
lbest.append(bestnow)
ll=[]
def iters(n,arr):
    for i in range(0,n):
        D=distance(arr)
        F=fitness(D,len(arr))
        P=prob(F,len(F))
        P=sum1(P)
        c=chose(P)
        arr=newgroup(arr,c)
        arr=vir(arr,len(arr))
        D=distance(arr)
        bestnow=min(D)
        lbest.append(bestnow)
        ll.append(min(lbest))
    return arr
G=iters(200,G)

D=distance(G)


bestall=min(lbest)


    
D=D.tolist()
m=D.index(min(D))

x=range(len(lbest))
y=lbest
plt.plot(x, y, mec='r', mfc='w')
plt.show()
print(min(lbest))
end=time.clock()
print((end-start))
