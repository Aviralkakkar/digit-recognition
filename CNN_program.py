import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

imgx=cv2.cvtColor(cv2.imread(r"C:\Python36\pythonprogs\ximage.png"),cv2.COLOR_BGR2GRAY)
imgo=cv2.cvtColor(cv2.imread(r"C:\Python36\pythonprogs\oimage.png"),cv2.COLOR_BGR2GRAY)
print(imgx.shape)
print(imgo.shape)

kernel=np.array([[1,0,0],[0,1,0],[0,0,1]])
features=[]


#for image imgX
f=[]

for i in range(0,18,2):
    horz=[]
    for j in range(0,18,2):
        mat=imgx[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    f.append(horz)
print(np.shape(f))



g=[]
    
f=np.asarray(f)
for i in range(0,len(f),2):
    horz=[]
    for j in range(0,len(f),2):
        mat=f[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    g.append(horz)
print(np.shape(g))


h=[]
g=np.asarray(g)
for i in range(0,len(g),2):
    horz=[]
    for j in range(0,len(g),2):
        mat=g[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    h.append(horz)
print(np.shape(h))



h=np.asarray(h)
a=h.flatten()
a=list(a)
features.append(a)




#for image img0
print("for image o")
x=[]
for i in range(0,18,2):
    horz=[]
    for j in range(0,18,2):
        mat=imgo[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    x.append(horz)

print(np.shape(x))


y=[]
x=np.asarray(x)
    

for i in range(0,len(x),2):
    horz=[]
    for j in range(0,len(x),2):
        mat=x[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    y.append(horz)
print(np.shape(y))


z=[]
y=np.asarray(y)
        
for i in range(0,len(y),2):
    horz=[]
    for j in range(0,len(y),2):
        mat=y[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    z.append(horz)
print(np.shape(z))



z=np.asarray(z)
b=z.flatten()
b=list(b)

#features
features.append(b)
print(features)

#labels
l=['X','0']



#TESTING THE IMAGE
print("TESTING THE IMAGE")

img_test=cv2.cvtColor(cv2.imread(r"C:\Python36\pythonprogs\test_ximage2.png"),cv2.COLOR_BGR2GRAY)
print(img_test.shape)

pred=[]
p=[]


for i in range(0,18,2):
    horz=[]
    for j in range(0,18,2):
        mat=img_test[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    p.append(horz)



q=[]
p=np.asarray(p)

for i in range(0,len(p),2):
    horz=[]
    for j in range(0,len(p),2):
        mat=p[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    q.append(horz)
print(np.shape(q))


r=[]
q=np.asarray(q)
        
for i in range(0,len(q),2):
    horz=[]
    for j in range(0,len(q),2):
        mat=q[i:i+3,j:j+3]
        newmat=np.multiply(mat,kernel)
        cell=np.sum(newmat)
        horz.append(cell/3)
    
    r.append(horz)
print(np.shape(r))



r=np.asarray(r)
c=r.flatten()
c=list(c)
pred.append(c)



clf=KNeighborsClassifier(n_neighbors=1)
t=clf.fit(features,l)
s=t.predict(pred)
print(s)






