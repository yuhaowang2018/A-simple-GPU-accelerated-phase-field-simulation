# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:46:42 2018

@author: yuhao_wang
"""

import tensorflow as tf
import numpy as np
import math
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import h5py

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))



''' concentration '''
cin=0.136
c=2*cin
c2=1-c
# size
N=512
print('run')
''' real space build'''
nn=N-1
x=np.array(range(0,N),dtype=np.float64)
y=np.array(range(0,N),dtype=np.float64)
z=np.array(range(0,N),dtype=np.float64)
X,Y,Z=np.meshgrid(x,y,z)
# primitive cell vector real space
a1=np.array([1.0,0.0,0.0])
a2=np.array([0.0,1.0,0.0])
a3=np.array([0.0,0.0,1.0])
# real space coordinates
XX=X*a1[0]+Y*a2[0]+Z*a3[0]
YY=X*a1[1]+Y*a2[1]+Z*a3[1]
ZZ=X*a1[2]+Y*a2[2]+Z*a3[2]

''' reciprocal space build'''
# primitive cell vector reciprocal space
b1=np.cross(a2,a3)/(np.dot(a1,np.cross(a2,a3)))
b2=np.cross(a3,a1)/(np.dot(a1,np.cross(a2,a3)))
b3=np.cross(a1,a2)/(np.dot(a1,np.cross(a2,a3)))

lk = 1 #total length k space
dk = lk/N
h=np.array(range(0,int(N/2)+1),dtype=np.float64) # vector of wavenumbers
h=np.append(h,np.array(range(1-int(N/2),0),dtype=np.float64))
h=h*dk
k=h
l=h
H,K,L = np.meshgrid(h,k,l)
# reciprocal space coordinates
HH=H*b1[0]+K*b2[0]+L*b3[0]
KK=H*b1[1]+K*b2[1]+L*b3[1]
LL=H*b1[2]+K*b2[2]+L*b3[2]

'''Initial condition '''
nr0=np.complex64(c+0.05*(np.random.rand(h.shape[0],h.shape[0],h.shape[0])-0.5))

nk0=np.complex64(np.fft.fftn(nr0))



'''Parameters'''
R=8.3144598
T=600
T0=1100
Tr1=T/T0
w1=(592+T/R+205*(c-c2)/2)
pi=tf.constant(math.pi,dtype=tf.complex64)
' ''Lk=-2 sum(L1*sin^2(1/2*k*r))'''
Lk1=0
coordl=[-1,1]
for xx in coordl:
    Lk1=Lk1-2*((np.sin(math.pi*(HH*xx)))**2)
    Lk1=Lk1-2*((np.sin(math.pi*(KK*xx)))**2)
    Lk1=Lk1-2*((np.sin(math.pi*(LL*xx)))**2)
Lk1=np.complex64(Lk1)

# fourier transformation of exchange energy
Vk1=np.complex64(2*w1*(np.cos(2*math.pi*HH)+np.cos(2*math.pi*KK)+np.cos(2*math.pi*LL))) 

Lk=tf.placeholder("complex64",shape=(N,N,N))
Vk=tf.placeholder("complex64",shape=(N,N,N))
nkplace=tf.placeholder("complex64",shape=(N,N,N))
nrplace=tf.placeholder("complex64",shape=(N,N,N))
nk=tf.Variable(nkplace)
nr=tf.Variable(nrplace)

''' Control time step'''
dt=0.0001
dtr=tf.constant(dt,dtype=tf.complex64)

Tr=tf.constant(Tr1,dtype=tf.complex64)

def elastic_constant_3d(T0,HH,KK,LL,N):
  mod=((HH)**2+(KK)**2+(LL)**2)**(0.5)
  mod[mod==0]=99999999999999999999
  n1=HH/mod
  n2=KK/mod
  n3=LL/mod
  n=np.zeros((3,N,N,N))
  n[0,:,:,:]=n1
  n[1,:,:,:]=n2
  n[2,:,:,:]=n3
  
  epsilon0=0.01
  epsilond=np.array([0.015, 0.015, 0.015, 0, 0, 0])
  lambda0=np.array([  [239, 210,  210, 0, 0, 0],    # unit gpa
          [210, 239,  210, 0, 0, 0], 
          [210, 210,  239, 0, 0, 0],
          [0,    0,    0,  179, 0, 0],
          [0,    0,    0,  0, 179, 0],
          [0,    0,    0,  0,  0, 179]])
  lambdadel=lambda0*0.1
  kB=1.38e-23
  V=(3.04e-10)**3/2 # Volume of a single atom
  lambda0=lambda0*10**9/(kB*T0)*V # unit conversion
  delta_kl=np.array([ 1, 1, 1, 0, 0, 0])
  delta=np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])
  # index shift for 3D
  voigt=np.array([[1, 4, 5],
       [4, 2, 6],
       [5, 6, 3]
       ])

  sigma0=np.zeros((1,6))
  for i in range(6):
    for k in range(6):
        sigma0[i]=sigma0[i]+lambda0[i,k]*epsilon0*delta_kl[k]

  SIGMA0=np.zeros((3,3))
  SIGMA0[0,0]=sigma0[0]
  SIGMA0[1,1]=sigma0[1]
  SIGMA0[2,2]=sigma0[2]
  SIGMA0[0,1]=sigma0[3]
  SIGMA0[1,0]=sigma0[3]
  SIGMA0[0,2]=sigma0[4]
  SIGMA0[2,0]=sigma0[4]
  SIGMA0[1,2]=sigma0[5]
  SIGMA0[2,1]=sigma0[5]

  sigmadel=np.zeros((1,6))
  for i in range(6):
      for k in range(6): 
          sigmadel[i]=sigmadel[i]+lambdadel[i,k]*epsilon0*delta_kl[k]

  SIGMADEL=np.zeros((3,3))
  SIGMADEL[0,0]=sigmadel[0]
  SIGMADEL[1,1]=sigmadel[1]
  SIGMADEL[2,2]=sigmadel[2]
  SIGMADEL[0,1]=sigmadel[3]
  SIGMADEL[1,0]=sigmadel[3]
  SIGMADEL[0,2]=sigmadel[4]
  SIGMADEL[2,0]=sigmadel[4]
  SIGMADEL[1,2]=sigmadel[5]
  SIGMADEL[2,1]=sigmadel[5]
  
  omega=np.zeros((3,3,N,N,N))
  omega2=np.zeros((3,3,N,N,N))

  for i in range(3):
    for j in range(3):
      for k in range(3):
        for l in range(3):
          omega[i,j,:,:,:]=omega[i,j,:,:,:]+(lambda0[voigt[i,k],voigt[l,j]]*n[k,:,:,:]*n[l,:,:,:])

  deter=omega[0,0,:,:,:]*(omega[1,1,:,:,:]*omega[2,2,:,:,:]-omega[1,2,:,:,:]*omega[2,1,:,:,:])-omega[0,1,:,:,:] \
     *(omega[1,0,:,:,:]*omega[2,2,:,:,:]-omega[1,2,:,:,:]*omega[2,0,:,:,:])+omega[0,2,:,:,:]*(omega[1,0,:,:,:]*omega[2,1,:,:,:]-omega[1,1,:,:,:]*omega[2,0,:,:,:])
  deter[deter==0]=999999999999999999
  deter=1.0/deter
  omega2[0,0,:,:,:]=deter*(omega[1,1,:,:,:]*omega[2,2,:,:,:]-omega[1,2,:,:,:]*omega[2,1,:,:,:])
  omega2[0,1,:,:,:]=deter*(omega[0,2,:,:,:]*omega[2,1,:,:,:]-omega[0,1,:,:,:]*omega[2,2,:,:,:])
  omega2[0,2,:,:,:]=deter*(omega[0,1,:,:,:]*omega[1,2,:,:,:]-omega[1,1,:,:,:]*omega[0,2,:,:,:])
  omega2[1,0,:,:,:]=deter*(omega[1,2,:,:,:]*omega[2,0,:,:,:]-omega[1,0,:,:,:]*omega[2,2,:,:,:])
  omega2[1,1,:,:,:]=deter*(omega[0,0,:,:,:]*omega[2,2,:,:,:]-omega[0,2,:,:,:]*omega[2,0,:,:,:])
  omega2[1,2,:,:,:]=deter*(omega[0,2,:,:,:]*omega[1,0,:,:,:]-omega[1,2,:,:,:]*omega[0,0,:,:,:])
  omega2[2,0,:,:,:]=deter*(omega[1,0,:,:,:]*omega[2,1,:,:,:]-omega[1,1,:,:,:]*omega[2,0,:,:,:])
  omega2[2,1,:,:,:]=deter*(omega[0,1,:,:,:]*omega[2,0,:,:,:]-omega[0,0,:,:,:]*omega[2,1,:,:,:])
  omega2[2,2,:,:,:]=deter*(omega[0,0,:,:,:]*omega[1,1,:,:,:]-omega[1,0,:,:,:]*omega[0,1,:,:,:])
  omega=omega2

  sigmad=np.zeros((1,6))
  for i in range(6):
      for k in range(6):
          sigmad[i]=sigmad[i]+lambda0[i,k]*epsilond[k]


  SIGMAd=np.zeros((3,3))
  SIGMAd[0,0]=sigmad[0]
  SIGMAd[1,1]=sigmad[1]
  SIGMAd[2,2]=sigmad[2]
  SIGMAd[0,1]=sigmad[3]
  SIGMAd[1,0]=sigmad[3]
  SIGMAd[0,2]=sigmad[4]
  SIGMAd[2,0]=sigmad[4]
  SIGMAd[1,2]=sigmad[5]
  SIGMAd[2,1]=sigmad[5]

  g=np.zeros((3,N,N,N))
  g[0,:,:,:]=HH
  g[1,:,:,:]=KK
  g[2,:,:,:]=LL

  mod2=np.zeros((3,3,N,N,N))
  for i in range(3):
      for j in range(3):
          mod2[i,j,:,:,:]=mod[:,:,:]

  G=omega/(mod2**2)    

  # convert epsilond from vertor to tensor
  temp=epsilond
  epsilond[0,0]=temp[0]
  epsilond[1,1]=temp[1]
  epsilond[2,2]=temp[2]
  epsilond[0,1]=temp[3]
  epsilond[1,0]=temp[3]
  epsilond[0,2]=temp[4]
  epsilond[2,0]=temp[4]
  epsilond[1,2]=temp[5]
  epsilond[2,1]=temp[5]

  iGgsigma0=np.zeros((3,N,N,N))
  for k in range(3):
    for i in range(3):
      for j in range(3):
        
        aaa=1j*np.squeeze(G[i,k,:,:,:])*np.squeeze(g[j,:,:,:])*SIGMA0[i,j] 
        iGgsigma0[k,:,:,:]=np.squeeze(iGgsigma0[k,:,:,:])-aaa # -iG_ik*g_j*sigma0_ij; zeroth order multiplier before delta_n 
      
  SIGMA00=np.zeros((3,3,N,N,N))
  for i in range(3):
      for j in range(3):
          SIGMA00[i,j,:,:,:]=SIGMA0[i,j]+SIGMA00[i,j,:,:,:]
      
  

  SIGMADEL0=np.zeros((3,3,N,N,N))
  for i in range(3):
      for j in range(3):
          SIGMADEL0[i,j,:,:,:]=SIGMADEL[i,j]+SIGMADEL0[i,j,:,:,:]


  temp=epsilon0
  epsilon00=np.zeros((3,3,N,N,N))
  epsilon00[0,0,:,:,:]=temp
  epsilon00[1,1,:,:,:]=temp
  epsilon00[2,2,:,:,:]=temp
  epsilon00[0,1,:,:,:]=0
  epsilon00[1,0,:,:,:]=0
  epsilon00[0,2,:,:,:]=0
  epsilon00[2,0,:,:,:]=0
  epsilon00[1,2,:,:,:]=0
  epsilon00[2,1,:,:,:]=0

  return SIGMA00,SIGMADEL0,omega,G,g,n,delta,epsilon00,lambda0,lambdadel,iGgsigma0


'''Update rule '''
SS=tf.log(nr/(1-nr))
Sk=tf.fft3d(SS)
dnkdt=Lk*(Vk*nk/T0+Tr*(Sk))
nk1=nk+dnkdt*dtr
nr_=tf.ifft3d(nk1)

step1=tf.group(
        nk.assign(nk1),
        nr.assign(nr_),
        )



'''Calculate order parameter '''
def make_kernel(a):
  """Transform a 3D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype="complex64")


def calc_op(nr):
    nrop=nr*tf.exp(-1j*2*pi*(1/2*X+0.5*Y+0.5*Z))
    paddings=tf.constant([[0,1],[0,1],[0,1]])
    nrpad=tf.pad(nrop,paddings,"REFLECT")
    nrpad=tf.expand_dims(tf.expand_dims(nrpad,0),-1)
    filt=np.ones((2,2,2))
    kernel=make_kernel(filt)
    xr=tf.nn.conv3d(nrpad,kernel,[1,1,1,1,1],padding="VALID")
    xr=xr/8
    return xr[0,:,:,:,0]

order_parameter=calc_op(nr)
''''''
Nt=50
time=np.arange(dt,Nt,dt)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer(), feed_dict={nkplace:nk0, nrplace:nr0})


print('finish build')
hf= h5py.File('store_n_x.h5', 'w')

for t in time:

    step1.run(session=sess,feed_dict={Lk: Lk1, Vk: Vk1})
    
    if t<=5:
     if (10*t % 1) <1e-5:
        print(t)
        
        nsetname='n_time'+str(int(10*t)/10)
        hf.create_dataset(nsetname,  data=np.real(nr.eval(session=sess)), compression="gzip", compression_opts=9)

    else:
     if (t % 1) <1e-5:
        print(t)
        nsetname='n_time'+str(int(10*t)/10)
        hf.create_dataset(nsetname,  data=np.real(nr.eval(session=sess)), compression="gzip", compression_opts=9)
        

sess.close()
hf.close()


    
             



























