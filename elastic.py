import tensorflow as tf
import numpy as np
import math
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import h5py

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


class elastic_part_update:

    def __init__(self,iGgsigma0,  G, SIGMA00 , SIGMADEL0, g ):
        iGgsigma0tf: 
        Gtf:
        SIGMA00tf:
        SIGMADEL0tf:
        gtf:


    def assigndeltac(nr,c,N):
        deltack=tf.Variable(tf.zeros((N,N,N),dtype=tf.complex64))
        deltacr=tf.Variable(tf.zeros((N,N,N),dtype=tf.complex64))
        deltack2=tf.Variable(tf.zeros((N,N,N),dtype=tf.complex64))
        assigndelt=tf.group(deltack.assign(tf.fft3d(nr-c)),
                            deltacr.assign(nr-c),
                            deltack2.assign(tf.fft3d((nr-c)**2)))
        return assigndelt, deltack, deltacr, deltack2

    #iGgsigma0tf=tf.placeholder("complex64",shape=(3,N,N,N))
    def first_order_disp(iGgsigma0tf,deltack,N):

        # calculate first order approximation of displacement  A PHASE-FIELD MODEL FOR EVOLVING MICROSTRUCTURES WITH STRONG ELASTIC INHOMOGENEITY
        v_0=tf.Variable(tf.zeros((3,N,N,N),dtype=tf.complex64))
        v_0assignstep=tf.group(
                v_0[0,:,:,:].assign(tf.squeeze(iGgsigma0tf[0,:,:,:])*deltack),
                v_0[1,:,:,:].assign(tf.squeeze(iGgsigma0tf[1,:,:,:])*deltack),
                v_0[2,:,:,:].assign(tf.squeeze(iGgsigma0tf[2,:,:,:])*deltack)
                )
        return v_0, v_0assignstep

    def higherorderdisp(v_0,Gtf,SIGMA00tf,SIGMADEL0tf,N,gtf,deltacr,lambdadel,deltack,deltack2):
        du_0=tf.Variable(tf.zeros((3,3,N,N,N),dtype=tf.complex64))
        for l in range(3):
            for m in range(3):
                with tf.control_dependencies([du_0[l,m,:,:,:].assign(tf.ifft3d(1j*tf.squeeze(v_0[l,:,:,:])*tf.squeeze(gtf[m,:,:,:]))),\
                    du_0[l,m,:,:,:].assign(tf.fft3d(deltacr*tf.squeeze(du_0[l,m,:,:,:])))]):
                    du_0=tf.identity(du_0)
        
        delcdu_0=tf.Variable(tf.zeros((6,N,N,N),dtype=tf.complex64))
        with tf.control_dependencies(
        [delcdu_0[0,:,:,:].assign(du_0[0,0,:,:,:]),
        delcdu_0[1,:,:,:].assign(du_0[1,1,:,:,:]),
        delcdu_0[2,:,:,:].assign(du_0[2,2,:,:,:]),
        delcdu_0[3,:,:,:].assign(du_0[0,1,:,:,:]+du_0[1,0,:,:,:]),
        delcdu_0[4,:,:,:].assign(du_0[0,2,:,:,:]+du_0[2,0,:,:,:]),
        delcdu_0[5,:,:,:].assign(du_0[1,2,:,:,:]+du_0[2,1,:,:,:]),]):
            delcdu_0=tf.identity(delcdu_0)
        
        sigma3=tf.reshape(tf.matmul(lambdadel,tf.reshape(delcdu_0,[6,-1])),[6,N,N,N]) # third term in V_k1(g) expression in square paracenes
        
        SIGMA3=tf.Variable(tf.zeros((3,3,N,N,N),dtype=tf.complex64))
        with tf.control_dependencies(
        [SIGMA3[0,0,:,:,:].assign(sigma3[0,:,:,:]),
        SIGMA3[1,1,:,:,:].assign(sigma3[1,:,:,:]),
        SIGMA3[2,2,:,:,:].assign(sigma3[2,:,:,:]),
        SIGMA3[0,1,:,:,:].assign(sigma3[3,:,:,:]),
        SIGMA3[1,0,:,:,:].assign(sigma3[3,:,:,:]),
        SIGMA3[0,2,:,:,:].assign(sigma3[4,:,:,:]),
        SIGMA3[2,0,:,:,:].assign(sigma3[4,:,:,:]),
        SIGMA3[1,2,:,:,:].assign(sigma3[5,:,:,:]),
        SIGMA3[2,1,:,:,:].assign(sigma3[5,:,:,:])]):
            SIGMA3=tf.identity(SIGMA3)

        SIGMAtotal=tf.Variable(tf.zeros((3,3,N,N,N),dtype=tf.complex64))

        for l in range(3):
            for m in range(3):
                with tf.control_dependencies([SIGMAtotal[l,m,:,:,:].assign(tf.squeeze(SIGMA00tf[l,m,:,:,:])*deltack+squeeze(SIGMADEL0tf[l,m,:,:,:])*deltack2-tf.squeeze(SIGMA3[l,m,:,:,:]))]):
                    SIGMAtotal=tf.identity(SIGMAtotal)
        
        
        gtfexp=tf.expand_dims(gtf,1)
        u_1=-1j*tf.squeeze(tf.matmul(tf.transpose(Gtf,perm=[1,0,2,3,4]),tf.matmul(SIGMAtotal,gtfexp)))
        
        dif=tf.abs(u_1-v_0)
        dif=tf.reduce_sum(dif)
        assignup=v_0.assign(u_1)
        return u_1,dif,assignup
            
        

            
        


    def high_order_disp_assign_uel(u_1,v_0,dif,assignup,sess,gtf,N):  
    
        for i in range(100):
            difval=sess.run(dif,feed_dict={iGgsigma0tf: iGgsigma0, Gtf: G, SIGMA00tf:SIGMA00 , SIGMADEL0tf:SIGMADEL0, gtf:g })
            if np.abs(difval)<0.0001:
                break
            else:
                sess.run(assignup)
    
    # Calculate delta epsilon
        deltaep=tf.Variable(tf.zeros((3,3,N,N,N),dtype=tf.complex64)) # deltaepsilon(r) in real space
        gtf=tf.placeholder("complex64",shape=(3,N,N,N))
    
        for i in range(3):
            for j in range(3):
                with tf.control_dependencies([deltaep[i,j,:,:,:].assign(tf.ifft3d(tf.squeeze(1j/2*tf.squeeze(u_1[i,:,:,:]*gtf[j,:,:,:]+u_1[j,:,:,:]*gtf[i,:,:,:]))))]):
                    deltaep=tf.identity(deltaep)
            
        deltaep_0=tf.Variable(tf.zeros((6,N,N,N),dtype=tf.complex64))
        with tf.control_dependencies(
        [deltaep_0[0,:,:,:].assign(deltaep[0,0,:,:,:]),
        deltaep_0[1,:,:,:].assign(deltaep[1,1,:,:,:]),
        deltaep_0[2,:,:,:].assign(deltaep[2,2,:,:,:]),
        deltaep_0[3,:,:,:].assign(2*deltaep[0,1,:,:,:]),
        deltaep_0[4,:,:,:].assign(2*deltaep[0,2,:,:,:]),
        deltaep_0[5,:,:,:].assign(2*deltaep[1,2,:,:,:])]):
            deltaep_0=tf.identity(deltaep_0)
        
        SIGMADELDEL=tf.reshape(tf.matmul(lambdadel,tf.reshape(deltaep_0,[6,-1])),[6,N,N,N])
        
        uel=tf.fft3d(-tf.reduce_sum(SIGMA00*deltaep,[0,1])+tf.reduce_sum(SIGMA00*epsilon00,[0,1])*deltacr+1.0/2*tf.reduce_sum(SIGMADELDEL*deltaep_0,[0])\
        -2*tf.reduce_sum(SIGMADEL0*deltaep,[0,1])*deltacr+3/2*tf.reduce_sum(SIGMADEL0*epsilon00,[0,1])*deltacr**2)

        return uel

    

  

  