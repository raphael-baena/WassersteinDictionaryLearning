#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:07:14 2020

@author: raphaelbaena
"""


#### WDL CODE ###
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import scipy.optimize as sopt


class WDL:

    def __init__(self,**kwargs):
        self.set_defaults()
        self.set_pars(**kwargs)
   
   
    def set_defaults(self):
        self.def_tau = 0
        self.def_gamma = 7.
        self.def_n_iter_sink = 100
        self.def_factr = 1e7
        self.def_pgtol = 1e-5
        self.def_varscale = 100
        self.def_unbalanced =False 
       
        self.def_rho = 20 #float('inf')
        self.def_loss = 'L2'
        self.def_maxiter =2
        
        self.def_Cost = None
        self.def_feat_0 = None 
        self.def_feat_init="random"
        self.def_wgt_0 = None
        self.def_wgt_init = "uniform"
        self.def_n_process = 1
        
        self.def_Verbose = False
        self.def_savepath=''
        self.def_logpath=''
        self.def_checkplots = False,

   
    def set_pars(self,**kwargs):
        self.tau = kwargs.get('tau', self.def_tau)
        self.gamma = tf.constant(kwargs.get('gamma', self.def_gamma),dtype=tf.float64)
        self.n_iter_sink = tf.constant(kwargs.get('n_iter_sink',self.def_n_iter_sink ),dtype=tf.int32)
        self.factr = tf.constant(kwargs.get('factr',  self.def_factr))
        self.pgtol = tf.constant(kwargs.get('pgtol', self.def_pgtol))
        self.varscale = kwargs.get('varscale',self.def_varscale )
        self.unbalanced = kwargs.get('unbalanced', self.def_unbalanced )
        self.rho =  tf.constant(kwargs.get('rho', self.def_rho))
        self.loss = kwargs.get('loss', self.def_loss)
        self.maxiter = tf.constant(kwargs.get("maxiter", self.def_maxiter))
        self.feat_0 = kwargs.get("feat_0", self.def_feat_0)
        self.feat_init = kwargs.get("feat_init", self.def_feat_init)
        
        self.wgt_0 =  kwargs.get("wgt_0", self.def_wgt_0 )
        self.wgt_init =  kwargs.get("wgt_init", self.def_wgt_init )
        
        self.n_process =  kwargs.get("n_process", self.def_n_process )
    
        self.Verbose = kwargs.get("Verbose", self.def_Verbose )
        self.savepath= kwargs.get("savepath", self.def_savepath)
        self.logpath= kwargs.get("logpath", self.def_logpath)
        self.checkplots = kwargs.get("checkplots", self.def_checkplots )

        
        self.rho = kwargs.get('rho', self.def_rho )
        
        self.Datapoint = tf.convert_to_tensor(kwargs.get('Datapoint'))
        
        if kwargs.get('Cost') is None :
            n, p = self.Datapoint.shape   
            self.Cost  = tf.convert_to_tensor(EuclidCost(int(np.sqrt(p)), int(np.sqrt(p)), 
                                                     divmed=False, timeit=True))
        
        else :
             self.Cost = tf.convert_to_tensor(kwargs.get('Cost'),dtype=tf.float64)
            
        self.Ker = tf.math.exp(-self.Cost/self.gamma)
        self.n_components = tf.constant(kwargs.get('n_components'),dtype =tf.int32)



    
    
    def LBFGSDescent(self):

        n, p = self.Datapoint.shape

        if self.feat_0 is None:
            if self.feat_init == "sampled":
                Ys = np.empty((self.n_components, p))
                Ys[:] = self.Datapoint[np.random.randint(0, n, self.n_components), :]
            elif self.feat_init == "uniform":
                Ys = np.ones((self.n_components, p)) / p
            elif self.feat_init == "random":
                Ys = np.random.rand(self.n_components, p)
                Ys = (Ys.T / np.sum(Ys, axis = 1)).T
        else:
            Ys = self.feat_0
        if self.wgt_0 is None:
            if self.wgt_init == "uniform":
                w = tf.divide(tf.ones((n, self.n_components)) ,
                              tf.cast(self.n_components,dtype=tf.float32))
            elif self.wgt_init == "random":
                w = np.random.rand(n, self.n_components)
                
                w = (w.T / np.sum(w, axis=1)).T
        else:
            w = self.wgt_0
        w = tf.cast(w,tf.float64)
        
        dicw0 = tf.reshape(log10(tf.concat([tf.transpose(Ys),w],axis=0)),[-1])
        #dicw0 = tf.concat([self.Datapoint[0],self.Datapoint[1]],axis=0)
        #dicw0= tf.reshape(log10(tf.concat([tf.transpose(dicw0),w],axis =0)),[-1])
        #err ,fullgrad = self.LBFGSFunc(dicw0)
        #dic = tfp.optimizer.lbfgs_minimize(self.LBFGSFunc, initial_position = dicw0, max_iterations=15000,parallel_iterations=1)
        dic= lbfgs(self.func, dicw0.numpy(),factr=10, pgtol=1e-10, maxiter=30)

        return   dic, dicw0
    

 #x, f, dic = tfp.optimizer.lbfgs_minimize(self.LBFGSFunc, dicw0, max_iterations=self.maxiter)





    @tf.function
    def varchange(self,newvar):

      return tf.transpose(tf.transpose(tf.math.exp(newvar))/tf.math.reduce_sum(tf.math.exp(newvar),axis=-1))
     
    @tf.function
    def unwrap_rep(self,dicweights, datashape):
        n,p = datashape
        Ys, w = dicweights[:p,:], dicweights[p:,:]
        return Ys, w
    
    @tf.function
    def Loss_func(self,datapoint,bary): 
        return  tf.nn.l2_loss(tf.math.subtract(datapoint,bary))

    
    @tf.function
    def condition(self,i,*args):
        return tf.less(i, self.n_iter_sink)

 

    @tf.function
    def wass_grad(self,arg):
        Ys,wi = arg
        Newvar_D, Newvar_lbda =Ys,wi
        p =  self.sinkhorn_algo(Newvar_D, Newvar_lbda)
  
        return  p
    @tf.function

    def StabKer(self, alpha, beta):
      #tf.expand_dims( tf.ones((10,20)) ,axis=-1)
#tf.expand_dims( tf.ones((10,20)) ,axis=1)
#tf.expand_dims( tf.ones((10,20)) ,axis=0)
        M = tf.expand_dims(-self.Cost,axis = -1) + tf.expand_dims(alpha, axis = 1) + tf.expand_dims(beta, axis = 0) 
        M = tf.math.exp(M / self.gamma)
        return M

# Log Sinkhorn iteration
    def log_sinkhorn_step(self,i,alpha, beta, logp, logD, lbda, Epsilon=1e-100):
          M = self.StabKer(alpha,beta)
          #M = tf.clip_by_value(M, clip_value_min=tf.cast(Epsilon,dtype=tf.float64), clip_value_max=1e7)
          
          newalpha = self.gamma * (logD - tf.math.log(tf.math.reduce_sum(M,axis=1) + Epsilon)) + alpha
        
          alpha = self.tau*alpha + (1.-self.tau)*newalpha
          
          M = self.StabKer(alpha,beta)
          #M = tf.clip_by_value(M, clip_value_min=tf.cast(Epsilon,dtype=tf.float64), clip_value_max=1e7)
          lKta = tf.math.log(tf.math.reduce_sum(M, axis=0) + Epsilon) - beta/self.gamma
          logp = tf.math.reduce_sum(lbda*lKta, axis=1)
          newbeta = self.gamma * (tf.expand_dims(logp,-1) - lKta)

          beta = self.tau*beta + (1.-self.tau)*newbeta
          return i+1,alpha, beta, logp ,logD, lbda


    @tf.function
    def sinkhorn_step(self,i,a,b,p,Newvar_D,Newvar_lbda):
      newa = Newvar_D/tf.matmul(self.Ker,b)
      a = a**self.tau * newa**(1.-self.tau)
      p =  tf.math.reduce_prod(tf.matmul(tf.transpose(self.Ker),a)**Newvar_lbda, axis=1)
      newb = tf.reshape(p,(p.shape[0],1))/tf.linalg.matmul(tf.transpose(self.Ker),a)
      b = b**self.tau * newb**(1.-self.tau)
      return i+1,a,b,p,Newvar_D, Newvar_lbda


    def unbal_sinkhorn_step(self,i,a,b,p,D,lbda):
      newa = (D/tf.matmul(self.Ker,b))**(self.rho / (self.rho+self.gamma))
      a = a**self.tau * newa**(1.-self.tau)
      #p = T.prod(T.dot(Ker.T,a)**lbda, axis=1)
      p = tf.math.reduce_sum(tf.matmul(tf.transpose(self.Ker),a)**(self.gamma/(self.gamma+self.rho))*lbda, axis=1)**((self.rho+self.gamma)/self.gamma)
     # p = T.sum(T.dot(self.Ker.T,a)**(self.gamma/(self.gamma+self.rho))*lbda, axis=1)**((self.rho+self.gamma)/self.gamma)
      newb = tf.reshape(p,(p.shape[0],1))/tf.linalg.matmul(tf.transpose(self.Ker),a)
      newb = newb**(self.rho / (self.rho+self.gamma))
      b = b**self.tau * newb**(1.-self.tau)
      return i+1,a,b,p,D,lbda


    @tf.function
    def sinkhorn_algo(self,Newvar_D, Newvar_lbda):
      a,b,p =tf.ones_like(Newvar_D),tf.ones_like(Newvar_D),tf.ones_like(Newvar_D[:,0])
      i= 0
      def return_False(i,a ,b,p,Newvar_D, Newvar_lbda):
          return i<self.n_iter_sink
      
      i,a,b,p,Newvar_D, Newvar_lbda= tf.while_loop(return_False,self.log_sinkhorn_step,loop_vars=[i,a,b,p,tf.math.log(Newvar_D), Newvar_lbda])
      #tf.while_loop(return_False,self.unbal_sinkhorn_step,loop_vars=[i,a,b,p,Newvar_D, Newvar_lbda])
      return p

    def Loss(self,Newvar_D, Newvar_lbda):
      arg=Newvar_D, Newvar_lbda
      p =  self.wass_grad(arg)
      p=tf.math.exp(p)
      #p=p/tf.math.reduce_sum(p)
      return tf.norm( self.datapoint-p, ord='euclidean')**2*1/2*1000 #tf.math.reduce_sum(p*tf.math.log(p/self.datapoint - p + self.datapoint))#tf.math.reduce_sum((self.datapoint-p)**2)*1/2

    def varchange_wass_grad(self,arg):
        self.datapoint,wi = arg
        Newvar_D, Newvar_lbda = tf.transpose(self.varchange(tf.transpose(self.Ys))),self.varchange(wi)
        
        Loss = self.Loss(Newvar_D, Newvar_lbda)
        varchange_Grads = tf.gradients([ Loss], [Newvar_D, Newvar_lbda])


        return [Loss]+varchange_Grads    


#tf.vectorized_map(self.varchange_Theano_wass_grad,(self.Datapoint,YS,w))
    @tf.function
    def LBFGSFunc(self,dicweights):
        n, p = self.Datapoint.shape
        dicweights = tf.reshape(tf.cast(dicweights,dtype = tf.float64),(n+p,self.n_components))
        Ys,w = self.unwrap_rep(dicweights, (n,p))
        self.Ys = Ys
        err = 0
        x,y = dicweights.shape
        if self.unbalanced:
            print("UNBALANCED")
        else:
            if self.loss=='L2':
                this_err,Vgrad,Vgraw = tf.map_fn(self.varchange_wass_grad,(self.Datapoint,w),dtype =[tf.float64,tf.float64,tf.float64])
                grad = tf.cast(tf.math.reduce_sum(Vgrad,axis=0),tf.float64)
    
                Vgraw= tf.reshape(Vgraw,(n,self.n_components))*self.varscale
                fullgrad = tf.concat((grad,tf.cast(Vgraw,tf.float64)),axis =0)
                err = tf.math.reduce_sum(this_err)
                
  
        tf.print(err)
        return err ,tf.reshape(fullgrad,[-1])

    def func(self,x):
      return [vv.numpy().astype(np.float64)  for vv in self.LBFGSFunc(tf.constant(x, dtype=tf.float32))]


@tf.function
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator #/ denominator


@tf.function
def EuclidCost(Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
    N = Nr * Nc
    C = np.zeros((N,N))
    for k1 in range(N):
        for k2 in range(k1):
            r1, r2 = int(float(k1) / Nc)+1, int(float(k2) / Nc)+1
            c1, c2 = k1%Nc + 1, k2%Nc + 1
            C[k1, k2] = (r1-r2)**2 + (c1-c2)**2
            C[k2, k1] = C[k1, k2]
    if divmed:
        C /= np.median(C)
    if trunc:
        C[C>maxtol] = truncval
    return C



       
def plot_func(flatarr, wind=False, savepath='', cmap='gist_stern'):
    sqrtN = int(np.sqrt(flatarr.shape[0]))
    if not wind:
        plt.imshow(flatarr.reshape(sqrtN,sqrtN), cmap=cmap, 
                   interpolation='Nearest')
    else:
        vmin, vmax = wind
        plt.imshow(flatarr.reshape(sqrtN,sqrtN), cmap=cmap, 
                   interpolation='Nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
    
        
def alphatolbda(alpha):
    return (np.exp(alpha).T / np.sum(np.exp(alpha), axis=1)).T