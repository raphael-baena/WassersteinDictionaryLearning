#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:07:14 2020

@author: raphaelbaena
"""
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from functools import partial
from scipy.optimize import fmin_l_bfgs_b as lbfgs
import matplotlib.pyplot as plt

class WDL:

    def __init__(self,**kwargs):
        self.set_defaults()
        self.set_pars(**kwargs)
        
    def set_defaults(self):
        self.def_Tau = 0
        self.def_Gamma = 7.
        self.def_n_iter_sink = 20
        self.def_factr = 1e7
        self.def_pgtol = 1e-5
        self.def_varscale = 100
        self.def_unbalanced =False 
       
        self.def_rho = float('inf')
        self.def_loss = 'L2'
        self.def_maxiter =15000
        
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
        self.Tau = tf.constant(kwargs.get('Tau', self._def_Tau))
        self.Gamma = tf.constant(kwargs.get('Gamma', self._def_Gamma))
        self.n_iter_sink = tf.constant(kwargs.get('n_iter_sink',self.def_n_iter_sink ))
        self.factr = tf.constant(kwargs.get('factr',  self.def_factr))
        self.pgtol = tf.constant(kwargs.get('pgtol', self.def_pgtol))
        self.varscale = tf.constant( kwargs.get('varscale',self.def_varscale ))
        self.unbalanced = kwargs.get('unbalanced', self.def_unbalanced )
        self.rho =  tf.constant(kwargs.get('rho', self.def_rho))
        self.loss = kwargs.get('loss', self.def_loss)
        self.maxiter = tf.constant(kwargs.get("maxiter", self.def_maxiter))
        self.feat_0 = kwargs.get("feat_0", self.def_feat_0)
        self.feat_init = kwargs.get("feat_init", self.def_feat_init)
        
        self.wgt_0 =  kwargs.get("wgt_0 ", self.def_wgt_0 )
        self.wgt_init =  kwargs.get("wgt_init ", self.def_wgt_init )
        
        self.n_process =  kwargs.get("n_process ", self.n_process )
    
        self.def_Verbose = kwargs.get("Verbose ", self.Verbose )
        self.def_savepath= kwargs.get("savepath ", self.savepath)
        self.def_logpath= kwargs.get("logpath= ", self.logpath)
        self.def_checkplots = kwargs.get("checkplots ", self.checkplots )

        
        self.rho = kwargs.get('rho', self.def_rho )
        
        self.Datapoint = tf.Variable(kwargs.get('DataPoint'))
        
        if kwargs.get('Cost') is None :
            n, p = self.DataPoint.shape   
            self.Cost  = tf.Variable(self.EuclidCost(int(np.sqrt(p)), int(np.sqrt(p)), 
                                                     ivmed=False, timeit=True))
        
        else :
             self.Cost = tf.Variable(kwargs.get('Cost'))
            
        self.Ker = tf.Variable(tf.math.exp(self.Cost/self.Gamma))
        self.n_components = tf.Variable(kwargs.get('n_components'))



    @tf.function
    def LBFGSDescent(self):
        #self,X, n_components, rho=float('inf'), varscale=100, unbalanced=False, loss='L2',
             #   feat_0=None, feat_init="random", wgt_0=None, wgt_init="uniform",
#                n_process=1, Verbose=False, savepath='', logpath='', checkplots=False,
             #   factr=1e7, pgtol=1e-05, maxiter=15000
        """Compute Wasserstein dictionary and weights.
    
            Parameters
            ----------
            X : np.ndarray
                Input data to learn the representation on.
                
            n_components : int
                Number of desired atoms in the representation.
                
            gamma : float
                Entropic penalty parameter.
                
            n_iter_sinkhorn : int
                Number of 'Sinkhorn' iterations for barycenter computation.
                
            C : np.ndarray
                Cost function for the ground metric. If None, assumes data are square images
                flattened in np.flatten() convention and computes naturel Euclidean pixel distances.
                
            tau : float (<=0)
                Heavyball parameter (see section 4.3. in Schmitz et al 2017). Default is 0, i.e. no
                acceleration.
                
            rho : float (>=0)
                Unbalanced KL trade-off parameter (see section 4.4. in Schmitz et al 2017): 
                the smaller, the further from balanced OT we can stray. Ignored if unbalanced=False.
                
            varscale : float
                Trade-off hyperparameter between dictionary and weight optimization (see 
                end of section 3.1. in Schmitz et al 2017).
                
            unbalanced : bool
                If True, learn Unbalanced Wasserstein dictionary (with KL divergence for marginals).
                See section 4.4. in Schmitz et al 2017.
                
            loss : string
                Which fitting loss to use for the data similarity term. Implemented options are
                'L2' (default), 'L1' and 'KL'.
                
            feat_0 : np.ndarray or None
                Initialization of the dictionary. If None (default), initialize it as prescribed
                in feat_init.
                
            feat_init : str
                How to initialize dictionary atoms. Implemented options are:
                    - 'sampled': initialize each atom as a randomly drawn datapoint.
                    - 'uniform': initialize each atom as a uniform histogram.
                    - 'random' (default): intialize each atom as white gaussian noise.
                Ignored if feat_0 is given.
                
            wgt_0 : np.ndarray or None
                Initialization of the weights. If None (default), initialize it as prescribed
                in wgt_init.
                
            wgt_init : str
                How to initialize the weights. Implemented options are:
                    - 'uniform': give constant equal weight to each atom.
                    - 'random' (default): draw random initial weights.
                Ignored if wgt_0 is given.
                
            n_process : int (deprecated)
                If higher than 1, run several parallel batches. Not recommended as Theano
                already parallelizes computation (in a much better way).
            
            Verbose : bool
                Whether to print (or save to log) information as the learning goes on.
                
            savepath : str
                Path to folder where results will be saved. If empty (default), results are not 
                written to disk.
                
            logpath : str
                Path to folder where logfile should be written. If empty (default), info is printed 
                to stdout instead. Ignored if Verbose=False.
                
            checkplots : bool
                If true, plots of atoms are saved (and overwritten each time). Assumes squared
                images in np.flatten() convention.
                
            factr, pgtol, maxiter:
                SciPy L-BFGS parameters. See: 
                https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
            
    
            Returns
            -------
            (D,lbda) : tuple
                Learned dictionary and weights.
                
            f : float
                Final error achieved.
                
            dic : dictionary
                L-BFGS output dictionary.
        """  

        n, p = self.DataPoint.shape

        # INITIALIZATION
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
                w = np.ones((n, self.n_components)) / self.n_components
            elif self.wgt_init == "random":
                w = np.random.rand(n, self.n_components)
                w = (w.T / np.sum(w, axis=1)).T
        else:
            w = self.wgt_0
        dicw0 = np.log(np.vstack((Ys.T,w))).flatten()
        args = (self.Datapoint, self.gamma, self.Cost, self.n_components, self.n_iter_sinkhorn, self.tau, self.rho, self.varscale, 
                self.unbalanced, self.loss, self.n_process, self.Verbose, self.savepath, self.logpath, self.checkplots)
        x, f, dic = lbfgs(LBFGSFunc, dicw0, args=args, factr=factr, pgtol=pgtol, maxiter=maxiter)
        print (dic)
        print( 'FINAL ERROR:\t{}'.format(f))
        return unwrap_rep(x.reshape(n+p,n_components), (n,p)), f, dic
    
    
    @tf.function
    def EuclidCost(self,Nr, Nc, divmed=False, timeit=False, trunc=False, maxtol=745.13, truncval=745.13):
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




    @tf.function
    def sinkhorn_step(self,a,b,p,D,lbda,Ker,Tau):
        newa = D/tf.matmul((Ker,b))
        a = a**Tau * newa**(1.-Tau)
        #a = Tau*a + (1-Tau)*newa
    
        p = tf.math.reduce_prod(tf.matmul(tf.transpose(Ker),a)**lbda, axis=1)
    
        newb = p/tf.matmul(tf.transpose(Ker,a))
        b = b**Tau * newb**(1.-Tau)
        b = Tau*b + (1-Tau)*newb
        return a,b,p
    
    @tf.function
    def varchange(self,newvar):
        
         return tf.math.exp(newvar)/tf.math.reduce_sum(tf.math.exp(newvar))
     
    @tf.function
    def unwrap_rep(dicweights, datashape):
        n,p = datashape
        Ys, w = dicweights[:p,:], dicweights[p:,:]
        return Ys, w
    
    @tf.function
    def Loss_func(self): 
        return  tf.nn.l2_loss(self.Datapoint-self.bary)
    # def varchangetehano(newvar):
    #     return T.exp(newvar)/T.sum(T.exp(newvar))
    
    
    ###Convert this
    #result, updates = theano.scan(sinkhorn_step, outputs_info=[)], 
                            #      non_sequences=[D,lbda,Ker,Tau], n_steps=n_iter)
                            
    #bary = result[2][-1]
    
    @tf.function
    def update():
    
        return None
    
    varchange_Grads = tf.gradients(Loss, [self.Newvar_D,self.Newvar_lbda])
    
    Grads = tf.gradients(Loss, [self.D,self.lbda])
    
    
    

    @tf.function
    def Theano_wass_grad(self,Datapoint, D, lbda, Gamma, Cost, n_iter, Tau=0):
        return [Loss]+Grads
    
    
    
    
    @tf.function
    def varchange_Theano_wass_grad(self,Datapoint, Newvar_D, Newvar_lbda, Gamma, Cost, n_iter, Tau):
        for i in range(n_iter):
            a,b,p=self.sinkhorn_step(a,b,p,newvar_d,self.l)
        
        return [Loss]+varchange_Grads
    
    
    @tf.function
    def mp_Theano_grad(self,Xw, Ys, gamma, C, n_iter_sinkhorn, tau, rho, unbalanced, loss):
        datapoint, wi = Xw
        if unbalanced:
           # if loss=='L2':
            #    return unbal_varchange_wass_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
           # elif loss=='L1':
           #     return unbal_varchange_L1_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
           # elif loss=='KL':
           #     return unbal_varchange_KL_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
           print("UNBALANCED")
        else:
            if loss=='L2':
                return self.varchange_Theano_wass_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
          #  elif loss=='L1':
            #    return varchange_Theano_L1_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
           # elif loss=='KL':
             #   return varchange_Theano_KL_grad(datapoint, Ys, wi, gamma, C, n_iter_sinkhorn, tau)
    
    
    
    
    
    
    @tf.function
    def LBFGSFunc(self,dicweights, X, gamma, C, n_components, n_iter_sinkhorn=20, 
              tau=0, rho=float('inf'), varscale=100, 
              unbalanced=False, loss='L2', n_process=4, 
              Verbose=False, savepath='', logpath='', checkplots=False):
        n, p = X.shape
        dicweights = dicweights.reshape(n+p,n_components)
        Ys, w = self.unwrap_rep(dicweights, (n,p))
        
        if n_process>1:
            pool = mp.Pool(n_process)
            mp_grads = partial(self.mp_Theano_grad, Ys=Ys, gamma=gamma, C=C,
                                    n_iter_sinkhorn=n_iter_sinkhorn, tau=tau, rho=rho,
                                    unbalanced=unbalanced, loss=loss)
            Xw = zip(X,w)
            res = pool.map(mp_grads, Xw)
            err = 0 
            fullgrad = np.zeros((dicweights.shape))
            for i, (this_err, grad, graw) in enumerate(res):
                err += this_err
                fullgrad[:p,:] += grad/n
                fullgrad[p+i,:] = varscale*graw
            pool.close()
            pool.join()
        else:
            err = 0
            fullgrad = np.zeros((dicweights.shape))
            for i,(datapoint,wi) in enumerate(zip(X,w)):
                if unbalanced:
                #    if loss=='L2':
                #        this_err, grad, graw = unbal_varchange_wass_grad(datapoint, 
                #                              Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
                #    elif loss=='L1':
                #        this_err, grad, graw = unbal_varchange_L1_grad(datapoint, 
                ##                              Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
                 #   elif loss=='KL':
                 #       this_err, grad, graw = unbal_varchange_KL_grad(datapoint, 
                                         #     Ys, wi, gamma, C, n_iter_sinkhorn, rho, tau)
                    print("UNBALANCED")
                else:
                    if loss=='L2':
                        this_err, grad, graw = self.varchange_Theano_wass_grad(datapoint, 
                                              Ys, wi, gamma, C, n_iter_sinkhorn, tau)
                    #elif loss=='L1':
                   #     this_err, grad, graw = varchange_Theano_L1_grad(datapoint, 
                    #                          Ys, wi, gamma, C, n_iter_sinkhorn, tau)
                   # elif loss=='KL':
                     #   this_err, grad, graw = varchange_Theano_KL_grad(datapoint, 
                     #                         Ys, wi, gamma, C, n_iter_sinkhorn, tau)
                err += this_err
                fullgrad[:p,:] += grad/n
                fullgrad[p+i,:] = varscale*graw
      #  if Verbose:
       #     info = 'Current Error: {} - Duration: {} - Time: {}:{}'.format(
       #                 err, time.time()-start, time.localtime()[3], time.localtime()[4])
       ##     if logpath:
      #          logging.info(info)
       #     else:
        #        print (info)
        if savepath and self.LBFGSFunc.besterr>err:
            self.LBFGSFunc.besterr = err
            np.save(savepath+'dicweights.npy',dicweights)
            if checkplots:
                for i,yi in enumerate(alphatolbda(Ys.T)):
                    plot_func(yi, savepath=savepath+'atom_{}.png'.format(i))
        return err, fullgrad.flatten()
        
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