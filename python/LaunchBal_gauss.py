import numpy as np
import matplotlib.pyplot as plt
import time
start = time.time()
import WDL_tensorflow as wdl
# print( 'Theano compilation done in {}s.'.format(time.time()-start))

# ##############
# # Launch WDL #
# ##############
gamma = 7.
n_iter_sink = 100
n_components = 3
results_path = 'example/'

data = np.load(results_path+'data.npy')    
C = wdl.EuclidCost(data.shape[1],1)

factr = 10
pgtol = 1e-10
print(data.shape)
WDL_obj= wdl.WDL(Datapoint = data,n_components= n_components, gamma = gamma, n_iter_sinkhorn=n_iter_sink, Cost=C,
                 varscale=1,
                feat_0=None, feat_init='random',wgt_0=None, wgt_init='uniform',
                savepath=results_path,
               factr=factr, pgtol=pgtol)

dic = WDL_obj.LBFGSDescent()
# LBFGSDescent(data, n_components, gamma, n_iter_sinkhorn=n_iter_sink, C=C,
#                 varscale=1,
#                feat_0=None, feat_init='random',wgt_0=None, wgt_init='uniform',
#                savepath=results_path,
#               factr=factr, pgtol=pgtol)


def alphatolbda(alpha):
    return (np.exp(alpha).T / np.sum(np.exp(alpha), axis=1)).T







################
# Create plots #
# ################
plots_path = results_path + 'plots/'
grid_size = 20

xs = np.array(range(3*grid_size))-.5 # x-axis for plots

# # plot datapoints as single images
plt.rcParams['figure.figsize'] = 4., 4.

# # plot data
# for i, dat in enumerate(data):
#     plt.bar(xs,dat) 
#     plt.xlim([-.5,3*grid_size-.5])
#     plt.ylim(np.min(data)-.05,np.max(data)+.1)
#     plt.xticks(range(grid_size))
#     plt.xticks([])
#     plt.show()
#     plt.close()


# # read WDL outputs and convert them (reverse the logit variable change)
# dicw = np.load(results_path+'dicweights.npy')

DIC = dic.position.numpy()
n,p=data.shape
DIC = DIC.reshape(n+p,n_components )
Ys, w = DIC[:p,:], DIC[p:,:]
Ys_bal = alphatolbda(Ys.T)
w_bal = alphatolbda(w)


for i, atom in enumerate(Ys_bal):
    plt.bar(xs,atom, alpha=.5)
    plt.xlim([-.5,3*grid_size-.5])
    plt.ylim(np.min(data)-.05,np.max(data)+.1)
    plt.xticks([])
    plt.show()
    plt.close()

# # generate reconstructions
# recs_bal = np.array([WDL_obj.wass_grad(tf.transpose(Ys_bal),wi)   for wi in w_bal])
WDL_obj= WDL(Datapoint = data,n_components= n_components, gamma = gamma, n_iter_sinkhorn=n_iter_sink, Cost=C,
                 varscale=1,
                feat_0=None, feat_init='random',wgt_0=None, wgt_init='uniform',
                savepath=results_path,
               factr=factr, pgtol=pgtol)
recs_bal = np.array([WDL_obj.wass_grad((tf.transpose(Ys_bal),wi))   for wi in w_bal])

# # # plot balanced WDL atoms and reconstructions

for i, rec in enumerate(recs_bal):
    plt.bar(xs,rec)
    
    plt.xlim([-.5,3*grid_size-.5])
    plt.ylim(np.min(data)-.05,np.max(data)+.1)
    plt.xticks([])
    plt.show()
    plt.close()
        
