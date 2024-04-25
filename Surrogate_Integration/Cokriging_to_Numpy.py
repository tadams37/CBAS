import numpy as np
from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
import pickle

"""
This script will convert the SMT Co-Kriging surrogate models created by CBAS for the Cl and Cd (as SMT objects) 
and convert them into numpy functions, thereby removing the interactions with the SMT object. The process here 
follows closely that by Meliani and SMT, both cited here.

M. Meliani, “High dimensional efficient global optimization via multi-fidelity surro-
gate modeling,” Ph.D. dissertation, Sep. 2018. doi: 10.13140/RG.2.2.25899.59685/1.

P. Saves, R. Lafage, N. Bartoli, et al., “SMT 2.0: A surrogate modeling toolbox
with a focus on hierarchical and mixed variables gaussian processes,” Advances in
Engineering Sofware, vol. 188, p. 103 571, 2024. doi: https : / / doi . org / 10 . 1016 / j .
advengsoft.2023.103571.
"""

##Import Co-Kriging Surrogate Models (.pkl files)
cl_file = "[name_1].pkl"
cd_file = "[name_2].pkl"

sm_cl = None    #"surrogate model - Cl (SMT object)"
sm_cd = None

with open(cl_file, "rb") as f:
    sm_cl = pickle.load(f)

with open(cd_file, "rb") as f:
    sm_cd = pickle.load(f)

##################### Getting parameters from SMT object #######################################
lvl_cl = sm_cl.nlvl    
x_offset_cl = sm_cl.X_offset
x_scale_cl = sm_cl.X_scale 
y0_cl = sm_cl.X_norma_all[0].copy()
y1_cl = sm_cl.X_norma_all[1].copy()
optimal_theta0_cl = sm_cl.optimal_theta[0]
optimal_theta1_cl = sm_cl.optimal_theta[1]
nt_cl = sm_cl.nt
nt_all_cl = sm_cl.nt_all
nx_cl = sm_cl.nx   
beta0_cl = sm_cl.optimal_par[0]["beta"]
gamma0_cl = sm_cl.optimal_par[0]["gamma"]
beta1_cl = sm_cl.optimal_par[1]["beta"]
gamma1_cl = sm_cl.optimal_par[1]["gamma"]
y_mean_cl = sm_cl.y_mean
y_std_cl = sm_cl.y_std

lvl_cd = sm_cd.nlvl   
x_offset_cd = sm_cd.X_offset
x_scale_cd = sm_cd.X_scale 
y0_cd = sm_cd.X_norma_all[0].copy()
y1_cd = sm_cd.X_norma_all[1].copy()
optimal_theta0_cd = sm_cd.optimal_theta[0]
optimal_theta1_cd = sm_cd.optimal_theta[1]
nt_cd = sm_cd.nt
nt_all_cd = sm_cd.nt_all
nx_cd = sm_cd.nx   
beta0_cd = sm_cd.optimal_par[0]["beta"]
gamma0_cd = sm_cd.optimal_par[0]["gamma"]
beta1_cd = sm_cd.optimal_par[1]["beta"]
gamma1_cd = sm_cd.optimal_par[1]["gamma"]
y_mean_cd = sm_cd.y_mean
y_std_cd = sm_cd.y_std

################### Need a point (or points) to predict at ####################################################
#x = an array of shape (n,2) where [angle of attack (degrees), Mach number]
x = np.array([[5,19.5]]) 
x = x.astype(float)

################### Building Cl surrogate ######################################################################

n_eval_cl = x.shape[0]
mu_cl = np.zeros((n_eval_cl, lvl_cl))   #get kriging mean and variance at level 0
f_cl = np.ones([n_eval_cl,1])   #Regression is constant 
f0_cl = np.ones([n_eval_cl,1])

### get the offset and scaling for x using the parameters from the SMT model ####
x_cont_cl = (x - x_offset_cl) / x_scale_cl

### Get the distance matrix between the known points and the unkown point
dx_cl0 = (x_cont_cl[:,np.newaxis,:] - y0_cl[np.newaxis,:,:]).reshape((-1, x_cont_cl.shape[1]))

d_cl0 =  np.zeros((nt_all_cl[0],nx_cl))
d_cl0[0:nt_all_cl[0],:] = (dx_cl0[0:nt_all_cl[0],:])**2 #Squared because we're using square exponential

#Use the optimal theta vector from SMT model and distance matrix to compute correlation kernel
r0_cl = np.zeros((d_cl0.shape[0], 1))
n_components_cl = d_cl0.shape[1]
r0_cl[0 : nt_all_cl[0], 0] = (np.exp(-np.sum(optimal_theta0_cl.reshape(1,n_components_cl) * 
                                             d_cl0[0 : nt_all_cl[0],:],axis=1)))
r0_cl = r0_cl.reshape(1,nt_all_cl[0])

# Scaled Predictor
mu_cl[:,0] = (np.dot(f_cl,beta0_cl) + np.dot(r0_cl, gamma0_cl)).ravel()

## We only have 2 levels (two sets of data), so we don't need to recursively calculate at level i. Just level 1.

g_cl = np.ones([n_eval_cl,1]) #rho regression is also constant
dx_cl1 = (x_cont_cl[:,np.newaxis,:] - y1_cl[np.newaxis,:,:]).reshape((-1, x_cont_cl.shape[1]))
d_cl1 =  np.zeros((nt_all_cl[1],nx_cl))
d_cl1[0:nt_all_cl[1],:] = (dx_cl1[0:nt_all_cl[1],:])**2 #Squared because we're using square exponential
r1_cl = np.zeros((d_cl1.shape[0], 1))
n_components_cl1 = d_cl1.shape[1]
r1_cl[0 : nt_all_cl[1], 0] = (np.exp(-np.sum(optimal_theta1_cl.reshape(1,n_components_cl1) * 
                                             d_cl1[0 : nt_all_cl[1],:],axis=1)))
r1_cl = r1_cl.reshape(1,nt_all_cl[1])

f1_cl = np.vstack((g_cl.T * mu_cl[:,0], f0_cl.T))

mu_cl[:,1] = (np.dot(f1_cl.T,beta1_cl) + np.dot(r1_cl, gamma1_cl)).ravel()

mu_cl = mu_cl * y_std_cl + y_mean_cl

cl_guess = mu_cl[:,-1].reshape((n_eval_cl, 1))
cl = cl_guess[0][0]

################### Building Cd surrogate ######################################################################
n_eval_cd = x.shape[0]
mu_cd = np.zeros((n_eval_cd, lvl_cd))   #get kriging mean and variance at level 0
f_cd = np.ones([n_eval_cd,1])   #Regression is constant 
f0_cd = np.ones([n_eval_cd,1])

### get the offset and scaling for x using the parameters from the SMT model ####
x_cont_cd = (x - x_offset_cd) / x_scale_cd

### Get the distance matrix between the known points and the unkown point
dx_cd0 = (x_cont_cd[:,np.newaxis,:] - y0_cd[np.newaxis,:,:]).reshape((-1, x_cont_cd.shape[1]))

d_cd0 =  np.zeros((nt_all_cd[0],nx_cd))
d_cd0[0:nt_all_cd[0],:] = (dx_cd0[0:nt_all_cd[0],:])**2 #Squared because we're using square exponential

#Use the optimal theta vector from SMT model and distance matrix to compute correlation kernel
r0_cd = np.zeros((d_cd0.shape[0], 1))
n_components_cd = d_cd0.shape[1]
r0_cd[0 : nt_all_cd[0], 0] = (np.exp(-np.sum(optimal_theta0_cd.reshape(1,n_components_cd) * 
                                             d_cd0[0 : nt_all_cd[0],:],axis=1)))
r0_cd = r0_cd.reshape(1,nt_all_cd[0])

# Scaled Predictor
mu_cd[:,0] = (np.dot(f_cd,beta0_cd) + np.dot(r0_cd, gamma0_cd)).ravel()

## We only have 2 levels (two sets of data), so we don't need to recursively calculate at level i. Just level 1.

g_cd = np.ones([n_eval_cd,1]) #rho regression is also constant
dx_cd1 = (x_cont_cd[:,np.newaxis,:] - y1_cd[np.newaxis,:,:]).reshape((-1, x_cont_cd.shape[1]))
d_cd1 =  np.zeros((nt_all_cd[1],nx_cd))
d_cd1[0:nt_all_cd[1],:] = (dx_cd1[0:nt_all_cd[1],:])**2 #Squared because we're using square exponential
r1_cd = np.zeros((d_cd1.shape[0], 1))
n_components_cd1 = d_cd1.shape[1]
r1_cd[0 : nt_all_cd[1], 0] = (np.exp(-np.sum(optimal_theta1_cd.reshape(1,n_components_cd1) * 
                                             d_cd1[0 : nt_all_cd[1],:],axis=1)))
r1_cd = r1_cd.reshape(1,nt_all_cd[1])

f1_cd = np.vstack((g_cd.T * mu_cd[:,0], f0_cd.T))

mu_cd[:,1] = (np.dot(f1_cd.T,beta1_cd) + np.dot(r1_cd, gamma1_cd)).ravel()

mu_cd = mu_cd * y_std_cd + y_mean_cd

cd_guess = mu_cd[:,-1].reshape((n_eval_cd, 1))
cd = cd_guess[0][0]

############ Compare to SMT Prediction ######################################################################

# smt_cl = sm_cl.predict_values(x)
# smt_cd = sm_cd.predict_values(x)

# print(smt_cl, ' ', cl)
# print(smt_cd, ' ', cd)
