import numpy as np
import casadi as ca
import pickle

"""
This script will in essence follow the same steps as the Cokriging_to_Numpy code and will look very similar 
to it. However, this script will convert the SMT surrogate objects into symbolic expressions utilizing CasADi.
Therefore, the syntax will be different and the final result will be two expressions representing the Cl and 
Cd surrogates.

Note that this script is for Co-Kriging surrogate models. However, it is simple to use this script and the 
Kriging_to_Numpy.py code to build CasADi expressions for the Kriging models.
"""

# Define alpha and M as symbolic variables using CasADi
alpha = ca.MX.sym('alpha', 1)
M = ca.MX.sym('M', 1)

## Import Co-Kriging Surrogate Models (.pkl files)
cl_file = "[name_1].pkl"
cd_file = "[name_2].pkl"

sm_cl = None    #"surrogate model - Cl (SMT object)"
sm_cd = None

with open(cl_file, "rb") as f:
    sm_cl = pickle.load(f)

with open(cd_file, "rb") as f:
    sm_cd = pickle.load(f)

##################### Getting parameters from SMT object #######################################
    #Cl stuff
lvl_cl = sm_cl.nlvl    
x_offset_cl = ca.MX(sm_cl.X_offset)
x_scale_cl = ca.MX(sm_cl.X_scale) 
y0_cl = ca.MX(sm_cl.X_norma_all[0].copy())
y1_cl = ca.MX(sm_cl.X_norma_all[1].copy())
optimal_theta0_cl = ca.MX(sm_cl.optimal_theta[0])
optimal_theta1_cl = ca.MX(sm_cl.optimal_theta[1])
nt_cl = sm_cl.nt
nt_all_cl0 = sm_cl.nt_all[0]
nt_all_cl1 = sm_cl.nt_all[1]
nx_cl = sm_cl.nx 
beta0_cl = ca.MX(sm_cl.optimal_par[0]["beta"])
gamma0_cl = ca.MX(sm_cl.optimal_par[0]["gamma"])
beta1_cl = ca.MX(sm_cl.optimal_par[1]["beta"])
gamma1_cl = ca.MX(sm_cl.optimal_par[1]["gamma"])
y_mean_cl = ca.MX(sm_cl.y_mean)
y_std_cl = ca.MX(sm_cl.y_std)

    #Cd stuff
lvl_cd = sm_cd.nlvl   
x_offset_cd = ca.MX(sm_cd.X_offset)
x_scale_cd = ca.MX(sm_cd.X_scale) 
y0_cd = ca.MX(sm_cd.X_norma_all[0].copy())
y1_cd = ca.MX(sm_cd.X_norma_all[1].copy())
optimal_theta0_cd = ca.MX(sm_cd.optimal_theta[0])
optimal_theta1_cd = ca.MX(sm_cd.optimal_theta[1])
nt_cd = sm_cd.nt
nt_all_cd0 = sm_cd.nt_all[0]
nt_all_cd1 = sm_cd.nt_all[1]
nx_cd = sm_cd.nx 
beta0_cd = ca.MX(sm_cd.optimal_par[0]["beta"])
gamma0_cd = ca.MX(sm_cd.optimal_par[0]["gamma"])
beta1_cd = ca.MX(sm_cd.optimal_par[1]["beta"])
gamma1_cd = ca.MX(sm_cd.optimal_par[1]["gamma"])
y_mean_cd = ca.MX(sm_cd.y_mean)
y_std_cd = ca.MX(sm_cd.y_std)

################### Need to define prediction point #################################################
x = ca.vertcat(alpha, M)
################### Getting cl ######################################################################


n_eval_cl = 1
mu_cl = ca.MX.zeros(n_eval_cl, lvl_cl)   #get kriging mean and variance at level 0
f_cl = ca.MX.ones(n_eval_cl,1)   #Regression is constant 
f0_cl = ca.MX.ones(n_eval_cl,1)

### get the offset and scaling for x using the parameters from the SMT model ####
x_cont_cl = (x - x_offset_cl) / x_scale_cl

x_cont_cl_reshaped = ca.vertcat(x_cont_cl[0], x_cont_cl[1])
# Repeat x_cont_cl along the first axis to match the shape of y_cl
x_cont_cl_repeated = ca.repmat(x_cont_cl_reshaped, 1, nt_all_cl0).T

### Get the distance matrix between the known points and the unkown point
dx_cl0 = (x_cont_cl_repeated - y0_cl).reshape((-1, x_cont_cl.shape[0]))

d_cl0 =  ca.MX.zeros(nt_all_cl0,nx_cl)
d_cl0[0:nt_all_cl0,:] = (dx_cl0[0:nt_all_cl0,:])**2 #Squared because we're using square exponential

#Use the optimal theta vector from SMT model and distance matrix to compute correlation kernel
r0_cl = ca.MX.zeros(d_cl0.shape[0], 1)
optimal_theta_broadcasted_cl0 = ca.repmat(optimal_theta0_cl.T, nt_all_cl0, 1)
r0_cl[0 : nt_all_cl0, 0] = ca.exp(-ca.sum2(optimal_theta_broadcasted_cl0 * d_cl0[0 : nt_all_cl0,:]))

# Scaled Predictor
mu_cl[:,0] = (ca.dot(f_cl,beta0_cl) + ca.dot(r0_cl, gamma0_cl))

## We only have 2 levels (two data sets), so we don't need to recursively calculate at level i. Just level 1.

g_cl = ca.MX.ones(n_eval_cl,1) #rho regression is also constant
x_cont_cl_repeated1 = ca.repmat(x_cont_cl_reshaped, 1, nt_all_cl1).T
dx_cl1 = (x_cont_cl_repeated1 - y1_cl).reshape((-1, x_cont_cl.shape[0]))
d_cl1 =  ca.MX.zeros(nt_all_cl1,nx_cl)
d_cl1[0:nt_all_cl1,:] = (dx_cl1[0:nt_all_cl1,:])**2 #Squared because we're using square exponential
r1_cl = ca.MX.zeros((d_cl1.shape[0], 1))
optimal_theta_broadcasted_cl1 = ca.repmat(optimal_theta1_cl.T, nt_all_cl1, 1)
r1_cl[0 : nt_all_cl1, 0] = ca.exp(-ca.sum2(optimal_theta_broadcasted_cl1 * d_cl1[0 : nt_all_cl1,:]))

f1_cl = ca.vertcat(g_cl.T * mu_cl[:,0], f0_cl.T)

mu_cl[:,1] = (ca.dot(f1_cl,beta1_cl) + ca.dot(r1_cl, gamma1_cl))

mu_cl = mu_cl * y_std_cl + y_mean_cl

# All for this: A symbolic CasADi expression for the Cl surrogate model.
cl = mu_cl[:,-1].reshape((n_eval_cl, 1))

#############Getting Cd #################################################################################
n_eval_cd = 1
mu_cd = ca.MX.zeros(n_eval_cd, lvl_cd)   #get kriging mean and variance at level 0
f_cd = ca.MX.ones(n_eval_cd,1)   #Regression is constant 
f0_cd = ca.MX.ones(n_eval_cd,1)

### get the offset and scaling for x using the parameters from the SMT model ####
x_cont_cd = (x - x_offset_cd) / x_scale_cd

x_cont_cd_reshaped = ca.vertcat(x_cont_cd[0], x_cont_cd[1])
# Repeat x_cont_cl along the first axis to match the shape of y_cl
x_cont_cd_repeated = ca.repmat(x_cont_cd_reshaped, 1, nt_all_cd0).T

### Get the distance matrix between the known points and the unkown point
dx_cd0 = (x_cont_cd_repeated - y0_cd).reshape((-1, x_cont_cd.shape[0]))

d_cd0 =  ca.MX.zeros(nt_all_cd0,nx_cd)
d_cd0[0:nt_all_cd0,:] = (dx_cd0[0:nt_all_cd0,:])**2 #Squared because we're using square exponential

#Use the optimal theta vector from SMT model and distance matrix to compute correlation kernel
r0_cd = ca.MX.zeros(d_cd0.shape[0], 1)
optimal_theta_broadcasted_cd0 = ca.repmat(optimal_theta0_cd.T, nt_all_cd0, 1)
r0_cd[0 : nt_all_cd0, 0] = ca.exp(-ca.sum2(optimal_theta_broadcasted_cd0 * d_cd0[0 : nt_all_cd0,:]))

# Scaled Predictor)
mu_cd[:,0] = (ca.dot(f_cd,beta0_cd) + ca.dot(r0_cd, gamma0_cd))

## We only have 2 levels (two data sets), so we don't need to recursively calculate at level i. Just level 1.

g_cd = ca.MX.ones(n_eval_cd,1) #rho regression is also constant
x_cont_cd_repeated1 = ca.repmat(x_cont_cd_reshaped, 1, nt_all_cd1).T
dx_cd1 = (x_cont_cd_repeated1 - y1_cd).reshape((-1, x_cont_cd.shape[0]))
d_cd1 =  ca.MX.zeros(nt_all_cd1,nx_cd)
d_cd1[0:nt_all_cd1,:] = (dx_cd1[0:nt_all_cd1,:])**2 #Squared because we're using square exponential
r1_cd = ca.MX.zeros((d_cd1.shape[0], 1))
optimal_theta_broadcasted_cd1 = ca.repmat(optimal_theta1_cd.T, nt_all_cd1, 1)
r1_cd[0 : nt_all_cd1, 0] = ca.exp(-ca.sum2(optimal_theta_broadcasted_cd1 * d_cd1[0 : nt_all_cd1,:]))

f1_cd = ca.vertcat(g_cd.T * mu_cd[:,0], f0_cd.T)

mu_cd[:,1] = (ca.dot(f1_cd,beta1_cd) + ca.dot(r1_cd, gamma1_cd))

mu_cd = mu_cd * y_std_cd + y_mean_cd

#All for this: A symbolic CasADi expression for the Cd surrogate model.
cd = mu_cd[:,-1].reshape((n_eval_cd, 1))

############ Compare to SMT Prediction ######################################################################

# Test functions
# f = ca.Function('f', [alpha, M], [cl])
# fd = ca.Function('fd', [alpha, M], [cd])
# cl_value = f(5, 10)
# cd_value = fd(5, 10)