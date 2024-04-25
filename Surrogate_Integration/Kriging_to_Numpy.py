import numpy as np
from smt.surrogate_models import KRG
import pickle

"""
This script will convert the SMT Kriging surrogate models created by CBAS (as SMT objects) and convert them 
into numpy functions, thereby removing the interactions with the SMT object. The process here follows closely 
that by Meliani and SMT, both cited here.

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
qdot_file = "[name_2].pkl"

sm_cl = None    #"surrogate model - Cl (SMT object)"
sm_cd = None
sm_qdot = None

with open(cl_file, "rb") as f:
    sm_cl = pickle.load(f)

with open(cd_file, "rb") as f:
    sm_cd = pickle.load(f)

with open(qdot_file, "rb") as f:
    sm_qdot = pickle.load(f)


################################## Getting constants ###############################################
    
x_offset_cl = sm_cl.X_offset
x_scale_cl = sm_cl.X_scale 
y_cl = sm_cl.X_norma.copy()
optimal_theta_cl = sm_cl.optimal_theta
nt_cl = sm_cl.nt
nx_cl = sm_cl.nx  
beta_cl = sm_cl.optimal_par["beta"]
gamma_cl = sm_cl.optimal_par["gamma"]
y_mean_cl = sm_cl.y_mean
y_std_cl = sm_cl.y_std

x_offset_cd = sm_cd.X_offset
x_scale_cd = sm_cd.X_scale 
y_cd = sm_cd.X_norma.copy()
optimal_theta_cd = sm_cd.optimal_theta
nt_cd = sm_cd.nt
nx_cd = sm_cd.nx 
beta_cd = sm_cd.optimal_par["beta"]
gamma_cd = sm_cd.optimal_par["gamma"]
y_mean_cd = sm_cd.y_mean
y_std_cd = sm_cd.y_std

x_offset_qdot = sm_qdot.X_offset
x_scale_qdot = sm_qdot.X_scale 
y_qdot = sm_qdot.X_norma.copy()
optimal_theta_qdot = sm_qdot.optimal_theta
nt_qdot = sm_qdot.nt
nx_qdot = sm_qdot.nx
beta_qdot = sm_qdot.optimal_par["beta"]
gamma_qdot = sm_qdot.optimal_par["gamma"]
y_mean_qdot = sm_qdot.y_mean
y_std_qdot = sm_qdot.y_std

################### Need a point (or points) to predict at ##########################################
x = np.array([[6,19]])
x = x.astype(float)

################### Getting cl ######################################################################

### get the offset and scaling for x using the parameters from the SMT model ####
x_cont_cl = (x - x_offset_cl) / x_scale_cl

### Get the distance matrix between the known points and the unkown point
dx_cl = (x_cont_cl[:,np.newaxis,:] - y_cl[np.newaxis,:,:]).reshape((-1, 2))

d_cl =  np.zeros((nt_cl,nx_cl))
d_cl[0:nt_cl,:] = (dx_cl[0:nt_cl,:])**2 #Squared because we're using square exponential

#Use the optimal theta vector from SMT model and distance matrix to compute correlation kernel
r_cl = np.zeros((nt_cl, 1))
r_cl[0 : nt_cl, 0] = np.exp(-np.sum(optimal_theta_cl.reshape(1,2) * d_cl[0 : nt_cl,:],axis=1))
r_cl = r_cl.reshape(1,nt_cl)

#We are using a constant F array of 1's, see documentation (Forrester)
y_cl = np.zeros(1)
f_cl = np.ones([1,1])

#Use beta and gamma from SMT model, then use the mean and std to compute the guess at unknown point.
y__cl = np.dot(f_cl, beta_cl) + np.dot(r_cl, gamma_cl)
cl_guess = y_mean_cl + y_std_cl*y__cl
cl = cl_guess[0][0]

#############Getting Cd #################################################################################
x_cont_cd = (x - x_offset_cd) / x_scale_cd

dx_cd = (x_cont_cd[:,np.newaxis,:] - y_cd[np.newaxis,:,:]).reshape((-1, 2))

d_cd =  np.zeros((nt_cd,nx_cd))
d_cd[0:nt_cd,:] = (np.abs(dx_cd[0:nt_cd,:])**2)

r_cd = np.zeros((nt_cd, 1))
r_cd[0 : nt_cd, 0] = np.exp(-np.sum(optimal_theta_cd.reshape(1,2) * d_cd[0 : nt_cd,:],axis=1))
r_cd = r_cd.reshape(1,nt_cd)

y_cd = np.zeros(1)
f_cd = np.ones([1,1])

y__cd = np.dot(f_cd, beta_cd) + np.dot(r_cd, gamma_cd)
cd_guess = y_mean_cd + y_std_cd*y__cd
cd = cd_guess[0][0]

#############Getting Qdot #################################################################################
# Note that xq is different than x because qdot takes three inputs: alpha, Mach, and dynamic pressure
xq = np.array([[15,22,20]])
xq = xq.astype(float)

x_cont_qdot = (xq - x_offset_qdot) / x_scale_qdot

dx_qdot = (x_cont_qdot[:,np.newaxis,:] - y_qdot[np.newaxis,:,:]).reshape((-1, 3))

d_qdot =  np.zeros((nt_qdot,nx_qdot))
d_qdot[0:nt_qdot,:] = (np.abs(dx_qdot[0:nt_qdot,:])**2)

r_qdot = np.zeros((nt_qdot, 1))
r_qdot[0 : nt_qdot, 0] = np.exp(-np.sum(optimal_theta_qdot.reshape(1,3) * d_qdot[0 : nt_qdot,:],axis=1))
r_qdot = r_qdot.reshape(1,nt_qdot)

y_qdot = np.zeros(1)
f_qdot = np.ones([1,1,1])

y__qdot = np.dot(f_qdot, beta_qdot) + np.dot(r_qdot, gamma_qdot)
qdot_guess = y_mean_qdot + y_std_qdot*y__qdot
qdot = qdot_guess[0][0]

############ Compare to SMT Prediction ######################################################################

# smt_cl = sm_cl.predict_values(x)
# smt_cd = sm_cd.predict_values(x)
# smt_qdot = sm_qdot.predict_values(xq)

# print(smt_cl, ' ', cl)
# print(smt_cd, ' ', cd)
# print(smt_qdot, ' ', qdot)