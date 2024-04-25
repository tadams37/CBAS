import numpy as np

from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
import pickle

from matplotlib import pyplot as plt
import matplotlib as mpl

N_VALS = 10_000

MED_FIG_SIZE = (6.5, 5)
SML_FIG_SIZE = (6.5, 3)
LAR_FIG_SIZE = (8, 9)

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

"""In get get_coefficients, the surrogate models will be loaded and then several cl and cd arrays will be returned. 
Because there are two input variables and an output variable (alpha, mach, and then cl or cd), we have to hold one of 
the inputs constant in order to create 2D plots. In the example here, we are plotting Cl/Cd versus Mach number,
so we will vary the Mach and hold the angle of attack (alpha) constant at several different values. 

get_qdot will have a similar function, but will require a third input variable, dynamic pressure. So we will need to hold two constant.
For the example here, we are holding Alpha and Mach constant and varying dynamic pressure (q).
"""

def get_coefficients(cl, cd):

    #Load in the surrogate models created in "cbaerosurrogate.py" using pickle
    sm_cl = None
    sm_cd = None
    cl_filename = cl
    cd_filename = cd

    with open(cl_filename, "rb") as f:
        sm_cl = pickle.load(f)

    with open(cd_filename, "rb") as f:
        sm_cd = pickle.load(f)

    mach_min = 5        ## CHANGE THESE TO MATCH YOURS
    mach_max = 22       

    mach_data = np.linspace(mach_min, mach_max, N_VALS)  # Lots of Mach numbers, can change this to alpha if you want to hold Mach constant instead

    alpha_hold_n5 = [-5]*N_VALS
    alpha_hold_0 = [0]*N_VALS
    alpha_hold_7p5 = [7.5]*N_VALS
    alpha_hold_15 = [15]*N_VALS

    # Hold Alpha = -5 
    x_values_n5 = np.array([alpha_hold_n5, mach_data]) # x_values_... is just the input array (nx2 array of alphas and mach number)
    x_values_n5 = x_values_n5.T

    cl_array_an5 = sm_cl.predict_values(x_values_n5)   # Use SMT's predict_values function to create an array of predicted values at those input points
    cd_array_an5 = sm_cd.predict_values(x_values_n5)
    
    
    # Hold Alpha = 0 
    x_values_0 = np.array([alpha_hold_0, mach_data])    
    x_values_0 = x_values_0.T

    cl_array_a0 = sm_cl.predict_values(x_values_0)  
    cd_array_a0 = sm_cd.predict_values(x_values_0)

    # Hold Alpha = 7.5 
    x_values_7p5 = np.array([alpha_hold_7p5, mach_data])
    x_values_7p5 = x_values_7p5.T

    cl_array_a7p5 = sm_cl.predict_values(x_values_7p5)
    cd_array_a7p5 = sm_cd.predict_values(x_values_7p5)
    
    # Alpha = 15
    x_values_15 = np.array([alpha_hold_15, mach_data])
    x_values_15 = x_values_15.T

    cl_array_a15 = sm_cl.predict_values(x_values_15)
    cd_array_a15 = sm_cd.predict_values(x_values_15)


    return mach_data, cl_array_an5, cl_array_a0, cl_array_a7p5, cl_array_a15, cd_array_an5, cd_array_a0, cd_array_a7p5, cd_array_a15

def get_qdot(qdot):

    #Load in the surrogate model created in "cbaerosurrogate.py" using pickle
    sm_qdot = None
    qdot_filename = qdot
    
    with open(qdot_filename, "rb") as f:
        sm_qdot = pickle.load(f)

    q_data = np.linspace(0.0015, 30, N_VALS)  # Lots of dynamic pressures numbers, can change this to alpha if you want to hold q constant instead

    alpha_hold_0 = [0]*N_VALS
    alpha_hold_15 = [15]*N_VALS

    mach_hold_5 = [5]*N_VALS
    mach_hold_13p5 = [13.5]*N_VALS
    mach_hold_20 = [22]*N_VALS

    # Alpha = 0 and mach = 5
    x_values_0 = np.array([alpha_hold_0, mach_hold_5, q_data])
    x_values_0 = x_values_0.T
    qdot_array_a0M5 = sm_qdot.predict_values(x_values_0)

    # Alpha = 0 and mach = 13.5
    x_values_13p5 = np.array([alpha_hold_0, mach_hold_13p5, q_data])
    x_values_13p5 = x_values_13p5.T
    qdot_array_a0M13p5 = sm_qdot.predict_values(x_values_13p5)

    # Alpha = 0 and mach = 20
    x_values_1 = np.array([alpha_hold_0, mach_hold_20, q_data])
    x_values_1 = x_values_1.T
    qdot_array_a0M20 = sm_qdot.predict_values(x_values_1)

    # Alpha = 15 and mach = 5
    x_values_2 = np.array([alpha_hold_15, mach_hold_5, q_data])
    x_values_2 = x_values_2.T
    qdot_array_a15M5 = sm_qdot.predict_values(x_values_2)

    # Alpha = 15 and mach = 13.5
    x_values_4 = np.array([alpha_hold_15, mach_hold_13p5, q_data])
    x_values_4 = x_values_4.T
    qdot_array_a15M13p5 = sm_qdot.predict_values(x_values_4)

    # Alpha = 15 and mach = 20
    x_values_3 = np.array([alpha_hold_15, mach_hold_20, q_data])
    x_values_3 = x_values_3.T
    qdot_array_a15M20 = sm_qdot.predict_values(x_values_3)

    return q_data, qdot_array_a0M5, qdot_array_a0M20, qdot_array_a15M5, qdot_array_a15M20, qdot_array_a0M13p5, qdot_array_a15M13p5 


#### Change these file names to match yours #####
cl_filename = "[cl_model].pkl"
cd_filename = "[cd_model].pkl"
qdot_filename = "[qdot_model].pkl"


### Call the two functions and get all of the arrays for plotting ##### 

q_data, qdot_array_a0M5, qdot_array_a0M20, qdot_array_a15M5, qdot_array_a15M20, qdot_array_a0M13p5, qdot_array_a15M13p5 = get_qdot(qdot_filename)
mach_data, cl_array_an5, cl_array_a0, cl_array_a7p5, cl_array_a15, cd_array_an5, cd_array_a0, cd_array_a7p5, cd_array_a15 = get_coefficients(cl_filename, cd_filename)

# FIGURE 1 (CL)
fig1 = plt.figure(figsize=LAR_FIG_SIZE)

ax11 = fig1.add_subplot(211)
ax11.plot(mach_data, cl_array_an5, color=cols[0], label='-5° Kriging')
ax11.plot(mach_data, cl_array_a0, color=cols[1], label='0° Kriging')
ax11.plot(mach_data, cl_array_a7p5, color=cols[2], label='7.5° Kriging')
ax11.plot(mach_data, cl_array_a15, color=cols[3], label='15° Kriging')
ax11.grid()
ax11.legend()
ax11.set_ylabel(r'$C_L$')
ax11.set_xlabel(r'$Mach$')
ax11.set_title(r'Cav-L Kriging - $C_L$')


# FIGURE 2 (CD)
fig2 = plt.figure(figsize=LAR_FIG_SIZE)

ax21 = fig2.add_subplot(211)
ax21.plot(mach_data, cd_array_an5, color=cols[0], label='-5° Kriging')
ax21.plot(mach_data, cd_array_a0, color=cols[1], label='0° Kriging')
ax21.plot(mach_data, cd_array_a7p5, color=cols[2], label='7.5° Kriging')
ax21.plot(mach_data, cd_array_a15, color=cols[3], label='15° Kriging')
ax21.grid()
ax21.legend()
ax21.set_ylabel(r'$C_D$')
ax21.set_xlabel(r'$Mach$')
ax21.set_title(r'Cav-L Kriging - $C_D$')


q_data, qdot_array_a0M5, qdot_array_a0M20, qdot_array_a15M5, qdot_array_a15M20, qdot_array_a0M13p5, qdot_array_a15M13p5
# FIGURE 3 (Q dot)
fig3 = plt.figure(figsize=LAR_FIG_SIZE)

ax31 = fig3.add_subplot(211)
ax31.plot(q_data, qdot_array_a0M5, color=cols[0], label='0°, Mach 5 Kriging')
ax31.plot(q_data, qdot_array_a0M20, color=cols[1], label='0°, Mach 20 Kriging')
ax31.plot(q_data, qdot_array_a15M5, color=cols[2], label='15°, Mach 5 Kriging')
ax31.plot(q_data, qdot_array_a15M20, color=cols[3], label='15°, Mach 20 Kriging')
ax31.plot(q_data, qdot_array_a0M13p5, color=cols[4], label='0°, Mach 13.5 Kriging')
ax31.plot(q_data, qdot_array_a15M13p5, color=cols[5], label='15°, Mach 5 Kriging')
ax31.grid()
ax31.legend()
ax31.set_ylabel(r'$C_L$')
ax31.set_xlabel(r'$Mach$')
ax31.set_title(r'Cav-L Kriging - $C_L$')


plt.show()

