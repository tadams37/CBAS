import casadi as ca
import numpy as np
from smt.surrogate_models import KRG
from smt.applications.mfk import MFK
import pickle

"""
This script will build n x n lookup tables for Cl and Cd using the surrogate models obtained from CBAS and CasADi. 
The user might have to change a few things here, depending on what they have:
 1. The filenames for the cl and cd surrogate models to match whatever they have
 2. The num_pts which is the 'n' in the n x n table size. Basically, how dense do you want the lookup table
 3. The ranges for the two lists (alpha_list, mach_list) depending on the range of alpha and mach they used (alpha_min, alpha_max, mach_min, mach_max)
"""

def get_coefficient_tables():

    #Load in the surrogate models created in "cbaerosurrogate.py" using pickle
    sm_cl = None
    sm_cd = None
    cl_filename = "[name].pkl"
    cd_filename = "[name].pkl"

    with open(cl_filename, "rb") as f:
        sm_cl = pickle.load(f)

    with open(cd_filename, "rb") as f:
        sm_cd = pickle.load(f)

    # How many rows and columns in the table?
    num_pts = 10

    # Create lists of alphas and mach numbers to run through the surrogate models
    alpha_min = 0
    alpha_max = 10
    mach_min = 5
    mach_max = 15

    alpha_list = np.linspace(alpha_min, alpha_max, num_pts)
    mach_list = np.linspace(mach_min, mach_max, num_pts) 

    #Create an array for the x_values to plug into the surrogate models (Needs to be in nx2 array format for 
    # the surrogate model)
    result = []

    for i in range(len(alpha_list)):
        for j in range(len(mach_list)):
            result.append([alpha_list[i], mach_list[j]])
    
    x_values = np.array(result)

    #Run the values through the surrogate model to get the prediction arrays
    cl_array = sm_cl.predict_values(x_values)
    cd_array = sm_cd.predict_values(x_values)

    #Reshape the arrays to what CasADi needs.
    cl_array = np.reshape(cl_array, (num_pts,num_pts))
    cd_array = np.reshape(cd_array, (num_pts,num_pts))


    return alpha_list, mach_list, cl_array, cd_array

#####################################################
######## Create lookup tables using CasADi ##########
#####################################################

alpha_list, mach_list, cl_array, cd_array = get_coefficient_tables()

data_flat_cl = cl_array.ravel(order='F')
data_flat_cd = cd_array.ravel(order='F')

#These next two lines, cl_table_bspline and cd_table_bspline, are exactly what this script is for. These are 
# the lookup tables.

cl_table = ca.interpolant('cl_table', 'bspline', (alpha_list, mach_list), data_flat_cl)
cd_table = ca.interpolant('cd_table', 'bspline', (alpha_list, mach_list), data_flat_cd)