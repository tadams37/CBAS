# CBAero_Surrogate (CBAS)
Creates Kriging and Co-Kriging surrogate models for a geometry using CBAero. There is one version of the script as of 04/25/2024: 'cbaerosurrogate.py'. There are multiple ways to run this script, whether it's locally in serial, using a cluster in serial or parallel, or using the GUI provided. The following will attempt to cover everything within the Git repository:

- README: Comprehensive description of the 'cbaerosurrogate.py' script so the user doesn't need the have extensive knowledge of the code itself.
- Surrogate.yml: This will help the user setup an environment that contains the correct packages (SMT) that will enable the use of this script. It's there for user convenience only.
- cbaerosurrogate.py: This script will create kriging or co-kriging surrogate models for a geometry using CBAero and will run on a local machine or up on a cluster. The README and everything below is based on this script.
- cbas_gui.ipynb: This is the Jupyter Notebook that runs the GUI. This GUI will either run CBAS locally using the specified inputs, or it will generate a script (.sh) that can be run up on a cluster in parallel.
- plotting_template.py: This script is a simple template to help with plotting the surrogate models you create using CBAS.

NOTE: _You must have CBAero installed on your machine to run this. This tool specifically uses CBAero to build surrogate models, meaning you need to have access to CBAero from NASA before this will be of any use._

## Setup 

### SMT
- To install SMT manually:
    - `conda install -c conda-forge smt`
- https://smt.readthedocs.io/en/latest/

### CBAero
- Add CBAero's binaries path (based on distro) to your PATH environment variable
    - Set all binaries in that directory to executable ("chmod +x *')
- CBAero requires an environment variable named "CBAERO_TABLES" that holds the path to its tables dir.
    - Example: `CBAERO_TABLES='/home/dkinney/cbaero.3.3.1.distro.9.04.2008/tables'`
    - Note that single quotes around path are required!

## Run

- NOTE: _You must run in the model directory corrsponding to your model file._ That is, your current working direcotry must contain the file for your selected model. CBAero does not like working with paths to the model!

### User Inputs (Command line arguments)

- Positional Arguments:
    1. model_file (The path, name, and extension of the model file. For example, if you were using NASA's HL20 model, it might be: "home/user/CBAS/HL20/hl20.cbaero".)
    1. start (Starting number of training points)
    1. stop (Ending number of training points)
    1. step (Step size for intervals if running multiple amounts of training points.)

- Optional Arguments:
    1. --run_type (The type of run being used. There are three types: \
                    "serial", "pstart", "pend". "serial" will generate Latin Hypercube samples, run CBAero for each point, and save the CBAero output as .otis files \
                    before reading those files and building a Kriging Surrogate. \
                    If the user provides a data file (high fidelity data), the code will build cokriging surrogate models using the given data as "high fidelity" and \
                    CBAero as "low fidelity". "serial" will build kriging or cokriging models depending on if you provide a data file or not. \
                    "pstart" runs "serial" logic, in parallel, only for the training points indexed by "task_id". \
                    "pend" runs the model building processes after CBAero has been run (logic on data produced using "pstart"). \
                    "serial" is the default value.)
    1. --min_alpha (The minumum angle of attack (in degrees) to test over. Default = 0)
    1. --max_alpha (The maximum angle of attack (in degrees) to test over. Default = 30)
    1. --min_mach (The minumum mach to test over. Default = 1.2. If it is less than 1.2, CBAERO must be run using inviscid solvers for those cases. CBAS will automatically change the solver in CBAERO for each case with Mach < 1.2, but this is a source of significant uncertainty for the surrogate model predictions.)
    1. --max_mach (The maximum mach to test over. Default = 25)
    1. --min_q (The minumum dynamic pressure (in bars) to test over. Default = 0.0035)
    1. --max_q (The maximum dynamic pressure (in bars) to test over. Default = 10)
    1. --cokrig_file (The path to the high fidelity training data file (must be an .xlsx type in the correct format, see example in this git repo) that will be used to create Co-Kriging surrogate models.)
    1. --save_as (This will be used to form the name of the save file for the Kriging Surrogate Models. There are two surrogates \
                              that will be saved: one for the lift coefficient (Cl) and one for the drag coefficient (Cd). "_cl" and "_cd" will be appended to the filename you input \
                              to differentiate the two models. For instance, if you use the default file name, you will have two files in the current directory: "kriging_cl.pkl" and \
                              kriging_cd.pkl". If you use your own name (maybe to distinguish the model geometry you are using), it will be "[user_input]_cl.pkl" for the Cl model. \
                              The number of training points will also be appended to the file name, thus saving each model during a convergence study. "[user_input]_cl_[num_training_pts]pts.pkl" \
                                Note that these files will be pkl files (we are using pickle for the save feature). Also note that if this is left empty, the models (or objects) will not be saved.)
    1. --task_id TASK_ID  Current task ID if running job array (usually $SLURM_ARRAY_TASK_ID). Only valid when
                        "run_type" is "pstart".

### Example:
  - `python /home/user/shared/CBAero_Surrogate/cbaerosurrogate.py home/user/CBAS/HL20/hl20.cbaero 15 16 1 --run_type=serial --min_alpha=0 --max_alpha=35 --min_mach=1.2 --max_mach=22 --min_q=0.0035 --max_q=10 --cokrig_file=/home/user/CBAS/HL20/HL20_Test_Data.xlsx --save_as=hl20_serial`

  - Note that this example is for a local serial run. 

  - It is recommended to save the command line input as a txt file so the user can simply run the file instead of having to type out that long line every single time. For example,
  the example above was saved as a text file named "call_script.txt". This allowed us to edit it using "nano call_script" from the command line and when we wanted to run the code, 
  we simply needed to type "./call_script" from the command line rather than typing all of that out every single time.

## Runnning in parallel on a Slurm cluster
A Slurm job array can be used to make cbaerosurrogate.py run CBAero in multiple process. Depending on the CPU resources available, these separate instances of cbaerosurrogate.py process will run at the same time. This can reduce the total runtime for larger runs. Running in parallel is done in two steps:

1. Run cbaerosurrogate.py, in a Slurm job array, with `--run_mode` set to "pstart" and `--task_id` set to `$SLURM_ARRAY_TASK_ID`. This causes each "task" in the job array to create a separate set of OTIS files, reads the OTIS files, and generates the surrogate models.
1. Run Call_CBAero_Teset.py, in single job, with  `--run_mode` set to "pend". This output plots and text data.

# Running CBAS Video
[![Watch the video](https://img.youtube.com/vi/ojdatBbXg-o/0.jpg)](https://www.youtube.com/watch?v=ojdatBbXg-o)

# Functions Within CBAS (cbaerosurrogate.py)

## get_hf_data
This function will read the high fidelity data file specified by the user (.xlsx file type required). The function simply reads each column of the file, stores the angle of attack, Mach number, Cl, and Cd data, and returns the data as arrays.

## latin_hypercube
This function will take in the number of training points and the minimums and maximums for angle of attack and mach number. Using pyDOE2's lhs function, the function will then create two arrays: one for the training points and one for the testing points. For simplicity, and as per common practice, the number of testing points is just a third of the number of training points (num_testing_pts = num_training_pts * 0.33). The dimensions of the the training point matrix will be the number of training points x 2 (num_training_pts x 2). The first column is the angle of attack, or alpha, and the second column is the mach number. Each row represents one LHS point: a combination of one alpha and one mach number. The testing point matrix will be in the same format, just 1/3 the size of the training point matrix. 
- Inputs: 
    - num_training: This is the number of training points that the user input as one of the required command line arguments
    - alpha_min: This is the minimum angle of attack (in degrees) the user wishes to test and is another command line argument
    - alpha_max: This is the maximum angle of attack (in degrees) the user wishes to test and is another command line argument
    - mach_min: This is the minimum mach number the user wishes to test and is another command line argument
    - mach_max: This is the maximum mach number the user wishes to test and is another command line argument
    - random_state: Fixed random state so the results are always the same.
- Outputs:
    - x_train_lhs: An n x 2 numpy array (where n is the number of training points) that comprises all of the Latin Hypercube Sampling points generated
    - x_test_lhs: An m x 2 numpy array (where m is the number of training points * 0.33) that comprises all of the Latin Hypercube Sampling points generated
- Sample:
    lb = np.array([alpha_lb, mach_lb])
    ub = np.array([alpha_ub, mach_ub])
    x_train_lhs = lhs(n=2, samples=num_training_pts, criterion="cm", random_state=random_state)
    x_train_lhs = lb + (ub - lb)*x_train_lhs

For more information on Latin Hypercube Sampling and pyDOE2's functionality, please refer to its documentation here: https://pypi.org/project/pyDOE2/

## latin_hypercube_qdot
This function is almost identical to the previous function. However, instead of just angle of attack and Mach number, we are also including dynamic pressure because our maximum heat flux will depend on that as well. Thus, we will have an n x 3 array for both the training and testing data for the maximum heat flux surrogate model.

## get_latin_hypercube_cokriging
Again, this function is identical to the first latin_hypercube function. However, because the testing data for the Co-Kriging models will come from the high fidelity data, this function only creates and returns Latin Hypercube training sets.

## run_cbaero
This function will call CBAero behind the scenes, run CBAero for each one of the LHS training points and testing points, and save the output .otis files in a separate location. CBAero takes in a text file "[name].cbaero" that contains some model data and parameters, as well as a list of angles of attack, mach numbers, and dynamic pressures to test over. CBAero will iterate over these lists and run the simulation for each combination of the three variables (so if you list 3 AoAs, 3 mach numbers, and 3 dynamic pressures, CBAero will run 27 cases: each combination, or, 3x3x3). This means that in order to use Latin Hypercube Sampling, we need to run a single LHS point (one angle of attack, one mach number, and one dynamic pressure) at a time and modify that text file after every single run. This is tedious, so the process has been automated using this function. 

The function takes in the number of training points, the x_train_lhs and x_test_lhs arrays created in the previous function, the model name, the .cbaero file name, the model directory, and the output .otis file directory (where the user wants the output files saved). It runs in two loops: the outer loop is a for loop that makes sure the function iterates over the training point array and then the testing point array, and the inner loop is a for loop that iterates over every single one of the LHS points for each array. This inner loop's job is to write over the text file that CBAero reads and make sure that the proper angle of attack, mach number, and dynamic pressure are being used for each run. This ensures that each LHS point is run in CBAero and the design space is well covered. Finally, the function will convert CBAero's output file to a .otis file (or a tecplot file for heat flux) and save that file with a unique name. The same approach is taken for the testing point array. By the end of the function's run, there should be a number of .otis files saved equal to the number of training points and testing points (if the user said 240 training points, there should be 240 _train.otis files and 80 _test.otis files). 

- Inputs: 
    - mode: Training or Testing (nothing to change here, these are just constants and used for the save file names)
    - iterations: This is the number of training points currently being run
    - lhs_matrix: The LHS matrix for the training or testing data
    - qdot_opt: An integer that defines whether or not to include/run the heat flux portion of the code or just the lift and drag code. 
    - task_id: Nothing to change here, this is automatically updated and done.
- Outputs:
    - There are two checks for failures within the run:
        - If there is a problem with CBAero, the command prompt will show "ERROR: Command failed with return code: ". Note that the CBAero output will be displayed in the command prompt window allowing you to see it's progress and track it if you wish.
        - If there is a problem with the otis file creation, the command prompt will show an error with a different code.

## read_otis
This function simply reads an .otis file and extracts the coefficient of lift and the coefficient of drag. These values are then stored in arrays that will be used for the surrogate model creation.

- Inputs:
    - filename: the path to the otis file that is being read. This is done internally.
- Outputs: 
    - alpha_list: A list of each angle of attack associated with each lift and drag coefficient. Each one of these lists should correspond to their respective columns in the _lhs arrays created previously
    - mach_list: A list of each mach number associated with each lift and drag coefficient
    - cl_array: A simple n x 1 array containing all of the lift coefficients associated with each LHS point tested
    - cd_array: A simple n x 1 array containing all of the drag coefficients associated with each LHS point tested

## tecplot_reader
This function will read the Tecplot file containing the heat flux data and extract the maximum heat flux for the training/testing point. This value is then returned.

- Inputs:
    - file: the path to the tecplot file being used. Done Internally.
- Outputs:
    - Value of maximum heat flux

## extract_otis
This function will call the read_otis function, extract the lift and drag coefficients, and build arrays for the training points, the testing points, and the lift and drag coefficients for the training and testing points. It's a simple function the just takes the outputs of the read_otis function and appends them to arrays for the training data and the testing data.

- Inputs:
    - num_training_pts: This is the number of training points that the user input as one of the required command line arguments
    - qdot_opt: Again, an integer that is used to define whether or not to include/run the heat flux portion of the code or just the lift and drag code.
- Outputs:
    - x: An array containing the training or testing input points. 
    - cl: An array containing the training or testing output points for Cl. 
    - cd: An array containing the training or testing output points for Cd. 
    - qdot: An array containing the training or testing output points for Qdot. 


## choose_correlation
Note that this function is no longer used as "squar_exp" will always be chosen. However, we will leave this here as a reference.

The Kriging surrogate model has several different correlation functions that are used for the covariance matrix when building the actual model. There are four types of correlation functions available in SMT (the surrogate modeling package we're using here): Exponential, Squared Exponential (Gaussian), Matern 5/2, and Matern 3/2. Each function has different advantages and disadvantages, and to save the user from needing to select a function themselves, this choose_correlation function will create surrogate models for the lift and drag coefficients using each one of the correlation functions and then keep the correlation function that yields the model with the lowest Root Mean Square Error (RMSE). 

This function takes in the training and testing matricies created in the previous function, whether or not it's for the lift or drag coefficients, and whether the error should be measured in lift or drag counts. It outputs the name of the best correlation function, as well as two plots: first, a plot that shows the RMSE vs. Correlation Function and then second, a plot that shows the RMSE reported in lift/drag counts vs. Correlation function. These two plots will look the exact same, however, lift/drag counts are easier to interpret for some so both plots are given. 

The function itself is very basic and just creates a surrogate model within a loop that iterates four times using each one of the correlation functions. It saves the RMSE for each model and then saves the name of the correlation function associated with the model that had the lowest RMSE. The RMSE is also used for rplotting purposes, as described previously.

- Inputs:
    - x_train: The x_values_train array created previously
    - x_test: The x_values_test array created previously
    - coe_train: Either the cl_values_train array or the cd_values_train array created previously depending on if we're building the surrogate for the lift or drag coefficient
    - coe_test: Either the cl_values_test array or the cd_values_test array created previously depending on if we're building the surrogate for the lift or drag coefficient
    - name: Either "Lift" or "Drag" and is used for plotting purposes. 
    - count_type: A divisor that converts RMSE to lift counts (0.001) or drag counts (0.0001), used for plotting. 
- Outputs:
    - best_corr: A string containing the name of the best correlation function found. For example, "matern52".
    - Plot Named "RMSE_{name}": A plot displaying the RMSE vs. Correlation function for either the lift or drag coefficient
    - Plot Named "RMSE_{name}_Counts": A plot displaying the RMSE (in lift/drag counts) vs. Correlation function for either the lift or drag coefficient

This function will be called twice: once for the coefficient of lift and then for the coefficient of drag because each one requires slightly different inputs for "name" and "count_type". For the lift, name="Lift" and count_type=0.001, and the drag is "Drag" and 0.0001 respectively.

Note that these plots will be saved in the current directory.

## fit_krig
This function simply creates and saves a Kriging surrogate model. It takes in the training point and testing point arrays for the data, the save file name (if any), and where the user wants the model saved. It will print the RMSE, the lift or drag counts, and the optimal theta. For examples of using SMT's functionality to build Kriging Models, see: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/krg.html

- Input: 
    - mode: specify whether we are building a model for the Cl, Cd, or Qdot
    - x: Training or testing point array
    - other: The Cl, Cd, or Qdot array
    - best_corr: 'squar_exp'
    - num_training_pts: current number of training points
    - qdot_opt: Again, an integer that is used to define whether or not to include/run the heat flux portion of the code or just the lift and drag code.
- Output:
    - sm: The kriging surrogate model 
    - The save file (.pkl)

## fit_cokrig
This function is almost identical to the previous function, only it is used to create Co-Kriging models. As such, it must include both the low fidelity and high fidelity training and testing arrays. The outputs are the same.

## plot_convergence
Because the number of training points needed to yield a surrogate model that's "good enough" is usually not known beforehand, the user may choose to test a range of training points to build models over. For example, the user can run the script for 100, 200, 300, and 400 training points to determine what amount of training points is appropriate (eg. run a convergence study). If this is the case, then all of the previous functions will be run for every single training point amount (meaning four cl surrogate models and four cd surrogate models will be created for the example above). This function will plot the RMSE for each one of those models against the number of training points and allow the user to see the trend as the number of training points increases. The user can then choose the appropriate surrogate model. 

- Output:
    - Plot Named "Convergence_Cl": Plot of the RMSE vs. the number of training points for the lift coefficient models
    - Plot Named "Convergence_Cd": Plot of the RMSE vs. the number of training points for the drag coefficient models
    - Plot Named "Convergence_QDot": Plot of the RMSE vs. the number of training points for the qdot (max heat flux) models

## write_convergence_data
Should the user want to see the data behind the convergence plots listed above, this function will create a simple text file that contains a table with the following columns: "Num Training Points", "Lift Count", "Drag Count", and "Qdot NRMSE". This way, the user can see the actual numbers associated with the convergence plot from the previous function.

- Output:
    - Text File Named "num_train_pts_vs_counts.txt": A file containing a table that displays the data from the convergence plot function


## write_point_data
Should the user want to see the all of the point data listed in one place, this function will create two text files (one for the training data and one for the testing data) that contain tables with the following columns: "Alpha", "Mach Number", "Cl", and "Cd", or, "Alpha", "Mach Number", "Dynamic Pressure (Bars)", "QDot". This script will simply write all of the point data, both training and testing, for each surrogate model created (for each variable).

- Inputs:
    - x: Array containing training or testing inputs
    - cl: Array containing training or testing outputs for Cl
    - cd: Array containing training or testing outputs for Cd
    - qdot: Array containing training or testing outputs for QDot
    - num_training_pts: Current number of training points
    - qdot_opt: Again, an integer that is used to define whether or not to include/run the heat flux portion of the code or just the lift and drag code.
- Outputs:
    -Text files for the training or testing data for both Cl & Cd, and QDot.

## plot_lhs_training
This simple function is designed to display the Latin Hypercube Sample points so the user can visualize how much of the design space is truly being covered by the sampling method. This is another way to just verify that the model is being built on a good foundation and it allows the user another glimpse into the "behind the scenes." This function reads in the training data set and plots each point. The x axis is angle of attack and the y axis is Mach number with the limits of each axis being the minimums and maximums for each variable. Again, this is just to allow the user to actually see how much of the design space is being covered by the training data.

- Inputs:
    - x_values_train: Array of training or testing inputs
    - num_training_pts: Current number of training points
    - qdot_opt: Again, an integer that is used to define whether or not to include/run the heat flux portion of the code or just the lift and drag code.
- Outputs:
    - Plots for the latin hypercube sampling arrays for both the Cl & Cd points, and the QDot points.

## plot_lhs_training_cokrig
This function is exactly like the previous function except it plots the latin hypercube points along with the high fidelity points used for the Co-Kriging surrogate models.

## copy_model
This function simply copies the model files and dir into the newly created otis directory.

## clear_lists
This function clears the lists that are used for the plotting and text file writing functions outlined above.

## update_lists
This function will update the lists for plotting/writing with the relevant information. This is an internal function.

## append_lists
This is called by update_lists and appends the data to the corresponding lists. This is an internal function.

## get_list_entry
This function reads the lift and drag counts from the saved data files.

## run
The is the big function that will run CBAS by calling all of the relevant functions described above. A lengthy description will not be given here. However, it is important to note that this function is divided into several loops that will be run based on the run type (serial, parallel, local, cluster, etc) and the type of models being created (Kriging or Co-Kriging). The time to run depends entirely on CBAERO - the amount of time it takes to run CBAERO for every point will constrain the total time to run CBAS. The model building and output files take mere seconds to run.

# Surrogate Integration Scripts
We have several options to integrate these surrogate models (saved as .pkl files) into other solvers/codes. First, and most simple, is directly incorporating the objects into your code. However, that is not always an option (say, for instance, you are using symbolic integration for a trajectory solver). Thus, this folder contains a few useful scripts that might help the user with integration. See https://figshare.com/articles/thesis/CBAS_A_Multi-Fidelity_Surrogate_Modeling_Tool_For_Rapid_Aerothermodynamic_Analysis/25668738. 

## Lookup_table_Creation.py
This script is a template for creating lookup tables from the surrogate models. See the script for more details.

## Kriging_to_Numpy.py
This script is a template for converting the Kriging surrogate model objects into numpy functions. This removes the "object" constraints. See the script for more details.

## Cokriging_to_Numpy.py
This script is a template for converting the Co-Kriging surrogate model objects into numpy functions. This removes the "object" constraints. See the script for more details.

## Cokriging_to_Casadi.py
This script is a template for converting the Co-Kriging surrogate model objects into Casadi's symbolic expressions functions. This removes the numerical constraints. See the script for more details.

# Graphical User Interface (GUI)

This GUI is a simple Jupyter Notebook that makes it easier for the user to run CBAS. It provides a simple interface whereby the user inputs can be obtained and will run CBAS directly if the user wants to run locally in serial. If the user wishes to run in parallel on a cluster, the GUI will create a Slurm job script for the user in the appropriate format which will allow the user to run up on a cluster.

There are four tabs in the GUI, and they are as follows:
1. Parameters: Here, the user will select the model path, the training point range, the minimumns and maximums for the angle of attack, Mach number, and dynamic pressure, the name for the saved Kriging (or Co-Kriging) file, and the high fidelity data file if the user is building Co-Kriging models. Note that the model file selection will search for ".cbaero" file extensions because that is the file type necessary. Additionally, the Cokrig file selection will search for only "xlsx" file extensions. The Cokrig file tab may be left blank if the user does not have high fidelity data to add and thus only wishes to create Kriging models.
1. Settings: In this tab, the user will define the paths for the CBAERO executables (see CBAERO documentation) and the CBAERO tables (see CBAERO documentaion). It also has an option to add the Run directory. This is for local, serial jobs and will be the location of all output files. It is strongly recommended that the user add a folder for the outputs to avoid cluttering the current directory.
1. Local Run: This tab will be used if the user wants to run locally in serial. CBAS can be run directly from this tab.
1. Job Script: This tab has a few additional parameters that must be added in order to generate a Slurm job script that can be run on a cluster. After filling in the appropriate parameters in the Parameters, Settings, and Job Script tabs you may select "Create Job Script" which will generate that job script.

To run the CBAS GUI:

1. Enter `voila cbas_gui.ipynb` via command line.
2. A browser window may open display the CBAS GUI. If it does not, or if a error is displayed:  
    1. Find the URL listed in the output of the voila. It will be similar to: `Voilà is running at: http://localhost:8866/`.
    2. Open a browser and enter that URL.
