# CBAero_Surrogate (CBAS)
Creates Kriging and Co-Kriging surrogate models for a geometry using CBAero. There is one version of the script as of 04/25/2024: 'cbaerosurrogate.py'. There are multiple ways to run this script, whether it's locally in serial, using a cluster in serial or parallel, or using the GUI provided. The following will attempt to cover everything within the Git repository:

- README: Comprehensive description of the 'cbaerosurrogate.py' script so the user doesn't need the have extensive knowledge of the code itself.
- Surrogate.yml: This will help the user setup an environment that contains the correct packages (SMT) that will enable the use of this script. It's there for user convenience only.
- cbas_util.py and all test-[].sh scripts: These scripts have helped in the testing and development of the 'cbaerosurrogate.py' script which allows us to run in parallel and on the super computer. Again, these are for the user's reference and convenience, but not necessary to run the code.
- cbas_local.py: This script will create kriging surrogate models for a geometry using CBAero and will run on a local machine. The README and everything below is based on this script.
- cbas_cluster.py: This script will do the same thing as the 'local' script, but with the additional ability to run in parallel and using a super computer. The script will look different, as it uses object-oriented programming, but the functions implementation are the same.

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
    1. start (Starting number of training points)
    1. stop (Ending number of training points)
    1. step (Step size for intervals)
    1. model_name (The name of the model being used in CBAero. Do not include the file extension. For example, if you are using the \
                        NASA hl20 model, "hl20" would be your model name. NOT "hl20.cbaero".)

- Optional Arguments:
    1. --min_alpha (The minumum angle of attack (in degrees) to test over. Default = 0)
    1. --max_alpha (The maximum angle of attack (in degrees) to test over. Default = 30)
    1. --min_mach (The minumum mach to test over. Default = 1.2. It cannot be less than 1.2)
    1. --max_mach (The maximum mach to test over. Default = 25)
    1. --min_q (The minumum dynamic pressure (in bars) to test over. Default = 0.0035)
    1. --max_q (The maximum dynamic pressure (in bars) to test over. Default = 10)
    1. --run_type {fullstart,skipcbaero,pstart,pend}
                        The type of run being used. There are four types: "fullstart", "skipcbaero", "pstart",
                        "pend". "fullstart" will generate Latin Hypercube samples, run CBAero for each point,
                        and save the CBAero output as .otis files before reading those files and building the
                        Kriging Surrogate. If the user already has .otis files saved, they can skip the LHS and
                        CBAero steps and simply create a surrogate model using those .otis file. "pstart" runs
                        "fullstart" logic, in parallel, only for the training points indexed by "task_id".
                        "pend" runs "skipcbaero" logic on data produced using "pstart". "fullstart" is the
                        default value.
    1. --model_directory (The directory that contains your model. The default value is just the current working directory.)
    1. --otis_directory (The directory where the otis files created in CBAero will be saved. The default value is \
                         simply the current working directory. However, it is highly recommended that a separate folder be created because there will be a significant number of \
                         these files.)
    1. --save (Do you want to save the Kriging Surrogate Models for Cl and Cd? If they are saved, \
                        you will not need to create the surrogate each time you run this code and will only need to do so once, then load in the model for each subsequent run.\
                        Note that the save will be located in the current directory. Default is "Yes", the other option is "No".)
    1. --save_filename (This will be the name of the save file for the Kriging Surrogate Models, the default is "kriging". There are two surrogates \
                        that will be saved: one for the lift coefficient (Cl) and one for the drag coefficient (Cd). "_cl" and "_cd" will be appended to the filename you input \
                        to differentiate the two models. For instance, if you use the default file name, you will have two files in the current directory: "kriging_cl.pkl" and \
                        kriging_cd.pkl". If you use your own name (maybe to distinguish the model geometry you are using), it will be "[user_input]_cl.pkl" for the Cl model. \
                        Note that these files will be pkl files (we are using pickle for the save feature).)
    1. --task_id TASK_ID  Current task ID if running job array (usually $SLURM_ARRAY_TASK_ID). Only valid when
                        "run_type" is "pstart".

### Example:
  - `python /home/tyler/shared/CBAero_Surrogate/cbas_local.py 15 16 1 Biconic1_cbaero --min_alpha=0 --max_alpha=35 --min_mach=1.2 --max_mach=22 --min_q=0.0035 --max_q=10 --run_type=fullstart --model_directory=/home/tyler/shared/CBAero/cbaero_distribution.August.2022/models/BiConics_Vlad --otis_directory=/home/tyler/shared/Practice_Otis_Biconic1
  --save=Yes --save_filename=kriging`

  - Note that this example is for the 'local_surrogate_build.py' script and not for the cluster build. If the user is running on the cluster, there are additional inputs as outlined above. 
  
  - It is recommended to save the command line input as a txt file so the user can simply run the file instead of having to type out that long line every single time. For example,
  the example above was saved as a text file named "call_script.txt". This allowed us to edit it using "nano call_script" from the command line and when we wanted to run the code, 
  we simply needed to type "./call_script" from the command line rather than typing all of that out every single time.

  See `test.h` for running a test using a batch script.

## Runnning in parallel on a Slurm cluster
A Slurm job array can be used to make Call_CBAero_Teset.py run CBAero in multiple process. Depending on the CPU resources available, these separate instances of Call_CBAero_Teset.py process will run at the same time. This can reduce the total runtime for larger runs. Running in parallel is done in two steps:

1. Run Call_CBAero_Teset.py, in a Slurm job array, with `--run_mode` set to "pstart" and `--task_id` set to `$SLURM_ARRAY_TASK_ID`. This causes each "task" in the job array to create a separate set of OTIS files.
1. Run Call_CBAero_Teset.py, in single job, with  `--run_mode` set to "pend". This reads all of the OTIS files and generates output plots and text data.

See `test-job-parallel.sh` for running a parallel Slurm job. See `test-job-baseline.sh` for running a regular, non-parallel Slurm job.

# Functions Within _surrogate_build Scripts

## latin_hypercube
This function will take in the number of training points and the minimums and maximums for angle of attack, mach number, and dynamic pressure (all from the user inputs). Using pyDOE2's lhs function, the function will then create two arrays: one for the training points and one for the testing points. For simplicity, and as per common practice, the number of testing points is just a third of the number of training points (num_testing_pts = num_training_pts * 0.33). The dimensions of the the training point matrix will be the number of training points x 3 (num_training_pts x 3). The first column is the angle of attack, or alpha, the second column is the mach number, and the third column is dynamic pressure (in bars). Each row represents one LHS point: a combination of one alpha, one mach number, and one dynamic pressure. The testing point matrix will be in the same format, just 1/3 the size of the training point matrix. 
- Inputs: 
    - num_training: This is the number of training points that the user input as one of the required command line arguments
    - alpha_min: This is the minimum angle of attack (in degrees) the user wishes to test and is another command line argument
    - alpha_max: This is the maximum angle of attack (in degrees) the user wishes to test and is another command line argument
    - mach_min: This is the minimum mach number the user wishes to test and is another command line argument
    - mach_max: This is the maximum mach number the user wishes to test and is another command line argument
    - q_min: This is the minimum dynamic pressure (in bars) the user wishes to test and is another command line argument
    - q_max: This is the maximum dynamic pressure (in dbars) the user wishes to test and is another command line argument
- Outpus:
    - x_train_lhs: An n x 3 numpy array (where n is the number of training points) that comprises all of the Latin Hypercube Sampling points generated
    - x_test_lhs: An m x 3 numpy array (where m is the number of training points * 0.33) that comprises all of the Latin Hypercube Sampling points generated
- Sample:
    lb = np.array([alpha_lb, mach_lb, q_lb])
    ub = np.array([alpha_ub, mach_ub, q_ub])
    x_train_lhs = lhs(n=3, samples=num_training_pts, criterion="cm", random_state=random_state)
    x_train_lhs = lb + (ub - lb)*x_train_lhs

For more information on Latin Hypercube Sampling and pyDOE2's functionality, please refer to its documentation here: https://pypi.org/project/pyDOE2/

## cbaero_processing
This function will call CBAero behind the scenes, run CBAero for each one of the LHS training points and testing points, and save the output .otis files in a user-specified location. CBAero takes in a text file "[name].cbaero" that contains some model data and parameters, as well as a list of angles of attack, mach numbers, and dynamic pressures to test over. CBAero will iterate over these lists and run the simulation for each combination of the three variables (so if you list 3 AoAs, 3 mach numbers, and 3 dynamic pressures, CBAero will run 27 cases: each combination, or, 3x3x3). This means that in order to use Latin Hypercube Sampling, we need to run a single LHS point (one angle of attack, one mach number, and one dynamic pressure) at a time and modify that text file after every single run. This is tedious, so the process has been automated using this function. 

The function takes in the number of training points, the x_train_lhs and x_test_lhs arrays created in the previous function, the model name, the .cbaero file name, the model directory, and the output .otis file directory (where the user wants the output files saved). It runs in two loops: the outer loop is a while loop that makes sure the function iterates over the training point array and then the testing point array, and the inner loop is a for loop that iterates over every single one of the LHS points for each array. This inner loop's job is to write over the text file that CBAero reads and make sure that the proper angle of attack, mach number, and dynamic pressure are being used for each run. This ensures that each LHS point is run in CBAero and the design space is well covered. Finally, the function will convert CBAero's output file to a .otis file and save that file with a unique name to the location specified by the user. The same approach is taken for the testing point array. By the end of the function's run, there should be a number of .otis files saved equal to the number of training points and testing points (if the user said 240 training points, there should be 240 _train.otis files and 80 _test.otis files). 

- Inputs: 
    - num_training_pts: This is the number of training points that the user input as one of the required command line arguments
    - x_train_lhs: An n x 3 numpy array (where n is the number of training points) that comprises all of the Latin Hypercube Sampling points generated
    - x_test_lhs: An m x 3 numpy array (where m is the number of training points * 0.33) that comprises all of the Latin Hypercube Sampling points generated
    - modelname: The name of the model which is another command line argument. For example, if you were using the hl20 model and your mesh file was hl20.msh, you would input "hl20"
    - filename: This is created within the script and simply combines the model name to ".cbaero". CBAero needs the model name, we need the full filename in order to open it and overwrite it. Thus, we need both the modelname and the filename. 
    - model_dir: This is the model directory that you are running CBAero from and is a command line argument. This should be your current directory (see instructions above) because you need to be running this all from the directory that contains your CBAero model. However, it's best of you enter the whole path in the command line just to make sure.
    - otis_dir: This is the directory you want all of the output .otis files saved and is another command line argument. Please note that there will be a significant amount of output files, so it's recommended that you have a separated folder to store all of them.
- Outputs:
    - If it has run successfully, the command prompt will show "Done with CBAero processing!" and all of the .otis files should appear in the directory you specified.
    - There are two checks for failures within the run:
        - If there is a problem with CBAero, the command prompt will show "ERROR: Command failed with return code: ". Note that the CBAero output will be displayed in the command prompt window allowing you to see it's progress and track it if you wish.
        - If there is a problem with the otis file creation, the command prompt will show "ERROR: Command failed with return code: ". 

## read_otis
This function simply reads an .otis file and extracts the coefficient of lift and the coefficient of drag. These values are then stored in arrays that will be used for the surrogate model creation. Because each file is formatted the exact same, reading in the data is simply a matter of knowing how many angles of attack, mach numbers, and dynamic pressures we're testing because the line containing the information we want is just a function of those three numbers. Therefore, the function takes in the number of angles of attack being tested, the number of mach numbers being tested, the number of dynamic pressures being tested, and the filename described in the previous function and then outputs a list of alphas, a list of mach numbers, a list of dynamic pressures, and arrays for the lift and drag coefficients. 

- Inputs:
    - alphas: The amount of angles of attack being tested. For the LHS method, this will always be 1 because we ran one point at a time (each .otis file will only have one point). 
    - mach: The amount of mach numbers being tested. Again, this number will always be 1 here. 
    - q: The amount of dynamic pressures being tested. Again, this number will always be 1 here.
    - filename: This is created within the script and simply combines the model name to ".cbaero". See the previous function for an explanation.
- Outputs: 
    - alpha_list: A list of each angle of attack associated with each lift and drag coefficient. Each one of these lists should correspond to their respective columns in the _lhs arrays created previously
    - mach_list: A list of each mach number associated with each lift and drag coefficient
    - q_list: A list of each dynamic pressure associated with each lift and drag coefficient
    - cl_array: A simple n x 1 array containing all of the lift coefficients associated with each LHS point tested
    - cd_array: A simple n x 1 array containing all of the drag coefficients associated with each LHS point tested

## extract_otis
This function will call the read_otis function, extract the lift and drag coefficients, and build arrays for the training points, the testing points, and the lift and drag coefficients for the training and testing points. It's a simple function the just takes the outputs of the read_otis function and appends them to arrays for the training data and the testing data.

- Inputs:
    - num_training_pts: This is the number of training points that the user input as one of the required command line arguments
    - modelname: The name of the model which is another command line argument. For example, if you were using the hl20 model and your mesh file was hl20.msh, you would input "hl20"
    - otis_dir: This is the directory you want all of the output .otis files saved and is another command line argument. Please note that there will be a significant amount of output files, so it's recommended that you have a separated folder to store all of them.
- Outputs:
    - x_values_train: An n x 3 array which should be identical to the x_train_lhs array created previously. Relatively redundant. 
    - x_values_train: An m x 3 array which should be identical to the x_test_lhs array created previously. Relatively redundant.
    - cl_values_train: A simple n x 1 array that comprises all the lift coefficient values associated with the points (rows) in the x_values_train array
    - cl_values_test: A simple m x 1 array that comprises all the lift coefficient values associated with the points (rows) in the x_values_test array
    - cd_values_train: A simple n x 1 array that comprises all the drag coefficient values associated with the points (rows) in the x_values_train array
    - cd_values_test: A simple m x 1 array that comprises all the drag coefficient values associated with the points (rows) in the x_values_test array

## choose_correlation
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

## fit_krig_cl
This function simply creates and saves the surrogate model for the lift coefficient. It takes in the training point and testing point arrays for the data, the best correlation function name, whether or not the user wants to save the surrogate model, and where the user wants the model saved. It will print the RMSE, the Lift Count, and the Average Uncertainty across all of the points, and then return the surrogate model and lift count. For examples of using SMT's functionality to build Kriging Models, see: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/krg.html

- Input: 
    - x_train_cl: The x_values_train array created previously
    - x_test_cl: The x_values_test array created previously
    - cl_train: The cl_values_train array created previously
    - cl_test: The cl_values_test array created previously
    - best_corr: A string containing the best correlation function found in the previous function
    - save_check: The user command line input whether or not the user wanted to save the surrogate model. Default is "Yes"
    - save_filename: The name of the save file that the user specified in the command line argument. The file will be "{user_input}_cl.pkl"
- Output:
    - sm_cl: The surrogate model for the lift coefficient
    - lift_count: The lift count associated with the model
    - The save file in the location the user specified

## fit_krig_cd
This function will be the exact same as the previous function, just for the drag coefficient and not the lift coefficient.

- Input: 
    - x_train_cd: The x_values_train array created previously
    - x_test_cd: The x_values_test array created previously
    - cd_train: The cd_values_train array created previously
    - cd_test: The cd_values_test array created previously
    - best_corr: A string containing the best correlation function found in the "choose_corr" function
    - save_check: The user command line input whether or not the user wanted to save the surrogate model. Default is "Yes"
    - save_filename: The name of the save file that the user specified in the command line argument. The file will be "{user_input}_cd.pkl"
- Output:
    - sm_cd: The surrogate model for the lift coefficient
    - lift_count: The lift count associated with the model
    - The save file in the location the user specified

## plot_convergence
Because the number of training points needed to yield a surrogate model that's "good enough" is usually not known beforehand, the user may choose to test a range of training points to build models over. For example, the user can run the script for 100, 200, 300, and 400 training points to determine what amount of training points is appropriate (eg. run a convergence study). If this is the case, then all of the previous functions will be run for every single training point amount (meaning four cl surrogate models and four cd surrogate models will be created for the example above). This function will plot the RMSE for each one of those models against the number of training points and allow the user to see the trend as the number of training points increases. The user can then choose an appropriate number of training points to create a surrogate model with. 

- Input:
    - train_pts_list: A list containing the number of training points tested. For the example above, the list would contain "100, 200, 300, 400"
    - cl_count_list: A list containing the RMSE (in lift counts) for each lift coefficient surrogate model built
    - cd_count_list: A list containing the RMSE (in drag counts) for each drag coefficient surrogate model built
- Output:
    - Plot Named "Convergence_Cl": Plot of the RMSE vs. the number of training points for the lift coefficient models
    - Plot Named "Convergence_Cd": Plot of the RMSE vs. the number of training points for the drag coefficient models

The command window will display "Finished creating Convergence Plots." when this function finishes. 
Note that these plots will be saved in the current directory. 

## write_convergence_data
Should the user want to see the data behind the convergence plots listed above, this function will create a simple text file that contains a table with the following columns: "Num Training Points", "Lift Count", and "Drag Count". This way, the user can see the actual numbers associated with the convergence plot from the previous function.

- Input:
    - train_pts_list: A list containing the number of training points tested. For the example above, the list would contain "100, 200, 300, 400"
    - cl_count_list: A list containing the RMSE (in lift counts) for each lift coefficient surrogate model built
    - cd_count_list: A list containing the RMSE (in drag counts) for each drag coefficient surrogate model built
- Output:
    - Text File Named "num_train_pts_vs_counts.txt": A file containing a table that displays the data from the convergence plot function

The command window will display "Finished creating Convergence Data file." when this function finishes. 
Note thata this text file will be saved in the current directory.

## write_point_data
Should the user want to see the all of the point data listed in one place, this function will create two text files (one for the training data and one for the testing data) that contain tables with the following columns: "Alpha", "Mach Number", "Dynamic Pressure (Bars)", "Cl", and "Cd". This script will simply write all of the data in x_values_train, cl_values_train, and cd_values train to a text file and then do the same thing with the _test arrays. 

- Inputs:
    - x_values_train: An n x 3 array which should be identical to the x_train_lhs array created previously. 
    - x_values_train: An m x 3 array which should be identical to the x_test_lhs array created previously. 
    - cl_values_train: A simple n x 1 array that comprises all the lift coefficient values associated with the points (rows) in the x_values_train array
    - cl_values_test: A simple m x 1 array that comprises all the lift coefficient values associated with the points (rows) in the x_values_test array
    - cd_values_train: A simple n x 1 array that comprises all the drag coefficient values associated with the points (rows) in the x_values_train array
    - cd_values_test: A simple m x 1 array that comprises all the drag coefficient values associated with the points (rows) in the x_values_test arrayv
- Outputs:
    -Text File Named "train_pts_data.txt": A text file containing a table that displays all of the training point data
    -Text File Named "test_pts_data.txt": A text file containing a table that displays all of the test point data

The command window will display "Finished creating Training and Testing Points Data files." when this function finishes. 

## compare_cbaero
Essentially, this function will run everything through "fit_krig_cd" in order to give the user a "quick and dirty" way of checking the validity of the surrogate model. This function does everything described in the functions from latin_hypercube through fit_krig_cd and compares 25 points between CBAero and the surrogate models created previously. All it does is generate 25 points evenly spaced between the min and maxes of each variable (AoA, mach number, and dynamic pressure), runs those through CBAero, and then compares the CBAero results with the results when those same 25 points are run through the surrogate models. The function will output two plots: one for the lift coefficient and one for the drag coefficient, with both displaying the surrogate model evaluations and the CBAero results. Note that in order to plot these, one of the variables is held constant. We have opted to hold dynamic pressure constant at 1.25 bars because the dynamic pressure has little to no actual effect on the lift and drag coefficients. 

The reaason why this is a "quick and dirty" way of visually checking the validity of the surrogate model is simple: because these are 3D plots, it is difficult to see if the points align exactly because there's no easy way to determine the depth into the screen for each point. That being said, it is a quick sanity check that allows the user to see if the model is totally off or if they're on the right track.

- Inputs:
    - min_mach: This is the minimum mach number the user wishes to test and is a command line argument
    - max_mach: This is the maximum mach number the user wishes to test and is another command line argument
    - min_alpha: This is the minimum angle of attack (in degrees) the user wishes to test and is another command line argument
    - max_alpha: This is the maximum angle of attack (in degrees) the user wishes to test and is another command line argument
    - sm_cl: This is the surrogate model for the lift coefficient that was created previously
    - sm_cd: This is the surrogate model for the drag coefficient that was created previously
    - modelname: The name of the model which is another command line argument. For example, if you were using the hl20 model and your mesh file was hl20.msh, you would input "hl20"
    - model_dir: This is the model directory that you are running CBAero from and is a command line argument. This should be your current directory (see instructions above) because you need to be running this all from the directory that contains your CBAero model. However, it's best of you enter the whole path in the command line just to make sure.
    - filename: This is created within the script and simply combines the model name to ".cbaero". See the previous functions for an explanation.
    - otis_dir: This is the directory you want all of the output .otis files saved and is another command line argument. Please note that these .otis files will have unique names to separate them from the other .otis files created.
- Outputs:
    - Plot named "Comparison_Cl": This plot will compare the lift coefficient outputs of CBAero and the surrogate model
    - Plot named "Comparison_Cd": This plot will compare the drag coefficient outputs of CBAero and the surrogate model

Note that the command prompt window should display two messages throughout the completion of this function: "Done with CBAero processing for comparison!" and "Finished creating Comparison Plots."

## plot_lhs_training
This simple function is designed to display the Latin Hypercube Sample points so the user can visualize how much of the design space is truly being covered by the sampling method. This is another way to just verify that the model is being built on a good foundation and it allows the user another glimpse into the "behind the scenes." This function reads in the training data set, extracts each column as separate lists (remember, the first column is angle of attack, the second column is mach number, and the third colum is dynamic pressure), and then plots each point on a 3D plot. The x axis is angle of attack, the y axis is mach number, and the z axis is dynamic pressure with the limits of each axis being the minimums and maximums for each variable. Again, this is just to allow the user to actually see how much of the design space is being covered by the training data.

- Inputs:
    - x_values_train: An n x 3 array which should be identical to the x_train_lhs array created previously. This is the same x_values_train used before.
    - alpha_min: This is the minimum angle of attack (in degrees) the user wishes to test and is a command line argument
    - alpha_max: This is the maximum angle of attack (in degrees) the user wishes to test and is another command line argument
    - mach_min: This is the minimum mach number the user wishes to test and is another command line argument
    - mach_max: This is the maximum mach number the user wishes to test and is another command line argument
    - q_min: This is the minimum dynamic pressure (in bars) the user wishes to test and is another command line argument
    - q_max: This is the maximum dynamic pressure (in bars) the user wishes to test and is another command line argument
- Outputs:
    - Plot named "LHS_TrainingPTS": A 3D plot that displays all of the training points within the design space. This allows the user to see how much of the space is covered.

Note that the command prompt window will display "Finished creating LHS plot for training data." This is the final message that will be displayed in the program.

## main
The main function is relatively simple in that all of the previously described functions do all of the work. The first chunk of the function just reads in all of the user inputed command line arguments using python's argparse package. Documentation on that can be found here: https://docs.python.org/3/library/argparse.html .  The function then takes all of those arguments and assigns them to variables that will be used in the program. 

Because the user can run this program using several different training point amounts, all of the functions from latin_hypercube through fit_krig_cd are run within a 'for' loop that iterates over the range of training points (specified by the command line inputs "start", "stop", and "step"). There is a nested 'if' statement within the loop that checks the run_type: if it is a 'fullstart', then the latin_hypercube and cbaero_processing functions will be run. However, if the user already has all of the .otis files created (maybe they ran this program previously and have all of those files saved), they can skip these two functions by inputing 'skipcbaero' as the run_type in the command line. The remaining functions within the loop simply appear in the order listed above and usually are called twice: once for the lift coefficient and once for the drag coefficient. 

Outside of the loop, each function from plot_convergence through plot_lhs_training are called. Keep in mind, if a convergence study is being run the most helpful plots will be the convergence plots. The other data files and plots will only be for the surrogate model using the final training point amount. Once the proper training point amount has been determined, the program should be run again for that single case and all of the files and plots will be associated with this correct model. 

# Giuseppe Integration Scripts
Below will be the scripts found in the Giuseppe_Integration folder. Each subfolder corresponds to a geometry that we are working with and trying to integrate within Giuseppe. The lookup table scripts for each geometry work, however, the actual Giuseppe implementation script has not been fleshed out yet. This will be an ongoing process and we will update this as we continue to work on it. As of 10/31/2023, the _strike_surrogate scripts are not completely functioning. 

Note: _All of the cbaero mesh files are found within each individual folder and should be able to be loaded into and used with CBAero. These are .msh files._

Also note, in order to run any of the Giuseppe scripts, you need to have Giuseppe installed. This is a simple process and you can pull it from Sean Nolan's public github page found here:

https://github.com/SeanMatthewNolan/Giuseppe/tree/master

## Lookup Tables Surrogate
These scripts (found for each geometry) create lookup tables as CasADi objects for Cl and Cd using the surrogate models. The scripts will also generate several plots to allow the user to see the classic aerodynamic plots for the model. These scripts only need two things to run: the saved surrogate model for Cl and the saved surrogate model for Cd, both as .pkl files that should have been obtained from the surrogate model script outlined previously. The user will need to move those saved models into whatever folder the lookup table script is in and will need to make sure that the lookup table script is reading the right file name (if the user has changed the .pkl file name from what's already in the folder currently). After that, everything can be run. 

These scripts will create lookup tables as CasADi objects which will then be used for the Giuseppe integration script by simply importing the lookup tables.

### Run
- NOTE: _More to come once the code is a little more fleshed out._ 

## _Strike Surrogate
This is the actual script that will run Giuseppe and in this case, we're optimizing for max final velocity. Each geometry has a similar script in its folder, with some form of '_strike_surrogate' as its name. None of these scripts (as of 10/31/2023) work completely and we have tried a variety of different approaches. The files are here though and the user if free to explore the different repos and use their own controls knowledge to find solutions as they see fit. This is going to be continually worked on and updated when this function is up and running.

This Hypersonic Strike Surrogate Example is a cross between two examples created by Sean Nolan and Winston Levin. Sean Nolan created a 
'Hypersonic_Strike.py' script for the cav-h and cav-l geometries (this code follows the max_vf script). This script can be found on his 
github as a public repository:

https://github.com/SeanMatthewNolan/dissertation/tree/master/examples/hypersonic_strike/max_vf

Winston Levin created a 'minimum_time_to_climb.py script that uses Giuseppe and can also be found on Sean Nolan's public repository. This script utilizes
lookup tables in CasADi which is the reason we are splicing the two codes together. The script can be found here:

https://github.com/SeanMatthewNolan/Giuseppe/tree/master/examples/ocp/minimum_time_to_climb

Also note that Winston Levin has a 'maneuvers' branch for Giuseppe that is also public. He's continuing to update his scripts/examples there, so the user might find it helpful to check out that repository. Winston employs the use of lookup tables and is heavily involved in assisting us as we continue to figure out the use of surrogate models with Giuseppe, so the user is welcome to explore all the resources. The repo is found here:

https://github.com/winstonlevin/Giuseppe/tree/maneuvers/examples/ocp/maneuvers



### Run
- NOTE: _More to come once the code is a little more fleshed out._ 

## Plot Max Vf
Plots the output from the Giuseppe run in hypersonic_strike_surrogate. Because the Giuseppe optimization is not working yet, these scripts are unused. 10/31/2023

### Run
- NOTE: _More to come once the code is a little more fleshed out._ 

## Pressure Plot Max Vf
Plots the pressure output from the Giuseppe run in hypersonic_strike_surrogate. Because the Giuseppe optimization is not working yet, these scripts are unused. 10/31/2023

### Run
- NOTE: _More to come once the code is a little more fleshed out._ 

# Graphical User Interface (GUI)

To run the CBAS GUI:

1. Enter `voila cbas_gui.ipynb` via command line.
2. A browser window may open display the CBAS GUI. If it does not, or if a error is displayed:  
    1. Find the URL listed in the output of the voila. It will be similar to: `Voilà is running at: http://localhost:8866/`.
    2. Open a browser and enter that URL.