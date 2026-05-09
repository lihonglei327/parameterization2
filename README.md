## Parameterization of PC-SAFT models

## Introduction
This is a suite of MATLAB-based programmes for the Parameterization of PC-SAFT models. There are three core files (ParamOpt.m, Sensitivity_Analysis.m and Validation_Para.m) for this work.

## Structure
- ParamOpt.m: Heuristic algorithm-based programme to output the parameters of PC-SAFT models.
- Validation_Para.m: Evaluating the fitting error of the parameterization work.
- Sensitivity_Analysis.m: Conducting the sensitivity analysis of parameters of PC-SAFT models.

More information about all modules are shown in the sequence diagrams in sensitivity_analysis.pdf, Parameterization.pdf and validation_param.pdf.

## Running
STEP 1.  Preparing the raw data into the trial_data.m file.
STEP 2.  Setting the values of PC_SAFT_Params in ParamOpt.m, then running the programme to output the parameters of polymer species. Args in CONST, ul and ub are optionally set.
STEP 3.  Setting the values of PC_SAFT_Params and running the Validation_Para.m to evaluate the fitting error of the parameterization work.
STEP 4. (Optional).  Running the Sensitivity_Analysis.m to analyze the sensitivity of parameters.
