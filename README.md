# Logistics for Cell and Gene Therapies: Challenges and Strategies for Scaling Personalized Medicine
Master Thesis Project Josephine Becker

This repository contains the code and data setsfor the Master's thesis, "Logistics for Cell and Gene Therapies: Challenges and Strategies for Scaling Personalized Medicine" The project analyses the performance of decentralized and centralized network configurations under uncertainties, more specifically randomly generated delays and disruptions. 

The repository is organised as follows:

## 1. Supercloud_model
This folder contains the python scripts for running the model on an HPC cluste or local machine. The "Delays" folder contains the script for delay-only runs, the "Disruptions" folder contains the scripts for delays & disruptions, and the "Normal" folder contains the script for the base model runs.  Before running the script, the directory must be changed to the user directory. The variable "n" can be initialied to the desired annual demand level, and the "simult" variable should be set to the number of parallel simulations to be run at a given time. Before running the script, ensure the data files for the patient demand, delays and disruptions are in the same directory. 
When the code is run, it generates a new output directory "temp-delays" with the temporay data files incorporating the delays and disruptions.
The results from the model are stored in 2 csv files; one contains the costs and return times as positive real values, and the other is binary for the manufacturing site activations.

## 2. Data_generator
This folder contains the scripts used for generating the models' input data. 
The "delay&disrupt" folder contains the data files for delays and disruptions for each demand level which were created using the notebook "delay&disrupt_maker". To run the notebook, initialise the demand using "n_patients", the maximum number of delay days "m", the distribution from which to sample the lambda rate values "dist", the range for lambda "min_val" and "max_val", the number of simulations to generate delays for "num_vectors", the probability of a disruption "p_disrupt". The notebook then creates a data file for each demand level.
In the "demandgen" folder, the "generate_demand" notebook creates the input data without delays and disrtuptions. To do so, initialize "n_patients" with the chosen annual demand level. The outputs for each demand level are stored in data files.

## 3. Int_results
This folder contains all outputs from the models used for the results section of the thesis. The outputs are in csv files: one containing the costs and return times for each model, one containing the site activations. "1211" corresponds to delay-only runs, "1218" to delays & disruptions runs, and 1303 to the base model runs.
The notebook "graphs_plots" contains the functions used to plot the graphs that were used to visualize the results in the thesis paper.

## How to Use
1. Generate the demand, delays and disruption files using the notebooks in the "Data_generator" folder.
2. Run the python scripts for each demand level and model set-up in the "Supercloud_model" making sure to choose the appropriate number of parallel simulations for the size of the demand level (higher demand levels require lower numbers of parallel runs).
3. Analyze the output files using the graphing notebook in "Int_results".

## Requirements
- Python 3.10.12
- CPLEX IBM Solver 22.1.1
- Pyomo 6.8.2 (for running the MILP models)
- Additional dependencies mentioned in the Python scripts (e.g., pandas, numpy, matplotlib)


## License
This repository is licensed under the MIT License


## Optimization model code
The base optimization model code used for the simulation experiments in this thesis is the open-source i-SHIPMENT platform developed by Niki Triantafyllou at the Papathanasiou Lab at Imperial College London, which can be found in the following repository:
GitHub repository: [https://github.com/papathanlab/i-SHIPMENT](https://github.com/papathanlab/i-SHIPMENT)

## Citation
Triantafyllou, N., Bernardi, A., Lakelin, M. et al.  
*A digital platform for the design of patient-centric supply chains.*  
Scientific Reports *12*, 17365 (2022).  
[https://doi.org/10.1038/s41598-022-21290-5](https://doi.org/10.1038/s41598-022-21290-5)

