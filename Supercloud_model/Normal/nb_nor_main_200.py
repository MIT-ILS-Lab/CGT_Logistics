#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Importing all relevant libraries
from pyomo.environ import *
from pyomo.common.timing import TicTocTimer, report_timing
from time import process_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
import shutil
import os


# %%


import psutil
available_mem = psutil.virtual_memory().available
print(available_mem/10**9)


# %%


os.chdir('/home/gridsan/jbecker/thesis_code/normal')
print(os.getcwd())



###################### INITIALISE WITH SPECIFIC DATA ######################
###########################################################################

n = 200
simult = 3
date = 1317
data = f'/home/gridsan/jbecker/thesis_code/normal/Data{n}_profileA.dat'
delays = pd.read_csv(f'/home/gridsan/jbecker/thesis_code/normal/delays_{n}.dat', delim_whitespace=True, header=None)

t1_start = process_time()
n_delays = 1

columns = ['p'] + [f'd{i}' for i in range(1, n_delays+1)]
delays.columns = columns

gamma = 19
t_lim = 50

print(f'Running {n} patients per quarter')
###########################################################################

############################# FUNCTIONS ###################################
###########################################################################


def create_delays_files(runs, delaysdata, modeldata):
    # Create a new folder named 'temp_delays' if it doesn't exist
    output_dir = 'temp_delays'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(runs):
        # Create a copy of modeldata in the temp_delays directory
        output_file = os.path.join(output_dir, f'data{i+1}.dat')
        shutil.copy(modeldata, output_file)
        
        # Write the i+1th set of delays into the copied file
        p_column = delaysdata['p']
        d_chosen = f'd{i+1}'
        d_column = delaysdata[d_chosen]
        delays_txt = "param DELAY :=\n"
        for p, d in zip(p_column, d_column):
            delays_txt += f"{p} {d}\n"
        delays_txt += ";"
        
        with open(modeldata, 'r') as file:
            data_txt = file.read()
        
        # Append delaysdata to the copied file
        data_txt += '\n' + delays_txt
        
        with open(output_file, 'w') as file:
            file.write(data_txt)


create_delays_files(n_delays, delays, data)



###################### OPTIMISATION MODELS ######################
#################################################################

print('#################### DECENTRALISED MODEL RUN #########################')

model = AbstractModel()
#function to run 1 simulation
def simulation_with_delay1(data, model = model):
    timer = TicTocTimer()
    timer.tic('start')
    #report_timing()
    instance = model.create_instance(data)
    timer.toc('Built model')

    #print('Solving')
    opt = SolverFactory('cplex', executable='/home/gridsan/jbecker/linux_cplex/download/cplex/bin/x86-64_linux/cplex')
    myoptions = dict()
    #myoptions['log'] = '200_profileA.log'
    #myoptions['timelimit'] = t_lim # adapt for optimality gap
    opt.options['timelimit'] = t_lim
    results = opt.solve(instance, options=myoptions, tee=True) # solves and updates instance -> tee=False avoids printing results from CPLEX solver
    timer.toc('Time to solve')
    t1_stop = process_time()
    obj_val = value(instance.obj)
    manu_facilities = []
    for m in instance.m:
        if (value(instance.E1[m]) == 1):
            #print(f"Establish manufacturing facility {m}")
            manu_facilities.append(1)
        else:
            manu_facilities.append(0)
            
    
    # create a table to store the results
    result = np.array([obj_val, 
                       value(sum(instance.CTM[p] for p in instance.p))/len(instance.p), 
                       value(sum((instance.TRT[p]-19)*instance.DELAYPEN for p in instance.p))/len(instance.p), 
                       value(sum(instance.DELAY[p]*instance.MANUPEN for p in instance.p))/len(instance.p),
                       value(sum(instance.TTC[p] for p in instance.p))/len(instance.p), 
                       (10476+9312), 
                       obj_val/len(instance.p), 
                       value(instance.ATRT)])
    return result, manu_facilities


## Initialising model parameters & variables
# SETS
model.c = Set() # Leukapheresis sites
model.h = Set() # Hospitals
model.j = Set() # Transport mode
model.m = Set() # Manufacturing sites
model.p = Set() # Patients
model.t = RangeSet(130) # Time
model.tt = Set(initialize=model.t) # Alias of set t

# Indexed PARAMETERS
model.CIM = Param(model.m) # Capital investment for manufacturing facility
model.FCAP = Param(model.m) # Total capacity of manufacturing site
model.TT1 = Param(model.j) # Transport time LS to MS using transport mode j
model.TT3 = Param(model.j) # Transport time MS to hospital using transport mode j
model.U1 = Param(model.c, model.m, model.j) # Unit transport cost LS to MS using transport mode j
model.U3 = Param(model.m, model.h, model.j) # Unit transport cost MS to hospital using transport mode j
model.INC = Param(model.p, model.c, model.t, initialize=0) # Demand therapy p arriving for leukapheresis at LS site c at time t
model.CVM = Param(model.m, default={'m1':20920, 'm2':156900, 'm3':52300, 'm4':20920, 'm5':156900, 'm6':52300}) #Fixed variable costs


# Scalar PARAMETERS
model.FMAX = Param() # Maximum flow
model.FMIN = Param() # Minimum flow
model.TAD = Param(within=NonNegativeReals) # Duration of administration
model.TLS = Param(within=NonNegativeReals) # Duration of leukapheresis
model.TMFE = Param(initialize=7) # Duration of manufacturing excluding QC


# Binary VARIABLES
model.E1 = Var(model.m, within=Binary) # 1 if manufacturing facility m is established
model.X1 = Var(model.c, model.m, within=Binary) # 1 if a match between LS site c and MS site m is established
model.X3 = Var(model.m, model.h, within=Binary) # 1 if a match between MS site m and a hospital h is established
model.Y1 = Var(model.p, model.c, model.m, model.j, model.t, within=Binary) # 1 if a sample p is transferred from a LS site c to a MS site m via mode j at time t
model.Y3 = Var(model.p, model.m, model.h, model.j, model.t, within=Binary) # 1 if a sample p is transferred from a MS site m to a hospital h via mode j at time t

# Integer variables
model.INH = Var(model.p, model.h, model.t, within=NonNegativeIntegers) # Therapy p arriving at hospital h at time t

# Positive VARIABLES
model.CTM = Var(model.p, within=NonNegativeReals)
model.FTD = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
model.TTC = Var(model.p, within=NonNegativeReals)
model.LSA = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
model.LSR = Var(model.p, model.c, model.m, model.j, model.t, within=NonNegativeReals)
model.MSO = Var(model.p, model.m, model.h, model.j, model.t, within=NonNegativeReals)
model.OUTC = Var(model.p, model.c, model.t, within=NonNegativeReals)
model.OUTM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.INM = Var(model.p, model.m, model.t, within=NonNegativeReals)
model.DURV = Var(model.p, model.m, model.t, within=NonNegativeReals) # 1 only for the time period t in which a therapy p is manufactured in facility m
model.RATIO = Var(model.m, model.t, within=NonNegativeReals) # the percentage of utilisation of MS site m at time t


# VARIABLES
model.TOTCOST = Var() # Total cost
model.CAP = Var(model.m, model.t) # Capacity of MS m at time t
model.TRT = Var(model.p) # Total return time of therapy
model.ATRT = Var() # Average return time
model.STT = Var(model.p) # Starting time of treatment for patient p
model.CTT = Var(model.p) # Completion time of treatment for patient p

######################### CAN BE MODIFIED ########################
# Extra parameter for delay
model.DELAY = Param(model.p, within=NonNegativeReals)
model.DELAYPEN = Param(default=2000)
model.MANUPEN = Param(default=1000)
##################################################################

# %%


## Objective function
def obj_rule(model):
    return sum( model.CTM[p] for p in model.p )+ sum( model.TTC[p] for p in model.p ) + (10476+9312)*len(model.p) #+ sum((model.TRT[p]-19)*model.DELAYPEN for p in model.p)
    
model.obj = Objective( rule=obj_rule )

## Constraints
# Manufacturing cost
# C1
# base case: 58000 $ 80% utilisation
def C1_rule(model,p):
    return model.CTM[p] == sum((model.E1[m]*(model.CIM[m]+model.CVM[m]))*len(model.t)/len(model.p) for m in model.m) + model.DELAY[p]*model.MANUPEN
model.C1 = Constraint(model.p, rule=C1_rule)


#RATIOEQ
def RATIOEQ_rule(model,m,t):
    return model.RATIO[m,t] == sum(model.DURV[p,m,t]/model.FCAP[m] for p in model.p)
model.RATIOEQ = Constraint(model.m, model.t, rule=RATIOEQ_rule)


#MSBnew
def MSBnew_rule(model,p,m,t):
     return model.DURV[p,m,t] == sum(model.INM[p,m,tt-1]-model.OUTM[p,m,tt] for tt in model.tt if tt<=t and tt>1) + model.OUTM[p,m,t] 
model.MSBnew = Constraint(model.p, model.m, model.t, rule=MSBnew_rule)


# Transport cost
#C2
def C2_rule(model,p):
    return model.TTC[p] == sum(model.Y1[p,c,m,j,t]*model.U1[c,m,j] for c in model.c for m in model.m for j in model.j for t in model.t) + sum(model.Y3[p,m,h,j,t]*model.U3[m,h,j] for m in model.m for h in model.h for j in model.j for t in model.t)
model.C2 = Constraint(model.p, rule=C2_rule)


#MSB1
def MSB1_rule(model,p,c,t,tt):
    if tt == t + model.TLS:
        return model.INC[p,c,t] == model.OUTC[p,c,tt]
    else:
        return Constraint.Skip
model.MSB1 = Constraint(model.p, model.c, model.t, model.tt, rule=MSB1_rule)


#MSB2
def MSB2_rule(model,p,m,t,tt):
    if tt == t + model.TMFE + model.DELAY[p]:
        return model.INM[p,m,t] == model.OUTM[p,m,tt]
    else:
        return Constraint.Skip
model.MSB2 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB2_rule)



#MSB8
def MSB8_rule(model,p,m,t,tt):
    if tt == t + 7:
        return model.OUTM[p,m,t] == sum(model.MSO[p,m,h,j,tt] for h in model.h for j in model.j)
    else:
        return Constraint.Skip
model.MSB8 = Constraint(model.p, model.m, model.t, model.tt, rule=MSB8_rule)


#MSB3
def MSB3_rule(model,p,c,m,j,t,tt):
    if tt == t + model.TT1[j]:
        return model.LSR[p,c,m,j,t] == model.LSA[p,c,m,j,tt]
    else:
        return Constraint.Skip
model.MSB3 = Constraint(model.p, model.c, model.m, model.j, model.t, model.tt, rule=MSB3_rule)


#MSB4
def MSB4_rule(model,p,m,h,j,t,tt):
    if tt == t + model.TT3[j]:
        return model.MSO[p,m,h,j,t] == model.FTD[p,m,h,j,tt]
    else:
        return Constraint.Skip
model.MSB4 = Constraint(model.p, model.m, model.h, model.j, model.t, model.tt, rule=MSB4_rule)


#MSB5
def MSB5_rule(model,p,m,t):
    return model.INM[p,m,t] == sum(model.LSA[p,c,m,j,t] for c in model.c for j in model.j)
model.MSB5 = Constraint(model.p, model.m, model.t, rule=MSB5_rule)


#MSB6
def MSB6_rule(model,p,h,t):
    return model.INH[p,h,t] == sum(model.FTD[p,m,h,j,t] for m in model.m for j in model.j)
model.MSB6 = Constraint(model.p, model.h, model.t, rule=MSB6_rule)


#MSB7
def MSB7_rule(model,p,c,t):
    return model.OUTC[p,c,t] == sum(model.LSR[p,c,m,j,t] for m in model.m for j in model.j)
model.MSB7 = Constraint(model.p, model.c, model.t, rule=MSB7_rule)


# Capacity equation
#CAP1
def CAP1_rule(model,m,t):
    return model.CAP[m,t] == model.FCAP[m]-sum(model.INM[p,m,tt] for p in model.p for tt in model.tt if tt<t and tt>=t-model.TMFE)
model.CAP1 = Constraint(model.m, model.t, rule=CAP1_rule)


# Capacity constraint
#CAPCON1
def CAPCON1_rule(model,m,t):
    return sum(model.INM[p,m,t] for p in model.p)-sum(model.OUTM[p,m,t] for p in model.p) <= model.CAP[m,t]
model.CAPCON1 = Constraint(model.m, model.t, rule=CAPCON1_rule)


# Constraints to ensure that no matches are established with non-existent facilities
#CON1
#def CON1_rule(model):
#    return sum(model.E1[m] for m in model.m) <= 2
#model.CON1 = Constraint(rule=CON1_rule)


#CON2
def CON2_rule(model,c,m):
    return model.X1[c,m] <= model.E1[m]
model.CON2 = Constraint(model.c, model.m, rule=CON2_rule)


#CON3
def CON3_rule(model,m,h):
    return model.X3[m,h] <= model.E1[m]
model.CON3 = Constraint(model.m, model.h, rule=CON3_rule)


#CON4
def CON4_rule(model,p,c,m,j,t):
    return model.Y1[p,c,m,j,t] <= model.X1[c,m]
model.CON4 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON4_rule)


#CON5
def CON5_rule(model,p,m,h,j,t):
    return model.Y3[p,m,h,j,t] <= model.X3[m,h]
model.CON5 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON5_rule)


#CON6
def CON6_rule(model,p):
    return sum(model.Y1[p,c,m,j,t] for c in model.c for m in model.m for j in model.j for t in model.t) == 1
model.CON6 = Constraint(model.p, rule=CON6_rule)


#CON7
def CON7_rule(model,p):
    return sum(model.Y3[p,m,h,j,t] for m in model.m for h in model.h for j in model.j for t in model.t) == 1
model.CON7 = Constraint(model.p, rule=CON7_rule)


# Demand satisfaction
#DEM
def DEM_rule(model):
    return sum(model.INH[p,h,t] for p in model.p for h in model.h for t in model.t) <= len(model.p)
model.DEM = Constraint(rule=DEM_rule)


# Flow constraints
#CON8
def CON8_rule(model,p,c,m,j,t):
    return model.LSR[p,c,m,j,t] >= model.Y1[p,c,m,j,t]*model.FMIN
model.CON8 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON8_rule)


#CON9
def CON9_rule(model,p,c,m,j,t):
    return model.LSR[p,c,m,j,t] <= model.Y1[p,c,m,j,t]*model.FMAX
model.CON9 = Constraint(model.p, model.c, model.m, model.j, model.t, rule=CON9_rule)


#CON10
def CON10_rule(model,p,m,h,j,t):
    return model.MSO[p,m,h,j,t] >= model.Y3[p,m,h,j,t]*model.FMIN
model.CON10 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON10_rule)


#CON11
def CON11_rule(model,p,m,h,j,t):
    return model.MSO[p,m,h,j,t] <= model.Y3[p,m,h,j,t]*model.FMAX
model.CON11 = Constraint(model.p, model.m, model.h, model.j, model.t, rule=CON11_rule)


#CON12
def CON12_rule(model,p):
    return sum(model.Y3[p,m,'h1',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c1',t] for t in model.t)
model.CON12 = Constraint(model.p, rule=CON12_rule)


#CON13
def CON13_rule(model,p):
    return sum(model.Y3[p,m,'h2',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c2',t] for t in model.t)
model.CON13 = Constraint(model.p, rule=CON13_rule)


#CON14
def CON14_rule(model,p):
    return sum(model.Y3[p,m,'h3',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c3',t] for t in model.t)
model.CON14 = Constraint(model.p, rule=CON14_rule)


#CON15
def CON15_rule(model,p):
    return sum(model.Y3[p,m,'h4',j,t] for m in model.m for j in model.j for t in model.t) == sum(model.INC[p,'c4',t] for t in model.t)
model.CON15 = Constraint(model.p, rule=CON15_rule)


#START
def START_rule(model,p):
    return model.STT[p] == sum(model.INC[p,c,t]*t for c in model.c for t in model.t)
model.START = Constraint(model.p, rule=START_rule)


#END
def END_rule(model,p):
    return model.CTT[p] == sum(model.INH[p,h,t]*t for h in model.h for t in model.t)
model.END = Constraint(model.p, rule=END_rule)


#TIME
def TIME_rule(model,p):
    return model.TRT[p] == model.CTT[p] - model.STT[p]
model.TIME = Constraint(model.p, rule=TIME_rule)


#ATIME
def ATIME_rule(model):
    return model.ATRT == sum(model.TRT[p] for p in model.p)/len(model.p)
model.ATIME = Constraint(rule=ATIME_rule)


#TSEQ
def TSEQ_rule(model,p):
    return model.STT[p] <= model.CTT[p]
model.TSEQ = Constraint(model.p, rule=TSEQ_rule)




if __name__ == '__main__':
    start_time = time.time()
    num_processes = 24 # use 24! crashes with >48 processes
    chunk_size = n_delays // num_processes

    # create the multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    # create an array of the arguments you want to pass to simulation_with_delay, where each argument is a data file from temp_delays folder
    datas = [f'/home/gridsan/jbecker/thesis_code/normal/temp_delays/data{i+1}.dat' for i in range(n_delays)]
    # use the pool.map function to run the simulation_with_delay function on all the arguments in parallel
    results = pool.imap(simulation_with_delay1, datas)
    pool.close()
    pool.join()
    
    run_results = pd.DataFrame(columns=['Total cost', 
                                    'Average manufacturing cost per therapy', 
                                    'Average TRT delay penalty per therapy',
                                    'Average manufacturing delay penalty per therapy', 
                                    'Average transport cost per therapy', 
                                    'Average QC cost per therapy', 
                                    'Average cost per therapy', 
                                    'Average return time'])
 
    manu_config = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4', 'm5', 'm6'])
    trans_config = pd.DataFrame(columns=['transport_mode1', 'transport_mode2'])

        
    
    for result in results:
        # Convert result[0] into a DataFrame with the correct column names
        result_df = pd.DataFrame([result[0]], columns=run_results.columns)

        # Append the data to run_results
        run_results = pd.concat([run_results, result_df], ignore_index=True)

        # Convert result[1] into a DataFrame with the correct column names for manu_config
        manu_config_df = pd.DataFrame([result[1]], columns=manu_config.columns)

        # Append the data to manu_config
        manu_config = pd.concat([manu_config, manu_config_df], ignore_index=True)
    
    run_results.to_csv(f'{date}_normal_d_{n}.csv')
    manu_config.to_csv(f'{date}_normal_d_manu_{n}.csv')
    end_time = time.time()
    print(f'total runtime DECENTRALISED: {(end_time - start_time)/60:.2f} min')



print('#################### CENTRALISED MODEL RUN #########################')

model2 = AbstractModel()


def simulation_with_delay2(data, model = model2):
    timer = TicTocTimer()
    timer.tic('start')
    #report_timing()
    instance = model.create_instance(data)
    timer.toc('Built model')

    #print('Solving')
    opt = SolverFactory('cplex', executable='/home/gridsan/jbecker/linux_cplex/download/cplex/bin/x86-64_linux/cplex')
    myoptions = dict()
    #myoptions['log'] = '200_profileA.log'
    myoptions['timelimit'] = t_lim # adapt for optimality gap
    results = opt.solve(instance, options=myoptions, tee=False) # solves and updates instance -> tee=False avoids printing results from CPLEX solver
    timer.toc('Time to solve')
    t1_stop = process_time()
    obj_val = value(instance.obj)
    manu_facilities = []
    for m in instance.m:
        if (value(instance.E1[m]) == 1):
            #print(f"Establish manufacturing facility {m}")
            manu_facilities.append(1)
        else:
            manu_facilities.append(0)
            
    
    # create a table to store the results
    result = np.array([obj_val, 
                       value(sum(instance.CTM[p] for p in instance.p))/len(instance.p), 
                       value(sum((instance.TRT[p]-19)*instance.DELAYPEN for p in instance.p))/len(instance.p), 
                       value(sum(instance.DELAY[p]*instance.MANUPEN for p in instance.p))/len(instance.p),
                       value(sum(instance.TTC[p] for p in instance.p))/len(instance.p), 
                       (10476+9312), 
                       obj_val/len(instance.p), 
                       value(instance.ATRT)])
    return result, manu_facilities

## Initialising model parameters & variables
# SETS
model2.c = Set() # Leukapheresis sites
model2.h = Set() # Hospitals
model2.j = Set() # Transport mode
model2.m = Set() # Manufacturing sites
model2.p = Set() # Patients
model2.t = RangeSet(130) # Time
model2.tt = Set(initialize=model2.t) # Alias of set t

# Indexed PARAMETERS
model2.CIM = Param(model2.m) # Capital investment for manufacturing facility
model2.FCAP = Param(model2.m) # Total capacity of manufacturing site
model2.TT1 = Param(model2.j) # Transport time LS to MS using transport mode j
model2.TT3 = Param(model2.j) # Transport time MS to hospital using transport mode j
model2.U1 = Param(model2.c, model2.m, model2.j) # Unit transport cost LS to MS using transport mode j
model2.U3 = Param(model2.m, model2.h, model2.j) # Unit transport cost MS to hospital using transport mode j
model2.INC = Param(model2.p, model2.c, model2.t, initialize=0) # Demand therapy p arriving for leukapheresis at LS site c at time t
model2.CVM = Param(model2.m, default={'m1':20920, 'm2':156900, 'm3':52300, 'm4':20920, 'm5':156900, 'm6':52300}) #Fixed variable costs


# Scalar PARAMETERS
model2.FMAX = Param() # Maximum flow
model2.FMIN = Param() # Minimum flow
model2.TAD = Param(within=NonNegativeReals) # Duration of administration
model2.TLS = Param(within=NonNegativeReals) # Duration of leukapheresis
model2.TMFE = Param(initialize=7) # Duration of manufacturing excluding QC


# Binary VARIABLES
model2.E1 = Var(model2.m, within=Binary) # 1 if manufacturing facility m is established
model2.X1 = Var(model2.c, model2.m, within=Binary) # 1 if a match between LS site c and MS site m is established
model2.X3 = Var(model2.m, model2.h, within=Binary) # 1 if a match between MS site m and a hospital h is established
model2.Y1 = Var(model2.p, model2.c, model2.m, model2.j, model2.t, within=Binary) # 1 if a sample p is transferred from a LS site c to a MS site m via mode j at time t
model2.Y3 = Var(model2.p, model2.m, model2.h, model2.j, model2.t, within=Binary) # 1 if a sample p is transferred from a MS site m to a hospital h via mode j at time t

# Integer variables
model2.INH = Var(model2.p, model2.h, model2.t, within=NonNegativeIntegers) # Therapy p arriving at hospital h at time t

# Positive VARIABLES
model2.CTM = Var(model2.p, within=NonNegativeReals)
model2.FTD = Var(model2.p, model2.m, model2.h, model2.j, model2.t, within=NonNegativeReals)
model2.TTC = Var(model2.p, within=NonNegativeReals)
model2.LSA = Var(model2.p, model2.c, model2.m, model2.j, model2.t, within=NonNegativeReals)
model2.LSR = Var(model2.p, model2.c, model2.m, model2.j, model2.t, within=NonNegativeReals)
model2.MSO = Var(model2.p, model2.m, model2.h, model2.j, model2.t, within=NonNegativeReals)
model2.OUTC = Var(model2.p, model2.c, model2.t, within=NonNegativeReals)
model2.OUTM = Var(model2.p, model2.m, model2.t, within=NonNegativeReals)
model2.INM = Var(model2.p, model2.m, model2.t, within=NonNegativeReals)
model2.DURV = Var(model2.p, model2.m, model2.t, within=NonNegativeReals) # 1 only for the time period t in which a therapy p is manufactured in facility m
model2.RATIO = Var(model2.m, model2.t, within=NonNegativeReals) # the percentage of utilisation of MS site m at time t


# VARIABLES
model2.TOTCOST = Var() # Total cost
model2.CAP = Var(model2.m, model2.t) # Capacity of MS m at time t
model2.TRT = Var(model2.p) # Total return time of therapy
model2.ATRT = Var() # Average return time
model2.STT = Var(model2.p) # Starting time of treatment for patient p
model2.CTT = Var(model2.p) # Completion time of treatment for patient p

# Extra parameter for delay
model2.DELAY = Param(model2.p, within=NonNegativeReals)
model2.DELAYPEN = Param(default=2000)
model2.MANUPEN = Param(default=1000)


## Objective function
def obj_rule(model2):
    return sum( model2.CTM[p] for p in model2.p )+ sum( model2.TTC[p] for p in model2.p ) + (10476+9312)*len(model2.p) #+ sum((model2.TRT[p]-19)*model2.DELAYPEN for p in model2.p)
    
model2.obj = Objective( rule=obj_rule )

## Constraints
# Manufacturing cost
# C1
# base case: 58000 $ 80% utilisation
def C1_rule(model2,p):
    return model2.CTM[p] == sum((model2.E1[m]*(model2.CIM[m]+model2.CVM[m]))*len(model2.t)/len(model2.p) for m in model2.m) + model2.DELAY[p]*model2.MANUPEN
model2.C1 = Constraint(model2.p, rule=C1_rule)


#RATIOEQ
def RATIOEQ_rule(model2,m,t):
    return model2.RATIO[m,t] == sum(model2.DURV[p,m,t]/model2.FCAP[m] for p in model2.p)
model2.RATIOEQ = Constraint(model2.m, model2.t, rule=RATIOEQ_rule)


#MSBnew
def MSBnew_rule(model2,p,m,t):
     return model2.DURV[p,m,t] == sum(model2.INM[p,m,tt-1]-model2.OUTM[p,m,tt] for tt in model2.tt if tt<=t and tt>1) + model2.OUTM[p,m,t] 
model2.MSBnew = Constraint(model2.p, model2.m, model2.t, rule=MSBnew_rule)


# Transport cost
#C2
def C2_rule(model2,p):
    return model2.TTC[p] == sum(model2.Y1[p,c,m,j,t]*model2.U1[c,m,j] for c in model2.c for m in model2.m for j in model2.j for t in model2.t) + sum(model2.Y3[p,m,h,j,t]*model2.U3[m,h,j] for m in model2.m for h in model2.h for j in model2.j for t in model2.t)
model2.C2 = Constraint(model2.p, rule=C2_rule)


#MSB1
def MSB1_rule(model2,p,c,t,tt):
    if tt == t + model2.TLS:
        return model2.INC[p,c,t] == model2.OUTC[p,c,tt]
    else:
        return Constraint.Skip
model2.MSB1 = Constraint(model2.p, model2.c, model2.t, model2.tt, rule=MSB1_rule)


#MSB2
def MSB2_rule(model2,p,m,t,tt):
    if tt == t + model2.TMFE + model2.DELAY[p]:
        return model2.INM[p,m,t] == model2.OUTM[p,m,tt]
    else:
        return Constraint.Skip
model2.MSB2 = Constraint(model2.p, model2.m, model2.t, model2.tt, rule=MSB2_rule)



#MSB8
def MSB8_rule(model2,p,m,t,tt):
    if tt == t + 7:
        return model2.OUTM[p,m,t] == sum(model2.MSO[p,m,h,j,tt] for h in model2.h for j in model2.j)
    else:
        return Constraint.Skip
model2.MSB8 = Constraint(model2.p, model2.m, model2.t, model2.tt, rule=MSB8_rule)


#MSB3
def MSB3_rule(model2,p,c,m,j,t,tt):
    if tt == t + model2.TT1[j]:
        return model2.LSR[p,c,m,j,t] == model2.LSA[p,c,m,j,tt]
    else:
        return Constraint.Skip
model2.MSB3 = Constraint(model2.p, model2.c, model2.m, model2.j, model2.t, model2.tt, rule=MSB3_rule)


#MSB4
def MSB4_rule(model2,p,m,h,j,t,tt):
    if tt == t + model2.TT3[j]:
        return model2.MSO[p,m,h,j,t] == model2.FTD[p,m,h,j,tt]
    else:
        return Constraint.Skip
model2.MSB4 = Constraint(model2.p, model2.m, model2.h, model2.j, model2.t, model2.tt, rule=MSB4_rule)


#MSB5
def MSB5_rule(model2,p,m,t):
    return model2.INM[p,m,t] == sum(model2.LSA[p,c,m,j,t] for c in model2.c for j in model2.j)
model2.MSB5 = Constraint(model2.p, model2.m, model2.t, rule=MSB5_rule)


#MSB6
def MSB6_rule(model2,p,h,t):
    return model2.INH[p,h,t] == sum(model2.FTD[p,m,h,j,t] for m in model2.m for j in model2.j)
model2.MSB6 = Constraint(model2.p, model2.h, model2.t, rule=MSB6_rule)


#MSB7
def MSB7_rule(model2,p,c,t):
    return model2.OUTC[p,c,t] == sum(model2.LSR[p,c,m,j,t] for m in model2.m for j in model2.j)
model2.MSB7 = Constraint(model2.p, model2.c, model2.t, rule=MSB7_rule)


# Capacity equation
#CAP1
def CAP1_rule(model2,m,t):
    return model2.CAP[m,t] == model2.FCAP[m]-sum(model2.INM[p,m,tt] for p in model2.p for tt in model2.tt if tt<t and tt>=t-model2.TMFE)
model2.CAP1 = Constraint(model2.m, model2.t, rule=CAP1_rule)


# Capacity constraint
#CAPCON1
def CAPCON1_rule(model2,m,t):
    return sum(model2.INM[p,m,t] for p in model2.p)-sum(model2.OUTM[p,m,t] for p in model2.p) <= model2.CAP[m,t]
model2.CAPCON1 = Constraint(model2.m, model2.t, rule=CAPCON1_rule)


# Constraints to ensure that no matches are established with non-existent facilities
#CON1
def CON1_rule(model2):
    return sum(model2.E1[m] for m in model2.m) <= 2
model2.CON1 = Constraint(rule=CON1_rule)


#CON2
def CON2_rule(model2,c,m):
    return model2.X1[c,m] <= model2.E1[m]
model2.CON2 = Constraint(model2.c, model2.m, rule=CON2_rule)


#CON3
def CON3_rule(model2,m,h):
    return model2.X3[m,h] <= model2.E1[m]
model2.CON3 = Constraint(model2.m, model2.h, rule=CON3_rule)


#CON4
def CON4_rule(model2,p,c,m,j,t):
    return model2.Y1[p,c,m,j,t] <= model2.X1[c,m]
model2.CON4 = Constraint(model2.p, model2.c, model2.m, model2.j, model2.t, rule=CON4_rule)


#CON5
def CON5_rule(model2,p,m,h,j,t):
    return model2.Y3[p,m,h,j,t] <= model2.X3[m,h]
model2.CON5 = Constraint(model2.p, model2.m, model2.h, model2.j, model2.t, rule=CON5_rule)


#CON6
def CON6_rule(model2,p):
    return sum(model2.Y1[p,c,m,j,t] for c in model2.c for m in model2.m for j in model2.j for t in model2.t) == 1
model2.CON6 = Constraint(model2.p, rule=CON6_rule)


#CON7
def CON7_rule(model2,p):
    return sum(model2.Y3[p,m,h,j,t] for m in model2.m for h in model2.h for j in model2.j for t in model2.t) == 1
model2.CON7 = Constraint(model2.p, rule=CON7_rule)


# Demand satisfaction
#DEM
def DEM_rule(model2):
    return sum(model2.INH[p,h,t] for p in model2.p for h in model2.h for t in model2.t) <= len(model2.p)
model2.DEM = Constraint(rule=DEM_rule)


# Flow constraints
#CON8
def CON8_rule(model2,p,c,m,j,t):
    return model2.LSR[p,c,m,j,t] >= model2.Y1[p,c,m,j,t]*model2.FMIN
model2.CON8 = Constraint(model2.p, model2.c, model2.m, model2.j, model2.t, rule=CON8_rule)


#CON9
def CON9_rule(model2,p,c,m,j,t):
    return model2.LSR[p,c,m,j,t] <= model2.Y1[p,c,m,j,t]*model2.FMAX
model2.CON9 = Constraint(model2.p, model2.c, model2.m, model2.j, model2.t, rule=CON9_rule)


#CON10
def CON10_rule(model2,p,m,h,j,t):
    return model2.MSO[p,m,h,j,t] >= model2.Y3[p,m,h,j,t]*model2.FMIN
model2.CON10 = Constraint(model2.p, model2.m, model2.h, model2.j, model2.t, rule=CON10_rule)


#CON11
def CON11_rule(model2,p,m,h,j,t):
    return model2.MSO[p,m,h,j,t] <= model2.Y3[p,m,h,j,t]*model2.FMAX
model2.CON11 = Constraint(model2.p, model2.m, model2.h, model2.j, model2.t, rule=CON11_rule)


#CON12
def CON12_rule(model2,p):
    return sum(model2.Y3[p,m,'h1',j,t] for m in model2.m for j in model2.j for t in model2.t) == sum(model2.INC[p,'c1',t] for t in model2.t)
model2.CON12 = Constraint(model2.p, rule=CON12_rule)


#CON13
def CON13_rule(model2,p):
    return sum(model2.Y3[p,m,'h2',j,t] for m in model2.m for j in model2.j for t in model2.t) == sum(model2.INC[p,'c2',t] for t in model2.t)
model2.CON13 = Constraint(model2.p, rule=CON13_rule)


#CON14
def CON14_rule(model2,p):
    return sum(model2.Y3[p,m,'h3',j,t] for m in model2.m for j in model2.j for t in model2.t) == sum(model2.INC[p,'c3',t] for t in model2.t)
model2.CON14 = Constraint(model2.p, rule=CON14_rule)


#CON15
def CON15_rule(model2,p):
    return sum(model2.Y3[p,m,'h4',j,t] for m in model2.m for j in model2.j for t in model2.t) == sum(model2.INC[p,'c4',t] for t in model2.t)
model2.CON15 = Constraint(model2.p, rule=CON15_rule)


#START
def START_rule(model2,p):
    return model2.STT[p] == sum(model2.INC[p,c,t]*t for c in model2.c for t in model2.t)
model2.START = Constraint(model2.p, rule=START_rule)


#END
def END_rule(model2,p):
    return model2.CTT[p] == sum(model2.INH[p,h,t]*t for h in model2.h for t in model2.t)
model2.END = Constraint(model2.p, rule=END_rule)


#TIME
def TIME_rule(model2,p):
    return model2.TRT[p] == model2.CTT[p] - model2.STT[p]
model2.TIME = Constraint(model2.p, rule=TIME_rule)


#ATIME
def ATIME_rule(model2):
    return model2.ATRT == sum(model2.TRT[p] for p in model2.p)/len(model2.p)
model2.ATIME = Constraint(rule=ATIME_rule)


#TSEQ
def TSEQ_rule(model2,p):
    return model2.STT[p] <= model2.CTT[p]
model2.TSEQ = Constraint(model2.p, rule=TSEQ_rule)


# %%


if __name__ == '__main__':
    start_time = time.time()
    num_processes = 24 # use 24! crashes with >48 processes
    chunk_size = n_delays // num_processes

    # create the multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    # create an array of the arguments you want to pass to simulation_with_delay, where each argument is a data file from temp_delays folder
    datas = [f'/home/gridsan/jbecker/thesis_code/normal/temp_delays/data{i+1}.dat' for i in range(n_delays)]
    # use the pool.map function to run the simulation_with_delay function on all the arguments in parallel
    results = pool.imap(simulation_with_delay2, datas)
    pool.close()
    pool.join()
    
    run_results = pd.DataFrame(columns=['Total cost', 
                                    'Average manufacturing cost per therapy', 
                                    'Average TRT delay penalty per therapy',
                                    'Average manufacturing delay penalty per therapy', 
                                    'Average transport cost per therapy', 
                                    'Average QC cost per therapy', 
                                    'Average cost per therapy', 
                                    'Average return time'])
 
    manu_config = pd.DataFrame(columns=['m1', 'm2', 'm3', 'm4', 'm5', 'm6'])
    trans_config = pd.DataFrame(columns=['transport_mode1', 'transport_mode2'])

        
    
    for result in results:
        # Convert result[0] into a DataFrame with the correct column names
        result_df = pd.DataFrame([result[0]], columns=run_results.columns)

        # Append the data to run_results
        run_results = pd.concat([run_results, result_df], ignore_index=True)

        # Convert result[1] into a DataFrame with the correct column names for manu_config
        manu_config_df = pd.DataFrame([result[1]], columns=manu_config.columns)

        # Append the data to manu_config
        manu_config = pd.concat([manu_config, manu_config_df], ignore_index=True)
    
    run_results.to_csv(f'{date}_normal_c_{n}.csv')
    manu_config.to_csv(f'{date}_normal_c_manu_{n}.csv')
    end_time = time.time()
    print(f'total runtime: {(end_time - start_time)/60:.2f} min')


# %%





# %%




