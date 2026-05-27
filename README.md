# Bayesian ML Human Trafficking

**Date:** March 29, 2026

**Group members:** Bhavya Sharma, Jiwon Choi, Noah Fisher, Sieon Lee, and Mateo Galvis

## Agenda
1. [Problematic](#problematic)
2. [Datasets](#datasets)
3. [EDA](#eda) 
4. [Methodology](#methodology)
    * [Model 1: Bayesian State Space Decomposition](#model-1-bayesian-state-space-decomposition)
    * [Model 2: Hierarchical Beta Binomial](#model-2-hierarchical-beta-binomial)
    * [Model 3: Bayesian Resource Optimizer](#model-3-bayesian-resource-optimizer)
5. [Conclusions and next steps](#conclusions-and-next-steps)

## Problematic
Human trafficking is a silent pandemic that our world is facing with 49.6 M victims in modern
slavery globally. Almost 80% of victims globally face a gap in access to reintegration services
after rehabilitation. Fragmented case management data across agencies exist in silos and there is
no comprehensive solution to this problem. We look at the data on victim services in the United
States to understand the trends in this sector.

## Datasets

## EDA

![image](figures/total_signal_potential_situation.png)<br>
The timeseries plot shows that the Total Signals have been increasing since 2013 until 2021, with a downward trend from 2021 to 2024. However, the potential situations identified got their highest value in 2024, with 12K victims found, despite a negative trend from 2019 to 2023, probably related to the COVID period.

![image](figures/proportion_signals_reported.png)<br>
The proportion of signals reported by potential victims has constantly decreased over the last decade, reaching a minimum of 5%. This timeseries plot suggests that the victims have been less listened to, depending more on witnesses, family, friends, law enforcement, or anonymous reporters.


![image](figures/ovc_trafficking_chart.png)<br>
The OVC displays that in recent years, the most common cause is sex trafficking, with an increase of 15K victims in 2024 compared with 2023, reaching 90K annually, followed by labor trafficking, with 2403 in 2024, in contrast with 2056 in 2023. It is essential that governance start tracking these metrics to develop more efficient public policies that reduce the number of victims.


![image](figures/gender_distribution_age.png)<br>
The pie chart shows that 72.8% of human traffic is female. Taking a closer look, by age, the most affected are 9-17 and 18-20 year olds, the most likely target victims are teenagers and young adults. In the case of Men, they represents 27% of the human trafficking cases, and the most frequent age group is 18-20 years old.


## Methodology

### Model 1: Bayesian State Space Decomposition
The idea behind developing this type of model was to identify how potential situations of human trafficking are evolving over time and whether public policy decisions are improving the identified cases. Therefore, the FOSTA policy intervention of 2018 was taken as a reference.

The Bayesian structural time series implemented was using PyMC:
   1.	Separating trend + policy effect + noise
   2.	Estimating everything probabilistically with MCMC (NUTS)
   3.	Extracting a "cleaned" residual signal after removing the structure
      
$$y_t = \text{trend}_t + \text{policy effect}_t + \text{noise}_t$$
* $$\text{trend}_t =$$ slowly evolving hidden baseline
* $$\text{policy effect}_t =$$ step change starting in 2018
* $$\text{noise}_t =$$ observation randomness

**Results:**

![image](figures/posterior_distribution.png)<br>
Posterior Distribution: The model estimates that FOSTA increased situations identified by 0.34, but the HDI is between -1.58 and +2.20, which crosses zero, leading to uncertainty about the impact of the policy.<br>

![image](figures/trace_plot.png)<br>
Trace plot: All the chains agree on the same shape, indicating strong convergence, but like the previous graph, it suggests the model leans toward a positive policy effect, but the zero value appears in the interval, the variance is too wide to be conclusive. 

![image](figures/obs_vs_pred.png)<br>
The observation vs. prediction time series shows great performance, the model was able to learn the trend, the level, handle the policy effect, and control the noise in the data.

This type of model can help decompose and evaluate how decisions and resources related to human trafficking assistance impact victims.


### Model 2: Hierarchical Beta Binomial

### Model 3: Bayesian Resource Optimizer

## Conclusions and next steps

