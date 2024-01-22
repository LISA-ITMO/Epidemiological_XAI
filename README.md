# Research and analysis of interpretable AI and ML methods for modeling disease dynamics during an epidemic.  

## Abstract  
In recent years, the growing popularity of Explainable Artificial Intelligence (XAI) and Machine Learning (ML) techniques in epidemiology has underscored the critical need to enhance our understanding of these transparent and interpretable models for effective public health decision-making. This study explores the feasibility of working with PINN (Physics-Informed Neural Networks) in pair with boosting methods, CatBoost in this study case, using SHAP (SHapley Additive exPlanations). In addition, search engine trend data was used to gain a better view and improve predictions as a measure of social sentiment. 

## Introduction
TODO


## Methods

### Used Data  
In study two data sources were used: historical data for 220 days on the number of Infected, Recovered and Dead cases (IRD data) and GoogleTrends data on 14 search requests. Setting n-days to predict, SIDR and GoogleTrends data was taken to show situation n days before prediction.      
##### PINN Data
Data preprocessing for PINN consists of MinMax normalisation only, after which it is fed to the neural network. In study case we perform training cycle on 190 days and predict number of SIDR cases in next 30 days with PINN. 
##### CatBoost Data
CatBoost utilises 3 data sources: historical SIDR data, forecast and fitted PINN data and Google Trends statistics.
For this model, each feature was smoothed using the Savitzky-Golay filter ([wiki](https://en.wikipedia.org/wiki/Savitzky–Golay_filter), [paper](https://eigenvector.com/wp-content/uploads/2020/01/SavitzkyGolay.pdf)), which significantly improved the performance of the model, reducing the MAPE from 0,39 to 0,27. The window was set to the number of days and the order of the polynomial was set to 5, as this showed the best performance. 


### Used models
#### PINN - [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)

<p align="center">
<img src="https://i.ibb.co/y8d4fSP/unnamed2.png" alt="drawing2" width="100%"/>   
</p>

To build confidence when using the DL model, we can modify our loss function so that the model behaves better and more meaningfully. Based on one or a system of differencial equations. In this case the a system of differential equations knowns as SIRD mathematical model is used. This equation contains the variables S(Susceptible), I(Infected), R(Recovered), D(Dead) and coefficients alpha, beta and gamma to represent transmission, recovery and mortality rate, respectivelly. 

<p align="center">
  <img src="https://i.ibb.co/VBK49w3/unnamed3.png" alt="drawing" width="200"/>
</p>

Hence, loss is formed as sum of RMSE and functions shown earlier:  
<p align="center">
$Loss = u * loss_N + (1-u) * loss_F, u-$ regularization rate,
$loss_N = mean(S-S')^2 + mean(I-I')^2 + mean(R-R')^2+ mean(D-D')^2$  
$loss_F = mean(f_S)^2 + mean(f_I)^2 + mean(f_R)^2 + mean(f_D)^2$  
</p>


Neural Net itself features 4 fully-connected hidden layers with ReLU activation function. _(Optionally, ReLU can be changed to tanh as performance depends on prediction length and might vary.)_ 

#### CatBoost - [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)

<p align="center">
<img src="https://i.ibb.co/rcjqChs/unnamed4.png" alt="drawing2" width="100%"/>   
</p>

The data and preprocessing methods used have been described previously. Within the model, two methods, GridSearch with cross-validation and FeatureSelection based on SHAP values, were used to improve the results. The ordering of these methods was set according to the best performance. Due to the SHAP architecture, it is not possible to reproduce and show the same result every time. However, the feature selection method was set to select 7 features. This reduced the MAPE from 0.07 to 0.04, which is the best result that could be obtained.  

### Intepretation of results
TODO
<p align="center">
<img src="https://i.ibb.co/7JFS65y/unnamed5.png" alt="drawing2" width="100%"/>   
</p>

## Results
TODO
<p align="center">
<img src="https://i.ibb.co/NWKR40p/end-graph.png" alt="drawing2" width="100%"/>   
</p>

## Discussion
TODO


 


