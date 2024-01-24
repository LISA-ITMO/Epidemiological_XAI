# Research and analysis of interpretable AI and ML methods for modeling disease dynamics during an epidemic.  

## Abstract  
In recent years, the growing popularity of Explainable Artificial Intelligence (XAI) and Machine Learning (ML) techniques in epidemiology has underscored the critical need to enhance our understanding of these transparent and interpretable models for effective public health decision-making. This study explores the feasibility of working with PINN (Physics-Informed Neural Networks) in pair with boosting methods, CatBoost in this study case, using SHAP (SHapley Additive exPlanations). In addition, search engine trend data was used to gain a better view and improve predictions as a measure of social sentiment. The result shows average 0.03 MAPE amongst 30 days of prediction in best scenario predicting pre-peak scenario. 

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/xai_pipeline.png" alt="drawing2" width="100%"/>  
    <p align="center">Final model pipeline</p>
</div>

## Methods

### Used Data  
In study two data sources were used: historical data for 220 days of covid-19 epidemy starting 05.07.2020 on the number of Infected, Recovered and Dead cases (IRD data) and GoogleTrends data on 24 search requests. SIRD and GoogleTrends data was taken from earlier period to show situation before prediction.      
#### PINN data preprocessing
Data preprocessing for PINN consists of MinMax normalisation only, after which it is fed to the neural network.  
#### CatBoost data preprocessing
CatBoost utilises 3 data sources: historical SIRD data, forecast and fitted PINN data and Google Trends statistics.
For this model, each feature is normalized via MinMax and smoothed using the Savitzky-Golay filter ([wiki](https://en.wikipedia.org/wiki/Savitzky–Golay_filter), [paper](https://eigenvector.com/wp-content/uploads/2020/01/SavitzkyGolay.pdf)), which significantly improves the performance of the model, reducing the MAPE from 0,39 to 0,27. For the smoothing filter window is set to the number of days and the order of the polynomial is set to 5, as it showed the best performance. 



### Used models
#### PINN - [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/pinn_pipeline.png" alt="drawing2" width="100%"/>  
    <p align="center">PINN pipeline</p>
</div>

To build confidence when using the DL model, we can modify our loss function so that the model behaves better and more meaningfully. The idea behind PINN is to provide the model with an understanding of a particular physical process passing one or a system of differencial equations. In this case the a system of differential equations is known as [SIRD mathematical model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) is used. This equation contains the variables S(Susceptible), I(Infected), R(Recovered), D(Dead) and coefficients alpha, beta and gamma to represent transmission, recovery and mortality rate, respectivelly. 

<p align="center">

$$
\begin{cases}
\displaystyle\frac{dS}{dt} = - \beta S I, \\
\displaystyle\frac{dI}{dt} = \beta S I - (\alpha + \gamma )I, \\ 
\displaystyle\frac{dD}{dt} = \gamma I, \\  
\displaystyle\frac{dR}{dt} = \alpha I.  
\end{cases}
$$
  
</p>

Hence, loss is formed as sum of RMSE and functions shown earlier: 

<p align="center"> 
  
$$
\begin{cases}
Loss = u * loss_N + (1-u) * loss_F, \quad u - regularization \ rate \\ 
loss_N = mean(S-S')^2 + mean(I-I')^2 + mean(R-R')^2+ mean(D-D')^2 \\ 
loss_F = mean(f_S)^2 + mean(f_I)^2 + mean(f_R)^2 + mean(f_D)^2
\end{cases}
$$
  
</p>

Neural Net itself features 4 fully-connected hidden layers with ReLU activation function. _(Optionally, ReLU can be changed to tanh as performance depends on prediction length and might vary.)_   
In study case we perform training cycle on 190 days and predict number of SIDR cases in next 30 days with PINN. It is worth noting that the PINN architecture allows for long-term predictions to be made e.g. with 160 training points up to 200 days of prediction having reasonable performance. Total RMSE for the model is 5180.

#### CatBoost - [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/catboost_pipeline.png" alt="drawing2" width="100%"/>  
    <p align="center">CatBoost pipeline</p>
</div>


The data and preprocessing methods used have been described previously. Within the model, two methods, GridSearch with cross-validation and FeatureSelection based on SHAP values, were used to improve the results. The ordering of these methods was set according to the best performance. Due to the SHAP architecture, it is not possible to reproduce and show the same result every time. However, the feature selection method was set to select 24 features. This reduced the RMSE from 3939 to 3555, which is the best result that could be obtained. 
Experiments were made with the loss funcion and various modifications and ways of tuning the bousting model. The best RMSE (3555) was shown having MAPE as loss function. It is worth noting that the second best result (5% higher loss) was shown without PINN predictions in features.


## Results
The best result to achieve by RMSE is 3,555 as average in predicted 30 days (0,03 MAPE). As we can see raw CatBoost model showed 27% better result and ensemble of CatBoost and PINN gives 31% improvement. 

| Model                     	| Tuning&ast; 	| Loss function 	| RMSE     	| Loss change&ast;&ast; (%) 	|
|---------------------------	|-------------	|---------------	|----------	|---------------------------	|
| CatBoost with PINN        	|     GCV     	|      RMSE     	|   5632   	| +8.73%                    	|
| CatBoost without PINN     	|     GCV     	|      MAPE     	|   5523   	| +6.62%                    	|
| **Raw PINN**              	|      -      	|  Custom RMSE  	| **5180** 	| 0.00%                     	|
| CatBoost without PINN     	|   GCV + SF  	|      MAPE     	|   5154   	| -0.50%                     	|
| CatBoost with PINN        	|   GCV + SF  	|      RMSE     	|   4988   	| -3.70%                     	|
| CatBoost without PINN     	|     GCV     	|      RMSE     	|   4576   	| -11.66%                   	|
| CatBoost with PINN        	|     GCV     	|      MAPE     	|   3939   	| -23.95%                   	|
| **CatBoost without PINN** 	|   GCV + SF  	|      RMSE     	| **3753** 	| **-27.54%**               	|
| **CatBoost with PINN**    	|   GCV + SF  	|      MAPE     	| **3555** 	| **-31.37%**               	|

&ast; GCV - Grid Search with Cross Validation, SF - Select Features method,  
&ast;&ast; In comparison with raw PINN result.

                                    
<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/cb_wo_pinn_graph.png" alt="drawing2" width="100%"/>   
    <p align="center">Best CatBoost result without PINN</p>
</div>

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/cb_w_pinn_graph.png" alt="drawing2" width="100%"/>   
    <p align="center">Best CatBoost result with PINN</p>
</div>

### Intepretation of results
Earlier it was defined that crucial part of this study is to go through interpretation process and to gain confidence in the prediction. One might say that PINN has played more supportive role, as results for CatBoost without one are only 5% worse. Nevertheless, as loss is calculated as average of 30 days the raw CatBoost prediction is more linear and does not represent unevenness of new infected cases. Moreover, even 5% are crucial in such sphere. 

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/shap.jpg" alt="drawing2" width="100%"/>   
    <p align="center">10 most valuable features by SHAP interpretation for CatBoost with PINN</p>
</div>

<div class="img-with-text">
    <img src="https://github.com/LISA-ITMO/Epidemiological_XAI/blob/main/img/shap_wo_pinn_2.jpg" alt="drawing2" width="100%"/>   
    <p align="center">10 most valuable features by SHAP interpretation for CatBoost without PINN</p>
</div>

## Discussion
I'd be delighted to discuss the work and answer your questions.  
Email: tsowehh@gmail.com   
[<img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" width="64" height="60">](https://github.com/ssenichev)



 


