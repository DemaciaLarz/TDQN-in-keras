# Implementing TDQN in Keras
Consider as the basic building block of this project the Trading Deep Q-Network algorithm (TDQN) as it is put forward in the paper named: An Application of Deep Reinforcement Learning to Algorithmic Trading which you can find [here](https://arxiv.org/abs/2004.06627).

#### Table of Contents
1. Objectives
2. Underlying Assets - Data
3. User-Values / Downstream Application
4. Content
5. Results
6. TDQN Implementation Notes

### 1 Objectives
The objective was to implement the TDQN algorithm on a set of shares from the upcoming hydrogen sector in order to obtain valuable insights into market movements. 

As it turned out it got applied to historical gold prices for the initial training, and then quite successfully to one hydrogen stock, Powercell Sweden. Here are some nice results:

![Powercell results](https://github.com/DemaciaLarz/implementing-TDQN-in-keras/blob/main/files/img_results_powercell_1.png "Powercell results")

### 2 Underlying Assets - Data
Powercell Sweden is a fuel cell manufacturer listed on the First North GM Sweden market, [here](http://www.nasdaqomxnordic.com/aktier/microsite?Instrument=SSE105121&name=PowerCell%20Sweden) is information on the share and the historical prices, and [here](https://www.powercell.se/en/start/) is info on the company.

You can find analysis and preprocessing as it relates to this project of the actual data [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_2_powercell.html).

When it comes to gold, [here](https://www.kaggle.com/omdatas/historic-gold-prices) are the historical prices, and [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_1_gold.html) is the analysis from this project.

### 3 User-Values / Downstream Application
The use-case is to apply one or more successfully trained models such that they are able to bring some actual useful intel on a daily basis when it comes to the Powercell share movements.

This is achieved through an application, in which daily results based on the two models’ actions alongside the underlying asset as a baseline is being presented. The results are obtained by running an inference procedure as per the [pipeline.py]() script. 

In a nutshell, a cronjob runs the script around 5:00 am each morning. It collects yesterday’s closing data through some selenium code, performs the inference procedure, and updates a set of databases so that the actions of the two models can get interpreted as buy or sell signals in perfect time a few hours before the markets open.

This could provide some opportunity. Either by simply dedicating a bit of capital for the purpose and executing the daily trades each day as per one’s favorite model, or by using the model behavior as merely additional intel in one’s own decision-making process. 

Executing trades automatically through some API has always been outside the scope of this project. The pursuit has rather been towards a more intelligent trading screen. 

Below is a screenshot from the application. It can at the time of writing be found [here](http://35.158.207.95/).

![Application img](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_application.png "Application 1")

### 4 Content
* train.py is the code on which the most successful model was trained. It takes Powercell CSV data and trains a TDQN agent.
* pipeline.py is the inference procedure.
* the two model files are saved Tensorflow Keras model objects. These are the two models in the application.
* in the notebooks folder there are some notebooks on preprocessing, results and training.
* the CSV file is Powercell data up until 2020-09-25.

### 5 Results
On **gold**, the first results that came in are [these](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/results_1_gold.html). What really made the difference from flat to actual learning were a proper implementation of the X2 state, and a reward clipping procedure. See more about this in the TDQN implementation notes below. 

You can follow the training of a gold model [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/training_1_gold.html). 

After some further runs the following results were obtained:

![Gold results image](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_results_gold.png "Gold results image")

On **Powercell**, we got numerous results worth mentioning:

![Powercell results 2](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_results_powercell_2.png "Powercell results 2")

These stood strong even though further attempts were made later on to achieve better, [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/results_4_powercell%20.html) are some of them.

The "BH" is the baseline, the Powercell shares actual movements scaled to the portfoilio value. 

[Here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/notebooks/htmls/training_2_powercell.html) is a notebook in which one can follow the training of a Powercell model.

### 6 TDQN Implementation Notes

#### 6.1 State Representation
* in general, the X1 state
* specifically about the inner X2 state, the normalizing and the one-hot things

#### 6.2 Actions
* formulating the long and short position
* multiple actions, dueling
reducing the action space, three fold reason

#### 6.3 Scalar Reward Signal
* the portfolion return
* all the transforming, normalizing and clipping procedures

#### 6.4 Data Augmentation


#### 6.5 Model Architecture
* FFNN, comment on LSTM and dueling
* capacity
* batchnorm vs dropout
* seperate input streams

#### 6.6 Hyperparameters
* alpha, very, very sensitive. only the current is working.
* gamma, kept quite constant.
* batchsize
* num steps were very important
* weight update

#### 6.7 Loss and Optimizer
* huber and adam. Tried others but nah

#### 6.8 Selected issues
* diverging q values, the single largest obstacle
* 
*
