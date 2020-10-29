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
* helpers/base.py contains the prioritized experience replay buffer.
* helpers/data.py gets the data and preprocess it.
* helpers.hydro.py holds various helper functions.
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
The agents observations consist at each timestep of the following: 
* S(t) represents the agents inner state.
* D(t) is the information concerning the OHLCV (Open-High-Low-Close-Volume) data characterising the stock market.
* I(t) represent a number of technical indicators.

This is what was used in order to get the results baove. In some experiements macro related information N(t), and information about date, week and year T(t) also were used. In those particular settings though it disturbed the training to much for any results to appear.

You will find in the context of this implementation that D(t) and I(t) bot are in the X1 input stream. Preprocessed and normalized as one they are fed into the model a single input. This was very beneficial. A number of settings with seperatae input streams for various types of observation data was experemented on without success.

When it comes to the agents inner state S(t) it is made up by three parts:
1. cash - the amount of cash the agent has at its disposal at timestep t.
2. stock value - the value at timestep t of the stock which the agent is sitting on.
3. number of stock - the number of stock that the agent holds.

One question that arised was how to encode this properly such that core pieces of information does not get lost. Many variations were tested. 

First of all the cash and stock value made sense to simply put in a two dimensional vector and apply some normalizer of choice and they would under all circumstances maintain their opposite proportions meanwhile bottoming out when zero, this would signal to the agent which kind of position it is capable to formulate. 

A number of scaling methods were tested, standard normal scaling, removing the mean and scale to unit variance is what worked best. The reason is simply the stability in training that it brings which allows for results to maybe or maybe not come to light. 

When it comes to the number of stock, it can easily become very large depending on the price of the stock and the initial cash value the agent is allowed to start trading with. This led me to be intitally suspicious about just clamping it in there with the cash and stock value. Hence a setup in which information about the number of stock was encoded via a one-hot procedure. 

This was made possible by adjusting the initial cash such that the number of stock were doomed to fall in between a managable range between zero and one hundred. Obviously this approach is sub optimal, however we were able to get our first results on this setup. 

At this point the cash and stock value doublet was fed into the model via their own input stream which in the code is denoted x2a, while the one-hot number of stock vector had its own stream, denoted x2b. Together the inner state of the model, S(t) is denoted as the X2 stream.

In the end though, it turned out that the simplest path often is the most beneficial. As of the best performing models, the number of stock, x2b, is simply preprocessed with the x2a, the cash and stock value such that they get normalized together in an online fashion. 

However, this setup is currently bounded by the fact that the initial cash is set to 10 000. Even though not pursuid fully, a couple of attempts were made to raise other arbitrary initial values with very poor outcomes. Rationale one can assume would be that proportions matter in terms of encoding information about the agents inner state.

#### 6.3 Scalar Reward Signal
The portfolio value is the sum of cash and stock value, and the strategy as per the paper is to provide daily returns as rewards for reasons proposed one has to say makes sense. After numerous experiments with a number of varying rewards schemes, clipping the rewards to (-1, 0, 1) really made the difference in terms of creating the stability of training necessary.

During most part of the early project we suffered hard from diverging Q values during training. By that we mean that as training progressed the action values kept growing apart. In effect this gave rise to a sub optimal / useless dominant either buy or sell strategy. The reward signal was idnetified as a prosperous angle to attack the problem from. Here are some the attempts:
* episodic scaling - holding all the transiitons until the end of the episode allows for the applience applying some scaling procedure on the block of rewards, and only to after that store all of it into memory. 
* scaling - we did alot of scaling, unit norm, log, min max, standard, standard without shifting the mean. The same thing with respect to larger polulations of rewards from memory and from preivous runs and so on. Still unstable training.
* clipping - when we added clipping into the mix, things changed. The initial hesitance came from the obvious risk of loosing valuable information by making the rewards much more sparse. Given that clipping the rewards added to the rise of a very stable training, trading off the potential of loosing information was not a difficult desicion to make. Recall also, that information about all the subtelties about the changes in portfolio value is in fact encoded into the X2 stream. This is in turn propagated through the network and has assumably very much infleunce over the estimated action values. One can perhaps express it such that the TD error is simply kept in line by a sparse reward clipping procedure, meanwhile the inner state provides the intel needed for the actual position taking. In any case it is what worked for us, so we ran with it.

#### 6.2 Actions

**The action:** at each timestep t the agent executes a trading action as per its policy. It can buy, sell or in effect hold its position. In a DQN context one can consider the chain of the action from the estimated Q values, to an action-preference throuhg an argmax operation (or whatever policy is in effect), Whereas the action-preference goes into the TDQN position formulation procedure and out comes the trading action of its choice. This trading action ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img4.png) is the actual action of the agent, and it can be either long or short.

**Effects of the action:** each trading action has an effect on both of the two components of the portfolio value, that is cash and stock value. Here are their update rules:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img3.png)

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img2.png)

This can be understood as such that the cash value change with the cash needed for the taken position. Whereas the stock value is made up from the number of stocks owned post the trade times the price.

**Simplifying the short position:** moving on to the formulaiton of the ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img4.png) long and short positions. A modification made within this project was to simplify the short position, such that the agent is not able to sell any shares it does not hold. Nevertheless the lack of opportunity this would bring, the reasons for the simplification of the short position in this project are as follows:
1. user-value - our downstream use-case will not really be in a position to capitalize on any opportunity brought by the agent through the execution of a short position, mainly due to access, price and hazzle.
2. accuracy - what is the proper market volatility parameter ![m5](http://www.sciweavers.org/tex2img.php?eq=%5Cepsilon&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=), and what is the cost estimation for shorting the stock? It is not something of immedieate clarity.
3. comfort - it reduces complexity in the implementation, makes it easier.

**The long and short positions:** with this in mind, the long and short positions can now described. Note that it is a reduced action space that is in question now, one in which there only exist buying or selling everything each trade, nothing in between. Note furthermore that this is the setup our best models ran on. However, we also experiemnted with larger action spaces, see more below.

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img6.png)

![m7](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img1.png)

This can be understood such that the long position represents the maximum number of shares it can get out of the cash that the agent currently holds and with transaction costs taken into account. The short position is simply the number of shares held by the agent. 

**Expanding the action space**
Consider the reduced action space as a set that consist of exactly one long and one short position:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img8.png)

Consider furthermore the set of all legal actions the agent can take:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img9.png)

The expression above can be interpreted such that each integer between 







* formulating the long and short position
* multiple actions, dueling
reducing the action space, three fold reason


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


![alt text](http://www.sciweavers.org/tex2img.php?eq=%3Cb%3E%3Ci%3Ep_t%28x%5E2%29%20%3D%205%20%2A%20%5Calpha%3C/i%3E%3C/b%3E&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)
