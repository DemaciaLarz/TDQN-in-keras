# Implementing TDQN in Keras
Consider as the basic building block of this project the Trading Deep Q-Network algorithm (TDQN) as it is put forward in the paper named: An Application of Deep Reinforcement Learning to Algorithmic Trading which you can find [here](https://arxiv.org/abs/2004.06627).

#### Table of Contents
1. Objectives
2. Underlying Assets - Data
3. User-Values / Downstream Application
4. Content
5. Results
6. TDQN Implementation Notes

## 1 Objectives
The objective was to implement the TDQN algorithm on a set of shares from the upcoming hydrogen sector in order to obtain valuable insights into market movements. 

As it turned out it got applied to historical gold prices for the initial training, and then quite successfully to one hydrogen stock, Powercell Sweden. Here are some nice results:

![Powercell results](https://github.com/DemaciaLarz/implementing-TDQN-in-keras/blob/main/files/img_results_powercell_1.png "Powercell results")

## 2 Underlying Assets - Data
Powercell Sweden is a fuel cell manufacturer listed on the First North GM Sweden market, [here](http://www.nasdaqomxnordic.com/aktier/microsite?Instrument=SSE105121&name=PowerCell%20Sweden) is information on the share and the historical prices, and [here](https://www.powercell.se/en/start/) is info on the company.

You can find analysis and preprocessing as it relates to this project of the actual data [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_2_powercell.html).

When it comes to gold, [here](https://www.kaggle.com/omdatas/historic-gold-prices) are the historical prices, and [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_1_gold.html) is the analysis from this project.

## 3 User-Values / Downstream Application
The use-case is to apply one or more successfully trained models such that they are able to bring some actual useful intel on a daily basis when it comes to the Powercell share movements.

This is achieved through an application, in which daily results based on the two models’ actions alongside the underlying asset as a baseline is being presented. The results are obtained by running an inference procedure as per the [pipeline.py]() script. 

In a nutshell, a cronjob runs the script around 5:00 am each morning. It collects yesterday’s closing data through some selenium code, performs the inference procedure, and updates a set of databases so that the actions of the two models can get interpreted as buy or sell signals in perfect time a few hours before the markets open.

This could provide some opportunity. Either by simply dedicating a bit of capital for the purpose and executing the daily trades each day as per one’s favorite model, or by using the model behavior as merely additional intel in one’s own decision-making process. 

Executing trades automatically through some API has always been outside the scope of this project. The pursuit has rather been towards a more intelligent trading screen. 

Below is a screenshot from the application. It can at the time of writing be found [here](http://35.158.207.95/).

![Application img](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_application.png "Application 1")

## 4 Content
* train.py is the code on which the most successful model was trained. It takes Powercell CSV data and trains a TDQN agent.
* pipeline.py is the inference procedure.
* helpers/base.py contains the prioritized experience replay buffer.
* helpers/data.py gets the data and preprocess it.
* helpers.hydro.py holds various helper functions.
* the two model files are saved Tensorflow Keras model objects. These are the two models in the application.
* in the notebooks folder there are some notebooks on preprocessing, results and training.
* the CSV file is Powercell data up until 2020-09-25.

## 5 Results
On **gold**, the first results that came in are [these](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/results_1_gold.html). What really made the difference from flat to actual learning were a proper implementation of the X2 state, and a reward clipping procedure. See more about this in the TDQN implementation notes below. 

You can follow the training of a gold model [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/training_1_gold.html). 

After some further runs the following results were obtained:

![Gold results image](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_results_gold.png "Gold results image")

On **Powercell**, we got numerous results worth mentioning:

![Powercell results 2](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/image_results_powercell_2.png "Powercell results 2")

These stood strong even though further attempts were made later on to achieve better, [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/results_4_powercell%20.html) are some of them.

The "BH" is the baseline, the Powercell shares actual movements scaled to the portfoilio value. 

[Here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/notebooks/htmls/training_2_powercell.html) is a notebook in which one can follow the training of a Powercell model.

## 6 TDQN Implementation Notes

### 6.1 State Representation  

***X1***

The agents observations consist at each timestep of the following: 
* S(t) represents the agents inner state.
* D(t) is the information concerning the OHLCV (Open-High-Low-Close-Volume) data characterising the stock market.
* I(t) represent a number of technical indicators.

This is what was used in order to get the results baove. In some experiements macro related information N(t), and information about date, week and year T(t) also were used. In those particular settings though it disturbed the training to much for any results to appear.

You will find in the context of this implementation that D(t) and I(t) bot are in the X1 input stream. Preprocessed and normalized as one they are fed into the model a single input. This was very beneficial. A number of settings with seperatae input streams for various types of observation data was experemented on without success.

***X2***

When it comes to the agents inner state S(t) it is made up by three parts:
1. cash - the amount of cash the agent has at its disposal at timestep t.
2. stock value - the value at timestep t of the stock which the agent is sitting on.
3. number of stock - the number of stock that the agent holds.

***Preprocessing***

One question that arised was how to encode this properly such that core pieces of information does not get lost. Many variations were tested. 

First of all the cash and stock value made sense to simply put in a two dimensional vector and apply some normalizer of choice and they would under all circumstances maintain their opposite proportions meanwhile bottoming out when zero, this would signal to the agent which kind of position it is capable to formulate. 

A number of scaling methods were tested, standard normal scaling, removing the mean and scale to unit variance is what worked best. The reason is simply the stability in training that it brings which allows for results to maybe or maybe not come to light. 

When it comes to the number of stock, it can easily become very large depending on the price of the stock and the initial cash value the agent is allowed to start trading with. This led me to be intitally suspicious about just clamping it in there with the cash and stock value. Hence a setup in which information about the number of stock was encoded via a one-hot procedure. 

This was made possible by adjusting the initial cash such that the number of stock were doomed to fall in between a managable range between zero and one hundred. Obviously this approach is sub optimal, however we were able to get our first results on this setup. 

At this point the cash and stock value doublet was fed into the model via their own input stream which in the code is denoted x2a, while the one-hot number of stock vector had its own stream, denoted x2b. Together the inner state of the model, S(t) is denoted as the X2 stream.

In the end though, it turned out that the simplest path often is the most beneficial. As of the best performing models, the number of stock, x2b, is simply preprocessed with the x2a, the cash and stock value such that they get normalized together in an online fashion. 

However, this setup is currently bounded by the fact that the initial cash is set to 10 000. Even though not pursuid fully, a couple of attempts were made to raise other arbitrary initial values with very poor outcomes. Rationale one can assume would be that proportions matter in terms of encoding information about the agents inner state.

### 6.3 Scalar Reward Signal
The portfolio value is the sum of cash and stock value, and the strategy as per the paper is to provide daily returns as rewards for reasons proposed one has to say makes sense. After numerous experiments with a number of varying rewards schemes, clipping the rewards to (-1, 0, 1) really made the difference in terms of creating the stability of training necessary.

During most part of the early project we suffered hard from diverging Q values during training. By that we mean that as training progressed the action values kept growing apart. In effect this gave rise to a sub optimal / useless dominant either buy or sell strategy. The reward signal was idnetified as a prosperous angle to attack the problem from. Here are some the attempts:

**Episodic scaling** Holding all the transiitons until the end of the episode allows for the applience of some scaling procedure of choice, and only to after that store all of it into memory. 

**Scaling:** We did alot of scaling, unit norm, log, min max, standard, standard without shifting the mean. With respect to many things such as larger polulations of rewards from memory and from preivous runs and so on. Still unstable training.

** Clipping:** When we added clipping into the mix, things changed. The initial hesitance came from the obvious risk of loosing valuable information that may arise from making the rewards much more sparse. 

Given that clipping the rewards added to the rise of a very stable training, trading off the potential of loosing information was not a difficult desicion to make. Recall also, that information about all the subtelties about the changes in portfolio value is in fact encoded into the X2 stream. 

### 6.2 Actions

***The Action:***

At each timestep t the agent executes a trading action as per its policy. It can buy, sell or in effect hold its position. In a DQN context one can consider a journey from the estimated Q values, to an action-preference via argmax operation (or whatever policy is in effect), Whereas the action-preference goes into the TDQN position formulation procedure and out comes out as a trading action. This trading action ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img4.png) is the actual action of the agent, and it can be either long or short.

***Effects of the Action:*** 

Each trading action has an effect on both of the two components of the portfolio value, that is cash and stock value. Here are their update rules:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img3.png)

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img2.png)

Where [](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img_nt.png) and [](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img_pt.png) are the number of shares owned by the agent and the price at timestep t.

***Simplifying the Short Position:*** 

Moving on to the formulaiton of the ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img4.png) long and short positions. A modification was made to simplify the short position, such that the agent is not able to sell any shares it does not hold. Nevertheless the lack of opportunity this would bring, the reasons for the simplification appeared as follows:
1. user-value - our downstream use-case will not really be in a position to capitalize on any opportunity brought by the agent through the execution of a short position, mainly due to access and hazzle.
2. accuracy - what is the proper market volatility parameter ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img5.png), and what is the cost estimation for shorting the stock? It is not something of immedieate clarity.
3. comfort - it reduces complexity in the implementation, makes it easier.

***The Long and Short Positions:*** 

With this in mind, the long and short positions can now described. this is the setup our best models ran on. 

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img6.png)

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img1.png)

***Expanding the Action Space***

Consider the reduced action space as a set that consist of exactly one long and one short position:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img8.png)

**Position Sizer:** Given that one wants to provide the agent with more actions to choose from, that is more options when it comes to the actual sizing of the position, one could simply add more outputs to the newtwork let them represent different slices. 

The position_sizer method found in helpers/hydro.py implements just that. Given an action preference it takes the total number of actions in the network into account and returns a sizer parameter alongside a boolean which represents either a long or a short. 

Consider the sizer parameter as the fraction  ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/img1s.png) where ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/imgs2.png)  and where ![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/imgO.png) is the total number of action values in the network. Note that this needs to be an even number.

Inserted into the positions as follows:

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/imgs3.png)

![](https://github.com/DemaciaLarz/TDQN-in-keras/blob/main/files/imgs4.png)

In effect this allows the positions to be cut into halfs, triplets, quadruples and so on. We ran several experiements on this setup ranging from two up to sixteen model outputs. We obtained results, no doubt. However, it never got any better than using simply two actions only. The agent never really seemed to fully take advantage of the fact that it could diversify its positions, but it rather kept on falling back on a simple buy and sell everything strategy only that it was worse then a pure one. There was alot of less optimal activity going on in the action values under the hood. With this being said, there is obviously alot of potential to explore here if one would push a bit to get it, if one would in a more dedicated manner adjust model capacity and hyperparameters for the purpose.

**In a dueling context:** So that is what we did, and now comes a bit of what we wanted to do. Now wouldnt it be nice to have the agent being able to freely, in a continous manner, size up its positions as it pleases? Lets say that it is, the discrete fashion of action value estimation in a DQN context quickly limits any such endevours. 

Consider the Dueling DQN model architecture. You can find the paper [here](https://arxiv.org/abs/1511.06581), some implementation notes [here](https://github.com/DemaciaLarz/Rainbow/blob/master/Implementation_Notes.ipynb) and some code for a Keras implementation [here](https://github.com/DemaciaLarz/rainbow/blob/master/Train.ipynb). 

The idea was that one could use the reduced set of actions containing only one for long, and one for short as model output propagated through the advantage stream. Then let the the state value as it is propagated throught the value stream be suitably mapped to a range between zero and one, and use that real number as a multiplicative scaler instead of the sizer parameter in the equations above.

Rationale for a procedure such as this could be:
* it would allow for an in effect continous action space with a DQN model architecture.
* while the action values represent the direction that a trade is going, the state value asks, "how nice is it where im standing right now?". One can at least imagine that the state value could encapsulate things such as tendency towards risk at that particular state and so on. 
* one would optimize the positioning sizer signal, "in model", which is appealing.
* recall the intel about the agents internal state that the X2 stream propagates thorugh the net all the way to the state value estimation. Its all there.

Our hypothesis was under any circumstances that this was worth to pursue. We encountered problems in the implementation of the dueling architecture itself though which simply led down a different path, the results were elsewhere at that point in time. However, the idea is still appealing.

### 6.4 Data Augmentation
Note that the Powercell data set is not more than 1315 observations after that 98 rows was sliced off as the test set. This is not much and it was to our surprise that any results at all were possible. Two things were crucial here. First of all the data augmenttaion techniques such as those that are mentioned in the paper, and secondly the fact that the X2 state is in effect generating new unseen data continously.

### 6.5 Model Architecture
Even though many attempts were made on various LSTM setups and dueling architecture, we ended up on a feed forward neural network. We maintained two input streams with proportions such that the X1 held two thirds of the capacity and one third was given to X2. This worked nicely. On top of this the layers were concatenated and together they went through one additional layer and into output. 

Many attempts were made to pump up the capacity but the sweetspot in this setting were simply ((200, 100), (200), (2)) and nothing else.

Results were only possible when a full carpet of batchnormalization layers were applied. Since these layers add quite alot to training time, targeted attempts were made towards the removal of them with poor outcomes. Dropout for example did not do the job, either on its own or combined witn batchnormalization.  

#### DQN Settings
Besides some of the original DQN techniques we made use of Double DQN and Prioritized experience replay. You can find the papers for these implementations [here](https://arxiv.org/abs/1509.06461) and [here](https://arxiv.org/abs/1511.05952). 

### 6.6 Hyperparameters
* aplha / learning rate - the sweet spot were exactly 1e-4, and nothing else.
* gamma / discount factor - this was kept to 0.5 as per the paper suggests.
* batchsize - it was found that 32 worked, but so did other settings aswell.
* steps - this turned out to be very important. We started out much lower and kept raisning it, it turned out that 400, a good third of the entire dataset worked best.
* weight update - keeping it at 1000 worked nicely.

### 6.7 Loss and Optimizer
We ran Huber loss and Adam optimizer as the paper suggests. Tried a few others at times of desperation but the Huber and Adam combo is what worked.

