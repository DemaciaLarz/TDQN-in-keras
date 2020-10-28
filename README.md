# Implementing TDQN in Keras
Consider as the basis of this project the Trading Deep Q-Network algorithm (TDQN) as it is put forward in the paper named: An Application of Deep Reinforcement Learning to Algorithmic Trading which you can find [here](https://arxiv.org/abs/2004.06627).

## Summary

#### Objectives
The objective was to implement the TDQN algorithm on a set of stocks from the upcoming hydrogen sector. 

In the end it got applied to historical gold prices for the initial training, and then quite successfully to one hydrogen stock, Powercell Sweden. Here are some nice results:

![Powercell results](https://github.com/DemaciaLarz/implementing-TDQN-in-keras/blob/main/files/img_results_powercell_1.png "Powercell results")

#### Underlying assets - data
Powercell Sweden is a fuel cell manufacturer listed on the First North GM Sweden market, [here](http://www.nasdaqomxnordic.com/aktier/microsite?Instrument=SSE105121&name=PowerCell%20Sweden) are information on the share and the historical prices, and [here](https://www.powercell.se/en/start/) is info on the company.

You can find analysis and preprocessing as it realtes to this project of the actual data [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_2_powercell.html)

When it comes to gold [here](https://www.kaggle.com/omdatas/historic-gold-prices) are the historical prices, and [here](http://htmlpreview.github.io/?https://github.com/DemaciaLarz/trading-hydro/blob/main/notebooks/htmls/know_your_data_1_gold.html) is the analysis from this project.

#### User-values / downstream application
The use-case is to apply one or more successfully trained models such that they are able to bring some actual useful intel on a daily basis when it comes to the Powercell share movements.

This is acheived through an application in which daily results from the models actions alongside the underlying asset as a basline is being presented. The results are obtained by running an inference procedure as per the [pipeline.py]() script. 

In a nutshell a cronjob runs the script around 5:00am each morning. It collects yesterdays closing data through some selenium code, performs the inference procedure and updates a set of databases so that its action can get interpreted as a buy or sell signal in perfect time a few hours before the markets open.

This could provide some opportunity. Either by simply dedicating a bit of capital for the purpose and executing the daily trades each day as per ones favourite model, or by using the model behaiviour as merely additional intel in ones own decision making process. Executing trades automatically through some api has not been a pursuit of this particular project.

Below is a screenshot of the application. It can currently be found [here](http://35.158.207.95/).

![Application img](https://github.com/DemaciaLarz/implementing-TDQN-in-keras/blob/main/files/application_1.png "Application 1")
