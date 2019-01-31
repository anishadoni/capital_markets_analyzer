# capital_markets_analyzer
An algorithm that tracks correlations between stock price movements and general online sentiment from news sources and social media.


An LSTM is used to generate a stream of sentiment data from online news sources and social media. Then stock-price data is gathered. 
Training data is created by matching sentiment data for one company with stock-price movements over a long period of time. Then a standard neural network is trained on this data to predict stock-price movement from sentiment data. 


NOTE:
This project contains functions that allow it be used more easily with a jupyter-notebook style platform like Google Collaboratory. s
