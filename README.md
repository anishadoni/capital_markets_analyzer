# capital_markets_analyzer
An algorithm that tracks correlations between stock price movements and general online sentiment from news sources and social media.


First a stream of sentiment data from online news sources and social media is generated. Then stock-price data is gathered. 
Training data is created by matching sentiment data for one company with stock-price movements over a correlated period of time. Then a standard neural network is trained on this data to predict stock-price movement from sentiment data. 

