import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import yfinance as yf
import pypfopt as pfopt
import alpha_vantage as av
import requests
import json
API_KEY = 'ENTER ALPHAVANTAGE KEY'
from datetime import datetime, timedelta



api_key = "ENTER OPENAI KEY"

class Get_Data:
    def __init__(self, dictionary):

        self.horizon = dictionary['horizon']
        self.stocks = dictionary['stocks']
        self.money_to_invest = dictionary['money_to_invest']
        self.method = dictionary['method']
        self.today_date = self.get_today_date()
        self.start_date = self.get_start_date()
        self.data = self.get_data()
        self.risk_free_rate = 0.02


    ## Lacks risk_free_rate

    def get_start_date(self):
        return self.today_date - pd.Timedelta(days = self.horizon)
    
    def get_today_date(self):
        return pd.to_datetime('today').normalize()
    
    def get_data(self):
        stock_data = yf.download(self.stocks, start = self.start_date, end = self.today_date)['Adj Close']
        return stock_data
    


class Get_Efficent_Frontier(Get_Data):
    def __init__(self, dictionary):

        Get_Data.__init__(self, dictionary)

        self.mu = pfopt.expected_returns.mean_historical_return(self.data)
        self.S = pfopt.risk_models.sample_cov(self.data)
        self.ef = pfopt.efficient_frontier.EfficientFrontier(self.mu, self.S, weight_bounds=(-1, 1))



class Min_Variance(Get_Efficent_Frontier):

    def __init__(self, dictionary):

        Get_Efficent_Frontier.__init__(self, dictionary)

        self.raw_weights = self.ef.min_volatility()
        self.weights_min_var = self.ef.clean_weights()
        (
        self.expected_return, 
        self.expected_vol, 
        self.expected_sharpe
        ) = self.ef.portfolio_performance()

        self.allocation = self.get_allocation()

    def get_allocation(self):
        allocation = {stock: round(weight * self.money_to_invest, 2) for stock, weight in self.weights_min_var.items()}
        return allocation
    


class Max_Sharpe(Get_Efficent_Frontier):
    
    def __init__(self, dictionary):

        Get_Efficent_Frontier.__init__(self, dictionary)

        self.raw_weights = self.ef.max_sharpe()
        self.weights_max_sharpe = self.ef.clean_weights()
        (
        self.expected_return, 
        self.expected_vol, 
        self.expected_sharpe
        ) = self.ef.portfolio_performance()

        self.allocation = self.get_allocation()

    def get_allocation(self):
        allocation = {stock: round(weight * self.money_to_invest, 2) for stock, weight in self.weights_max_sharpe.items()}
        return allocation
        

class Get_Portfolio:
    def __init__(self, dictionary):
        if dictionary['method'] == 'min_variance':
            self.strategy = Min_Variance(dictionary)
        elif dictionary['method'] == 'max_sharpe':
            self.strategy = Max_Sharpe(dictionary)

        self.allocation = self.strategy.get_allocation()
        (
        self.expected_return, 
        self.expected_vol, 
        self.expected_sharpe
        ) = self.strategy.ef.portfolio_performance()
        self.portfolio_value = self.get_cumulative_returns()


    def get_cumulative_returns(self):
        cumulative_returns = self.strategy.data.pct_change().dropna().add(1).cumprod()
        cumulative_returns_weighted = sum([cumulative_returns[stock] * weight for stock, weight in self.allocation.items()])
        cumulative_returns_weighted = 100*(cumulative_returns_weighted/cumulative_returns_weighted[0])
        return cumulative_returns_weighted
    

class AlphaVantageNewsSentiment:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_sentiment(self, ticker):
        function = "NEWS_SENTIMENT"
        
        url = self.base_url + "?function=" + function + "&tickers=" + ticker + "&apikey=" + self.api_key
        response = requests.get(url)
        if response.status_code == 200:
            news_data = json.loads(response.text)
        else : return None
        sentiment_scores = []
        sentiment_relevance = []
        for article in news_data['feed']:
            for dicos in article['ticker_sentiment']:
                if dicos['ticker'] == ticker:
                    sentiment_scores.append(float(dicos['ticker_sentiment_score']))
                    sentiment_relevance.append(float(dicos['relevance_score']))
        weighted_sentiment_score = np.mean([score*weight for score, weight in zip(sentiment_scores, sentiment_relevance)])*1000 if sentiment_scores else None
        return weighted_sentiment_score




    


    









