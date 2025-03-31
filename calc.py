"""CSC111 Project 2: Smart Trades- Calc

Instructions (READ THIS FIRST!)
===============================

Based on the user input, this module recieves the data for a specific stock ticker.
With this stock ticker data, it performs the necessary calculations, using indicators like RSA and MACD.
Then, the module advises the user whether to invest in the stock, after thorough technical analysis, using the trees module.

Copyright and Usage Information
===============================

This file is provided solely for the professional use of CSC111 adminstrators
at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult CSC111 Course Syllabus.

This file is Copyright (c) Smart Trades Team- Shaurya Sareen, Abish Kulkarni, Irin Jin
"""

import datetime
import numpy as np
import pandas as pd
from trees import DecisionTree


class Calculation:
    """
    Performs the necessary calculations to advise whether the stock is a good investment, provided the given data for a stock ticker.

    Instance Attributes:
        - data: Contains a ticker's stock price, RSI and MACD data

    Representation Invariants:
        - self.data is valid and present in the dataset, containing a ticker's stock price, RSI and MACD data 
    """
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the data for the specific stock we are looking to analyze.

        Preconditions:
        - data is valid and present in the dataset, containing a ticker's stock price, RSI and MACD data 
        """
        self.data = data

    def make_prediction(self, pred_data: pd.DataFrame, tree: DecisionTree) -> np.ndarray:
        """
        Makes a trade prediction with the latest available stock data.
        The function finds the most recent valid date within a 30-day lookback period and
        uses the trained Decision Tree model to predict whether to trade.
        It returns an integer, 1 corresponding to buy and 0 to not buy.

        Preconditions:
        - pred_data is valid and present in the dataset, containing a ticker's stock price, RSI and MACD data 
        - tree is a valid DecisionTree object, holding all its attributes (like root) and nodes
        """
        target_date = datetime.datetime.now().date()
        pred_data.index = pd.to_datetime(pred_data.index).date

        max_lookback_days = 30
        days_checked = 0

        while target_date not in pred_data.index:
            target_date -= datetime.timedelta(days=1)
            days_checked += 1
            if days_checked > max_lookback_days:
                raise ValueError("No valid date found in the dataset within the lookback period.")

        data_row = pred_data.loc[target_date]

        features_for_prediction = data_row[['RSI', 'MACD', 'Signal_Line', 'SMA']].values.reshape(1, -1)

        predicted_trade = tree.predict(features_for_prediction)

        return predicted_trade

    def build_model(self, mod_data: pd.DataFrame) -> DecisionTree:
        """
        Builds a Decision Tree model to predict trade signals.

        Preconditions:
        - mod_data contains RSI, MACD, Signal Line, SMA, and Trade columns
        """
        x = mod_data[['RSI', 'MACD', 'Signal_Line', 'SMA']].values
        y = mod_data['Trade'].astype(int).values

        tree = DecisionTree(max_depth=5)
        tree.fit(x, y)
        return tree

    def compute_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates new columns in the dataframe and populates them with RSI, MACD, Signal Line, SMA, Percentage Profit
        data using the helpers defined earlier. Also creates a Trade column which contains a true or false boolean
        based on whether the percentage profit is greater than 14, indicating a Buy Signal.

        >>> data = pd.DataFrame({'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
        >>> calcObj = Calculation(data)
        >>> pred_data, model_data = calcObj.compute_data()
        >>> 'RSI' in pred_data.columns and 'MACD' in pred_data.columns
        True
        >>> 'Trade' in model_data.columns
        True
        """
        self.data['RSI'] = self.calculate_rsi(window=14)
        self.data['MACD'], self.data['Signal_Line'] = self.calculate_macd(short_window=12, long_window=26, signal_window=9)
        self.data['SMA'] = self.calculate_sma(window=14)
        self.data['Percentage_Profit'] = self.calculate_percentage_profit(days=30)

        prediction_data = self.data.copy()
        model_data = self.data.dropna().copy()
        model_data['Trade'] = model_data['Percentage_Profit'] > 14

        return prediction_data, model_data

    def calculate_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        Calculates the Relative Strength Index (RSI), for the given stock data. RSI measures
        the speed and change of price movements and helps identify overbought or oversold conditions.

        Preconditions:
        - self.data contains a 'Close' price column

        >>> data = pd.DataFrame({'Close': [50, 51, 52, 53, 50, 48, 49, 50, 52, 54]})
        >>> calcObj = Calculation(data)
        >>> data['RSI'] = calcObj.calculate_rsi(window=5).round(2)
        >>> print(data['RSI'])
        0      NaN
        1      NaN
        2      NaN
        3      NaN
        4    50.00
        5    37.50
        6    37.50
        7    37.50
        8    44.44
        9    75.00
        Name: RSI, dtype: float64
        """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates the Moving Average Convergence Divergence (MACD) and its signal line.
        MACD helps traders identify trends and reversals by comparing short-term and long-term moving averages.

        Preconditions:
        - self.data contains a 'Close' price column

        >>> data = pd.DataFrame({'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
        >>> calcObj = Calculation(data)
        >>> data['MACD'], data['Signal'] = calcObj.calculate_macd(short_window=3, long_window=6, signal_window=3)
        >>> data
            Close      MACD    Signal
        0      10  0.000000  0.000000
        1      11  0.214286  0.107143
        2      12  0.474490  0.290816
        3      13  0.713921  0.502369
        4      14  0.911729  0.707049
        5      15  1.066414  0.886732
        6      16  1.183599  1.035165
        7      17  1.270651  1.152908
        8      18  1.334505  1.243707
        9      19  1.380952  1.312330
        10     20  1.414548  1.363439
        """
        short_ema = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = self.data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def calculate_sma(self, window: int = 14) -> pd.DataFrame:
        """
        Calculates the Simple Moving Average (SMA) for a given stock data.
        SMA smooths out price fluctuations to help identify trends over a specified period.

        Preconditions:
        - self.data contains a 'Close' price column

        >>> data = pd.DataFrame({'Close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> calcObj = Calculation(data)
        >>> calcObj.calculate_sma(window=3).tolist()
        [nan, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        """
        sma = self.data['Close'].rolling(window=window).mean()
        return sma

    def calculate_percentage_profit(self, days: int = 30) -> pd.DataFrame:
        """
        Calculates the percentage profit for a given stock over a specified number of days.

        Preconditions:
        - self.data contains a 'Close' price column
        - days >= 0

        >>> data = pd.DataFrame({'Close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]})
        >>> calcObj = Calculation(data)
        >>> calcObj.calculate_percentage_profit(days=3).round(2).tolist()
        [15.0, 14.29, 13.64, 13.04, 12.5, 12.0, 11.54, nan, nan, nan]
        """
        future_price = self.data['Close'].shift(-days)
        percentage_profit = ((future_price - self.data['Close']) / self.data['Close']) * 100
        return percentage_profit


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['datetime', 'pandas', 'trees', 'numpy'],
        'allowed-io': [],
        'max-line-length': 140
    })
