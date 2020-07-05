#Alcapa
API_KEY = "PKZSYQS4N9O5GQYOG735"
API_SECRET_KEY = "6/JQ7OHMqJnVNmRl4KfruiCwRRGagGMdWNu5d4a2"

#Datahub
wti_daily_path = 'https://pkgstore.datahub.io/core/oil-prices/wti-daily/archive/06c5d4808369fd5ec71b045210162db7/wti-daily.csv'
brent_daily_path = 'https://pkgstore.datahub.io/core/oil-prices/brent-daily/archive/b220560ec00f25407b0cc9ed2198a687/brent-daily.csv'
vix_daily_path = 'https://pkgstore.datahub.io/core/finance-vix/vix-daily/archive/d4a51363da29db079f13cd2351f72145/vix-daily.csv'

#pyEX
TOKEN_pyEX = 'sk_555e253314b9446c9a75c75399741744'

#AlphaVantage
AV_API = 'OJEOM5EZYRVK9GBF'

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from pprint import pprint
import json
import argparse

import pandas as pd
from sklearn import preprocessing
import numpy as np


def save_dataset(symbol, time_window):
    api_key = AV_API
    print(symbol, time_window)

    #frames = []

    ts = TimeSeries(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data_ts, meta_data_ts = ts.get_intraday(
            symbol=symbol, interval='5min', outputsize='full')
    elif time_window == 'daily':
        data_ts, meta_data_ts = ts.get_daily(symbol, outputsize='full')
    elif time_window == 'daily_adj':
        data_ts, meta_data_ts = ts.get_daily_adjusted(symbol, outputsize='full')

    data_ts = data_ts.reset_index().set_index('date', drop = False)
    data_ts.index.name = None
    data_ts['date'] = pd.to_datetime(data_ts['date'])
    data_ts = data_ts.rename(columns={'1. open': 'Open', '2. low': 'Low', '3. high': 'High', '4. close': 'Close'})

    brent_df = pd.read_csv(brent_daily_path)
    wti_df = pd.read_csv(wti_daily_path)
    vix_df = pd.read_csv(vix_daily_path)
    all_df = pd.merge(pd.merge(brent_df, wti_df, on = 'Date'), vix_df, on = 'Date')
    all_df = all_df.rename(columns={'Date': 'date'})

    all_df['date'] = pd.to_datetime(all_df['date'])
    #frames.append(data_ts)
    ############################################################################
    '''tech = TechIndicators(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data_sma, meta_data_sma = tech.get_sma(
            symbol=symbol, interval='5min', time_period=20, series_type='close')
        data_ema, meta_data_ema = tech.get_ema(
            symbol=symbol, interval='5min', time_period=20, series_type='close')
        data_macd, meta_data_macd = tech.get_macd(
            symbol=symbol, interval='5min', series_type='close')
    elif time_window == 'daily' or time_window == 'daily_adj':
        data_sma, meta_data_sma = tech.get_sma(
            symbol=symbol, interval='daily', time_period=20, series_type='close')
        data_ema, meta_data_ema = tech.get_ema(
            symbol=symbol, interval='daily', time_period=20, series_type='close')
        data_macd, meta_data_macd = tech.get_macd(
            symbol=symbol, interval='daily', series_type='close')
    frames.append(data_sma)
    frames.append(data_ema)
    frames.append(data_macd)'''
    ############################################################################ 
    cc = pd.merge(data_ts, all_df, on = 'date')
    cc.dropna(inplace = True)
    cc.to_csv(f'./{symbol}_{time_window}.csv')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument('symbol', type=str, help="the stock symbol you want to download")
#     parser.add_argument('time_window', type=str, choices=[
#                         'intraday', 'daily', 'daily_adj'], help="the time period you want to download the stock history for")

#     namespace = parser.parse_args()
#     save_dataset(**vars(namespace))

history_points = 50

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    hist = data.loc[:, '1. open':'5. volume']  #timeseries
    tech = data.loc[:, 'MACD'] #technical
    
    hist = hist.values
    tech = tech.values.reshape(-1, 1)

    normaliser = preprocessing.MinMaxScaler()
    hist_normalised = normaliser.fit_transform(hist)
    tech_normalised = normaliser.fit_transform(tech)

    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([hist_normalised[i:i + history_points].copy() for i in range(len(hist_normalised) - history_points)])
    tech_indicators_normalised = np.array([tech_normalised[i:i + history_points].copy() for i in range(len(tech_normalised) - history_points)])
    next_day_open_values_normalised = np.array([hist_normalised[:, 0][i + history_points].copy() for i in range(len(hist_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([hist[:, 0][i + history_points].copy() for i in range(len(hist) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    # def calc_ema(values, time_period):
    #     sma = np.mean(values[:, 3])
    #     ema_values = [sma]
    #     k = 2 / (1 + time_period)
    #     for i in range(len(his) - time_period, len(his)):
    #         close = his[i][3]
    #         ema_values.append(close * k + ema_values[-1] * (1 - k))
    #     return ema_values[-1]

    # technical_indicators = []
    # for his in ohlcv_histories_normalised:
    #     # note since we are using his[3] we are taking the SMA of the closing price
    #     sma = np.mean(his[:, 3])
    #     macd = calc_ema(his, 5) - calc_ema(his, 20)
    #     technical_indicators.append(np.array([sma]))
    #     technical_indicators.append(np.array([sma,macd]))

    #tech_ind_scaler = preprocessing.MinMaxScaler()
    #technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == tech_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, tech_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def multiple_csv_to_dataset(test_set_name):
    import os
    ohlcv_histories = 0
    technical_indicators = 0
    next_day_open_values = 0
    for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('./'))):
        if not csv_file_path == test_set_name:
            print(csv_file_path)
            if type(ohlcv_histories) == int:
                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset(csv_file_path)
            else:
                a, b, c, _, _ = csv_to_dataset(csv_file_path)
                ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
                technical_indicators = np.concatenate((technical_indicators, b), 0)
                next_day_open_values = np.concatenate((next_day_open_values, c), 0)

    ohlcv_train = ohlcv_histories
    tech_ind_train = technical_indicators
    y_train = next_day_open_values

    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset(test_set_name)

    return ohlcv_train, tech_ind_train, y_train, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser

