import alpaca_backtrader_api
import backtrader as bt
import yfinance as yf

import config #xgb_predict #neuralnetwork_tech
import datetime as dt
import os.path, time


import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
from keras.models import load_model

class Backtesting():

    def __init__(self):
        self.stock = None
        self.time_window = None
        self.thresh = None

    def Backtest(self):
        
        filename = f'./{self.stock}_{self.time_window}_xgbpredict.csv'

        #modelfile = f'./technical_{self.stock}_model.h5'
        
        # if not path.exists(filename): # and time.ctime(os.path.getctime(filename))[:10] == time.ctime()[:10]):
        #     config.save_dataset(self.stock, self.time_window)
        # if not path.exists(modelfile):
        #     neuralnetwork_tech.Build_Model(self.stock, self.time_window)
        #     xgb_model.Build_Model()

        #model = load_model(modelfile)


        
        # ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = config.csv_to_dataset(filename)

        # technical_indicators = technical_indicators.reshape(-1, config.history_points)
        # test_split = 0.9
        # n = int(ohlcv_histories.shape[0] * test_split)

        # ohlcv_train = ohlcv_histories[:n]
        # tech_ind_train = technical_indicators[:n]
        # y_train = next_day_open_values[:n]

        # ohlcv_test = ohlcv_histories[n:]
        # tech_ind_test = technical_indicators[n:]
        # y_test = next_day_open_values[n:]

        # unscaled_y_test = unscaled_y[n:]

        # y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
        # y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

        predict = pd.read_csv(filename)
        y_predict = predict['est']
        y_real = predict['y_sample']


        buys = [(0,)]
        sells = [(1,)] #buy first, sell later

        start = 0
        end = -1

        x = 0

        for i in range(len(y_predict)-1):      
            pct = (y_predict[i+1] - y_predict[i])/y_predict[i]
            if pct > self.thresh and buys[-1][0] <= sells[-1][0]: #you can't buy more
                buys.append((x, y_real[i]))
            elif pct < (self.thresh * -1) and buys[-1][0] > sells[-1][0]: #you can't sell more: no action
                sells.append((x, y_real[i]))
            x += 1

        # for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        #     normalised_price_today = ohlcv[-1][0]
        #     normalised_price_today = np.array([[normalised_price_today]])
        #     price_today = y_normaliser.inverse_transform(normalised_price_today)
        #     predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))          
        #     pct = (predicted_price_tomorrow - price_today)/predicted_price_tomorrow
        #     if pct > self.thresh and buys[-1][0] <= sells[-1][0]: #you can't buy more
        #         buys.append((x, price_today[0][0]))
        #     elif pct < (self.thresh * -1) and buys[-1][0] > sells[-1][0]: #you can't sell more: no action
        #         sells.append((x, price_today[0][0]))
        #     x += 1
        buys.pop(0)
        sells.pop(0)
        print(f"buys: {len(buys)}")
        print(f"sells: {len(sells)}")


        def compute_earnings(buys_, sells_):
            #purchase_amt = 10
            shares = 0
            balance = 10000
            while len(buys_) > 0 and len(sells_) > 0:
                if buys_[0][0] < sells_[0][0] : #and buys_[0][1] * purchase_amt < balance:
                    # time to buy 10 stocks
                    #balance -= purchase_amt * buys_[0][1]
                    shares +=  balance // buys_[0][1]
                    balance = balance % buys_[0][1]
                    buys_.pop(0)
                else:
                    # time to sell all of our stock
                    balance += shares * sells_[0][1]
                    shares = 0
                    sells_.pop(0)
            print(f"Portfolio: ${balance} & {shares} shares of Stock")
                
            
        compute_earnings([b for b in buys], [s for s in sells])
            

        # we create new lists so we dont modify the original
        #print(f'for threshold {self.thresh} your portfolio is worth {compute_earnings([b for b in buys], [s for s in sells])}')

        plt.gcf().set_size_inches(22, 15, forward=True)

        real = plt.plot(y_real[start:end], label='real')
        pred = plt.plot(y_predict[start:end], label='predicted')

        if len(buys) > 0:
            plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
        if len(sells) > 0:
            plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

        # real = plt.plot(unscaled_y[start:end], label='real')
        # pred = plt.plot(y_predicted[start:end], label='predicted')

        plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

        plt.show()

btst = Backtesting()
btst.stock = 'AAPL'
btst.time_window = 'daily_adjusted'
btst.thresh = 0.005
btst.Backtest()


# class EWMACross(bt.SignalStrategy):
#     def __init__(self):
#         self.order = None
#         ewma1, ewma2 = bt.ind.WMA(period=5), bt.ind.WMA(period=20)
#         crossover = bt.ind.CrossOver(ewma1, ewma2)
#         self.signal_add(bt.SIGNAL_LONG, crossover)

# cerebro = bt.Cerebro()
# cerebro.addstrategy(EWMACross)
# store = alpaca_backtrader_api.AlpacaStore(
#         key_id=config.API_KEY,
#         secret_key=config.API_SECRET_KEY,
#         paper=None)

# DataFactory = store.getdata
# fromdate= datetime.today() - timedelta(days = 2 * 365)
# todate= datetime.today() - timedelta(days= 1)
# data0 = DataFactory(
#     dataname= 'AAPL',
#     timeframe= bt.TimeFrame.TFrame("Days"),
#     fromdate= pd.Timestamp(fromdate.strftime('%Y-%m-%d')),
#     todate= pd.Timestamp(todate.strftime('%Y-%m-%d')),
#     historical= True)
# cerebro.adddata(data0)
# cerebro.broker.setcash(10000)
# cerebro.broker.setcommission(commission=0.0)
# cerebro.addsizer(bt.sizers.PercentSizer, percents=20)
# cerebro.run()
# print("Final Portfolio Value: %.2f"  %cerebro.broker.getvalue())
# cerebro.plot()
