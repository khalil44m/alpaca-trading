import requests, json
import config, neuralnetwork_tech
from keras.models import load_model
from datetime import datetime, timedelta

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDER_URL = "{}/v2/orders".format(BASE_URL)
HEADERS = {'APCA-API-KEY-ID' : config.API_KEY, 'APCA-API-SECRET-KEY' : config.API_SECRET_KEY}

#################################### POST/GET #####################################
###################################################################################
###################################################################################
def get_account():
    r = requests.get(ACCOUNT_URL, headers = HEADERS)
    return json.loads(r.content)

def post_order(symbol, qty, side, type, time_in_force, limit_price, stop_price):
    data = {
        "symbol" : symbol,
        "qty" : qty,
        "side" : side,
        "type" : type,
        "time_in_force" : time_in_force
    }

    r = requests.post(ORDER_URL, json = data, headers = HEADERS)
    return json.loads(r.content)

def get_orders():
    r = requests.get(ORDER_URL, headers = HEADERS)
    return json.loads(r.content)

def get_position(stock):
    r = requests.get("{}/v2/positions/{}".format(BASE_URL, stock), headers = HEADERS)
    return json.loads(r.content)



#################################### TRAIN MODEL ##################################
###################################################################################
###################################################################################
class PaperTrade():
    def __init__(self):
        self.stock = None
        self.time_window = None
        self.y_test_predicted = None
        self.thresh = None
    def predict(self):
        
        #config.save_dataset(self.stock, self.time_window)

        #neuralnetwork_tech.Build_Model(self.stock, self.time_window)

        model = load_model(f'technical_{self.stock}_model.h5')

        ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = config.csv_to_dataset(f'./{self.stock}_{self.time_window}.csv')

        ohlcv_train = ohlcv_histories[0:-1]
        tech_ind_train = technical_indicators[0:-1]
        y_train = next_day_open_values[0:-1]

        ohlcv_test = [ohlcv_histories[-1]]
        tech_ind_test = [technical_indicators[-1]]
        y_test = [next_day_open_values[-1]]

        unscaled_y_test = unscaled_y[-1]

        y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
        y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

        return y_test_predicted, unscaled_y_test

##################################### BUY/SELL ####################################
###################################################################################
###################################################################################
    def buy_sell(self, y_test_predicted, unscaled_y_test):

        pct = (y_test_predicted - unscaled_y_test)/ unscaled_y_test
        if pct > self.thresh:
            post_order(self.stock, 50, 'buy', 'market', 'gtc', None, None)
            print(f'Got you some {self.stock} shares')
        elif pct < - 0.02:
            position = get_position(self.stock)
            if 'message' in position:
                print(f'No {self.stock} shares to sell')
                pass
            else:
                n = position['qty']
                post_order(self.stock, n , 'sell', 'market', 'gtc', None, None)
                print(f'Sold all {n} {self.stock} shares for you')
        else:
            print('No action taken')

papertrade = PaperTrade()
papertrade.stock = 'AAPL'
papertrade.time_window = 'daily'
papertrade.thresh = 0.005

y_test_predicted, unscaled_y_test = papertrade.predict()
papertrade.buy_sell(y_test_predicted, unscaled_y_test)