from chalice import Chalice
import requests, json

app = Chalice(app_name='alpaca-trading')

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDER_URL = "{}/v2/orders".format(BASE_URL)
HEADERS = {'APCA-API-KEY-ID' : 'PKZSYQS4N9O5GQYOG735', 'APCA-API-SECRET-KEY' : 'JQ7OHMqJnVNmRl4KfruiCwRRGagGMdWNu5d4a2'}

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/buy_stock', methods=['POST'])
def buy_stock():
    # This is the JSON body the user sent in their POST request.
    request = app.current_request
    message = request.json_body
    # We'll echo the json body back to the user in a 'user' key.
    return {'message': message}

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
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
    
    symbol = 'AAPL' #str(input('enter ticker: '))
    time_window = 'daily' #str(input('enter time window: '))

    #config.save_dataset(symbol, time_window)

    #neuralnetwork_tech.Build_Model(symbol, time_window)

    #pct = papertrade.predict(symbol, time_window)

    thresh = 0.005
    pct = 0.05
    if pct > thresh:
        post_order(symbol, 50, 'buy', 'market', 'gtc', None, None)
        return {'message': f'Got you some {symbol} shares'}
    elif pct < -0.02:
        position = get_position(symbol)
        if position['message'] == 'position does not exist':
            return {'message': f'No {symbol} shares to sell'}
            
        else:
            n = position['qty']
            post_order(symbol, n , 'sell', 'market', 'gtc', None, None)
            return {'message' : f'Sold all {n} {symbol} shares for you'}
    else:
        return {'message' : 'No action taken'}