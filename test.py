from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract as IBcontract
from threading import Thread
import queue
import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

#reference: https://github.com/peterbi/stockpredict1
#reference: https://gist.github.com/robcarver17/f50aeebc2ecd084f818706d9f05c1eb4



DEFAULT_HISTORIC_DATA_ID=50
DEFAULT_GET_CONTRACT_ID=43

## marker for when queue is finished
FINISHED = object()
STARTED = object()
TIME_OUT = object()

class finishableQueue(object):

    def __init__(self, queue_to_finish):

        self._queue = queue_to_finish
        self.status = STARTED

    def get(self, timeout):
        """
        Returns a list of queue elements once timeout is finished, or a FINISHED flag is received in the queue
        :param timeout: how long to wait before giving up
        :return: list of queue elements
        """
        contents_of_queue=[]
        finished=False

        while not finished:
            try:
                current_element = self._queue.get(timeout=timeout)
                if current_element is FINISHED:
                    finished = True
                    self.status = FINISHED
                else:
                    contents_of_queue.append(current_element)
                    ## keep going and try and get more data

            except queue.Empty:
                ## If we hit a time out it's most probable we're not getting a finished element any time soon
                ## give up and return what we have
                finished = True
                self.status = TIME_OUT


        return contents_of_queue

    def timed_out(self):
        return self.status is TIME_OUT





class TestWrapper(EWrapper):
    """
    The wrapper deals with the action coming back from the IB gateway or TWS instance
    We override methods in EWrapper that will get called when this action happens, like currentTime
    Extra methods are added as we need to store the results in this object
    """

    def __init__(self):
        self._my_contract_details = {}
        self._my_historic_data_dict = {}

    ## error handling code
    def init_error(self):
        error_queue=queue.Queue()
        self._my_errors = error_queue

    def get_error(self, timeout=5):
        if self.is_error():
            try:
                return self._my_errors.get(timeout=timeout)
            except queue.Empty:
                return None

        return None

    def is_error(self):
        an_error_if=not self._my_errors.empty()
        return an_error_if

    def error(self, id, errorCode, errorString):
        ## Overriden method
        errormsg = "IB error id %d errorcode %d string %s" % (id, errorCode, errorString)
        self._my_errors.put(errormsg)


    ## get contract details code
    def init_contractdetails(self, reqId):
        contract_details_queue = self._my_contract_details[reqId] = queue.Queue()

        return contract_details_queue

    def contractDetails(self, reqId, contractDetails):
        ## overridden method

        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(contractDetails)

    def contractDetailsEnd(self, reqId):
        ## overriden method
        if reqId not in self._my_contract_details.keys():
            self.init_contractdetails(reqId)

        self._my_contract_details[reqId].put(FINISHED)

    ## Historic data code
    def init_historicprices(self, tickerid):
        historic_data_queue = self._my_historic_data_dict[tickerid] = queue.Queue()

        return historic_data_queue


    def historicalData(self, tickerid , bar):

        ## Overriden method
        ## Note I'm choosing to ignore barCount, WAP and hasGaps but you could use them if you like
        bardata=(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume)

        historic_data_dict=self._my_historic_data_dict

        ## Add on to the current data
        if tickerid not in historic_data_dict.keys():
            self.init_historicprices(tickerid)

        historic_data_dict[tickerid].put(bardata)

    def historicalDataEnd(self, tickerid, start:str, end:str):
        ## overriden method

        if tickerid not in self._my_historic_data_dict.keys():
            self.init_historicprices(tickerid)

        self._my_historic_data_dict[tickerid].put(FINISHED)




class TestClient(EClient):
    """
    The client method
    We don't override native methods, but instead call them from our own wrappers
    """
    def __init__(self, wrapper):
        ## Set up with a wrapper inside
        EClient.__init__(self, wrapper)


    def resolve_ib_contract(self, ibcontract, reqId=DEFAULT_GET_CONTRACT_ID):

        """
        From a partially formed contract, returns a fully fledged version
        :returns fully resolved IB contract
        """

        ## Make a place to store the data we're going to return
        contract_details_queue = finishableQueue(self.init_contractdetails(reqId))

        print("Getting full contract details from the server... ")

        self.reqContractDetails(reqId, ibcontract)

        ## Run until we get a valid contract(s) or get bored waiting
        MAX_WAIT_SECONDS = 10
        new_contract_details = contract_details_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if contract_details_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        if len(new_contract_details)==0:
            print("Failed to get additional contract details: returning unresolved contract")
            return ibcontract

        if len(new_contract_details)>1:
            print("got multiple contracts using first one")

        new_contract_details=new_contract_details[0]

        resolved_ibcontract=new_contract_details.summary

        return resolved_ibcontract


    def get_IB_historical_data(self, ibcontract, date, durationStr="4 Y", barSizeSetting="1 day",
                               tickerid=DEFAULT_HISTORIC_DATA_ID):

        """
        Returns historical prices for a contract, up to today
        ibcontract is a Contract
        :returns list of prices in 4 tuples: Open high low close volume
        """


        ## Make a place to store the data we're going to return
        historic_data_queue = finishableQueue(self.init_historicprices(tickerid))

        # Request some historical data. Native method in EClient
        self.reqHistoricalData(
            tickerid,  # tickerId,
            ibcontract,  # contract,
            datetime.dateime(date[0:4], date[4:6], date[6:]).strftime("%Y%m%d %H:%M:%S %Z"),
            durationStr,  # durationStr,
            barSizeSetting,  # barSizeSetting,
            "TRADES",  # whatToShow,
            1,  # useRTH,
            1,  # formatDate
            False,  # KeepUpToDate <<==== added for api 9.73.2
            [] ## chartoptions not used
        )



        ## Wait until we get a completed data, an error, or get bored waiting
        MAX_WAIT_SECONDS = 10
        print("Getting historical data from the server... could take %d seconds to complete " % MAX_WAIT_SECONDS)

        historic_data = historic_data_queue.get(timeout = MAX_WAIT_SECONDS)

        while self.wrapper.is_error():
            print(self.get_error())

        if historic_data_queue.timed_out():
            print("Exceeded maximum wait for wrapper to confirm finished - seems to be normal behaviour")

        self.cancelHistoricalData(tickerid)


        return historic_data



class TestApp(TestWrapper, TestClient):
    def __init__(self, ipaddress, portid, clientid):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)

        self.connect(ipaddress, portid, clientid)

        thread = Thread(target = self.run)
        thread.start()

        setattr(self, "_thread", thread)

        self.init_error()


#if __name__ == '__main__':
app = TestApp("127.0.0.1", 7497, 1)




securities = ["MMM", "AEP", "AAPL"]

def cibcontract(symbol):
    ibcontract = IBcontract()
    ibcontract.secType = "STK"
    #ibcontract.lastTradeDateOrContractMonth="201809"
    ibcontract.symbol=symbol
    ibcontract.currency = "USD"
    ibcontract.exchange="ISLAND"
    return ibcontract

def download(symbol, date):
    d = datetime.dateime(date[0:4], date[4:6], date[6:]).strftime("%Y%m%d %H:%M:%S %Z")
    ibcontract = cibcontract(symbol)
    resolved_ibcontract=app.resolve_ib_contract(ibcontract)

    historic_data = app.get_IB_historical_data(resolved_ibcontract,date)
    pds = pd.DataFrame(historic_data)
    print(pds)

def stock_prediction(string, date, epo):

    # Collect data points from csv
    dataset = []
    #data = pd.read_csv(FILE_NAME)

    dataset = download(string, date)['4'].values ###

    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]

    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epo, batch_size=2, verbose=2)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    result = 'The price will move from %s to %s' % (dataset[0], prediction[0][0])

    return result

def stock_prediction2(string, date, epo):

    # Collect data points from csv
    dataset = []
    #data = pd.read_csv(FILE_NAME)

    dataset = download(string, date)['4'].values ###
    startdate = dataset[0][0][0:4] + '-' + dataset[0][0][4:6] + '-' +dataset[0][0][6:]
    dataset = pd.merge(dataset, getScores(string, startdate, date) , on='date', how='inner')

    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]

    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epo, batch_size=2, verbose=2)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    result = 'The price will move from %s to %s' % (dataset[0], prediction[0][0])

    return result

auth_data = {
    'grant_type'    : 'client_credentials',

    'scope'         : 'read_product_data read_financial_data read_content'
}
# create Session instance
session = requests.Session()

# make a POST to retrieve access_token
auth_request = session.post('https://idfs.gs.com/as/token.oauth2', data = auth_data)
access_token_dict = json.loads(auth_request.text)
access_token = access_token_dict['access_token']

# update session headers
session.headers.update({'Authorization':'Bearer '+ access_token})

print(access_token)

def getScores(symbol, startdate, enddate):
    payload = {
        "startDate": startdate,
        "endDate": enddate,
        "where": {
            "ticker": [ str(symbol) ]
        }
    }

    request_url = 'https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query'
    request = session.post(url=request_url, json = payload)
    results = json.loads(request.text)
    dat = results['data']
    inteS=[]
    date=[]
    rs=[]
    growth=[]
    multiple=[]
    for i in range(len(dat)):
        inteS.append((dat[i]['integratedScore']))
        rs.append((dat[i]['financialReturnsScore']))
        growth.append(dat[i]['growthScore'])
        multiple.append(dat[i]['multipleScore'])
        spli=dat[i]['date'].split('-')
        date.append(spli[0]+spli[1]+spli[2])

    d = {'date': date, 'integrated score': inteS, 'finantial return score': rs, 'growth score': growth, 'multiple score':multiple}
    df = pd.DataFrame(data=d)
    return df

# generate weights:
def generate_weights(function, date):
    num #number of stocks
    ret = []

    for i in range(x) # for each predition date
        vec = ones[len(num, 1)]*(.9/len(num, 1))
        for j in securities # for each security
            if(function(j, i, 200)/prices[i][j]>1.01):
                vec[j]+=0.01
        ret.append(vec)

    return ret

# genearate returns
def generate_rets(function, date):
    ret = []
    for i in range(1, x):
        ret.append(prices[x]/prices[x-1])

    return ret

# runs experiment
