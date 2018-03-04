
# coding: utf-8

# In[ ]:


import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

#import urllib.request

auth_data = {
    'grant_type'    : 'client_credentials',
    'client_id'     : '**********',
    'client_secret' : '***********',
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


# In[169]:


def getScores(symbol):
    payload = {
        "startDate": "2012-12-31",
        "endDate": "2017-06-30",
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

# plt.figure(0)
# plt.plot(date,inteS)
# plt.title("integrated score")

# plt.figure(1)
# plt.plot(date,rs)
# plt.title("finantial return score")

# plt.figure(2)
# plt.plot(date,growth)
# plt.title("growth score")
# plt.figure(3)
# plt.plot(date,multiple)
# plt.title("multiple score")
# plt.show()
dff=getScores('FB')
dff


# In[ ]:





# In[ ]:





# In[170]:


payload = {
    "startDate": "2015-11-02",
    "endDate": "2015-11-04",
    "where": {
        "ticker": [ "FB", "AAPL", "GOOGL" ]
    }
}

request_url = 'https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query'
request = session.post(url=request_url, json = payload)
results = json.loads(request.text)
data = results['data']
print(data)


# In[ ]:




