#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import streamlit as st



# In[2]:

st.title(" Perrin Freres sales of champagne")
st.header("The dataset is monthly sales of champagne from year 1964 to 1972.")

data = pd.read_csv('./champagne.csv')


# In[3]:


data.head()


# In[4]:


data.columns = ['Month','Sales']


# In[5]:


data.drop(106, axis=0, inplace = True)


# In[6]:


data.info()


# In[7]:


data.Month = pd.to_datetime(data.Month)


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.plot()


# In[11]:


sns.lineplot(data = data, x= 'Month', y = 'Sales')


# In[12]:


import statsmodels as stat
from statsmodels.tsa.stattools import adfuller


# In[13]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[14]:


adfuller_test(data['Sales'])


# In[15]:


data['Sales diff']= data['Sales'] - data['Sales'].shift(12) #desfasa 12 meses con el fin de volverlo estacionario


# In[16]:


adfuller_test(data['Sales diff'].dropna())


# In[17]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf #correlacion parcial


# In[18]:


plot_acf(data['Sales'].iloc[13:].dropna())


# In[19]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data['Sales diff'].iloc[13:].dropna(),lags=40,ax=ax1)#autocorrelacion relacion de datos con ellos mismos
ax2 = fig.add_subplot(212)
fig = plot_pacf(data['Sales diff'].iloc[13:].dropna(),lags=40,ax=ax2)#autocorrelacion relacion los mismos datos y un paso de tiempo 


# In[20]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data['Sales'].dropna(),lags=40,ax=ax1)#autocorrelacion relacion de datos con ellos mismos
ax2 = fig.add_subplot(212)
fig = plot_pacf(data['Sales'].dropna(),lags=40,ax=ax2)#autocorrelacion relacion los mismos datos y un paso de tiempo 


# In[21]:


from statsmodels.tsa.arima.model import ARIMA


# In[22]:


model = ARIMA(data['Sales'],order = (1,1,1))#parametros (1,1,1)(p,q,d)


# In[23]:


model_fit= model.fit()


# In[24]:


model_fit.summary()


# In[25]:


data['Forecast'] = model_fit.predict(start = 90, end = 103 , dynamic = True)
data[['Sales', 'Forecast']].plot()


# In[26]:


import statsmodels.api as sm


# In[27]:


model_ciclico = sm.tsa.statespace.SARIMAX(data['Sales'],order = (1,1,1), seasonal_order = (1,1,1,12))
model_ciclico_fit = model_ciclico.fit()


# In[29]:


data['Forecast_s'] = model_ciclico_fit.predict(start = 100, end = 130 , dynamic = True)
data[['Sales', 'Forecast_s']].plot()


# slider p,q, D. 
# 3 slider p , q , d 
# 2 numericas start y enter 
# grafica el resultado variables de entrada start y end

# In[30]:
st.line_chart(data['Sales'])

#select values for model sarimax
st.subheader("P")
p1 = st.slider("Choose P value between 0-10",min_value= 0.0, max_value= 10.0, step=0.1) 

st.subheader("Q")
q1 = st.slider("Choose Q value between 0-10",min_value= 0.0, max_value= 10.0, step=0.1)

st.subheader("D")
d1 = st.slider("Choose D value between 0-10",min_value= 0.0, max_value= 10.0, step=0.1)

# In[31]

def prediccion_sales(start , end ):
       
    inicio = int(start)
    final = int(end)
        
    model_ciclico = sm.tsa.statespace.SARIMAX(data['Sales'],order = (1,1,1), seasonal_order = (1,1,1,12))
    model_ciclico_fit = model_ciclico.fit()
    
    data['Forecast'] = model_ciclico_fit.predict(start = inicio, end = final , dynamic = True)
    
    data_transformada = data[['Sales','Forecast']]
    
    return data_transformada
    



p_start = st.text_input("parametro inicial")
p_end = st.text_input("parametro final")

if st.button("Accept"):
    result = prediccion_sales(start=p_start, end=p_end)
    st.line_chart(result)



# In[32]

def prediccionp(start, end):
    inicio =int(start)
    final = int(end)
    
    model_ciclico = sm.tsa.statespace.SARIMAX(data['Sales'],order = (1,1,1), seasonal_order = (1,1,1,12))
    model_ciclico_fit = model_ciclico.fit()
    
    data['Forecast'] = model_ciclico_fit.predict(start = inicio, end = final , dynamic = True)
    data[['Sales','Forecast']].plot()

prediccionp(start=100,end=110)
    
    
    
    
    
