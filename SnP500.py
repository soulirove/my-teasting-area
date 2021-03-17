# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 01:06:37 2021

@author: Souli
"""
import streamlit as st 
import pandas as pd 
import numpy as np 
import yfinance as yf 
import matplotlib.pyplot as plt 
import base64


st.title('Simple Stock Price App of The S&P 500')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, yfinance
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

@st.cache
def load_data(): #a function to net scrape out data 
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' 
    html = pd.read_html(url, header = 0) #to read the table 
    df = html[0] # to select the first table on the page
    return df
df = load_data() # web scraping data assigned to df 
# sidebar options 
st.sidebar.subheader('Stocks Component') 
sector = df.groupby('GICS Sector')
# Sidebar - GICS Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('GICS Sector', sorted_sector_unique,sorted_sector_unique[0])
# Filtering data
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]
# Sidebar - GICS Sub-Industry Sector selection
sorted_sector_unique_sub = sorted( df_selected_sector['GICS Sub-Industry'].unique() )
selected_sector_sub = st.sidebar.multiselect('GICS Sub-Industry', sorted_sector_unique_sub,sorted_sector_unique_sub[0])
# Filtering data
df_selected_sector_sub = df[ (df['GICS Sub-Industry'].isin(selected_sector_sub)) ]
# showing the DATA
st.header('Display Companies in Selected Sectors')
st.write('Data Dimension: ' + str(df_selected_sector_sub.shape[0]) + ' rows and ' + str(df_selected_sector_sub.shape[1]) + ' columns.')
st.dataframe(df_selected_sector_sub)
# Download S&P500 data in csv format
# this link shows how to do it https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="FILE.csv">CSV File Download</a>'
    return href

st.markdown(filedownload(df_selected_sector_sub), unsafe_allow_html=True)
# explained on this website pypi.org/project/yfinance/
data = yf.download(
        tickers = list(df_selected_sector_sub.Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )
#disable After December 1st, 2020, Matplotlib's global figure object, which is not thread-safe warning 
st.set_option('deprecation.showPyplotGlobalUse', False)

# Plot Opening Price of Query Symbol
def price_plot_Open(symbol):
  df = pd.DataFrame(data[symbol].Open)
  df['Date'] = df.index
  plt.plot(df.Date, df.Open, color='red', alpha=0.5) # the plot line is in red 
  plt.fill_between(df.Date, df.Open, color='red', alpha=0.1) #the filling under the plot line is red but much lighter 
  plt.title(symbol, fontweight='bold') #the title of the graph is the symbol of the company
  plt.xlabel('Date', fontweight='bold') #labeling the X axis Date
  plt.ylabel('Opening Price', fontweight='bold') ##labeling the Y axis Opening Price
  plt.xticks(rotation=90) #rotating the ticks by 90 degrees so they become more visable 
  return st.pyplot()
# Plot High Price of Query Symbol
def price_plot_High(symbol):
  df = pd.DataFrame(data[symbol].High)
  df['Date'] = df.index
  plt.plot(df.Date, df.High, color='blue', alpha=0.5)
  plt.fill_between(df.Date, df.High, color='blue', alpha=0.1)
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('High Price', fontweight='bold')
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  return st.pyplot()
# Plot Low Price of Query Symbol
def price_plot_Low(symbol):
  df = pd.DataFrame(data[symbol].Low)
  df['Date'] = df.index
  plt.plot(df.Date, df.Low, color='green', alpha=0.5)
  plt.fill_between(df.Date, df.Low, color='green', alpha=0.1)
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Low Price', fontweight='bold')
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  return st.pyplot()
# Plot Closing Price of Query Symbol
def price_plot_Close(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.plot(df.Date, df.Close, color='magenta', alpha=0.5)
  plt.fill_between(df.Date, df.Close, color='magenta', alpha=0.1)
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  return st.pyplot()
# Plot Volume Price of Query Symbol
def price_plot_Volume(symbol):
  df = pd.DataFrame(data[symbol].Volume)
  df['Date'] = df.index
  plt.plot(df.Date, df.Volume, color='cyan', alpha=0.5)
  plt.fill_between(df.Date, df.Volume, color='cyan', alpha=0.1)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Volume Price', fontweight='bold')
  plt.xticks(rotation=90)
  return st.pyplot()
# Plot Dividends Price of Query Symbol
def price_plot_Dividends(symbol):
  df = pd.DataFrame(data[symbol].Dividends)
  df['Date'] = df.index
  plt.plot(df.Date, df.Dividends, color='yellow', alpha=0.5)
  plt.fill_between(df.Date, df.Dividends, color='yellow', alpha=0.1)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Dividends Price', fontweight='bold')
  plt.xticks(rotation=90)
  return st.pyplot()
# selecting the data of the Query
dataset_select = st.sidebar.selectbox('select dataset', ('Open','High','Low','Close','Volume','Dividends'))
# choosing the number of companies
num_company = st.sidebar.slider('Number of Companies', 1, 10)
# to show the Plots
if st.button('Show Plots'): #checking which dataset is being queried
    if dataset_select =='Open':
            st.header('Stock Opening Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]: 
                        price_plot_Open(i) #ploting all the selected data
    elif dataset_select =='High':
            st.header('Stock High Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]:
                        price_plot_High(i)
    elif dataset_select =='Low':
            st.header('Stock Low Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]:
                        price_plot_Low(i)
    elif dataset_select =='Close':
            st.header('Stock Closing Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]:
                        price_plot_Close(i)
    elif dataset_select =='Volume':
            st.header('Stock Volume Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]:
                        price_plot_Volume(i)
    else:
            st.header('Stock Dividends Price')
            for i in list(df_selected_sector_sub.Symbol)[:num_company]:
                        price_plot_Dividends(i)