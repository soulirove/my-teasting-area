# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:46:27 2021

@author: Souli
"""

import yfinance as yf
import streamlit as st

st.write("""
# Simple Stock Price App of The Top 11

shown are the stock performance

""")
complete_name= ['Apple inc.','Microsoft','Amazon Inc.','Alphabet Inc.','Tesla, Inc.','Facebook','Tencent','Alibaba Group','Visa Inc.','Johnson & Johnson','JPMorgan Chase']
abriviation=['AAPL','MSFT','AMZN','GOOG','TSLA','FB','TCEHY','BABA','V','JNJ','JPM']
company_select = st.sidebar.selectbox('select company', complete_name)


# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = abriviation[complete_name.index(company_select)]
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

dataset_select = st.sidebar.selectbox('select dataset', ('Open','High','Low','Close','Volume','Dividends'))
if dataset_select =='Open':
        st.line_chart(tickerDf.Open)
elif dataset_select =='High':
        st.line_chart(tickerDf.High)
elif dataset_select =='Low':
        st.line_chart(tickerDf.Low)
elif dataset_select =='Close':
        st.line_chart(tickerDf.Close)
elif dataset_select =='Volume':
        st.line_chart(tickerDf.Volume)
else:
        st.line_chart(tickerDf.Dividends)




