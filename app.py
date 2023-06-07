import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from st_on_hover_tabs import on_hover_tabs
from PIL import Image
import setup

st.set_page_config(layout="wide")

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
image = Image.open("C:/Users/dell/OneDrive/Desktop/dard/Logo1.png")

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'Prediction' ,'About Us'], iconName=['home', 'online_prediction', 'economy'], default_choice=0)

if tabs =='Home':
    st.image(image, width=320)
    st.title("Home")
    st.write("The popularity of cryptocurrencies had grown greatly in 2017 due to several consecutive months of exponential growth of their market capitalization, it has continued growth on the large scale and peaked at more than $1.8 trillion in 2021. Today, there are more than 1,900 actively traded cryptocurrencies.There are estimated between 2.9 and 5.8 million private as well as institutional investors in the different transaction networks, according to a recent survey , and access to the market has become easier over time with the growing number of online exchanges. Today, cryptocurrencies can be bought using fiat currency from a number of online exchanges.")
    st.write("The emergence of a self-organised market of virtual currencies and/or assets whose value is generated primarily by social consensus has naturally attracted interest from the scientific community.There have been various studies that have focused on the analysis and forecasting of price fluctuations, using mostly traditional approaches for financial markets analysis and prediction.")

elif tabs == 'Prediction':
    st.title("Trend Prediction")
    #st.write('Name of option is {}'.format(tabs))
    
    start = st.date_input("Enter a start date")
    end = st.date_input("Enter a end date")

    user_input = st.selectbox('Enter Crypto Ticker', ["BTC-INR", "DOGE-INR", "ETH-INR", "SHIB-INR", "DNT-INR"])
    df=yf.download(user_input,start,end)

    st.subheader("data from 2010-2023")
    st.write(df.describe())

    st.subheader('Closing Price vs Time chart')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart')
    ma100=df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)


    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array=scaler.fit_transform(data_training)

    x_train=[]
    y_train=[]

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train=np.array(x_train), np.array(y_train)

    model=load_model('keras_model.h5')

    past_100_days=data_training.tail(100)
    final_df=pd.concat([past_100_days,data_testing], ignore_index=True)
    input_data=scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor=1/scaler[0]
    y_predicted=y_predicted*scale_factor
    y_test=y_test*scale_factor

    st.subheader('Prediction vs Original')
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    
elif tabs == 'About Us':
    st.title("About Us")
    st.write("Hi Everyone this is Piyush Nigam student of Galgotias College of Engineering and Technology pursuing Master of Computer Application.")
    st.write("We would like to acknowledge and thank our Head of Department,for his contributions towards the development of our skillset.We wish to thank our parents as well for their undivided support and interest who inspired us and encouraged us to go our own way, without whom we would be unable to complete our project.In the end, we would also like to express our heartfelt gratitude towards our friends who displayed appreciation for our work and motivated us to continue our work.")


