# -*- coding: utf-8 -*-
from json import load
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import time
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')
import unicodedata
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

#functions to read saved models 
union_model = load_model("python_script/saved_models/Union.h5") #put the location where model is saved
mul_model = load_model("python_script/saved_models/lstm_mv.h5")
s_model = load_model("python_script/saved_models/Sta.h5")

#website from where news to be sacraped
finviz_url = 'https://finviz.com/quote.ashx?t='

#trained models
allmodels = {'LSTM': union_model, 'LSTM(mult)': mul_model, 'slstm': s_model}

stocks_data = ('LSTM', 'LSTM(mult)','slstm')



n_steps = 30

#about section displayed at bottom in side bar
def about_section():
    st.sidebar.markdown("Saurabh Sharma")
    st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/S4urabhShrma/) ', unsafe_allow_html=True)


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

#Extracting the news from the link above and creating a data frame
def get_news_df(stock_name,finviz_url):
    
    news_tables = {}
    url = finviz_url + stock_name

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[stock_name] = news_table
    
    parsed_data = []

    for stock_name, news_table in news_tables.items():

        for row in news_table.findAll('tr'):

            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([stock_name, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['stock_name', 'date', 'time', 'title'])

    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date
    df["Negative"] = ''
    df["Neutral"] = ''
    df["Positive"] = ''
    
    sentiment_i_a = SentimentIntensityAnalyzer()
    for indexx, row in df.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', df.loc[indexx, 'title'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            df['Negative'].iloc[indexx] = sentence_sentiment['neg']
            df['Neutral'].iloc[indexx] = sentence_sentiment['neu']
            df['Positive'].iloc[indexx] = sentence_sentiment['pos']
        except TypeError:
            print (df.loc[indexx, 'title'])
            print (indexx)
    
    
    return df


def plot_cgraph(df):
    posi=0
    nega=0
    neu=0
    for i in range (0,len(df)):
        get_val=df.compound[i]
        if(float(get_val)<(0)):
            nega=nega+1
        if(float(get_val>(0))):
            posi=posi+1
        if(float(get_val==(0))):
            neu=neu+1    
    posper=(posi/(len(df)))*100
    negper=(nega/(len(df)))*100
    neuper=(neu/(len(df)))*100

    fig,ax = plt.subplots()
    arr=np.asarray([posper,negper,neuper], dtype=int)
    ax.pie(arr,labels=['positive','negative','neutral'])
    ax.plot()
    return fig,posper,negper,neuper  
  
def wordCloudFunction(df,column,numWords):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=numWords,
                          width=1000,height=1000,
                         ).generate(word_string)
    
    fig , ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis('off')
    return fig
    
def wordBarGraphFunction(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    fig , ax = plt.subplots()
    ax.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    return fig   


def plot_predict(df, model, name):
    
    
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.transform(np.array(close).reshape(-1,1))
    
    training_size=int(len(new_df)*0.67)
    test_size=len(new_df)-training_size
    train_data,test_data=new_df[:training_size],new_df[training_size:]
    Date_train, Date_test = Date[:training_size], Date[training_size:]
    
    n_steps = 30
    time_step=n_steps
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    print(X_train.shape, X_test.shape)
    
    
    
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    print(train_predict.shape, test_predict.shape)
    
    from sklearn.metrics import mean_squared_error
    print(f'Train error - {mean_squared_error(train_predict, Y_train)*100}')
    print(f'Test error - {mean_squared_error(test_predict, Y_test)*100}')
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    X_train=X_train.reshape(-1, 1)
    X_test=X_test.reshape(-1, 1)
    close_train=scaler.inverse_transform(train_data)
    close_test=scaler.inverse_transform(test_data)
    close_train = close_train.reshape(-1)
    close_test = close_test.reshape(-1)
    prediction = test_predict.reshape((-1))
    
    trace1 = go.Scatter(
        x = Date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = Date_test[n_steps:],
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = Date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = name,
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    

    st.plotly_chart(fig)
    #fig.show()
    

def plot_forecast_data(df, days, model, name):
    
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler(feature_range=(0,1))
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.transform(np.array(close).reshape(-1,1))
    
    
    
    test_data = close
    test_data = scaler.transform(np.array(close).reshape(-1,1))
    test_data = test_data.reshape((-1))
    
    def predict(num_prediction, model):
        prediction_list = test_data[-n_steps:]
        
        for _ in range(num_prediction):
            x = prediction_list[-n_steps:]
            x = x.reshape((1, n_steps, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[n_steps-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
    
    num_prediction =days
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    forecast = forecast.reshape(1, -1)
    forecast = scaler.inverse_transform(forecast)
    forecast
    test_data = test_data.reshape(1, -1)
    test_data = scaler.inverse_transform(test_data)
    test_data = test_data.reshape(-1)
    forecast = forecast.reshape(-1)
    res = dict(zip(forecast_dates, forecast))
    date = df["Date"]
    trace1 = go.Scatter(
        x = date,
        y = test_data,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode = 'lines',
        name = 'Prediction'
    )
    layout = go.Layout(
    title = name,
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    #fig.show()
    choose_date = st.selectbox("Date", forecast_dates)
    for itr in res:
        if choose_date==itr:
            res_price=res[itr]
    
    if res_price >= stock_df['Close'][-1]:
        st.success(f"On {choose_date} the stock price will be Rs. {res_price}")
    else:
        st.error(f"On {choose_date} the stock price will be Rs. {res_price}")

    

def plot_raw_data(data):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update( xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    	
#screen displayed at the beginning
def landing_ui():
    st.header("Welcome to Stock Price Predictor")
    st.write("")
    st.write("")
    st.write("Welcome to this site")
    #st.write("As the model is trained with data having time steps of 30 days so it will give its best results for a forecast till 30days ")
    st.write("")
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    #st.write("To see the data representation please uncheck the hide button in the sidebar")
    st.write("")
    st.write("Share market investments are subject to market risks, read all scheme related documents carefully. The NAVs of the schemes may go up or down depending upon the factors and forces affecting the securities market including the fluctuations in the interest rates. The past performance of the stocks is not necessarily indicative of future performance of the schemes.")
    

if __name__ == "__main__":

    st.set_page_config(layout="centered", page_icon="🔮", page_title="Stock Price Prediction")
    img = Image.open("Images/logo.png")
    
    st.sidebar.image(img,width=None)
    st.sidebar.image("https://bit.ly/2RgH8BC", width=None)
    st.sidebar.markdown("---")

    st.sidebar.subheader("Query Parameters")
    start_date = st.sidebar.date_input("Start Date",datetime.date(2018,8,1))
    end_date = st.sidebar.date_input("End Date",datetime.date(2021,4,19))
    st.sidebar.markdown("---")
    check = st.sidebar.checkbox("Warning", value=True, key='1')
    st.sidebar.markdown("---")


    
    #about_section()
    #print(temp)
    if not check:
        #to remove padding from top
        st.markdown(
                    f'''
                    <style>
                        .reportview-container .sidebar-content {{
                            padding-top: {1}rem;
                        }}
                        .reportview-container .main .block-container {{
                            padding-top: {1}rem;
                        }}
                    </style>
                    ''',unsafe_allow_html=True)

        st.title("Stock Price Prediction 🔮")

        stock_name = st.text_input('Enter Stock ticker','TSLA')
        #getting data of desired stock
        stock_df = yf.download(stock_name,start =start_date,end = date.today().strftime("%Y-%m-%d"),interval='1d')
        temp=stock_df.copy()
        temp.reset_index(inplace=True)
        
        ticker_data=yf.Ticker(stock_name)
        string_name = ticker_data.info['longName']
        stock_img = ticker_data.info['logo_url']

        col1, col2,col3= st.columns(3)
        with col1:
            st.header(string_name)
        with col3:
            st.image(stock_img)

        string_website = ticker_data.info["website"]
        st.info(string_website)
            
        string_summary = ticker_data.info['longBusinessSummary']
        st.caption("About the Company ")
        st.info(string_summary)
        st.markdown(" ")

        #to download and display data
        end=date.today().strftime("%Y-%m-%d")
        col1, col2,col3= st.columns(3)
        with col1:
            st.subheader('Stock Data ')
        with col3:
            st.download_button(
            "Download", stock_df.to_csv(), file_name=f"stock_data_{end}.csv"
            )  
        st.write(temp)
        
        st.subheader("Raw Data - Visualized")
        plot_raw_data(temp)

        
        name = st.sidebar.selectbox( "", stocks_data, key='1' )

        for itr in stocks_data:
            if name==itr:
                model=allmodels[itr]

        st.subheader("Predicted data")
        plot_predict(temp, model, name)

        #Getting news headlines
        ndf = get_news_df(stock_name,finviz_url)
        
        st.subheader(' Sentiment Analysis of news and tweets ')
        
        
        st.markdown("**Sentiment Views of users**")
        cgs,pstr,nstr,nlstr = plot_cgraph(ndf)
        
        col1, col2= st.columns(2)
        with col1:
            st.pyplot(cgs)
            st.set_option('deprecation.showPyplotGlobalUse', True)   

        with col2:
            st.caption("Positive views= "+str(pstr)+"%")
            st.caption("Negative views= "+str(nstr)+"%")
            st.caption("Neutral views= "+str(nlstr)+"%")
        
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most used words in tweets/headlines**")
            wcf = wordCloudFunction(ndf,'title',10000)
            st.pyplot(wcf)

        with col2:
            st.markdown("**Word Count**")
            wbg = wordBarGraphFunction(ndf,'title',"Popular Words in News")
            st.pyplot(wbg)
        
        
        if pstr >=40 and nlstr >=25:
            st.success('Overall Positive : According to Sentiment Analysis of the data , A rise in price trend to be expected.') 
        elif nstr >=40 and nlstr >=25:
            st.error('Negative : According to Sentiment Analysis of the data , A fall in price trend to be expected.')
        else:
            st.info('Neutral : According to Sentiment Analysis of the data , Stalement in price to be expected.')


        #st.text(STOPWORDS)
        st.sidebar.subheader("Forecasted Data")
        forecast_check = st.sidebar.checkbox("See the results", value=False)
        st.sidebar.markdown("---")
        about_section()
        if forecast_check:
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

            forecast = st.slider("Days to forecast",min_value=30,max_value=100,step=5)
            st.subheader("Forecasted data")
            
            plot_forecast_data(temp, forecast, model, name)


    else:
        landing_ui()
