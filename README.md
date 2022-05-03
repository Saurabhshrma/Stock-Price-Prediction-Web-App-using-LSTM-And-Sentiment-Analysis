# Stock-Price-Prediction-Web-App-using-LSTM
**Stock Market Prediction** Web App based on **Machine Learning** and **Sentiment Analysis** of Tweets and News **(API keys included in code)**. The front end of the Web App is based on **Streamlit**. The App forecasts stock prices of the next sixty days for any given stock under **NASDAQ** or **NSE** as input by the user. Predictions are made using algorithm: **LSTM**. The Web App combines the predicted prices of the next seven days with the **sentiment analysis of tweets** to give recommendation whether the price is going to rise or fall


# Screenshots
<img src="https://github.com/Saurabhshrma/Stock-Price-Prediction-Web-App-using-LSTM-And-Sentiment-Analysis/blob/master/Images/homepage.jpg">
<img src="https://github.com/Saurabhshrma/Stock-Price-Prediction-Web-App-using-LSTM-And-Sentiment-Analysis/blob/master/Images/senti.jpg">
<img src="https://github.com/Saurabhshrma/Stock-Price-Prediction-Web-App-using-LSTM-And-Sentiment-Analysis/blob/master/Images/pred.jpg">


# File and Directory Structure
<pre>
Images - Screenshots of Web App
python_scripts - contains code to create ML and sentiment analysis model
webapp.py - main code that runs to display model and predictions on web app
</pre>

# Technologies Used
<ol>
<a href="https://streamlit.io/"><li>Streamlit</a></li>
<a href="https://www.tensorflow.org/"><li>Tensorflow</a></li>
<a href="https://keras.io/"><li>Keras</a></li>
<a href="https://pypi.org/project/yfinance/"><li>Yahoo Finance</a></li>
<a href="https://scikit-learn.org/"><li>Scikit-Learn</a></li>
<a href="https://www.tweepy.org/"><li>Tweepy</a></li>
<a href="https://www.python.org/"><li>Python</a></li>
</ol>

# How to Install and Use
<b>Python 3.8.5 is required for the python packages to install correctly</b><br>
<ol>
<li>Clone the repo</li>
<li>use webapp.py file to run the web application.</li>
<li>Select the wordpress database and click on <b>Import</b> and select the <b>wordpress.sql</b> file from the repo.</li> 
<li>OPen command prompt:</li>
<li>type - streamlit run webapp.py</li>
<li>1. make sure the command line is pointing at the current directory(directory in which webapp.py is present.)</li>
<li>2. streamlit must be installed in your systems.</li>
# Authors
## SaurabhSharma
<ul>
<li>Github:https://github.com/Saurabhshrma</li>
<li>LinkedIn:https://www.linkedin.com/in/s4urabhshrma/</li>
</ul>