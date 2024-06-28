# Project: Market Sentiment Integration for Improved Stock Price Forecasting

### Description:

This repository contains research code for sentiment analysis and stock price prediction using financial news, Twitter, and StockTwits data. The project aims to forecast the extent of stock price changes by analyzing sentiment from financial news and social media.

### The primary objectives of this project are the following:

1. **Comparative Analysis of Sentiment Models:** Undertake a systematic comparison of state-of-the-art NLP and sentiment analysis models in the context of financial markets to facilitate the selection of the most high-performing sentiment analysis model.

2. **Integration of Sentiment into Stock Prediction Models:** Incorporate the selected sentiment analysis model into stock prediction models. Evaluate the impact of sentiment analysis on the predictive accuracy of machine learning-based stock market prediction models.

### Dependencies:

- Python 3.8.12
- Jupyter Notebook
- Pandas 2.2.0
- NumPy 1.26.3
- Matplotlib 3.8.2
- Scikit-learn 1.4.0
- finbert 0.1.4
- nltk 3.8.1
- requests 2.31.0
- scipy 1.7.3
- tensorflow 2.15.0
- tensorflow-estimator 2.15.0
- textblob 0.17.1
- torch 1.11.0
- transformers 4.36.2
- xgboost 2.0.3

### How to Run:

1. Clone the repository:

`git clone https://git.cs.bham.ac.uk/projects-2023-24/oxe002.git`

`cd oxe002`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Open and execute the notebooks in the corresponding directory.

## Structure:

Directory Structure Explanation:

The directory structure is organized in a way that fulfills the research objectives related to comparative studies, sentiment analysis, correlation analysis, data collection, and final prediction model development. Here's a brief explanation of each directory:

- `comparative_studies_nlp_regression`: This directory contains two sets of notebooks and documents related to comparative studies involving NLP and regression techniques. The first comparison to decide on which sentiment analysis models to use, and the second to compare regression models to decide on one of them of the final prediction model.

- `correlation_analysis`: Here, the correlation analysis of sentiment scores derived from different sources with stock price movements is conducted. Subdirectories like `correlation_analysis_financial_news`, `correlation_analysis_stock_twits`, and `correlation_analysis_twitter` contain notebooks specific to correlation analysis with financial news, StockTwits, and Twitter data on a daily, weekly, and monthly basis.

- `data`: This directory contains data collection and storage docuemntation. Notebooks like `data_collection_financial_news.ipynb` and `data_collection_stock_prices.ipynb` are used for collecting financial news and stock price data. Data obtained is stored in subdirectories like `financial_news/` and `stock_prices/`.

- `final_prediction_model_stocktwits`: This directory hosts notebooks for developing the final prediction model utilizing sentiment analysis from StockTwits data. Notebooks like `lstm_vader_daily.ipynb`, `lstm_vader_monthly.ipynb`, and `lstm_vader_weekly.ipynb` are involved in model development. They also include comparison of baseline LSTM and Linear regression models with sentiment-integrated LSTM, and Linear regression models.

- `utils`: Contains utility scripts such as `NewsProcessing.py`, `SentimentAnalysis.py`, and `StockPricesProcess.py`, which provide functionalities related to data processing, sentiment analysis, and correlation analysis that are used in multiple notebooks. These functions are reusable throughout the project.

- `requirements.txt`: This file lists all the dependencies required for running the code in the project, ensuring reproducibility and ease of setup.

This structured organization enables systematic exploration, analysis, and development of sentiment-based stock price prediction models.

```
+ comparative_studies_nlp_regression/
  + nlp_sentiment_analysis/
    - Against_Majority_Vote.ipynb
    - Against_StockNews_Labels.ipynb
    - FinancialPhraseBank.csv
    - FinancialPhraseExperiments.ipynb
    + comparison docs/
      - nlp_comparison_against_api_labels.docx
      - nlp_comparison_against_each_other.docx
  + stock_prediction_experiments/
    - gradient_boosting_prediction.ipynb
    - lstm_prediction.ipynb
    - random_forest_prediction.ipynb
    - stock_prediction_comparison.docx
+ correlation_analysis/
  + correlation_analysis_financial_news/
    - correlation_vader_aapl.ipynb
    - correlation_vader_aapl_monthly.ipynb
    - correlation_vader_aapl_weekly.ipynb
  + correlation_analysis_stock_twits/
    - stocktwits_vader_aapl.ipynb
    - stocktwits_vader_social_monthly.ipynb
    - stocktwits_vader_social_weekly.ipynb
  + correlation_analysis_twitter/
    - correlation_vader_twitter_aapl.ipynb
    - correlation_vader_twitter_aapl_monthly.ipynb
    - correlation_vader_twitter_aapl_weekly.ipynb
+ data/
  + data_collection/
    - data_collection_financial_news.ipynb
    - data_collection_stock_prices.ipynb
  + financial_news/
    + eodhd_api/
      - financial_news_data_eodhd.csv
      - financial_news_data_eodhd_AAPL.csv
    + stock_news_api/
      - all_tickers.csv
      - financial_news_data_stocknewsapi_AAPL.csv
      - financial_news_data_stocknewsapi_alltickers.csv
  + social_media_tweets/
    - stock_tweets.csv
  + stock_prices/
    + yfinance/
      - AAPL_prices.csv
      - AAPL_prices_2016-01-01_2020-01-01.csv
      - AAPL_prices_2019-01-01_2020-01-01.csv
  + stocktwits/
    - stocktwits_AAPL.csv
+ final_prediction_model_stocktwits/
  - lstm_vader_daily.ipynb
  - lstm_vader_monthly.ipynb
  - lstm_vader_weekly.ipynb
+ utils/
  - CorrelationHelpers.py
  - NewsProcessing.py
  - SentimentAnalysis.py
  - StockPricesProcess.py
- requirements.txt

```

### References

1. **Python Requests Tutorial: Request Web Pages, Download Images, POST Data**

   - Tutorial: [Python Requests Tutorial: Request Web Pages, Download Images, POST Data](https://realpython.com/python-requests/) (Accessed on April 3, 2024)
     - Reference: Stack Abuse. "Python Requests Tutorial: Request Web Pages, Download Images, POST Data." _Real Python_.

2. **Reading and Writing CSV Files in Python**

   - Tutorial: [Reading and Writing CSV Files in Python](https://realpython.com/python-csv/) (Accessed on April 3, 2024)
     - Reference: Stack Abuse. "Reading and Writing CSV Files in Python." _Real Python_.

3. **Using EOD Historical Data API**

   - Documentation: [Using EOD Historical Data API](https://eodhistoricaldata.com/financial-apis/) (Accessed on April 3, 2024)

4. **Using Stock News API**

   - Documentation: [Using EOD Historical Data API](https://stocknewsapi.com/) (Accessed on April 3, 2024)

5. **How to Get Stock Data using Python**

   - Tutorial: [How to Get Stock Data using Python](https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/) (Accessed on April 3, 2024)
     - Reference: Learn Data Science. "How to Get Stock Data using Python." _Learn Data Sci_.

6. **Using YFinance Library**

   - Tutorial: [yfinance Library – A Complete Guide](https://algotrading101.com/learn/yfinance-guide/) (Accessed on April 3, 2024)
     - Reference: AlgoTrading. "yfinance Library – A Complete Guide."

7. **Python Pandas Tutorial: A Complete Introduction for Beginners**

   - Tutorial: [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/) (Accessed on April 3, 2024)
     - Reference: Learn Data Science. "Python Pandas Tutorial: A Complete Introduction for Beginners." _Learn Data Sci_.

8. **Python OS Module: Your One-Stop Guide**

   - Tutorial: [Python OS Module: Your One-Stop Guide](https://realpython.com/python-os-module/) (Accessed on April 3, 2024)
     - Reference: Stack Abuse. "Python OS Module: Your One-Stop Guide." _Real Python_.

9. **IPython Magics - Autoreload**

   - Tutorial: [IPython Magics - Autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html)
     - Reference: IPython Project. "IPython Magics - Autoreload".

10. **NumPy Quickstart Tutorial**

    - Tutorial: [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
      - Reference: NumPy Developers. "NumPy Quickstart Tutorial".

11. **Pyplot Tutorial**

    - Tutorial: [Pyplot Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
      - Reference: Matplotlib Development Team. "Pyplot Tutorial".

12. **Python Seaborn Tutorial For Beginners**

    - Tutorial: [Seaborn Tutorial](https://www.datacamp.com/tutorial/seaborn-python-tutorial)
      - Reference: "Python Seaborn Tutorial For Beginners: Start Visualizing Data".

13. **Statistics in SciPy**

    - Tutorial: [SciPy](https://docs.scipy.org/doc/scipy/)
      - Reference: SciPy.org. "Statistics in SciPy".

14. **Understanding the Basics of Correlation in Python**

    - Tutorial: [Understanding the Basics of Correlation in Python](https://realpython.com/numpy-scipy-pandas-correlation-python/) (Accessed on April 3, 2024)
      - Reference: Real Python. "Understanding the Basics of Correlation in Python".

15. **NLTK Kit**

    - Tutorial: [NLTK Documentation](https://www.nltk.org/)
      - Reference: Steven Bird, Edward Loper, and Ewan Klein. "Natural Language Processing with Python." _NLTK Documentation_.

16. **Scikit-learn: Machine Learning in Python**

    - Tutorial: [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
      - Reference: Pedregosa et al. "Scikit-learn: Machine Learning in Python." _Journal of Machine Learning Research_, 12, 2825-2830, 2011.

17. **VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text**

    - Tutorial: [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://github.com/cjhutto/vaderSentiment)
      - Reference: Hutto, C.J. & Gilbert, E.E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." _Eighth International Conference on Weblogs and Social Media (ICWSM-14)_.

18. **TextBlob: Simplified Text Processing**

    - Tutorial: [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
      - Reference: Steven Loria. "TextBlob: Simplified Text Processing." _TextBlob Documentation_.

19. **Transformers: State-of-the-Art Natural Language Processing**

    - Tutorial: [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
      - Reference: Hugging Face Inc. "Transformers: State-of-the-Art Natural Language Processing." _Hugging Face Transformers Documentation_.

20. **Time Series Analysis in Python with Pandas**

    - Tutorial: [Time Series Analysis in Python with Pandas](https://realpython.com/python-time-series/)
      - Reference: Real Python. "Time Series Analysis in Python with Pandas." _Real Python_.

21. **VADER Implementation**

- Tutorial: [A Comprehensive Guide to Sentiment Analysis in Python](https://medium.com/@rslavanyageetha/vader-a-comprehensive-guide-to-sentiment-analysis-in-python-c4f1868b0d2e#:~:text=Sentiment%20analysis%20is%20a%20popular,trained%20model%20for%20sentiment%20analysis.)
  - Reference: R. Slavanya Geetha. "VADER: A Comprehensive Guide to Sentiment Analysis in Python." _Medium_.

24. **TextBlob Implementation**

- Tutorial: [A Step-by-Step Guide to Using TextBlob for NLP with Python](https://betterprogramming.pub/a-step-by-step-guide-to-using-textblob-for-nlp-with-python-157a7365a17b)
  - Reference: Better Programming. "A Step-by-Step Guide to Using TextBlob for NLP with Python."

25. **FinBERT Implementation**

- Tutorial: [FinBERT: Financial Sentiment Analysis with BERT](https://huggingface.co/ProsusAI/finbert)
  - Reference: Hugging Face Inc. "FinBERT: Financial Sentiment Analysis with BERT."

26. **Kaggle Dataset Used in the Comparison**

- Dataset: [Financial News Headlines](https://www.kaggle.com/notlucasp/financial-news-headlines)
  - Reference: Kaggle. "Financial News Headlines."

21. **Feature Engineering for Machine Learning in Python**

    - Tutorial: [Feature Engineering for Machine Learning in Python](https://www.datacamp.com/community/tutorials/feature-engineering-python)
      - Reference: DataCamp Community. "Feature Engineering for Machine Learning in Python." _DataCamp_.

22. **Python Sentiment Analysis using VADER**

    - Tutorial: [Python Sentiment Analysis using VADER](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)
      - Reference: [GeeksforGeeks](https://www.geeksforgeeks.org/). "Python Sentiment Analysis using VADER."

23. **Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras**

    - Tutorial: [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
      - Reference: Jason Brownlee. "Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras." _Machine Learning Mastery_.

24. **Time Series Data Visualization with Python**

    - Tutorial: [Working with Time Series Data](https://machinelearningmastery.com/time-series-data-visualization)

25. **Linear Regression in Python**
    - Tutorial: [Linear Regression in Python](https://realpython.com/linear-regression-in-python/)
      - Reference: [Real Python](https://realpython.com/). "Linear Regression in Python".
