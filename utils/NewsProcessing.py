import pandas as pd

# Function to load news data from a CSV file within a specified date range
def load_news_df(path, start_date_string, end_date_string):
    news_data = pd.read_csv("../../data/financial_news/stock_news_api/financial_news_data_stocknewsapi_AAPL.csv")
    
    # Converting 'date' column to datetime objects with specified format
    news_data["Date"] = pd.to_datetime(news_data["date"], format="%a, %d %b %Y %H:%M:%S %z")

    # Converting start_date_string and end_date_string to datetime objects with UTC timezone
    start_date = pd.to_datetime(start_date_string, utc=True)
    end_date = pd.to_datetime(end_date_string, utc=True)

    # Filtering news_data DataFrame to include only rows within the specified date range
    news_data = news_data.loc[(news_data["Date"] >= start_date) & (news_data["Date"] <= end_date)]
    
    # Combining 'text' and 'title' columns into a single column
    news_data["concat_text"] = news_data["text"] + " " + news_data["title"]
    
    # Converting 'Date' column to date format
    news_data['Date'] = pd.to_datetime(news_data['Date'], utc=True).dt.date

    return news_data