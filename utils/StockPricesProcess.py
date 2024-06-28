# ### Tutorials

# 1. **Loading and Manipulating Time Series Data with pandas**
#    - Tutorial: [Time Series Analysis in Python with Pandas](https://realpython.com/python-time-series/)
#      - Reference: Real Python. "Time Series Analysis in Python with Pandas." *Real Python*.

# 2. **Data Preprocessing and Feature Engineering**
#    - Tutorial: [Feature Engineering for Machine Learning in Python](https://www.datacamp.com/community/tutorials/feature-engineering-python)
#      - Reference: DataCamp Community. "Feature Engineering for Machine Learning in Python." *DataCamp*.



import pandas as pd

# Function to load stock price data from a CSV file within a specified date range
def load_prices_df(path, start_date_string, end_date_string):
    # Reading stock data from CSV file into a DataFrame
    stock_data = pd.read_csv(path)
    
    # Converting 'Date' column to datetime objects with specified format and UTC timezone
    stock_data["Date"] = pd.to_datetime(stock_data["Date"], format="%Y-%m-%d %H:%M:%S%z", utc=True)

    # Converting start_date_string and end_date_string to datetime objects with UTC timezone
    start_date = pd.to_datetime(start_date_string, utc=True)
    end_date = pd.to_datetime(end_date_string, utc=True)

    # Filtering stock_data DataFrame to include only rows within the specified date range
    stock_data = stock_data.loc[(stock_data["Date"] >= start_date) & (stock_data["Date"] <= end_date)]
    
    # Setting 'Date' column as index and sorting DataFrame by date
    stock_data = stock_data.set_index("Date")
    stock_data.sort_values(by="Date", inplace=True)

    return stock_data

# Function to add lagged 'Close' prices as features to stock data
def add_lags(stock_data, numLags):
    # Adding lagged 'Close' prices as features
    for i in range(1, numLags + 1):
        stock_data[f"Close_Lag{i}"] = stock_data["Close"].shift(i)
    
    # Filling missing values in lagged features with backward fill
    for i in range(1, numLags + 1):
        stock_data[f"Close_Lag{i}"] = stock_data[f"Close_Lag{i}"].bfill()

    return stock_data