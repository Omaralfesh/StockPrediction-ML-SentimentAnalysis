# def get_correlaton_for_lags(df, lags):
#     for lag in lags:
#         df[f'Sentiment_Lag_{lag}'] = df['sentiment_score'].shift(lag)
#         pearson_corr_lag = df[f'Sentiment_Lag_{lag}'].corr(df['Close'], method='pearson')
#         print(f"Pearson Correlation with Lag {lag}: {round(pearson_corr_lag, 4)}")

