# Google Trends for Stock Price Prediction

Author: Jack Olson

---

## Problem Statment

Could using Google Trends improve the prediction of stock price changes?

![PTON Close and Trend Score](./Pictures/PTON_Close_and_Trend_Score.png)

---

## Outline

1. Introduction
2. Predicting Stock Prices With and Without Google Trends
3. History of Google Trends as a Stock Trading Tool
4. Conclusions

---

## Introduction

**What is Google Trends?**

Google Trends is a free online tool that helps visualize trends in a term's searching popularity. The output of Google Trends is simply a time series of integers between 1 and 100. A 100 indicates the time when the search was most popular over the time range selected. A 1 indicates when the search was least popular over that period.

To collect Trends data I used the PyTrends API. For each ticker in my dataset I pulled the trend score for the period 1.1.18-6.1.21. If a ticker's IPO was after 1.1.18 the trend score before IPO was dropped from my data.

**Google Trends as a leading Indicator for stock price**

For Google Trends to be useful for stock trading it would need to be a leading indicator of stock price. The first stock to catch my interest in this subject was the popular exercise bike company Peloton.

As the stock chart above shows, PTON was trading at about 84 dollars as of 9.11.20, the period where Google Trends shows the maximum interest in searches for "PTON". In the months following the price of the stock rose by 104%.

## Predicting Stock Prices With and Without Google Trends

### **ARIMA**

ARIMA uses historical prices to predict future prices. The only input to the model is the target time series. This means that ARIMA is simple to use and can be applied to any ticker. It is popular for stock modeling and a good baseline model for my purposes. 

ARIMA worked well on some of the tickers in my dataset but badly on others. This is because ARIMA tended to predict the same shape for any ticker. It would predict a flat trend at first and after a few weeks predict exponential gains or losses. If a ticker happened to follow this shape in the testing period then ARIMA would look very good. See the images below.

![ARIMA Results](./Pictures/ARIMA_Results.png)

### **ARIMA with an Exogeneous Variable (ARIMAX)**

ARIMAX is the first model I used that takes in Google Trends as an input. ARIMAX made trend score available as a weighted input. Unlike my third model, VAR, ARIMAX does not directly model or predict trend score.

ARIMAX performed worse for all five tickers in my dataset compared to ARIMA. See the scores below. AIC is the most popular error metric for time series. The lower the AIC, the better. Negative AIC is acceptable, and the lower the negative number the better.

| Ticker                  | ARIMA AIC | ARIMAX AIC | Variance | Variance/ARIMA AIC |
|-------------------------|-----------|-------------|----------|--------------------|
| PTON (Peloton)          | -1178.22  | -1143.44    | 34.78    | 2.95%              |
| IBM                     | -4061.98  | -3997.10    | 64.88    | 1.60%              |
| GME (GameStop)          | -1493.96  | -1350.03    | 143.93   | 9.63%              |
| Apple                   | -3826.80  | -3706.63    | 120.17   | 3.14%              |
 BXP (Boston Properties) | -3750.09  | -3722.37    | 27.72    | 0.74%              |

The explanations for why ARIMAX performed worse than ARIMA fall into two buckets, in my opinion:
- Google Trends made ARIMAX worse because trend score was misleading
- My error: I may have selected better parameters for ARIMA than I did for ARIMAX. Generally, comparisons between different models can be misleading for this reason.

### **Vector Autoregression (VAR) Modeling**

VAR assumes that multiple time series in a system influence each other. One way to test this assumption on a ticker is Granger's Causality Test. Granger's Causality tests the null hypothesis that two time series do not influence each other. If the p-value returned by the test is small enough (<5%), the null can be rejected. Rejecting the null implies that the two time series do have an influencing relationship and can be put into VAR. Below are the results of applying Granger's Causality Test to the five tickers in my dataset.

| Ticker                  | GT Influences Price       | Price Influences GT      |
|-------------------------|---------------------------|--------------------------|
| PTON (Peloton)          | 5.79%, CAN'T REJECT NULL  | 0.17%, REJECT NULL       |
| IBM                     | 28.84%, CAN'T REJECT NULL | 6.03%, CAN'T REJECT NULL |
| GME (GameStop)          | 0%, REJECT NULL           | 0%, REJECT NULL          |
| Apple                   | 5.7%, CAN'T REJECT NULL   | 0%, REJECT NULL          |
| BXP (Boston Properties) | 10.8%, CAN'T REJECT NULL  | 0%, REJECT NULL          |

The only ticker where I could reject the null in both cases is GME. That means I should only use VAR for GME. 

The results of VAR were great for GME. But for the first two tickers I found that also passed Granger's Causality Test VAR did not perform well at all. See below. I did not score VAR compared to my other two models because VAR predicts multiple time series so the AIC numbers aren't comparable.

![GME VAR](./Pictures/GME_VAR.png)

![Other Tickers VAR](./Pictures/other_VAR.png)

## History of Google Trends as a Stock Trading Tool

**[Google Trends for Investor Sentiment Analysis](https://seekingalpha.com/article/4191521-using-google-trends-to-predict-stocks)**

Harrison Schwartz of Seeking Alpha constructed an investor sentiment index using Google Trends. The concept was that certain words are associated with a bullish market and others are associated with a bearish market. Using the trend score of these words Schwartz would assess if the market sentiment was overall bearish or bullish. When the index was bearish the algorithm would buy consumer staples. When it was bullish it would buy consumer discretionary. 

The results of this trading strategy were very encouraging. It outperformed the S&P 500 by a considerable amount in simulations. This was the best example I found of how Google Trends has potential as a stock trading tool. 

The difference between Schwartz's sentiment index and my research is that I am studying Google Trends for individual stock prediction. Schwartz's sentiment index only works for the market as a whole.

**[Correlation Between Other Trend Scores and Stock Price](https://www.ccom.ucsd.edu/~cdeotte/papers/GoogleTrends.pdf)**

Dartanyon Shivers of UCSD wrote about using selected keywords to predict stock price movement. Shivers' methodology was:
1. Search for a keyword whos' trend score correlated with a stock's price during the training period
2. See if that correlation continued into the testing period

For example, Shivers found that the price of RCL had a very high correlation with the Google Trend score for "popcorn" 1.1.12-12.31.16. Unsurprisingly, this correlation went away immediately over the testing period 1.1.17-4.30.17. This is because there isn't an explanation of why interest in popcorn would have anything to do with the stock price of a cruise line.

Shiver's research is a reminder to avoid confirmation bias when it comes to using Google Trends to predict stock prices. In my research I used the trend score of a stock ticker's name because that is not biased.

## Conclusion

**The usefulness of Google Trends for predicting the price of individual stocks is uncertain.** ARIMA was my best performing model and it did not use trend score as an input. That being said, it's hard to say that ARIMA performing better can be attributed to Google Trends. It might be that ARIMA was set up better and I made errors with my use of the other two model types. 

Despite the lackluster performance of my models that used trend score, there were encouraging findings. First, Granger's Causality test shows that for some tickers Google Trend score and stock price do have a relationship. In addition, Schwartz's Google Trends Sentiment Index was successful in simulations.

## Next Steps

1. Improve Model Accuracy
   - The better I can make my models the more comparable they will become. At this point, the difference between models is due party to the fact that aren't optimized.
2. Investigate Using Google Trends for a Trading Algorithm
   - I would be interested to see if Google Trends could be used as a trading strategy at the individual stock level. For example, I wonder how an algorithm would do that purchases stocks when the trend score is high and sells when the trend score is low.
3. Test and Score Models On a Shorter prediction Time Frame
   - Time series models aren't the best for predicting more than a few periods out. I would be interested to compare which models are the best for predicting the next day or just a week out. The problem with this is that Google Trends data is weekly.