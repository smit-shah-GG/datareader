"""
Already logged:
>> Google
>> Microsoft
>> Apple
"""

import yfinance as yf

company = "AAPL"
filename = "%s_Info.txt" % company

google = yf.Ticker(company)
data = yf.download(company)

with open(filename, 'w') as f:
    f.write("\n \n " + company + " Stock price data: \n \n")    
    f.writelines(str(data))
    f.write("\n \n")
    f.write(company + ' info: \n\n')
    f.writelines(str(google.major_holders))
    f.writelines(str(google.sustainability))
    f.writelines(str(google.financials))
    f.writelines(str(google.cashflow))
    f.writelines(str(google.recommendations))
    f.writelines(str(google.balance_sheet))



"""
List of available methods:

>> .info
>> .history
>> .actions
>> .dividends
>> .splits
>> .financials, .quaterly_financials
>> .major_holders, .institutional_holders
>> .balance_sheet, .quaterly_balance_sheet
>> .cashflow, .quaterly_cashflow
>> .earnings, .quaterly_earnings
>> .sustainability
>> .recommendations
>> .calendar
>> .isin                                   SHOWS INTERNATIONAL SECURITIES IDENTIFICATION NUMBER, IS EXPERIMENTAL
>> .options
>> .news

Returns a pandas df as object, eg. googl here is a pandas df

Can also make a 2D dataframe that includes multiple tickers:
Can be done as data = yf.download("MSFT GOOGL")

"""

