import pandas as pd
from datetime import datetime
import helper as h

def main( df30m : pd.DataFrame, days, PIP_SIZE ):
    new_data = None
    data = None
    cur_day = 0

    tmp_volume = 0
    tmp_trades = 0

    for index, row in df30m.iterrows():
        dt = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S")

        # # # # # # # # #
        #if( dt.month == 12 and dt.day == 5 and dt.year == 2018 ): continue # @ES
        # # # # # # # # #
        if( dt.weekday() < 0 or dt.weekday() > 4 ): continue

        if dt.hour != 8 or dt.minute != 0: continue

        tmp_high = h.string_to_num(row["High"])
        tmp_low = h.string_to_num(row["Low"])

        if cur_day == 0:
            low = 0
            high = 0
            cur_day = dt.day
        elif cur_day != dt.day:
            #day = [cur_day, low, high, ((high-low)/PIP_SIZE)+1]
            days.append([cur_day, low, high, ((high-low)/PIP_SIZE)+1])
            data = pd.concat([data, new_data], ignore_index=True)
            low = 0
            high = 0
            cur_day = dt.day

        #print(df30m.loc[[index]])
        #print(df30m.loc[[index]][['Volume', 'Trades', 'Bar Size', 'High', 'Low', 'Open', 'Close']])
        #exit()

#"Date";"Volume";"Trades";"Bar Size";"High";"Low";"Open";"Close";"Delta T/D";"Delta Aggressor";"H/L Side";"O/C Side";"O/C Size";"Max Volume";"M/V Price";"M/V Trades";"M/V Delta T/D";"M/V Delta Aggressor";"Max Trade";"Max Tick";"Ticks Number";"Hidden Volume";"TPO High";"TPO Low";"COT High";"COT Low";"Max Delta T/D";"Max Delta Aggressor"

        #data = pd.concat([data, df30m.loc[[index]]], ignore_index=True)
        mv_price = h.string_to_num(row['M/V Price'])
        full = tmp_high - tmp_low
        to_mv = mv_price - tmp_low
        #print(full)
        #print(to_mv)
        #print(to_mv/full)

        new_data = df30m.loc[[index]][['Date', 'Volume', 'Trades', 'Bar Size']]#, 'Max Volume']]
        if full == 0:
            new_data['Uprise'] = 0
        else:
            new_data['Uprise'] = to_mv/full

        if low == 0 and high == 0: 
            low = tmp_low
            high = tmp_high
            continue

        if low > tmp_low:
            low = tmp_low
        if high < tmp_high:
            high = tmp_high
    
    days.append([cur_day, low, high, ((high-low)/PIP_SIZE)+1])
    data = pd.concat([data, new_data], ignore_index=True)

    return data