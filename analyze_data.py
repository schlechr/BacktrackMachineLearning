import pandas as pd
from datetime import datetime
import helper as h
import config as cfg

def main( df30m : pd.DataFrame, days ):
    new_data = None
    data = None
    cur_day = 0

    for index, row in df30m.iterrows():
        dt = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S")

        if( dt.weekday() < 0 or dt.weekday() > 4 ): continue

        if dt.hour != 8 or dt.minute != 0: continue

        tmp_high = h.string_to_num(row["High"])
        tmp_low = h.string_to_num(row["Low"])

        if cur_day == 0:
            low = 0
            high = 0
            cur_day = dt.day
        elif cur_day != dt.day:
            days.append([cur_day, low, high, ((high-low)/cfg.PIP_SIZE)+1])
            data = pd.concat([data, new_data], ignore_index=True)
            low = 0
            high = 0
            cur_day = dt.day

        new_data = create_new_data(df30m.loc[[index]], cfg.USING_VALUES)

        if low == 0 and high == 0: 
            low = tmp_low
            high = tmp_high
            continue

        if low > tmp_low:
            low = tmp_low
        if high < tmp_high:
            high = tmp_high
    
    days.append([cur_day, low, high, ((high-low)/cfg.PIP_SIZE)+1])
    data = pd.concat([data, new_data], ignore_index=True)

    return data

def create_new_data(data : pd.DataFrame, fields = []):
    pos_fields = ["Date", "Volume","Trades","Bar Size","High","Low","Open","Close","Delta T/D","Delta Aggressor","H/L Side","O/C Side","O/C Size","Max Volume","M/V Price","M/V Trades","M/V Delta T/D","M/V Delta Aggressor","Max Trade","Max Tick","Ticks Number","Hidden Volume","TPO High","TPO Low","COT High","COT Low","Max Delta T/D","Max Delta Aggressor"]

    if fields == []:
        fields = pos_fields
    if "Date" not in fields:
        fields.append("Date")

    new_data = pd.DataFrame(columns=fields)

    for f in pos_fields:
        if f in fields:
            new_data[f] = h.string_to_num(data[f])

    #new_data['Weekday'] = datetime.strptime(data["Date"].values[0],"%Y-%m-%d %H:%M:%S").weekday()
    #new_data['Day'] = datetime.strptime(data["Date"].values[0],"%Y-%m-%d %H:%M:%S").day
    #new_data['Month'] = datetime.strptime(data["Date"].values[0],"%Y-%m-%d %H:%M:%S").month
    return new_data