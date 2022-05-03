import pandas as pd
from datetime import datetime
import helper as h
import config as cfg

def main( df1m : pd.DataFrame, all_trades, days):
    cur_day = 0
    low_limit = 0
    high_limit = 0
    open_trade = []
    skip_day = False
    last_dt = None

    #print(data)
    
    for index, row in df1m.iterrows():
        dt = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S")

        if( dt.weekday() < 0 or dt.weekday() > 4 ): continue

        if dt.hour < 8 or (dt.hour == 8 and dt.minute < 30): continue
        #if dt.hour <= 8: continue

        if last_dt != None and dt.date() != last_dt.date():
            if all_trades[len(all_trades)-1][0] != last_dt.date():
                all_trades.append([last_dt.date(), 0, 0])
        
        if dt.hour >= 15: 
            if open_trade != []:
                close = h.string_to_num(row["Close"])
                if open_trade[0] == "LONG":
                    all_trades.append([dt.date(), open_trade[0],((close - high_limit) / cfg.PIP_SIZE) * cfg.GV_PER_PIP])
                elif open_trade[0] == "SHORT":
                    all_trades.append([dt.date(), open_trade[0],((low_limit - close) / cfg.PIP_SIZE) * cfg.GV_PER_PIP])
                open_trade = []
            continue
        elif dt.hour >= 12 and open_trade == [] and skip_day == False and found_day == True:
            open_trade = ["No Trade"]
            all_trades.append([dt.date(), 0, 0])
            skip_day = True

        last_dt = dt

        #if PIP_SIZE != 0.5 and ( dt.year > 2021 or (dt.year == 2021 and dt.month > 6) or (dt.year == 2021 and dt.month == 6 and dt.day >= 21)):
        #    set_pip_size(0.5)

        if cur_day != dt.day:
            skip_day = False
            found_day = False
            cur_day = dt.day
            open_trade = []
            for i in days:
                if i[0] == cur_day:
                    low_limit = i[1] - cfg.PIP_SIZE
                    high_limit = i[2] + cfg.PIP_SIZE
                    tp_sl = ((i[2]-i[1])/cfg.PIP_SIZE)+cfg.PIP_SIZE
                    found_day = True
                    break

        if skip_day == True: continue
        if found_day == False: continue

        tmp_high = h.string_to_num(row["High"])
        tmp_low = h.string_to_num(row["Low"])

        if open_trade == []:
            if tmp_high > high_limit > tmp_low:
                open_trade = ["LONG", high_limit + tp_sl*cfg.PIP_SIZE, low_limit]
            elif tmp_high > low_limit > tmp_low:
                open_trade = ["SHORT", low_limit - tp_sl*cfg.PIP_SIZE, high_limit]

        else:
            if tmp_high > open_trade[1] > tmp_low:
                all_trades.append([dt.date(), open_trade[0], tp_sl*cfg.GV_PER_PIP])
                open_trade = []
                skip_day = True
            elif tmp_high > open_trade[2] > tmp_low:
                all_trades.append([dt.date(), open_trade[0], (tp_sl*cfg.GV_PER_PIP)*-1])
                open_trade = []
                skip_day = True