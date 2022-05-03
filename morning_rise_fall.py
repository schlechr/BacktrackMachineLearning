import pandas as pd
from datetime import datetime
import time
import analyze_data as ad
import daily_result as dr
import config as cfg
import machine_learning as ml
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# # # # # # # # # # # # # # # # # # # # # #
# Following values needs to be configurated in a seperate "config.py file"
# YEARS = range(2020, 2021)
# MONTHS = range(1,13)
# MARKET = 'ES'
# PIP_SIZE = 0.25
# GV_PER_PIP = 12.5
# RATE_TO_USD = 1
# # # # # # # # # # # # # # # # # # # # # #

def create_data(all_trades, y, m):
    try:
        df1m = pd.read_csv(f"./data/{cfg.MARKET}/1M/{y}/{m:02d}.csv", sep=";")
        df30m = pd.read_csv(f"./data/{cfg.MARKET}/30M/{y}/{m:02d}.csv", sep=";")
    except:
        return
    
    days = []

    data = ad.main(df30m, days)

    dr.main(df1m, all_trades, days)

    return data

def get_result( data : pd.DataFrame, all_trades ):
    pos = 0
    res = 0
    result = []

    remove_no_trades = []
    # Check if there is a issue with the date of the data
    for index, row in data.iterrows():
        dt = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S")
        if dt.date() != all_trades[index - len(remove_no_trades)][0]:
            if dt.date() < all_trades[index - len(remove_no_trades)][0]:
                remove_no_trades.append(index)
                continue
            print("Date does not match")
            print(dt.date())
            print(all_trades[index][0])
            exit()

    remove_no_trades.reverse()
    for r in remove_no_trades:
        data = data.drop(r)
    data = data.reset_index(drop=True)

    for a in all_trades:
        res += a[2]
        if a[2] > 0:
            pos += 1
            result.append(1)
        elif a[2] < 0:
            result.append(-1)
        else:
            result.append(0)

    data['Result'] = result

    # print(data)
    print(f"{len(all_trades)} -> Pos: {pos} -> Res: {res}")
    print(f"Processing time...{round(time.time() - start, 2)} sec")

    return data

def plot_all( data : pd.DataFrame ):

    features = data.loc[:, data.columns != 'Result']
    labels = data.loc[:,'Result']

    if features.shape[1] <= 1:
        return

    color = []
    for l in labels:
        if l == 1:
            color.append("g")
        elif l == -1:
            color.append("r")
        else:
            color.append("b")    

    pca = PCA()
    pca_data = pca.fit_transform(features)
    x = []
    y = []
    for p in pca_data:
        x.append(p[0])
        y.append(p[1])
    
    plt.scatter(x, y, marker="x", color=color)
    
    plt.show()

if __name__ == "__main__":
    start = time.time()

    all_trades = []
    data = None
    for y in cfg.YEARS:
        for m in cfg.MONTHS:
            data = pd.concat([data, create_data(all_trades, y, m)], ignore_index=True)

    data = get_result( data, all_trades )
    
    if cfg.MODE == 0:
        ml.machine_learning(data.loc[:, data.columns != 'Date'])
    elif cfg.MODE == 1:
        ml.local_machine_learning(data.loc[:, data.columns != 'Date'])
    elif cfg.MODE == 2:
        ml.new_machine_learning_monthly(data)
    elif cfg.MODE == 3:
        ml.new_machine_learning_timeset(data)
    elif cfg.MODE == 4:
        ml.new_machine_learning_weekday(data)

    print(f"Processing time...{round(time.time() - start, 2)} sec") 

    plot_all(data.loc[:, data.columns != 'Date'])

    # res = 0
    # for i in all_trades:
    #     res += i[2]
    # print(round(res,2))