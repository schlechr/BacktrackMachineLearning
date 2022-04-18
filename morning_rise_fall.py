import pandas as pd
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import analyze_data as ad
import daily_result as dr
import config as c
from sklearn.model_selection import KFold

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
        df1m = pd.read_csv(f"./data/{c.MARKET}/1M/{y}/{m:02d}.csv", sep=";")
        df30m = pd.read_csv(f"./data/{c.MARKET}/30M/{y}/{m:02d}.csv", sep=";")
    except:
        return
    
    days = []

    data = ad.main(df30m, days, c.PIP_SIZE)

    dr.main(df1m, all_trades, days, c.PIP_SIZE, c.GV_PER_PIP)

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
            print(dt.date())
            print(all_trades[index][0])
            #exit()

    remove_no_trades.reverse()
    for r in remove_no_trades:
        data = data.drop(r)

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

    print(data)
    print(f"{len(all_trades)} -> Pos: {pos} -> Res: {res}")
    print(f"Processing time...{round(time.time() - start, 2)} sec")

    return data

def machine_learning(data):
    accs = 0
    splits = 5

    features = data.loc[:, data.columns != 'Result']
    labels = data.loc[:,'Result']

    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(data):
        trainF, testF = features.iloc[train_index], features.iloc[test_index]
        trainL, testL = labels.iloc[train_index], labels.iloc[test_index]

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainF, trainL)

        prediction = rf.predict(testF)
        accuracy = metrics.accuracy_score(testL, prediction) * 100
        print('Accuracy:', round(accuracy,4), '%')
        accs += accuracy

    print(f"Avg. Accuracy: {round(accs/splits, 2)} %")
    print(f"Processing time...{round(time.time() - start, 2)} sec")

if __name__ == "__main__":
    start = time.time()

    all_trades = []
    data = None
    for y in c.YEARS:
        for m in c.MONTHS:
            data = pd.concat([data, create_data(all_trades, y, m)], ignore_index=True)

    data = get_result( data, all_trades )
    
    machine_learning(data.loc[:, data.columns != 'Date'])

    # res = 0
    # for i in all_trades:
    #     res += i[2]
    # print(round(res,2))