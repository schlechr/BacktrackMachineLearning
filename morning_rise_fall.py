import pandas as pd
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import analyze_data as ad
import daily_result as dr
import helper as h

#YEARS = range(2010, 2022)
YEARS = range(2020, 2021)
MONTHS = range(1,13)
MARKET = 'FESX'
PIP_SIZE = 0.5
GV_PER_PIP = 5
RATE_TO_USD = 1.13

def create_data(all_trades, y, m):
    try:
        df1m = pd.read_csv(f"./data/{MARKET}/1M/{y}/{m:02d}.csv", sep=";")
        df30m = pd.read_csv(f"./data/{MARKET}/30M/{y}/{m:02d}.csv", sep=";")
    except:
        return
    
    days = []

    data = ad.main(df30m, days, PIP_SIZE)

    dr.main(df1m, all_trades, days, PIP_SIZE, GV_PER_PIP)

    return data

def get_result( data, all_trades ):
    pos = 0
    result = []
    for a in all_trades:
        if a[2] > 0:
            pos += 1
            result.append(1)
        elif a[2] < 0:
            result.append(-1)
        else:
            result.append(0)

    data['Result'] = result

    print(data)
    print(f"{len(all_trades)} -> Pos: {pos}")
    print(f"Processing time...{round(time.time() - start, 2)} sec")

def machine_learning(data):
    accs = 0
    trys = 10
    for _ in range(trys):
        trainF, testF, trainL, testL = train_test_split(data.loc[:, data.columns != 'Result'], data.loc[:,'Result'], test_size=0.25)

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainF, trainL)

        prediction = rf.predict(testF)
        accuracy = metrics.accuracy_score(testL, prediction) * 100
        print('Accuracy:', accuracy, '%')
        accs += accuracy
        #print(prediction)
        #print(testL)
    print(f"Avg. Accuracy: {round(accs/trys,2)} %")

if __name__ == "__main__":
    start = time.time()

    all_trades = []
    data = None
    for y in YEARS:
        for m in MONTHS:
            data = pd.concat([data, create_data(all_trades, y, m)], ignore_index=True)

    get_result( data, all_trades )
    
    machine_learning(data.loc[:, data.columns != 'Date'])

    # res = 0
    # for i in all_trades:
    #     res += i[2]
    # print(round(res,2))