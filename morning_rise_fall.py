import pandas as pd
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import analyze_data as ad
import daily_result as dr
import config as cfg
from sklearn.model_selection import KFold
from pathlib import Path
import pickle
from joblib import dump, load
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
    
def machine_learning(data):
    tmp_acc = 0
    accs = 0
    splits = 5

    features = data.loc[:, data.columns != 'Result']
    labels = data.loc[:,'Result']

    # pca = PCA()
    # features = pca.fit_transform(features)
    # features = pd.DataFrame(features)

    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(data):
        trainF, testF = features.iloc[train_index], features.iloc[test_index]
        trainL, testL = labels.iloc[train_index], labels.iloc[test_index]

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainF, trainL)

        prediction = rf.predict_proba(testF)
        res = []

        for p in prediction:
            tmp_prob = 0
            tmp_pred = ''
            for i, c in enumerate(rf.classes_):
                if p[i] >= cfg.MIN_PROB and p[i] > tmp_prob:
                    tmp_prob = p[i]
                    tmp_pred = c
            res.append(tmp_pred)

        new_labels = []
        new_res = []
        addM = 0

        for i in range(0, len(res)):
            if res[i] != '':
                try:
                    new_labels.append(testL[test_index[i]+addM])
                except:
                    addM += 1
                    new_labels.append(testL[test_index[i]+addM])
                new_res.append(res[i])
        accuracy = metrics.accuracy_score(new_labels, new_res) * 100
        accs += accuracy
        print(f'Found a prediction for {round((len(new_res)/len(res))*100, 2)} %')
        print('Accuracy:', round(accuracy, 4), '%')

        if tmp_acc < accuracy:
            exportClassifier( rf )
            tmp_acc = accuracy

    print(f"Avg. Accuracy: {round(accs/splits, 2)} %")
    print(f"Processing time...{round(time.time() - start, 2)} sec")

def local_machine_learning(data):
    features = data.loc[:, data.columns != 'Result']
    labels = data.loc[:,'Result']
    rf = getLocalClassifier()

    prediction = rf.predict_proba(features)
    res = []

    for p in prediction:
        tmp_prob = 0
        tmp_pred = ''
        for i, c in enumerate(rf.classes_):
            if p[i] >= cfg.MIN_PROB and p[i] > tmp_prob:
                tmp_prob = p[i]
                tmp_pred = c
        res.append(tmp_pred)

    new_labels = []
    new_res = []

    for i in range(len(res)):
        if res[i] != '':
            new_labels.append(labels[i])
            new_res.append(res[i])
    accuracy = metrics.accuracy_score(new_labels, new_res) * 100
    print(f'Found a prediction for {round((len(new_res)/len(res))*100, 2)} %')
    print('Accuracy:', round(accuracy, 4), '%')

    # prediction = rf.predict(features)
    # accuracy = metrics.accuracy_score(labels, prediction) * 100
    # print('Accuracy:', round(accuracy,4), '%')

    print(f"Processing time...{round(time.time() - start, 2)} sec")

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

def exportClassifier( clf ):
    Path(f"data/classifier").mkdir(parents=True, exist_ok=True)
    s = pickle.dumps(clf)
    dump(s, 'data/classifier/new.joblib')

def getLocalClassifier():
    s = load('data/classifier/new.joblib')
    return( pickle.loads(s) )

if __name__ == "__main__":
    start = time.time()

    all_trades = []
    data = None
    for y in cfg.YEARS:
        for m in cfg.MONTHS:
            data = pd.concat([data, create_data(all_trades, y, m)], ignore_index=True)

    data = get_result( data, all_trades )
    
    if True:
        machine_learning(data.loc[:, data.columns != 'Date'])
    else:
        local_machine_learning(data.loc[:, data.columns != 'Date'])

    plot_all(data.loc[:, data.columns != 'Date'])

    # res = 0
    # for i in all_trades:
    #     res += i[2]
    # print(round(res,2))