import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from pathlib import Path
import pickle
from joblib import dump, load
import config as cfg

def new_machine_learning_monthly(data : pd.DataFrame):
    # # # # # # # # # # # #
    # Die Daten werden separiert pro Monat, für das erste Monat wird ein RF erstellt und damit das zweite Monat vorhergesagt
    # Danach wird für das zweite Monat ein RF erstellt und das dritte vorhergesagt, usw.
    # # # # # # # # # # # #
    month = 99          # current month
    month_data = []     # List of all monthly data
    new_month = pd.DataFrame()    # Temporary data of new month
    
    accs = 0
    accs_count = 0
    
    for index, row in data.iterrows():
        dt = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S")
        if dt.month != month:  # Check if current entry is new month
            month = dt.month   # Save month as new month for check
            if new_month.empty == False:       # If there is data available ( not at the first run )
                # Add data to month_data and reset new_month
                month_data.append(new_month.loc[:, new_month.columns != "Date"])    
                new_month = pd.DataFrame()
        
        # Add data without Date column
        if new_month.empty == True:
            new_month = pd.DataFrame(data.iloc[[index]])
        else:
            new_month = pd.concat([new_month, data.iloc[[index]]], ignore_index=True)

    month_data.append(new_month.loc[:, new_month.columns != "Date"])

    rf = None
    for md in month_data:
        # seperate features and lables for each month
        features = md.loc[:, md.columns != 'Result']
        labels = md.loc[:,'Result']
        
        # if ther is already a RF (not for the first run), make a prediction for the current data
        if rf != None:
            prediction = rf.predict(features)
            accuracy = metrics.accuracy_score(labels, prediction) * 100
            accs += accuracy
            accs_count += 1
            print('Accuracy:', round(accuracy,4), '%')
        
        # create a new RF for the current data        
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(features, labels)
    
    if accs_count > 0:
        print(f"Avg. Accuracy: {round(accs/accs_count, 2)} %") 
    else:
        print("No prediction done!")      

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

def exportClassifier( clf ):
    Path(f"data/classifier").mkdir(parents=True, exist_ok=True)
    s = pickle.dumps(clf)
    dump(s, 'data/classifier/new.joblib')

def getLocalClassifier():
    s = load('data/classifier/new.joblib')
    return( pickle.loads(s) )