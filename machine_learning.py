import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from pathlib import Path
import pickle
from joblib import dump, load
import config as cfg

def new_machine_learning_weekday(data):
    # # # # # # # # # # # #
    # Die Daten werden separiert pro Wochentag gespeichert und es wird ein normaler KFold darüber gemacht
    # # # # # # # # # # # #
    weekly = [ pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() ]
    splits = 3
    #new_data = None     
    
    accs = 0
    accs_count = 0
    tmp_accs = 0
    tmp_accs_count = 0
    
    for index, row in data.iterrows():
        wd = datetime.strptime(row["Date"],"%Y-%m-%d %H:%M:%S").weekday()
        
        # Add data without Date column
        if weekly[wd].empty == True:
            weekly[wd] = data.iloc[[index]]
        else:
            weekly[wd] = pd.concat([weekly[wd], data.iloc[[index]]], ignore_index=True)
    
    kf = KFold(n_splits=splits)
    for i, wd_data in enumerate(weekly):
        data = wd_data.loc[:, wd_data.columns != "Date"]
        for train_index, test_index in kf.split(data):
            trainF, testF = data.iloc[train_index, data.columns != "Result"], data.iloc[test_index, data.columns != "Result"]
            trainL, testL = data.iloc[train_index, data.columns == "Result"], data.iloc[test_index, data.columns == "Result"]

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(trainF, trainL["Result"])
            
            prediction = rf.predict(testF)
            accuracy = metrics.accuracy_score(testL["Result"], prediction) * 100
            tmp_accs += accuracy
            tmp_accs_count += 1
            print(f'{i}: Accuracy: {round(accuracy,4)}%')
        
        print(f"{i}: Avg. Accuracy: {round(tmp_accs/tmp_accs_count, 2)} %")
        accs += tmp_accs
        accs_count += tmp_accs_count
        tmp_accs = 0
        tmp_accs_count = 0
        
    if accs_count > 0:
        print(f"Avg. Accuracy: {round(accs/accs_count, 2)} %")
    else:
        print("No prediction done!")

def new_machine_learning_timeset(data):
    # # # # # # # # # # # #
    # Es wird für die ersten 20 (*counts*) Einträge ein RF erstellt und der 21. Vorhergesagt
    # Der vorderste Eintrag wird gelöscht und der 21. hinzugefügt, es wird ein RF erstellt und der 22. Vorhergesagt
    # usw.
    # # # # # # # # # # # #
    counts = 30
    learning_data = pd.DataFrame()
    rf = None
    accs = 0
    accs_count = 0
    
    for index, row in data.iterrows():
        # if the learning data matches the choosen amount, create a RF and make a prediction for the next day
        if len(learning_data) >= counts:
            features = learning_data.loc[:, learning_data.columns != 'Result']
            features = features.loc[:, features.columns != 'Date']
            labels = learning_data.loc[:,'Result']
                
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(features, labels)
            
            test_feat = data.iloc[[index]]
            test_feat = test_feat.loc[:,test_feat.columns != 'Date']
            prediction = rf.predict(test_feat.loc[:, test_feat.columns != 'Result'])
            
            # accuracy = metrics.accuracy_score(row["Result"], prediction) * 100
            # print('Accuracy:', round(accuracy,4), '%')
            # accs += accuracy
            if row["Result"] == prediction[0]:
                accs += 1
            accs_count += 1
            # remove the first entry of the learning datas
            learning_data = learning_data.iloc[1: , :]
        

        # add row to learning data
        learning_data = pd.concat([learning_data, data.iloc[[index]]], ignore_index=True)
    
    if accs_count > 0:
        print(f"Avg. Accuracy: {round(accs/accs_count, 2)*100} %")
    else:
        print("No prediction done!")  

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