# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:14:50 2023

@author: ala.blidi
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks as pic

doc_path_phone='../csv_data_builder/calls_diabolocom.csv'
df_phone=pd.read_csv(doc_path_phone, sep=';')
df_phone.rename(columns={'call_start' : 'Consideration Start'}, inplace=True)
df_phone['Consideration Start'] = pd.to_datetime(df_phone['Consideration Start'], format="%d/%m/%Y %H:%M:%S")

### c'est inutile de le faire ici puisqu'après les groupby il faut le refaire ou drop 
### ou drop avant ; sauf le cas où on veut faire un hist, mais il suffit de refaire ça avant

# df_phone=df_phone.assign(Year=df_phone['Consideration Start'].dt.isocalendar()['year'], 
#              Week=df_phone['Consideration Start'].dt.isocalendar()['week'], 
#              Day=df_phone['Consideration Start'].dt.isocalendar()['day'],
#              Hour=df_phone['Consideration Start'].dt.hour,
#              Minute=df_phone['Consideration Start'].dt.minute)
#df_phone.loc[:,['Year','Week','Day','Hour','Minute']] = df_phone.loc[:,['Year','Week','Day','Hour','Minute']].astype(int)



##packages des vacance et jours fériés
import datetime
from vacances_scolaires_france import SchoolHolidayDates
d = SchoolHolidayDates()

from workalendar.europe import France
cal = France()

def df_agg(df_,freq, y_name, key) : #return à vérifier avant usage
    ##freq est la fréquence à laquelle on agreg : 1H,30T,2H
    df=df_.copy()
    # Agréger les vases par jour et compter le nombre de cases pour chaque jour    
    df = df.groupby(pd.Grouper(key=key, freq=freq)).size().reset_index(name=y_name)
            
    # Ajout des features temporelles détaillées
    df=df.assign(year=df[key].dt.isocalendar()['year'], 
                 week=df[key].dt.isocalendar()['week'],
                 day=df[key].dt.isocalendar()['day'],
                 #quarter=df['Consideration Start'].dt.quarter,
                 month=df[key].dt.month,
                 day_mon=df[key].dt.day,
                 hour=df[key].dt.hour*100+df[key].dt.minute)

    # Éjection des H off-office lorsque l'on est freq=min
    df.drop(index=df.loc[(df['hour']<800) | (df['hour']>=1800)].index, inplace=True)

    # Conversion de la colonne de date en index
    df.set_index(key, inplace=True)

    # Éjection des samedis et dimanches lorsque l'on est en phone only
    df.drop(index=df.loc[(df['day']==6) | (df['day']==7)].index, inplace=True)

    # Conversion dans le type adéquat pour LGBM
    df = df.astype(int)

    # Rajoute les vacances, jours fériés et d-1 fériés
    df=df.assign(vac_A=df.apply(lambda x : d.is_holiday_for_zone(datetime.date(x['year'],x['month'],x['day_mon']),'A'),axis=1),
                 vac_B=df.apply(lambda x : d.is_holiday_for_zone(datetime.date(x['year'],x['month'],x['day_mon']),'B'),axis=1),
                 vac_C=df.apply(lambda x : d.is_holiday_for_zone(datetime.date(x['year'],x['month'],x['day_mon']),'C'),axis=1),
                 ouvre=df.apply(lambda x : cal.is_working_day(datetime.date(x['year'],x['month'],x['day_mon'])),axis=1),
                 begin=df.apply(lambda x : 800<=x['hour']<900, axis=1),
                 lunch=df.apply(lambda x : 1200<=x['hour']<1300, axis=1),
                 end=df.apply(lambda x : 1700<=x['hour']<1800, axis=1)
                 )


    df['d-1_fer']=((df['ouvre'].shift(1,fill_value=True)-1)*-1).apply(bool)
    df.loc[(df['day']==1) & (df['d-1_fer'])==True,'d-1_fer']=False 

    # Éjection des samedis et dimanches lorsque l'on est en phone only ainsi que les jours fériés
    df.drop(index=df.loc[df.ouvre==False].index, inplace=True)
    df.drop(columns='ouvre', inplace=True)
    return df

def train_test_split(df, train_beg, train_end, test_beg, test_end, y_name) :
    #Données train et test
    train=df.loc[train_beg : train_end]
    test=df.loc[test_beg : test_end]

    # Variables d'entrée
    X_train = train.drop([y_name], axis=1)
    X_test = test.drop([y_name], axis=1)

    # Variable de sortie
    y_train = train[y_name]
    y_test = test[y_name]
    return X_train, X_test, y_train, y_test

## fonctions rendant des tableaux d'indicateurs de performances

from sklearn.metrics import mean_absolute_error as mae, r2_score as r2_s, mean_squared_error as mse, mean_squared_log_error as msle

def tabl_err(list_model, X_test, y_test) : ###le but est de minimiser la valeur présente dans chaque colonne, c'est pour ça que l'on prend >20 d'ailleurs
    tabl=pd.DataFrame()
    for model in list_model :
        y_pred=model.predict(X_test).astype(int)
        j_trav=y_test.shape[0]
        ##les indicateurs manuels
        abs_err_perc=np.abs((y_pred - y_test) / y_test) * 100
        mape=round(np.ma.masked_invalid(abs_err_perc).mean(),2)
        
        err_20=abs_err_perc[abs_err_perc>20].shape[0]
        err_20=round(err_20*100/j_trav,2)
        
        # sqr_err_perc=np.square((y_pred - y_test) / y_test) * 100
        # rmspe=round(np.ma.masked_invalid(sqr_err_perc).sum()/(y_pred.max,2)
        ##ajout dans la table
        tabl.loc[model,"MAE"]=round(mae(y_test,y_pred),2)
        tabl.loc[model,"MAPE"]=mape
        tabl.loc[model,"RMSE"]=round(mse(y_test,y_pred,squared=False),2)
        tabl.loc[model,"RMSPE"]=round(mse(y_test,y_pred,squared=False)/(y_pred.mean())*100,2)
        tabl.loc[model,"R²_score"]=round(r2_s(y_test,y_pred),2)
        tabl.loc[model,"Perc_with_err_>20%"]=err_20

    ###ajouter un indicateur qui mesure si le modèle a prédit le pic au bon moment i.e si pic dans la pred est le même que dans vrai/test
    return tabl

def tabl_err_all(list_model, X_train, y_train, X_test, y_test) : #idem qu'en haut mais en distinguant selon Data du train et du test
    list_ind=[['Train', 'Test'], list_model]
    index=pd.MultiIndex.from_product(list_ind, names=["Data", "Model"])
    tabl=pd.DataFrame(index=index)
    
    for Data in ["Train", "Test"] :
        X_Data = X_train if Data=="Train" else X_test
        y_Data = y_train if Data=="Train" else y_test
        for model in list_model :
            y_pred=model.predict(X_Data).astype(int)
            j_trav=y_Data.shape[0]
            ##les indicateurs manuels
            abs_err_perc=np.abs((y_pred - y_Data) / y_Data) * 100
            mape=round(np.ma.masked_invalid(abs_err_perc).mean(),2)

            err_20=abs_err_perc[abs_err_perc>20].shape[0]
            err_20=round(err_20*100/j_trav,2)

            # sqr_err_perc=np.square((y_pred - y_Data) / y_Data) * 100
            # rmspe=round(np.ma.masked_invalid(sqr_err_perc).sum()/(y_pred.max,2)
            ##ajout dans la table
            tabl.loc[(Data, model),"MAE"]=round(mae(y_Data,y_pred),2)
            tabl.loc[(Data, model),"MAPE"]=mape
            tabl.loc[(Data, model),"RMSE"]=round(mse(y_Data,y_pred,squared=False),2)
            tabl.loc[(Data, model),"RMSPE"]=round(mse(y_Data,y_pred,squared=False)/(y_pred.mean())*100,2)
            tabl.loc[(Data, model),"R²_score"]=round(r2_s(y_Data,y_pred),2)
            tabl.loc[(Data, model),"Perc_with_err_>20%"]=err_20
            
    return tabl

def predict_test(model, test, X_test, y_test):
    preds_test = model.predict(X_test)
    test["actual"] = y_test
    test["forecast"] = preds_test
    test["forecast"] = test["forecast"].astype(int)
    test.loc[test["forecast"] < 0, "forecast"] = 0
    test["absolute_error"] = np.abs(test["forecast"] - test["actual"])
    test["absolute_error_%"] = (
        np.abs((test["forecast"] - test["actual"]) / test["actual"]) * 100
    )
    return test


### Hyperopt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

mae_scorer=make_scorer(mae, greater_is_better=True)
##### c'est quoi score : ce que renvoie cross_val_score et c'est quoi son rôle dans fmin de hyperopt ? 
##### le laisser ou le remplacer par score renvoyé par rmse mape etc ?