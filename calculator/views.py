from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
#from rest_framework.response import Response
import json


## Load the required libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn import tree,ensemble
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score

# Create your views here.

def index(request):
    return render(request, "input.html")


def addition(request):
    model_name = request.GET['model_name']
    JobName = request.GET['jobname']
    num1 = request.GET['max_depth']
    num2 = request.GET['train_test_split']
    criterion = request.GET['criterion']

    if num1.isdigit() and num2.isdigit():
        a = int(num1)
        b = int(num2)
        b=b/100

        
        data=pd.read_csv(r"C:\Users\yugandhar.gantala\Documents\ML_and_AI_Labs\django_Projects\FirstProject\calculator\employee_data.csv")
        
        data=data.drop(["filed_complaint","recently_promoted"],axis=1)
        data['tenure']=data['tenure'].replace(np.NaN,0)
        data['department']=data['department'].replace(np.NaN,'sales')
        
        cat_cols=["department","salary","n_projects","tenure","status"]
        num_cols=["last_evaluation","satisfaction","avg_monthly_hrs"]
        data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))
        data[num_cols] = data[num_cols].apply(lambda x: x.astype('float'))
        
        
        ## Convert Categorical Columns to Dummies
        cat_cols=["department","salary","n_projects","tenure","status"]
        data = pd.get_dummies(data,columns=cat_cols,drop_first=True,)
        
        ## Split the data into X and y
        X = data.copy().drop("status_Left",axis=1)
        y = data["status_Left"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=b)
        
        X_train['satisfaction'].fillna(X_train['satisfaction'].mean(),inplace=True)
        X_train['last_evaluation'].fillna(X_train['last_evaluation'].mean(),inplace=True)
        
        X_test['satisfaction'].fillna(X_train['satisfaction'].mean(),inplace=True)
        X_test['last_evaluation'].fillna(X_train['last_evaluation'].mean(),inplace=True)
        
        scaler=StandardScaler()
        scaler.fit(X_train.iloc[:,:3])
        
        X_train.iloc[:,:3]=scaler.transform(X_train.iloc[:,:3])
        X_test.iloc[:,:3]=scaler.transform(X_test.iloc[:,:3])
        
        ####random forest
        
        rfc = RandomForestClassifier(criterion=criterion,
                       max_depth=a)
        rfc.fit(X = X_train,y = y_train)
        
        train_predictions = rfc.predict(X_train)
        test_predictions = rfc.predict(X_test)
        
        Train_Score=accuracy_score(y_train,train_predictions)
        Test_Score=accuracy_score(y_test,test_predictions)
    
        predictions=pd.DataFrame(test_predictions)
        predictions['JobID']=JobName
        predictions1=predictions.to_json(orient='records')
        #predictions1 = pd.DataFrame({'bla':[1,2,3],'bla2':['a','b','c']}).to_json(orient='records')
        
        return JsonResponse(json.loads(predictions1), safe = False)

        #return predictions1
    else:
        res = "Only digits are allowed"
        return render(request, "result.html", {"result": res})
