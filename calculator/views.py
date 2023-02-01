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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Create your views here.

def index(request):
    return render(request, "input.html")


def addition(request):
    model_name = request.GET['model_name']
    JobName = request.GET['jobname']
    num1 = request.GET['max_depth']
    num2 = request.GET['train_test_split']
    criterion = request.GET['criterion']

    if num1.isdigit():
        a = int(num1)
        b = float(num2)

#         data=pd.read_csv(r"calculator/employee_data.csv")
        data=pd.read_csv(r"calculator/train.csv")
        test_df=pd.read_csv(r"calculator/test.csv")
        test_df_copy=test_df.copy()

        drop_list = ["Cabin","Name","Ticket"]
        data = data.drop(drop_list, axis=1)
        test_df = test_df.drop(drop_list, axis=1)
        # data = data.drop(["PassengerId"], axis=1)

#         cat_cols=["Survived","Pclass","Sex","SibSp","Embarked","Parch"]
#         num_cols=["PassengerId","Age","Fare"]
#         data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))
#         data[num_cols] = data[num_cols].apply(lambda x: x.astype('float'))

#         test_cat_cols=["Pclass","Sex","SibSp","Embarked","Parch"]

#         test_df[test_cat_cols] = test_df[test_cat_cols].apply(lambda x: x.astype('category'))
#         test_df[num_cols] = test_df[num_cols].apply(lambda x: x.astype('float'))

        test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

        sexReplacer = lambda x:x.map({'male':0,'female':1})
        data['Sex'] = data[['Sex']].apply(sexReplacer)
        data["Age"] = data["Age"].fillna(data["Age"].mean())
        embarkedReplacer = lambda x:x.map({'S':0,'C':1,'Q':2})
        data['Embarked'] = data[['Embarked']].apply(embarkedReplacer)
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mean())

        sexReplacer = lambda x:x.map({'male':0,'female':1})
        test_df['Sex'] = test_df[['Sex']].apply(sexReplacer)
        test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
        embarkedReplacer = lambda x:x.map({'S':0,'C':1,'Q':2})
        test_df['Embarked'] = test_df[['Embarked']].apply(embarkedReplacer)
        test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mean())

        ## Split the data into X and y
        X = data.copy().drop("Survived",axis=1)
        y = data["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=b)

        rfc = RandomForestClassifier(criterion=criterion,
                       max_depth=a)
        rfc.fit(X = X_train,y = y_train)

        train_predictions = rfc.predict(X_train)
        test_predictions = rfc.predict(X_test)
        predictions = rfc.predict(test_df)

        Train_Score=accuracy_score(y_train,train_predictions)
        Test_accuracy=accuracy_score(y_test,test_predictions)
        Test_f1_score=f1_score(y_test,test_predictions)
        Test_precision_score=precision_score(y_test,test_predictions)
        Test_recall_score=recall_score(y_test,test_predictions)

        predictions=pd.DataFrame(predictions)
        predictions.rename(columns={0:'Values'},inplace=True)
        predictions['PassengerId']=test_df['PassengerId']
        predictions=pd.merge(predictions,test_df_copy,on='PassengerId',how='Left')
        
     
        predictions['Parameter']='Predictions'

        predictions=predictions.append(pd.DataFrame({'Values':[Test_accuracy,Test_f1_score,Test_precision_score,Test_recall_score], 'Parameter':['Accuracy', 'f1_score', 'precision_score', 'recall_score']})).reset_index(drop=True)
        predictions['JobID']=JobName
        predictions1=predictions.to_json(orient='records')
        return JsonResponse(json.loads(predictions1), safe = False)

        #return predictions1
    else:
        res = "Only digits are allowed"
        return render(request, "result.html", {"result": res})
