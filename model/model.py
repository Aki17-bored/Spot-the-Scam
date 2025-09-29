from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda import preprocessing



class Model:
    def __init__(self,text,target):
        super().__init__()
        self.text = text
        self.target = target
        _,self.vectorizer = preprocessing.vectorizer()
        self.text = np.array(self.text)
        self.target = np.array(self.target)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.text,self.target,random_state=42,shuffle=True)
        self.logistic_regression = LogisticRegression(max_iter=1000,random_state=0,class_weight='balanced')
        self.logistic_regression.fit(self.X_train,self.Y_train)

        self.prediction = self.logistic_regression.predict(self.X_test)

        self.accuracy = accuracy_score(self.Y_test,self.prediction)
        self.precision = precision_score(self.Y_test,self.prediction)
        self.recall = recall_score(self.Y_test,self.prediction)

    def Accuracy(self):
        return self.accuracy

    def F1_score(self):
        return  2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    def save_model(self, filename='fake_job_model.pkl'):
        import joblib
        joblib.dump(self.logistic_regression, filename)
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
        print(f"Model saved to {filename}")

if __name__ == '__main__':
    text,_ = preprocessing.vectorizer()
    target =preprocessing.df['fraudulent']
    model = Model(text=text,target=target)
    print('Precision Score : ', model.precision )
    print('Recall Score : ', model.recall)
    print('Model Accuracy : ', model.accuracy)
    print('F1 score : ', model.F1_score())
    model.save_model()