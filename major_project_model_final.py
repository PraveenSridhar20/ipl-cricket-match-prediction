#!/usr/bin/env python
# coding: utf-8

# # ML Major project group 11
# 
# The task at hand is to come up with a machine learning model to predict the outcome of an IPL cricket match given some statistics such as batting averages, bowling averages, run rates, etc about the team. Data was web scraped from https://www.iplt20.com/stats/all-time for individual player statistics. Season wise match data was collected from kaggle. Since the percentage of matches which get tied are very small, they have not been considered for this model. Also, due the fact that a lot of teams changed dramatically in 2016, 17, these years were not included. Years 2008-12 were also not included because they were too far back to possibly make a meaningful prediction today. 2014-20 were used to train the model and 2013 was used as the test set. The reason for picking a completely different year as the test set was because this would be a good way to check how well the model generalizes to new teams and slightly different distrubutions because, in all probability, this model would be used to predict match outcomes for the following years. 

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import random
random.seed(20)


# All csv files are read from the folder and then loaded into different data frames. They are all concatenated into a single data frame later 

# In[2]:


df14=pd.read_csv("final14.csv",header=0)
df15=pd.read_csv("final15.csv",header=0)
df18=pd.read_csv("final18.csv",header=0)
df19=pd.read_csv("final19.csv",header=0)
df20=pd.read_csv("final20.csv",header=0)


# In[3]:


df_matches=pd.concat([df14,df15,df18,df19,df20])


# In[4]:


df_matches.info()


# <br>The model we are trying to build is provided with the the following features. 
# * The batting averages and strike rates of all 11 batsmen of team 1 and 2
# * The bowling averages and economies of 7 bowlers of team 1 and 2
# * The win and lost count of team 1 and team 2
# * The team which won the toss
# * The toss decision (field or bat)
# * The net run rate for team 1 and 2
# 
# 
# The model does not attempt to learn data based on how many matches a specific team has played / won against another specific team. Rather, it attempts to learn a set of features each team must posses to win or lose.
# 
# ***Therefore, this is being handled as a binary classification problem***
# 
# Venue has not been included as a feature because a lot of the matches were out of the country where neither of the teams had the home advantage

# In[5]:


df_matches_model=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)


# In[6]:


df_matches_model.head()


# In[7]:


corr=np.triu(df_matches_model.corr())


f,ax=plt.subplots(figsize=(110,90))

sns.heatmap(df_matches_model.corr(),annot=True,vmin=-1,vmax=1,center=0,mask=corr)
plt.show()


# <br>For this problem, 5 algorithms have been chosen after a lot of consideration about the nature of the problem and the data. They are:
# * Gaussian Naive Bayes
# * Random Forest
# * Simple Logistic Regression 
# * SVM with an RBF Kernel 
# * XGBoost
# 
# <br>Additionally, PCA was used in combination with all possible model structures as well to hopefully improve performance<br>
# 
# All of the 5 listed algorithms are run on all chosen model structures. All of the structures were chosen based on some form of prior knowledge about the game and the relative importance of each player. ***5 fold cross validation was used with accuracy as the metric***. Other metrics such as roc auc were not considered because they are meant for classes which deal with positive/negative classes. While this is being handled as a binary classification problem, the initial proposition was to handle this as multi class for which the other metrics would not work so well. Besides, as this is not a skewed class problem, accuracy with k fold cross validation gave a good idea of how well the model was generalising to new data sets as well
# 
# Note: Comparison of accuracy across multiple cells is done relative to the best model structure. In some cells (most of the cells represent a model structure each. Exceptions have been specified if not self evident), certain models like logistic regression might see a 10% jump in accuracy. However, it often turned out that that one of the 4 algorithms showing a significant improvement still did not do better than the best algorithm from the previous model structure. So, for one model structure (feature selection/feature scaling) to be better than another, it is not necessary that all algorithms should do better than the other model. 
# Therefore, the following comparison metric was used to compare 2 model structures. 
# 
# ***For model structure A to be called superior to model structure B, there must exist at least one algorithm that ran on model structure A that outperformed all algorithms on model structure B, i.e.***
# 
# ***
# $structA$ $>$ $structB$ iff  $\exists$  $i$   $\ni$    $algorithm_i(structA)$  > max ($algorithm_j(structB)$) $\forall$ $j$
# ***

# <br>***The first structure is the raw X and y data***. We run all 5 algorithms Vanilla and Naive Bayes outperforms all algorithms.

# In[8]:


#vanilla

X=df_matches_model.drop(['winner'],axis=1)
y=df_matches_model['winner']
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# <br><br>The following cells were run based on the results of the correlation matrix. Highly correlated features were removed with the hope of improving the accuracy. This slightly improved the accuracy for Naive Bayes but better approaches were found and have been described later. Furthermore, the correlation of the features win and lost was -0.97. But removing the lost feature dropped the accuracy of the model on average by atleast 3-4 % and therefore, it was evident that ***removing correlated features was not the way go***

# In[9]:


X=df_matches_model.drop(['winner','avg51','sr11','boavg31','boavg51','boavg32','avg112','avg52','boavg52'],axis=1)
y=df_matches_model['winner']

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=7000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
#correlation 1


# In[10]:


X=df_matches_model.drop(['winner','avg51','boavg51','avg52','boavg52'],axis=1)
y=df_matches_model['winner']

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
#attempt to remove based on correlation. Reduction in performance 


# <br> The following 2 cells run all 4 algorithms with feature scaling (Standard Scaler and  MinMaxScaler on X). ***Running feature scaling on the data without modifying the model structure did not improve the accuracy substantially***

# In[11]:


#standard scaler
print('Standard Scaler')
X=df_matches_model.drop(['winner'],axis=1)
scaler=StandardScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# In[12]:


print('\nMinMaxScaler')
X=df_matches_model.drop(['winner'],axis=1)
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# After having tried quite a few models on the raw data, it became clear that ***feature selection was necessary to further improve model accuracy***. This is what has been attempted in the following cells
# 
# The first cell runs all the 5 algorithms on all possible structures (77). The second also runs PCA before running the 5 algorithms.

# In[13]:


maxAccNb=int(0)
mbaNb=int(0)
mbbNb=int(0)

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        gnb=GaussianNB()
        gnb.fit(X,y)
        acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
        print(np.mean(acc_nb))
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
        if (maxAccNb<np.mean(acc_nb)):
            maxAccNb=np.mean(acc_nb)
            mbaNb=batcount
            mbbNb=bowlcount
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best NB:",maxAccNb,' ',mbaNb,' ',mbbNb)
print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# In[14]:


print("With PCA")
maxAccNb=int(0)
mbaNb=int(0)
mbbNb=int(0)

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        pca=PCA()
        X=pca.fit_transform(X)

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        gnb=GaussianNB()
        gnb.fit(X,y)
        acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
        print(np.mean(acc_nb))
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
        if (maxAccNb<np.mean(acc_nb)):
            maxAccNb=np.mean(acc_nb)
            mbaNb=batcount
            mbbNb=bowlcount
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best NB+PCA:",maxAccNb,' ',mbaNb,' ',mbbNb)
print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# The above cells picked a very small number of features. This could possibly be because picking a larger number of features resulted in overfitting which performed poorly on cross validation. However, based on prior knowledge of the game, it is unrealistic to predict the outcome of a cricket match based on just the top 2 or 3 batsmen or bowlers. It is possible that these models also overfit the cross validation set and while they ended up performing better on average, they cannot be expected to generalize well to the test set

# <br>Given the fact that scaling seemed to improve model accuracy, ***the following cell run all the algorithms except Naive Bayes with the StandardScaler. This is because Naive Bayes is not affected by scaling. The cell after that does the same thing but with a MinMaxScaler***

# In[15]:


#standard scaler
maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        scaler=StandardScaler()
        X=scaler.fit_transform(X)

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# In[16]:


#minmaxscaler
maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# ***The following 2 cells run the all the algorithms except Naive bayes with Standard and MinMaxScaling. But with PCA this time***

# In[17]:


print("Standard Scaler with PCA")

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        pca=PCA()
        X=pca.fit_transform(X)

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        

            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# In[18]:


print("MinMax Scaler with PCA")

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXBB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        team=1


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)
        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)


        
        X=df_matches_model[list0]
        y=df_matches_model['winner']
        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        pca=PCA()
        X=pca.fit_transform(X)

        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        

            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)


# We have the same issue as we did without scaling. The models which perform the best in both of the above cells pick too small a number of features to be expected to generalize well to the test set. The SVM seemed to pick a larger number of features with 7 batsmen but ended up picking only one bowler. Despite having a lower number of features, some of these structures have been run on the test set in a later section of this notebook. However, it became quite evident that we needed a way to reduce the number of features without reducing the information about the data.

# <br>The following cell takes into account prior knowledge about the game in the hopes of selecting some features better.The prior knowledge can be summarized as follows:
# 
# ***Lower order batsmen come in the death overs and therefore a higher strike rate would be expected of them. And a better batting average would be expected of the batsmen before. Further, if other bowlers keep a low economy, then strike bowlers have more chances to get a wicket***
# 
# Therefore the following features were selected:
# * Batting averages of batsmen 1-3 for both teams 
# * Strike rates of batsmen 4-7 for both teams
# * Bowling averages for bowlers 1-2 for both teams
# * Bowling economies for bowlers 3-5 for both teams 
# * Toss winner
# * Toss result 
# * Win count of both teams 
# * Loss count of both teams 
# * Net run rate of both teams
# 
# The same protocol is followed. Run all algorithms raw, with PCA, with scaling, with scaling and PCA

# In[19]:


X=df_matches_model.drop(['winner','avg41','avg51','avg61','avg71','avg81','avg91','avg101','avg111','sr11','sr21','sr31','sr81','sr91','sr101','sr111','boavg31','boavg41','boavg51','boavg61','boavg71','econ11','econ21','avg42','avg52','avg62','avg72','avg82','avg92','avg102','avg112','sr12','sr22','sr32','sr82','sr91','sr102','sr112','boavg32','boavg42','boavg52','boavg62','boavg72','econ12','econ22'],axis=1)
y=df_matches_model['winner']
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
#prior knowledge data 


# <br>***The following 2 cells run the above structure with both forms of scaling and PCA.*** This did not offer a substantial improvement either.

# In[20]:


#standard scaler
print('Standard Scaler')
X=df_matches_model.drop(['winner','avg41','avg51','avg61','avg71','avg81','avg91','avg101','avg111','sr11','sr21','sr31','sr81','sr91','sr101','sr111','boavg31','boavg41','boavg51','boavg61','boavg71','econ11','econ21','avg42','avg52','avg62','avg72','avg82','avg92','avg102','avg112','sr12','sr22','sr32','sr82','sr91','sr102','sr112','boavg32','boavg42','boavg52','boavg62','boavg72','econ12','econ22'],axis=1)
scaler=StandardScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# In[21]:


#minmax scaler
print('MinMax Scaler')
X=df_matches_model.drop(['winner','avg41','avg51','avg61','avg71','avg81','avg91','avg101','avg111','sr11','sr21','sr31','sr81','sr91','sr101','sr111','boavg31','boavg41','boavg51','boavg61','boavg71','econ11','econ21','avg42','avg52','avg62','avg72','avg82','avg92','avg102','avg112','sr12','sr22','sr32','sr82','sr91','sr102','sr112','boavg32','boavg42','boavg52','boavg62','boavg72','econ12','econ22'],axis=1)
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# One way of retaining the information about the data while also reducing the number of features was to take the averaged team statistics.
# <br>***From this cell forward, individual player features shall not be used. Rather, an average of all the features shall be used.***
# Namely:
# * Average of all batsmen batting averages
# * Average of all batsmen strike rates
# * Average of all bowler bowling averages
# * Average of all bowler economies 
# 
# 
# This was done with the hope of further improving model accuracy by reducing the number of features (thereby preventing overfitting) while also retaining as much information as possible
# 

# In[22]:


pd.set_option('mode.chained_assignment', None)
team=0
#match_count is the number of matches 
list0=[]
for i in range(11):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg1=df_matches_model[list0]
list0=[]


for i in range(11):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr1=df_matches_model[list0]
list0=[]

for i in range(7):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo1=df_matches_model[list0]
list0=[]

for i in range(7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(11):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg2=df_matches_model[list0]
list0=[]


for i in range(11):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr2=df_matches_model[list0]
list0=[]

for i in range(7):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo2=df_matches_model[list0]
list0=[]

for i in range(7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco2=df_matches_model[list0]
dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)
pd.set_option('mode.chained_assignment', 'warn')


# In[23]:


df_mat_avg.info()


# <br> ***The following cells run all of the aforementioned algorithms on the averages of all the features used till this point.***

# In[24]:


X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# <br> ***The following 2 cells run the algorithm on the data with feature scaling now in place.*** 

# In[25]:


#standard scaler
print('Standard Scaler')
X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]

scaler=StandardScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# In[26]:


#minmaxscaler scaler
print('MinMaxScaler')
X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))
print('With pca')
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))


# <br>As seen above, using the average of all the features did not really improve model performance substantially (by substantially we mean did not outperform the best model till now by a significant enough margin) irrespective of scaling. So in order to identify the best combination of averages that we might want to take, ***the following 2 cells repeat the structure search problem this time using the averages statistics and all algorithms.*** The first cell does not use PCA and the next cell does

# In[27]:


pd.set_option('mode.chained_assignment', None)
maxAccNb=int(0)
mbaNb=int(0)
mbbNb=int(0)

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']


        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        gnb=GaussianNB()
        gnb.fit(X,y)
        acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
        print(np.mean(acc_nb))
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
        if (maxAccNb<np.mean(acc_nb)):
            maxAccNb=np.mean(acc_nb)
            mbaNb=batcount
            mbbNb=bowlcount
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best NB:",maxAccNb,' ',mbaNb,' ',mbbNb)
print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# In[28]:


pd.set_option('mode.chained_assignment', None)
#pca
print("With PCA")
maxAccNb=int(0)
mbaNb=int(0)
mbbNb=int(0)

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']


        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        pca=PCA()
        X=pca.fit_transform(X)
        gnb=GaussianNB()
        gnb.fit(X,y)
        acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
        print(np.mean(acc_nb))
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        
        if (maxAccNb<np.mean(acc_nb)):
            maxAccNb=np.mean(acc_nb)
            mbaNb=batcount
            mbbNb=bowlcount
            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best NB+PCA:",maxAccNb,' ',mbaNb,' ',mbbNb)
print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# ***The following 2 cells run 4 algorithms (no NB) with standard scaling. First without PCA and then with PCA***

# In[29]:


pd.set_option('mode.chained_assignment', None)
#standard scaler
maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']

        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
 
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        

            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# In[30]:


pd.set_option('mode.chained_assignment', None)
#pca standard scaler
print("With PCA")

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']

        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        pca=PCA()
        X=pca.fit_transform(X)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')

        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# ***The following 2 cells run 4 algorithms (no NB) with MinMax scaling. First without PCA and then with PCA***

# In[31]:


pd.set_option('mode.chained_assignment', None)
#standard scaler
maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']

        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
 
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')
        

            
        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    
print ("best RF:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# In[32]:


pd.set_option('mode.chained_assignment', None)
#pca standard scaler
print("With PCA")

maxAccRf=int(0)
mbaRf=int(0)
mbbRf=int(0)

maxAccLr=int(0)
mbaLr=int(0)
mbbLr=int(0)

maxAccSVM=int(0)
mbaSVM=int(0)
mbbSVM=int(0)

maxAccXGB=int(0)
mbaXGB=int(0)
mbbXGB=int(0)

for batcount in range(1,12):
    for bowlcount in range(1,8):
        
        team=0
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg1=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo1=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco1=df_matches_model[list0]



        team=1
        #match_count is the number of matches 
        list0=[]
        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('avg'+x)

        dfavg2=df_matches_model[list0]
        list0=[]


        for i in range(batcount):
            x=str(i+1)+str(team+1)
            list0.append('sr'+x)

        dfsr2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('boavg'+x)

        dfbo2=df_matches_model[list0]
        list0=[]

        for i in range(bowlcount):
            x=str(i+1)+str(team+1)
            list0.append('econ'+x)

        dfeco2=df_matches_model[list0]

        dfavg1['avg1']=dfavg1.mean(axis=1)
        dfsr1['sr1']=dfsr1.mean(axis=1)
        dfeco1['eco1']=dfeco1.mean(axis=1)
        dfbo1['boavg1']=dfbo1.mean(axis=1)

        dfavg2['avg2']=dfavg2.mean(axis=1)
        dfsr2['sr2']=dfsr2.mean(axis=1)
        dfeco2['eco2']=dfeco2.mean(axis=1)
        dfbo2['boavg2']=dfbo2.mean(axis=1)

        df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

        df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

        X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
        y=df_mat_avg['winner']

        scaler=MinMaxScaler()
        X=scaler.fit_transform(X)
        print("Batsman bowler count:",' ',batcount,' ',bowlcount)
        pca=PCA()
        X=pca.fit_transform(X)
        rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
        rf.fit(X,y)
        acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
        print(np.mean(acc_rf))
        lr=LogisticRegression(max_iter=8000,class_weight='balanced')
        lr.fit(X,y)
        acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
        print(np.mean(acc_lr))
        svc=SVC()
        svc.fit(X,y)
        acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
        print(np.mean(acc_svc))
        xgb=XGBClassifier(n_jobs=-1)
        xgb.fit(X,y)
        acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
        print(np.mean(acc_xgb),'\n')

        if (maxAccRf<np.mean(acc_rf)):
            maxAccRf=np.mean(acc_rf)
            mbaRf=batcount
            mbbRf=bowlcount
            
        if (maxAccLr<np.mean(acc_lr)):
            maxAccLr=np.mean(acc_lr)
            mbaLr=batcount
            mbbLr=bowlcount
            
        if (maxAccSVM<np.mean(acc_svc)):
            maxAccSVM=np.mean(acc_svc)
            mbaSVM=batcount
            mbbSVM=bowlcount
            
        if (maxAccXGB<np.mean(acc_xgb)):
            maxAccXGB=np.mean(acc_xgb)
            mbaXGB=batcount
            mbbXGB=bowlcount
        #print(batcount,' ',bowlcount,'\n')
    

print ("best RF+PCA:",maxAccRf,' ',mbaRf,' ',mbbRf)
print ("best LR+PCA:",maxAccLr,' ',mbaLr,' ',mbbLr)
print ("best SVM+PCA:",maxAccSVM,' ',mbaSVM,' ',mbbSVM)
print ("best XGB+PCA:",maxAccXGB,' ',mbaXGB,' ',mbbXGB)
pd.set_option('mode.chained_assignment', 'warn')


# ***Logistic regression with MinMaxScaling and with 8 batsmen and 5 bowlers features averaged significantly outperformed all models till now. This was regardless of whether or not PCA was used***

# <br>***The following cells are based on prior knowledge. First raw data. Then standard scaling and then minmax scaling.*** This degraded performance compared to simple logistic regression. 

# In[33]:


pd.set_option('mode.chained_assignment', None)
team=0
batcount=6
bowlcount=7
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg1=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr1=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo1=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg2=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr2=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo2=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

#features based on prior knowledge 
pd.set_option('mode.chained_assignment', 'warn')


# In[34]:


pd.set_option('mode.chained_assignment', None)
#standard scaler
print("Standard scaler")
team=0
batcount=6
bowlcount=7
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg1=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr1=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo1=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg2=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr2=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo2=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']
scaler=StandardScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

#features based on prior knowledge 
pd.set_option('mode.chained_assignment', 'warn')


# In[35]:


pd.set_option('mode.chained_assignment', None)
#MinMax scaler
print("MinMax scaler")
team=0
batcount=6
bowlcount=7
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg1=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr1=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo1=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(1,4):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
    
dfavg2=df_matches_model[list0]
list0=[]


for i in range(4,8):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
    
dfsr2=df_matches_model[list0]
list0=[]

for i in range(1,3):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
    
dfbo2=df_matches_model[list0]
list0=[]

for i in range(3,7):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)
    
dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

print('\nWith pca')
pca=PCA()
X=pca.fit_transform(X)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))

rf=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))

lr=LogisticRegression(max_iter=5000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))

svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))

xgb=XGBClassifier()
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb))

#features based on prior knowledge 
pd.set_option('mode.chained_assignment', 'warn')


# The following models were trained on the data:
# ***
# * NB,RF,LR,SVM,XGB on raw data (with and without PCA):***10***
# * NB,RF,LR,SVM,XGB on data after removing correlated features (with and without PCA):10+10=***20***
# * NB,RF,LR,SVM,XGB on data with Standard and MinMaxScaling (with and without PCA):10+10=***20***
# ***
# * NB,RF,LR,SVM,XGB on data over all possible 77 structures with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 5 x 2=***770***
# * RF,LR,SVM,XGB on data over all possible 77 structures with Standard scaling, with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 4 x 2=***616***
# * RF,LR,SVM,XGB on data over all possible 77 structures with MinMaxScaling scaling, with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 4 x 2=***616***
# ***
# * NB,RF,LR,SVM,XGB on prior knowledge data (with and without PCA):***10***
# * NB,RF,LR,SVM,XGB on prior knowledge data with Standard and MinMaxScaling (with and without PCA):10+10=***20***
# ***
# * NB,RF,LR,SVM,XGB on averaged data (with and without PCA):***10***
# * NB,RF,LR,SVM,XGB on averaged data with Standard and MinMaxScaling (with and without PCA):10+10=***20***
# ***
# * NB,RF,LR,SVM,XGB on averaged data over all possible 77 structures with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 5 x 2=***770***
# * RF,LR,SVM,XGB on averaged data over all possible 77 structures with Standard scaling, with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 4 x 2=***616***
# * RF,LR,SVM,XGB on averaged data over all possible 77 structures with MinMaxScaling scaling, with and without PCA (top x batsmen and y bowlers where x in range(1,12) and y in range(1,8): 77 x 4 x 2=***616***
# ***
# * NB,RF,LR,SVM,XGB on prior knowledge averaged data (with and without PCA):***10***
# * NB,RF,LR,SVM,XGB on prior knowledge averaged data with Standard and MinMaxScaling (with and without PCA):10+10=***20***
# *** 
# 
# After running more than 4100 models, Naive Bayes seemed to be the clear winner initially and in most cases, it dramatically outperformed other algorithms with a much better accuracy on quite a few structures. <br>However, towards the end, ***Logistic regression with MinMaxScaling gained a massive boost and achieved a prediction accuracy of 65.8% with 8 batsmen and 5 bowlers averaged/cumulative stats***. This lined up well with the prior knowledge of the game as well. 

# In[36]:


df_test=pd.read_csv("final13.csv",header=0)
df_test.head()


# In[37]:


df_test_model=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)


# Now we run some of the best models on the test set. Matches from the year 2013. However, as seen from the following 2 cells, ***it has been found that PCA no longer works because the features that is learns on this new data are not the same.*** So, one possible option alleviate this would be to shuffle the training and test data and apply pca on both of them as a whole. But this does not solve the problem on principle as if this is applied on a new data set (like 2021), it won't perform well. Compounding this is the fact that the best models worked just as well if not better without PCA. 

# In[38]:


pd.set_option('mode.chained_assignment', None)
#pca standard scaler
print("With PCA")

batcount=8
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)
pca=PCA()
X=pca.fit_transform(X)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')


pd.set_option('mode.chained_assignment', 'warn')


# In[39]:


pd.set_option('mode.chained_assignment', None)
#pca standard scaler
print("With PCA")

batcount=8
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_test_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_test_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)
pca=PCA()
X=pca.fit_transform(X)
y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))


pd.set_option('mode.chained_assignment', 'warn')


# ***From this point forward, PCA shall no longer be used*** 
# The following 7 cells run the best model structures (ie the structures on which at least one algorithm had an accuracy of greater than 65%. These structures for testing were selected in conjunction with prior knowledge of the game. So structures with a very small number of features were not considered and structures with cumulative statistics were preferred because they reduced the number of features and thereby reduced the risk of overfitting

# In[40]:


pd.set_option('mode.chained_assignment', None)

print ("Cumulative stats")

batcount=8
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=8
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_test_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_test_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))


pd.set_option('mode.chained_assignment', 'warn')


# In[41]:


print ("Player wise stats")
team=0
batcount=3
bowlcount=2
list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

team=1


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)



X=df_matches_model[list0]
y=df_matches_model['winner']


print("Batsman bowler count:",' ',batcount,' ',bowlcount)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=3
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

team=1


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

X=df_test_model[list0]
y=df_test_model['winner']
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))
    


# In[42]:


#standard scaling
print ("Standard scaling")
print ("Player wise stats")
team=0
batcount=7
bowlcount=2
list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

team=1


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)



X=df_matches_model[list0]
y=df_matches_model['winner']
scaler=StandardScaler()
X=scaler.fit_transform(X)

print("Batsman bowler count:",' ',batcount,' ',bowlcount)
gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=7
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2']
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

team=1


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)
for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

X=df_test_model[list0]
y=df_test_model['winner']
scaler=StandardScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))
    


# In[43]:


pd.set_option('mode.chained_assignment', None)

print ("Cumulative stats")
batcount=3
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

print("Batsman bowler count:",' ',batcount,' ',bowlcount)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=3
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_test_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_test_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']


print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))


pd.set_option('mode.chained_assignment', 'warn')


# In[44]:


pd.set_option('mode.chained_assignment', None)

print ("Cumulative stats")
batcount=5
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

print("Batsman bowler count:",' ',batcount,' ',bowlcount)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=5
bowlcount=2
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_test_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_test_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']


print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))


pd.set_option('mode.chained_assignment', 'warn')


# In[45]:


pd.set_option('mode.chained_assignment', None)

print ("Cumulative stats")
batcount=6
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_matches_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_matches_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_matches_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_matches_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_matches.drop(['Venue','Team 1','Team 2','res'],axis=1)

df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

gnb=GaussianNB()
gnb.fit(X,y)
acc_nb=cross_val_score(gnb,X,y,scoring='accuracy')
print(np.mean(acc_nb))
rf=RandomForestClassifier(class_weight='balanced',n_estimators=200,n_jobs=-1)
rf.fit(X,y)
acc_rf=cross_val_score(rf,X,y,scoring='accuracy')
print(np.mean(acc_rf))
lr=LogisticRegression(max_iter=8000,class_weight='balanced')
lr.fit(X,y)
acc_lr=cross_val_score(lr,X,y,scoring='accuracy')
print(np.mean(acc_lr))
svc=SVC()
svc.fit(X,y)
acc_svc=cross_val_score(svc,X,y,scoring='accuracy')
print(np.mean(acc_svc))
xgb=XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
acc_xgb=cross_val_score(xgb,X,y,scoring='accuracy')
print(np.mean(acc_xgb),'\n')

print("Test")
batcount=6
bowlcount=5
        
team=0
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg1=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo1=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco1=df_test_model[list0]



team=1
#match_count is the number of matches 
list0=[]
for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('avg'+x)

dfavg2=df_test_model[list0]
list0=[]


for i in range(batcount):
    x=str(i+1)+str(team+1)
    list0.append('sr'+x)

dfsr2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('boavg'+x)

dfbo2=df_test_model[list0]
list0=[]

for i in range(bowlcount):
    x=str(i+1)+str(team+1)
    list0.append('econ'+x)

dfeco2=df_test_model[list0]

dfavg1['avg1']=dfavg1.mean(axis=1)
dfsr1['sr1']=dfsr1.mean(axis=1)
dfeco1['eco1']=dfeco1.mean(axis=1)
dfbo1['boavg1']=dfbo1.mean(axis=1)

dfavg2['avg2']=dfavg2.mean(axis=1)
dfsr2['sr2']=dfsr2.mean(axis=1)
dfeco2['eco2']=dfeco2.mean(axis=1)
dfbo2['boavg2']=dfbo2.mean(axis=1)

df_mat_avg=df_test.drop(['Venue','Team 1','Team 2','res'],axis=1)
df_mat_avg=pd.concat([df_mat_avg,dfavg1['avg1'],dfsr1['sr1'],dfbo1['boavg1'],dfeco1['eco1'],dfavg2['avg2'],dfsr2['sr2'],dfbo2['boavg2'],dfeco2['eco2']],axis=1)

X=df_mat_avg[['Toss winner','toss result','won1','lost1','rr1','won2','lost2','rr2','avg1','sr1','boavg1','eco1','avg2','sr2','boavg2','eco2']]
y=df_mat_avg['winner']

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
print("Batsman bowler count:",' ',batcount,' ',bowlcount)

y_test_nb=gnb.predict(X)
print(accuracy_score(y,y_test_nb))
y_test_rf=rf.predict(X)
print(accuracy_score(y,y_test_rf))
y_test_lr=lr.predict(X)
print(accuracy_score(y,y_test_lr))
y_test_svm=svc.predict(X)
print(accuracy_score(y,y_test_svm))
y_test_xgb=xgb.predict(X)
print(accuracy_score(y,y_test_xgb))
pd.set_option('mode.chained_assignment', 'warn')


# Conclusions:
# 
# * From all of the above cells which ran the model on the test set, one thing became self evident. ***The models which used individual player stats were much more likely to overfit the data as the models which performed the best on cross validation performed much worse on the test set.*** There were some models which did not do well on cross validation but did great on the test set. However, it can be concluded that such models are overfitting the test set. 
# * ***Cumulative models fit the data much better simply by virtue of having lesser features.*** This helped majorly because the data set wasn't particularly large to begin with. However, even in this, some of the models seemed to great on the test set did not do as well on the cross validation. 
# * After a thorough analysis of all the models and their performances on the training, cross validation and test sets, the algorithm which performs the best is ***Logistic regression with MinMaxScaling. The features used for this are the cumulative statistics of the top 8 batsmen and the top 5 bowlers***. This model gave an accuracy of 73.33% on the test set and outperformed all algorithms on the training-cross validation set. One possible explanation as to why it could beat algorithms such as random forest could be the fact that logistic regression converges to a global optima. 
# * ***No algorithm which had near the same cross validation accuracy as Logistic regression with MinMax scaling did as well on the test set***. While some of the other algorithms performed better on the test set, it was not enough to outweigh the poor performance on the training cross validation set. This was because using 5 fold cross validation already checks how well the model generalizes on 5 different splits. 
# * Therefore, it has been concluded that ***Logistic Regression with MinMax scaling, cumulative features of 8 batsmen and 5 bowlers*** generalizes the best and predicts the correct winner in almost 3/4ths of instances. 
