#!/usr/bin/env python
# coding: utf-8

# In[36]:


get_ipython().system('pip install selenium')


# In[37]:


import selenium
import pandas as pd
from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')
from selenium.webdriver.common.by import By
import requests
import time
from selenium.common.exceptions import NoSuchElementException


# In[38]:


# Connect to webdriver
driver = webdriver.Chrome(r'C:\Users\COTMAC\Downloads\chromedriver_win32\chromedriver.exe')


# In[39]:


driver.get("https://www.makemytrip.com/")
time.sleep(2)


# In[41]:


search=driver.find_element(By.XPATH,"/html/body/div[1]/div/div[2]/div/div[1]/div[3]/p/a")
search.click()


# In[42]:


FlightName=[]
SourceName=[]
DestinationName=[]
DepartureTime=[]
ArrivalTime=[]
TotalStops=[]
Duration=[]


# In[43]:


# 5000 time we scroll down by 50000 in order to generate more Comments
for _ in range(2000):
    driver.execute_script("window.scrollBy(0,2000)")
    
time.sleep(5)


# In[46]:


try:   
    FlightName1=driver.find_elements(By.XPATH,"//p[@class='boldFont blackText airlineName']")
    for i in FlightName1[0:2001]:
        FlightName.append(i.text.split('\n')[0])
except NoSuchElementException:
        FlightName.append('-') 
        
time.sleep(500)


# In[47]:


len(FlightName)


# In[49]:


try:
    DepartureTime1=driver.find_elements(By.XPATH,"//p[@class='appendBottom2 flightTimeInfo']")
    for i in DepartureTime1[0:2001]:
        DepartureTime.append(i.text.split('\n')[0])
except NoSuchElementException:
        DepartureTime.append('-') 
        
time.sleep(500)


# In[50]:


len(DepartureTime)


# In[51]:


try:
    SourceName1=driver.find_elements(By.XPATH,"//p[@class='blackText']")
    for i in SourceName1[0:2001]:
        SourceName.append(i.text.split('\n')[0])
except NoSuchElementException:
        SourceName.append('-') 
        
time.sleep(500)


# In[52]:


len(SourceName)


# In[53]:


try:
    TotalStops1=driver.find_elements(By.XPATH,"//p[@class='flightsLayoverInfo']")
    for i in TotalStops1[0:2001]:
        TotalStops.append(i.text.split('\n')[0])
except NoSuchElementException:
        TotalStops.append('-') 
        
time.sleep(500)


# In[54]:


len(TotalStops)


# In[55]:


try:
    Duration1=driver.find_elements(By.XPATH,"//div[@class='stop-info flexOne']")
    for i in Duration1[0:2001]:
        Duration.append(i.text.split('\n')[0])
except NoSuchElementException:
        Duration.append('-') 
        
time.sleep(500)


# In[56]:


len(Duration)


# In[58]:


Price=[]
try:
    Price1=driver.find_elements(By.XPATH,"//p[@class='blackText fontSize18 blackFont white-space-no-wrap']")
    for i in Price1[0:2001]:
        Price.append(i.text.split('\n')[0])
except NoSuchElementException:
        Price.append('-') 
        
time.sleep(500)


# In[59]:


len(Price)


# In[73]:


Dep_From = []
for i in range(0,len(DepartureTime),2):
    Dep_From.append(DepartureTime[i])
time.sleep(2)


# In[74]:


len(Dep_From)


# In[77]:


Arrivel = []
for i in range(1,len(DepartureTime),2):
    Arrivel.append(DepartureTime[i])
time.sleep(2)


# In[78]:


len(Arrivel)


# In[79]:


SourceName_From = []
for i in range(0,len(SourceName),2):
    SourceName_From.append(SourceName[i])
time.sleep(2)


# In[80]:


len(SourceName_From)


# In[81]:


Destination = []
for i in range(1,len(SourceName),2):
    Destination.append(SourceName[i])
time.sleep(2)


# In[82]:


len(Destination)


# In[83]:


print(len(FlightName),len(SourceName_From),len(Destination),len(Dep_From),len(Arrivel),len(TotalStops),len(Duration),len(Price))


# In[84]:


df=pd.DataFrame({'FlightName':FlightName,'SourceName_From':SourceName_From,'Destination':Destination,'Dep_From':Dep_From,'Arrivel':Arrivel,'TotalStops':TotalStops,'Duration':Duration,'Price':Price})
df


# In[85]:


df.to_csv(r'D:\I\02.Flip Robo Assignments\FlightPrice.csv', index=False)


# In[86]:


df


# In[87]:


df.head(2)


# In[88]:


df.tail(2)


# In[89]:


df.isnull().sum()


# In[90]:


df.shape


# In[91]:


df.dtypes


# In[92]:


#EDA Part Using countplot
import seaborn as sns


# In[93]:


ax=sns.countplot(x="FlightName",data=df)
print(df["FlightName"].value_counts())


# In[94]:


ax=sns.countplot(x="SourceName_From",data=df)
print(df["SourceName_From"].value_counts())


# In[95]:


ax=sns.countplot(x="Destination",data=df)
print(df["Destination"].value_counts())


# In[96]:


ax=sns.countplot(x="Dep_From",data=df)
print(df["Dep_From"].value_counts())


# In[97]:


ax=sns.countplot(x="Arrivel",data=df)
print(df["Arrivel"].value_counts())


# In[98]:


ax=sns.countplot(x="TotalStops",data=df)
print(df["TotalStops"].value_counts())


# In[99]:


ax=sns.countplot(x="Duration",data=df)
print(df["Duration"].value_counts())


# In[100]:


ax=sns.countplot(x="Price",data=df)
print(df["Price"].value_counts())


# In[101]:


#Encoding of data


# In[102]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[103]:


for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[104]:


df


# In[105]:


df.columns


# In[106]:


#Using Scatter Plot
sns.scatterplot(x='FlightName',y='Price',data=df)


# In[107]:


sns.scatterplot(x='SourceName_From',y='Price',data=df)


# In[108]:


sns.scatterplot(x='Destination',y='Price',data=df)


# In[109]:


sns.scatterplot(x='Dep_From',y='Price',data=df)


# In[110]:


sns.scatterplot(x='Arrivel',y='Price',data=df)


# In[111]:


sns.scatterplot(x='TotalStops',y='Price',data=df)


# In[112]:


sns.scatterplot(x='Duration',y='Price',data=df)


# In[113]:


import matplotlib.pyplot as plt
sns.pairplot(df)
plt.savefig('pairplot.png')
plt.show()


# In[114]:


#Plotting the histogram for univarience analysis


# In[115]:


df.hist(figsize=(20,20),grid=True,layout=(8,8),bins=30)


# In[116]:


df.shape


# In[117]:


df.describe()


# In[118]:


#Correlation


# In[119]:


df.corr()


# In[120]:


df.corr()['Price'].sort_values()


# In[121]:


# Correlation using Heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,linecolor='Black',fmt='.2f')


# In[122]:


df.plot(kind='density',subplots=True,layout=(6,11),sharex=False,legend=False,fontsize=1,figsize=(18,12))
plt.show()


# In[123]:


x=df.drop("Price",axis=1)
y=df["Price"]


# In[124]:


x


# In[125]:


y


# In[126]:


#No need to check skewness & outliers for Categorical data.Its a invalid operation


# In[127]:


import warnings
warnings.filterwarnings('ignore')


# In[128]:


from sklearn.feature_selection import mutual_info_classif


# In[129]:


mutual_info_classif(x,y)


# In[130]:


imp=pd.DataFrame(mutual_info_classif(x,y),index=x.columns)
imp


# In[131]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[132]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# In[133]:


#splitting data into 80% training and 20% testing
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[134]:


from sklearn import metrics
import pickle
def predict(ml_model):
    model=ml_model.fit(X_train,y_train)
    print('Training score: {}'.format(model.score(X_train,y_train)))
    predictions=model.predict(X_test)
    print('Predictions are: {}'.format(predictions))
    print('\n')
    r2_score=metrics.r2_score(y_test,predictions)
    print('r2 score is {}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(y_test,predictions))
    print('MSE:',metrics.mean_squared_error(y_test,predictions))
    print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test,predictions)))
    sns.distplot(y_test-predictions)


# In[137]:


from sklearn.ensemble import RandomForestRegressor
import numpy as np


# In[138]:


predict(RandomForestRegressor())


# In[139]:


predict(LinearRegression())


# In[140]:


predict(DecisionTreeRegressor())


# In[141]:


predict(KNeighborsRegressor())


# In[ ]:





# In[142]:


#As checked above Regressor RandomForestRegressor is the best model
from sklearn.ensemble import RandomForestRegressor


# In[143]:


n_estimators=[int(x) for x in np.linspace(start=100, stop=1200, num=6)]
max_depth=[int(x) for x in np.linspace(start=5, stop=30, num=4)]


# In[144]:


RF= RandomForestRegressor()


# In[145]:


from sklearn.model_selection import RandomizedSearchCV

parameters={
    'n_estimators': n_estimators,
    'max_features': ['auto','sqrt'],
    'max_depth':max_depth,
    'min_samples_split':[5,10,15,100]}


# In[147]:


RCV=RandomizedSearchCV(RandomForestRegressor(),parameters,cv=3, verbose=2, n_jobs=-1)
RCV.fit(X_train,y_train)
RCV.best_params_


# In[148]:


type(RCV)


# In[149]:


RCV.best_estimator_


# In[151]:


RCV_pred=RCV.best_estimator_.predict(X_test)


# In[152]:


from sklearn.metrics import r2_score
r2_score(y_test,RCV_pred)


# In[166]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=y_test,y=RCV_pred, color='y')
plt.plot(y_test,y_test,color='b')
plt.xlabel("Actual Price",fontsize=14)
plt.ylabel("Predicted Price",fontsize=14)
plt.title('RandomForestRegressor',fontsize=18)
plt.show()


# In[167]:


#Regularization
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[168]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[169]:


from sklearn.linear_model import Lasso


# In[171]:


parameters = {'alpha':[.0002,.002,.02,.2,2,20],'random_state':list(range(0,20))}
ls=Lasso()
clf=GridSearchCV(ls,parameters)
clf.fit(X_train,y_train)


# In[172]:


print(clf.best_params_)


# In[173]:


ls=Lasso(alpha=0.2,random_state=0)
ls.fit(X_train,y_train)
ls.score(X_train,y_train)
pred_ls=ls.predict(X_test)


# In[174]:


lss=r2_score(y_test,pred_ls)
lss


# In[178]:


cv_score=cross_val_score(ls,x,y,cv=3)
cv_mean=cv_score.mean()
cv_mean


# In[179]:


#Ensemble technoque


# In[181]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parameters = {'criterion':['mse','mae'],'max_features':["auto","sqrt","log2"]}
rf=RandomForestRegressor()
clf=GridSearchCV(RF,parameters)
clf.fit(X_train,y_train)

print(clf.best_params_)


# In[182]:


rf=RandomForestRegressor(criterion="mae",max_features="log2")
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
pred_decision=rf.predict(X_test)

rfs=r2_score(y_test,pred_decision)
print('R2 Score:',rfs*100)

rfscore=cross_val_score(rf,x,y,cv=3)
rfc=rfscore.mean()
print('cross val score:', rfc*100)


# In[183]:


#difference between r2 score & cross_val_score is very high so using different method method


# In[184]:


import pickle
filename ='Flip_FlightPrice'
pickle.dump(rf,open(filename,'wb'))


# In[185]:


#Conclusion


# In[187]:


loaded_model=pickle.load(open('Flip_FlightPrice','rb'))
result=loaded_model.score(X_test,y_test)
print(result)


# In[189]:


conclusion = pd.DataFrame([loaded_model.predict(X_test)[:],pred_decision[:]],index=["predicted","origional"])


# In[190]:


conclusion


# In[ ]:





# In[ ]:





# In[ ]:





# In[191]:


features=df.drop("Price",axis=1)
target=df["Price"]


# In[192]:


features


# In[193]:


target


# In[194]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# In[195]:


from sklearn.preprocessing import MinMaxScaler
mns=MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
lr=LinearRegression()
rf=RandomForestRegressor()
kn=KNeighborsRegressor()
dt=DecisionTreeRegressor()
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[196]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    lr.fit(features_train,target_train)
    pred_train=lr.predict(features_train)
    pred_test=lr.predict(features_test)
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_train,pred_train)}")
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_test,pred_test)}")
    print("\n")


# In[199]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    rf.fit(features_train,target_train)
    pred_train=rf.predict(features_train)
    pred_test=rf.predict(features_test)
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_train,pred_train)}")
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_test,pred_test)}")
    print("\n")


# In[200]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    dt.fit(features_train,target_train)
    pred_train=dt.predict(features_train)
    pred_test=dt.predict(features_test)
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_train,pred_train)}")
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_test,pred_test)}")
    print("\n")


# In[201]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    kn.fit(features_train,target_train)
    pred_train=kn.predict(features_train)
    pred_test=kn.predict(features_test)
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_train,pred_train)}")
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_test,pred_test)}")
    print("\n")


# In[202]:


# As compare above RandomForestRegressor is best


# In[203]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=i)
    rf.fit(features_train,target_train)
    pred_train=rf.predict(features_train)
    pred_test=rf.predict(features_test)
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_train,pred_train)}")
    print(f"At Random state {i},the training accuracy is:- {r2_score(target_test,pred_test)}")
    print("\n")


# In[204]:


features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=76)


# In[205]:


rf.fit(features_train,target_train)


# In[206]:


pred_test=rf.predict(features_test)


# In[207]:


print(r2_score(target_test,pred_test))


# In[208]:


#Cross Validation of Model


# In[209]:


Train_accuracy=r2_score(target_train,pred_train)
Test_accuracy=r2_score(target_test,pred_test)


# In[210]:


from sklearn.model_selection import cross_val_score
for j in range(2,20):
    cv_score=cross_val_score(rf,features,target,cv=j)
    cv_mean=cv_score.mean()
    print(f"At cross fold {j} the cv score is {cv_mean} and accuracy score for training is {Train_accuracy} and accuracy for the testing is {Test_accuracy}")
    print("\n")


# In[211]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=target_test,y=pred_test, color='y')
plt.plot(target_test,target_test,color='b')
plt.xlabel("Actual Charges",fontsize=14)
plt.ylabel("Predicted Charges",fontsize=14)
plt.title('RandomForestRegressor',fontsize=18)
plt.show()


# In[ ]:





# In[212]:


#Regularization


# In[214]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[215]:


from sklearn.linear_model import Lasso


# In[216]:


parameters = {'alpha':[.0002,.002,.02,.2,2,20],'random_state':list(range(0,20))}
ls=Lasso()
clf=GridSearchCV(ls,parameters)
clf.fit(features_train,target_train)


# In[217]:


print(clf.best_params_)


# In[218]:


ls=Lasso(alpha=0.02,random_state=0)
ls.fit(features_train,target_train)
ls.score(features_train,target_train)
pred_ls=ls.predict(features_test)


# In[219]:


lss=r2_score(target_test,pred_ls)
lss


# In[220]:


cv_score=cross_val_score(ls,features,target,cv=3)
cv_mean=cv_score.mean()
cv_mean


# In[221]:


#Ensemble technoque


# In[222]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parameters = {'criterion':['mse','mae'],'max_features':["auto","sqrt","log2"]}
rf=RandomForestRegressor()
clf=GridSearchCV(rf,parameters)
clf.fit(features_train,target_train)

print(clf.best_params_)


# In[223]:


rf=RandomForestRegressor(criterion="mae",max_features="auto")
rf.fit(features_train,target_train)
rf.score(features_train,target_train)
pred_decision=rf.predict(features_test)

rfs=r2_score(target_test,pred_decision)
print('R2 Score:',rfs*100)

rfscore=cross_val_score(rf,features,target,cv=3)
rfc=rfscore.mean()
print('cross val score:', rfc*100)


# In[224]:


#difference between r2 score & cross_val_score is very less so random forest is best method


# In[225]:


import pickle
filename ='Flip1_FlightPrice'
pickle.dump(rf,open(filename,'wb'))


# In[226]:


#Conclusion


# In[227]:


loaded_model=pickle.load(open('Flip1_FlightPrice','rb'))
result=loaded_model.score(features_test,target_test)
print(result)


# In[228]:


conclusion = pd.DataFrame([loaded_model.predict(features_test)[:],pred_decision[:]],index=["predicted","origional"])


# In[229]:


conclusion


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




