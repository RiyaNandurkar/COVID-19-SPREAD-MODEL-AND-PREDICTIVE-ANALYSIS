import pandas as pd
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df = pd.read_csv('covid_19_india.csv',parse_dates=['Date'],dayfirst=True,na_values=["??","????","########"])
df.dropna(axis=0,inplace=True)
print(df.isnull().sum())
print(df.head())
df = df[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
df.columns = ['Date','States','Cured','Death','Confirmed']
print(df.tail())
present = df[df.Date=='2021-05-19'] 
print(present.shape)
print(present.head())
#max_confirmed_cases = present.sort_values(by='Confirmed',ascending=False)
#print(max_confirmed_cases.head())
#top_states_confirmed = max_confirmed_cases[0:5]
#print(top_states_confirmed.head())
#sns.set(rc={'figure.figsize': (10,5)})
#sns.barplot(x="States",y="Confirmed",data=top_states_confirmed,hue="States")#plt.show()

#max_death_cases = present.sort_values(by='Death',ascending=False)
#print(max_death_cases.head())
#top_states_deaths = max_death_cases[0:5]
#sns.set(rc={'figure.figsize': (10,5)})
#sns.barplot(x="States",y="Death",data=top_states_deaths,hue="States")
#plt.show()

mh = df[df.States == "Maharashtra"]
#print(mh.head())
#print(mh.tail())
#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Confirmed",data=mh,color='g')
#plt.title('Maharashtra Confirmed Cases')
#plt.show()

#sns.set(rc={'fig/ure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Death",data=mh,color='r')
#plt.title('Maharashtra Death Cases')
#plt.show()

#ka = df[df.States=="Karnataka"]
#print(ka.head())
#print(ka.tail())
#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Confirmed",data=ka,color='g')
#plt.title('Karnataka Confirmed Cases')
#plt.show()

#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Death",data=ka,color='r')
#plt.title('Karnataka Death Cases')
#plt.show()

#ke = df[df.States=="Kerala"]
#print(ke.head())
#print(ke.tail())
#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Confirmed",data=ke,color='g')
#plt.title('Kerala Confirmed Cases')
#plt.show()

#sns.set(rc={'figure.figsize':(10,5)})
#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Death",data=ta,color='r')
#plt.title('Tamil Nadu Death Cases')
#plt.show()

#up = df[df.States=="Uttar Pradesh"]
#print(up.head())
#print(up.tail())
#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Confirmed",data=up,color='g')
#plt.title('Uttar Pradesh Confirmed Cases')
#plt.show()

#sns.set(rc={'figure.figsize':(10,5)})
#sns.lineplot(x="Date",y="Death",data=up,color='r')
#plt.title('Uttar Pradesh Death Cases')
#plt.show()

tests = pd.read_csv('StatewiseTestingDetails.csv')
tests.dropna(axis=0,inplace=True)
print(tests.head())
print(tests.tail())

mh['Date'] = mh['Date'].map(dt.datetime.toordinal)
print(mh.tail())
x = mh['Date']
y = mh['Confirmed']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

rf = RandomForestRegressor(n_estimators=20,random_state=0)
rf.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
print(rf.predict([[737930]]))

rf = RandomForestRegressor(n_estimators=20,random_state=0)
rf.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
