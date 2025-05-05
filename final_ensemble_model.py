from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GB,VotingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error as RMSE

#Read Dataset
df=read_csv("Dataset.csv")
#features
x=df.drop('f_type',axis='columns')
#labels
y=df.f_type

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,random_state=42,stratify=y)
m1=RandomForestClassifier(criterion='entropy',n_estimators=150)
m2=SVC(C=7,probability=True)
m3=GB(learning_rate=0.2)

bag_model=VotingClassifier(estimators=[('m1',m1),('m2',m2),('m3',m3)],weights=[4,3,1.5],voting='soft')
bag_model.fit(x_train,y_train)
y_pred=bag_model.predict(x_test)
score=cross_val_score(bag_model,x_scaled,y,cv=5)
error=RMSE(y_pred,y_test)
print (f'Accuracy with CV: {score.mean()}')
print (error)
