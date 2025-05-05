#this program optimizes weights for soft voting 

from pyswarms.single import GlobalBestPSO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import root_mean_squared_error as RMSE
from pandas import read_csv
from numpy import array,append

#Read Dataset
df=read_csv("dataset.csv")
#features
x=df.drop('f_type',axis='columns')
#labels
y=df.f_type

#scaling
scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)

#splitting
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,random_state=42,test_size=0.2)
#model
m1=RandomForestClassifier(criterion='entropy',n_jobs=-1)
m2=GradientBoostingClassifier()
m3=SVC(probability=True)
#n_jobs for parallel operation

#Fitness function:
def func(param_set):
    rmse_set=array([])
    for param in param_set:
        #ensemble Models
        model=VotingClassifier(estimators=[('m1',m1),('m2',m2),('m3',m3)],voting='soft',weights=[param[0],param[1],param[2]])
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        error=RMSE(y_pred,y_test)
        rmse_set=append(rmse_set,error)
    return(rmse_set)

#lower and upper limit parameters
b=(array([1,1,1]),array([5,5,5]))

#parameters of PSO
op={'c1':0.5,'c2':0.5,'w':0.9}

#optimizer
optimizer=GlobalBestPSO(n_particles=20,options=op,bounds=b,dimensions=3)
best_cost,best_position=optimizer.optimize(func,iters=20)

print (best_cost,best_position)


