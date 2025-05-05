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

#Fitness function:
def func(param_set):
    rmse_set=array([])
    for param in param_set:
        #model
        m1=RandomForestClassifier(n_estimators=int(param[0]))
        m2=GradientBoostingClassifier(learning_rate=param[1])
        m3=SVC(C=param[2],probability=True)

        #ensemble Models
        model=VotingClassifier(estimators=[('m1',m1),('m2',m2),('m3',m3)],voting='soft')
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        error=RMSE(y_pred,y_test)
        rmse_set=append(rmse_set,error)
    return(rmse_set)

#hyper parameters: 
# n_estimators for RF: 50-100
# learning_rate for GB: 0.05 to 0.2
# regularization parameter (C) for SVM: 0.1 to 10

#lower and upper limit parameters
b=(array([50,0.05,0.1]),array([200,0.2,10]))

#parameters of PSO
op={'c1':0.5,'c2':0.5,'w':0.9}

#optimizer
optimizer=GlobalBestPSO(n_particles=20,options=op,bounds=b,dimensions=3)
best_cost,best_position=optimizer.optimize(func,iters=20)

print (best_cost,best_position)


