import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time
from sklearn import tree
import graphviz
from IPython.display import display
import pickle
import joblib

data_path = 'predictive_maintenance.csv'
data = pd.read_csv(data_path)
n = data.shape[0]
print('Features non-null values and data type:')
data.info()
print('Check for duplicate values:',data['ProductID'].unique().shape[0]!=n)

data['Tool_wear'] = data['Tool_wear'].astype('float64')
data['Rotational_speed'] = data['Rotational_speed'].astype('float64')
data['ProductID'] = data['ProductID'].apply(lambda x: x[1:])
data['ProductID'] = pd.to_numeric(data['ProductID'])

df = data.copy()
df.drop(columns=['UDI','ProductID'], inplace=True)

value = data['Type'].value_counts()
Type_percentage = 100*value/data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
features = [col for col in df.columns if df[col].dtype=='float64' or col =='Type']
target = ['Target','Failure_Type']
idx_RNF = df.loc[df['Failure_Type']=='Random Failures '].index
df.loc[idx_RNF,target]

first_drop = df.loc[idx_RNF,target].shape[0]
#print('Number of observations where RNF=1 but Machine failure=0:',first_drop)
df.drop(index=idx_RNF, inplace=True)

idx_ambiguous = df.loc[(df['Target']==1) & (df['Failure_Type']=='No Failure ')].index
second_drop = df.loc[idx_ambiguous].shape[0]
#print('Number of ambiguous observations:', second_drop)

df.drop(index=idx_ambiguous, inplace=True)

#print('Global percentage of removed observations:',(100*(first_drop+second_drop)/n))
df.reset_index(drop=True, inplace=True)   # Reset index
n = df.shape[0]
#print(df.describe())
num_features = [feature for feature in features if df[feature].dtype=='float64']
idx_fail = df.loc[df['Failure_Type'] != 'No Failure '].index
df_fail = df.loc[idx_fail]
df_fail_percentage = 100*df_fail['Failure_Type'].value_counts()/df_fail['Failure_Type'].shape[0]
#print('Failures percentage in data:',round(100*df['Target'].sum()/n,2))
n_working = df['Failure_Type'].value_counts()['No Failure ']
desired_length = round(n_working/0.8)
spc = round((desired_length-n_working)/4)  #samples per class
balance_cause = {'No Failure ':n_working,
                 'Overstrain Failure ':spc,
                 'Heat Dissipation Failure ':spc,
                 'Power Failure ':spc,
                 'Tool Wear Failure ':spc}
sm = SMOTENC(categorical_features=[0,7], sampling_strategy=balance_cause, random_state=0)
df_res, y_res = sm.fit_resample(df, df['Failure_Type'])

idx_fail_res = df_res.loc[df_res['Failure_Type'] != 'No Failure '].index
df_res_fail = df_res.loc[idx_fail_res]
fail_res_percentage = 100*df_res_fail['Failure_Type'].value_counts()/df_res_fail.shape[0]
#print('Percentage increment of observations after oversampling:',round((df_res.shape[0]-df.shape[0])*100/df.shape[0],2))
#print('SMOTE Resampled Failures percentage:',round(df_res_fail.shape[0]*100/df_res.shape[0],2))
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('Original Features distribution')
enumerate_features = enumerate(num_features)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('Features distribution after oversampling')
enumerate_features = enumerate(num_features)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,7))
fig.suptitle('Features distribution after oversampling - Diving deeper')
enumerate_features = enumerate(num_features)
sc = StandardScaler()
type_dict = {'L': 0, 'M': 1, 'H': 2}
cause_dict = {'No Failure ': 0,
              'Power Failure ': 1,
              'Overstrain Failure ': 2,
              'Heat Dissipation Failure ': 3,
              'Tool Wear Failure ': 4}
df_pre = df_res.copy()
df_pre['Type'].replace(to_replace=type_dict, inplace=True)
df_pre['Failure_Type'].replace(to_replace=cause_dict, inplace=True)
df_pre[num_features] = sc.fit_transform(df_pre[num_features]) 
pca = PCA(n_components=len(num_features))
X_pca = pd.DataFrame(data=pca.fit_transform(df_pre[num_features]), columns=['PC'+str(i+1) for i in range(len(num_features))])
var_exp = pd.Series(data=100*pca.explained_variance_ratio_, index=['PC'+str(i+1) for i in range(len(num_features))])
#print('Explained variance ratio per component:', round(var_exp,2), sep='\n')
#print('Explained variance ratio with 3 components: '+str(round(var_exp.values[:3].sum(),2)))
pca3 = PCA(n_components=3)
X_pca3 = pd.DataFrame(data=pca3.fit_transform(df_pre[num_features]), columns=['PC1','PC2','PC3'])
X_pca3.rename(mapper={'PC1':'Temperature',
                      'PC2':'Power',
                      'PC3':'Tool Wear'}, axis=1, inplace=True)
X, y = df_pre[features], df_pre[['Target','Failure_Type']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df_pre['Failure_Type'], random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure_Type'], random_state=0)

def eval_preds(model,X,y_true,y_pred,task):
    if task == 'binary':
        y_true = y_true['Target']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)[:,1]
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
    elif task == 'multi_class':
        y_true = y_true['Failure_Type']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics = pd.Series(data={'ACC':acc, 'AUC':auc, 'F1':f1, 'F2':f2})
    metrics = round(metrics,3)
    return cm, metrics

def tune_and_fit(clf,X,y,params,task):
    if task=='binary':
        f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Target'])
    elif task=='multi_class':
        f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Failure_Type'])
        
    #print('Best params:', grid_model.best_params_)
    train_time = time.time()-start_time
    mins = int(train_time//60)
    #print('Training time: '+str(mins)+'m '+str(round(train_time-mins*60))+'s')
    return grid_model

def predict_and_evaluate(fitted_models,X,y_true,clf_str,task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models,clf_str):
        y_pred[model_name] = fit_model.predict(X)
        if task == 'binary':
            cm, scores = eval_preds(fit_model,X,y_true,y_pred[model_name],task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model,X,y_true,y_pred[model_name],task)
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return y_pred, cm_dict, metrics

def fit_models(clf,clf_str,X_train,X_val,y_train,y_val):
    metrics = pd.DataFrame(columns=clf_str)
    for model, model_name in zip(clf, clf_str):
        model.fit(X_train,y_train['Target'])
        y_val_pred = model.predict(X_val)
        metrics[model_name] = eval_preds(model,X_val,y_val,y_val_pred,'binary')[1]
    return metrics

lr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC(probability=True)
rfc = RandomForestClassifier()
xgb = XGBClassifier() 

clf = [lr,knn,svc,rfc,xgb]
clf_str = ['LR','KNN','SVC','RFC','XGB'] 

metrics_0 = fit_models(clf,clf_str,X_train,X_val,y_train,y_val)

XX_train = X_train.drop(columns=['Process_temperature','Air_temperature'])
XX_val = X_val.drop(columns=['Process_temperature','Air_temperature'])
XX_train['Temperature']= X_train['Process_temperature']*X_train['Air_temperature']
XX_val['Temperature']= X_val['Process_temperature']-X_val['Air_temperature']
metrics_1 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)

XX_train = X_train.drop(columns=['Rotational_speed','Torque'])
XX_val = X_val.drop(columns=['Rotational_speed','Torque'])
XX_train['Power'] = X_train['Rotational_speed']*X_train['Torque']
XX_val['Power'] = X_val['Rotational_speed']*X_val['Torque']     
metrics_2 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)

XX_train = X_train.drop(columns=['Process_temperature','Air_temperature','Rotational_speed','Torque'])
XX_val = X_val.drop(columns=['Process_temperature','Air_temperature','Rotational_speed','Torque'])
XX_train['Temperature']= X_train['Process_temperature']*X_train['Air_temperature']
XX_val['Temperature']= X_val['Process_temperature']*X_val['Air_temperature']
XX_train['Power'] = X_train['Rotational_speed']*X_train['Torque']
XX_val['Power'] = X_val['Rotational_speed']*X_val['Torque']       
metrics_3 = fit_models(clf,clf_str,XX_train,XX_val,y_train,y_val)
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train['Target'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'binary')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'binary')
#print('Validation set metrics:',metrics_val_lr, sep='\n')
#print('Test set metrics:',metrics_test_lr, sep='\n')

'''cm_labels = ['Not Failure', 'Failure']
cm_lr = [cm_val_lr, cm_test_lr]
d = {'feature': X_train.columns, 'odds': np.exp(lr.coef_[0])}
odds_df = pd.DataFrame(data=d).sort_values(by='odds', ascending=False)
odds_df
knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier() 
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']

knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[300,500,700],
              'max_depth':[5,7],
              'learning_rate':[0.01,0.1],
              'objective':['binary:logistic']}
params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params],index=clf)

#print('GridSearch start')
fitted_models_binary = []
for model, model_name in zip(clf, clf_str):
    #print('Training '+str(model_name))
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'binary')
    fitted_models_binary.append(fit_model)

task = 'binary'
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_binary,X_val,y_val,clf_str,task)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_binary,X_test,y_test,clf_str,task)'''

#print('')
#print('Validation scores:', metrics_val, sep='\n')
#print('Test scores:', metrics_test, sep='\n')

'''f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
importances = pd.DataFrame()
for clf in fitted_models_binary:
    result = permutation_importance(clf, X_train,y_train['Target'],
                                  scoring=f2_scorer,random_state=0)
    result_mean = pd.Series(data=result.importances_mean, index=X.columns)
    importances = pd.concat(objs=[importances,result_mean],axis=1)
importances.columns = clf_str'''

lr = LogisticRegression(random_state=0,multi_class='ovr')
lr.fit(X_train, y_train['Failure_Type'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)
cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'multi_class')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'multi_class')
#print('Validation set metrics:',metrics_val_lr, sep='\n')
#print('Test set metrics:',metrics_test_lr, sep='\n')

cm_lr = [cm_val_lr, cm_test_lr]
cm_labels = ['No Fail','PWF','OSF','HDF','TWF']
odds_df = pd.DataFrame(data = np.exp(lr.coef_), columns = X_train.columns,index = df_res['Failure_Type'].unique())
odds_df

knn = KNeighborsClassifier()
svc = SVC(decision_function_shape='ovr')
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']
knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[100,300,500],
              'max_depth':[5,7,10],
              'learning_rate':[0.01,0.1],
              'objective':['multi:softprob']}
params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params],index=clf)

#print('GridSearch start')
fitted_models_multi = []
for model, model_name in zip(clf, clf_str):
    #print('Training '+str(model_name))
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'multi_class')
    fitted_models_multi.append(fit_model)

task = 'multi_class'
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(fitted_models_multi,X_val,y_val,clf_str,task)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(fitted_models_multi,X_test,y_test,clf_str,task)
metrics_final = metrics_val*metrics_test

#print('')
#print('Validation scores:', metrics_val, sep='\n')
#print('Test scores:', metrics_test, sep='\n')
'''f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
importances = pd.DataFrame()
for clf in fitted_models_multi:
    result = permutation_importance(clf, X_train,y_train['Failure_Type'],scoring=f2_scorer,random_state=0)
    result_mean = pd.Series(data=result.importances_mean, index=X.columns)
    importances = pd.concat(objs=[importances,result_mean],axis=1)
importances.columns = clf_str'''

# Calculating best model
macc = 0
bestm = 0
for i, j in enumerate(clf_str):
    acc = metrics_final[j][0]
    if acc > macc:
        macc = acc
        bestm = i
print(bestm, "\t", clf_str[bestm])

# Saving model to file
best_model = fitted_models_multi[bestm]
joblib.dump(best_model, 'newfile.pkl')