#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import binned_statistic

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import svm


from mlxtend.feature_selection import SequentialFeatureSelector
import collections

from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix



from sklearn.naive_bayes import GaussianNB


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold


from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict


import warnings

import copy
warnings.filterwarnings("ignore")




# In[2]:


file_location = "./data/creditcard.csv"


# In[3]:


df = pd.read_csv(file_location)


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# Class column determines if transaction is Fraudulent
# 

# In[7]:


df['Class'].value_counts()


# Determine percentage of Fraudlent and Crediable Transactions

# In[8]:


def get_percentage(value, total):
    return round((value / total) * 100, 2)


# In[9]:


def print_transaction_percentage(df):
    CREDIBLE = 0
    FRAUDULENT = 1
    total_transactions = len(df['Class'])
    total_credible_transactions = df['Class'].value_counts()[CREDIBLE]
    total_fraud_transactions = df['Class'].value_counts()[FRAUDULENT]
    print("Credible Transactions: "+ str(get_percentage(total_credible_transactions, total_transactions)))
    print("Fraudulent Transactions: "+ str(get_percentage(total_fraud_transactions, total_transactions)))


# In[10]:


print_transaction_percentage(df)


# In[11]:


def plot_count(df, title, col):
    sns.countplot(col, data=df)
    plt.title(title, fontsize=20)


# In[12]:


def create_distributed_plot(sub_df, title):
    f, ax = plt.subplots(1, 1, figsize=(18,4))
    col_array_vals = sub_df.values
    sns.distplot(col_array_vals, ax=ax, color='r')
    ax.set_title(title, fontsize=14)
    ax.set_xlim([min(col_array_vals), max(col_array_vals)])


# In[ ]:





# In[13]:


plot_count(df, 'Class Count [ 0 == Credible, 1 == Fraudulent ]', 'Class')
create_distributed_plot(df['Amount'], 'Distrubtion of Amount')
create_distributed_plot(df['Time'], 'Distrubtion of Time')
plt.show()


# In[ ]:





# In[14]:


bins_amount = 100


# In[15]:


# equal sized bins
df['bin_time'] = pd.cut(df['Time'], bins=bins_amount, labels=False )


# In[16]:


df['bin_time'].unique()


# In[17]:


len(df['bin_time'].unique())


# In[30]:


df['bin_amount'] = pd.cut(df['Amount'], bins=100, labels=False )


# In[32]:


df['bin_amount'].unique()


# In[ ]:





# In[33]:


df.drop(['Amount', 'Time'], axis=1, inplace=True )


# In[ ]:





# In[34]:


df.head(10)


# In[35]:


def show_correlation_matrix(data, title):
    f, ax = plt.subplots(1, 1, figsize=(12,10))
    # Entire DataFrame
    corr = data.corr()
    sns.heatmap(corr, cmap='coolwarm_r', ax=ax)
    ax.set_title(title, fontsize=14)


# In[36]:


show_correlation_matrix(df, 'Correlation Map Before Undersampleing')


# In[37]:


# shuffle
df = df.sample(frac=1)


# In[38]:


# Creating a balanced df
fraud_df = df.loc[df['Class'] == 1]
crediable_df = df.loc[df['Class'] == 0][:len(fraud_df)]
balanced_df = pd.concat([fraud_df, crediable_df]).sample(frac=1, random_state = 50)


# In[39]:


len(balanced_df)


# In[40]:


print_transaction_percentage(balanced_df)


# In[ ]:





# In[41]:


plot_count(balanced_df, 'Class Count [ 0 == Credible, 1 == Fraudulent ] for blanaced_df', 'Class')
create_distributed_plot(balanced_df['bin_amount'], 'Distrubtion of Amount')
create_distributed_plot(balanced_df['bin_time'], 'Distrubtion of Time')
show_correlation_matrix(balanced_df, 'Correlation Map After Undersampleing')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


def filter_relative_features(data, target, start):
    cor = data.corr()
    cor_target = abs(cor[target])
    relevant_features = cor_target[cor_target>start]
    return relevant_features


# In[43]:


def get_columns_from_series(s):
    result = []
    for i,r in s.items():
        result.append(i)
    return result


# In[44]:


filtered_features_based_on_overall_correlation = filter_relative_features(balanced_df, 'Class', 0.5)


# In[45]:


filtered_features_based_on_overall_correlation = get_columns_from_series(filtered_features_based_on_overall_correlation)


# In[46]:


# excluding target feature
filtered_features_based_on_overall_correlation.remove('Class')


# In[47]:


# Shallow feature correlation shows 9. 
# Will now use backward feature selection to determine the best features per model 
# Will use this size for and this feature selection to determine the best possible features
size_of_filtered_featured_based_on_correlation = len(filtered_features_based_on_overall_correlation)


# In[ ]:





# In[48]:


random_forest = RandomForestClassifier(n_estimators=20)


# In[49]:


naive_bayes = GaussianNB()


# In[ ]:





# In[50]:


supervised = {
    naive_bayes.__class__.__name__ : naive_bayes,
    random_forest.__class__.__name__ : random_forest,
}


# In[51]:


supervised


# In[52]:


balanced_df_features = balanced_df.drop('Class', axis=1)
balanced_df_target_feature_only = balanced_df['Class']


# In[ ]:





# In[ ]:





# In[53]:


def get_backward_selected_features(amount_of_selected_features, m, df_with_features, df_with_target_feature):
#     Uses the backward feature selection approach
    feature_selector = SequentialFeatureSelector(m,
           k_features=amount_of_selected_features,
           forward=False,
           verbose=2,
           scoring='roc_auc',
           cv=4)
    features = feature_selector.fit(df_with_features, df_with_target_feature)
    filtered_features = df_with_features.columns[list(features.k_feature_idx_)]
    return list(filtered_features)
    


# In[54]:


def get_filtered_features_by_model(amount_of_selected_features, supervised, df_with_features, df_with_target_feature):
    filtered_features_by_model = {}
    for modelName, modelObj in supervised.items():
        filtered_features_by_model[modelName+"_backward_selected_features"] = get_backward_selected_features(amount_of_selected_features, modelObj, df_with_features, df_with_target_feature)
    return filtered_features_by_model


# In[55]:


filtered_features_by_correlation = get_filtered_features_by_model(
    size_of_filtered_featured_based_on_correlation,
    supervised,
    balanced_df_features, balanced_df_target_feature_only )


# In[ ]:





# In[56]:


filtered_features_by_correlation


# In[ ]:





# In[57]:


filtered_features_by_correlation['filtered_correlation_selected_features']  = filtered_features_based_on_overall_correlation


# In[58]:


# This is the set a features I will test the balanced data set on 
filtered_features_by_correlation


# In[ ]:





# In[59]:


models_to_test = supervised


# In[ ]:





# In[60]:


def create_confustion_matrix_and_score(correlation_name, model_name, model_to_test, X_test, y_test, filtered_features, version):
    score = model_to_test.score(X_test, y_test)
    y_predicted = model_to_test.predict(X_test)
    
    cm = confusion_matrix(y_test, y_predicted)
    
    accuray_score_data = accuracy_score(y_test, y_predicted)
    classification_report_data = classification_report(y_test, y_predicted)
    recall_score_data = recall_score(y_test, y_predicted)
    percision_score_data = precision_score(y_test, y_predicted)
    f1_score_data = f1_score(y_test, y_predicted)
    roc_auc_score_data = roc_auc_score(y_test, y_predicted)
    
    figure = plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    title =  "VERSION: " +version
    title += "\nModel used: " 
    title += model_to_test.__class__.__name__
    title += "\nScore: "
    title += str(roc_auc_score_data)
    title += "\nCorelation Name : "
    title += correlation_name
    title += "\nFiltered Features Used: "
    title += str(filtered_features)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    file_name = 'version_'+version+'confusion_matrix_'+str(model_to_test.__class__.__name__)+"_correlation_name_"+correlation_name+".jpg"
    figure.savefig(file_name)
    return {
        'cm' : cm,
        'title' : title,
        'file_name' : file_name,
        'model_name' : model_name,
        'correlation_name' : correlation_name,
        'model' : model_to_test,
        'score' : score,
        'accuray_score_data' : accuray_score_data,
        'classification_report_data' : classification_report_data,
        'recall_score_data' : recall_score_data,
        'percision_score_data' : percision_score_data,
        'f1_score_data' : f1_score_data,
        'roc_auc_score_data' : roc_auc_score_data
    } 


# In[ ]:





# In[ ]:





# In[61]:


def test_models_by_feature_list(correlation_name,
                                list_of_filtered_features,
                                model_name,
                                model_to_test,
                                X_train,
                                X_test,
                                y_train,
                                y_test, version):
    
    model_to_test.fit(X_train, y_train)
    training_score = cross_val_score(model_to_test, X_train, y_train, cv=5)
    data = create_confustion_matrix_and_score(correlation_name, model_name, model_to_test, X_test, y_test, list_of_filtered_features, version)
    data['training_score'] = training_score
    model_pred = cross_val_predict(model_to_test, X_train, y_train, cv=5)
    roc_auc_score_corss_validation = roc_auc_score(y_train, model_pred)
    data['roc_auc_score_corss_validation']  = roc_auc_score_corss_validation
    data['roc_curve_data'] = roc_curve(y_train, model_pred)
    return data
    
    
    


# In[ ]:





# In[62]:


def test_data(data_frame, version):
    
    results  = []

    for correlation_name, list_of_filtered_features in filtered_features_by_correlation.items():

        df_with_high_correlated_features = data_frame[list_of_filtered_features]

        df_target = data_frame['Class']

        X_train, X_test, y_train, y_test = train_test_split(df_with_high_correlated_features, df_target,test_size=.2)

        for model_name, model_to_test in models_to_test.items():
            res = test_models_by_feature_list(correlation_name,
                                                                list_of_filtered_features,
                                                                model_name,
                                                                model_to_test,
                                                                X_train,
                                                                X_test,
                                                                y_train,
                                                                y_test, version)
            results.append(res)
        plt.show()
    return results


# In[ ]:





# In[63]:


def graph_roc_curve_multiple(data):
    fpr, tpr, thresold = data['roc_curve_data']
    plt.title('ROC Curve \n 2 Classifiers', fontsize=18)
    plt.plot(fpr, tpr, label=data['model_name']+'_'+data['correlation_name']+'{:.4f}'.format(data['roc_auc_score_data']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()


# In[64]:


def create_roc_cruve(results):    
    plt.figure(figsize=(16,8))
    for i, d in enumerate(results):
        graph_roc_curve_multiple(d)
    plt.show()


# In[65]:


def print_result(d):
    CM = d['cm']
    TP = CM[0][0]
    FN = CM[0][1]
    FP = CM[1][0]
    TN = CM[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    model_and_feature_used = 'Model and Feature used: '+ d['title'] 
    TN = "TN: " + str( np.round(TN, 2) )
    FN = "FN: " + str( np.round(FN, 2) )
    TP = "TP: " + str( np.round(TP, 2) )
    FP = "FP: " + str( np.round(FP, 2) )
    TPR = "TPR: " + str( np.round(TPR, 2) )
    TNR = "TNR: " + str( np.round(TNR, 2) )
    PPV = "PPV: " + str( np.round(PPV, 2) )
    NPV = "NPV: " + str( np.round(NPV, 2) )
    FPR = "FPR: " + str( np.round(FPR, 2) )
    FNR = "FNR: " + str( np.round(FNR, 2) )
    FDR = "FDR: " + str( np.round(FDR, 2) )
    Recall_Score = 'Recall Score: {:.2f}'.format(d['recall_score_data']) 
    Precision_Score = 'Precision Score: {:.2f}'.format(d['percision_score_data']) 
    F1_Score = 'F1 Score: {:.2f}'.format(d['f1_score_data'])
    Accuracy_Score = 'Accuracy Score: {:.2f}'.format(d['accuray_score_data']) 



    line =  TN +', '+ FN +', '+ TP +', '+ FP +', '+ TPR +', '+ TNR +', '+ PPV +', '+ NPV +', '+ FPR +', '+ FNR +', '+ FDR +', '+ Recall_Score +', '+ Precision_Score +', '+ F1_Score +', '+ Accuracy_Score

    print('---' * 10)
    print( model_and_feature_used )
    print( TN )
    print( FN )
    print( TP )
    print( FP )
    print( TPR )
    print( TNR )
    print( PPV )
    print( NPV )
    print( FPR )
    print( FNR )
    print( FDR )
    print( Recall_Score )
    print( Precision_Score )
    print( F1_Score )
    print( Accuracy_Score )
    print(line)
    print('---' * 10)


# In[66]:


def print_all(results):
    for i, d in enumerate(results):
        print_result(d)
   


# In[67]:


def show_all_confusion_matrix(results):
    stack_results = copy.deepcopy(results)
    fig, ax = plt.subplots(3, 2,figsize=(30,30))
    r = 0;
    c = 0;

    while (r < 3):
        while(c < 2):
            d = stack_results.pop()
            sns.heatmap(d['cm'], ax=ax[r][c], annot=True, cmap=plt.cm.copper, square=True, linewidths=0.1, annot_kws={"size":30}) 
            ax[r, c].set_title(d['title'], fontsize=16)
            ax[r, c].set_xticklabels(['', ''], fontsize=100, rotation=90)
            ax[r, c].set_yticklabels(['', ''], fontsize=100, rotation=360)
            c = c + 1
        c = 0
        r = r + 1 

    plt.show()


# In[ ]:





# In[68]:


def determine_max_correlation(cm1, cm1Index , cm2, cm2Index):

    correct_credible_transactions_cm1 = cm1[0][0]
    incorrect_fraudulent_transactions_cm1 = cm1[0][1]
    incorrect_credible_transactions_cm1 = cm1[1][0]
    correct_fraudulent_transactions_cm1 = cm1[1][1]
    
    correct_credible_transactions_cm2 = cm2[0][0]
    incorrect_fraudulent_transactions_cm2 = cm2[0][1]
    incorrect_credible_transactions_cm2 = cm2[1][0]
    correct_fraudulent_transactions_cm2 = cm2[1][1]
    
#     Misclassifying fradulent transactions has the highest bussiness cost
    if(incorrect_fraudulent_transactions_cm1 > incorrect_fraudulent_transactions_cm2):
        return cm2Index
    elif(incorrect_fraudulent_transactions_cm1 < incorrect_fraudulent_transactions_cm2):
        return cm1Index

#     Checking count for in correct credible transactions
    if(incorrect_credible_transactions_cm1 > incorrect_credible_transactions_cm2):
        return cm2Index
    elif(incorrect_credible_transactions_cm1 < incorrect_credible_transactions_cm2):
        return cm1Index
    
#  Who has the most correct credible transactions
    if(correct_credible_transactions_cm1 > correct_credible_transactions_cm2):
        return cm1Index
    elif(correct_credible_transactions_cm1 < correct_credible_transactions_cm2):
        return cm2Index
    
    if(correct_fraudulent_transactions_cm1 > correct_fraudulent_transactions_cm2):
        return cm1Index
    elif(correct_fraudulent_transactions_cm1 < correct_fraudulent_transactions_cm2):
        return cm2Index
    
    return cm2Index
        


# In[69]:


def get_best_result(res):
    maxResultIndex = 0
    initCm = res[0]['cm']
    for i, r in enumerate(res):
        maxResultIndex = determine_max_correlation(res[maxResultIndex]['cm'], maxResultIndex, r['cm'], i)    
    print("MAX_RESULT_INDEX: "+ str(maxResultIndex))
    return res[maxResultIndex]


# In[ ]:





# In[ ]:





# In[70]:


version = "balanced_df"


# In[71]:


results = test_data(balanced_df, version)


# In[72]:


create_roc_cruve(results)


# In[73]:


show_all_confusion_matrix(results)


# In[74]:


print_all(results)


# In[75]:


best_result = get_best_result(results)


# In[76]:


print_result(best_result)


# In[77]:


print("Best result from test data: ")
print(best_result['title'])


# In[78]:


results


# In[ ]:





# In[ ]:





# In[79]:


results_based_off_og = test_data(df, "OG_DATA")


# In[80]:


create_roc_cruve(results_based_off_og)


# In[81]:


show_all_confusion_matrix(results_based_off_og)


# In[82]:


print_all(results_based_off_og)


# In[83]:


best_result = get_best_result(results_based_off_og)


# In[84]:


print_result(best_result)


# In[85]:


print("Best result from test data: ")
print(best_result['title'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




