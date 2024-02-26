#!/usr/bin/env python
# coding: utf-8

# VARIABLE DESCRIPTION:
# 
# >age (numeric)
# 
# >job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
# 
# >marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
# 
# >education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
# 
# >default: has credit in default? (categorical: "no","yes","unknown")
# 
# >housing: has housing loan? (categorical: "no","yes","unknown")
# 
# >loan: has personal loan? (categorical: "no","yes","unknown")
# 
# >Last Telephone Contact
# contact: contact communication type (categorical: "cellular","telephone")
# 
# >month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# >day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
# 
# >duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# Other Attributes
# >campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# >pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 
# >previous: number of contacts performed before this campaign and for this client (numeric)
# 
# >poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
# 
# Social and Economic Context
# >emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 
# >cons.price.idx: consumer price index - monthly indicator (numeric)
# 
# >cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 
# >euribor3m: euribor 3 month rate - daily indicator (numeric)
# 
# >nr.employed: number of employees - quarterly indicator (numeric)
# 

# In[1]:


P_TARGETED = .066
AVG_REVENUE = 1083
AVG_COST = -8


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('bank-additional-full.csv', sep=';')
df.head()


# In[4]:


df.shape


# In[5]:


cols=df.columns.tolist()


# In[6]:


for _ in cols:
    print(_,"------",df[_].unique(),"\n")


# In[7]:


poutcome=df[df['poutcome']!='nonexistent'].poutcome.apply(lambda x:1 if x=='success' else 0)
coutcome=df['y'].apply(lambda x:1 if x=='yes' else 0)

print("Total Records: ",df.shape[0])
print("Previous markting campaign success rate: ",poutcome.sum()/len(poutcome))
print("Current markting campaign success rate: ",coutcome.sum()/df.shape[0])


# In[8]:


Y=df.y
X=df.drop('y',axis=1)


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=5, stratify=Y, shuffle=True, test_size=.2)


# In[10]:


df_test=pd.concat([x_test,y_test],axis=1)


# In[11]:


bank=pd.concat([x_train,y_train],axis=1)


# In[12]:


bank.dtypes.groupby(bank.dtypes).size()


# In[13]:


dtypes = pd.DataFrame(bank.dtypes.rename('type')).reset_index().astype('str')
dtypes = dtypes[dtypes['index']!='duration']
numeric = dtypes[(dtypes.type.isin(['int64', 'float64'])) & (dtypes['index'] != 'duration')]['index'].values
categorical = dtypes[~(dtypes['index'].isin(numeric)) & (dtypes['index'] != 'y')]['index'].values

print('Numeric:\n', numeric)
print('Categorical:\n', categorical)


# ## ANALYSIS : Categorical and Numeric

# In[14]:


bank[categorical].isnull().sum()


# In[15]:


bank[numeric].isnull().sum()


# In[16]:


for i in categorical:
    plt.figure(figsize=(15,10))
    sns.countplot(data=bank,x=i)
    plt.xlabel(i.upper())
    plt.show()


# >A very few proportion of customers are illiterate
# 
# >No or very few customers who have defaulted on a loan - not surprising, the bank probably does not want to extend an offer to customers with bad credit
# 
# >Similar proportions of customers with and without housing loans
# 
# >Few customers have personal loans
# 
# >Almost double as many cellular as landline phone calls
# 
# >Fewer calls made in the second half of the year
# 
# >Calls are uniform accross days of the week
# 
# >Some customers were already contacted previously by the bank but many were never contacted at all

# In[17]:


marital_resp_rate=(bank.groupby('marital').y.value_counts()/bank.groupby('marital').size()).rename('rate').reset_index()


# In[18]:


marital_pos_rate=marital_resp_rate[marital_resp_rate['y'] == "yes"]
marital_pos_rate


# >marital status isn't very predictive of outcome

# In[19]:


ct_resp_rate=(bank.groupby('contact').y.value_counts()/bank.groupby('contact').size()).rename('rate').reset_index()


# In[20]:


ct_pos_rate=ct_resp_rate[ct_resp_rate['y'] == "yes"]
ct_pos_rate


# > contact medium appears to have good predictive power - nearly 4x increase in conversion rate for customers who were contacted on their mobile phone

# In[21]:


for i in categorical:
    plt.figure(figsize=(15,10))
    sns.countplot(data=bank,x=i,hue=bank.y)
    plt.xlabel(i.upper())
    plt.show()


# >The ‘default’ attribute seems to have significant predictive potential, particularly due to a substantial number of instances being ‘no’.
# 
# >In terms of job categories, individuals who are administrators, retired, students, or unemployed tend to have a higher response rate. However, the groups of retired individuals, students, and unemployed individuals are smaller in size. .
# 
# >A number of attributes (such as job, marital status, education, default, housing, and loan) have missing values. However, for all these variables, except ‘default’, the occurrence of the ‘unknown’ value is quite low. Moreover, the response rate associated with the ‘unknown’ value is comparable to that of other known values. This suggests that there may not be a need to employ techniques for handling missing values for these variables. 

# In[22]:


bank[numeric].hist(figsize=(12,12));


# In[23]:


bank[numeric].describe()


# >outliers in age indicating customers who are very old
# 
# >The fact that a majority of instances have a pdays value of 999 (missing) is going to be problematic if we want to use the attribute as a model feature. Since so few instances have an associated pdays, we could remove it from the analysis and modelling process. 
# 
# >previous can be treated as a discrete variable

# In[24]:


plt.figure(figsize=(10,7))
corr = bank[numeric].drop('pdays', axis=1).corr(method='spearman')
sns.heatmap(corr, annot=True)


# Correlation between the following are significant:
# 
# *emp.var.rate(employment variation rate) and cons.price.idx*
#     
# *emp.var.rate(employment variation rate) and euribor3m (euriboro 3 month rate)*
#     
# *emp.var.rate and nr.employed (number of employees)*
#     
# *nr.employed and euribor3m*

# In[25]:


customer_att=['age','campaign','previous','y']
numeric_outcome=pd.concat([bank[numeric],bank['y']],axis=1)
sns.pairplot(numeric_outcome[customer_att],hue='y',aspect=1.4)


# >customers who were contacted more than ten times in 18-60 were not likely to respond to the campaign. 

# In[26]:


camp1=bank[bank['campaign']<10]
camp1=camp1.groupby("y").size()/len(camp1)


# In[27]:


camp1


# In[28]:


camp2=bank[bank['campaign']>=10]
camp2=camp2.groupby("y").size()/len(camp2)


# In[29]:


camp2


# In[30]:


y_dt=bank[bank.y=='yes']['age']
n_dt=bank[bank.y=='no']['age']

sns.kdeplot(y_dt, color='steelblue', label='yes')
sns.kdeplot(n_dt, color='red', label='no')
plt.legend(title='y')
plt.xlabel("Age")
plt.show()


# >Customers at the younger and older ends of the age spectrum appear to have a higher conversion rate compared to those in the middle age range. This may be because younger customers are typically interested in growing their savings, while older customers may be looking to invest their capital.

# In[31]:


soc_Att=['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed','y']
soc_ec = bank[soc_Att]

for i in soc_Att[:-1]:  
    
    plt.figure(figsize=(10,5))
    for outcome in ['yes', 'no']:
        data = soc_ec[soc_ec.y == outcome][i]
        sns.kdeplot(data, label=outcome)

    # Add legend and title
    plt.legend(title='y')
    plt.show()


# Customers are likely to convert when:
# 
# *emp.var.rate is low*
# 
# *consumer price index is in lower spectrum*
# 
# *euribor-3 month rate is low is low*
# 
# *no. of employed is low*
# 
# *consumer confidence index is in either of the lowest or highest spectrum*

#  What we can do is modify pdays as an indicator variable 
#  
#  Transform previous to categorical variable
#  
# Binning age and campaign
# 

# In[32]:


def prev_contacted(X):
        return (X != 999)

def partoflast_campaign(X):
        pcampaign = ~(X == 'nonexistent')
        return pcampaign

def contacted_10(X):
        return (X >= 10)

def prev_discrete(X):
        return str(X)


# In[33]:


X['prev_contacted']=X['pdays'].apply(prev_contacted)


# In[34]:


X['p_lastcamp']=X['poutcome'].apply(partoflast_campaign)
X['previous_disc']=X['previous'].apply(prev_discrete)
X['camp_gte10']=X['campaign'].apply(contacted_10)


# In[35]:


categorical


# In[36]:


CATEGORICAL_FEATURES =  [
  'job',
  'marital',
  'education',
  'default',
  'housing',
  'loan',
  'contact',
  'month',
  'day_of_week',
  'poutcome',
    'p_lastcamp','previous_disc','camp_gte10','prev_contacted']


# In[37]:


NUMERIC_FEATURES = [
    'age', 
    'campaign', 
    'previous', 
    'emp.var.rate', 
    'cons.price.idx', 
    'cons.conf.idx', 
    'euribor3m', 
    'nr.employed'
]


# In[38]:


X['age']=np.log(X['age'])


# In[39]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_cols = encoder.fit_transform(X[CATEGORICAL_FEATURES])


encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))

# Now you can join this DataFrame with your original DataFrame
X = X.join(encoded_df)


# In[40]:


del encoded_df


# In[41]:


from sklearn.preprocessing import RobustScaler


# In[42]:


scaler=RobustScaler()


# In[43]:


X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])


# In[44]:


data=X.copy()


# In[45]:


data=data.drop(CATEGORICAL_FEATURES,axis=1)


# In[46]:


data.columns


# In[47]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif, chi2,SelectKBest


# In[48]:


# Assuming df is your DataFrame and 'target' is your target column
df['y'] = df['y'].map({'yes': 1, 'no': 0})


# In[49]:


y=df['y']


# In[50]:


def cv_score(model, rstate = 45, cols = data.columns):
    dff=data[cols]
    cv=[]
    ac=[]
    i=1
    kf=StratifiedKFold(n_splits=5,random_state=rstate,shuffle=True)
    for df_index,test_index in kf.split(dff,y):
        xtr,xval=dff.loc[df_index],dff.loc[test_index]
        ytr,yval=y.loc[df_index],y.loc[test_index]
        print("\n{} of KFold{}".format(i,kf.n_splits))
        
        model=model
        model.fit(xtr,ytr)
        pred_pr=model.predict_proba(xval)
        pp=model.predict(xval)
        
        
        recall=recall_score(yval,pp)
        precision=precision_score(yval,pp)
        f1score=f1_score(yval,pp)
        
        print(" Recall Score {} Precision Score {:.4f} F1 score{:.4f}".format(recall,precision,f1score))
        acc=accuracy_score(yval,pp)
        cv.append(precision)
        ac.append(acc)
        i+=1
    return cv


# In[51]:


model = GradientBoostingClassifier()
cv=cv_score(model)


# #### Our fundamental model operates on a simple principle: 
# "If a customer converted during the last campaign, predict that they will convert again when contacted for the current campaign." The outcome of the previous campaign is represented by the attribute 'poutcome', which can have one of three values: 'success', 'failure', or 'nonexistent'. We interpret 'nonexistent' to mean that the customer was not contacted during the previous campaign.

# In[52]:


mapping={'yes':1,'no':0}
Y=Y.map(mapping)


# In[53]:


n_instances = len(X)
p_instances = Y.sum() / len(Y)
p_targeted = .066
n_targeted = int(n_instances*p_targeted)

print('Number of instances: {:,}'.format(n_instances))
print('Number of conversions {:,}'.format(y.sum()))
print('Conversion rate: {:.2f}%'.format(p_instances*100.))
print('6.6% of the population {:,}'.format(n_targeted))
print('Expected number of conversions targetting {:,} @ {:.2f}%: {:,}'.format(n_targeted, p_instances*100., int(p_instances * n_targeted)))


# In[54]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=5, stratify=Y, shuffle=True, test_size=.2)


# In[55]:


n_instances = len(x_train)
p_instances = y_train.sum() / len(y_train)
p_targeted = .066
n_targeted = int(n_instances*p_targeted)

print('Number of instances: {:,}'.format(n_instances))
print('Number of conversions {:,}'.format(y_train.sum()))
print('Conversion rate: {:.2f}%'.format(p_instances*100.))
print('6.6% of the population {:,}'.format(n_targeted))
print('Expected number of conversions targetting {:,} @ {:.2f}%: {:,}'.format(n_targeted, p_instances*100., int(p_instances * n_targeted)))


# In[56]:


X


# In[57]:


n_targeted_test = int(len(x_test) * p_targeted)
# Get all of the instances where the previous campaign was a success
x_test_success = x_test[x_test.poutcome == 'success']

# Calcuate how many more instances we need
n_rest = n_targeted_test - len(x_test_success)


# In[58]:


rest = x_test[~(x_test.index.isin(x_test_success.index))].sample(n=n_rest, random_state=1)


# In[59]:


baseline_targets = pd.concat([x_test_success, rest], axis=0)
baseline_ys = y_test.loc[baseline_targets.index]
baseline_outcomes = baseline_ys.apply(lambda x: AVG_COST if x == 0 else AVG_COST+AVG_REVENUE)


# In[60]:


baseline_profit = sum(baseline_outcomes)

print('Number of customers targeted: {:,}/{:,}\n'.format(len(baseline_targets), len(x_test)))

print('Conversion rate under baseline policy: {:.1}%'.format(baseline_ys.sum() / len(baseline_ys)*100.))
print('Expected profit under baseline policy: ${:,}'.format(baseline_profit))


# In[63]:


X_train, X_test, Y_train, Y_test = train_test_split(data,y, random_state=5, stratify=y, shuffle=True, test_size=.2)


# In[64]:


# Create a Gradient Boosting Classifier instance
gb_classifier = GradientBoostingClassifier()

# Fit the model to your training data
gb_classifier.fit(X_train, Y_train)

# Predict probabilities for test data
probs = gb_classifier.predict_proba(X_test)

# Predict class labels for test data
preds = gb_classifier.predict(X_test)


# In[65]:


probs_df = pd.DataFrame(np.hstack([probs, Y_test.values.reshape(-1,1), preds.reshape(-1,1)]), columns=['p_no', 'p_yes', 'actual', 'predicted'])

# Sort customers by the probability that they will convert
model_targets = probs_df.sort_values('p_yes', ascending=False)

# Take the top 6.6%
model_targets = model_targets.head(n_targeted_test)

# Calculate financial outcomes
model_outcomes = model_targets.actual.apply(lambda x: AVG_COST if x == 0 else AVG_COST+AVG_REVENUE)


# In[66]:


model_profit = sum(model_outcomes)

print('Number of customers targeted: {:,}/{:,}'.format(len(model_targets), len(X_test)))
print('Conversion rate of model policy: {:.2f}%'.format(model_targets.actual.sum() / len(model_outcomes)*100.))
print('Expected profit of model policy: ${:,}'.format(model_profit))

print('Lift over baseline: {:.1f} or ${:,}'.format(model_profit / baseline_profit, model_profit - baseline_profit))


# In[ ]:




