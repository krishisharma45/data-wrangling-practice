#Feature Selection Exercise

'''
Data Cleaning
This data does not have any null values and is relatively clean. There are three features (hour, segments, device) and a target column ('result'). There's an almost equal split between 'Conversion' and 'No Conversion' results, making this data set well balanced.
There is one column containing string values (device) that needs to be converted to integers before we implement feature selection techniques.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from numpy import loadtxt
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

campaign_data = pd.read_csv('campaign_data.csv')
campaign_data.head()
campaign_data.shape
campaign_data.isnull().values.any()
campaign_data['device'].value_counts()
campaign_data['result'].value_counts()

df = campaign_data
le = preprocessing.LabelEncoder()
le.fit(df['device'])
df['device_int'] = le.transform(df['device'])
df = df[['hour', 'segments', 'device', 'device_int', 'result']]
print(df['device_int'].isnull().any())
print(dict(zip(df['device'], df['device_int'])))
df.head()

'''
Exploratory Data Analysis
Certain hours of the day correlate with higher conversion rates, and certain devices during those hours have a higher likelihood of conversion. Notably, tablets used around 10 am were extremely successful, whereas desktops (in general) were unsuccessful.
'''


def distplot(feature):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    sns.distplot(df[feature][df['result']==1] ,color='hotpink', ax=ax)
    sns.distplot(df[feature][df['result']==0] ,color='mediumblue', ax=ax )
    plt.legend(['Conversion', 'No Conversion'])

distplot('hour')
plt.title('Campaign Conversion Highest Midday and Late Night')
plt.xlabel('Hour')

'''
We see the highest conversion rates happen during lunch time (10 am to 12 pm). Around 3 pm, there is a significant decline in conversion rates, however it increases significantly at 4 pm and very slowly tapers down from 5 pm to midnight. The chances of non-conversion outweight conversion during the hours of 11 am to 9 pm, however, which is important to note, as it means that even if we campaign more in the evening, it may not be as sucessful as campaigning aggressively at midnight or 10 am, when the probability of conversion  is higher than the probability of non-conversion.
'''

num = 10
hour_df = df[df['hour'] == num]
len(hour_df)

sns.countplot(hour_df['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(hour_df['device'], palette=("autumn"), hue=hour_df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')


'''
If we isolate the hour in which we are most successful, 10 am, then we see that although our campain reaches more smartphone users, we are most successful with tablet users. Tablet users are around 11 times more likely to convert than not convert. To understand if the potential for tapping further into this opportunity, we can explore other time slots.
'''

hour_df_two = df[df['hour'] != num]
sns.countplot(hour_df_two['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(hour_df_two['device'], palette=("autumn"), hue=hour_df_two['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')

'''
When we look at the hours that are not 10 am, we see that tablets still performly strongly (in terms of probability), however the conversion rate is lower than it was at 10 am (nearly all tablet users at 10 am converted, whereas only half converted at hours that were not 10 am. This suggests that device may be a weaker feature for conversion prediction compared to hour.
'''

num_two = 15
hour_df_three = df[df['hour'] == num_two]
sns.countplot(hour_df_three['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(hour_df_three['device'], palette=("autumn"), hue=hour_df_three['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')

'''
The worst time for conversion was 3 pm (hour 15), when non-conversion count was highest, and converstion count was lowest. Tablet still performs better in terms of conversion probability, but is much weaker at 3 am than our previous comparisons. This supports the notion that hour is stronger at predicting conversion over device.
'''

sns.countplot(df['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(df['device'], palette=("autumn"), hue=df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')

'''
In general, the success that we see overall in the tablet device is an average of a very successful hour for conversion at 10 am, when most tablet users converted. Tablet users do have a higher probability of conversion, but the device choice alone is not a good enough feature for prediction.
'''

distplot('segments')
plt.title('Campaign Conversion Highest Midday and Late Night')
plt.xlabel('Hour')

'''
From this distplot, we can see that segmentation itself does not look like a decisive factor because there are equal chances of conversoin and non conversion across all time periods. There is more activity within certain segments though (the first and last segment), and the last segment has the best conversion probability. We can drill down further to learn more about what causes this peak.
'''

df['segments'].value_counts()[:5]
df['segments'].value_counts()[-5:]

seg = 341
segment_df = df[df['segments'] == seg]
sns.countplot(segment_df['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(segment_df['device'], palette=("autumn"), hue=segment_df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')

segment_one = df[df['segments'] == 251]
sns.countplot(segment_one['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(segment_one['device'], palette=("autumn"), hue=segment_one['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')


'''
In the segments with higher conversion probability than non-conversion, we see that there is no noticeable correlation between the device used and result. This means that device may not be a meaningful feature when paired with segments.
'''

segment_df_two = df[df['segments'] == 201]
sns.countplot(segment_df_two['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(segment_df_two['device'], palette=("autumn"), hue=segment_df_two['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')


'''
With our most successful segment, segment 341, we see that there is no strong connection between the segment and device in terms of conversion rate. However, in our least successful segment, segment 201, we see that tablet users are most likely to convert. We can look at segments around it that performed similarly to verify this hypthesis.
'''

segment_df_three = df[df['segments'] == 191]
sns.countplot(segment_df_three['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(segment_df_three['device'], palette=("autumn"), hue=segment_df_three['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')

segment_df_four = df[df['segments'] == 171]
sns.countplot(segment_df_four['device'], palette=("autumn"))
plt.title('Campaign Reaches Smartphones Most')
plt.xlabel('Device')

sns.countplot(segment_df_four['device'], palette=("autumn"), hue=segment_df_four['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Device')


'''
In our least successful segments, where non-conversion was higher than conversion, we see that tablet users were much more likely to convert. This means that the device feature relies on the segment feature, and the segment feature could be important in helping us campain better.
'''

seg = 191
segment_df = df[df['segments'] == seg]
sns.countplot(segment_df['hour'], palette=("autumn"), hue=segment_df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Hour')

seg = 341
segment_df = df[df['segments'] == seg]
sns.countplot(segment_df['hour'], palette=("autumn"), hue=segment_df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Hour')

seg = 251
segment_df = df[df['segments'] == seg]
sns.countplot(segment_df['hour'], palette=("autumn"), hue=segment_df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Hour')

sns.countplot(df['segments'], palette=("autumn"), hue=df['result'])
plt.title('Tablet Users Have Highest Conversion Rate')
plt.xlabel('Segments')


'''
The patterns for different segments vary by each hour, meaning that the hour and segment features are both independent. The There are certain segments that are more successful in terms of conversion than others, and if we target specific devices in certain segments that can help us with our campaign. Targeting certain segments at specific hours can also help us encourage conversion.
'''

'''
Modeling Data
# From the exploratory data analysis, we hypothesize that time and device are the strongest features for predicting which users will convert. This is because segmentation does not seem to correlate strongly with a particular trend, and targeting specific devices at specific times (such as tablet users at 10 am) looks like a strong strategy to consider, according to our analysis.
#
# We build an ML pipeline to see which algorithm gives us the most accurate prediction for our data. From there, we can narrow down and see which features improve accuracy the most and plot feature importances to gather more support for our hypothesis.
'''

data = df[['hour', 'segments', 'device_int']]
target = df[['result']]

#Split data into 70% train, 30% validation (test) split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size= 0.3, random_state=42)
X_train.shape

#Clean the data to either replace or remove string columns
X_train.dtypes

#Build ML pipeline to perform quick analysis with classification algorithms
def quick_eval(pipeline, X_train, y_train, X_test, y_test, verbose=True):
    '''More advanced pipeline that uses DecisionTree,
    RandomForest, XGBoost, CatBoost and LightGBM as estimators'''

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    s = [train_score, test_score, train_rmse, test_rmse, train_r2, test_r2]
    s_df = pd.DataFrame(s, columns=[pipeline.named_steps['classification'].__class__.__name__],
                        index=['train_score', 'test_score', 'train_rmse', 'test_rmse',
                              'train_r2', 'test_r2'])
    return s_df

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    KNeighborsClassifier(),
    BaggingClassifier()
]

score_df = pd.DataFrame()

for c in classifiers:
    pipe = Pipeline(steps = [
        ('minmax', MinMaxScaler()),
        ('classification', c)
    ])

    result_df = quick_eval(pipe, X_train, y_train, X_test, y_test)
    score_df = pd.concat([score_df, result_df], axis=1)

score_df

'''
From our ML pipeline, we see that the tree classifiers perform best, with the DecisionTreeClassifier and ExtraTreesClassifier performing better than the Random Forest Classifier. We can dig inot the feature importance modules of tree classifiers to understand what features were most prominent in these algorithms.
'''

clf = DecisionTreeClassifier().fit(data, target)
dict(zip(data.columns, clf.feature_importances_))

'''
The Decision Tree Classifier feature importances attribute shows us that the hour and segments features are prominent in predicting conversion, whereas the device is not nearly as important.
'''

#Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators=250,
                              random_state=0)

n_features = 3
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

#Print the feature ranking
print("Feature ranking:")

for f in range(n_features):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#Plot the feature importances of the forest
plt.figure(figsize=(16,8))
plt.title("Feature importances")
plt.bar(range(n_features), importances[indices][:n_features],
       color="r", yerr=std[indices][:n_features], align="center")
plt.xticks(range(n_features), indices)
plt.xlim([-1, n_features])
plt.show()


X_train_features = [x for i,x in enumerate(X_train) if i!=3]

def plot_feature_importances(model):
    plt.figure(figsize=(10,5))
    n_features = 3
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("")
print('Random Forest')
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("")
print('Random Forest - Max Depth = 3')
print("Accuracy on training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf1.score(X_test, y_test)))

print("")
print('Random Forest Feature Importance')
plot_feature_importances(rf)


'''
The feature importance of the Random Forest Classifier supports the feature importance of the Decision Tree Classifier - hour and segments are both more important than device. This could be because hour and segments are not correlated with each other, and due to this lack of correlation, they can work better together to predict conversion, rather than with the device feature, which seems to mildly correlate with both the hour and segments feature.
'''

X = data.values
Y = target.values

model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)

'''
Feature Extraction + Modeling
# Using the results from our first round of modeling and feature importance plots, we can try to improve the accuracy of our results by picking out the hour and segments feature and running our ML pipeline again. We can also run our hypothesis (hours and device are best features). If the results improve, then we know that these features are better at predicting conversion than using all three features. If not, we can see if our exploratory data analysis and hypothesis are stronger than the feature importance plots offered by sklearn.
'''

features_hypothesis = df[['hour', 'device_int']]
#Split data into 70% train, 30% validation (test) split
X_train, X_test, y_train, y_test = train_test_split(features_hypothesis, target, test_size= 0.5, random_state=42)
X_train.shape
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    KNeighborsClassifier(),
    BaggingClassifier()
]
score_df = pd.DataFrame()

for c in classifiers:
    pipe = Pipeline(steps = [
        ('minmax', MinMaxScaler()),
        ('classification', c)
    ])

    result_df = quick_eval(pipe, X_train, y_train, X_test, y_test)
    score_df = pd.concat([score_df, result_df], axis=1)

score_df


features_observed = df[['hour', 'segments']]
#Split data into 70% train, 30% validation (test) split
X_train, X_test, y_train, y_test = train_test_split(features_observed, target, test_size= 0.5, random_state=42)
X_train.shape
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    KNeighborsClassifier(),
    BaggingClassifier()
]
score_df = pd.DataFrame()

for c in classifiers:
    pipe = Pipeline(steps = [
        ('minmax', MinMaxScaler()),
        ('classification', c)
    ])

    result_df = quick_eval(pipe, X_train, y_train, X_test, y_test)
    score_df = pd.concat([score_df, result_df], axis=1)

score_df

features_all = df[['hour', 'segments', 'device_int']]
#Split data into 70% train, 30% validation (test) split
X_train, X_test, y_train, y_test = train_test_split(features_all, target, test_size= 0.5, random_state=42)
X_train.shape
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    XGBClassifier(),
    KNeighborsClassifier(),
    BaggingClassifier()
]
score_df = pd.DataFrame()

for c in classifiers:
    pipe = Pipeline(steps = [
        ('minmax', MinMaxScaler()),
        ('classification', c)
    ])

    result_df = quick_eval(pipe, X_train, y_train, X_test, y_test)
    score_df = pd.concat([score_df, result_df], axis=1)

score_df

'''
With regards to our test, we can see that our hypothesis provided us with a better test score, but the feature importances suggestion provided us with a better training score. We can see that using all three features with a 50/50 train/test split provides us with the best accuracy. This means that we benefit from using all three features, although the hour and segments features are more important in determining conversion than if either were paired with the device feature.

In summary, we can see that the most predictive features are hour, followed by segments. We see in our exploratory data analysis that there are certain hours that correlate with a higher likelihood of conversion, and if we pair it a certain device, that likelihood increases. We also see that there are certain segments that are likely to convert and that if we target those segments at certain hours, we can increase chances of conversion. This makes the best features hour, followed second by segments.
'''
