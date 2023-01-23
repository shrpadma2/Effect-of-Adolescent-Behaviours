#%%[markdown]
# # Introduction
# For our final project, we analyzed data collected by the CDC’s Youth Risk Behavior Surveillance System. This is a survey that is conducted every 2 years by states and local school districts across the country to collect information about adolescent tendencies as it relates to their physical and educational outcomes.
#
# We hypothesize that technology use may have the potential to impede on the adolescents' well-being and lead to adverse effects on their physical health and educational performance. Furthermore, we beleive risky behaviors such as drug use can also have detrimental effects.

# # SMART Questions
# 1. How does technology and drug use relate to positive health and academic outcomes in adolescents?
# 2. Do adolescents of various races differ in their physical health and academic success?

# # Variables of Interest
# - Physical activity
#       - The amount of days with the past week that respondent was active for at least 60 minutes
# - Television use
#       - Hours of TV watched on an average school day
# - Electronics use
#       - Hours of non-school related electronic device usage (computers, smartphones, video games) on an average school day
# - Marijuana use
#       - Has smoked marijuana
# - Vape use
#       - Has used an electronic vapor product
# - Grades
#       - Description of overall grades
# - Race
#       - Race/ethnicity they classified as
# - Body Mass Index
#       - Calculated using height and weight

#%%
# package imports
import pandas as pd
import numpy as np

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
 

# stats and models
import statsmodels.api as sm
from statsmodels.formula.api import glm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,\
    roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import researchpy as rp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import mnlogit 
from sklearn.metrics import confusion_matrix
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from imblearn.over_sampling import SMOTE
import math
# Reading the Survey Data and Analysing the Dataset
#%%
raw_data = pd.read_csv('data.csv')


#Sanity Checking of the Dataset
#%%
raw_data.head()


#%%
raw_data.columns

# As We See From Above There are multiple columns in the survey 
# Data we will select only the required columns required for our analysis

#%%
raw_data.info()

#The Dataset has 311 Columns and 217340 rows. We will Now Select the columns Required

#%%
data_subset = raw_data[['year','bmi','q34','q78','q79','q80','q89','qn45','race4','race7','sex','qn24',"stweight"]]
data_subset.head()
data= data_subset[data_subset['year']>=2009]
#%%
data.head()
#%%
data.tail()
#%%
data.info()

# We will now check for the missing values in the dataset.
#%%
data = data.replace(' ', float('NaN'), regex = True)  # Replace blanks by NaN

# Lets Plot the Missing Values
#%%
missing_df = data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

#%%
ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots(figsize=(6,10))
rects = ax.barh(ind,missing_df.missing_count.values,color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.ticklabel_format(useOffset=False, style='plain', axis='x')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
#%%
data.isnull().sum()

#%%
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


city_day_missing= missing_values_table(data)
city_day_missing


#%%
import missingno as msno #install missingno using pip install missingno
msno.bar(data)


#%%
#Droping the NA values and race7 column
data = data.drop('race7', axis=1)
data.dropna(axis=0,inplace=True)
data.head()


#%%
# Converting the strings to integers
# displaying the datatypes
print(data.dtypes)
  
# converting 'Field_2' and 'Field_3' from float to int
data['q34'] = data['q34'].apply(np.int64)
data['q78'] = data['q78'].apply(np.int64)
data['q79'] = data['q79'].apply(np.int64)
data['q80'] = data['q80'].apply(np.int64)
data['q89'] = data['q89'].apply(np.int64)
data['qn45'] = data['qn45'].apply(np.int64)
data['qn24'] = data['qn24'].apply(np.int64)
data['race4'] = data['race4'].apply(np.int64)
data['sex'] = data['sex'].apply(np.int64)
data['bmi'] = data['bmi'].apply(np.float)

# displaying the datatypes
print(data.dtypes)


#%%
# Changing the column names
data=data.rename(columns={"q34": "Vape_Use", "q78": "Physical_Activity", "q79": "Television", "q80": "Electronic_Devices", "q89": "Grades", "race4": "race", "stweight":"weight","qn45":"marijuana_use","qn24":"cyber_bullied", "stweight":"weight"})
data=data.reset_index()
data
#%%
#recoding television from factors to numeric
data["Television"]=data["Television"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()
#recoding electronic devices from factors to numeric
data["Electronic_Devices"]=data["Electronic_Devices"].replace([1,2,3,4,5,6,7],[0,0.5,1,2,3,4,5])
data.head()
#recoding physical activity from factors to numeric
data["Physical_Activity"]=data["Physical_Activity"].replace([1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7])
data.head()
#%%
import missingno as msno
msno.bar(data)




#%%

# The cleaned datasets can be found here: https://github.com/tejas-rawal/DATS6103-FinalProject/tree/main/Datasets
# load the data

# analyze structure
print(f"Shape (rows, columns): {data.shape}")
print("\n\n")
print("Dataframe info:\n")
print(data.info())
print("\n\n")
print("Dataframe - first 5 rows\n")
print(data.head())

#%%[markdown]
# # Adolescent behaviors and BMI
# The goal of this analysis is to determine if there are differences between the mean BMI across race, sex, and responses for: days of physical activity, hours of watching TV, hours of electronic device usage.

#%%
# response labels
tv_answers = ['0', '< 1', '1', '2', '3', '4', '>= 5']
phys_answers = ['0', '1', '2', '3' , '4', '5', '6', '7']
electronics_answers = ['0', '< 1', '1', '2', '3', '4', '>=5']
race_groups = ['White', 'Black or African American', 'Hispanic/Latino', 'All Other Races']
sex = ['Female', 'Male']
grades = ["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"]

# helper functions
def get_unique(df: pd.DataFrame, column: str):
    """
    Returns all unique values from the column in the provided dataframe.
    :param: :df: the dataframe to filter
    :param: :column: string name of column
    :return:  the sorted unique elements of df or raises error if column does not belong to df

    """
    try:
        return np.unique(df[column])
    except Exception as err:
        raise(err)
    
#%%[markdown]
# To begin, we analyze the value counts of responses to the behaviors and labels we will utilize in our models

#%%
# Television use
sns.countplot(y=data.Television, color='#1B065E')
plt.yticks(list(range(len(tv_answers))), tv_answers)
plt.xlabel('Count')
plt.ylabel('Hours of TV watched')
plt.title('Counts for hours of television watched responses')
plt.show()

# Physical Activity
sns.countplot(y=data.Physical_Activity, color='#34E5FF')
plt.yticks(list(range(len(phys_answers))), phys_answers)
plt.xlabel('Count')
plt.ylabel('Days physically active')
plt.title('Counts for days physically active responses')
plt.show()

# Electronic device use
sns.countplot(y=data.Electronic_Devices, color='#3E8989')
plt.yticks(list(range(len(electronics_answers))), electronics_answers)
plt.xlabel('Count')
plt.ylabel('Hours of electronic device use')
plt.title('Counts for hours of electronic device usage')
plt.show()

# Race
sns.countplot(y=data.race, color='#6B0F1A',
    order=[1, 3, 2, 4])
plt.yticks(list(range(len(race_groups))), race_groups,  rotation=45)
plt.xlabel('Count')
plt.ylabel('Race')
plt.title('Counts of race responses in survey population')
plt.show()

# Sex
sns.countplot(y=data.sex, color='#F49E4C')
plt.yticks(list(range(len(sex))), sex)
plt.xlabel('Count')
plt.ylabel('Sex')
plt.title('Counts of each sex in survey population')
plt.show()

#%%[markdown]
# For the televisons hours watched survey questions, most participants responded that they watched no TV on an average school day.
#
# For the question addressing days of physcical activity within a typical school week, a majority of participants responded that they were active for at least 60 minutes on all 7 days of the week.
#
# In our survey population, the majority of respondents identified as White.
#
# There is a near 50-50 split of each sex (male, female) in the survey population.

#%%[markdown]
# #### BMI Distribution
# Let's start by examining the distribution of our target variable
#%%
sns.distplot(data.bmi, color="#60D394", bins=40,
    hist_kws=dict(edgecolor="#000000", linewidth=1),
    kde_kws=dict(linewidth=2, color="#313715"))
plt.xlabel('BMI (kg/in²)')
plt.ylabel('Density')
plt.title('Density Plot of Survey Population BMI')
plt.show()

#%%[markdown]
# The distribution of BMI within our population seems failry normal, with a slight right-skewness. This can be expalined by respondendts with unusually high BMIs shifting the distribution.

# #### BMI across hours spent watching TV
# Distribution of BMI across responses to television watching

#%%
# violin plot of BMI distribution across hours of TV watched answers
sns.violinplot(y=data.bmi, x=data.Television, alpha=0.6, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Television, palette='husl')
plt.title('BMI by hours of TV watched')
plt.xlabel('TV watched (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(7)), tv_answers)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for hours of television watched

# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_tv = get_unique(data, 'Television')
samples_by_tv = [
    data[data.Television == answer]['bmi']
        for answer in unique_by_tv
]

print("Number of samples: ", len(samples_by_tv))
print("Size of each sample: ", [len(sample) for sample in samples_by_tv])

tv_anova_result = stats.f_oneway(*samples_by_tv)
print("TV ANOVA result:\n", tv_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across TV hours watched survey answeres are equal. Our result indicates that the BMI is significantly different between adolescents who watch TV for differing amounts of time on an average school day.

#%%[markdown]
# #### BMI across electronic device usage
# Distribution of BMI across hours of electronic device usage

#%%
# violin plot
sns.violinplot(y=data.bmi, x=data.Electronic_Devices, alpha=0.6, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(electronics_answers))), electronics_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Electronic_Devices, palette='husl')
plt.title('BMI by hours of electronic device use')
plt.xlabel('Device usage (# of hours)')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(electronics_answers))), electronics_answers)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for electronic device usage
# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_device = get_unique(data, 'Electronic_Devices')
samples_by_device = [
    data[data.Electronic_Devices == answer]['bmi']
        for answer in unique_by_device
]

print("Number of samples: ", len(samples_by_device))
print("Size of each sample: ", [len(sample) for sample in samples_by_device])

device_anova_result = stats.f_oneway(*samples_by_device)
print("Electronic device ANOVA result:\n", device_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across adolescent electronic device usage are equal. Our result indicates that the BMI is significantly different between adolescents who use electronic devices for varying amounts of time. 

#%%[markdown]
# #### BMI across days of physical activity
# Distribution of BMI across responses to the physical activity question.

#%%
sns.violinplot(y=data.bmi, x=data.Physical_Activity, alpha=0.6, palette='husl')
plt.title('BMI by days physical active')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.Physical_Activity, palette='husl')
plt.title('BMI by days physical active')
plt.xlabel('Days physically active')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(phys_answers))), phys_answers)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for physical activity
# The hypothesis setup for this test looks as follows:
#
# * Hٖ₀ = The mean BMIs for each answer choice are equal
# * Hₐ = The mean BMIs is significantly different across answer choices
# * alpha = 0.5
#%%
# code for ANOVA here
unique_by_phys = get_unique(data, 'Physical_Activity')
samples_by_phys = [
    data[data.Physical_Activity == answer]['bmi']
        for answer in unique_by_phys
]

print("Number of samples: ", len(samples_by_phys))
print("Size of each sample: ", [len(sample) for sample in samples_by_phys])

phy_anova_result = stats.f_oneway(*samples_by_phys)
print("Physical activity ANOVA result:\n", phy_anova_result)

#%%[markdown]
# With a p-value close to 0, this test yields a significant result. We must reject Hٖ₀ that the mean BMI of samples across physical activity survey answers are equal. Our result indicates that the BMI is significantly different between adolescents who are active for different amount of days within the past week.

#%%[markdown]
# #### BMI across race
# Distribution of BMI by race

#%%
sns.violinplot(y=data.bmi, x=data.race, alpha=0.6, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.race, palette='husl')
plt.title('BMI by race')
plt.xlabel('Race')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(race_groups))), race_groups, rotation=45)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for race 
# The hypothesis setup for this test looks as follows:
#
# - Hٖ₀ = The mean BMI for each race is equal
# - Hₐ = The mean BMI is significantly different across race
# - alpha = 0.5
#%%
# code for ANOVA here
unique_by_race = get_unique(data, 'race')
samples_by_race = [
    data[data.race == answer]['bmi']
        for answer in unique_by_race
]

print("Number of samples: ", len(samples_by_race))
print("Size of each sample: ", [len(sample) for sample in samples_by_race])

race_anova_result = stats.f_oneway(*samples_by_race)
print("Race ANOVA result:\n", race_anova_result)

#%%[markdown]
# Our results again yield a significant result. With a p-value close to 0, we must reject Hٖ₀ that the mean BMI across race are equal. Our result indicates that the BMI is significantly different between adolescents belonging to different race groups.

#%%[markdown]
# #### BMI by sex
# Distribution of BMI by the sex of the participant.
#%%
sns.violinplot(y=data.bmi, x=data.sex, alpha=0.6, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

# boxplot of BMI vs hours of TV
sns.boxplot(y=data.bmi, x=data.sex, palette='husl')
plt.title('BMI by sex')
plt.xlabel('Sex')
plt.ylabel('BMI (kg/in²)')
plt.xticks(list(range(len(sex))), sex)
plt.show()

#%%[markdown]
# #### One-way ANOVA test for sex
# The hypothesis setup for this test looks as follows:
# 
# - Hٖ₀ = The mean BMIs for each sex are equal
# - Hₐ = The mean BMIs is significantly different between each sex
# - alpha = 0.5
#%%
# code for ANOVA here
unique_by_sex = get_unique(data, 'sex')
samples_by_sex = [
    data[data.sex == sex]['bmi']
        for sex in unique_by_sex
]

print("Number of samples: ", len(samples_by_sex))
print("Size of each sample: ", [len(sample) for sample in samples_by_sex])

sex_anova_result = stats.f_oneway(*samples_by_sex)
print("Sex ANOVA result:\n", sex_anova_result)

#%%[markdown]
# With a p-value of 0.0002, we can reject our Hٖ₀ that the mean BMI for adolescents of each sex is equal. Our result indicates that the BMI is significantly different between female and male adolescents in the survey population.

#%%[markdown]
# # Effect of behaviors on physical outcomes (Shreyas)
# We Will Now Plot the Graphs to see
# how adolescents behaviours are affecting Physical outcomes

#%%[markdown]
# For better visulaization we will 
# add random values to the Television and Electronic Device Usage

#%%
data['value1'] = np.round(np.random.uniform(0.0, 0.4, size=(len(data), 1)), 2)
data['value2'] = np.round(np.random.uniform(0.0, 0.9, size=(len(data), 1)), 2)
data['value3'] = np.round(np.random.uniform(0.0, 0.4, size=(len(data), 1)), 2)
data['value4'] = np.round(np.random.uniform(0.0, 0.9, size=(len(data), 1)), 2)

data.loc[data['Television'] < 0.5, 'RESULT1'] = 0
data.loc[data['Television'] == 0.5, 'RESULT1'] = data['Television']+data['value1']
data.loc[data['Television'] >= 1, 'RESULT1'] = data['Television']+data['value2']
data.loc[data['Electronic_Devices'] < 0.5, 'RESULT2'] = 0
data.loc[data['Electronic_Devices'] == 0.5, 'RESULT2'] = data['Electronic_Devices']+data['value3']
data.loc[data['Electronic_Devices'] >= 1, 'RESULT2'] = data['Electronic_Devices']+data['value4']

data=data.drop(['value1','value2','value3','value4'], axis = 1)

data=data.rename(columns={"RESULT1": "Television1", "RESULT2": "Electronic_Devices1"})

#%%
data.head()

#%%[markdown]
#Lets Convert the BMI and Physical Activity into Categories
#%%
data['bmi']=data['bmi'].round(decimals = 2)

#%%
data.loc[(data['bmi']) < 18.5, 'BMI_class'] = 'underweight'
data.loc[(data['bmi'] <= 24.99 ) & (data['bmi'] >= 18.5), 'BMI_class'] = 'healthy'
data.loc[(data['bmi'] <= 29.99 ) & (data['bmi'] >= 25), 'BMI_class'] = 'overweight'
data.loc[(data['bmi']) >= 30, 'BMI_class'] = 'obese'

#%%

data.loc[data['Physical_Activity'] == 0, 'PA_Class'] = 'No Activity'
data.loc[(data['Physical_Activity'] <= 2 ) & (data['Physical_Activity'] >= 1), 'PA_Class'] = 'Minimal'
data.loc[(data['Physical_Activity'] <= 5 ) & (data['Physical_Activity'] >= 3), 'PA_Class'] = 'Moderate'
data.loc[(data['Physical_Activity'] <= 7 ) & (data['Physical_Activity'] >= 6), 'PA_Class'] = 'High'

#%%

data["BMI_num"]=data["BMI_class"].replace(['overweight','healthy','underweight','obese'],[3,2,1,0])
data["PA_num"]=data["PA_Class"].replace(['High','Minimal','Moderate','No Activity'],[3,2,1,0])
data["MU_Class"]=data["marijuana_use"].replace([1,2],['yes','no'])
data.head()

#%%
data_2009=data.copy()

#%%[markdown]
# Effect of Electronic Device Usage on BMI and Physical Activity
#%%
sns.boxplot(x=data_2009["BMI_class"], y=data_2009["Electronic_Devices1"], width=0.3).set(
    xlabel='BMI', 
    ylabel='Electronic Device Usage in hours',
title='BMI vs Electronic Devices usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["BMI_class"], data_2009["Electronic_Devices"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
# We can rule out the null hypothesis that BMI and 
# the number of hours spent using electronic devices 
# are independent variables with a p-value that is close to 0. 
# There is evidence to support the relationship between BMI 
# categories and electronic device usage hours.


#%%
sns.boxplot(x=data_2009["PA_Class"], y=data_2009["Electronic_Devices1"], width=0.3).set(
    xlabel='Physical Activity', 
    ylabel='Electronic Device Usage in hours',
title='Physical Activity vs Electronic Devices usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["PA_Class"], data_2009["Electronic_Devices"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
#We can rule out the null hypothesis that Physical Activity 
# and the number of hours spent using electronic devices are 
# independent variables with a p-value that is close to 0. 
# There is evidence to support the relationship between BMI 
# categories and electronic device usage hours.

#%%[markdown]
# Effect of Television Usage on BMI and Physical Activity
#%%
sns.boxplot(x=data_2009["BMI_class"], y=data_2009["Television1"], width=0.3).set(
    xlabel='BMI', 
    ylabel='Television Usage in hours',
title='BMI vs Television usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["BMI_class"], data_2009["Television"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
#We can rule out the null hypothesis that BMI and the
#  number of hours spent watching television are independent
#  variables with a p-value that is close to 0. 
# There is evidence to support the relationship 
# between BMI categories and television viewing hours.


#%%
sns.boxplot(x=data_2009["PA_Class"], y=data_2009["Television1"], width=0.3).set(
    xlabel='Physical Activity', 
    ylabel='Television Usage in hours',
title='Physical Activity vs Television usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["PA_Class"], data_2009["Television"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
#We can rule out the null hypothesis that Physical Activity and 
# the number of hours spent using Television are independent
#  variables with a p-value that is close to 0. 
# There is evidence to support the relationship
#  between BMI categories and electronic device usage hours.
#%%[markdown]
# Effect of Marijuana Usage on BMI and Physical Activity
#%%
sns.histplot(data = data_2009, x = "BMI_class", kde = True, hue = "MU_Class").set(
    xlabel='BMI', 
title='BMI vs Marijuana usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["BMI_class"], data_2009["MU_Class"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
#We can rule out the null hypothesis that BMI 
# and the Marijuana use are independent variables with a
#  p-value that is close to 0. There is evidence to support the 
# relationship between BMI categories and Marijuana Usage.


#%%
sns.histplot(data = data_2009, x = "PA_Class", kde = True, hue = "MU_Class").set(
    xlabel='Physical Activity', 
title='Physical Activity vs Marijuana usage')
plt.show()

crosstab, test_results, expected = rp.crosstab(data_2009["PA_Class"], data_2009["MU_Class"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")


print(test_results)

#%%[markdown]
#With a p-value larger than 0.05, the null hypothesis that 
# Physical Activity and Marijuana Use are independent variables
#  cannot be ruled out. There is no proof that certain BMI 
# categories and marijuana use are related.

#%%[markdown]
# Lets Build a multinominal Logistic Regression Model
#
#
# Model 1
# Target Variable : Physical Activity
#
# Feautre Variables : Television Usage, Electronic Device Usage, Marijuana Usage

#%%
y = data_2009['PA_Class']
X = data_2009[['Television','Electronic_Devices','marijuana_use']]

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
LR_Model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', 
                           C=1.0, max_iter = 1000000, class_weight='balanced')
LR_Model.fit(X_train, y_train)
LR_Predict = LR_Model.predict(X_test)

#%%[markdown]
#Confusion Matrix

#%%
import pylab as pl

cm = confusion_matrix(y_test, LR_Predict)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nFalse Positives(FP) = ', cm[1,0]+cm[2,0]+cm[3,0])

print('\nTrue Negatives(TN) = ', cm[1,1]+cm[1,2]+cm[1,3]+cm[2,1]+cm[2,2]+cm[2,3]+cm[3,1]+cm[3,2]+cm[3,3])

print('\nFalse Negatives(FN) = ', cm[0,1]+cm[0,2]+cm[0,3])


TP = cm[0,0]
TN = cm[1,1]+cm[1,2]+cm[1,3]+cm[2,1]+cm[2,2]+cm[2,3]+cm[3,1]+cm[3,2]+cm[3,3]
FP = cm[1,0]+cm[2,0]+cm[3,0]
FN = cm[0,1]+cm[0,2]+cm[0,3]




   

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['No Activity','Minimal','Moderate','High']); ax.yaxis.set_ticklabels(['No Activity','Minimal','Moderate','High']);


#%%
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#%%[markdown]
# Model building Using stats model

#%%
modellogit = mnlogit(formula='PA_num ~ C(Television)+C(Electronic_Devices)+C(marijuana_use)', data=data_2009)
modellogitfit = modellogit.fit()
print( modellogitfit.summary() )

#%%
print(f'The model\'s pseudo r-squared value is {modellogitfit.prsquared.__round__(4)}')

# Model 2
# Target Variable : BMI
#
# Feautre Variables : Television Usage, Electronic Device Usage, Marijuana Usage

#%%
y = data_2009['BMI_class']
X = data_2009[['Television','Electronic_Devices','marijuana_use']]

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
LR_Model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', 
                           C=1.0, max_iter = 1000000, class_weight='balanced')
LR_Model.fit(X_train, y_train)
LR_Predict = LR_Model.predict(X_test)

#%%[markdown]
#Confusion Matrix

#%%


cm = confusion_matrix(y_test, LR_Predict)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nFalse Positives(FP) = ', cm[1,0]+cm[2,0]+cm[3,0])

print('\nTrue Negatives(TN) = ', cm[1,1]+cm[1,2]+cm[1,3]+cm[2,1]+cm[2,2]+cm[2,3]+cm[3,1]+cm[3,2]+cm[3,3])

print('\nFalse Negatives(FN) = ', cm[0,1]+cm[0,2]+cm[0,3])


TP = cm[0,0]
TN = cm[1,1]+cm[1,2]+cm[1,3]+cm[2,1]+cm[2,2]+cm[2,3]+cm[3,1]+cm[3,2]+cm[3,3]
FP = cm[1,0]+cm[2,0]+cm[3,0]
FN = cm[0,1]+cm[0,2]+cm[0,3]




import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['No Activity','Minimal','Moderate','High']); ax.yaxis.set_ticklabels(['No Activity','Minimal','Moderate','High']);


#%%
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#%%[markdown]
# Model building Using stats model

#%%
modellogit = mnlogit(formula='BMI_num ~ C(Television)+C(Electronic_Devices)+C(marijuana_use)', data=data_2009)
modellogitfit = modellogit.fit()
print( modellogitfit.summary() )

#%%
print(f'The model\'s pseudo r-squared value is {modellogitfit.prsquared.__round__(4)}')

#%%[markdown]
#--------------------------Grades BY Television-------------------------------#
x, y ="Television", "Grades"

df1=data.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of Television on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    
plt.show()    
#%%[markdown]

#--------------------------Grades BY Electronic_Devices-------------------------------#
x, y ="Electronic_Devices", "Grades"

df1=data.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of Electronic Devices on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    
plt.show()
#%%[markdown]
#--------------------------Grades BY marijuana_use-------------------------------#
x, y ="marijuana_use", "Grades"

df1=data.groupby(x)[y].value_counts(normalize=True)
df1=df1.mul(100)
df1=df1.rename('percent').reset_index()

g=sns.catplot(x=x, y='percent', hue=y, kind="bar", data=df1)
g.ax.set_ylim(0,100)
g.fig.set_size_inches(20,10)
plt.title('Effect of marijuana_use on Adolescent Grades')

for p in g.ax.patches:
    txt=str(p.get_height().round(1))+"%"
    txt_x=p.get_x()
    txt_y=p.get_height()
    
    g.ax.text(txt_x,txt_y,txt)
    
plt.show()

#%%
data.head()
#%%[markdown]
#--------------------Synthetic Minority Oversampling Technique (SMOTE)---------------------#

X=data[['race','Electronic_Devices','Television','marijuana_use']]
#dependent_variable
y=data[['Grades']]

os=SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

columns=X_train.columns
smote_data_X, smote_data_y=os.fit_resample(X_train, y_train)
smote_data_X=pd.DataFrame(data=smote_data_X, columns=columns)
smote_data_y=pd.DataFrame(data=smote_data_y, columns=['Grades'])

'''
Now, as not all the feature will be contributing towards the prediction, 
we need to first figure out the most important features that will contribute towards the prediction. 
This is very crucial to improve the efficiency of the model.
'''
feature_names=smote_data_X.columns.to_list()
model=LogisticRegression(random_state=0).fit(data[feature_names].values, data['Grades'].values)
#Get the score
score=model.score(data[feature_names].values, data['Grades'].values)
print("Logistic Regression score is:",score)


w0=model.intercept_[0]
w=model.coef_[0]

feature_importance=pd.DataFrame(feature_names, columns=['feature'])
feature_importance['importance']=pow(math.e,w)
feature_importance=feature_importance.sort_values(by=['importance'], ascending=False)
feature_importance=feature_importance[:10].sort_values(by=['importance'], ascending=False)

ax=feature_importance[:10].sort_values(by=['importance'], ascending=True).plot.barh(x="feature", y="importance")
plt.title('Important Feature')
plt.savefig('feature.png')
# print(feature_importance)

#%%[markdown]
#Electronic Devices,Television, Marijuana Use,Race – Decision Tree Classifier
#MAke a list of all feature

feature_importance_list=feature_importance['feature'].to_list()

X=smote_data_X[feature_importance_list]
y=smote_data_y['Grades']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=0)


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

# Check the model performance with the training data
predictions_dt = model_dt.predict(X_train)
print("DecisionTreeClassifier", accuracy_score(y_train, predictions_dt))

predictions_dt = model_dt.predict(X_test)
print("DecisionTreeClassifier TEST: ", accuracy_score(y_test, predictions_dt))

#%%[markdown]
#Confusion Matrix
conf_matrix =confusion_matrix(y_test, predictions_dt)
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["A","B","C","D","F","NG","NS"]); 
ax.yaxis.set_ticklabels(["A","B","C","D","F","NG","NS"]);


#%%
# recoding race from numeric to categorical 
data["race"]=data["race"].replace([1,2,3,4],["White","Black or African American","Hispanic/Latino","All Other Races"])
#recoding grades from numeric to categorical
data["Grades"]=data["Grades"].replace([1,2,3,4,5,6,7],["Mostly A's","Mostly B's","Mostly C's","Mostly D's","Mostly F's","None of these grades","Not sure"])
#%%[markdown]
# #### Contingency Table of Race and Grades and Chi-Squared Test of Independence
contigency = pd.crosstab(index=data['race'], columns=data['Grades'], margins=True, margins_name="Total")
plt.figure(figsize=(9,4))
sns.heatmap(contigency, annot=True, cmap="Blues", vmin= 40, vmax=36000,fmt='g')
plt.title("Contingency Table of Race and Grades")
plt.ylabel("Race")
#chi-squared test of independence
stat, p, dof, expected = stats.chi2_contingency(contigency)
#checking the significance
alpha = 0.05
print("The results of the chi-squared test of independence showed that the p value is " + str(p) + " which indicates a significant dependent relationship between race and grades.")
#%%[markdown]
#The contingency table between racial groups and their grades reveals that a majority of individuals, regardless of race, report having mostly A's and B's for their grades. A majority of white individuals and individuals of other races have mostly A's while a majority of Black/African American and Hispanic/Latino students report having mostly B's. 


#%%[markdown]
# # Adolescent behaviors and vape use classification (Carrie)
#%%
# recoding variables for tables 
data["Vape_Use"]=data["Vape_Use"].replace([1,2],["Yes","No"])
vape_yes = data[data["Vape_Use"]=="Yes"]
vape_no = data[data["Vape_Use"]=="No"]

#%%[markdown]
#Barplot of Vape Use in Adolescents
ax = sns.countplot(x=data["Vape_Use"],data=data,palette='husl')
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Vape Use" , size = 12 )
plt.ylabel( "Frequency" , size = 12 )
plt.title("Distribution of Vape Use")

#%%[markdown]
#This figure shows the proportion of individuals in the sample who do and do not use electronic vapor products. More specifically, there are 17,102 individuals who do not engage in vaping and 14,482 who do engage in vaping which makes about a 2,620 person difference. 
#%%[markdown]
# #### Vape Use by Racial Groups
plt.figure(figsize=(9,4))
ax = sns.countplot(x=data["Vape_Use"],hue="race",data=data,palette='husl')
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel( "Vape Use" , size = 12 )
plt.ylabel( "Frequency" , size = 12 ) 
plt.title("Distribution of Vape Use by Race")
plt.legend(loc='upper center')
#%%[markdown]
# In relation to race and vaping habits, there is a pretty similar distribution between races in terms of individuals that vape and do not vape. More specifically, there is less than a 200 person difference between Hispanic/Latino individuals who vape and do not vape. There is about a 1000 person difference between White individuals, Black/African American individuals, and individuals of all other races who vape and do vape.
#%%[markdown]
# #### Hours of Watching Television per Day by Vape Use
ax = sns.countplot(x="Television",hue="Vape_Use", data=data,palette='husl')
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bar Chart of Hours Watching Television per Day by Vape Use")
plt.ylabel( "Frequency" , size = 12 )
plt.xlabel( "Time Watching Television (Hours)" , size = 12 )
ts = stats.ttest_ind(a=vape_yes["Television"], b=vape_no["Television"], equal_var=True)
print("Two-Sample T-test:",ts)

#%%[markdown]
# The plot shows the differences in hours of television watched per day by individuals who do and do not vape. Interestingly, between the hours of 0.0 and 2.0 there are many more individuals who report not vaping. In comparison, between the hours of 3.0 and 5.0 it is apparent that a greater proportion of individuals report vaping. It is important to note the pattern that the more hours of television watched in the day, the more the individuals report vaping in comparison to not vaping. These results may imply a relationship between number of hours of television per day and vaping habits considering that the gap between those who vape and those who do not vape becomes smaller and smaller with every extra hour of television watched per day. In addition, after running a two-sample t-test between those who and those who do not vape, the results indicate that there is a significant difference in the average number of hours of television watch per day between groups (p<0.05).
#%%[markdown]
# #### Hours of Electronic Device Use per Day by Vape Use
ax = sns.countplot(x="Electronic_Devices",hue="Vape_Use", data=data,palette='husl')
for container in ax.containers:
    ax.bar_label(container)
plt.title("Bar Chart of Hours on Electronic Devices per Day by Vape Use")
plt.ylabel( "Frequency" , size = 12 )
plt.xlabel( "Time on Electronic Devices (Hours)" , size = 12 )
plt.legend(loc='center right')
s = stats.ttest_ind(a=vape_yes["Electronic_Devices"], b=vape_no["Electronic_Devices"], equal_var=True)
print("Two-Sample T-test:",s)

#%%[markdown]
# The plot shows mixed results with the two ends of the hour distribution having the smallest differences between those who do and do not report vaping.The largest difference between groups clusters at 2.0 hours. Overall, there is no definite trend in this graph depicting differences in time spent on electronic devices per day between vaping and non-vaping individuals. This conclusion is further supported by the insignificant (p>0.05) t-test which indicates that there is no significant difference in the average number of hours spent on electronic devices between the vaping and non-vaping groups.

#%%[markdown]
# #### Contingency Table of Marijuana and Vape Use and Chi-Squared Test of Independence
data["marijuana_use"]=data["marijuana_use"].replace([1,0],["Yes","No"])
data["Vape_Use"]=data["Vape_Use"].replace([0,1],["No","Yes"])
#creating a contingency table for race and grades
contigency1 = pd.crosstab(index=data['marijuana_use'], columns=data['Vape_Use'], margins=True, margins_name="Total")
plt.figure(figsize=(9,4))
sns.heatmap(contigency1, annot=True, cmap="Blues", vmin= 40, vmax=36000,fmt='g')
plt.title("Contingency Table of Marijuana Use and Vape Use")
plt.xlabel('Vape Use')
plt.ylabel('Marijuana Use')

stat, p, dof, expected = stats.chi2_contingency(contigency1)
#checking the significance
alpha = 0.05
print("The results of the chi-squared test of independence showed that the p value is " + str(p) + " which indicates a significant dependent relationship between marijuana use and e-cig use.")
#%%[markdown]
#The contingency table between marijuana and vape usage for shows that about half of the sample neither vape or smoke marijuana (~46%) while about 30% of the sample did report vaping and smoking marijuana. A very small percentage of the sample either smoke marijuana or vape but do not engage in both. In addition, our chi-squared test of independence indicaed a significant dependent relationship between marijuana and vape use. 
#%%
#recoding race from numeric to categorical
data["race"]=data["race"].replace(["White","Black or African American","Hispanic/Latino","All Other Races"],[0,1,2,3])
data["Vape_Use"]=data["Vape_Use"].replace(["No","Yes"],[0,1])
data["marijuana_use"]=data["marijuana_use"].replace(["No","Yes"],[0,1])

#%%
#recoding race from numeric to categorical
#data["race"]=data["race"].replace(["White","Black or African American","Hispanic/Latino","All Other Races"],[0,1,2,3])
# recording vape use to numeric
#data["Vape_Use"]=data["Vape_Use"].replace(["No","Yes"],[0,1])
# recoding marijuana use to numeric
data["marijuana_use"]=data["marijuana_use"].replace([1,2],[1,0])
#%%
#splitting data for logit regression
xdata = data[["Television","Electronic_Devices",'marijuana_use',"race"]]
ydata = data[["Vape_Use"]]
features = ["Television","Electronic_Devices", 'marijuana_use','race']
#%%[markdown]
# #### Logistic Regression Model of Vape Use
#%% 
# running logistic regression and splitting into training and testing data 
model = glm(formula="Vape_Use ~ Television + Electronic_Devices + C(marijuana_use)+ C(race)",data=data, family=sm.families.Binomial())
model = model.fit()
print(model.summary())

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=1)

logit = LogisticRegression()  # instantiate
logit.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', logit.score(x_test, y_test))
print('Logit model accuracy (with the train set):',logit.score(x_train, y_train))

#creating prediction function to test test data  with model 
y_pred = logit.predict(x_test)
#%%[markdown]
#After converting the coefficients to their exponential form, it can be said that in comparison to individuals that do not smoke marijuana, the odds-ratio of using vape products is multiplied by 12.7 for individuals who do smoke marijuana, when holding all other predictors constant. Also, holding all other predictors constant, for African American individuals the odds-ratio of using vape products is multiplied by 2.46, multiplied by 1.19 for Hispanic/Lation individuals, and multiplied by 1.35 for individuals of all other races in comparison to White individuals. Lastly, for every hour increase in television use per day, the odds-ratio of using vape products is multiplied by 1.03. Furthermore, the relationship between electronic device usage and vape use was not signifcant. 

#%%[markdown]
# #### Classification Report and Confusion Matrix of Logistic Regression Predicting Vape Use
y_true, y_pred = y_test, logit.predict(x_test)
print("Classification Report:",end='\n')
print(classification_report(y_true, y_pred))

c = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",end='\n')
ax = sns.heatmap(c, annot=True,fmt='g')
ax.xaxis.set_label_position('top')
ax.set_xticklabels(['Positive', 'Negative'])
ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
ax.add_patch(plt.Rectangle((0, 1), 1, 0.1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
ax.add_patch(plt.Rectangle((0, 0), -0.1, 1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
plt.tight_layout()
plt.title("Confusion Matrix")

#%%[markdown]
# According to the classification report of our logistic regression model, out of all adolescents that the model predicted would use vape products, only about 79% actually do use vape products. Out of all the adolescents that actually do vape, the model only predicted this outcome correctly for 66% of those adolescents. Since the F1-Score is somewhat close to 1, we can assume that the model does an good job of predicting whether or not adolescents will use vape products. The overall accuracy of the model was 77% which is a good sign that the model is efficient at classifying between adolescents who vape and who do not vape.
#%%[markdown]
# #### ROC-AUC of Logistic Regression Model
ns_probs = [0 for _ in range(len(y_test))]
lr_probs = logit.predict_proba(x_test)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic - Logisitc Regression")
plt.legend()
plt.show()
#%%[markdown]
# The ROC curve ad AUC score help us to understand separability. More specifically, it tells use how much our model is capable of distinguishing between adolescents who do and do not vape. The ROC curve trends somewhat to the upper left corner which indicates a pretty good model. The AUC score is about 0.78 which is acceptable but it would be preferred to be closer to 0.8. Overall, the ROC curve and AUC score indicate that our logistic regression does a good job at discriminating between classes. 

#%%[markdown]
# #### Decision Tree Classifier of Vape Use
x_train1, x_test1, y_train1, y_test1 = train_test_split(xdata, ydata, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier(class_weight='balanced',max_depth=3)
#creating prediction function to test test data  with model 
y_pred1 = clf.fit(x_train1, y_train1).predict(x_test1)

print("Accuracy of Decision Tree Classifier is:", accuracy_score(y_test1, y_pred1))

y_true1, y_pred1 = y_test1, clf.predict(x_test1)
print("Classification Report:",end='\n')
print(classification_report(y_true1, y_pred1))

cm = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cm, annot=True,fmt='g')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted')
ax.tick_params(length=0, labeltop=True, labelbottom=False)
ax.xaxis.set_label_position('top')
ax.set_xticklabels(['Positive', 'Negative'])
ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
ax.add_patch(plt.Rectangle((0, 1), 1, 0.1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
ax.add_patch(plt.Rectangle((0, 0), -0.1, 1, color='white', clip_on=False, zorder=0, transform=ax.transAxes))
plt.tight_layout()
plt.title("Confusion Matrix")
plt.show()

#%%[markdown]
# Similar to our logistic regression model classifying vape use, our decision tree predicted that out of all adolescents that the model predicted would use vape products, only about 79% actually do use vape products. Out of all the adolescents that actually do vape, the model only predicted this outcome correctly for 66% of those adolescents. Since the F1-Score is somewhat close to 1, we can assume that the model does an good job of predicting whether or not adolescents will use vape products. The overall accuracy of the model was 77% which is a good sign that the model is efficient at classifying between adolescents who vape and who do not vape.
#%%[markdown]
# #### ROC-AUC of Decision Tree Classifier Model
ns_probs = [0 for _ in range(len(y_test))]
lr_probs = clf.predict_proba(x_test)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('DecisionTree: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='DecisionTree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic - DecisionTree ")
plt.legend()
plt.show()


#%%[markdown]
# The ROC curve ad AUC score tells use how much our model is capable of distinguishing between adolescents who do and do not vape. Like the logistic regression model, the ROC curve trends somewhat to the upper left corner which indicates a pretty good model. The AUC score is about 0.78 which is acceptable but it would be preferred to be closer to 0.8. Overall, the ROC curve and AUC score indicate that our logistic regression does a good job at discriminating between classes of vaping and not vaping. 
#%%[markdown]
# # Conclusion
# We received statistically significant results from performing ANOVA and Chi-squared tests agains the relationships between our variables of interest.
# This allowed us to continue our pursuit of studying the effects of different behaviors on health and education outcomes for adolescent youth by developing classification models.
#
# The models we built to classify health outcomes and grades did not return a significantly high accuracy. We believe this was due to weak correlation amongst our predictors.
#
# The classification models we built to predict adolescent vape use did result in a significantly high accuracy.
# 
# Some of our challenges were:
# - Cleaning a relatively large dataset with many missing values for responses from earlier years.
# - We had a large number of behaviors to choose from which can be a good and bad problem.
# 
# 
# Overall, modeling adolescent social behaviors and outcomes is a complicated task. We could potentially include responses from other behavioral questions in the survey to improve the accuracy of our model, but that will require deeper analysis of each new variable we decide to use.

#%%