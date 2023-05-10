#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from scipy.stats import skew
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[6]:


aapl = pd.read_excel(r'C:\Users\Esihle.Mlaka\OneDrive - MultiChoice\Desktop\Assignments\MacroTrends_Data_Apple.xlsx')
print(aapl)


# In[3]:


msft = pd.read_excel(r'C:\Users\Esihle.Mlaka\OneDrive - MultiChoice\Desktop\Assignments\MacroTrends_Data_Microsoft.xlsx')
print(msft)


# In[4]:


ssl = pd.read_excel(r'C:\Users\Esihle.Mlaka\OneDrive - MultiChoice\Desktop\Assignments\MacroTrends_Data_Sasol.xlsx')
print(ssl)


# In[5]:


#Size and Shape
print(aapl.size)
print(aapl.shape)


# In[6]:


#Size and Shape
print(msft.size)
print(msft.shape)


# In[7]:


#Size and Shape
print(ssl.size)
print(ssl.shape)


# In[8]:


#Clean Data
aapl.drop(aapl.columns[0],axis = 1, inplace =True)
aapl.head(10)


# In[9]:


#Clean Data
msft.drop(msft.columns[0],axis = 1, inplace =True)
msft.head(10)


# In[10]:


#Clean Data
ssl.drop(ssl.columns[0],axis = 1, inplace =True)
ssl.head(10)


# In[11]:


#Descriptive Statistics(Count, Mean, Standard Dev, Minimum, Maximum, 1st Quartile, 2nd Quartile, 3rd Quartile)
aapl.describe().round(2)


# In[12]:


#Descriptive Statistics(Count, Mean, Standard Dev, Minimum, Maximum, 1st Quartile, 2nd Quartile, 3rd Quartile)
msft.describe().round(2)


# In[13]:


#Descriptive Statistics(Count, Mean, Standard Dev, Minimum, Maximum, 1st Quartile, 2nd Quartile, 3rd Quartile)
ssl.describe().round(2)


# In[14]:


# Mode(aapl)
print('Mode:\n',aapl[['open','high','low','close','volume']].mode())


# In[15]:


#Range For APPLE
selected_columns = ['open','high','low','close','volume']
selected_data = aapl[selected_columns]

range_ = selected_data.max() - selected_data.min()
print('The range is:\n',range_)


# In[16]:


#Range For Microsoft
selected_columns = ['open','high','low','close','volume']
selected_data = msft[selected_columns]

range_ = selected_data.max() - selected_data.min()
print('The range is:\n',range_)


# In[17]:


#Range For Sasol
selected_columns = ['open','high','low','close','volume']
selected_data = ssl[selected_columns]

range_ = selected_data.max() - selected_data.min()
print('The range is:\n',range_)


# In[18]:


#Variances For APPLE
variance = aapl[['open','high','low','close','volume']].var().round(2)
print('Variance for :\n',variance)


# In[19]:


#Variances For MSFT
variance = msft[['open','high','low','close','volume']].var().round(2)
print('Variance for :\n',variance)


# In[20]:


#Geometric Mean
print('Geometric Mean For open:','\t',stats.gmean(aapl['open']).round(2))
print('Geometric Mean For high:','\t',stats.gmean(aapl['high']).round(2))
print('Geometric Mean For low:','\t',stats.gmean(aapl['low']).round(2))
print('Geometric Mean For close:','\t',stats.gmean(aapl['close']).round(2))
print('Geometric Mean For volume:','\t',stats.gmean(aapl['volume']).round(2))


# In[21]:


#Geometric Mean
print('Geometric Mean For open:','\t',stats.gmean(msft['open']).round(2))
print('Geometric Mean For high:','\t',stats.gmean(msft['high']).round(2))
print('Geometric Mean For low:','\t',stats.gmean(aapl['low']).round(2))
print('Geometric Mean For close:','\t',stats.gmean(aapl['close']).round(2))
print('Geometric Mean For volume:','\t',stats.gmean(aapl['volume']).round(2))


# In[22]:


#Geometric Mean
print('Geometric Mean For open:','\t',stats.gmean(ssl['open']).round(2))
print('Geometric Mean For high:','\t',stats.gmean(ssl['high']).round(2))
print('Geometric Mean For low:','\t',stats.gmean(ssl['low']).round(2))
print('Geometric Mean For close:','\t',stats.gmean(ssl['close']).round(2))
print('Geometric Mean For volume:','\t',stats.gmean(ssl['volume']).round(2))


# In[44]:


#kurtosis For Apple
from scipy.stats import kurtosis
print( '\nopen =',round(kurtosis(aapl['open']),2) )
print( '\nhigh =', round(kurtosis(aapl['high']),2))
print( '\nlow =', round(kurtosis(aapl['low']),2))
print( '\nclose =',round(kurtosis(aapl['close']),2))
print( '\nvolume =',round(kurtosis(aapl['volume']),2))


# In[45]:


#kurtosis For MSFT
from scipy.stats import kurtosis
print( '\nopen =',round(kurtosis(msft['open']),2) )
print( '\nhigh =', round(kurtosis(msft['high']),2))
print( '\nlow =', round(kurtosis(msft['low']),2))
print( '\nclose =',round(kurtosis(msft['close']),2))
print( '\nvolume =',round(kurtosis(msft['volume']),2))


# In[46]:


#kurtosis For SSL
from scipy.stats import kurtosis
print( '\nopen =',round(kurtosis(ssl['open']),2) )
print( '\nhigh =', round(kurtosis(ssl['high']),2))
print( '\nlow =', round(kurtosis(ssl['low']),2))
print( '\nclose =',round(kurtosis(ssl['close']),2))
print( '\nvolume =',round(kurtosis(ssl['volume']),2))


# In[26]:


#standard_deviation for APPLE
print('standard_deviation for Open =', aapl['open'].std().round(2))
print('standard_deviation for High =', aapl['high'].std().round(2))
print('standard_deviation for Low =', aapl['low'].std().round(2))
print('standard_deviation for Close =', aapl['close'].std().round(2))
print('standard_deviation forVolume =', aapl['volume'].std().round(2))


# In[27]:


#standard_deviation for MSFT
print('standard_deviation for Open =', msft['open'].std().round(2))
print('standard_deviation for High =', msft['high'].std().round(2))
print('standard_deviation for Low =', msft['low'].std().round(2))
print('standard_deviation for Close =', msft['close'].std().round(2))
print('standard_deviation forVolume =', msft['volume'].std().round(2))


# In[28]:


#standard_deviation for SSL
print('standard_deviation for Open =', ssl['open'].std().round(2))
print('standard_deviation for High =', ssl['high'].std().round(2))
print('standard_deviation for Low =', ssl['low'].std().round(2))
print('standard_deviation for Close =', ssl['close'].std().round(2))
print('standard_deviation forVolume =', ssl['volume'].std().round(2))


# In[29]:


#Skewness for APPLE
from scipy.stats import skew
print('Skewness for Open =',round((skew(aapl['open'], axis=0, bias=True)),4))
print('Skewness for High =',round((skew(aapl['high'], axis=0, bias=True)),4))
print('Skewness for Low =',round((skew(aapl['low'], axis=0, bias=True)),4))
print('Skewness for Close =',round((skew(aapl['close'], axis=0, bias=True)),4))
print('Skewness for Volume=',round((skew(aapl['close'], axis=0, bias=True)),4))


# In[30]:


#Skewness for MSFT
from scipy.stats import skew
print('Skewness for Open =',round((skew(msft['open'], axis=0, bias=True)),4))
print('Skewness for High =',round((skew(msft['high'], axis=0, bias=True)),4))
print('Skewness for Low =',round((skew(msft['low'], axis=0, bias=True)),4))
print('Skewness for Close =',round((skew(msft['close'], axis=0, bias=True)),4))
print('Skewness for Volume=',round((skew(msft['close'], axis=0, bias=True)),4))


# In[31]:


#Skewness for SSL
from scipy.stats import skew
print('Skewness for Open =',round((skew(ssl['open'], axis=0, bias=True)),4))
print('Skewness for High =',round((skew(ssl['high'], axis=0, bias=True)),4))
print('Skewness for Low =',round((skew(ssl['low'], axis=0, bias=True)),4))
print('Skewness for Close =',round((skew(ssl['close'], axis=0, bias=True)),4))
print('Skewness for Volume=',round((skew(ssl['close'], axis=0, bias=True)),4))


# In[32]:


#Apple
#IQR For Open
q3=np.quantile(aapl['open'],0.75)
q1=np.quantile(aapl['open'],0.25)
IQR=(q3-q1).round(2)
print('IQR for Open:',IQR)

#IQR For high
q3=np.quantile(aapl['high'],0.75)
q1=np.quantile(aapl['high'],0.25)
IQR=(q3-q1).round(2)
print('IQR for high:',IQR)

#IQR For low
q3=np.quantile(aapl['low'],0.75)
q1=np.quantile(aapl['low'],0.25)
IQR=(q3-q1).round(2)
print('IQR for low:',IQR)

#IQR For close
q3=np.quantile(aapl['close'],0.75)
q1=np.quantile(aapl['close'],0.25)
IQR=(q3-q1).round(2)
print('IQR for close:',IQR)


#IQR For volume
q3=np.quantile(aapl['volume'],0.75)
q1=np.quantile(aapl['volume'],0.25)
IQR=(q3-q1).round(2)
print('IQR for volume:',IQR)


# In[33]:


#MSFT
#IQR For Open
q3=np.quantile(msft['open'],0.75)
q1=np.quantile(msft['open'],0.25)
IQR=(q3-q1).round(2)
print('IQR for Open:',IQR)

#IQR For high
q3=np.quantile(msft['high'],0.75)
q1=np.quantile(msft['high'],0.25)
IQR=(q3-q1).round(2)
print('IQR for high:',IQR)

#IQR For low
q3=np.quantile(msft['low'],0.75)
q1=np.quantile(msft['low'],0.25)
IQR=(q3-q1).round(2)
print('IQR for low:',IQR)

#IQR For close
q3=np.quantile(msft['close'],0.75)
q1=np.quantile(msft['close'],0.25)
IQR=(q3-q1).round(2)
print('IQR for close:',IQR)


#IQR For volume
q3=np.quantile(msft['volume'],0.75)
q1=np.quantile(msft['volume'],0.25)
IQR=(q3-q1).round(2)
print('IQR for volume:',IQR)


# In[34]:


#SSL
#IQR For Open
q3=np.quantile(ssl['open'],0.75)
q1=np.quantile(ssl['open'],0.25)
IQR=(q3-q1).round(2)
print('IQR for Open:',IQR)

#IQR For high
q3=np.quantile(ssl['high'],0.75)
q1=np.quantile(ssl['high'],0.25)
IQR=(q3-q1).round(2)
print('IQR for high:',IQR)

#IQR For low
q3=np.quantile(ssl['low'],0.75)
q1=np.quantile(ssl['low'],0.25)
IQR=(q3-q1).round(2)
print('IQR for low:',IQR)

#IQR For close
q3=np.quantile(ssl['close'],0.75)
q1=np.quantile(ssl['close'],0.25)
IQR=(q3-q1).round(2)
print('IQR for close:',IQR)


#IQR For volume
q3=np.quantile(ssl['volume'],0.75)
q1=np.quantile(ssl['volume'],0.25)
IQR=(q3-q1).round(2)
print('IQR for volume:',IQR)


# In[35]:


#Correlation Apple
aapl.corr()


# In[36]:


#Correlation
msft.corr()


# In[37]:


#Correlation
ssl.corr()


# In[38]:


#Coefficient of Variation Apple
Cols=aapl[['open','high','low','close','volume']]

print((Cols.mean()/Cols.std()).round(2))


# In[39]:


#Coefficient of Variation MSFT
Cols=msft[['open','high','low','close','volume']]

print((Cols.mean()/Cols.std()).round(2))


# In[40]:


#Coefficient of Variation MSFT
Cols=ssl[['open','high','low','close','volume']]

print((Cols.mean()/Cols.std()).round(2))


# In[6]:


#Apple
actual = aapl['close']
predicted = aapl['open']

# Compute the MAE
mae = mean_absolute_error(actual, predicted)

# Print the MAE
print('Mean Absolute Error:', mae)


# In[7]:


#MSFT
actual = msft['close']
predicted = msft['open']

# Compute the MAE
mae = mean_absolute_error(actual, predicted)

# Print the MAE
print('Mean Absolute Error:', mae)


# In[8]:


#SSL
actual = ssl['close']
predicted =ssl['open']

# Compute the MAE
mae = mean_absolute_error(actual, predicted)

# Print the MAE
print('Mean Absolute Error:', mae)


# In[9]:


#APPlE
# Define the actual and predicted values
actual = aapl['close']
predicted = aapl['open']

# Compute the MSE
mse = mean_squared_error(actual, predicted)

# Print the MSE
print('Mean Squared Error:', mse)


# In[10]:


#MSFT
# Define the actual and predicted values
actual = msft['close']
predicted = msft['open']

# Compute the MSE
mse = mean_squared_error(actual, predicted)

# Print the MSE
print('Mean Squared Error:', mse)


# In[11]:


#SSL
# Define the actual and predicted values
actual = ssl['close']
predicted = ssl['open']

# Compute the MSE
mse = mean_squared_error(actual, predicted)

# Print the MSE
print('Mean Squared Error:', mse)


# In[12]:


#APPLE
actual = aapl['close']
predicted = aapl['open']

# Compute the MAPE
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Print the MAPE
print('Mean Absolute Percentage Error:', mape)


# In[13]:


actual = msft['close']
predicted = msft['open']

# Compute the MAPE
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Print the MAPE
print('Mean Absolute Percentage Error:', mape)


# In[14]:


actual = ssl['close']
predicted = ssl['open']

# Compute the MAPE
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# Print the MAPE
print('Mean Absolute Percentage Error:', mape)


# In[22]:


# Calculate covariance for Apple
covariance = np.cov(aapl['close'], aapl['volume'])
print('Covariance:\n', covariance)


# In[30]:


# Calculate covariance for Apple
covariance = np.cov(ssl['close'], ssl['volume'])
print('Covariance:\n', covariance)


# In[31]:


# Calculate covariance for Apple
covariance = np.cov(msft['close'], msft['volume'])
print('Covariance:\n', covariance)


# In[32]:


covariance = aapl['volume'].pct_change().cov(ssl['volume'].pct_change())

# Round off covariance to 2 decimal places
covariance_rounded = round(covariance, 2)

print("Covariance (rounded): ", covariance_rounded)


# In[33]:


covariance = msft['volume'].pct_change().cov(ssl['volume'].pct_change())

# Round off covariance to 2 decimal places
covariance_rounded = round(covariance, 2)

print("Covariance (rounded): ", covariance_rounded)


# In[28]:


covariance = aapl['volume'].pct_change().cov(msft['volume'].pct_change())

# Round off covariance to 2 decimal places
covariance_rounded = round(covariance, 2)

print("Covariance (rounded): ", covariance_rounded)


# In[34]:


covariance = aapl['volume'].pct_change().cov(ssl['volume'].pct_change())

# Round off covariance to 2 decimal places
covariance_rounded = round(covariance, 2)

print("Covariance (rounded): ", covariance_rounded)


# In[35]:


covariance = msft['volume'].pct_change().cov(ssl['volume'].pct_change())

# Round off covariance to 2 decimal places
covariance_rounded = round(covariance, 2)

print("Covariance (rounded): ", covariance_rounded)


# In[36]:


standard_deviation = aapl[['open','high','low','close','volume']].std()
total_Sum = aapl['open'].count()
standard_error = standard_deviation/np.sqrt(total_Sum)
print("standard_error:\n", standard_error)


# In[37]:


standard_deviation = msft[['open','high','low','close','volume']].std()
total_Sum = msft['open'].count()
standard_error = standard_deviation/np.sqrt(total_Sum)
print("standard_error:\n", standard_error)


# In[38]:


standard_deviation = ssl[['open','high','low','close','volume']].std()
total_Sum = ssl['open'].count()
standard_error = standard_deviation/np.sqrt(total_Sum)
print("standard_error:\n", standard_error)


# In[92]:


df_mode =aapl[['high']].mode().transpose()
df_mode = df_mode.reset_index(drop=True)
df_mode.drop(df_mode.columns[0],axis = 1, inplace=True)
display('Mode for high:',df_mode)

df_Lmode =aapl[['low']].mode().transpose()
df_Lmode = df_Lmode.reset_index(drop=True)
df_Lmode.drop(df_Lmode.columns[0],axis = 1, inplace=True)
display('Mode for open:',df_Lmode)

df_modeC =aapl[['close']].mode().transpose()
df_modeC = df_modeC.reset_index(drop=True)
df_modeC.drop(df_modeC.columns[0],axis = 1, inplace=True)
display('Mode for close:',df_modeC)

df_modeV =aapl[['volume']].mode().transpose()
df_modeV = df_modeV.reset_index(drop=True)
df_modeV.drop(df_modeV.columns[0],axis = 1, inplace=True)
display('Mode for volume:',df_modeV)


# In[95]:


df_mode =msft[['high']].mode().transpose()
df_mode = df_mode.reset_index(drop=True)
df_mode.drop(df_mode.columns[0],axis = 1, inplace=True)
display('Mode for high:',df_mode)

df_Lmode =msft[['low']].mode().transpose()
df_Lmode = df_Lmode.reset_index(drop=True)
df_mode.drop(df_mode.columns[0],axis = 1, inplace=True)
display('Mode for open:',df_Lmode)

df_modeC =msft[['close']].mode().transpose()
df_modeC = df_modeC.reset_index(drop=True)
df_modeC.drop(df_modeC.columns[0],axis = 1, inplace=True)
display('Mode for close:',df_modeC)


# In[94]:


df_mode =ssl[['high']].mode().transpose()
df_mode = df_mode.reset_index(drop=True)
df_mode.drop(df_mode.columns[0],axis = 1, inplace=True)
display('Mode for high:',df_mode)

df_Lmode =ssl[['low']].mode().transpose()
df_Lmode = df_Lmode.reset_index(drop=True)
df_Lmode.drop(df_Lmode.columns[0],axis = 1, inplace=True)
display('Mode for open:',df_Lmode)

df_modeC =ssl[['close']].mode().transpose()
df_modeC = df_modeC.reset_index(drop=True)
df_modeC.drop(df_modeC.columns[0],axis = 1, inplace=True)
display('Mode for close:',df_modeC)

df_modeV =ssl[['volume']].mode().transpose()
df_modeV = df_modeV.reset_index(drop=True)
df_modeV.drop(df_modeV.columns[0],axis = 1, inplace=True)
display('Mode for volume:',df_modeV)


# In[97]:


Cols2=aapl[['open','high','low','close']]
sns.boxplot(Cols2)
plt.show()


# In[98]:


#AGE Boxplot
sns.boxplot(aapl['volume'])
plt.show()


# In[100]:


Cols2=ssl[['open','high','low','close']]
sns.boxplot(Cols2)
plt.show()


# In[101]:


#AGE Boxplot
sns.boxplot(ssl['volume'])
plt.show()


# In[102]:


#AGE Boxplot
sns.boxplot(msft['volume'])
plt.show()


# In[103]:


Cols2=msft[['open','high','low','close']]
sns.boxplot(Cols2)
plt.show()


# In[188]:


f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)
sns.violinplot(aapl.iloc[:,1], color="skyblue", ax=axes[0,0])
axes[0,0].set_xlabel('open')
sns.violinplot(aapl.iloc[:,2], color="olive", ax=axes[0,1])
axes[0,1].set_xlabel('close')
sns.violinplot(aapl.iloc[:,3], color="gold", ax=axes[1,0])
axes[1,0].set_xlabel('low')
sns.violinplot(aapl.iloc[:,5], color="skyblue", ax=axes[1,1])
axes[1,1].set_xlabel('volume')
plt.show()


# In[189]:


f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)
sns.violinplot(msft.iloc[:,1], color="skyblue", ax=axes[0,0])
axes[0,0].set_xlabel('open')
sns.violinplot(msft.iloc[:,2], color="olive", ax=axes[0,1])
axes[0,1].set_xlabel('close')
sns.violinplot(msft.iloc[:,3], color="gold", ax=axes[1,0])
axes[1,0].set_xlabel('low')
sns.violinplot(msft.iloc[:,5], color="skyblue", ax=axes[1,1])
axes[1,1].set_xlabel('volume')
plt.show()


# In[190]:


f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)
sns.violinplot(ssl.iloc[:,1], color="skyblue", ax=axes[0,0])
axes[0,0].set_xlabel('open')
sns.violinplot(ssl.iloc[:,2], color="olive", ax=axes[0,1])
axes[0,1].set_xlabel('close')
sns.violinplot(ssl.iloc[:,3], color="gold", ax=axes[1,0])
axes[1,0].set_xlabel('low')
sns.violinplot(ssl.iloc[:,5], color="skyblue", ax=axes[1,1])
axes[1,1].set_xlabel('volume')
plt.show()


# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=False)
sns.violinplot(aapl.iloc[:,1], color="skyblue", ax=axes[0,0])
axes[0,0].set_xlabel('open')
sns.violinplot(aapl.iloc[:,2], color="olive", ax=axes[0,1])
axes[0,1].set_xlabel('high')
sns.violinplot(aapl.iloc[:,3], color="gold", ax=axes[1,0])
axes[1,0].set_xlabel('low')
sns.violinplot(aapl.iloc[:,4], color="teal", ax=axes[1,1])
axes[1,1].set_xlabel('close')
plt.show()


# In[168]:


# Define the columns and dataframes to plot
columns = ['open', 'open','open']
dfs = [aapl, msft, ssl]

# Create a figure with two subplots
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Plot the histograms and display the skewness in the titles
for i, col in enumerate(columns):
    skewness = dfs[i][col].skew()
    sns.histplot(dfs[i][col], ax=axs[i], kde=True)
    

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
print("Skewness for open from AAPL, MSFT and SSL, respectively.")


# In[170]:


# Define the columns and dataframes to plot
columns = ['high', 'high','high']
dfs = [aapl, msft, ssl]

# Create a figure with two subplots
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Plot the histograms and display the skewness in the titles
for i, col in enumerate(columns):
    skewness = dfs[i][col].skew()
    sns.histplot(dfs[i][col], ax=axs[i], kde=True)
    

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
print("Skewness for High from AAPL, MSFT and SSL, respectively.")


# In[172]:


# Define the columns and dataframes to plot
columns = ['close', 'close','close']
dfs = [aapl, msft, ssl]

# Create a figure with two subplots
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Plot the histograms and display the skewness in the titles
for i, col in enumerate(columns):
    skewness = dfs[i][col].skew()
    sns.histplot(dfs[i][col], ax=axs[i], kde=True)
    

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
print("Skewness for close from AAPL, MSFT and SSL, respectively.")


# In[176]:


# Define the columns and dataframes to plot
columns = ['low', 'low','low']
dfs = [aapl, msft, ssl]

# Create a figure with two subplots
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Plot the histograms and display the skewness in the titles
for i, col in enumerate(columns):
    skewness = dfs[i][col].skew()
    sns.histplot(dfs[i][col], ax=axs[i], kde=True)
    

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
print("Skewness for low from AAPL, MSFT and SSL, respectively.")


# In[175]:


# Define the columns and dataframes to plot
columns = ['volume', 'volume','volume']
dfs = [aapl, msft, ssl]

# Create a figure with two subplots
fig, axs = plt.subplots(ncols=3, figsize=(12,5))

# Plot the histograms and display the skewness in the titles
for i, col in enumerate(columns):
    skewness = dfs[i][col].skew()
    sns.histplot(dfs[i][col], ax=axs[i], kde=True)
    

# Adjust the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
print("Skewness for volume from AAPL, MSFT and SSL, respectively.")


# In[208]:


# Define the predictor and target variables
X = pd.DataFrame(aapl['open'])
y = pd.DataFrame(aapl['close'])

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients and intercept
print('Regression Coefficients:', model.coef_.round(3))
print('Intercept:', model.intercept_.round(2))
#  R-squared
print('R-squared:', model.score(X, y).round(2))

# Plot the regression line and scatter plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression For AAPL stock Market Data')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.show()


# In[209]:


# Define the predictor and target variables
X = pd.DataFrame(msft['open'])
y = pd.DataFrame(msft['close'])

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients and intercept
print('Regression Coefficients:', model.coef_.round(3))
print('Intercept:', model.intercept_.round(2))
#  R-squared
print('R-squared:', model.score(X, y).round(2))

# Plot the regression line and scatter plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression For MSFT stock Market Data')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.show()


# In[210]:


# Define the predictor and target variables
X = pd.DataFrame(ssl['open'])
y = pd.DataFrame(ssl['close'])

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients and intercept
print('Regression Coefficients:', model.coef_.round(3))
print('Intercept:', model.intercept_.round(2))
#  R-squared
print('R-squared:', model.score(X, y).round(2))

# Plot the regression line and scatter plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression For AAPL stock Market Data')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.show()


# In[214]:


plt.scatter(aapl['close'],aapl['volume'])
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.title("Scatter Plot For APPLE(Close vs.Volume)")
plt.show


# In[215]:


plt.scatter(msft['close'],aapl['volume'])
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.title("Scatter Plot For Microsoft(Close vs.Volume)")
plt.show


# In[216]:


plt.scatter(ssl['close'],aapl['volume'])
plt.xlabel('Closing Price')
plt.ylabel('Volume')
plt.title("Scatter Plot For Sasol(Close vs.Volume)")
plt.show


# In[227]:


# convert index to datetime
aapl.index = pd.to_datetime(aapl.index)

# create plot
aapl['close'].plot()

# set x-axis ticks
plt.xticks(aapl.index, rotation=5)

# display plot
plt.show()


# In[232]:


# convert daily returns to arrays
apple_returns = np.array(aapl)
msft_returns = np.array(msft)
ssl_returns = np.array(ssl)

# create empty arrays for cumulative returns
apple_cumulative_returns = np.empty(len(aapl))
msft_cumulative_returns = np.empty(len(msft))
ssl_cumulative_returns = np.empty(len(ssl))

# set initial cumulative returns to 100
apple_cumulative_return = 100
msft_cumulative_return = 100
ssl_cumulative_return = 100

# calculate cumulative returns
# calculate cumulative returns using cumprod()
apple_cumulative_returns = (1 + apple_returns).cumprod() * 100
msft_cumulative_returns = (1 + msft_returns).cumprod() * 100
portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() * 100

# plot cumulative returns
plt.plot(apple_cumulative_returns, label='AAPL')
plt.plot(msft_cumulative_returns, label='MSFT')
plt.plot(portfolio_cumulative_returns, label='Portfolio')
plt.legend()
plt.show()

# print cumulative returns
print('Apple cumulative returns:', apple_cumulative_returns)
print('Microsoft cumulative returns:', msft_cumulative_returns)
print('Sasol cumulative returns:',ssl_cumulative_returns)



# In[27]:


import pandas_datareader as pdr
import datetime


# calculate cumulative returns using cumprod()
apple_cumulative_returns = (1 + aapl['close'].pct_change()).cumprod() * 100
msft_cumulative_returns = (1 + msft['close'].pct_change()).cumprod() * 100
ssl_cumulative_returns = (1 + ssl['close'].pct_change()).cumprod() * 100


# plot cumulative returns
plt.plot(apple_cumulative_returns, label='AAPL')
plt.plot(msft_cumulative_returns, label='MSFT')
plt.plot(portfolio_cumulative_returns, label='Portfolio')
plt.legend()
plt.show()



# In[19]:


apple_cumulative_return_list = []
apple_cumulative_return = 100

for ret in aapl['close']:
    apple_cumulative_return *= (1 + ret)
    apple_cumulative_return_list.append(apple_cumulative_return)

msft_cumulative_return_list = []
msft_cumulative_return = 100

for ret in msft['close']:
    msft_cumulative_return *= (1 + ret)
    msft_cumulative_return_list.append(msft_cumulative_return)

pf_cumulative_return_list = []
pf_cumulative_return = 100

for ret in ssl['close']:
    pf_cumulative_return *= (1 + ret)
    pf_cumulative_return_list.append(pf_cumulative_return)

plt.plot(aapl.date[::-1][::-1], apple_cumulative_return_list, label="AAPL")
plt.plot(msft.date[::-1][::-1], msft_cumulative_return_list, label="MSFT", color="red")
plt.plot(ssl.date[::-1][::-1], pf_cumulative_return_list, label="Equal weighted portfolio", color="blue", linewidth=1.7)

plt.xlabel("date")
plt.ylabel("close")
plt.legend()


# In[26]:


import matplotlib.pyplot as plt
import pandas as pd

# Convert date column to year only
aapl['year'] = pd.DatetimeIndex(aapl['date']).year
msft['year'] = pd.DatetimeIndex(msft['date']).year
ssl['year'] = pd.DatetimeIndex(ssl['date']).year

apple_cumulative_return_list = []
apple_cumulative_return = 100

for ret in aapl['close']:
    apple_cumulative_return *= (1 + ret)
    apple_cumulative_return_list.append(apple_cumulative_return)

msft_cumulative_return_list = []
msft_cumulative_return = 100

for ret in msft['close']:
    msft_cumulative_return *= (1 + ret)
    msft_cumulative_return_list.append(msft_cumulative_return)

pf_cumulative_return_list = []
pf_cumulative_return = 100

for ret in ssl['close']:
    pf_cumulative_return *= (1 + ret)
    pf_cumulative_return_list.append(pf_cumulative_return)

plt.plot(aapl.year[::-1], apple_cumulative_return_list, label="AAPL")
plt.plot(msft.year[::-1], msft_cumulative_return_list, label="MSFT", color="red")
#plt.plot(ssl.year[::-1], pf_cumulative_return_list, label="Equal weighted portfolio", color="blue", linewidth=1.7)

plt.xlabel("Year")
plt.ylabel("Close")
plt.legend()


# In[38]:


# Combine the data into a single DataFrame
data = pd.concat([aapl['close'], msft['close']], axis=1)
data.columns = ['aapl', 'msft']

# Plot the data
data.plot(figsize=(10, 5))
plt.xlabel('Date')
plt.ylabel('Price (R)')
plt.title('AAPL vs MSFT Stock Prices')
plt.show()

