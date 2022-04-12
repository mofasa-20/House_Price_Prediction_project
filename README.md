# House_Price_Prediction_project

**Introduction**
The real estate sector is an important industry with many stakeholders ranging from regulatory bodies to private companies and investors. Among these stakeholders, there is a high demand for a better understanding of the industry operational mechanism and driving factors.
Today there is a large amount of data available on relevant statistics as well as on additional contextual factors, and it is natural to try to make use of these in order to improve our understanding of the industry. Notably, this has been done in Zillow's Zestimate [4] and Kaggle's competitions on housing prices [2].

In some cases, non-traditional variables have proved to be useful predictors of real estate trends. For example, in [3] it is observed that Seattle apartments close to specialty food stores such as Whole Foods experienced a higher increase in value than average.

This project can be considered as a further step towards more evidence-based decision making for the benefit of these stakeholders. The project focused on assessment value for sales price based on data. The objective of this project is to build a predictive model for change in house prices based on certain dependent variables.

The data belong to Ames, Ames (/eɪmz/) is a city in Story County, Iowa, United States, located approximately 30 miles (48 km) north of Des Moines in central Iowa. It is best known as the home of Iowa State University (ISU), with leading agriculture, design, engineering, and veterinary medicine colleges. A United States Department of Energy national laboratory, Ames Laboratory, is located on the ISU campus.

In 2020, Ames had a population of 66,427, making it the state's ninth largest city. Iowa State University was home to 33,391 students as of fall 2019, which make up approximately one half of the city's population.

![image](https://user-images.githubusercontent.com/79675883/162915985-4d9013a0-4d7f-4518-8886-51e3a62abb67.png)

Output Label Analysis


```
**checking the distribution plot of the sales price**
plt.figure(figsize=(16, 6))
sns.set_style("whitegrid")*
output_plot = sns.distplot(df['SalePrice'], color ="red")
plt.title("Housing Data Sale price of houses (output) analysis", fontweight="bold", fontsize=18)
plt.xlabel('Sales Price',fontweight="bold",fontsize=14)
plt.ylabel("Frequncy of Sales",fontweight="bold",fontsize=14)
plt.show()
```

![image](https://user-images.githubusercontent.com/79675883/162913744-13cad444-7cb8-4bcf-8045-a52e36852c1c.png)

```
print("Skewness: %f" %df['SalePrice'].skew())
print("Kurtosis: %f" %df['SalePrice'].kurt())
Skewness: 1.882876
Kurtosis: 6.536282
```
The output column is a right skewed data with an elongated tail.

Data Cleaning - null value analysis
```
def null_df(df):
    """
    This function geenrates all the null values counts in a dataframe from a given dataframe.
    """
    count = df.isnull().sum() #getting missing value counts
    percnt = 100 * ((df.isnull().sum()) / (df.shape[0])) #getting missing value percentage
    #creating dataframe and renaming column
    tb = pd.concat([count, percnt], axis=1).rename(columns = {0:"count", 1:"percnt"})
    tb = tb[tb['count'] != 0]
    return tb.sort_values("count", ascending=False).round(3)
    print(f"missing df shape : {null_df(df).shape}")
null_df(df).T
```
**Outlier Detection**
```
figure = plt.figure(figsize=(15,12))
for idx, col in enumerate(numerical):
    plt.subplot(4, 5, idx+1)
    plot = sns.boxplot(df1[col])
    plt.ylabel(col)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162933938-fe36107d-e87e-4c41-9c0f-3f79dc4355ef.png)

from above analysis we can say that most of the features have outliers.

The features like PoolArea and Porch has majority values as 0. This is because these houses doesn't have those facilities, so it is given as zero.

```
## checking neighbourhood with number of overall quality of the house
f, ax = plt.subplots(figsize = (16,8))
#df_temp = df.sort_values(["YrSold", "SalePrice"])
df_temp1 = df.groupby("YrSold")["SalePrice"].count().reset_index()
plt.bar(df_temp1["YrSold"], df_temp1["SalePrice"], color = "red")
plt.title("House Sold between 2006 - 2010",fontweight='bold', fontsize=16)
plt.xlabel("Sold Year")
plt.ylabel("SalePrice")
plt.xticks(rotation = 90)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162934164-d756c1f5-6888-47d8-8015-39b4c7a752df.png)

```
## checking the distribution plot of the sales price
plt.figure(figsize=(16, 6))
sns.set_style("whitegrid")
output_plot = sns.distplot(df['YrSold'], color ="red")
plt.title("Houses Sold Between Year 2006 - 2010", fontweight="bold", fontsize=18)
plt.xlabel('Sold Year',fontweight="bold",fontsize=14)
plt.ylabel("Frequncy of Sales",fontweight="bold",fontsize=14)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162934261-2755784a-f646-41ab-89cc-691781b8a433.png)

```
corr_age_pace = df[["YrSold", "SalePrice"]].dropna().corr()
plot = sns.heatmap(corr_age_pace, square = True, linewidths = .5, annot = True, cmap = 'Set3') 
```
![image](https://user-images.githubusercontent.com/79675883/162934383-ec41d9cf-b289-4a72-9097-c455fda8c68e.png)

```
import plotly.express as px
fig = px.scatter(df1, x='SalePrice', y='YrSold', size='SalePrice',color='YrSold', width=1000, height=600)
fig.update_layout(showlegend=False, title={'text':'Houses Sold Between Year 2006 - 2010','x':0.5,'xanchor':'center'})
fig.show()
```
![image](https://user-images.githubusercontent.com/79675883/162934482-c05e244e-5d54-4458-ad54-5434af7d8bd9.png)

```
#scatter plot TotalBsmtSF/saleprice
sns.set_style("whitegrid")
var = "TotalBsmtSF"
data = pd.concat([df1[var], df1["SalePrice"]], axis = 1)
data.plot.scatter(x = var, y = "SalePrice", ylim = (0,800000))
plt.title("Housing Data",fontweight="bold", fontsize=18)
plt.xlabel('Total square feet of basement area',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
plt.gcf().set_size_inches((12, 8)) 
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162934554-a91254ed-7a41-48f9-9f53-53495276745a.png)

```
## checking outliers in Ground living area column with price value as this is one of the important features 
sns.set_style("whitegrid")
var = "GrLivArea"
data = pd.concat([df1[var], df1["SalePrice"]], axis = 1)
data.plot.scatter(x = var, y = "SalePrice", ylim = (0,800000))
plt.title("Housing Data",fontweight="bold", fontsize=18)
plt.xlabel('Ground Living Area',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
plt.gcf().set_size_inches((12, 8)) 
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162934698-58fc1db7-a7d5-4710-9b07-8bfe01ca1e33.png)


**Box Plot Analysis for categorical data**
```
#box plot overallqual/saleprice
var = "OverallQual"
data = pd.concat([df1[var], df1["SalePrice"]], axis = 1)
f, ax = plt.subplots(figsize = (12,10))
fig = sns.boxplot(x = var, y = "SalePrice", data=data)
fig.axis(ymin = 0, ymax = 800000);
plt.title("Overall quality of the house impacting the sale price",fontweight="bold", fontsize=18)
plt.xlabel('Over All Qual',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
```
![image](https://user-images.githubusercontent.com/79675883/162934987-05e38372-2a4a-4c8a-97ec-7589e3b49e4b.png)

```
#box plot overallqual/saleprice
var = "OverallCond"
data = pd.concat([df1[var], df1["SalePrice"]], axis = 1)
f, ax = plt.subplots(figsize = (14,8))
fig = sns.boxplot(x = var, y = "SalePrice", data=data)
fig.axis(ymin = 0, ymax = 800000);
plt.title("Overall condition of the house impacting the sale price",fontweight="bold", fontsize=18)
plt.xlabel('Over All Condition',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
```
![image](https://user-images.githubusercontent.com/79675883/162935123-2be99274-a681-4184-9631-2c937333e4ad.png)

```
## creating bin for year built
f, ax = plt.subplots(figsize = (14,8))
df_temp = df2.copy()
df_temp['YearBuilt_bin'] = pd.cut(df_temp['YearBuilt'], bins=[1850,1900, 1925, 1950, 1975, 2000, 2021], 
                              labels=['pre_1900', '1900_1925', '1925_1950', 
                                      '1950_1975', '1975_2000', 'post_2000'])

sns.violinplot(x="YearBuilt_bin", y="SalePrice", data=df_temp)

plt.title("house year building impacting the sale price", fontweight="bold", fontsize=18)
plt.xlabel('Year Built',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
plt.xticks(rotation = 90)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162935259-b070475f-6b2b-4d34-a0c7-8f5bce09a89f.png)

```
df_temp = df2.sort_values(["SalePrice", "Neighborhood"])
plt.figure(figsize=(16,8))
sns.boxplot(x="Neighborhood", y="SalePrice", data=df_temp)
plt.xticks(rotation = 90)
plt.title("Neighbourhood impacting the sale price", fontweight="bold", fontsize=15)
plt.xlabel('Neighbourhood',fontweight="bold",fontsize=14)
plt.ylabel("Sales Price",fontweight="bold",fontsize=14)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162935381-be1da102-7215-43f5-872c-d2131232ee75.png)

```
## checking neighbourhood with number of overall quality of the house
f, ax = plt.subplots(figsize = (16,8))
df_temp1 = df_temp.groupby("Neighborhood")["OverallQual"].count().reset_index()
plt.bar(df_temp1["Neighborhood"], df_temp1["OverallQual"], color = "red")
plt.title("total house present in each neighbourhood", fontsize=15)
plt.xlabel("neighbourhood")
plt.ylabel("count")
plt.xticks(rotation = 90)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162935502-21381094-edc5-4c3d-b294-ca5c71abd118.png)

**saleCondition**
```
piedf=pd.DataFrame(df2["SaleCondition"].value_counts())
piedf["Conditions"] = piedf.index
fig = px.pie(piedf, values="SaleCondition", names='Conditions', title ='sale condition during the sale of the house' )
fig.show()
```
![image](https://user-images.githubusercontent.com/79675883/162935847-67ae8302-2e06-4605-a901-6e4427dff8dc.png)
**very less percentage of houses were sold with an alloca and adjland way. partial and Abnormal way of sale consists of 12%.**

**numerical data analysis**
![image](https://user-images.githubusercontent.com/79675883/162936145-625609dc-75f6-4d8d-8e4f-3d3540113492.png)
![image](https://user-images.githubusercontent.com/79675883/162936237-20971394-4dd7-499e-bd03-14dd126cb7d4.png)

**Correlation**
```
fig = plt.figure(figsize=(20,20)) ## setting up figure size

## heatmap for correation analysis
sns.heatmap(df2.iloc[:,1:-1].corr(), square = True, 
            linewidths = .5, cbar_kws = {"shrink":0.8}, 
            annot = False) 
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162936543-2eef8928-7d3f-4f1d-84b3-1093503fe304.png)

```
corrmat = df2.corr()
f, ax = plt.subplots(figsize=(12,10))
k = 10
cols = corrmat.nlargest(k, 'SalePrice')["SalePrice"].index
cm = np.corrcoef(df2[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar = True, square=True, fmt='.2f',annot = True, annot_kws={'size' : 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162937065-abc52e24-026d-4fa1-8ca6-f81f7069dfe6.png)

```
## source code : - https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
    ```
```
corr_data = get_top_abs_correlations(df2.iloc[:,1:-1].corr(), n=16)
corr_data
```
![image](https://user-images.githubusercontent.com/79675883/162937776-6657ca38-746c-40a9-b12f-49c96177eccf.png)

```
##checking correation with output data
plot = df3.iloc[:,1:-1].corrwith(df3["SalePrice"]).plot.barh(figsize =(10,10), fontsize=12)
plt.title('Correlation with Target', fontsize = 20)
plt.show()
```
![image](https://user-images.githubusercontent.com/79675883/162938545-d331bbbb-1ceb-4772-9596-0ce25f0239e8.png)



    



