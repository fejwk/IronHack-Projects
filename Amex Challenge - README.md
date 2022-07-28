# Amex Predictive Model Challenge

## Project goal

This project is to be presented as the Final Paper of Ironhack's Data Analytics course, in which the knowledge and learnings were applied to the best of my ability to come up with an answer for this case.

The goal of this application is to predict the probability that a customer does not pay their credit card balance amount in the future. A predictive model was built using Amex's customer database. 

## Database

This study was performed by using the company's own database and it contains information about the customers' ID and features divided in categories, such as: 
- D_* = Delinquency variables
- S_* = Spend variables
- P_* = Payment variables
- B_* = Balance variables
- R_* = Risk variables

Features have been anonymized and all the data is already normalized.

All the descriptions are available in the following link: https://www.kaggle.com/competitions/amex-default-prediction/

Files 
- **train_data.csv** - training data with features per customer_ID
- **train_labels.csv** - target label for each customer_ID
- **test_data.csv** - corresponding test data; the objective is to predict the target label for each customer_ID
- **sample_submission.csv** - a sample submission file in the correct format

`Target = 1` meaning the the customer can be a future payment default.

## Methodologies

For this project the following libraries were used:
- pandas and numpy
- matplotlib.pyplot
- seaborn
- sklearn models and metrics
- dask dataframes
- pycaret classification
- joblib

## Reading data

Due to the size of the databases provided by Amex, pandas couldn't read the files. Therefore, we had to resort to Dask library in order to access and manipulate the data.
Dask is a Python parallel computing library that helps in scalling codes and machine learning workflows. It enables you to read large datasets by dividing it into partitions and also offers familiar user interface by mirroring other APIs including: pandas, scikit-learn and numpy

`amex_train_data = dd.read_csv('amex-default-prediction/train_data.csv')`

`amex_labels = dd.read_csv('amex-default-prediction/train_labels.csv')`

`amex_test_data = dd.read_csv('amex-default-prediction/test_data.csv')`

## First baseline model

### Data manipulation

#### The same treatment was made to train and test databases.

Since we had multiple statements for each customer, the first treatment was to group the numeric features by customer_ID using the mean as the aggregator.

`amex_train_unique_customers = amex_train_data.groupby(amex_train_data.customer_ID).mean().compute().reset_index()`

It was also perceived that our databases had many null values with some of the features having less than 10% of information filled.

In order to run the first baseline model as fast as possible, it was decided that null values would be dropped.

However, using `.dropna()` directly would remove almost all of the rows in the database, therefore the first step was to identify and drop the **features** with more than 10% of cells non-filled and later on drop the rows containing null values.

```
# identify columns with nulls
train_null_columns = amex_train_unique_customers.isna().sum() 
test_null_columns = amex_test_unique_customers.isna().sum() #colunas com valores nulos

# identify columns with more than 10% nulls
train_nulls_above10 = train_null_columns[train_null_columns>(0.1*train_null_columns.max())] 
test_nulls_above10 = test_null_columns[test_null_columns>(0.1*test_null_columns.max())] #mascara: colunas com mais de 10% dos dados nulos

# list of columns with more than 10% nulls
train_droplist = list(train_nulls_above10.index) 
test_droplist = list(test_nulls_above10.index) #listando colunas com mais de 10% dos dados nulos

# drop columns with more than 10% and remaining rows with nulls
amex_null_cleaned = amex_train_unique_customers.drop(columns=train_droplist).dropna().reset_index(drop=True) 
amex_test_null_cleaned = amex_test_unique_customers.drop(columns=test_droplist).dropna().reset_index(drop=True)

```

### Reducing dimensionality with PCA

The PCA method was used to reduce 150 features to fewer variables for the purpose of getting faster performance without losing data information.

#### Train data

We used the train dataset to fit the model:

```
from sklearn.decomposition import PCA

X_train = amex_null_cleaned.drop(columns='customer_ID')

pca = PCA(10, svd_solver='randomized', random_state=7)
X_pca = pca.fit_transform(X_train)
```

After the transformation, we still have the same number of rows but now with 10 principal components.

*original shape:    (395239, 150)*

*transformed shape: (395239, 10)*


When calculation the explained variance ratio, we noticed that 91% of the data is mantained with 10 components.

`pca.explained_variance_ratio_.sum() >>> 0.9127308150138991`

After reducing dimensions, the output dataframe was concatenated with customer_IDs.

#### Test data

The same columns as the train dataset was filtered for `X_test`:

```
X_test = amex_test_null_cleaned[X_train.columns]

X_test_pca = pca.transform(X_test)
```

The same transformation of the training data happened here in the test and also mantaining 91% of the data.

*original shape:    (728799, 150)*

*transformed shape: (728799, 10)*

`pca.explained_variance_ratio_.sum() >>> 0.9127308150138993`

After reducing dimensions, the output dataframe was concatenated with customer_IDs.

### Creating the model

Now we have train and test databases with reduced dimensions but containing almost all of the information that we had previously with 150 features. These dataframes can now be used on machine learning models to predict our target.

#### Appending PCA train data with amex target labels

To proceed with modelling, target variable have to be included in our PCA train database. With the merge function it was possible to append the target feature by having the customer_ID as the key.

`amex_train_treated_target = dd.merge(left=amex_train_treated, right=amex_labels, left_on='customer_ID',right_on='customer_ID').compute()`

## Considerações finais

## Next steps



