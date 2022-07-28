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
`amex_test_unique_customers = amex_test_data.groupby(amex_test_data.customer_ID).mean().compute().reset_index()`

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


When calculation the explained variance ratio, we noticed that 91% of the data is maintained with 10 components.

`pca.explained_variance_ratio_.sum() >>> 0.9127308150138991`

After reducing dimensions, the output dataframe was concatenated with customer_IDs.

#### Test data

The same columns as the train dataset was filtered for `X_test`:

```
X_test = amex_test_null_cleaned[X_train.columns]

X_test_pca = pca.transform(X_test)
```

The same transformation of the training data happened here in the test and also maintaining 91% of the data.

*original shape:    (728799, 150)*

*transformed shape: (728799, 10)*

`pca.explained_variance_ratio_.sum() >>> 0.9127308150138993`

After reducing dimensions, the output dataframe was concatenated with customer_IDs.

### Creating the model

Now we have train and test databases with reduced dimensions but containing almost all of the information that we had previously with 150 features. These dataframes can now be used on machine learning models to predict our target.

#### Appending PCA train data with amex target labels

To proceed with modelling, target variable have to be included in our PCA train database. With the merge function it was possible to append the target feature by having the customer_ID as the key.

`amex_train_treated_target = dd.merge(left=amex_train_treated, right=amex_labels, left_on='customer_ID',right_on='customer_ID').compute()`

#### Logistic Regression

Logistic Regression model was chosen to be the baseline for our study using the PCA databases.

##### Training the model

```
X_train=amex_train_treated_target.drop(columns=['customer_ID','target'])
y_train=amex_train_treated_target['target']

lrmodel = LogisticRegression(random_state=7)
lrmodel.fit(X_train,y_train)
```

Running `.predict` and `.predict_proba` methods we can see how was the performance of our model within the PCA train database.
By calculating the score, we can infer that the model created was able to predict 82.5% targets successfully.

`lrmodel.score(X_train,y_train) >>> 0.8247946179400312`

When evaluating with other metrics, the model appears to be non-efficient.

**Recall**: of all of the default customers, how many were predicted as default?

4 out of 10 default customers were not predicted.

`recall_score(y_train,y_pred) >>> 0.5867017345146335` 

**Precision**: of all of the default predictions, how many were actual defaults?

3 out of 10 default predictions, were not correct.

`precision_score(y_train,y_pred) >>> 0.7000332138398635`

`plot_roc_curve(lrmodel, X_train, y_train)`

##### Predicting with PCA test data

Predicting the target for PCA test dataset:
```
X_test = amex_test_treated.select_dtypes('number')
lrmodel.predict(X_test)
```

Getting the probabilities for both 0 and 1 for each customer:
`lrmodel.predict_proba(X_test)`

Although the model worked for the PCA test dataset, it was not possible to evaluate its performance through the model submission in kaggle since some of the rows have been dropped at the Data Manipulation step.

## Modelling with the entirety of the data

Since all the data was needed to upload the model in Kaggle, another way to treat the nulls had to be used. In this study the mean of each feature was selected to fill the null values in the datasets.

In the Data Manipulation step, all the features with more than 10% cells filled with null values were dropped and this rule is still maintained for the new modelling. And then, the train dataset grouped by customer_IDs was merged with `amex_labels` dataset that contains the target variable for each customer. Finally, all the remaining null values were filled with the feature's average.

This treatment was applied to both train and test datasets.

```
# dropping features with more than 10% null cells
new_train = amex_train_unique_customers.drop(columns=train_droplist)
new_test = amex_test_unique_customers[new_train.columns.drop('target')]

# merging the previous dataframe with target dataset
new_train = dd.merge(left=new_train, right=amex_labels,left_on='customer_ID',right_on='customer_ID').compute()

# filling nulls with feature's average
new_train_treated = new_train.fillna(new_train.select_dtypes('number').mean())
new_test_treated = new_test.fillna(new_test.select_dtypes('number').mean())
```

After the treatment both dataframes were exported to be used in Pycaret's ML library.

### Pycaret

The data had to be exported and imported due to Pycaret running in a different environment that the codes above.

`data_train = pd.read_csv('new_train_treated.csv')`
`data_test = pd.read_csv('new_test_treated.csv')`



`classification = setup(data = data_train, target = 'target', ignore_features=['customer_ID'], feature_selection=True, remove_multicollinearity=True, n_jobs=-1, use_gpu=True)`

<table id="T_86267">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_86267_level0_col0" class="col_heading level0 col0">Model</th>
      <th id="T_86267_level0_col1" class="col_heading level0 col1">Accuracy</th>
      <th id="T_86267_level0_col2" class="col_heading level0 col2">AUC</th>
      <th id="T_86267_level0_col3" class="col_heading level0 col3">Recall</th>
      <th id="T_86267_level0_col4" class="col_heading level0 col4">Prec.</th>
      <th id="T_86267_level0_col5" class="col_heading level0 col5">F1</th>
      <th id="T_86267_level0_col6" class="col_heading level0 col6">Kappa</th>
      <th id="T_86267_level0_col7" class="col_heading level0 col7">MCC</th>
      <th id="T_86267_level0_col8" class="col_heading level0 col8">TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_86267_level0_row0" class="row_heading level0 row0">lightgbm</th>
      <td id="T_86267_row0_col0" class="data row0 col0">Light Gradient Boosting Machine</td>
      <td id="T_86267_row0_col1" class="data row0 col1">0.8896</td>
      <td id="T_86267_row0_col2" class="data row0 col2">0.9510</td>
      <td id="T_86267_row0_col3" class="data row0 col3">0.7864</td>
      <td id="T_86267_row0_col4" class="data row0 col4">0.7862</td>
      <td id="T_86267_row0_col5" class="data row0 col5">0.7863</td>
      <td id="T_86267_row0_col6" class="data row0 col6">0.7119</td>
      <td id="T_86267_row0_col7" class="data row0 col7">0.7119</td>
      <td id="T_86267_row0_col8" class="data row0 col8">11.7400</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row1" class="row_heading level0 row1">gbc</th>
      <td id="T_86267_row1_col0" class="data row1 col0">Gradient Boosting Classifier</td>
      <td id="T_86267_row1_col1" class="data row1 col1">0.8846</td>
      <td id="T_86267_row1_col2" class="data row1 col2">0.9467</td>
      <td id="T_86267_row1_col3" class="data row1 col3">0.7740</td>
      <td id="T_86267_row1_col4" class="data row1 col4">0.7778</td>
      <td id="T_86267_row1_col5" class="data row1 col5">0.7759</td>
      <td id="T_86267_row1_col6" class="data row1 col6">0.6982</td>
      <td id="T_86267_row1_col7" class="data row1 col7">0.6982</td>
      <td id="T_86267_row1_col8" class="data row1 col8">1413.0500</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row2" class="row_heading level0 row2">rf</th>
      <td id="T_86267_row2_col0" class="data row2 col0">Random Forest Classifier</td>
      <td id="T_86267_row2_col1" class="data row2 col1">0.8830</td>
      <td id="T_86267_row2_col2" class="data row2 col2">0.9444</td>
      <td id="T_86267_row2_col3" class="data row2 col3">0.7629</td>
      <td id="T_86267_row2_col4" class="data row2 col4">0.7792</td>
      <td id="T_86267_row2_col5" class="data row2 col5">0.7710</td>
      <td id="T_86267_row2_col6" class="data row2 col6">0.6924</td>
      <td id="T_86267_row2_col7" class="data row2 col7">0.6925</td>
      <td id="T_86267_row2_col8" class="data row2 col8">336.2900</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row3" class="row_heading level0 row3">et</th>
      <td id="T_86267_row3_col0" class="data row3 col0">Extra Trees Classifier</td>
      <td id="T_86267_row3_col1" class="data row3 col1">0.8825</td>
      <td id="T_86267_row3_col2" class="data row3 col2">0.9441</td>
      <td id="T_86267_row3_col3" class="data row3 col3">0.7615</td>
      <td id="T_86267_row3_col4" class="data row3 col4">0.7785</td>
      <td id="T_86267_row3_col5" class="data row3 col5">0.7699</td>
      <td id="T_86267_row3_col6" class="data row3 col6">0.6910</td>
      <td id="T_86267_row3_col7" class="data row3 col7">0.6911</td>
      <td id="T_86267_row3_col8" class="data row3 col8">45.5500</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row4" class="row_heading level0 row4">lr</th>
      <td id="T_86267_row4_col0" class="data row4 col0">Logistic Regression</td>
      <td id="T_86267_row4_col1" class="data row4 col1">0.8821</td>
      <td id="T_86267_row4_col2" class="data row4 col2">0.9446</td>
      <td id="T_86267_row4_col3" class="data row4 col3">0.7454</td>
      <td id="T_86267_row4_col4" class="data row4 col4">0.7868</td>
      <td id="T_86267_row4_col5" class="data row4 col5">0.7656</td>
      <td id="T_86267_row4_col6" class="data row4 col6">0.6869</td>
      <td id="T_86267_row4_col7" class="data row4 col7">0.6874</td>
      <td id="T_86267_row4_col8" class="data row4 col8">67.5600</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row5" class="row_heading level0 row5">lda</th>
      <td id="T_86267_row5_col0" class="data row5 col0">Linear Discriminant Analysis</td>
      <td id="T_86267_row5_col1" class="data row5 col1">0.8800</td>
      <td id="T_86267_row5_col2" class="data row5 col2">0.9427</td>
      <td id="T_86267_row5_col3" class="data row5 col3">0.7447</td>
      <td id="T_86267_row5_col4" class="data row5 col4">0.7806</td>
      <td id="T_86267_row5_col5" class="data row5 col5">0.7622</td>
      <td id="T_86267_row5_col6" class="data row5 col6">0.6821</td>
      <td id="T_86267_row5_col7" class="data row5 col7">0.6824</td>
      <td id="T_86267_row5_col8" class="data row5 col8">9.5800</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row6" class="row_heading level0 row6">ridge</th>
      <td id="T_86267_row6_col0" class="data row6 col0">Ridge Classifier</td>
      <td id="T_86267_row6_col1" class="data row6 col1">0.8792</td>
      <td id="T_86267_row6_col2" class="data row6 col2">0.8275</td>
      <td id="T_86267_row6_col3" class="data row6 col3">0.7206</td>
      <td id="T_86267_row6_col4" class="data row6 col4">0.7927</td>
      <td id="T_86267_row6_col5" class="data row6 col5">0.7549</td>
      <td id="T_86267_row6_col6" class="data row6 col6">0.6750</td>
      <td id="T_86267_row6_col7" class="data row6 col7">0.6764</td>
      <td id="T_86267_row6_col8" class="data row6 col8">4.5000</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row7" class="row_heading level0 row7">ada</th>
      <td id="T_86267_row7_col0" class="data row7 col0">Ada Boost Classifier</td>
      <td id="T_86267_row7_col1" class="data row7 col1">0.8790</td>
      <td id="T_86267_row7_col2" class="data row7 col2">0.9418</td>
      <td id="T_86267_row7_col3" class="data row7 col3">0.7471</td>
      <td id="T_86267_row7_col4" class="data row7 col4">0.7758</td>
      <td id="T_86267_row7_col5" class="data row7 col5">0.7612</td>
      <td id="T_86267_row7_col6" class="data row7 col6">0.6802</td>
      <td id="T_86267_row7_col7" class="data row7 col7">0.6804</td>
      <td id="T_86267_row7_col8" class="data row7 col8">455.2000</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row8" class="row_heading level0 row8">svm</th>
      <td id="T_86267_row8_col0" class="data row8 col0">SVM - Linear Kernel</td>
      <td id="T_86267_row8_col1" class="data row8 col1">0.8752</td>
      <td id="T_86267_row8_col2" class="data row8 col2">0.8333</td>
      <td id="T_86267_row8_col3" class="data row8 col3">0.7466</td>
      <td id="T_86267_row8_col4" class="data row8 col4">0.7644</td>
      <td id="T_86267_row8_col5" class="data row8 col5">0.7554</td>
      <td id="T_86267_row8_col6" class="data row8 col6">0.6716</td>
      <td id="T_86267_row8_col7" class="data row8 col7">0.6717</td>
      <td id="T_86267_row8_col8" class="data row8 col8">8.3800</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row9" class="row_heading level0 row9">nb</th>
      <td id="T_86267_row9_col0" class="data row9 col0">Naive Bayes</td>
      <td id="T_86267_row9_col1" class="data row9 col1">0.8528</td>
      <td id="T_86267_row9_col2" class="data row9 col2">0.9120</td>
      <td id="T_86267_row9_col3" class="data row9 col3">0.7017</td>
      <td id="T_86267_row9_col4" class="data row9 col4">0.7209</td>
      <td id="T_86267_row9_col5" class="data row9 col5">0.7111</td>
      <td id="T_86267_row9_col6" class="data row9 col6">0.6124</td>
      <td id="T_86267_row9_col7" class="data row9 col7">0.6125</td>
      <td id="T_86267_row9_col8" class="data row9 col8">1.4400</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row10" class="row_heading level0 row10">knn</th>
      <td id="T_86267_row10_col0" class="data row10 col0">K Neighbors Classifier</td>
      <td id="T_86267_row10_col1" class="data row10 col1">0.8472</td>
      <td id="T_86267_row10_col2" class="data row10 col2">0.7770</td>
      <td id="T_86267_row10_col3" class="data row10 col3">0.6320</td>
      <td id="T_86267_row10_col4" class="data row10 col4">0.7384</td>
      <td id="T_86267_row10_col5" class="data row10 col5">0.6811</td>
      <td id="T_86267_row10_col6" class="data row10 col6">0.5814</td>
      <td id="T_86267_row10_col7" class="data row10 col7">0.5844</td>
      <td id="T_86267_row10_col8" class="data row10 col8">8.9500</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row11" class="row_heading level0 row11">qda</th>
      <td id="T_86267_row11_col0" class="data row11 col0">Quadratic Discriminant Analysis</td>
      <td id="T_86267_row11_col1" class="data row11 col1">0.8428</td>
      <td id="T_86267_row11_col2" class="data row11 col2">0.9081</td>
      <td id="T_86267_row11_col3" class="data row11 col3">0.6480</td>
      <td id="T_86267_row11_col4" class="data row11 col4">0.7162</td>
      <td id="T_86267_row11_col5" class="data row11 col5">0.6804</td>
      <td id="T_86267_row11_col6" class="data row11 col6">0.5765</td>
      <td id="T_86267_row11_col7" class="data row11 col7">0.5778</td>
      <td id="T_86267_row11_col8" class="data row11 col8">25.2900</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row12" class="row_heading level0 row12">dt</th>
      <td id="T_86267_row12_col0" class="data row12 col0">Decision Tree Classifier</td>
      <td id="T_86267_row12_col1" class="data row12 col1">0.8295</td>
      <td id="T_86267_row12_col2" class="data row12 col2">0.7788</td>
      <td id="T_86267_row12_col3" class="data row12 col3">0.6738</td>
      <td id="T_86267_row12_col4" class="data row12 col4">0.6686</td>
      <td id="T_86267_row12_col5" class="data row12 col5">0.6712</td>
      <td id="T_86267_row12_col6" class="data row12 col6">0.5561</td>
      <td id="T_86267_row12_col7" class="data row12 col7">0.5561</td>
      <td id="T_86267_row12_col8" class="data row12 col8">314.3300</td>
    </tr>
    <tr>
      <th id="T_86267_level0_row13" class="row_heading level0 row13">dummy</th>
      <td id="T_86267_row13_col0" class="data row13 col0">Dummy Classifier</td>
      <td id="T_86267_row13_col1" class="data row13 col1">0.7418</td>
      <td id="T_86267_row13_col2" class="data row13 col2">0.5000</td>
      <td id="T_86267_row13_col3" class="data row13 col3">0.0000</td>
      <td id="T_86267_row13_col4" class="data row13 col4">0.0000</td>
      <td id="T_86267_row13_col5" class="data row13 col5">0.0000</td>
      <td id="T_86267_row13_col6" class="data row13 col6">0.0000</td>
      <td id="T_86267_row13_col7" class="data row13 col7">0.0000</td>
      <td id="T_86267_row13_col8" class="data row13 col8">0.0300</td>
    </tr>
  </tbody>
</table>

The metrics AUC, Recall and Precision are the most relevant for our study and for that reason they were determinant in choosing the best model to be applied.

According to the table above the **Light Gradient Boosting Machine (LGBM)** showed the best performance for all three metrics compared to the rest thus being the chosen model.

After creating the model, a tuning of the hyperparameters of the LGBM was executed and optimized based on 'AUC' metric.

```
lgbm = create_model('lightgbm')
final_lgbm = finalize_model(lgbm)
tuned_model = tune_model(lgbm, optimize='AUC')
```

## Considerações finais

## Next steps



