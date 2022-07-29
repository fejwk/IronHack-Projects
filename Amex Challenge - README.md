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

![image](https://user-images.githubusercontent.com/105675184/181687701-8fb9678f-ddc8-42e0-b1a5-6f82a9065d37.png)

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

Pycaret classification only took arguments related to selecting features with more relevance to the model and removing highly correlated variables that could be sending the same information to the model.

`classification = setup(data = data_train, target = 'target', ignore_features=['customer_ID'], feature_selection=True, remove_multicollinearity=True, n_jobs=-1, use_gpu=True)`

<table id="T_6840c">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_6840c_level0_col0" class="col_heading level0 col0">Model</th>
      <th id="T_6840c_level0_col1" class="col_heading level0 col1">Accuracy</th>
      <th id="T_6840c_level0_col2" class="col_heading level0 col2">AUC</th>
      <th id="T_6840c_level0_col3" class="col_heading level0 col3">Recall</th>
      <th id="T_6840c_level0_col4" class="col_heading level0 col4">Prec.</th>
      <th id="T_6840c_level0_col5" class="col_heading level0 col5">F1</th>
      <th id="T_6840c_level0_col6" class="col_heading level0 col6">Kappa</th>
      <th id="T_6840c_level0_col7" class="col_heading level0 col7">MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6840c_level0_row0" class="row_heading level0 row0">lightgbm</th>
      <td id="T_6840c_row0_col0" class="data row0 col0">Light Gradient Boosting Machine</td>
      <td id="T_6840c_row0_col1" class="data row0 col1">0.8900</td>
      <td id="T_6840c_row0_col2" class="data row0 col2">0.9516</td>
      <td id="T_6840c_row0_col3" class="data row0 col3">0.7855</td>
      <td id="T_6840c_row0_col4" class="data row0 col4">0.7915</td>
      <td id="T_6840c_row0_col5" class="data row0 col5">0.7885</td>
      <td id="T_6840c_row0_col6" class="data row0 col6">0.7142</td>
      <td id="T_6840c_row0_col7" class="data row0 col7">0.7142</td>
    </tr>
    <tr>
      <th id="T_6840c_level0_row1" class="row_heading level0 row1">gbc</th>
      <td id="T_6840c_row1_col0" class="data row1 col0">Gradient Boosting Classifier</td>
      <td id="T_6840c_row1_col1" class="data row1 col1">0.8851</td>
      <td id="T_6840c_row1_col2" class="data row1 col2">0.9472</td>
      <td id="T_6840c_row1_col3" class="data row1 col3">0.7718</td>
      <td id="T_6840c_row1_col4" class="data row1 col4">0.7845</td>
      <td id="T_6840c_row1_col5" class="data row1 col5">0.7781</td>
      <td id="T_6840c_row1_col6" class="data row1 col6">0.7006</td>
      <td id="T_6840c_row1_col7" class="data row1 col7">0.7006</td>
    </tr>
    <tr>
      <th id="T_6840c_level0_row2" class="row_heading level0 row2">rf</th>
      <td id="T_6840c_row2_col0" class="data row2 col0">Random Forest Classifier</td>
      <td id="T_6840c_row2_col1" class="data row2 col1">0.8833</td>
      <td id="T_6840c_row2_col2" class="data row2 col2">0.9452</td>
      <td id="T_6840c_row2_col3" class="data row2 col3">0.7616</td>
      <td id="T_6840c_row2_col4" class="data row2 col4">0.7848</td>
      <td id="T_6840c_row2_col5" class="data row2 col5">0.7730</td>
      <td id="T_6840c_row2_col6" class="data row2 col6">0.6945</td>
      <td id="T_6840c_row2_col7" class="data row2 col7">0.6946</td>
    </tr>
    <tr>
      <th id="T_6840c_level0_row3" class="row_heading level0 row3">lr</th>
      <td id="T_6840c_row3_col0" class="data row3 col0">Logistic Regression</td>
      <td id="T_6840c_row3_col1" class="data row3 col1">0.8829</td>
      <td id="T_6840c_row3_col2" class="data row3 col2">0.9455</td>
      <td id="T_6840c_row3_col3" class="data row3 col3">0.7442</td>
      <td id="T_6840c_row3_col4" class="data row3 col4">0.7941</td>
      <td id="T_6840c_row3_col5" class="data row3 col5">0.7683</td>
      <td id="T_6840c_row3_col6" class="data row3 col6">0.6900</td>
      <td id="T_6840c_row3_col7" class="data row3 col7">0.6907</td>
    </tr>
    <tr>
      <th id="T_6840c_level0_row4" class="row_heading level0 row4">et</th>
      <td id="T_6840c_row4_col0" class="data row4 col0">Extra Trees Classifier</td>
      <td id="T_6840c_row4_col1" class="data row4 col1">0.8826</td>
      <td id="T_6840c_row4_col2" class="data row4 col2">0.9447</td>
      <td id="T_6840c_row4_col3" class="data row4 col3">0.7596</td>
      <td id="T_6840c_row4_col4" class="data row4 col4">0.7838</td>
      <td id="T_6840c_row4_col5" class="data row4 col5">0.7715</td>
      <td id="T_6840c_row4_col6" class="data row4 col6">0.6925</td>
      <td id="T_6840c_row4_col7" class="data row4 col7">0.6927</td>
    </tr>
  </tbody>
</table>

The metrics **AUC, Recall and Precision** are the most relevant for our study and for that reason they were determinant in choosing the best model to be applied.

According to the table above the **Light Gradient Boosting Machine (LGBM)** showed the best performance for all three metrics compared to the rest thus being the chosen model.

After creating the model, a tuning of the hyperparameters of the LGBM was executed and optimized based on 'AUC' metric.

```
lgbm = create_model('lightgbm')
final_lgbm = finalize_model(lgbm)
tuned_model = tune_model(lgbm, optimize='AUC')
```

A little improvement in performance is perceived following the tuning step.

<table id="T_71ffa">
  <thead>
    <tr>
      <th class="blank level0">&nbsp;</th>
      <th id="T_71ffa_level0_col0" class="col_heading level0 col0">Accuracy</th>
      <th id="T_71ffa_level0_col1" class="col_heading level0 col1">AUC</th>
      <th id="T_71ffa_level0_col2" class="col_heading level0 col2">Recall</th>
      <th id="T_71ffa_level0_col3" class="col_heading level0 col3">Prec.</th>
      <th id="T_71ffa_level0_col4" class="col_heading level0 col4">F1</th>
      <th id="T_71ffa_level0_col5" class="col_heading level0 col5">Kappa</th>
      <th id="T_71ffa_level0_col6" class="col_heading level0 col6">MCC</th>
    </tr>
    <tr>
      <th id="T_71ffa_level0_row11" class="row_heading level0 row11">Old LGBM</th>
      <td id="T_71ffa_row11_col0" class="data row11 col0">0.8900</td>
      <td id="T_71ffa_row11_col1" class="data row11 col1">0.9516</td>
      <td id="T_71ffa_row11_col2" class="data row11 col2">0.7855</td>
      <td id="T_71ffa_row11_col3" class="data row11 col3">0.7915</td>
      <td id="T_71ffa_row11_col4" class="data row11 col4">0.7885</td>
      <td id="T_71ffa_row11_col5" class="data row11 col5">0.7142</td>
      <td id="T_71ffa_row11_col6" class="data row11 col6">0.7142</td>
    </tr>
    <tr>
      <th id="T_71ffa_level0_row10" class="row_heading level0 row10">New LGBM</th>
      <td id="T_71ffa_row10_col0" class="data row10 col0">0.8923</td>
      <td id="T_71ffa_row10_col1" class="data row10 col1">0.9530</td>
      <td id="T_71ffa_row10_col2" class="data row10 col2">0.7897</td>
      <td id="T_71ffa_row10_col3" class="data row10 col3">0.7925</td>
      <td id="T_71ffa_row10_col4" class="data row10 col4">0.7911</td>
      <td id="T_71ffa_row10_col5" class="data row10 col5">0.7186</td>
      <td id="T_71ffa_row10_col6" class="data row10 col6">0.7186</td>
    </tr>
  </tbody>
</table>

*Quick reminder:*

*Recall: of all of the default customers, how many were predicted as default?*

*Precision: of all of the default predictions, how many were actual defaults?*

With this new model, Recall and Precision metrics shown considerable better performance in comparison to the Logistic Regression built before. 

**Recall**: aprox. 8 out of 10 default customers were predicted.

**Precision**: aprox. 8 out of 10 default predictions, were correct.

A high AUC metric means that the model has good capacity to differentiate between positive and negative targets.

![image](https://user-images.githubusercontent.com/105675184/181684474-639568b4-f744-4b5f-88a9-c8a462e55599.png)


According to the features importance graphic, it is possible to infer that Delinquency and Balance variables had more relevance for this model.

![image](https://user-images.githubusercontent.com/105675184/181684176-e5ee897b-8648-4579-b1f6-115c436172f2.png)

The learning curve graphic appears to show that the model still have space to learn more and get better scores for the same data used until now.

![image](https://user-images.githubusercontent.com/105675184/181690630-9363e509-7780-4ab8-8be0-9c84131d6eec.png)


The tuned model was then used to predict the probabilities of future defaults for each customer. By appending the `.predict_proba`s to the test dataset, it is now possible to upload the new model in Amex Challenge in Kaggle.

The overall score for the predictive model developed in this analysis was 0.75.

![image](https://user-images.githubusercontent.com/105675184/181684926-cda79666-ff6d-4b45-9d9a-e38e141f1bfe.png)


## Learnings & Conclusions

- When manipulating large datasets, it is possible to make use of alternative tools, such as Dask;
- Specifically for competitions, dropping rows due to null values, can lead you to a predictive model without scores;
- In this study, Logistic Regression combined with PCA reductions leaves too much gaps for the company to lose money;
- Light Gradient Boost Machine proved to be the best model considering all the relevant metrics;
- The most important features are related to Balance and Deliquency variables.

## Next steps

- Go back to data manipulation step and treat nulls with other strategies such as median, mode and KNN Imputer and compare the score with the developed models.
- Explore data modelling with time series
- Run more training iterations and folds in order to maximize the model's learning curve.
