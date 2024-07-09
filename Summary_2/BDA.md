- # Big Data and Text Analysis
	- # Pandas
		- ## Load Dataset
		  ```python
		  import pandas as pd
		  
		  df = pd.read_cvs('path/to/dataset')
		  ```
		  + To specify the csv file separator:
		  ```python
		  df = pd.read_cvs('path/to/dataset', sep=';')
		  ```
		  + To specify which line of the csv file to use as headers:
		  ```python
		  df = pd.read_cvs('path/to/dataset', header=0)
		  ```
		- ## Check dataframe data types
		  ```python
		  df.dtypes
		  ```
		- ## Check the number of rows and columns
		  + Rows
		  ```python
		  df.shape[0]
		  ```
		  + Columns
		  ```python
		  df.shape[1]
		  ```
		- ## Missing values
			- ### Checking whether the dataframe contains missing values:
			  ```python
			  pd.isna(df)					# Return a boolean same-sized object indicating if the values are NA
			  pd.isna(df).any()			# Return a boolean Series
			  pd.isna(df).any().any()		# Return a boolean value
			  
			  df.isna()
			  df.isna().any()
			  df.isna().any().any()
			  
			  if df.isna().any().any():
			    print('There are missing values')
			  else:
			    print('There are not missing values')
			  ```
			- ### Checking whether a columns contains missing values:
			  ```python
			  pd.isna(df['column_name'])
			  pd.isna(df['column_name']).any()
			  
			  df['column_name'].isna()
			  df['column_name'].isna().any()
			  df['column_name'].isna().any().any()
			  ```
			- ### Check whether a certain row contains missing values:
			  ```python
			  df.iloc[0].isna().any()
			  ```
			- ### Delete missing values
			  + Drop the rows where at least one element is missing:
			  ```python
			  df.dropna(inplace=True)
			  df.dropna(axis=0, inplace=True)
			  df.dropna(axis='index', inplace=True)
			  ```
			  + Drop the column where at least one element is missing:
			  ```python
			  df.dropna(axis=1, inplace=True)
			  df.dropna(axis='columns', inplace=True)
			  ```
			- ### Completing missing values
			  + SimpleImputer:
			  ```python
			  from sklearn.impute import SimpleImputer
			  
			  imp = SimpleImputer(strategy='mean')
			  
			  imp_array = imp.fit_transform(df)
			  
			  df = pd.DataFrame(imp_array, coloumns=df.columns)
			  ```
		- ## Check if classes are balanced
		  ```python
		  df['column_name'].value_counts()
		  
		  df['column_name'].value_counts(normalize=True)
		  
		  df.groupby('column_name').size()
		  ```
		- ## Check whether there are elements of a given dataframe column at a certain value/values
		  + One column:
		  ```python
		  df[df['column_name'] == value]
		  df[df['column_name'] == value].shape
		  
		  ```
		  + More columns:
		  ```python
		  df[(df['column_name_1'] == 'value') & (df['column_name_2'] == 'value')]
		  df[(df['column_name_1'] == 'value') & (df['column_name_2'] == 'value')].shape
		  ```
		- ## Check if all values in a dataframe have the same value
		  ```python
		  (df['column_name'] == 'value').all()
		  
		  (df['column_name'] == 'value').any()
		  
		  df['column_name'].unique()
		  ```
		- ## Creating a Pandas dataframe column based on a given condition
		  ```python
		  df['new_column'] = dataset.apply(lambda x: 2 if x['column_1'] > dataset['column_1'].mean() else 1, axis=1) 
		  ```
		- ## Deleting instances based on values in a dataframe column
		  ```python
		  df = df[df['column_name'] != 'value']
		  df.drop(df[df['column_name'] != 'value'].index, inplace=True)
		  ```
		- ## Delete column from a dataframe
		  ```python
		  df.drop('column_name', axis=1, inplace=True)
		  ```
		- ## Select columns based on their data type
		  ```python
		  df.select_dtypes(include=None, exclude=None)
		  ```
		- ## Mean
		  ```python
		  df['column_name'].mean()
		  ```
		- ## Convert categorical features into numerical features
		  id:: 645b5644-6f9c-4dd8-b467-67cc0cee6508
		  + Ordinal Encoding:
		  ```python
		  from sklearn.preprocessing import OrdinalEncoder
		  
		  # ordinal endoding for some columns of the dataframe
		  
		  features = ['x', 'y']
		  
		  encoder = OrdinalEncoder()
		  
		  df[features] = encoder.fit_transform(df[features])
		  
		  # ordinal endoding for the entire dataframe
		  
		  encoder = OrdinalEncoder()
		  
		  enc_df = encoder.fit_transform(df)
		  
		  df = pd.DataFrame(enc_df, columns=df.columns)
		  ```
		  + One-hot Encoding:
		  ```python
		  # identify the categorical features that need to be encoded
		  cat_features = ['x', 'y']
		  
		  # perform one-hot encoding
		  encoded_cat_features = pd.get_dummies(df[cat_features], prefix_sep='_')
		  
		  # join encoded features and numerical features
		  encoded_df = df[['numeric_x', 'numeric_y']].join(encoded_cat_features)
		  ```
		  + Replace:
		  ```python
		  df = pd.DataFrame({'column_1': ['A', 'B', 'C', 'A', 'B', 'C']})
		  
		  encoding = {'A': 0, 'B': 1, 'C': 2}
		  
		  df['column_1'] = df['column_1'].replace(encoding)
		  ```
		- ## Discretization of the values of a dataframe column
			- pandas wiki
			- ```python
			  pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True)
			  ```
				- > **__Esempio:__**
				  ```python
				  df['discrete column'] = pd.cut(df['column'], bins=6, labels=[1, 2, 3, 4, 5, 6])
				  ```
			- sklearn wiki
			- ```python
			  from sklearn.preprocessing import KBinsDiscretizer
			  
			  disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
			  
			  new_X_train = disc.fit_transform(X_train)
			  ```
		- ## Subset of data from the original one
		  ```python
		  new_columns = ['column_1', 'column_2',...]
		  
		  new_df = df.loc[:, new_columns]
		  ```
		- ## Pivot table
		  id:: 6460c401-df0f-4b99-9e02-8a010e36518a
		- pd.pivot_table
		  ```python
		  pivot_table = pd.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False, sort=True)
		  ```
		  where:
		  + `data`: the dataframe to create the  from.
		  + `values`: **the column of the dataframe to use to calculate the aggregations**. If not specified, all numeric columns will be used.
		  + `index`: the column or columns of the dataframe to use as the index of the pivot table. **These columns will become the rows of the table**.
		  + `columns`: the column or columns of the dataframe to use as the columns of the pivot table. **These columns will become the columns of the table**.
		  + `aggfunc`: the  to use to calculate the values of the pivot table. The default value is 'mean', which calculates the mean of the values. Other common options are 'sum', 'count', 'min', 'max', 'std', 'var', 'first', 'last', 'median', 'quantile', and 'sem'.
		  + `fill_value`: the value to use to fill any missing values.
		  + `margins`: if True, adds a 'All' row and column that show the totals of the rows and columns.
		  + `dropna`: if True, removes rows where there are missing values.
		  + `margins_name`: the name to use for the total row and column, if margins=True.
		  + `observed`: if True, uses only the existing combinations of values in `data`. If False, uses all possible combinations.
		  + `sort`: if True, sorts the rows and columns in alphabetical order.
		  ```python
		  pivot_table = pd.pivot_table(data=df, index=pd.cut(df['age'], bins=5, include_lowest=True), columns=['sex', 'smoking'], values='DEATH_EVENT', aggfunc='count')
		  ```
		- pd crosstab
		  ```python
		  cross_tab = pd.crosstab(index=pd.cut(df['age'], bins=5, include_lowest=True), columns=[df['sex'], df['smoking']], values=df['DEATH_EVENT'], aggfunc='count', normalize='index')
		  ```
	- # Matplotlib
		- Import Matplotlib:
		  ```python
		  import matplotlib.pyplot as plt
		  ```
		- Title:
		  ```python
		  plt.title('title name')
		  ```
		- x label and y label:
		  ```python
		  plt.xlabel('x label')
		  plt.ylabel('y label')
		  ```
	- # Seaborn
		- Import Seaborn:
		  ```python
		  import seaborn as sns
		  ```
		- [Barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html):
		  ```python
		  seaborn.barplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, units=None, seed=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, errcolor='.26', errwidth=None, capsize=None, dodge=True, ci='deprecated', ax=None, **kwargs)
		  ```
		- Histogram:
		  ```python
		  sns.histplot(data=None, *, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None, discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, ax=None, **kwargs)
		  ```
		- [Lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html):
		  ```python
		  seaborn.lineplot(data=None, *, x=None, y=None, hue=None, size=None, style=None, units=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None, dashes=True, markers=None, style_order=None, estimator='mean', errorbar=('ci', 95), n_boot=1000, seed=None, orient='x', sort=True, err_style='band', err_kws=None, legend='auto', ci='deprecated', ax=None, **kwargs)
		  ```
	- # Scikit-learn
		- ## Import
		  ```python
		  import sklearn
		  ```
		- ## Split the dataset
		  ```python
		  from sklearn.model_selection import train_test_split
		  
		  # extract features from the dataframe
		  X = df.drop("label", axis=1)
		  
		  # extract labels from the dataframe
		  y = df["label"]
		  
		  X_train, X_test, y_train, y_test = train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
		  ```
			- > **__Esempio:__**
			  ```python
			  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
			  ```
		- ## Processing data
			- ### Standardize features by removing the mean and scaling to unit variance
			  ```python
			  from sklearn.preprocessing import StandardScaler
			  
			  scaler = StandardScaler(*, copy=True, with_mean=True, with_std=True)
			  
			  new_X_train = scaler.fit_transform(X_train)
			  ```
			- ### Transform features by scaling each feature to a given range
			  ```python
			  from sklearn.preprocessing import MinMaxScaler
			  
			  scaler = MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
			  
			  new_X_train = scaler.fit_transform(X_train)
			  ```
			- ### Scale each feature by its maximum absolute value
			  ```python
			  from sklearn.preprocessing import MaxAbsScaler
			  
			  scaler = MaxAbsScaler()
			  
			  new_X_train = scaler.fit_transform(X_train)
			  ```
			- ### Normalize
			  ```python
			  from sklearn.preprocessing import Normalizer
			  
			  normalizer = Normalizer()
			  
			  new_X_train = normalizer.fit_transform(X_train)
			  ```
		- ## Classifiers
			- ### Logistic Regression #[[Logistic Regression]]
				- [wiki](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
				- ```python
				  from sklearn.linear_model import LogisticRegression
				  
				  lr_clf = LogisticRegression()
				  
				  lr_clf.fit(X_train, y_train)
				  
				  lr_y_pred = lr_clf.predict(X_test)
				  ```
			- ### Naive Bayes #[[Bayes Classifier]]
				- #### Multinomial Naive Bayes
				  + [wiki](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
				  ```python
				  from sklearn.naive_bayes import MultinomialNB
				  
				  nb_clf = MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)
				  
				  nb_clf.fit(X_train, y_train)
				  
				  y_pred = nb_clf.predict(X_test)
				  ```
				- #### Bernoulli Naive Bayes
				  + [wiki](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)
				  ```python
				  from sklearn.naive_bayes import BernoulliNB
				  
				  nb_clf = BernoulliNB()
				  
				  nb_clf.fit(X_train, y_train)
				  
				  y_pred = nb_clf.predict(X_test)
				  ```
			- ### Decision Tree #[[Random Forest]]
			  ```python
			  from sklearn.tree import DecisionTreeClassifier
			  
			  dt_clf = DecisionTreeClassifier()
			  
			  dt_clf.fit(X_train, y_train)
			  
			  dt_y_pred = dt_clf.predict(X_test)
			  ```
			- ### Random Forest #[[Random Forest]]
				- [wiki](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
				- ```python
				  from sklearn.ensemble import RandomForestClassifier
				  
				  rf_clf = RandomForestClassifier()
				  
				  rf_clf.fit(X_train, y_train)
				  
				  rf_y_pred = rf_clf.predict(X_test)
				  ```
			- ### KNN
			  ```python
			  from sklearn.neighbors import KNeighborsClassifier
			  
			  knn_clf = KNeighborsClassifier()
			  
			  knn_clf.fit(X_train, y_train)
			  
			  knn_y_pred = knn_clf.predict(X_test)
			  ```
			- ### Dummy Classifier
			  ```python
			  from sklearn.dummy import DummyClassifier
			  
			  dummy_clf = DummyClassifier()
			  
			  dummy_clf.fit(X_train, y_train)
			  
			  dummy_y_pred = dummy_clf.predict(X_test)
			  ```
		- ## K Fold
			- ```python
			  from sklearn.model_selection import cross_val_score
			  
			  # choose a model
			  model = SomeModel()
			  
			  # cross-validation for the model
			  scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
			  
			  
			  print("Accuracy: %0.2f (+/- %0.2f), Max Score: %0.2f" % (scores.mean(), scores.std() * 2, scores.max()))
			  ```
			- ```python
			  from sklearn.model_selection import cross_val_score
			  
			  def cross_validation(model_list, X, y, cv: int = 5, scoring='accuracy'):
			      
			      for model in model_list:
			          scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
			          print("{} accuracy: {:.2f} (+/- {:.2f}), max score: {:.2f}".format(type(model).__name__, scores.mean(), scores.std() * 2, scores.max()))
			  
			  models = [DecisionTreeClassifier(), LogisticRegression(solver='liblinear'), DummyClassifier()]
			  
			  cross_validation(model_list=models, X=X, y=y, cv=10)
			  ```
		- ## PCA #PCA 
		  ```python
		  from sklearn.decomposition import PCA
		  
		  X = ...
		  
		  dim_kept: int = ...
		  
		  pca = PCA(n_components=dim_kept)
		  
		  X_pca = pca.fit_transform(X)
		  ```
		- ## Evaluation
			- ### Accuracy score
			  + Basic:
			  ```python
			  from sklearn.metrics import accuracy_score
			  
			  accuracy = accuracy_score(y_test, y_pred)
			  
			  print('accuracy: {:.3f}'.format(accuracy))
			  ```
			  + Advanced
			  ```python
			  from sklearn.metrics import accuracy_score
			  
			  def test_clf(X_train, y_train, X_test, y_test, clf):
			  
			      clf.fit(X_train, y_train)
			      y_pred = clf.predict(X_train)
			      accuracy = accuracy_score(y_train, y_pred)
			      print('{} train accuracy: {:.3f}'.format(type(clf).__name__, accuracy))
			  
			      clf.fit(X_train, y_train)
			      y_pred = clf.predict(X_test)
			      accuracy = accuracy_score(y_test, y_pred)
			      print('{} test accuracy: {:.3f}'.format(type(clf).__name__, accuracy))
			      
			      display_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=classes, clf=clf)
			  
			  
			  test_clf(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, clf=my_clf)
			  ```
			- ### Confusion_matrix
			  + Basic:
			  ```python
			  import numpy as np
			  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
			  
			  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
			  
			  cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
			  
			  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			  
			  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot(ax=axs[0])
			  axs[0].set_title('Confusion Matrix')
			  
			  ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=clf.classes_).plot(ax=axs[1])
			  axs[1].set_title('Normalized Confusion Matrix')
			  ```
			  + Advanced:
			  ```python
			  import numpy as np
			  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
			  
			  classes = ['class 1', 'class 2', 'class 3']
			  
			  def display_confusion_matrix(y_true, y_pred, classes, clf):
			      cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)
			      cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			  
			      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
			  
			      ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axs[0])
			      axs[0].set_title('Confusion Matrix')
			      axs[0].set_xticklabels(labels=classes, rotation=45)
			      axs[0].set_yticklabels(labels=classes)
			  
			      ConfusionMatrixDisplay(confusion_matrix=cm_norm).plot(ax=axs[1])
			      axs[1].set_title('Normalized Confusion Matrix')
			      axs[1].set_xticklabels(labels=classes, rotation=45)
			      axs[1].set_yticklabels(labels=classes)
			  
			  
			  display_confusion_matrix(y_true=y_test, y_pred=my_y_pred, classes=classes, clf=my_clf)
			  ```
			- ### Grid Search
				- [wiki](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
				- ```python
				  from sklearn.model_selection import GridSearchCV
				  
				  # choose some model
				  model = SomeModel()
				  
				  # choose parameters to be tested
				  parameters = {'criterion': ..., 'param_2': ..., }
				  
				  clf = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
				  
				  clf.fit(X, y)
				  
				  
				  print("best parameters:", clf.best_params_)
				  print("best score:", clf.best_score_)
				  
				  # clf.get_params()
				  
				  # clf.cv_results_
				  
				  # clf.best_estimator_
				  
				  # clf.best_score_
				  
				  # clf.best_params_
				  ```
				  + Grid Search + Pipeline:
				  ```python
				  from sklearn.pipeline import Pipeline
				  from sklearn.decomposition import PCA
				  from sklearn.svm import SVC
				  from sklearn.model_selection import GridSearchCV
				  
				  # Creare la pipeline
				  pipe = Pipeline([
				      ('pca', PCA()),
				      ('svm', SVC())
				  ])
				  
				  # Definire la griglia di parametri da cercare
				  param_grid = {
				      'pca__n_components': [5, 10, 15],
				      'svm__kernel': ['linear', 'rbf'],
				      'svm__C': [0.1, 1, 10]
				  }
				  
				  # Creare l'oggetto GridSearchCV
				  grid_search = GridSearchCV(pipe, param_grid, cv=5)
				  
				  # Addestrare l'oggetto GridSearchCV sui dati
				  grid_search.fit(X, y)
				  
				  # Stampare i migliori parametri e lo score ottenuto
				  print("Migliori parametri:", grid_search.best_params_)
				  print("Miglior score:", grid_search.best_score_)
				  ```
		- ## Pipeline
			- ### Simple Pipeline
			  ```python
			  from sklearn.pipeline import Pipeline
			  
			  # preprocessing
			  from sklearn.decomposition import PCA
			  from sklearn.preprocessing import MinMaxScaler
			  from sklearn.preprocessing import Normalizer
			  
			  # model
			  from sklearn.linear_model import LogisticRegression
			  
			  pipeline = Pipeline(steps=[('normalize', Normalizer()),
			                             ('scaler', MinMaxScaler()),
			                             ('PCA', PCA()),
			                             ('model', LogisticRegression())])
			  
			  # preprocessing of training data, fit mode
			  pipeline.fit(X_train, y_train)
			  
			  y_pred = pipeline.predict(X_test)
			  
			  # accuracy stuff here ...
			  ```
		- ## ColumnTransformer
			- ```python
			  from sklearn.compose import ColumnTransformer
			  from sklearn.pipeline import Pipeline
			  from sklearn.impute import SimpleImputer
			  from sklearn.preprocessing import StandardScaler, OneHotEncoder
			  
			  # model
			  from sklearn.linear_model import LogisticRegression
			  
			  # Define transformations for numeric columns
			  numeric_transformer = Pipeline(steps=[
			      ('imputer', SimpleImputer(strategy='median')),
			      ('scaler', StandardScaler())])
			  
			  # Define transformations for categorical columns
			  categorical_transformer = Pipeline(steps=[
			      ('imputer', SimpleImputer(strategy='most_frequent')),
			      ('onehot', OneHotEncoder(handle_unknown='ignore'))])
			  
			  # Define the complete pipeline
			  preprocessor = ColumnTransformer(
			      transformers=[
			          ('num', numeric_transformer, num_features),
			          ('cat', categorical_transformer, cat_features)
			      ])
			  
			  pipeline = Pipeline(steps=[('preprocessor', preprocessor),
			                             ('model', LogisticRegression())])
			  
			  
			  # Preprocessing of training data, fit model 
			  pipeline.fit(X_train, y_train)
			  
			  # Preprocessing of validation data, get predictions
			  y_pred = pipeline.predict(X_test)
			  
			  # accuracy stuff here ...
			  ```
			- ```python
			  from sklearn.preprocessing import MinMaxScaler
			  from sklearn.pipeline import Pipeline
			  from sklearn.preprocessing import Normalizer
			  from sklearn.preprocessing import KBinsDiscretizer
			  from sklearn.compose import ColumnTransformer
			  
			  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
			  
			  preprocessor = ColumnTransformer(
			      transformers=[
			          ('ram_discretized', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'), ['ram']),
			          ('battery_power_discretized', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), ['battery_power'])
			      ],
			      remainder='passthrough'
			  )
			  
			  
			  my_pipeline = Pipeline([('preprocessor', preprocessor),
			                          ('scaler', MinMaxScaler()),
			                          ('normalizer', Normalizer()),
			                          ('dt', DecisionTreeClassifier())])
			  
			  # preprocessing of training data, fit mode
			  my_pipeline.fit(X_train, y_train)
			  
			  y_pred = my_pipeline.predict(X_test)
			  ```
		- ## FeatureUnion
			- ```python
			  from sklearn.preprocessing import MinMaxScaler
			  from sklearn.pipeline import Pipeline, FeatureUnion
			  from sklearn.preprocessing import Normalizer
			  from sklearn.preprocessing import KBinsDiscretizer
			  from sklearn.decomposition import PCA
			  from sklearn.feature_selection import SelectKBest
			  from sklearn.compose import ColumnTransformer
			  
			  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
			  
			  preprocessor = ColumnTransformer(
			      transformers=[
			          ('age_discretized', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'), ['THROMBOCYTE']),
			          ('thromocyte_discretized', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform'), ['AGE'])
			      ],
			      remainder='passthrough'
			  )
			  
			  preprocessing_pipeline = Pipeline([('preprocessor', preprocessor),
			                                     ('scaler', MinMaxScaler()),
			                                     ('normalizer', Normalizer())])
			  
			  feature_union = FeatureUnion([
			      ('preprocessing_pipeline', preprocessing_pipeline),
			      ('pca', PCA(n_components=2)),
			      ('selectkbest', SelectKBest(k=2))
			  ])
			  
			  dt_pipeline = Pipeline([('feature_union', feature_union),
			                          ('dt', DecisionTreeClassifier())])
			  
			  # dt pipeline
			  dt_pipeline.fit(X_train, y_train)
			  
			  dt_y_pred = dt_pipeline.predict(X_test)
			  
			  dt_accuracy = accuracy_score(y_test, dt_y_pred)
			  
			  print('dt accuracy: {:.3f}'.format(dt_accuracy))
			  ```
	- ## Imbalanced-learning
		- ### Import
		  ```python
		  import imblearn
		  ```
		- ### Oversample the smallest class
		  ```python
		  from imblearn.over_sampling import RandomOverSampler
		  
		  over_sampler = RandomOverSampler()
		  
		  X_res, y_res = over_sampler.fit_resample(X, y)
		  ```
		- ### Undersample the biggest class
		  ```python
		  from imblearn.under_sampling import RandomUnderSampler
		  
		  under_sampler = RandomUnderSampler()
		  
		  X_res, y_res = under_sampler.fit_resample(X, y)
		  ```
		- ### SMOTE
		  ```python
		  from imblearn.over_sampling import SMOTE
		  
		  sm = SMOTE()
		  
		  X_res, y_res = sm.fit_resample(X, y)
		  ```