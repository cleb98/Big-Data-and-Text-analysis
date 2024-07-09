#trasformatori e pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
#metriche e modelli
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression


#funzione per calcolare confusion matrix e il report
def plot_confusion_matrix(confusion_matrix):
    cm = confusion_matrix
    # Calcolo dei conteggi non normalizzati
    group_counts = ["{0:0.0f}\n({1:0.2%})".format(value, value/np.sum(cm)) for value in cm.flatten()]

    # Calcolo dei conteggi normalizzati
    group_percentages = ["{0:.2%}\n({1:0.0f})".format(value, value*np.sum(cm)) for value in cm.flatten()/np.sum(cm)]

    # Trasformazione dei conteggi in una matrice quadrata
    labels = np.asarray(group_counts).reshape(cm.shape[0],cm.shape[1])

    plot = sns.heatmap(cm, annot = labels, fmt='', cmap='Blues')
    plot.set_title('Confusion Matrix')
    plot.set_ylabel('True class')
    plot.set_xlabel('Predicted class')
    plt.show()


def gridSearch(model, param_grid, cv, X_train, y_train, X_test, y_test): #works also with pipelines
    clf = GridSearchCV(model, param_grid, cv=cv)
    clf.fit(X_train, y_train)
    print('best parameters: ', clf.best_params_)
    print('with accuracy: ', clf.best_score_)
    print()
    model.set_params(**clf.best_params_)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    print('acc sul train del modello con i parametri ottimizzati: ', acc_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('acc del modello con i parametri ottimizzati: : ', acc)
    print('--------------------------------------------------------------------')
    return model


# def compareModels(model_list, X_train, y_train, X_test, y_test):
#     for m in model_list:
#         m.fit(X_train, y_train)
#         y_pred_train = m.predict(X_train)
#         y_pred = m.predict(X_test)
#         print(m)
#         print('accuracy train: ', accuracy_score(y_train, y_pred_train))
#         print('accuracy: ', accuracy_score(y_test, y_pred))
#         cm = confusion_matrix(y_test, y_pred)
#         plot_confusion_matrix(cm)
#         print()
#         print()
#         print('classification report: ', classification_report(y_test, y_pred))

#         print('--------------------------------------------------------------------')

#prende in input un modello o una lista di modelli e fa il cross validation
def crossValidation(model,cv, X, y):
        cross_val = cross_val_score(m, X, y, cv = cv, scoring = 'accuracy') 
        print(model)
        # print()
        print('cross validation scores: ', cross_val)
        # print()
        # print('accuracy media', cross_val.mean())
        # print()
        print("Accuracy con incertezza: %0.2f (+/- %0.2f)" % (cross_val.mean(), cross_val.std() * 2))
        print('--------------------------------------------------------------------')
        return model



# allena il modello su X_train e y_train
# stampa l'accuracy sul train e sul test
def  testModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    print(model)
    print('accuracy train: ', accuracy_score(y_train, y_pred_train))
    print('accuracy test: ', accuracy_score(y_test, y_pred))
    print('--------------------------------------------------------------------')    
    return model

# allena il modello su X_train e y_train
# stampa l'accuracy sul train e sul test
# stampa la confusion matrix e il classification report
def testModelWithConfusionMatrix(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    print(model)
    print('accuracy train: ', accuracy_score(y_train, y_pred_train))
    print('accuracy test: ', accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
    print()
    print('--------------------------------------------------------------------')
    print('classification report: ')
    print()
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------------------')

# input d'esempio 
# models = [LogisticRegression(solver = 'liblinear'), DecisionTreeClassifier()]
# pipeline = Pipeline(steps=[('col_trasf', col_trasf), ('scaler', scaler), ('models_list', models)])
# def testPipelineOnModels(pipeline, models_list, X_train, y_train, X_test, y_test):
#     for m in models_list:
#         print(m)
#         pipeline.fit(X_train, y_train)
#         y_pred_train = pipeline.predict(X_train)
#         y_pred = pipeline.predict(X_test)
#         print('accuracy train: ', accuracy_score(y_train, y_pred_train))
#         print('accuracy: ', accuracy_score(y_test, y_pred))
#         print()
#         print()
#         print('classification report: ', classification_report(y_test, y_pred))

#         print('--------------------------------------------------------------------')
  

# per fare il feature union fra il dataset originale e le nuove colonne da aggiungere
def identity(X):
  return X
identity = FunctionTransformer(identity, validate=True)

#trasforma le colonne categoriche in numeriche con ordinal encoder
def categoricalToNumerical(dt):
    categorical_columns = dt.select_dtypes(include=['object']).columns.tolist()
    data = dt.copy()
    oe = OrdinalEncoder()
    data[categorical_columns] = oe.fit_transform(data[categorical_columns])
    data = pd.DataFrame(data, columns = dt.columns)
    return data

# ritorna lista delle colonne categoriche di un dataframe dato in input
def categoricalColumns(dt):
    categorical_columns = dt.select_dtypes(include=['object']).columns.tolist()
    return categorical_columns

# ritorna lista delle colonne numeriche di un dataframe dato in input
def numericalColumns(dt):
    numerical_columns = dt.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numerical_columns
