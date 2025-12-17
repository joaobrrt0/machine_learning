#%%
import pandas as pd

from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

import mlflow

from feature_engine import discretisation, encoding

import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_name='churn_exp')

df = pd.read_csv("../data/abt_churn.csv")
df.head()
# %%
oot = df[df["dtRef"]==df['dtRef'].max()].copy()

# %%
df_train = df[df["dtRef"]<df['dtRef'].max()].copy()

# %%
# Essas são as variaveis
features = df_train.columns[2: -1]
# Essa e a  nossa target
target = 'flagChurn'

X, y = df_train[features], df_train[target]
# %%

#SAMPLE
from sklearn import model_selection

X_train, X_teste , y_train, y_teste = model_selection.train_test_split(X,y,
                                                                       random_state=42,
                                                                       test_size=0.2,
                                                                       stratify=y,
                                                                       )

print("Taxa de variavel resposta geral", y.mean())
print("Taxa de variavel Treino", y_train.mean())
print("Taxa de variavel resposta Teste", y_teste.mean())

#%%
#EXPLORE (MISSINGS)

X_train.isna().sum().sort_values(ascending = False)

# %%

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg({"mean", "median"}).T
sumario["diff_abs"] = sumario[0] - sumario[1] 
sumario["diff_rel"] = sumario[0] / sumario[1]
sumario.sort_values(by =["diff_rel"], ascending=False)
# %%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

feature_importances = (pd.Series(arvore.feature_importances_,index=X_train.columns)
                       .sort_values(ascending=False)
                       .reset_index()
                       )

feature_importances['acum.']=feature_importances[0].cumsum()
feature_importances[feature_importances['acum.']< 0.96 ]
# %%
best_features = (feature_importances[feature_importances['acum.']< 0.96 ]['index']
                 .tolist())

best_features

# %
#MODIFY

## Discretizar
tree_discretization = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3,
)

#tree_discretization.fit(X_train[best_features], y_train)
#_train_transform =tree_discretization.transform(X_train[best_features])

# Onehot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)
#onehot.fit(x_train_transform, y_train)

#x_train_transform = onehot.transform(x_train_transform)
#x_train_transform


# %%
#MODEL

with mlflow.start_run():

    mlflow.sklearn.autolog() 

#model = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000,)
#model = naive_bayes.BernoulliNB()
    model = ensemble.RandomForestClassifier(random_state=42,
                                            n_jobs=2,
    )
    #model = ensemble.AdaBoostClassifier(random_stat e=42,
    #                                    n_estimators=500,
    #                                   learning_rate=0.01)



    params = {
        "min_samples_leaf":[15,20,25,30,50],
        "n_estimators":[100,200,500,1000],
        "criterion": ['gini', 'entropy', 'log_loss'],

    }

    grid = model_selection.GridSearchCV(model,
                                        params, 
                                        cv=3, 
                                        scoring='roc_auc',
                                        verbose=4,
                                        )



    model_pipeline = pipeline.Pipeline(
        steps=[
            ("Discretizar", tree_discretization),
            ("Onehot", onehot),
            ("Grid", grid),
        ]
    )

    model_pipeline.fit(X_train[best_features], y_train)




    


    #reg.fit(x_train_transform, y_train)
## ASSESS
    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)




    y_teste_predict = model_pipeline.predict(X_teste[best_features])
    y_teste_proba = model_pipeline.predict_proba(X_teste[best_features])[:,1]


    acc_teste = metrics.accuracy_score(y_teste, y_teste_predict)
    auc_teste = metrics.roc_auc_score(y_teste, y_teste_proba)
    roc_teste = metrics.roc_curve(y_teste, y_teste_proba)


    print("Acuracia Teste:", acc_teste)
    print("AUC Teste:", auc_teste)




    y_oot_predict = model_pipeline.predict(oot[best_features])
    y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]


    acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target], y_oot_proba)

    print("Acuracia oot:", acc_oot)
    print("AUC oot:", auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_teste":acc_teste,
    "auc_teste":auc_teste,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot,
    })

# %%
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_teste[0], roc_teste[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0,1],[0,1],"--", color="black")
plt.grid(True)
plt.title("Curva ROC")
plt.xlabel("Especificidade")
plt.ylabel("Sensibilidade")
plt.legend([
f"Treino {100*auc_train: .2f}",
f"Teste {100*auc_teste: .2f}",
f"Out of time {100*auc_oot: .2f}",

])

plt.show()
# %%
model_df = pd.Series({"model":model_pipeline,
           "features": best_features,
           })

model_df.to_pickle("model.pkl")
# %%
