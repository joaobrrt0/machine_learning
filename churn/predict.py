#%%
import mlflow.sklearn
import pandas as pd
import mlflow


mlflow.set_tracking_uri('http://localhost:5000')


models =mlflow.search_registered_models(filter_string="name = 'model_churn'")
latest_version = max([i for i in models[0].latest_versions])
latest_version


#import do modelo

model = mlflow.sklearn.load_model('models:/model_churn/2')
feature = model.feature_names_in_
#%%
model
# %%

#import de "novos" dados
df =    pd.read_csv("../data/abt_churn.csv")
amostra =df[df['dtRef']== df['dtRef'].max()].sample(3)
amostra =amostra.drop('flagChurn', axis=1)
# %%

#Predição
predicao = model.predict_proba(amostra[feature])[:,1]
amostra['proba_new'] = predicao
amostra
# %%

# %%
