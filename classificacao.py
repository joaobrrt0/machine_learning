#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df['aprovado'] = (df['nota'] >5).astype(int)
df
# %%
plt.plot(df['cerveja'], df['aprovado'], 'o', color = "blue")
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel("cerveja")
plt.ylabel("aprovado")


# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes
 
reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)

reg.fit(df[['cerveja']], df['aprovado'])

reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]


arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth= 2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba((df[['cerveja']].drop_duplicates()))[:,1]


plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color = "blue")
plt.grid(True)  
plt.title('Cerveja vs Aprovação')
plt.xlabel("cerveja")
plt.ylabel("aprovado")
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color = "red")
plt.plot(df['cerveja'].drop_duplicates(), reg_proba, color = "green")

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color = "purple")
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color = "orange")

#plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict, color = "black")
#plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_proba, color = "blue")

plt.hlines(0.5, xmin=1, xmax=9, linestyles= "--", colors= "black")  


plt.legend(['Observação', "Reg Predict", "Reg Pobra", 
            "Naive Bayes Predict", "Naive Bayes Pobra",
            #"Arvore D2 Predict", "Arvore D2 Pobra"
            ]
            )



     # %%
