#%%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df
# %%
df['General Jedi encarregado'].unique()

# %%

features = ['Massa(em kilos)', 'Estatura(cm)']

df.groupby('Status ')[features].mean()
# %%
y = df['Status ']

x = df[features]

# %%
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=4)
model.fit(X=x,y=y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi = 400)

tree.plot_tree(model, feature_names= features, class_names=model.classes_,
               filled=True)
# %%
