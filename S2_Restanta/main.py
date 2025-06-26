import pandas as pd
from pandas import merge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#  - - - - - - - - - - - - - - CERINTA 1 A- - - - - - - - - - - - - -
df_netflix = pd.read_csv('Data/Netflix.csv')

col_std = ['Librarie','CostLunarBasic','CostLunarStandard','CostLunarPremium','Internet','HDI','Venit','IndiceFericire','IndiceEducatie']

scaler = StandardScaler()
df_std = df_netflix.copy()

df_std[col_std] = scaler.fit_transform(df_netflix[col_std])

df_std = df_std.sort_values(by='Internet', ascending=False)

df_std_final = df_std[['Cod','Tara']+ col_std]
df_std_final.to_csv('Cerinta1.csv', index = False, float_format='%.3f')

#  - - - - - - - - - - - - - - CERINTA 2 A- - - - - - - - - - - - - -

df_netflix = pd.read_csv('Data/Netflix.csv')

df_tari = pd.read_csv('Data/CoduriTari.csv')

df_merge = merge(df_netflix,df_tari, on='Cod')

indicatori = ['Librarie','CostLunarBasic','CostLunarStandard','CostLunarPremium','Internet','HDI','Venit','IndiceFericire','IndiceEducatie']

def coef_variatie(gr):
    return gr.std(ddof=0)/gr.mean()

df_coef_var = df_merge.groupby('Continent')[indicatori].apply(coef_variatie).reset_index()

df_coef_var = df_coef_var.sort_values(by='Librarie',ascending=False)

df_coef_var.to_csv('Cerinta2.csv', index=False, float_format = '%.3f')

#  - - - - - - - - - - - - - - CERINTA 1 B- - - - - - - - - - - - - -
df_netflix = pd.read_csv('Data/Netflix.csv')

col = ['Librarie','CostLunarBasic','CostLunarStandard','CostLunarPremium','Internet','HDI','Venit','IndiceFericire','IndiceEducatie']
X = df_netflix[col]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_std)
print("Variantele componentelor principale: ")
for i, var in enumerate(pca.explained_variance_):
    print(f"Componenta {1+i}: {var:.3f}")

#  - - - - - - - - - - - - - - CERINTA 2 B- - - - - - - - - - - - - -
df_netflix = pd.read_csv('Data/Netflix.csv')

col = ['Librarie','CostLunarBasic','CostLunarStandard','CostLunarPremium','Internet','HDI','Venit','IndiceFericire','IndiceEducatie']
X = df_netflix[col]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA()
scores=pca.fit_transform(X_std)

scoruri_df = pd.DataFrame(
    scores,
    columns=[f'PC{i+1}' for i in range(scores.shape[1])]
)
scoruri_df.insert(0,'Cod',df_netflix['Cod'])
scoruri_df.insert(1,'Tara',df_netflix['Tara'])

scoruri_df.to_csv('scoruri.csv',index=False, float_format='%.3f')

#  - - - - - - - - - - - - - - CERINTA 3 B- - - - - - - - - - - - - -

plt.scatter(scores[:,0],scores[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scoruri PCA (primele doua componente)')
plt.show()