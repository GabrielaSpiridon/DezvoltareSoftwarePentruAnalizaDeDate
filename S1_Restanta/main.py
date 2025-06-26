import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

#  - - - - - - - - - - - - - - CERINTA 1 A- - - - - - - - - - - - - -
df_indicatori = pd.read_csv('Data/Indicatori.csv')

#calcul media coloanei CFA
media_CFA = df_indicatori['CFA'].mean()

df_filtrat = df_indicatori[df_indicatori['CFA']>media_CFA]

df_filtrat = df_filtrat.sort_values(by='CFA', ascending = False)

coloane = ['SIRUTA','NR_FIRME','NSAL','CFA','PROFITN','PIERDEREN']

df_filtrat[coloane].to_csv('Cerinta1.csv', index=False)

#  - - - - - - - - - - - - - - CERINTA 2 A - - - - - - - - - - - - - -

df_ind = pd.read_csv('Data/Indicatori.csv')
df_pop = pd.read_csv('Data/PopulatieLocalitati.csv')

#merge pe siruta
df_merge = pd.merge(df_ind,df_pop, left_on = 'SIRUTA', right_on='Siruta')

df_grup = df_merge.groupby('Judet').agg({
    'NR_FIRME':'sum',
    'NSAL':'sum',
    'CFA':'sum',
    'PROFITN':'sum',
    'PIERDEREN':'sum',
    'Populatie':'sum'
}).reset_index()


for col in ['NR_FIRME','NSAL','CFA','PROFITN','PIERDEREN']:
    df_grup[col] = (df_grup[col]*1000/ df_grup['Populatie']).round(3)

df_grup = df_grup[['Judet','NR_FIRME','NSAL','CFA','PROFITN','PIERDEREN']]
df_grup.to_csv('Cerinta2.csv', index=False)

#  - - - - - - - - - - - - - - CERINTA 1 B- - - - - - - - - - - - - -
df_locationQ = pd.read_csv('Data/LocationQ.csv')

X = df_locationQ.loc[:,'2008':'2021'].values

Z = linkage(X, method='ward')

print('cluster1 cluster2 distanta nr_elem_nou')
for row in Z:
    print(int(row[0]),int(row[1]),round(row[2],3),int(row[3]))

#  - - - - - - - - - - - - - - CERINTA 2 B- - - - - - - - - - - - - -

dendrogram(Z,labels=df_locationQ['Judet'].values)
plt.show()

#  - - - - - - - - - - - - - - CERINTA 3 B- - - - - - - - - - - - - -

nr_cluster = 3
labels = fcluster(Z,nr_cluster,criterion="maxclust")
df_locationQ['cluster'] = labels

df_locationQ[['Judet','cluster']].to_csv('popt.csv',index=False)
