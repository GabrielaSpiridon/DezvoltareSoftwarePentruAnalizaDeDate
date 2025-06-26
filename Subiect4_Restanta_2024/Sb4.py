import pandas as pd

# 1. Citește fișierul cu angajați (ani ca variabile)
df_ns = pd.read_csv('E_NSAL_2008-2021.csv')

# 2. Extrage anii (toate coloanele în afară de SIRUTA)
ani = [col for col in df_ns.columns if col != 'SIRUTA']

# 3. Determină anul cu cei mai mulți angajați pentru fiecare localitate
df_ns['Anul'] = df_ns[ani].idxmax(axis=1)

# 4. Salvează rezultatul
df_ns[['SIRUTA', 'Anul']].to_csv('Cerinta1.csv', index=False)

#-----------CERINTA 2-------------
# 1. Citește și fișierul cu populația localităților
df_pop = pd.read_csv('PopulatieLocalitati.csv')

# 2. Îmbină cele două seturi de date după codul SIRUTA
df_merge = pd.merge(df_ns, df_pop, left_on='SIRUTA', right_on='Siruta')

# 3. Grupează după județ și adună angajații și populația
df_group = df_merge.groupby('Judet')[ani + ['Populatie']].sum().reset_index()

# 4. Calculează rata ocupării = angajați / populație
for an in ani:
    df_group[an] = df_group[an] / df_group['Populatie']

# 5. Calculează rata medie (media pe ani)
df_group['RataMedie'] = df_group[ani].mean(axis=1)

# 6. Ordine descrescătoare după rata medie
df_group = df_group[['Judet'] + ani + ['RataMedie']].sort_values(by='RataMedie', ascending=False)

# 7. Salvează rezultatul
df_group.to_csv('Cerinta2.csv', index=False, float_format='%.3f')



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# 1. Pregătim datele
X = df_pacienti[['L_CORE', 'L_SURF', 'L_02', 'L_BP', 'SURF_ST', 'CORE_ST', 'BP_ST']]
y = df_pacienti['DECISION']

# 2. Aplicăm LDA
lda = LinearDiscriminantAnalysis()
Z = lda.fit_transform(X, y)

# 3. Salvăm scorurile discriminante în z.csv
df_z = pd.DataFrame(Z, columns=[f'LD{i+1}' for i in range(Z.shape[1])])
df_z.insert(0, 'Id', df_pacienti['Id'])
df_z.to_csv('z.csv', index=False)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
for label in y.unique():
    idx = y == label
    plt.scatter(Z[idx, 0], Z[idx, 1], label=label)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Scoruri Discriminante (LDA)')
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Prezicere pe același set (antrenare = testare, dacă nu avem alt set)
y_pred = lda.predict(X)

# 2. Matrice de confuzie
cm = pd.DataFrame(confusion_matrix(y, y_pred), 
                  index=lda.classes_, columns=lda.classes_)
cm.to_csv('matc.csv')

# 3. Afișare indicatori acuratețe
print(classification_report(y, y_pred, digits=3))

