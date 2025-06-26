import pandas as pd

# 1. Citește fișierul
df_ang = pd.read_csv('Data/CAEN2_2021_NSAL.csv')

# 2. Selectează coloanele cu coduri CAEN (presupunem că SIRUTA este prima coloană, restul sunt CAEN-uri)
caen_cols = [col for col in df_ang.columns if col != 'SIRUTA']

# 3. Calculează totalul angajaților pe fiecare localitate
df_ang['total_angajati'] = df_ang[caen_cols].sum(axis=1)

# 4. Calculează procentele (pe fiecare coloană CAEN)
df_procente = df_ang.copy()
for col in caen_cols:
    df_procente[col] =df_procente[col] / df_procente['total_angajati']

# 5. Păstrează doar coloanele cerute și salvează în Cerinta1.csv
df_procente = df_procente[['SIRUTA'] + caen_cols]
df_procente.to_csv('Cerinta1.csv', index=False, float_format='%.2f')  # cu 2 zecimale, ca în exemplu



# 1. Citește populația localităților
df_pop = pd.read_csv('Data/PopulatieLocalitati.csv')

# 2. Îmbină pe SIRUTA/Siruta
df_merged = pd.merge(df_ang, df_pop, left_on='SIRUTA', right_on='Siruta')

# 3. Grupează pe Judet, sumează CAEN-urile și populația
caen_cols = df_ang.columns.drop('SIRUTA')
df_group = df_merged.groupby('Judet')[list(caen_cols) + ['Populatie']].sum().reset_index()

# 4. Calculează angajații la 100.000 locuitori
for col in caen_cols:
    df_group[col] = df_group[col] * 100000 / df_group['Populatie']

df_out = df_group[['Judet'] + list(caen_cols)]
df_out.to_csv('Cerinta2.csv', index=False, float_format='%.2f')



# Ai nevoie de factor_analyzer (pip install factor_analyzer)
from factor_analyzer.factor_analyzer import calculate_kmo

# Selectează doar valorile CAEN (fără SIRUTA)
X = df_ang[caen_cols]
kmo_all, kmo_model = calculate_kmo(X)
print("KMO individual:", kmo_all)
print("KMO model:", kmo_model)

from factor_analyzer import FactorAnalyzer

# 1. Creează modelul
fa = FactorAnalyzer(rotation=None)
fa.fit(X)

# 2. Calculează scorurile
scores = fa.transform(X)
scoruri_df = pd.DataFrame(scores, columns=[f'F{i+1}' for i in range(scores.shape[1])])
scoruri_df.insert(0, 'SIRUTA', df_ang['SIRUTA'])
scoruri_df.to_csv('f.csv', index=False, float_format='%.3f')

import matplotlib.pyplot as plt

plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Scoruri factoriale pe primii doi factori')
plt.show()


