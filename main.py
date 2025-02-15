import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from factor_analyzer import calculate_kmo, FactorAnalyzer
from sklearn.preprocessing import StandardScaler

df_angajati = pd.read_csv("CAEN2_2021_NSAL.csv", index_col=0)
df_popLoc = pd.read_csv("PopulatieLocalitati.csv", index_col=0)
coduri = list(df_angajati)[0:]

#Cerinta1
total = df_angajati.sum()
cerinta1 = (df_angajati / total) * 100
df_cerinta1 = pd.DataFrame(
    data=cerinta1.round(2),
    columns=df_angajati.columns,
    index=df_angajati.index
).to_csv("Cerinta1.csv")

#Cerinta2
def fc(t):
    suma = t[coduri].sum()
    suma1 = t["Populatie"].sum()
    x = (suma * 10000) / suma1
    return pd.Series(x, coduri)


df_merged = df_angajati.merge(df_popLoc, left_index=True, right_index=True)
df_merged.index.name = "Siruta"
df_grouped = df_merged.groupby(by="Judet").apply(func=fc, include_groups=False)
df_grouped.round(2).to_csv("Cerinta2.csv")

#B
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_angajati), index=df_angajati.index, columns=df_angajati.columns)

#1KMO
kmo_all, kmo_model = calculate_kmo(df_scaled)
print(kmo_model) # 0.98 > 0.6, e bun, putem continua

#Scoruri factoriale
fa = FactorAnalyzer(rotation=None, n_factors=len(df_scaled.columns))
fa.fit(df_scaled)

ev, e = fa.get_eigenvalues()
plt.plot(range(1, len(ev) + 1), ev, 'bo-')
plt.show()
#Kaiser
n_factors = sum(ev > 1)
print(n_factors) # 9

fa = FactorAnalyzer(rotation=None, n_factors=n_factors)
scores = fa.fit_transform(df_scaled)

matrix_scores = pd.DataFrame(
    data=scores,
    index=df_scaled.index,
    columns=[f"F{i+1}" for i in range(n_factors)]
)
matrix_scores.to_csv("Matrix_scores.csv")

#Grafic primii 2 factori
seaborn.scatterplot(data=matrix_scores, x="F1", y="F2")
plt.show()

#Extra
#Corelațiile factoriale(corelațiile variabile - factori comuni)
cor_matrix = pd.DataFrame(
    data=fa.loadings_,
    index=df_scaled.columns,
    columns=[f"Factor{i+1}" for i in range(n_factors)]
)
cor_matrix.index.name = "Index"
cor_matrix.to_csv("Correlation.csv")

#Varianța factorilor comuni
table_variance = pd.DataFrame({
    "Var": fa.get_factor_variance()[0],
    "Proc": fa.get_factor_variance()[1],
    "CumProc": fa.get_factor_variance()[2]
})
table_variance.index.name = "Factor"
table_variance.to_csv("Variance.csv")
