from modeloMultiplo import MRLM
import pandas as pd
df = pd.read_excel("dados_projeto.xlsx")
df
estacao_ref = ["", "Baixa", "Media", "Alta"]
estacao_dict = {}
for i, ref in enumerate(estacao_ref):
    estacao_dict[ref] = i

tipo_destino_ref = ['', 'Praia', 'Urbano', 'Campo', 'Cultural']
tipo_destino_dict = {}
for i, ref in enumerate(tipo_destino_ref):
    tipo_destino_dict[ref] = i
df["estacao"] = df["estacao"].replace(estacao_dict)
df["tipo_destino"] = df["tipo_destino"].replace(tipo_destino_dict)
df
y = df.iloc[:, 0] 
X = df.iloc[:, 1:]
X
modelo = MRLM(X, y, X.columns.to_list(), y.name)
modelo.sumario_sm()
modelo.plot_correlacao_em_pares(show_model=True)