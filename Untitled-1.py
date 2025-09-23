# %%
from modeloMultiplo import MRLM
import pandas as pd

# %%
df = pd.read_excel("dados_projeto.xlsx")
df

# %%
# estacao_ref = ["", "Baixa", "Media", "Alta"]
# estacao_dict = {}
# for i, ref in enumerate(estacao_ref):
#     estacao_dict[ref] = i
# df["estacao"] = df["estacao"].replace(estacao_dict)

# tipo_destino_ref = ['', 'Praia', 'Urbano', 'Campo', 'Cultural']
# tipo_destino_dict = {}
# for i, ref in enumerate(tipo_destino_ref):
#     tipo_destino_dict[ref] = i
# df["tipo_destino"] = df["tipo_destino"].replace(tipo_destino_dict)

# %%
dummies = ["estacao", "tipo_destino"]
df = pd.get_dummies(df, columns=dummies, drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# %%
df

# %%
y_label = "Investimento_marketing"

y = df[y_label]
X = df.drop(columns=y_label)

# %%
modelo = MRLM(X, y, X.columns.to_list(), y.name, dummies)

X = df.drop(columns=["cliques", "taxa_ocupacao_hoteleira", "eventos_anuais", "Investimento_marketing"])
modelo2 = MRLM(X, y, X.columns.to_list(), y.name)

# %%
modelo_melhorado = modelo.selecionar_melhor_combinacao_de_variaveis()

# %%
modelo.sumario_sm()

# %%
modelo_melhorado.sumario_sm()

# %%
# modelo.plot_correlacao_em_pares(show_model=True)

# %%
# modelo.sumario_em_pares()

# %%
# modelo.residual_plots()


