from typing_extensions import Literal
from statsmodels.nonparametric.smoothers_lowess import lowess
from IPython.display import display, HTML
from scipy.stats import probplot
import matplotlib.pyplot as plt
from modeloSimples import MRLS
import statsmodels.api as sm
import math, itertools
from pprint import pprint
import pandas as pd
import numpy as np

class MRLM:

    def __init__(self, X, y, X_labels=[], y_label="y", dummies=[]):
        if X.shape[0] != y.shape[0]:
            raise AssertionError(f"O número de linhas de X e y devem ser iguais: X: {X.shape[0]} | y: {y.shape[0]}")

        self.original_X = X
        self.original_y = y

        df_comb = pd.concat([X, y], axis=1)
        df_clean = df_comb.dropna()


        self.y_label = y_label
        self.X_labels = X_labels

        self.dummies = dummies

        self.X = df_clean.iloc[:, :-1]
        self.y = df_clean.iloc[:, -1]
        self.n = len(y)
        self.k = X.shape[1]  # número de preditores
        self.observacoes_deletadas = y.shape[0] - self.y.shape[0]

        # Ajusta o modelo de regressão linear múltipla
        self.modelo: sm.OLS = sm.OLS(self.y, sm.add_constant(self.X)).fit()

        # Coeficientes do modelo
        self.coeficientes = self.modelo.params

        # Estatísticas do modelo
        self.R2 = self.modelo.rsquared
        self.R2_ajustado = self.modelo.rsquared_adj
        self.erro_padrao_residual = np.sqrt(self.modelo.mse_resid)
        self.F = self.modelo.fvalue
        self.p_F = self.modelo.f_pvalue
        self.gl = self.modelo.df_model
        self.glr = self.modelo.df_resid

        # Estatísticas dos coeficientes
        self.std_err = self.modelo.bse
        self.t_values = self.modelo.tvalues
        self.p_values = self.modelo.pvalues
        self.aic = self.modelo.aic
        self.bic = self.modelo.bic

    def html_display(self, data):
        if not (isinstance(data, pd.DataFrame) or isinstance(data, dict)):
            raise AssertionError(f"O input deve ser um DataFrame ou um dicionário. O tipo do input é {type(data)}")

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        display(
            HTML(
                data
                .head(10)
                .to_html(border=1, index=False, justify="center")
            )
        )

    def sumario_sm(self):
        print(self.modelo.summary())

    def plot_correlacao_em_pares(self, tamanho_do_plot=4, show_model=False):
        plot_elements = self.X_labels + [self.y_label]
        data = pd.concat([self.X, self.y], axis=1)

        # Cria todas as combinações únicas de pares (sem repetição)
        combinacoes = list(itertools.combinations(plot_elements, 2))
        matriz_combinacoes = []
        anterior = combinacoes[0][0]
        linha = []
        for a, b in combinacoes:
            if a != anterior:
                anterior = a
                if len(linha) < len(plot_elements)-1:
                    linha = linha + ([None]*(len(plot_elements)-len(linha)-1))

                matriz_combinacoes.append(linha)
                linha = []
            
            linha.append((a, b))
        linha = linha + [None]*(len(plot_elements)-1-len(linha))
        matriz_combinacoes.append(linha)
        # Encontra o melhor layout (linhas, colunas)
        linhas = len(plot_elements)-1
        colunas = linhas

        # Cria os subplots
        fig, axes = plt.subplots(nrows=linhas, ncols=colunas, figsize=(colunas*tamanho_do_plot, linhas*tamanho_do_plot))

        # Garante que axes seja 2D
        axes = np.array(axes)
        if linhas == 1:
            axes = axes.reshape(1, -1)
        elif colunas == 1:
            axes = axes.reshape(-1, 1)

        # axes = axes.flatten()  # Agora é 1D para iteração simples

        # Plota os pares
        for i_idx in range(len(matriz_combinacoes)):
            for j_idx in range(len(matriz_combinacoes[i_idx])):
                if matriz_combinacoes[i_idx][j_idx] is None:
                    idx = (i_idx, j_idx)
                    axes[idx].axis('off')
                    continue
                
                x_label, y_label = matriz_combinacoes[i_idx][j_idx]
                idx = (i_idx, j_idx)
                ax = axes[idx]
                x_data = data[x_label].tolist()
                y_data = data[y_label].tolist()
                ax.scatter(x_data, y_data)
                if show_model:
                    x_min = min(x_data)
                    x_max = max(x_data)

                    simple_model = MRLS(x_data, y_data)

                    ax.plot([x_min, x_max], [simple_model(x_min), simple_model(x_max)], color='g', linewidth=4)

                ax.set_title(f"{x_label} vs {y_label}")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

        # Esconde plots extras (caso existam)
        # for idx in range(len(combinacoes), len(axes)):
        #     axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    
    def reshape_lista(self, lista, linhas, colunas):
        matriz = []
        for i in range(linhas):
            inicio = i * colunas
            fim = inicio + colunas
            linha = lista[inicio:fim]
            if linha:
                matriz.append(linha)
        return matriz
    
    def sumario_em_pares(self):
        data = {
            "Coeficiente": [self.y_label + " (Intercept)"] + self.X_labels,
            "Estimativa": self.coeficientes.values,
            "Erro padrão": self.std_err.values,
            "T": self.t_values.values,
            "p-valor": [round(p, 3) for p in self.p_values.values]
        }

        self.html_display(data)

        print("\n")
        print(f"Erro padrão residual: {self.erro_padrao_residual} com gl={self.glr}\n")
        print(f"R2: {self.R2} -|- R2-ajustado: {self.R2_ajustado}")
        print(f"F: {self.F} com gl = {len(self.X_labels)} e {self.glr}")
        print(f"({self.observacoes_deletadas} observações deletadas por NAN)")

    def residual_plots(self):
        fitted = self.modelo.fittedvalues
        residuals = self.modelo.resid
        residuals_std = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)

        influence = self.modelo.get_influence()
        leverage = influence.hat_matrix_diag  # alavancagem

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(25, 5))

        # ---------- 1. Resíduos vs Ajustados ----------
        ax1.scatter(fitted, residuals)
        smooth = lowess(residuals, fitted)
        ax1.plot(smooth[:,0], smooth[:,1], color='red', linestyle='--')
        ax1.axhline(0, color='gray', linestyle='--')
        ax1.set_xlabel('Valores Ajustados (Ŷ)')
        ax1.set_ylabel('Resíduos')
        ax1.set_title('Resíduos vs Valores Ajustados')

        # ---------- 2. Q-Q Plot ----------
        (osm, osr), (slope, intercept, r) = probplot(residuals_std, dist="norm")
        ax2.scatter(osm, osr)
        ax2.plot(osm, slope*osm + intercept, color='red', linestyle='--')
        ax2.set_xlabel('Quantis Teóricos')
        ax2.set_ylabel('Resíduos Padronizados')
        ax2.set_title('Q-Q Plot')

        # ---------- 3. Scale-Location ----------
        y_vals = np.sqrt(np.abs(residuals_std))
        ax3.scatter(fitted, y_vals)
        smooth2 = lowess(y_vals, fitted)
        ax3.plot(smooth2[:,0], smooth2[:,1], color='red', linestyle='--')
        ax3.set_xlabel('Valores Ajustados (Ŷ)')
        ax3.set_ylabel('√|Resíduos Padronizados|')
        ax3.set_title('Homocedasticidade dos Resíduos')

        # ---------- 4. Resíduos vs Leverage ----------
        ax4.scatter(leverage, residuals_std)
        smooth3 = lowess(residuals_std, leverage)
        ax4.plot(smooth3[:,0], smooth3[:,1], color='red', linestyle='--')
        ax4.set_xlabel('Alavancagem (h)')
        ax4.set_ylabel('Resíduos Padronizados')
        ax4.set_title('Resíduos vs Alavancagem')

        plt.tight_layout()
        plt.show()

    def selecionar_melhor_combinacao_de_variaveis(self, print_relatorio=False, criterio: Literal["AIC", "BIC", "R2", "R2_aj"]="AIC"):
        if criterio.upper() not in ["AIC", "BIC", "R2", "R2_AJ"]:
            raise AssertionError("Critério deve ser AIC, BIC, R2 ou R2_aj")
    
        AIC, BIC, R2, R2_aj = [False] * 4
        match criterio:
            case "AIC":
                AIC = True
            case "BIC":
                BIC = True
            case "R2":
                R2 = True
            case "R2_aj":
                R2_aj = True

        def update_vars_(vars_, novo_modelo: MRLM):
            vars_["melhor_atual"] = "--".join(novo_modelo.X_labels)
            vars_["melhor"] = novo_modelo
            vars_["melhores"].append(novo_modelo)
            vars_["melhor_AIC"] = novo_modelo.aic
            vars_["melhor_BIC"] = novo_modelo.bic
            vars_["melhor_R2"] = novo_modelo.R2
            vars_["melhor_R2_ajus"] = novo_modelo.R2_ajustado
            vars_["melhor_F"] = novo_modelo.F
            vars_["melhor_p"] = novo_modelo.p_F

            return vars_

        dummies_find = []
        for dum in self.dummies:
            sub_list = [i for i in self.X_labels if dum in i]
            dummies_find.append(sub_list)

        todas_combinacoes = []
        for tamanho in range(1, len(self.X_labels)+1):
            combinacao_atual = list(itertools.combinations(self.X_labels, tamanho))

            for ca in combinacao_atual:
                passou = True
                for dum in dummies_find:
                    todos = all(i in ca for i in dum)
                    nenhum = all(i not in ca for i in dum)

                    if not(todos or nenhum):
                        passou = False
                        break

                if passou:
                    todas_combinacoes.append(ca)
        
        melhor = None
        melhor_AIC = float("inf")
        melhor_BIC = float("inf")
        melhor_R2 = 0
        melhor_R2_ajus = 0
        melhor_F = 0
        melhor_p = 2

        vars_ = {
            "melhor_atual": "",
            "melhor": melhor,
            "melhores": [],
            "melhor_AIC": melhor_AIC,
            "melhor_BIC": melhor_BIC,
            "melhor_R2": melhor_R2,
            "melhor_R2_ajus": melhor_R2_ajus,
            "melhor_F": melhor_F,
            "melhor_p": melhor_p
        }

        criterio_str = criterio
        tot_combinacoes = len(todas_combinacoes)
        for i, combinacao in enumerate(todas_combinacoes):
            if i % 300 == 0:
                print(f"\r{i} / {tot_combinacoes} considerando {criterio_str}", end="     ")
            X = self.original_X[list(combinacao)]
            novo_modelo = MRLM(X, self.original_y, X.columns.to_list(), self.original_y.name)

            bom_AIC = vars_["melhor_AIC"] > novo_modelo.aic
            bom_BIC = vars_["melhor_BIC"] > novo_modelo.bic
            bom_R2 = vars_["melhor_R2"] < novo_modelo.R2
            bom_R2_ajust = vars_["melhor_R2_ajus"] < novo_modelo.R2_ajustado
            criterio = bom_AIC if AIC else bom_BIC if BIC else bom_R2 if R2 else bom_R2_ajust

            if criterio:
                vars_ = update_vars_(vars_, novo_modelo)

        print(f"\r{i} / {tot_combinacoes} considerando {criterio_str}")
        if print_relatorio:
            pprint(vars_)

            print()
            for novo_modelo in vars_["melhores"]:
                relat = "*"*20+"\n\n"
                relat += "Modelo\n"
                relat += f"{' - '.join(novo_modelo.X_labels)}\n"
                relat += f"AIC: {novo_modelo.aic} -- BIC: {novo_modelo.bic}\n"
                relat += f"R2: {novo_modelo.R2} -- R2_ajus: {novo_modelo.R2_ajustado}\n"
                relat += f"F: {novo_modelo.F} -- p-valor: {novo_modelo.p_F}\n"
                relat += "*"*20+"\n\n"
                print(relat)

        return vars_["melhor"]