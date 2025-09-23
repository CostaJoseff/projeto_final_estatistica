from statsmodels.nonparametric.smoothers_lowess import lowess
from IPython.display import display, HTML
from scipy.stats import t, probplot, f
import matplotlib.pyplot as plt
from math import sqrt, isnan
import pandas as pd
import numpy as np

class MRLS:

    def __init__(self, x, y, x_label="", y_label="", dataset_title="", decimal_precision = 100):
        if len(x) != len(y):
            raise AssertionError(f"X não possui a mesma quantidade de valores que Y: X:{len(x)} -- Y:{len(y)}")

        xy = [(xi, yi) for xi, yi in zip(x, y) if not (isnan(xi) or isnan(yi))]
        x, y = zip(*xy)
        x, y = list(x), list(y)

        #################################
        ### Seção inicial ###############
        ### Calculo de parametros #######
        #################################

        self.decimal_precision = decimal_precision
        self.x_label = x_label
        self.y_label = y_label
        self.dataset_title = dataset_title

        self.x: list = x
        self.y: list = y
        self.n = len(x)

        self.x_ = round(sum(x) / self.n, self.decimal_precision)
        self.y_ = round(sum(y) / self.n, self.decimal_precision)

        # print(f"X_: {self.x_}")
        # print(f"Y_: {self.y_}")

        self.Sxy = round(sum([xi*yi for xi, yi in zip(self.x, self.y)]) - (self.n*self.x_*self.y_), self.decimal_precision)
        self.Sxx = round(sum([xi**2 for xi in self.x]) - (self.n*(self.x_**2)), self.decimal_precision)
        self.Syy = round(sum([yi**2 for yi in self.y]) - (self.n*(self.y_**2)), self.decimal_precision)

        # print(f"Sxy: {self.Sxy}")
        # print(f"Sxx: {self.Sxx}")
        # print(f"Syy: {self.Syy}")

        self.corr = self.correlacao()

        self.b1 = round(self.Sxy / self.Sxx, self.decimal_precision)
        self.b0 = round(self.y_ - self.b1*self.x_, self.decimal_precision)

        # print(f"b1: {self.b1}")
        # print(f"b0: {self.b0}")

        self.SQTot = round(sum([(yi-self.y_)**2 for yi in self.y]), self.decimal_precision)
        self.SQRes = round(sum([(yi - self(xi))**2 for xi, yi in zip(self.x, self.y)]), self.decimal_precision)
        self.SQReg = round(sum([(self(xi) - self.y_)**2 for xi in self.x]), self.decimal_precision)

        # print(f"SQTot {self.SQTot}")
        # print(f"SQReg {self.SQReg}")
        # print(f"SQRes {self.SQRes}")
        # print("\n" + str(self.SQRes + self.SQReg))

        # assert self.SQTot == (self.SQRes + self.SQReg)

        self.R2 = round(self.SQReg / self.SQTot, self.decimal_precision)
        # R2_2 = round((self.SQTot - self.SQRes) / self.SQTot, self.decimal_precision)
        # R2_3 = round(1 - (self.SQRes / self.SQTot), self.decimal_precision)

        # print(R2_1 == R2_2 == R2_3)
        # print(R2_1)
        # print(R2_2)
        # print(R2_3)

        # assert R2_1 == R2_2 == R2_3
        # assert R2_1 <= 1 and R2_1 >= 0
        # self.R2 = R2_1

        # print(f"R2: {self.R2}")

        self.fracao_explicada = self.R2
        self.fracao_nao_explicada = 1 - self.R2


        self.print_header = f"{self.x_label} -|- {self.y_label}"

        ###################################
        ########### Seção 2 ###############
        ### IC e teste de hipoteses #######
        ###################################

        self.QMRes = round(self.SQRes / (self.n - 2), decimal_precision)
        self.QMReg = round(self.SQReg, decimal_precision)
        self.var_b1 = round(self.QMRes / self.Sxx, decimal_precision)
        self.var_b0 = round(self.QMRes*((1/self.n) + ((self.x_**2)/self.Sxx)),decimal_precision)

        alpha = 0.05
        gl = self.n - 2
        t_critico = t.ppf(1 - alpha/2, gl)
        self.std_err = sqrt(self.var_b1)
        self.std_err_b0 = sqrt(self.var_b0)
        self.std_err_resitual = sqrt(self.QMRes)

        self.IC_b1 = (
            round(self.b1 - t_critico*self.std_err, decimal_precision), 
            round(self.b1 + t_critico*self.std_err, decimal_precision)
        )

        self.IC_b0 = (
            round(self.b0 - t_critico*self.std_err_b0, decimal_precision), 
            round(self.b0 + t_critico*self.std_err_b0, decimal_precision)
        )
        
        self.t = round(self.b1 / self.std_err, decimal_precision)
        self.p = round((1 - t.cdf(abs(self.t), gl))*2, decimal_precision)

        self.t_b0 = round(self.b0 / self.std_err_b0,decimal_precision)
        self.p_b0 = round((1 - t.cdf(abs(self.t_b0), gl)) * 2, decimal_precision)

        self.F_Star = round(self.QMReg / self.QMRes, decimal_precision)
        self.F_critico = round(f.ppf(1 - alpha, 1, self.n - 2), decimal_precision)
        self.p_f = round(1 - f.cdf(self.F_Star, 1, self.n - 2), decimal_precision)



    def __call__(self, data):
        if isinstance(data, list):
            return self.__batch_call__(data)

        if not isinstance(data, int) and not isinstance(data, float):
            raise AssertionError(f"O input deve ser int ou float. O tipo do input é {type(data)}")

        return self.b0 + self.b1*data
    
    def __batch_call__(self, data):
        for i in range(len(data)):
            if not isinstance(data[i], int) and not isinstance(data[i], float):
                raise AssertionError(f"O elemento do índice {i} não é do tipo int ou float. O tipo é {type(data[i])}")
            
        return [self(d) for d in data]
    
    def scatter_plot(self, show_model=False):
        plt.figure()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.dataset_title)
        plt.scatter(self.x, self.y)

        if show_model:
            min_x = min(self.x)
            max_x = max(self.x)
            plt.plot([min_x, max_x], [self(min_x), self(max_x)], color="g")

        plt.show()

    def __str__(self):
        return f"b0: {self.b0} -|- b1:{self.b1}"

    def tabela_de_valores(self):
        data = {
            self.x_label: self.x,
            self.y_label: self.y,
        }
        
        self.html_display(data)

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

    def tabela_de_valores_reais_e_preditos(self):
        data = {
            self.x_label: self.x,
            self.y_label: self.y,
            "y predito": self.__batch_call__(self.x)
        }
        
        self.html_display(data)

    def residual_plot(self):
        fitted = self.__batch_call__(self.x)
        residuals = [yi - fi for yi, fi in zip(self.y, fitted)]

        fig, (ax, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(25, 5))

        # ---------- 1. Residuals vs Fitted ----------
        ax.scatter(fitted, residuals)
        smooth = lowess(residuals, fitted)
        ax.plot(smooth[:,0], smooth[:,1], color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Valores Ajustados (Ŷ)')
        ax.set_ylabel('Resíduos (e)')
        ax.set_title('Gráfico de Resíduos vs Valores Ajustados')

        # ---------- 2. Q-Q Plot ----------
        residuals_std = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
        (osm, osr), (slope, intercept, r) = probplot(residuals_std, dist="norm")
        ax2.scatter(osm, osr)
        ax2.plot(osm, slope*osm + intercept, color='red', linestyle='--')
        ax2.set_xlabel('Quantis Teóricos')
        ax2.set_ylabel('Residuais Padronizados')
        ax2.set_title('Q-Q Plot dos Resíduos Padronizados')

        # ---------- 3. Scale-Location Plot ----------
        y_vals = np.sqrt(np.abs(residuals_std))
        ax3.scatter(fitted, y_vals)
        smooth2 = lowess(y_vals, fitted)
        ax3.plot(smooth2[:,0], smooth2[:,1], color='red', linestyle='--')
        ax3.set_xlabel('Valores Ajustados (Ŷ)')
        ax3.set_ylabel('Raiz Quadrada dos Resíduos Padronizados')
        ax3.set_title('Homocedasticidade dos Resíduos')

        # ---------- 4. Residuals vs Leverage ----------
        x_array = np.array(self.x)
        h = 1/self.n + ((x_array - self.x_)**2) / self.Sxx
        ax4.scatter(h, residuals_std)
        smooth3 = lowess(residuals_std, h)
        ax4.plot(smooth3[:,0], smooth3[:,1], color='red', linestyle='--')
        ax4.set_xlabel('Alavancagem (h)')
        ax4.set_ylabel('Resíduos Padronizados')
        ax4.set_title('Resíduos Padronizados vs Alavancagem')

        fig, (ax, ax2, ax3, ax4)
        fig.show()

    def sumario_completo(self):
        summary = f"Modelo de Regressão Linear Simples\n"
        summary += f"{self.print_header}\n\n"
        summary += f"y = b0 + b1*x\n"
        summary += f"b0 = {self.b0}\n"
        summary += f"b1 = {self.b1}\n\n"
        summary += f"Sxy = {self.Sxy}\n"
        summary += f"Sxx = {self.Sxx}\n"
        summary += f"Syy = {self.Syy}\n"
        summary += f"Correlação = {self.corr}\n\n"
        summary += f"SQTot = {self.SQTot}\n"
        summary += f"SQReg = {self.SQReg}\n"
        summary += f"SQRes = {self.SQRes}\n\n"
        summary += f"R2 = {self.R2}\n"
        summary += f"Fração explicada: {self.fracao_explicada} -> {self.fracao_explicada*100:.2f}%\n"
        summary += f"Fração não explicada: {self.fracao_nao_explicada} -> {self.fracao_nao_explicada*100:.2f}%\n\n"
        summary += f"QMRes = {self.QMRes}\n"
        summary += f"Var(b1) = {self.var_b1}\n"
        summary += f"StdErr(b1) = {self.std_err}\n"
        summary += f"IC_b1 = {self.IC_b1}\n"
        summary += f"t = {self.t}\n"
        summary += f"p-valor = {self.p}\n"
        summary += f"F* = {self.F_Star}\n"
        summary += f"F critico = {self.F_critico}\n"

        print(summary)

    def correlacao(self):
        return round(self.Sxy / (sqrt(self.Sxx)*sqrt(self.Syy)), self.decimal_precision)
    
    def sumario_estatistico(self):
        dados = {
            "Coeficiente": ["β0 (Interceptor)", f"β1 {self.x_label}"],
            "Estimativa": [self.b0, self.b1],
            "Erro padrão": [self.std_err_b0, self.std_err],
            "T": [self.t_b0, self.t],
            "p-valor": [self.p_b0, self.p]
        }

        self.html_display(pd.DataFrame(dados))

        print("\n")
        print(f"Erro padrão residual: {self.std_err_resitual} com gl={self.n-2}")
        print(f"R2: {self.R2} -> Adequação de {self.R2*100:.2f}%")
        print(f"F: {self.F_Star} com gl = 1 e {self.n-2} || p-valor: {self.p_f}")

        print("\n")

        anova = {
            " ": [self.x_label, "Residos"],
            "GL": [1, self.n-2],
            "SQ": [self.SQReg, self.SQRes],
            "QM": [self.SQReg, self.QMRes],
            "F": [self.F_Star, " "],
            "p-valor": [self.p_f, " "]
        }
        
        self.html_display(pd.DataFrame(anova))
