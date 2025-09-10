import matplotlib.pyplot as plt

class MRLS:

    def __init__(self, x, y, x_label="", y_label="", dataset_title="", decimal_precision = 100):
        if len(x) != len(y):
            raise AssertionError(f"X não possui a mesma quantidade de valores que Y: X:{len(x)} -- Y:{len(y)}")


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

        print(f"X_: {self.x_}")
        print(f"Y_: {self.y_}")

        self.Sxy = round(sum([xi*yi for xi, yi in zip(self.x, self.y)]) - (self.n*self.x_*self.y_), self.decimal_precision)
        self.Sxx = round(sum([xi**2 for xi in self.x]) - (self.n*(self.x_**2)), self.decimal_precision)
        self.Syy = round(sum([yi**2 for yi in self.y]) - (self.n*(self.y_**2)), self.decimal_precision)

        print(f"Sxy: {self.Sxy}")
        print(f"Sxx: {self.Sxx}")
        print(f"Syy: {self.Syy}")

        self.corr = self.correlacao()

        self.b1 = round(self.Sxy / self.Sxx, self.decimal_precision)
        self.b0 = round(self.y_ - self.b1*self.x_, self.decimal_precision)

        print(f"b1: {self.b1}")
        print(f"b0: {self.b0}")

        self.SQTot = round(sum([(yi-self.y_)**2 for yi in self.y]), self.decimal_precision)
        self.SQRes = round(self.SQE(), self.decimal_precision)
        self.SQReg = round(sum([(self(xi) - self.y_)**2 for xi in self.x]), self.decimal_precision)

        print(f"SQTot {self.SQTot}")
        print(f"SQReg {self.SQReg}")
        print(f"SQRes {self.SQRes}")
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

        print(f"R2: {self.R2}")

        self.fracao_explicada = self.R2
        self.fracao_nao_explicada = 1 - self.R2


        self.print_header = f"{self.x_label} -|- {self.y_label}"

        #################################
        ### Seção 2 ###############
        ###  #######
        #################################

        self.QMRes = self.SQRes / (self.n - 2)
        self.var_b1 = self.QMRes / self.Sxx

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
    
    def SQE(self):
        sqe = sum([(yi - self(xi))**2 for xi, yi in zip(self.x, self.y)])
        return sqe
    
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

    def table(self):
        rtrn = self.print_header + "\n"
        for xi, yi in zip(self.x, self.y):
            rtrn += f"{xi} -|- {yi}\n"
        
        return rtrn

    def correlacao(self):
        from math import sqrt

        return round(self.Sxy / (sqrt(self.Sxx)*sqrt(self.Syy)), self.decimal_precision)