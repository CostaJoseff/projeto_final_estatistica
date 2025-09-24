# install.packages("readxl")
library("readxl")

# 1. Ler o arquivo Excel (primeira planilha por padrão)
df <- read_excel("dados_projeto.xlsx")
df <- na.omit(df)
num_cols <- sapply(df, is.numeric)
df <- df[apply(df[, num_cols], 1, function(row) all(row >= 0)), ]
print(nrow(df))


# 2. Ajustar modelo MRLM (todas as variáveis menos a resposta como preditoras)
modelo <- lm(Investimento_marketing ~ ., data=df)

plot(modelo)

all <- lm(Investimento_marketing ~ ., data=df)
backward <- step(all, direction='backward', scope=formula(all), trace=1)