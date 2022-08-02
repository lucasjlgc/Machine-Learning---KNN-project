import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# 1- Carregando conjunto de dados
# X contem todas as variaveis indepedentes
# Y Contem a variavel depedente "Iphone comprado"

conjuntoDados = pd.read_csv("iphone_banco.csv")

#Obs: iloc[linha, coluna]
x = conjuntoDados.iloc[:,:-1].values #Pega todas as linhas e quase todas as colunas, menos a ultima
y = conjuntoDados.iloc[:,3].values #Pega todas as linhas e a ultima coluna apenas. (3 no caso)

# 2- Convertendo o Gênero em Numero utilizando a classe LabelEncoder
label_encoder = LabelEncoder()
x[:,0] = label_encoder.fit_transform(x[:,0]) #Converte todas as linhas da coluna 0(gender) em numero e atribui a ela mesmo.

#Homem = 1, mulher = 0

# Convertendo x para tipo de dado Float
x = np.vstack(x[:,:]).astype(np.float64)

# 3- Dividir os dados em conjunto de treinamento e teste.
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state=0)

# 4- Dimensionar Recursos com StandardScaler (Necessario dimensionar em KNN)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

# 5- Ajustando o classificador KNN usando KneighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
classificador.fit(X_train,y_train)

# 6- Fazendo Previsões
y_previsto = classificador.predict(X_test)



# 7- Verificando a precisão das previsões comparando os resultados previstos
# com os resultados reais usando matriz de confusão. Para isso importamos o metrics.

#mc = metrics.confusion_matrix(y_test,y_previsto)
#print(mc)
print(pd.crosstab(y_test,y_previsto,rownames=["Real"], colnames=["Predito"], margins=True))
print()
precisao = metrics.precision_score(y_test, y_previsto)
print(f"A precisão é: {precisao}")

acuracia = metrics.accuracy_score(y_test,y_previsto)
print(f"A acuracia é: {acuracia}")

recall = metrics.recall_score(y_test, y_previsto)
print(f"Pontuação do Recall: {recall}")



