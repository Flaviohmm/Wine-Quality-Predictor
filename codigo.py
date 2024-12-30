import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de estilo
sns.set(style="whitegrid")

# Carregar o conjunto de dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Exibir as primeiras linhas do conjunto de dados
print(data.head())

# Gráfico da Distribuição da Qualidade
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=data, hue='quality', palette='viridis')
plt.title('Distribuição da Qualidade dos Vinhos')
plt.xlabel('Qualidade')
plt.ylabel('Contagem')
plt.show()

# Gráfico de Dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(x='fixed acidity', y='alcohol', hue='quality', data=data, palette='viridis')
plt.title('Relação entre Acidez Fixa e teor Alcoólico')
plt.xlabel('Acidez Fixa')
plt.ylabel('Teor Alcoólico')
plt.legend(title='Qualidade')
plt.show()

# Dividir os Dados em Treinamento e Teste
from sklearn.model_selection import train_test_split

X = data.drop('quality', axis=1)    # Características
y = data['quality']                 # Alvo

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e Treinar o Modelo
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Avaliar o Modelo
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Fazer Previsões com Novos Dados
new_wine = pd.DataFrame([[7.4, 0.7, 0.00, 0.04, 0.5, 11.0, 34.0, 0.9968, 3.16, 0.58, 9.4]], 
                        columns=X.columns)
predicted_quality = model.predict(new_wine)
print(f'A qualidade prevista do vinho é: {predicted_quality[0]}')