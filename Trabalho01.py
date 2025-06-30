import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Simulação de um dataset fictício
data = pd.DataFrame({
    'url_length': [54, 66, 72, 45, 87, 54, 38, 91, 64, 70],
    'has_https': [1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    'has_at_symbol': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    'is_phishing': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
})

# Separar em features (X) e rótulo (y)
X = data.drop('is_phishing', axis=1)
y = data['is_phishing']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Acurácia:", accuracy)
print("Matriz de Confusão:\n", conf_matrix)
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))