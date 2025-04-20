import pandas as pd
from sklearn.preprocessing import StandardScaler

# Chargement des données
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# Filtrer uniquement les colonnes numériques
X_train_numeric = X_train.select_dtypes(include=['float64', 'int'])
X_test_numeric = X_test.select_dtypes(include=['float64', 'int'])

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Sauvegarde
pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)
