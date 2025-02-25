import pandas as pd
import glob
import numpy as np
import hashlib
import matplotlib.pyplot as plt


# === ETAPE 1 : Chargement, nettoyage et ajout de l'empreinte ===

def load_and_clean_data(path, dates):
    """
    Charge et nettoie les données CSV.
    """
    csv_files = sorted(glob.glob(f"{path}*{dates}*.csv"))
    if not csv_files:
        print("[!] Aucun fichier trouvé. Vérifiez le chemin et les dates.")
        return None

    print(f"Chargement de {len(csv_files)} fichiers CSV...")
    df = pd.concat((pd.read_csv(file, sep=";", decimal=".") for file in csv_files), ignore_index=True)

    # Conversion du champ datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Suppression des doublons basés sur 'src' et 'datetime'
    df = df.drop_duplicates(subset=["src", "datetime"])

    # Filtrage du RSSI (valeurs typiques entre -100 et 0 dBm)
    df = df[(df["rssi"] >= -100) & (df["rssi"] <= 0)]

    # Normalisation du RSSI pour d'éventuelles analyses futures
    df["rssi_norm"] = (df["rssi"] - df["rssi"].mean()) / df["rssi"].std()

    # Tri par date
    df = df.sort_values(by="datetime").reset_index(drop=True)

    print(f"✅ Nettoyage terminé : {len(df):,} entrées valides.")
    return df


def compute_fingerprint(row):
    """
    Calcule une empreinte (fingerprint) pour chaque enregistrement en combinant
    plusieurs champs pour pallier à la randomisation de l'adresse MAC.
    """
    parts = []
    for col in ['src_vendor', 'oui', 'randomized', 'ch_freq', 'seq_num', 'FCfield', 'dot11elt']:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).strip())
    fingerprint_str = '_'.join(parts)
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    return fingerprint


def add_fingerprint(df):
    """
    Ajoute une colonne 'fingerprint' au DataFrame.
    """
    df['fingerprint'] = df.apply(compute_fingerprint, axis=1)
    return df


# === ETAPE 2 : Agrégation des données pour extraire des features ===

def prepare_aggregated_features(df, interval='10min'):
    """
    Agrège les données par intervalle de temps pour extraire des caractéristiques utiles.

    Pour chaque intervalle, on calcule :
      - unique_devices: nombre d'appareils uniques (basé sur 'fingerprint')
      - total_requests: nombre total de probe requests
      - avg_rssi: moyenne du RSSI
      - std_rssi: écart-type du RSSI
      - occupancy: occupation moyenne (valeur de référence présente dans le CSV)

    :param df: DataFrame nettoyé et enrichi
    :param interval: Intervalle d'agrégation (ex : '10min')
    :return: DataFrame agrégé
    """
    agg_df = df.groupby(pd.Grouper(key='datetime', freq=interval)).agg(
        unique_devices=('fingerprint', 'nunique'),
        total_requests=('fingerprint', 'count'),
        avg_rssi=('rssi', 'mean'),
        std_rssi=('rssi', 'std'),
        occupancy=('occupancy', 'mean')
        # On suppose que la colonne 'occupancy' est présente et représente la ground truth
    ).reset_index()

    # On retire les intervalles où aucune mesure d'occupation n'est disponible
    agg_df = agg_df.dropna(subset=['occupancy'])

    return agg_df


# === ETAPE 3 : Modèle de Machine Learning pour la détection de l'occupation ===

def train_and_evaluate_model(agg_df):
    """
    Entraîne un modèle de régression (ici LinearRegression) pour prédire l'occupation.

    Utilise les features agrégées pour prédire la colonne 'occupancy' (ground truth).
    Affiche les métriques d'évaluation et trace la comparaison entre valeurs réelles et prédites.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Sélection des features et de la target
    features = ['unique_devices', 'total_requests', 'avg_rssi', 'std_rssi']
    target = 'occupancy'

    X = agg_df[features]
    y = agg_df[target]

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=== Évaluation du modèle ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.2f}")

    # Affichage d'un graphique comparant les valeurs réelles et prédites
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Occupancy Réelle")
    plt.ylabel("Occupancy Prédite")
    plt.title("Comparaison Occupancy Réelle vs. Prédite")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


# === Pipeline Principal ===

if __name__ == "__main__":
    # Définir le chemin et la période des fichiers CSV
    path = "../DATA/sc6-61/_CSV/position_1/"
    dates = "2024-03-"

    # 1. Chargement et nettoyage des données
    df = load_and_clean_data(path, dates)
    if df is None:
        exit(1)

    # 2. Ajout de l'empreinte pour identifier chaque appareil de façon robuste
    df = add_fingerprint(df)

    # (Optionnel) Affichage d'un aperçu des données
    print("\nAperçu des données enrichies :")
    print(df[['datetime', 'src', 'fingerprint', 'rssi', 'occupancy']].head())

    # 3. Agrégation des données pour extraire les caractéristiques par intervalle
    agg_df = prepare_aggregated_features(df, interval='10min')
    print("\nAperçu des features agrégées :")
    print(agg_df.head())

    # 4. Entraînement et évaluation du modèle de Machine Learning
    model = train_and_evaluate_model(agg_df)
