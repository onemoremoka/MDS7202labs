import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import load

def create_folders(ds, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    subfolders = ['raw', 'preprocessed', 'splits', 'models']
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    return base_path

def download_data(ds, download_data2=False, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    raw_path = os.path.join(base_path, 'raw')
    os.makedirs(raw_path, exist_ok=True)
    os.system(f"curl -o {os.path.join(raw_path, 'data_1.csv')} https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv")
    if download_data2:
        os.system(f"curl -o {os.path.join(raw_path, 'data_2.csv')} https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv")
    print(f"Archivos descargados en: {raw_path}")

def load_and_merge(ds, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    raw_path = os.path.join(base_path, 'raw')
    files = ['data_1.csv', 'data_2.csv']
    dataframes = []
    for file in files:
        file_path = os.path.join(raw_path, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)
    if dataframes:
        df_merged = pd.concat(dataframes, ignore_index=True)
        preprocessed_path = os.path.join(base_path, 'preprocessed')
        os.makedirs(preprocessed_path, exist_ok=True)
        merged_file_path = os.path.join(preprocessed_path, 'merged.csv')
        df_merged.to_csv(merged_file_path, index=False)
    else:
        raise FileNotFoundError("No se encontraron archivos data_1.csv o data_2.csv en 'raw'")

def split_data(ds, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    preprocessed_path = os.path.join(base_path, 'preprocessed', 'merged.csv')
    splits_path = os.path.join(base_path, 'splits')
    os.makedirs(splits_path, exist_ok=True)
    df = pd.read_csv(preprocessed_path)
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train.to_csv(os.path.join(splits_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(splits_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(splits_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(splits_path, 'y_test.csv'), index=False)

def train_model(ds, model_class, model_name, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    splits_path = os.path.join(base_path, 'splits')
    X_train = pd.read_csv(os.path.join(splits_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(splits_path, "y_train.csv")).values.ravel()
    num_cols = ['Age', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany', 
                'InterviewScore', 'SkillScore', 'PersonalityScore','Gender', 'EducationLevel', 'RecruitmentStrategy'] 
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_class(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    model_path = os.path.join(base_path, 'models', f'{model_name}.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Modelo {model_name} guardado en {model_path}")

def evaluate_models(ds, **kwargs):
    base_path = os.path.join(os.getenv("AIRFLOW_HOME", "/root/airflow"), 'runs', ds)
    splits_path = os.path.join(base_path, 'splits')
    X_test = pd.read_csv(os.path.join(splits_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(splits_path, 'y_test.csv')).squeeze()
    models_path = os.path.join(base_path, 'models')
    model_files = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
    best_model = None
    best_accuracy = 0.0
    best_model_name = None
    for model_file in model_files:
        model = load(os.path.join(models_path, model_file))
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Modelo: {model_file} - Accuracy: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = model_file
    if best_model:
        final_model_path = os.path.join(models_path, 'best_model.joblib')
        joblib.dump(best_model, final_model_path)
        print(f'Mejor modelo: {best_model_name} - Accuracy: {best_accuracy:.4f}')
    else:
        print("No se encontraron modelos para evaluar.")
