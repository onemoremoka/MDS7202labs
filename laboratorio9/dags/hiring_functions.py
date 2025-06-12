import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr
from sklearn.preprocessing import StandardScaler
import io

# 1. Crear carpetas para cada ejecución
def create_folders(**kwargs):
    execution_date = kwargs['ds']  # yyyy-mm-dd
    os.makedirs("dags", exist_ok=True)  # Aseguramos que la carpeta output exista
    base_path = f"dags/{execution_date}"

    raw_path = os.path.join(base_path, "raw")
    splits_path = os.path.join(base_path, "splits")
    models_path = os.path.join(base_path, "models")

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(splits_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)


# 2. Hold-out del dataset
def split_data(**kwargs):
    execution_date = kwargs['ds']
    raw_path = os.path.join("dags", execution_date, "raw")
    splits_path = os.path.join("dags", execution_date, "splits")

    df = pd.read_csv(os.path.join(raw_path, "data_1.csv"))

    X = df.drop("HiringDecision", axis=1)
    y = df["HiringDecision"]

    # genero un hold-out del 80% de los datos para entrenamiento y 20% para test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,random_state=42
    )

    X_train.to_csv(os.path.join(splits_path, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(splits_path, "y_train.csv"), index=False)
    
    X_test.to_csv(os.path.join(splits_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(splits_path, "y_test.csv"), index=False)

    print("Datos divididos correctamente.")


# 3. Preprocesamiento + Entrenamiento Random Forest + Guardar modelo
def preprocess_and_train(**kwargs):
    execution_date = kwargs['ds']
    model_path = os.path.join("dags", execution_date, "models")
    splits_path = os.path.join("dags", execution_date, "splits")
    

    # Cargar datos
    X_train = pd.read_csv(os.path.join(splits_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(splits_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(splits_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(splits_path, "y_test.csv")).values.ravel()

    # Preprocesamiento basico
    numeric_features = ['Age', 'ExperienceYears', 'PreviousCompanies', 'DistanceFromCompany', 
                        'InterviewScore', 'SkillScore', 'PersonalityScore','Gender', 'EducationLevel', 'RecruitmentStrategy'] 

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # Pipeline con modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Entrenamiento
    pipeline.fit(X_train, y_train)

    # Evaluación
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (Contratado): {f1:.4f}")

    # Guardar modelo
    model_path_joblib = os.path.join(model_path, "hiring_model.joblib")
    joblib.dump(pipeline, model_path_joblib)
    print(f"Modelo guardado en: {model_path_joblib}")

#enunciado
def predict_interface(file, model_path):
    pipeline = joblib.load(model_path)
    # ¡Lee el contenido del archivo directamente, sin .name!
    input_data = pd.read_json(io.BytesIO(file.read()))
    predictions = pipeline.predict(input_data)
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]
    return {
        'Predicción': labels[0]
    }

def gradio_interface(ds, **kwargs):
    print(f"Creando interfaz Gradio para el DAG con fecha {ds}")
    model_path = f"dags/{ds}/models/hiring_model.joblib"
    print(f"existe la ruta del modelo: {os.path.exists(model_path)}")
    interface = gr.Interface(
        fn=lambda file: predict_interface(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)