# Instrucciones de instalación y ejecución

## Instalación

1. Instala las dependencias desde la carpeta `apilab`:
   ```
   pip install -r requirements.txt
   ```

## Ejecución con Docker

1. Construye la imagen de Docker:
   ```
   docker build -t apilab:latest .
   ```

2. Inicia la aplicación usando el script:
   ```
   ./startdocker.sh
   ```

Esto dejará la aplicación ejecutándose en el puerto **8000**.

## Ejecución local (opcional)

Si prefieres ejecutar la API localmente sin Docker, se puede ejecutar desde el diretorio laboretario8:
```
python3 main.py
```

La aplicación estará disponible en [http://localhost:8000](http://localhost:8000).

---

**Nota:** Asegúrate de tener Docker y Python 3 instalados en tu sistema.