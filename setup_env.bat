@echo off
REM ============================================================
REM  setup_env.bat
REM  Crea un entorno virtual con Python 3.11 e instala
REM  todas las dependencias del Taller 6 CNN.
REM
REM  REQUISITO: tener Python 3.11 instalado.
REM  Descarga: https://www.python.org/downloads/release/python-3119/
REM ============================================================

echo.
echo [1/4] Verificando Python 3.11...
py -3.11 --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo  ERROR: Python 3.11 no encontrado.
    echo  Descargalo desde: https://www.python.org/downloads/release/python-3119/
    echo  Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)
py -3.11 --version
echo  OK

echo.
echo [2/4] Creando entorno virtual en .\venv ...
py -3.11 -m venv venv
IF ERRORLEVEL 1 ( echo ERROR creando venv & pause & exit /b 1 )
echo  OK

echo.
echo [3/4] Instalando dependencias (puede tardar unos minutos)...
venv\Scripts\pip install --upgrade pip -q
venv\Scripts\pip install -r requirements.txt
IF ERRORLEVEL 1 ( echo ERROR instalando dependencias & pause & exit /b 1 )
echo  OK

echo.
echo [4/4] Registrando kernel de Jupyter...
venv\Scripts\python -m ipykernel install --user --name taller6_cnn --display-name "Python 3.11 (Taller6 CNN)"
IF ERRORLEVEL 1 ( echo ERROR registrando kernel & pause & exit /b 1 )
echo  OK

echo.
echo ============================================================
echo  Entorno listo. Para abrir el notebook ejecuta:
echo.
echo    venv\Scripts\activate
echo    jupyter notebook Taller6_CNN_CIFAR10.ipynb
echo.
echo  En Jupyter selecciona el kernel: "Python 3.11 (Taller6 CNN)"
echo ============================================================
echo.
pause
