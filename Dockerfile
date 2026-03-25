FROM continuumio/miniconda3

# Créer environnement conda
RUN conda create -n gsk3b python=3.10 -y

# Utiliser cet environnement
SHELL ["conda", "run", "-n", "gsk3b", "/bin/bash", "-c"]

# Installer RDKit (clé du problème)
RUN conda install -c conda-forge rdkit -y

# Installer autres dépendances
RUN pip install streamlit scikit-learn xgboost joblib pandas numpy

# Copier fichiers
WORKDIR /app
COPY . /app

# Lancer app
CMD ["conda", "run", "-n", "gsk3b", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
