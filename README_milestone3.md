# Installer les packages

Dans le fichier environment.yml ici a la racine du projet, mentionner le nom d'environement souhaité, ensuite rouler la commande:

```python
conda env create -f environment.yml
```

# définition des variables d'environement

Dans le repertoire milestone3/serving, créer un fichier .env et y ajouter 
la variable ```WANDB_API_KEY=your_wand_db_api```

# Lancer le serveur

Depuis le repertoire milestone3/serving, rouler cette commande:

waitress-serve --listen=localhost:5000 app:app

# Tester un call HTTP

dans un jupyter notebook ou un fichier python, executer:
```python
import requests

WANDB_PROJECT_NAME = "IFT6758.2024-A11"
WANDB_TEAM_NAME = "youry-macius-universite-de-montreal"

data = {
    "project_name": WANDB_PROJECT_NAME,
    "entity_name": WANDB_TEAM_NAME,
    "model_name": "LogisticRegression_Distance_Angle", # valeurs possibles : LogisticRegression_Distance_Angle, LogisticRegression_Distance
}

r = requests.post("http://127.0.0.1:5000/download_registry_model", json=data)   
```

```python
import requests

data = {
    "distance": [8, 40],
    "angle": [20, 80],
}

r = requests.post("http://127.0.0.1:5000/predict", json=data)
```
