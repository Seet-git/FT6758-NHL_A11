# Installer les packages

Dans le fichier `environment.yml` (`environment_linux.yml` ou `environment_windows.yml`) situé à la racine du projet, spécifiez le nom de l'environnement souhaité.

Exécutez la commande suivante pour créer l'environnement :

Exemple pour windows
```bash
  conda env create -f environment_windows.yml
```

# Définition des variables d'environnement

Dans le fichier `milestone3/serving/.env`, ajoutez votre clé WandB: 

```python
WANDB_API_KEY=your_wand_db_api
```

# Lancer le serveur

Exécutez la commande suivante :
```bash
  cd milestone3/serving
  waitress-serve --listen=localhost:5000 app:app
```

# Lancer le client

Exécutez le fichier : `python main_Milestone-3.py` 
