# Installer les packages

Dans le fichier `environment.yml` situé à la racine du projet, spécifiez le nom de l'environnement souhaité.

Exécutez la commande suivante pour créer l'environnement :

```bash
  conda env create -f environment.yml
```

# Définition des variables d'environnement

Dans le repertoire `milestone3/serving`, ajoutez votre clé WandB: 

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
