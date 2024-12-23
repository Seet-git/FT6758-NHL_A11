# Installer les packages

Dans le fichier `environment.yml` (`environment_linux.yml` ou `environment_windows.yml`) situé à la racine du projet, spécifiez le nom de l'environnement souhaité.

Exécutez la commande suivante pour créer l'environnement :

Exemple pour windows
```bash
  conda env create -f environment_windows.yml
```

Pour créer un nouveau fichier yml apres l'ajout d'un nouveau package, faire ceci:

```bash
conda env export --no-builds > environment.yml
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

# Lancer predicteur flask comme conteneur Docker

- Installer l'app Docker desktop si necessaire
- Ouvrir l'app Docker desktop
- aller dans le repertoire milestone3 du projet
- dans le fichier .env du dossier milestone3, y mentionner votre clé API wandb
- executer les commandes:
```bash
  docker-compose build
```
ensuite

```bash
  docker-compose up
```

- Tester les calls API comme expliqué précédement pour vérifier que
le serveur a bien démarré


# Section Streamlit 

Pour lancer L'app StreamLit, aller dans le repertoire milestone3 et executer la commande:

```bash
  cd milestone3/
  streamlit run ./streamlit_app.py
```

### Setup navigateur

Une fois sur votre navigateur, choisissez votre ***Game Id*** et votre ***Model*** puis cliquez sur le bouton `Get model`.

### Lancement

Pour lancer la recherche, cliquez sur `Ping game`.

### Modèles chargés localement

Vous retrouverez les modèles chargés localement dans le dossier `./data/predictions` 

