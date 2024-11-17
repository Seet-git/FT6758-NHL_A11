# Projet NHL IFT 6758 - Milestone 1

## Objectifs du Projet
- **Acquisition de données** : Récupérer les données "play-by-play" de plusieurs saisons de la NHL à partir de l'API de statistiques.
- **Nettoyage des données** : Transformer ces données brutes en un dataframe pour des analyses ultérieures.
- **Visualisations** : Créer de simples graphiques interactifs.
- **Blog** : Un article de blog accompagné de visualisations interactives pour présenter les résultats du projet.

## Code Principal
Le script principal à exécuter pour ce projet est **`mainNHL.ipyb`**. Ce fichier contient tout le code nécessaire pour :
- Télécharger les données via l'API.
- Traiter et formater les données en dataframes.
- Générer des visualisations interactives.

## Ouverture du Projet
Pour exécuter le projet, il suffit de lancer le fichier `mainNHL.ipyb` dans un environnement Jupyter.

1. Ouvrez le terminal et accédez au répertoire du projet.
2. Lancez Jupyter Notebook avec la commande suivante :
   ```bash
   jupyter notebook main_Milestone-1.ipyb


# Projet NHL IFT 6758 - Milestone 2

## Objectifs du Milestone 2
- **Ingénierie des Caractéristiques** : Créer et sélectionner des caractéristiques à partir des données de tir pour améliorer la modélisation de la probabilité qu'un tir devienne un but.
- **Modélisation Statistique et Machine Learning** : Tester plusieurs modèles de classification et combiner diverses caractéristiques pour produire un classifieur pertinent.
- **Suivi des Expériences** : Utilisation de Wandb pour suivre les expériences et assurer la reproductibilité des résultats.
- **Blog** : Présentation des résultats et analyses dans un article de blog.

## Instructions d'Exécution
Pour exécuter le projet :
### Configuration environment :
   - Créez un environnement virtuel et installez les dépendances requises.
     ```bash
     pip install -r requirements.txt
     ```
   - Configurez votre compte Wandb pour suivre les expériences :
     ```bash
     wandb login
     ``` 

### Configuration de Optuna avec MySQL et Ngrok

#### Configuration de la Base de Données MySQL

Pour optimiser vos modèles avec `optuna` et stocker les résultats dans une base de données, configurez MySQL comme suit :

1. **Installer MySQL:** Commencez par installer MySQL

   - Télécharger sur Windows / Mac / Linux: https://dev.mysql.com/downloads/mysql/

   - Terminal Linux:
      ```bash
      sudo apt-get update
      sudo apt-get install mysql-server
      ```
     
   Une fois téléchargé, vous devriez pouvoir utiliser cette commande : 
   ```bash
   sudo mysql -u root -p
   ```

2. **Créer une base de données MySQL :**
   Connectez-vous à MySQL et exécutez les commandes suivantes :
   
   ```sql
   CREATE DATABASE optuna_db;
   CREATE USER 'optuna_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON optuna_db.* TO 'optuna_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Autorisation et Accès :**
   - Assurez-vous que `optuna_user` a les autorisations nécessaires pour créer, lire, écrire et supprimer les entrées dans `optuna_db`.
   - Configurez les accès réseau de votre base de données si nécessaire.


Si tout est correct, vous devriez voir la base de données, avec la commande suivante :
   ```sql
    SHOW GRANTS FOR 'optuna_user'@'localhost';
    SHOW databases;
   ```

#### Utilisation de Ngrok pour une Connexion Externe (OPTIONNEL)

Pour rendre la base de données accessible à distance (utile si vous ne pouvez pas accéder à `localhost`), vous pouvez utiliser `ngrok` :

**Installer Ngrok et Ouvrir une Connexion avec Ngrok :**
   ```bash
   ngrok tcp 3306
   ```
   Remplacez `3306` par le port utilisé par votre serveur MySQL si différent.


### Lancer le script

   ```bash
   jupyter notebook mainNHL_Milestone2.ipynb

