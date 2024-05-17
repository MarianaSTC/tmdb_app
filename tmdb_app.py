#python -m streamlit run tmdb_app.py
#créer un  dossier et mettre dedans le dossier data 2 et le fichier csv 
import streamlit as st
import numpy as np
import pandas as pd
import os
import zipfile
import io
#import ast  # data = ast.literal_eval(f.read())
#import cv2

import plotly.graph_objs as go
import plotly.io as pio
import missingno as msno
import datetime as dt
from datetime import datetime, date
import time
import math
import itertools
import random
import requests
import json
import matplotlib.pyplot as plt
from io import BytesIO

import plotly.express as px

from PIL import Image
from PIL import PngImagePlugin
from PIL import ImageDraw
from PIL import ImageFont

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import numpy as np




st.set_page_config(page_title="TMDB Dataset",
                   page_icon=":bar_chart:", layout='wide')

col1, col2 = st.columns([18, 4])  # Ajustez les proportions selon vos besoins

# Dans la première colonne, affichez l'image
with col1:
    st.title("Projet : Modèle de Deep Learning pour la Classification Multi-Étiquettes") # Ajustez le chemin et la largeur selon vos besoins

# Dans la deuxième colonne, affichez le titre
with col2:
    st.image(r'C:\Users\maria\Desktop\tmdb_dl\téléchargement.png', width=350) 
st.markdown("Ceci est une application web pour l'exploration des images du site TMDB")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)



# Fonction pour le chargement de fichier
st.sidebar.title("Sidebar")
fl = st.sidebar.file_uploader(":file_folder: Téléchargez un fichier", type=["csv", "txt", "xlsx", "xls"])
if fl is not None:
    filename = fl.name
    if filename.endswith('.csv'):
        df_clean = pd.read_csv(fl, sep=',', low_memory=False)
    elif filename.endswith(('.xlsx', '.xls')):
        df_clean = pd.read_excel(fl, engine='openpyxl')
    df_clean.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    st.success("Fichier chargé avec succès !")
else:
    st.info("Veuillez télécharger un fichier pour commencer.")



st.sidebar.title('Navigation')
pages = [
    'Accueil',
    "API TMDB",
    "Préparation du dataset",
    "Modèle de Classification", 
    "Conclusion"
]
choice = st.sidebar.radio(":bar_chart: Choisissez une page", pages)


def home (uploaded_file):
    if uploaded_file:
        st.subheader("Commencez à explorer les données en utilisant le menu à gauche.:point_left:")
    else: 
        st.subheader("Veuillez télécharger un fichier pour commencer.:inbox_tray:")

# Affichage conditionnel du contenu basé sur la page sélectionnée
if choice == 'Accueil':
    home(fl)  
    st.header('Missions :point_left:' , divider='rainbow')
    st.text("""Utilisez les données de TMDB pour entraîner un réseau de neurones profonds capable de classer automatiquement les films sous différents genres en utilisant leurs affiches. Ceci est un modèle de classification multi-étiquettes. Ne le confondez pas avec un modèle de classification multiclasses.
Utilisez vos connaissances en réseaux de neurones convolutifs et/ou en apprentissage par transfert pour construire les modèles les plus performants en termes de précision, de rappel et de vitesse d'exécution.
Construisez une interface utilisateur avec l'outil de votre choix (Streamlit, Dash, Anvil) pour permettre aux utilisateurs de télécharger leurs propres affiches à classifier.
Ceci est un travail de groupe, mais chaque membre doit soumettre le notebook auquel il a contribué dans cette quête.
Chaque membre doit charger le projet dans son portfolio GitHub personnel.
Préparez une présentation de 10 minutes de votre travail incluant le processus de construction de vos modèles, leurs métriques de performance et une démonstration en direct des modèles en action.""")
    

elif choice == 'API TMDB':
    st.header('Using the TMDB API', divider='rainbow')
    st.text("""L'API TMDB offre une interface web gratuite pour requêter et télécharger des données de films depuis sa base de données. La lecture de la documentation de l'API TMDB nous a aidé pour comprendre les différentes manières de l'utiliser avec Python. 
                Les données de l'API sont retournées au format JSON. Si vous n'êtes pas familier avec JSON, les quêtes suivantes devraient vous aider à démarrer.""")
    st.image(r'C:\Users\maria\Desktop\tmdb_dl\api tmdb.png')

    st.subheader("Code API")
    st.markdown("""
                
                
                """)

    code= """import requests

url = "https://api.themoviedb.org/3/movie/500"
params = {
  "api_key": "xxxxxxxxxxxx"
}

response = requests.get(url, params=params)
data = response.json()
print(data)"""
    st.code(code, language="python")
    st.markdown("""
                
                
                """)
    st.text("""Préparation du dataset...EDA""")      
    st.markdown("""
                
                
                """)
    col1, col2=st.columns([8,8])
    
    with col1:
        with st.expander("Le dataset:"):
            st.dataframe(df_clean.head().style.background_gradient(cmap='mako'))
            csv=df_clean.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("Télécharger csv", data=csv, file_name='le-dataset.csv', mime="text/csv",
                help='Clickez ici pout télécharger le fichier en csv avec; comme séparateur')
        
    with col2:
        with st.expander("Description:"):
            st.dataframe(df_clean.describe().style.background_gradient(cmap='mako'))
            csv=df_clean.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button("Télécharger csv", data=csv, file_name='description.csv', mime="text/csv",
                help='Clickez ici pout télécharger le fichier en csv avec ; comme séparateur')
    st.markdown("""
                
                
                """)
    total_lignes = df_clean.shape
    nombre_films = df_clean['original_title'].nunique()
    st.markdown("""
                
                
                """)
    
    with col1:
        st.metric(label="Taille du dataset", value=f"{df_clean.shape}")
            

    with col2:
        st.metric(label="Nombre unique de films", value=f"{nombre_films}")
    st.markdown("""
                
                
                """)
    with st.expander("Téléchargement d\'affiches pour classification"): 
# Chemin vers le dossier contenant les images
        folder_path = r'C:\Users\maria\Desktop\tmdb_dl\Data 2'
# Lister les fichiers dans le dossier
        files = os.listdir(folder_path)

# Affichage de la liste des fichiers
        st.write("Fichiers disponibles pour téléchargement :")
        st.write(files)
    


    # Fonction pour compresser et télécharger le dossier
        def zip_and_download_folder(folder_path):
            # Création d'un objet BytesIO pour stocker les données zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    zip_file.write(file_path, arcname=file_name)
            zip_buffer.seek(0)
            
            # Permettre le téléchargement du zip
            st.download_button(label="Télécharger le dossier compressé",
                            data=zip_buffer,
                            file_name='dossier_images.zip',
                            mime='application/zip')

    # Bouton pour déclencher la compression et le téléchargement
        st.button("Compresser et télécharger le dossier", on_click=zip_and_download_folder, args=(folder_path,))


elif choice == 'Préparation du dataset':
    st.header("Préparation du dataset", divider='rainbow')
    
    
# Créez trois colonnes
    col1, col2, col3 = st.columns([1,2,1])

# Utilisez la colonne du milieu pour afficher l'image, ce qui la centrera relativement
    with col2:
        st.image(r"C:\Users\maria\Desktop\tmdb_dl\Capture d'écran 2024-05-17 094149.png", width=700)
        st.caption("""Réalisation du one-hot-encoding""")

       
    st.markdown("""
    Le one-hot encoding est une technique de prétraitement des données utilisée principalement en apprentissage automatique pour gérer les variables catégorielles.     
                   
    Avantages:
                        
    Clarté des données : Chaque indice dans le vecteur encodé représente clairement la présence ou l'absence d'une catégorie.
                        
    Compatibilité : Facile à utiliser avec divers modèles de machine learning qui nécessitent des entrées numériques.
                        
    Efficient pour les classes multiples : Permet une représentation simple et efficace des cas où multiples classes peuvent être assignées à une observation.
                        
    Inconvénients:
                        
    Augmentation de la dimensionnalité : Pour les variables catégorielles avec beaucoup de valeurs uniques, le one-hot encoding peut grandement
                        augmenter la dimensionnalité de l'ensemble de données, ce qui peut entraîner une augmentation de la complexité du modèle et du risque de surajustement.
                        
    Sparsité des données : Les vecteurs résultants sont majoritairement composés de zéros, ce qui peut être inefficace en termes de stockage et de calcul.
                        
                        """)
    

elif choice == 'Modèle de Classification':
    st.header("Modele de classification multi-étiquettes", divider='rainbow')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(r"C:\Users\maria\Desktop\tmdb_dl\Data 2\1278183.jpg", width=350)

    with col2:
        st.image(r"C:\Users\maria\Desktop\tmdb_dl\Data 2\11.jpg", width=350)

    with col3: 
        st.image(r"C:\Users\maria\Desktop\tmdb_dl\Data 2\1280483.jpg", width=350)

    st.text("""Le modèle de classification multi-étiquettes est un type de modèle prédictif utilisé en apprentissage automatique pour assigner à chaque instance (comme un document, une image, un texte, etc.) plusieurs étiquettes ou classes simultanément. Contrairement à la classification binaire ou multiclasse où chaque instance est assignée à une seule catégorie, la classification multi-étiquettes permet à une instance d'être associée à plusieurs catégories en même temps.

Exemples d'utilisation :
Catégorisation de produits : Un produit peut être à la fois classé comme "électronique", "consommable", et "offre".
Systèmes de recommandation : Un film ou un livre pourrait être tagué dans plusieurs genres comme "action", "aventure", et "science-fiction".
Analyse de sentiments : Les textes des réseaux sociaux pourraient exprimer simultanément de la joie, de la surprise, et de la tristesse.
Reconnaissance d'images : Une photo peut être taguée avec plusieurs objets ou thèmes, comme "plage", "coucher de soleil", et "famille".
Fonctionnement :
Pour traiter une tâche de classification multi-étiquettes, les approches suivantes sont souvent utilisées :

Transformation de problème : Convertir le problème de classification multi-étiquettes en plusieurs problèmes de classification binaire ou multiclasse indépendants. Par exemple, pour chaque catégorie, un modèle distinct pourrait prédire si cette catégorie s'applique ou non à une instance donnée.

Adaptation d'algorithme : Modifier ou utiliser des algorithmes spécifiquement conçus pour gérer plusieurs étiquettes simultanément. Par exemple, certaines versions des arbres de décision, des réseaux de neurones, ou des machines à vecteurs de support (SVM) peuvent directement gérer les sorties multi-étiquettes.

Modèles de régression chaînée : Dans cette méthode, les étiquettes sont prédites séquentiellement, où chaque prédiction d'étiquette peut dépendre des étiquettes déjà prédites.

Mesure de Performance :
Évaluer les modèles de classification multi-étiquettes peut être complexe, car il faut considérer la précision des prédictions sur plusieurs étiquettes. Les métriques communes incluent :

Accuracy : Le pourcentage d'échantillons pour lesquels les ensembles d'étiquettes prédits et réels sont exactement les mêmes.
Hamming Loss : Le pourcentage de labels incorrects par rapport au nombre total de labels.
Precision, Recall, et F1-Score : Calculés pour chaque étiquette, puis moyennés ou pondérés par le support de l'étiquette.
La classification multi-étiquettes est donc une méthode puissante mais complexe qui nécessite des techniques spécifiques pour gérer et évaluer correctement la diversité et la corrélation des étiquettes.""")
    
    st.text("""Traitement des images""") 


    code1= """from PIL import Image
img = Image.open(poster_folder_path + film_id + ".jpg")
img = img.resize((224, 224))
img_array = np.array(img)
Benoit — ayer a las 17:28
df_train = df.sample(frac = 0.8, random_state=42)


import os
import shutil

Créer les dossiers train et test
os.makedirs(poster_folder_path + 'train', exist_ok=True)
os.makedirs(poster_folder_path + 'test', exist_ok=True)

Déplacer les fichiers vers les dossiers train et test
for id in df_train['id']:
    shutil.move(poster_folder_path + str(id) + '.jpg', poster_folder_path + 'train/' + str(id) + '.jpg')

for id in df_test['id']:
    shutil.move(poster_folder_path + str(id) + '.jpg', poster_folder_path + 'test/' + str(id) + '.jpg')
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image

Prétraitement des images avec PIL
def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    # Conversion en RGB si l'image est en noir et blanc
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Redimensionnement de l'image
    img = img.resize(target_size)
    # Conversion en array Numpy
    img_array = np.array(img)
    # Normalisation des valeurs des pixels
    img_array = img_array / 255.0  # Normaliser les valeurs de pixel
    return img_array
"""
    st.code(code1, language="python")