# Agent RAG Local


Ce projet fournit un exemple minimal d'agent conversationnel qui fonctionne en local. Il applique la méthode **Retrieval-Augmented Generation** (RAG) pour répondre aux questions des étudiants du BUT SD à partir des supports de cours.

## Importation des données

1. Décompressez l'archive `Supports programmation.zip` située à la racine du dépôt.
2. Copiez les fichiers PDF et notebooks dans le dossier `supports/` du répertoire `agent-rag-local` (à créer si besoin).

## Installation des dépendances


```bash
python install_dependencies.py
```


## Construction de l'index

Avant de pouvoir interroger l'agent, il faut extraire le texte des documents et construire l'index vectoriel FAISS :

```bash
python main.py --build-index
```


Cette commande :
- extrait le texte des supports dans `preprocess/output/` ;
- découpe les textes en « chunks » et génère leurs embeddings ;
- enregistre l'index FAISS et les métadonnées des chunks dans `vector_store/`.

## Utilisation de l'agent

Une fois l'index créé, lancez simplement :


```bash
python main.py
```

Saisissez votre question après l'invite `Question >` et l'agent répondra en utilisant les passages de cours pertinents.

## Organisation du projet

- `preprocess/extract_text.py` : extraction du texte des supports
- `vector_store/build_faiss_index.py` : création de l'index FAISS
- `retrieval/search_chunks.py` : recherche des chunks pertinents
- `generation/generate_answer.py` : génération de la réponse via Ollama
- `main.py` : interface en ligne de commande
- `requirements.txt` : liste des dépendances

