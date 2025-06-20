#Chatbot
Contexte : Tu es un ingénieur Data/IA chargé de développer un agent conversationnel local, sans appel à des API externes, pour aider des étudiants du BUT SD aux modules « Bases de la programmation » (semestres 1 & 2). L’agent doit utiliser la méthode Retrieval-Augmented Generation (RAG) pour puiser ses réponses dans des supports de cours (PDF, notebooks), et s’exécuter entièrement en local.  
Énoncé du projet (SAE6.EMS.01) :  
- Prétraitement des supports (extraction de texte depuis PDF et notebooks)  
- Découpage en « chunks » avec chevauchement  
- Vectorisation des chunks via SentenceTransformers  
- Indexation et stockage local des vecteurs dans Qdrant  
- Retrieval des chunks pertinents selon la question utilisateur  
- Génération de la réponse avec le modèle Ollama (exécuté via llama.cpp / ollama)  
- Interface CLI simple  
Plan détaillé à implémenter :  
1. Prétraitement (preprocess/) : extraire le texte des PDF (PyMuPDF ou pdfminer.six) et des notebooks (nbformat), nettoyer et sauvegarder chaque document en .txt  
2. Chunking & Vectorisation (vector_store/) : charger les .txt, découper en chunks de ~500 tokens avec overlap de 50, encoder chaque chunk avec SentenceTransformer("all-MiniLM-L6-v2"), indexer les vecteurs dans Qdrant via qdrant-client  
3. Retrieval & RAG (retrieval/ + generation/) :  
   a. Encoder la question utilisateur avec le même SentenceTransformer  
   b. Rechercher les K chunks les plus proches (cosine similarity) dans Qdrant  
   c. Construire un prompt structuré :  
      Tu es un assistant pour les étudiants de BUT SD.  
      Voici des extraits de cours pertinents :  
      ---  
      <chunk 1>  
      <chunk 2>  
      …  
      ---  
      Question : <texte de la question>  
      Réponds clairement et précisément.  
   d. Envoyer ce prompt à Ollama via subprocess ou SDK Python pour générer la réponse  
4. CLI (main.py) : boucle input("Question > ") → pipeline RAG → affichage de la réponse ; option --build-index pour (re)construire la base Qdrant  
5. Structure du projet :  
/agent-rag-local/  
├── install_dependencies.py  
├── main.py  
├── preprocess/  
│   └── extract_text.py  
├── vector_store/  
│   └── build_qdrant_index.py  
├── retrieval/  
│   └── search_chunks.py  
├── generation/  
│   └── generate_answer.py  
├── supports/  (PDF et .ipynb)  
├── README.md  
└── requirements.txt  
Contraintes techniques : Python 3.10+, LangChain pour orchestrer la RAG, modèle Ollama local, Qdrant en local, SentenceTransformers all-MiniLM-L6-v2, extraction PDF locale, exécution 100% locale.  
