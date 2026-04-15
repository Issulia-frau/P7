# P7

# Système RAG avec FastAPI


## Architecture du système

###  Recherche

* Les données sont transformées en documents texte
* Ces documents sont vectorisés via un modèle d’embeddings
* Stockage dans une base vectorielle (FAISS)
* Recherche des documents les plus proches via similarité

###  Construction du contexte

* Les documents récupérés sont filtrés selon la requête
* Un contexte textuel est construit à partir des documents pertinents

###  Génération

* Le contexte est injecté dans un prompt
* Un modèle de langage génère une réponse en français

---

## Choix technologiques

### Backend

* FastAPI : framework API rapide et moderne
* Pydantic : validation des requêtes

### Traitement des données

* Pandas : chargement et manipulation du dataset CSV

### Embeddings

* SentenceTransformers (`all-mpnet-base-v2`)

  * Bon compromis performance / qualité
  * Adapté à la similarité sémantique

### Vector Store

* FAISS

  * Très performant pour la recherche vectorielle
  * Facile à intégrer en local

### LLM

* Mistral-7B-Instruct

  * Modèle open-source
  * Utilisé via HuggingFace Pipeline

### Orchestration

* LangChain

  * Gestion du prompt
  * Chaînage de la récupération vers la génération

---

## Pipeline détaillé

1. L’utilisateur envoie `/ask`
2. Le système :
   * encode la question
   * récupère les `k` documents les plus proches
3. Construction du contexte
4. Injection dans un prompt
5. Génération de la réponse
6. Retour

---

## Résultats observés

### Points positifs

* Réponses cohérentes et contextualisées
* Bonne pertinence des résultats avec FAISS
* Pipeline rapide en local
* API simple à utiliser

### Limites observées

* Pas de reranking avancé
* Contexte parfois trop long

---

## Limites techniques

* Pas de mise en cache
* Pas de gestion fine des tokens

---

## Pistes d’amélioration

### Retrieval

* Ajouter un **reranker (cross-encoder)**
* Implémenter une recherche hybride (BM25 + embeddings)

### Performance

* Ajouter un cache
* Batch embeddings

---

## Utilisation

### Lancer l’API

```bash
uvicorn main:app --reload
```

### Endpoint principal

```bash
POST /ask
{
  "question": "Quels événements pop récents ?"
}
```

### Rebuild vector store

```bash
POST /rebuild
```

---

## Structure du projet
```
.
├── main.py
├── data.csv
├── README.md
├── test_api.py
├── Dockefile
├── Requirements.txt
├── P7-datasource.ipynb (data extration to csv)
├── auths.py (Your openagenda token)
```
