# 🦙 llama.cpp Docker Server

Serveur d'inférence LLM local basé sur [llama.cpp](https://github.com/ggml-org/llama.cpp), déployé via Docker avec support CUDA.

Ce projet utilise un **fork custom** (branche `mtp-clean` de [am17an/llama.cpp](https://github.com/am17an/llama.cpp)) pour bénéficier du **décodage spéculatif MTP** (Multi-Token Prediction).

## 📋 Prérequis

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Une carte GPU NVIDIA avec suffisamment de VRAM (optionnellement multi-GPU)

## 🚀 Démarrage rapide

```bash
# Copier le fichier d'exemple et remplir les paramètres
cp .env.example .env

# Modifier .env (au minimum HF_REPO)

# Démarrer le serveur
docker compose up -d --build
```

Le serveur sera accessible sur `http://localhost:${PORT}` (port `1234` par défaut).

## ⚙️ Configuration

Tous les paramètres de configuration sont centralisés dans le fichier `.env`. Créez-le à partir de `.env.example` :

### Modèle et HuggingFace

| Variable | Description | Défaut |
|---|---|---|
| `HF_REPO` | Repository HuggingFace au format `owner/repo:filename` | — |
| `MMPROJ` | Chemin du projecteur multimodal pour les modèles vision (optionnel) | — |

### Contexte et Performance

| Variable | Description | Défaut |
|---|---|---|
| `CTX_SIZE` | Taille du contexte en tokens | `262144` |
| `GPU_LAYERS` | Nombre de couches offloadées sur GPU (`99` = presque tout, `0` = CPU uniquement) | `99` |
| `THREADS` | Nombre de threads CPU pour l'inférence | `6` |
| `BATCH_SIZE` | Taille du batch d'évaluation | `512` |
| `PARALLEL` | Nombre de requêtes parallèles gérées simultanément | `1` |

### GPU et Mémoire

| Variable | Description | Défaut |
|---|---|---|
| `TENSOR_SPLIT` | Répartition multi-GPU (ex: `0.75,0.25` pour 2 GPU, `1` pour GPU unique) | `0.75,0.25` |

### Cache KV

| Variable | Description | Défaut |
|---|---|---|
| `CACHE_TYPE_K` | Type de quantification du cache key (`q4_0`, `q8_0`, `f16`, `f32`) | `q4_0` |
| `CACHE_TYPE_V` | Type de quantification du cache value (`q4_0`, `q8_0`, `f16`, `f32`) | `q4_0` |

### Décodage Spéculatif MTP

| Variable | Description | Défaut |
|---|---|---|
| `SPEC_TYPE` | Type de décodage spéculatif (`draft-mtp` pour Multi-Token Prediction) | `draft-mtp` |
| `SPEC_DRAFT_N_MAX` | Nombre maximum de tokens prédits par le modèle de draft | `2` |
| `SPEC_DRAFT_P_MIN` | Seuil minimum de probabilité pour accepter les tokens draft (0.0 à 1.0) | `0.75` |

### Flash Attention

| Variable | Description | Défaut |
|---|---|---|
| `FLASH_ATTENTION` | Activer Flash Attention pour un calcul plus rapide (`on`/`off`) | `on` |

### Paramètres de génération

| Variable | Description | Défaut |
|---|---|---|
| `TEMP` | Température d'échantillonnage (0 = déterministe, plus élevé = plus créatif) | `0.2` |
| `TOP_K` | Nombre maximum de tokens considérés pour chaque prédiction | `10` |
| `TOP_P` | Seuil de probabilité cumulée pour l'échantillonnage (0.0 à 1.0) | `0.9` |
| `REPEAT_PENALTY` | Pénalité de répétition (> 1 réduit la répétition, 1 = désactivé) | `1.05` |

### Réseau et Serveur

| Variable | Description | Défaut |
|---|---|---|
| `PORT` | Port exposé sur l'hôte | `1234` |

### Stockage

| Variable | Description | Défaut |
|---|---|---|
| `LLAMA_CACHE` | Répertoire de cache des modèles | `/app/modele` |

## 🌐 API

Le serveur expose une API compatible OpenAI :

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "user", "content": "Bonjour !"}
    ]
  }'
```

## 🚨 Arrêt

```bash
docker compose down
```

## 🧠 Fonctionnalités avancées

### Décodage spéculatif MTP
Le serveur utilise `--spec-type draft-mtp` pour prédire plusieurs tokens en avance grâce aux heads MTP du modèle, améliorant la vitesse d'inférence. Ajustez les paramètres via :
- `SPEC_DRAFT_N_MAX` : nombre de tokens draft maximum
- `SPEC_DRAFT_P_MIN` : seuil de probabilité minimum pour accepter les tokens

### Cache KV quantifié
Les caches key/value utilisent la quantification configurable via `CACHE_TYPE_K` et `CACHE_TYPE_V`. Les options disponibles sont :
- `q4_0` : réduit la mémoire au maximum
- `q8_0` : compromis mémoire/précision
- `f16` : précision élevée
- `f32` : précision maximale

### Flash Attention
Activé via `-fa on` par défaut (configurable avec `FLASH_ATTENTION`) pour un calcul d'attention plus rapide et moins gourmand en mémoire.

### Multi-GPU
Le paramètre `--tensor-split` permet de répartir le modèle sur plusieurs GPU. Ajustez `TENSOR_SPLIT` selon votre configuration (ex: `0.65,0.35` pour un GPU principal + un secondaire). Pour un GPU unique, utilisez `TENSOR_SPLIT=1`.

### Support Multimodal (Images)
Pour activer le support des images, spécifiez le chemin du fichier `mmproj` dans votre fichier `.env`. Ce fichier est généralement téléchargé avec le modèle.

Exemple pour Qwen3.6 :
```env
MMPROJ=models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/53b097416d6346f849b530e4bc1b5590dfe9d758/mmproj-BF16.gguf
```

Une fois activé, vous pouvez envoyer des images via l'API (format compatible OpenAI Vision).

## 📁 Structure du projet

```
.
├── .env            # Configuration personnalisée (ne pas committer)
├── .env.example    # Template de configuration avec tous les paramètres
├── .gitignore
├── docker-compose.yml   # Orchestration Docker
├── Dockerfile          # Image CUDA + llama.cpp
├── README.md
└── modele/             # Volume local pour les modèles (créé au premier lancer)
```

## 🗂️ Volume des modèles

Le dossier `./modele/` est monté dans le conteneur sur `/app/modele`. Les modèles téléchargés via `-hf` y sont stockés automatiquement. Vous pouvez aussi y placer manuellement des fichiers GGUF.

## ⚠️ Notes

- La taille de contexte par défaut est de **262 144 tokens** (256k), ce qui nécessite beaucoup de VRAM. Réduisez `CTX_SIZE` si nécessaire.
- Le fork `mtp-clean` est utilisé pour le support MTP. Assurez-vous que le modèle choisi supporte cette fonctionnalité.
- Pour un seul GPU, laissez `TENSOR_SPLIT=1` ou supprimez le paramètre.