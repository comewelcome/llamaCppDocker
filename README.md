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

Créez un fichier `.env` à partir de `.env.example` :

| Variable | Description | Défaut |
|---|---|---|
| `HF_REPO` | Repository HuggingFace au format `owner/repo` | — |
| `MMPROJ` | Chemin du projecteur multimodal (pour les images) | — |
| `CTX_SIZE` | Taille du contexte en tokens | `262144` |
| `GPU_LAYERS` | Nombre de couches offloadées sur GPU (`999` = tout) | `99` |
| `THREADS` | Nombre de threads CPU | `6` |
| `BATCH_SIZE` | Taille du batch d'évaluation | `512` |
| `PARALLEL` | Nombre de requêtes parallèles | `1` |
| `TENSOR_SPLIT` | Répartition multi-GPU (ex: `0.75,0.25` pour 2 GPU) | `0.75,0.25` |
| `PORT` | Port exposé sur l'hôte | `1234` |
| `LLAMA_CACHE` | Répertoire cache des modèles | `/app/modele` |

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
Le serveur utilise `--spec-type draft-mtp --spec-draft-n-max 2` pour prédire plusieurs tokens en avance grâce aux heads MTP du modèle, améliorant la vitesse d'inférence.

### Cache KV quantifié
Les caches key/value utilisent la quantification `q4_0` (`--cache-type-k q4_0 --cache-type-v q4_0`) pour réduire la consommation mémoire liée au contexte.

### Flash Attention
Activé via `-fa on` pour un calcul d'attention plus rapide et moins gourmand en mémoire.

### Multi-GPU
Le paramètre `--tensor-split` permet de répartir le modèle sur plusieurs GPU. Ajustez `TENSOR_SPLIT` selon votre configuration (ex: `0.65,0.35` pour un GPU principal + un secondaire).

### Support Multimodal (Images)
Pour activer le support des images, vous devez spécifier le chemin du fichier `mmproj` dans votre fichier `.env`. Ce fichier est généralement téléchargé avec le modèle.

Exemple pour Qwen3.6 :
```env
MMPROJ=models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/53b097416d6346f849b530e4bc1b5590dfe9d758/mmproj-BF16.gguf
```

Une fois activé, vous pouvez envoyer des images via l'API (format compatible OpenAI Vision).

## 📁 Structure du projet

```
.
├── .env            # Configuration personnalisée (ne pas committer)
├── .env.example    # Template de configuration
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