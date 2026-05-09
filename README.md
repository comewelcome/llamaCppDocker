# 🦙 llama.cpp Docker Server

Serveur d'inférence LLM local basé sur [llama.cpp](https://github.com/ggml-org/llama.cpp), déployé via Docker avec support CUDA.

## 📋 Prérequis

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Une carte GPU NVIDIA avec suffisamment de VRAM
- Un modèle GGUF (ex: téléchargé via LM Studio)

## 🚀 Démarrage rapide

```bash
# Copier le fichier d'exemple et modifier les paramètres
cp .env.example .env

# Démarrer le serveur
docker compose up -d
```

Le serveur sera accessible sur `http://localhost:3003`.

## ⚙️ Configuration

| Variable | Description | Exemple |
|---|---|---|
| `MODEL_PATH` | Chemin local vers le dossier contenant le modèle GGUF | `/home/user/.lmstudio/models/...` |
| `MODEL_FILE` | Nom du fichier modèle GGUF | `Qwen3.6-27B-Q4_K_M.gguf` |
| `CTX_SIZE` | Taille du contexte en tokens | `262144` |
| `GPU_LAYERS` | Nombre de couches offloadées sur le GPU (`999` = tout) | `64` |
| `THREADS` | Nombre de threads CPU | `6` |
| `BATCH_SIZE` | Taille du batch d'évaluation | `512` |
| `PARALLEL` | Nombre de requêtes parallèles | `4` |
| `TENSOR_SPLIT` | Répartition multi-GPU (optionnel) | `0.65,0.35` |
| `PORT` | Port du serveur | `3003` |

## 🌐 API

Le serveur expose une API compatible OpenAI :

```bash
curl http://localhost:3003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B",
    "messages": [
      {"role": "user", "content": "Bonjour !"
    ]
  }'
```

## 🛑 Arrêt

```bash
docker compose down
```

## 📁 Structure du projet

```
.
├── .env            # Configuration (ne pas committer)
├── .env.example    # Template de configuration
├── .gitignore
├── docker-compose.yml
├── README.md
└── cache/