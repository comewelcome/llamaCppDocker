# llamaCppDocker

Serveur d'inférence locale pour LLM basé sur [llama.cpp](https://github.com/ggml-org/llama.cpp), déployé via Docker Compose avec support GPU CUDA et API compatible OpenAI.

## 📋 Fonctionnalités

- **API OpenAI-compatible** — Compatible avec tout client OpenAI (langchain, litellm, Ollama clients, etc.)
- **Support multi-GPU** — Répartition du modèle sur plusieurs GPUs NVIDIA via tensor splitting
- **Flash Attention** — Calcul d'attention optimisé pour réduire la mémoire et accélérer l'inférence
- **Décodage spéculatif MTP** — Multi-Token Prediction pour accélérer la génération
- **Cache KV quantifié** — Réduction de la mémoire avec quantification q4/q8 du cache
- **Contexte étendu** — Support jusqu'à 256k tokens de contexte
- **Configuration centralisée** — Toutes les variables de configuration dans un seul fichier `.env`

## 🚀 Démarrage rapide

### Prérequis

- [Docker](https://docs.docker.com/get-docker/) (v24+)
- [Docker Compose](https://docs.docker.com/compose/) (v2+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Carte(s) GPU NVIDIA avec drivers CUDA

### Installation

```bash
# Cloner le repository
git clone https://github.com/comewelcome/llamaCppDocker.git
cd llamaCppDocker

# Copier le fichier de configuration
cp .env .env

# Télécharger le modèle GGUF dans le dossier models/
# Par défaut: Qwen3.6-27B-UD-Q5_K_XL.gguf
mkdir -p models
# Voir la section Modèles ci-dessous
```

### Lancement

```bash
docker compose up -d
```

Le serveur est accessible sur `http://localhost:1234` (port configurable dans `.env`).

### Arrêt

```bash
docker compose down
```

## 📁 Structure du projet

```
.
├── docker-compose.yml    # Configuration Docker Compose
├── .env                  # Variables d'environnement (configuration)
├── .gitignore            # Fichiers exclus de Git
├── .dockerignore         # Fichiers exclus du build Docker
├── models/               # Modèles GGUF (non versionné)
│   └── Qwen3.6-27B-UD-Q5_K_XL.gguf
└── cache/                # Cache Hugging Face (non versionné)
```

## ⚙️ Configuration

Toutes les options de configuration se trouvent dans `.env`.

### Modèle

| Variable | Défaut | Description |
|----------|--------|-------------|
| `HF_REPO` | `unsloth/Qwen3.6-27B-UD-Q5_K_XL.gguf` | Repository Hugging Face (format `owner/repo:filename`) |
| `MODEL` | `/models/Qwen3.6-27B-UD-Q5_K_XL.gguf` | Chemin vers le fichier GGUF dans le container |

### Performance et contexte

| Variable | Défaut | Description |
|----------|--------|-------------|
| `CTX_SIZE` | `262144` | Taille du contexte en tokens (256k max) |
| `GPU_LAYERS` | `99` | Couches offloadées sur GPU (0 = CPU uniquement) |
| `THREADS` | `12` | Threads CPU pour l'inférence |
| `BATCH_SIZE` | `2048` | Taille du batch d'évaluation |
| `PARALLEL` | `1` | Nombre de requêtes parallèles simultanées |

### GPU

| Variable | Défaut | Description |
|----------|--------|-------------|
| `TENSOR_SPLIT` | `0.80,0.20` | Répartition multi-GPU (ex: `0.80,0.20` pour 2 GPUs) |

### Cache KV

| Variable | Défaut | Description |
|----------|--------|-------------|
| `CACHE_TYPE_K` | `q8_0` | Quantification cache key (`q4_0`, `q8_0`, `f16`, `f32`) |
| `CACHE_TYPE_V` | `q8_0` | Quantification cache value (`q4_0`, `q8_0`, `f16`, `f32`) |

### Décodage spéculatif

| Variable | Défaut | Description |
|----------|--------|-------------|
| `SPEC_TYPE` | `draft-mtp` | Type de décodage spéculatif |
| `SPEC_DRAFT_N_MAX` | `2` | Tokens prédits max par le draft model |
| `SPEC_DRAFT_P_MIN` | `0.0` | Seuil minimum de probabilité d'acceptation |

### Flash Attention

| Variable | Défaut | Description |
|----------|--------|-------------|
| `FLASH_ATTENTION` | `1` | Activer Flash Attention (`1` = activé, `0` = désactivé) |

### Génération

| Variable | Défaut | Description |
|----------|--------|-------------|
| `TEMP` | `0.6` | Température d'échantillonnage (0 = déterministe) |
| `TOP_K` | `20` | Nombre max de tokens considérés par prédiction |
| `TOP_P` | `0.95` | Seuil de probabilité cumulée |
| `REPEAT_PENALTY` | `1.00` | Pénalité de répétition (> 1 réduit la répétition) |
| `PRESENCE_PENALTY` | `1.5` | Pénalité de présence |

### Réseau et stockage

| Variable | Défaut | Description |
|----------|--------|-------------|
| `PORT` | `1234` | Port exposé sur l'hôte |
| `LLAMA_MODELS` | `./models` | Répertoire des modèles sur l'hôte |
| `LLAMA_CACHE` | `./cache` | Répertoire du cache Hugging Face sur l'hôte |

## 🔌 API OpenAI-compatible

Le serveur expose une API compatible OpenAI sur le port configuré.

### Completions (messages)

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "user", "content": "Bonjour, comment ça va ?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Completions (completions legacy)

```bash
curl http://localhost:1234/v1/completions \
  -H "Content-Type: application/json"
  -d '{
    "model": "llama",
    "prompt": "Explique le machine learning en une phrase:",
    "max_tokens": 128
  }'
```

### Streaming

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Liste 3 frameworks Python pour le ML"}],
    "stream": true
  }'
```

### Liste des modèles

```bash
curl http://localhost:1234/v1/models
```

## 🤖 Modèles supportés

Le serveur accepte n'importe quel fichier GGUF. Voici quelques modèles recommandés :

### Par défaut

| Modèle | Taille | Fichier | VRAM estimée |
|--------|--------|---------|--------------|
| Qwen3.6-27B-UD (Q5) | ~14 Go | `Qwen3.6-27B-UD-Q5_K_XL.gguf` | ~24 Go |
| Qwen3.6-27B-UD (Q4) | ~12 Go | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | ~18 Go |

### Télécharger un modèle depuis Hugging Face

```bash
# Installation de l'outil hf
pip install huggingface-hub

# Télécharger un modèle
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF \
  UD-Q5_K_XL.gguf \
  --local-dir ./models
```

Puis mettre à jour la variable `MODEL` dans `.env`.

## 🖥️ Exemples de configuration

### GPU unique

```env
GPU_LAYERS=99
TENSOR_SPLIT=
FLASH_ATTENTION=1
```

### Double GPU (ex: RTX 3090 24Go + RTX 3060 12Go)

```env
GPU_LAYERS=99
TENSOR_SPLIT=0.80,0.20
FLASH_ATTENTION=1
```

### CPU uniquement

```env
GPU_LAYERS=0
FLASH_ATTENTION=0
```

### Mode performance (contexte réduit)

```env
CTX_SIZE=32768
BATCH_SIZE=4096
THREADS=16
GPU_LAYERS=99
```

## 🐛 Dépannage

### NVIDIA: device or resource busy

```bash
# Vérifier si un processus utilise le GPU
nvidia-smi
# Tuer le processus bloquant si nécessaire
sudo fuser -v /dev/nvidia*
```

### NVIDIA Container Toolkit non installé

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Pas assez de VRAM

Réduisez `CTX_SIZE`, passez à un modèle plus petit (Q4 au lieu de Q5), ou désactivez `FLASH_ATTENTION`.

### Le serveur ne démarre pas

```bash
# Vérifier les logs
docker compose logs -f llama-server
```

### Modifier la configuration à chaud

```bash
# Éditer .env puis redémarrer
docker compose restart llama-server
```

## 📄 Licence

Ce projet utilise [llama.cpp](https://github.com/ggml-org/llama.cpp) sous licence MIT.

## 🔗 Ressources

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Projet upstream
- [llama.cpp Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md) — Documentation du serveur
- [Hugging Face GGUF models](https://huggingface.co/models?library=gguf) — Modèles GGUF disponibles
- [gguf.org](https://gguf.org) — Spécification du format GGUF
