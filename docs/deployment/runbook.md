# MaintiDoc Demo Deployment & Rollback Runbook (SPEC §11)

Ce document formalise la procédure d'exploitation, de déploiement en commande unique et de retour arrière (rollback) pour la pile MaintiDoc en environnement de démonstration (SPEC §11).

---

## 1. Architecture Déployée et Versions Figées

La pile est orchestrée par Docker Compose (`docker-compose.yml`) avec des images et versions rigoureusement figées :

| Service | Image Figée | Port Host | Rôle |
|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.20.0` | 6333 | Base vectorielle (collections `qdrant_text` 1024-d et `qdrant_figures` 512-d) |
| `postgres` | `postgres:16-alpine` | 5432 | Métadonnées documentaires, catalogue et traçabilité |
| `seaweedfs` | `chrislusf/seaweedfs:3.79` | 8333, 9333 | Stockage objet S3 compatible (remplacement MinIO, ADR 0006) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.19.0` | 5000 | Serveur de suivi des exécutions et des métriques |
| `api` | Construit depuis `Dockerfile` (multi-stage) | 8000 | Backend FastAPI (`uvicorn maintidoc.api.app:create_app_from_env`) |
| `ui` | Construit depuis `Dockerfile` (multi-stage) | 8501 | Interface technicien Streamlit (`streamlit run src/maintidoc/ui/app.py`) |

---

## 2. Déploiement en Une Commande

### Étape 1 : Préparation de l'environnement
Copier le fichier d'exemple et vérifier les secrets :
```bash
cp .env.example .env
# Configurer les clés API MCP et tokens d'approbation si nécessaire
```

### Étape 2 : Lancement de la pile
```bash
docker compose up -d --build
```

### Étape 3 : Contrôle de disponibilité (Healthcheck)
```bash
# Vérification globale de l'état des conteneurs
docker compose ps

# Vérification du backend FastAPI
curl -s http://localhost:8000/health | grep '"status":"ok"'

# Vérification du serveur Qdrant
curl -s http://localhost:6333/healthz

# Vérification de l'interface Streamlit
curl -I http://localhost:8501
```

---

## 3. Procédure de Sauvegarde Pré-Déploiement

Avant toute mise à jour ou nouveau déploiement sur l'environnement de démo :
```bash
# 1. Sauvegarde de la base relationnelle PostgreSQL
docker compose exec postgres pg_dump -U postgres maintidoc > backup_postgres_$(date +%Y%m%d_%H%M%S).sql

# 2. Création d'un snapshot des collections Qdrant
curl -X POST http://localhost:6333/collections/qdrant_text/snapshots
curl -X POST http://localhost:6333/collections/qdrant_figures/snapshots

# 3. Sauvegarde des volumes SeaweedFS
docker run --rm -v maintidoc_seaweedfs_data:/data -v $(pwd):/backup alpine tar czf /backup/seaweedfs_$(date +%Y%m%d_%H%M%S).tar.gz /data
```

---

## 4. Procédure de Retour Arrière (Rollback)

En cas d'anomalie détectée en production (régression de latence, échec d'accès, instabilité VLM) :

### Étape 1 : Arrêt immédiat de la version régressée
```bash
docker compose down
```

### Étape 2 : Retour au tag ou commit git antérieur stable
```bash
git checkout <previous_stable_commit_or_tag>
```

### Étape 3 : Restauration des données si une migration est intervenue
```bash
# Redémarrage des seuls services de stockage
docker compose up -d postgres qdrant seaweedfs

# Restauration PostgreSQL
cat backup_postgres_<timestamp>.sql | docker compose exec -T postgres psql -U postgres maintidoc

# Restauration Qdrant depuis snapshot
curl -X PUT http://localhost:6333/collections/qdrant_text/snapshots/recover -H 'Content-Type: application/json' -d '{"location": "file:///qdrant/snapshots/..."}'
```

### Étape 4 : Redémarrage de la pile stable
```bash
docker compose up -d --build
```

### Étape 5 : Validation post-rollback
```bash
# Exécution du test de régression clinique et de sécurité
python -m pytest tests/test_security_regression.py tests/test_api.py
```
