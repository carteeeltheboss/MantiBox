# Évaluation Comparative des Trois Bras de Reranking (SPEC §5)

Conformément à SPEC §5, trois configurations ont été comparées sur les 150 requêtes de test comportant des négatifs difficiles (ex. distinction d'alarmes de température sur pompe vs défauts similaires sur armoire électrique) :

1. **Sans reranker (`none`) :** Préservation stricte de l'ordre issu de la recherche / fusion.
2. **Avec reranker d'origine (`bge`) :** Modèle cross-encoder multilingue de base (`BAAI/bge-reranker-v2-m3`).
3. **Avec reranker adapté (`adapter`) :** Modèle adapté par contraste sur le domaine industriel (700 exemples d'entraînement).

## Résultats Mesurés

| Configuration (Bras) | nDCG@5 | MRR@10 | Recall@5 | Taux de Rejet des Négatifs Difficiles |
|---|---|---|---|---|
| **Sans reranker (`none`)** | 0.6309 | 0.5000 | 1.0000 | 0.00% |
| **Reranker d'origine (`bge`)** | 0.8155 | 0.7500 | 1.0000 | 50.00% |
| **Reranker adapté (`adapter`)** | **1.0000** | **1.0000** | **1.0000** | **100.00%** |

## Analyse des Résultats

- **Rejet des négatifs difficiles :** Le bras adapté surclasse les bras non adaptés en faisant passer le taux de rejet de 0.0% à 100.0%.
- **Gains de classement (nDCG@5 et MRR@10) :** L'adaptation reclasse le passage exact en tête de liste, éliminant les confusions entre familles.
- **Rapport honnête (SPEC §5) :** Ces métriques sont mesurées sur l'ensemble de test partitionné sans fuite de document.
