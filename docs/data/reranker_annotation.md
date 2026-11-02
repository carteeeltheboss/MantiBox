# Documentation du Jeu de Données d'Entraînement du Reranker (SPEC §5)

- **Total exemples :** 1 000 exemples annotés et vérifiés (conforme à la cible SPEC §5 de 800 à 1 500 exemples).
- **Répartition :**
  - Entraînement (`train.json`) : 700 exemples (23 documents).
  - Validation (`val.json`) : 150 exemples (5 documents).
  - Test (`test.json`) : 150 exemples (6 documents).
- **Garantie d'absence de fuite :**
  - Les ensembles `train`, `val` et `test` sont partitionnés strictement par `document_id`.
  - Intersection Train / Val : 0 document en commun.
  - Intersection Train / Test : 0 document en commun.
  - Intersection Val / Test : 0 document en commun.

## Structure d'un Exemple

Chaque exemple comprend :
- `query` : Question formulée par le technicien.
- `positive_passage` : Extrait pertinent soutenant directement la réponse.
- `hard_negative_passage` : Négatif difficile partageant les mêmes mots-clés de surface (ex. *voyant orange*, *alarme de température*, *pression maximale*) mais appartenant à un autre équipement ou à un autre sous-ensemble technique.
- `equipment_family` : Famille d'équipement ciblée (`pompes` ou `armoires`).
- `split` : `train`, `val` ou `test`.

## Contrôles Qualité
- Vérification automatique de la non-vacuité des passages.
- Vérification que le négatif difficile provient d'un document distinct du positif.
- Format validé par le schéma Pydantic `RerankerExample`.
