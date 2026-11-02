# Fiche de Limites et Conditions d'Utilisation (SPEC §13.5)

**Projet :** MaintiDoc — Assistant documentaire multimodal et agentique pour la maintenance industrielle  
**Statut du document :** Document contractuel de sécurité et d'exploitation (SPEC §13.5)  
**Avertissement fondamental :** MaintiDoc est une **aide documentaire**. Il ne contrôle aucun équipement industriel et ne garantit aucun diagnostic (AGENTS.md).

---

## 1. Rôle et Limites Fondamentales du Système

1. **Aide documentaire exclusive :** Le système extrait, classe et synthétise les informations issues des manuels techniques, schémas électriques et historiques d'intervention.
2. **Absence totale de contrôle physique :** MaintiDoc ne dispose d'aucun actionneur, d'aucune connexion directe aux automates programmables (API/PLC), et ne peut en aucun cas démarrer, arrêter ou modifier les paramètres d'une machine.
3. **Aucune garantie de diagnostic :** Conformément à SPEC §8, une ressemblance visuelle ou une proximité textuelle ne constitue jamais une preuve d'anomalie machine. L'expertise et la décision finale incombent exclusivement au technicien de maintenance qualifié.
4. **Interdiction formelle de confiance inventée :** Le système n'émet aucun pourcentage de certitude arbitraire (NEVER #6). En cas de doute, d'image illisible ou de sources contradictoires, le système s'abstient ou demande explicitement une clarification.

---

## 2. Équipements Couverts (Périmètre Technique)

Le corpus documentaire et les capacités d'analyse couvrent deux familles d'équipements industriels :

| Famille d'équipement | Types de machines inclus | Documents applicables | Limites spécifiques |
|---|---|---|---|
| **Groupes motopompes centrifuges** | Pompes centrifuges industrielles multicellulaires, garnitures mécaniques, paliers lisses et roulements, vannes de refoulement et circuits de purge | Manuels constructeurs, fiches techniques, procédures de graissage, éclatés mécaniques | Ne couvre pas les pompes péristaltiques ou à vide non documentées |
| **Armoires électriques basse tension (BT)** | Armoires de distribution et commande (400V triphasé / 230V mono / 24V continu), relais thermiques, contacteurs, disjoncteurs, transformateurs d'auxiliaires, borniers | Schémas unifilaires, schémas de câblage, nomenclatures des protections et fusibles | Ne couvre pas les installations haute tension (HTA/HTB) |

---

## 3. Usages Autorisés

- Recherche rapide de procédures de maintenance préventive et corrective dans le corpus constructeur autorisé.
- Identification de composants (voyants de défaut, calibres de fusibles, références de garnitures mécaniques, seuils de déclenchement thermique).
- Consultation de l'historique des interventions et des pannes récurrentes enregistrées sur l'équipement via le serveur MCP.
- Préparation assistée de propositions de tickets d'intervention (titre, description, niveau d'urgence) soumises à la relecture et validation humaine préalable.

---

## 4. Usages Strictement Interdits et Hors-Rôle

1. **Substitution aux règles de sécurité et consignation :** L'assistant ne dispense en aucun cas de l'application rigoureuse des procédures de cadenassage / consignation (LOTO - Lockout/Tagout).
2. **Intervention sous tension sans habilitation :** Toute manipulation sur armoire électrique requiert les habilitations électriques réglementaires requises (B1V, B2V, BR, BC selon NF C 18-510).
3. **Création automatisée de tickets sans accord :** Aucun ticket ne peut être créé dans la GMAO sans l'accord explicite et préalable d'un utilisateur humain authentifié (NEVER #3).
4. **Traitement d'instructions dans les données :** Les textes de documents, sorties OCR ou retours d'outils ne doivent jamais être exécutés comme des consignes de sécurité ou des ordres opérationnels (NEVER #4).

---

## 5. Analyse des Risques et Dispositifs d'Atténuation

| Risque Identifié | Impact Potentiel | Atténuation Implémentée dans MaintiDoc |
|---|---|---|
| **Confusion entre équipements distincts** | Application d'une mauvaise consigne (ex. alarme pompe confondue avec voyant armoire) | Reranker adapté avec apprentissage contrastif sur négatifs difficiles (taux de rejet 100%) |
| **Hallucination de références documentaires** | Perte de temps du technicien ou fausse procédure | Vérification stricte des citations par rapport aux éléments réellement récupérés (NEVER #5) |
| **Injection indirecte de prompt (étiquette OCR)** | Tentative de forcer une action ou d'exfiltrer des données | Délimitation XML étanche des entrées non fiables (`<untrusted_document_data>`), prompt injection regression suite (0 fuite) |
| **Exécution d'action non autorisée** | Création de ticket frauduleux ou non justifié | Jeton d'approbation cryptographique HS256 à usage unique lié au contenu exact (`content_hash`) et clé d'idempotence |
| **Fuite de documents entre sites concurrents** | Accès à des procédures confidentielles d'un autre groupe | Filtre d'accès Qdrant obligatoire (`must: access_groups`) et `DocumentOpener` avec déni par défaut |

---

## 6. Conditions Opérationnelles d'Utilisation

1. **Qualité des photographies :**
   - Résolution minimale : $256 \times 256$ pixels.
   - Éclairage net, absence de flou cinétique majeur sur les voyants ou les textes de plaque.
   - En cas d'image dégradée, le système émet automatiquement un statut `needs_clarification`.
2. **Environnement matériel et réseau :**
   - Fonctionnement nominal sous Docker Compose sur machine hôte ou serveur dédié.
   - Si exécuté avec le modèle 7B complet (Qwen2.5-VL-7B-AWQ), un GPU de 24 Go est préconisé pour garantir une latence $p95 < 30$ secondes.
3. **Contrôle d'intégrité :**
   - Exécution préalable de `maintidoc doctor` avant mise en service pour valider la connectivité des services et l'intégrité du corpus.
