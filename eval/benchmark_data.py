"""Seed evaluation queries and ground-truth relevance annotations (SPEC §5, §10).

Curated across the two equipment families (pompes, armoires) and access tiers.
Every scenario defines:
- query: French user request
- relevant_element_ids: set of (document_id, element_id) pairs
- allowed_principal: Principal authorized to retrieve them
"""

from __future__ import annotations

from dataclasses import dataclass

from maintidoc.security.access import Principal

SUPERVISOR = Principal(user_id="u-sup", groups=frozenset({"maintenance", "supervisors", "site_a"}))
TECH_A = Principal(user_id="u-tech-a", groups=frozenset({"maintenance", "site_a"}))
TECH_B = Principal(user_id="u-tech-b", groups=frozenset({"maintenance", "site_b"}))


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    query: str
    principal: Principal
    # Set of target (document_id, element_id)
    relevant_targets: frozenset[tuple[str, str]]
    description: str


SEED_QUERIES: tuple[EvalQuery, ...] = (
    EvalQuery(
        query_id="q1_pompe_voyant_orange",
        query="Que signifie le voyant orange sur le groupe de pompage centrifuge ?",
        principal=TECH_A,
        relevant_targets=frozenset(
            {
                ("manuel_pompe_a", "paragraph_1_7"),
                ("manuel_pompe_a", "paragraph_1_9"),
                ("manuel_pompe_a", "figure_1_1"),
            }
        ),
        description="Alarme température voyant orange sur pompe A",
    ),
    EvalQuery(
        query_id="q2_armoire_voyant_orange",
        query="Quelle est la signification du voyant orange sur l'armoire de commande électrique ?",
        principal=TECH_B,
        relevant_targets=frozenset(
            {
                ("manuel_armoire_c", "paragraph_1_3"),
                ("manuel_armoire_c", "paragraph_1_5"),
                ("manuel_armoire_c", "figure_1_1"),
            }
        ),
        description="Défaut de phase voyant orange sur armoire C (hard negative for pompe)",
    ),
    EvalQuery(
        query_id="q3_armoire_fusibles",
        query="Quels fusibles protègent le circuit de commande de l'armoire électrique ?",
        principal=TECH_B,
        relevant_targets=frozenset(
            {
                ("manuel_armoire_c", "paragraph_1_6"),
            }
        ),
        description="Fusibles F3 et F4 sur armoire C",
    ),
    EvalQuery(
        query_id="q4_pompe_procedure_plaque",
        query=(
            "Quelle est la procédure à suivre et que vérifier sur la plaque signalétique "
            "en cas d'alarme température pompe ?"
        ),
        principal=TECH_A,
        relevant_targets=frozenset(
            {
                ("procedure_pompe_a", "paragraph_1_1"),
                ("procedure_pompe_a", "paragraph_1_2"),
            }
        ),
        description="Procédure d'intervention plaque signalétique pompe A",
    ),
    EvalQuery(
        query_id="q5_pompe_caracteristiques",
        query="Quelles sont les caractéristiques nominales et le débit du groupe de pompage ?",
        principal=TECH_A,
        relevant_targets=frozenset(
            {
                ("manuel_pompe_a", "table_1_1"),
            }
        ),
        description="Tableau caractéristiques nominales pompe A",
    ),
    EvalQuery(
        query_id="q6_pompe_restreint_consignation",
        query=(
            "Quelle est la procédure de consignation dans le manuel restreint "
            "du groupe de pompage ?"
        ),
        principal=SUPERVISOR,
        relevant_targets=frozenset(
            {
                ("manuel_pompe_a_restreint", "paragraph_1_5"),
                ("manuel_pompe_a_restreint", "figure_2_1"),
            }
        ),
        description="Manuel restreint accessible aux superviseurs uniquement",
    ),
)
