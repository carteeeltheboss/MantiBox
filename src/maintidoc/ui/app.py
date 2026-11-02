"""Streamlit frontend for MaintiDoc (SPEC §3, §6, §14).

Provides technician interface for:
- Asking multimodal questions with photo uploads.
- Inspecting sourced answers with strict citations and uncertainties.
- Reviewing candidate ticket proposals and granting human approval.
- Consulting equipment maintenance history and open tickets via MCP.
"""

from __future__ import annotations

import io
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

EQUIPMENT_OPTIONS = [
    "",
    "pompe_centrifuge_01",
    "pompe_centrifuge_02",
    "armoire_electrique_01",
    "armoire_electrique_02",
]


def render_ui() -> None:
    st.set_page_config(
        page_title="MaintiDoc — Assistant Documentaire",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 MaintiDoc — Assistant Maintenance")
    st.caption("Recherche documentaire multimodale, analyse d'images, MCP et réponses sourcées.")

    with st.sidebar:
        st.header("Profil & Authentification")
        api_url = st.text_input("URL API Backend", value=DEFAULT_API_URL)
        api_key = st.text_input("Clé API", value="dev-mcp-key-2026", type="password")
        user_id = st.text_input("Identifiant Technicien", value="technicien_site_a")
        user_groups_input = st.text_input(
            "Groupes d'accès (séparés par virgule)",
            value="maintenance, site_a",
        )
        st.divider()
        st.markdown("**Équipements connus :**")
        st.markdown("- `pompe_centrifuge_01` (Site A)")
        st.markdown("- `pompe_centrifuge_02` (Site A)")
        st.markdown("- `armoire_electrique_01` (Site B)")
        st.markdown("- `armoire_electrique_02` (Site B)")

    headers = {
        "X-User-Id": user_id.strip(),
        "X-User-Groups": user_groups_input.strip(),
        "X-API-Key": api_key.strip(),
    }

    tab_question, tab_history = st.tabs(["💬 Question & Photo", "📋 Historique & Tickets MCP"])

    with tab_question:
        col_input, col_view = st.columns([1, 1])

        with col_input:
            st.subheader("Demande d'intervention")
            default_q = (
                "Voici une photo du panneau de commande. Que signifie ce voyant orange ? "
                "Consulte les dernières interventions et prépare un ticket si nécessaire."
            )
            question = st.text_area("Question technique :", value=default_q, height=120)

            col_eq, col_opt = st.columns(2)
            with col_eq:
                equipment_id = st.selectbox(
                    "Équipement concerné (optionnel) :",
                    options=EQUIPMENT_OPTIONS,
                    index=1,
                )
            with col_opt:
                request_ticket = st.checkbox("Proposer un ticket si justifié", value=True)

            uploaded_file = st.file_uploader(
                "Photographie ou schéma (optionnel, max 10 Mo) :",
                type=["png", "jpg", "jpeg"],
            )

            submit_btn = st.button(
                "🚀 Interroger l'assistant",
                type="primary",
                use_container_width=True,
            )

        with col_view:
            if uploaded_file is not None:
                st.subheader("Aperçu visuel")
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if submit_btn:
            if not question.strip():
                st.warning("Veuillez saisir une question.")
                return

            with st.spinner("Analyse multimodale, consultation RRF et génération en cours..."):
                try:
                    data: dict[str, Any] = {
                        "question": question,
                        "request_ticket": "true" if request_ticket else "false",
                    }
                    if equipment_id:
                        data["equipment_id"] = equipment_id

                    files = None
                    if uploaded_file is not None:
                        bytes_data = uploaded_file.getvalue()
                        if len(bytes_data) > MAX_IMAGE_SIZE:
                            st.error("L'image dépasse la limite de 10 Mo autorisée.")
                            return
                        files = {
                            "image": (
                                uploaded_file.name,
                                io.BytesIO(bytes_data),
                                uploaded_file.type,
                            )
                        }

                    with httpx.Client(timeout=65.0) as client:
                        resp = client.post(
                            f"{api_url}/ask",
                            headers=headers,
                            data=data,
                            files=files,
                        )

                    if resp.status_code == 200:
                        payload = resp.json()
                        st.session_state["last_payload"] = payload
                    else:
                        st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Impossible de joindre le backend API : {exc}")

        # Render response if stored in session state
        payload = st.session_state.get("last_payload")
        if payload:
            st.divider()
            st.subheader("Réponse de l'assistant")

            status_str = payload.get("status", "")
            if status_str == "answered":
                st.success("✅ Statut : Réponse sourcée et vérifiée")
            elif status_str == "needs_clarification":
                st.warning("⚠️ Statut : Clarification nécessaire")
            elif status_str == "needs_approval":
                st.info("✋ Statut : Validation humaine requise avant création de ticket")
            elif status_str == "abstained":
                st.error("🛑 Statut : Abstention (sources insuffisantes ou contradictoires)")

            st.markdown(f"**Synthèse :**\n\n{payload.get('answer', '')}")

            clarification = payload.get("clarification_question")
            if clarification:
                st.markdown(f"**Question de clarification :** `{clarification}`")

            # Citations expander
            citations = payload.get("citations", [])
            if citations:
                lbl = f"📚 Références documentaires vérifiées ({len(citations)})"
                with st.expander(lbl, expanded=True):
                    for idx, c in enumerate(citations, 1):
                        st.markdown(
                            f"**{idx}. Document :** `{c.get('document_id')}` "
                            f"(Page {c.get('page')}, Élément `{c.get('element_id')}`, "
                            f"Type `{c.get('kind')}`)"
                        )
                        if c.get("quote"):
                            st.caption(f"> {c.get('quote')}")

            # Uncertainties expander
            uncertainties = payload.get("uncertainties", [])
            if uncertainties:
                lbl_u = f"⚠️ Incertitudes et éléments manquants ({len(uncertainties)})"
                with st.expander(lbl_u, expanded=False):
                    for u in uncertainties:
                        st.markdown(f"- {u}")

            # Action proposal (SPEC §6 approval interrupt)
            proposal = payload.get("proposed_action")
            if proposal:
                st.divider()
                st.warning("### ✍️ Proposition de ticket d'intervention")
                st.markdown(f"**Équipement :** `{proposal.get('equipment_id')}`")
                st.markdown(f"**Titre :** `{proposal.get('title')}`")
                st.markdown(f"**Description :** {proposal.get('description')}")
                st.caption(f"Empreinte (SHA-256) : `{proposal.get('content_hash')}`")

                if st.button("✅ Approuver et créer le ticket via MCP", type="primary"):
                    with st.spinner("Signature du jeton d'approbation et soumission sécurisée..."):
                        try:
                            with httpx.Client(timeout=30.0) as client:
                                # 1. Issue approval token
                                tok_resp = client.post(
                                    f"{api_url}/approval-token",
                                    headers=headers,
                                    json={
                                        "equipment_id": proposal.get("equipment_id"),
                                        "title": proposal.get("title"),
                                        "description": proposal.get("description"),
                                    },
                                )
                                tok_resp.raise_for_status()
                                tok_data = tok_resp.json()

                                # 2. Approve and create
                                app_resp = client.post(
                                    f"{api_url}/tickets/approve",
                                    headers=headers,
                                    json={
                                        "equipment_id": proposal.get("equipment_id"),
                                        "title": proposal.get("title"),
                                        "description": proposal.get("description"),
                                        "approval_token": tok_data["approval_token"],
                                        "idempotency_key": tok_data["idempotency_key"],
                                    },
                                )
                                app_resp.raise_for_status()
                                result = app_resp.json()
                                st.success(
                                    f"🎉 Ticket créé ! ID : **{result.get('ticket_id')}** "
                                    f"(Équipement: `{result.get('equipment_id')}`)"
                                )
                        except Exception as err:  # noqa: BLE001
                            st.error(f"Erreur lors de la validation du ticket : {err}")

    with tab_history:
        st.subheader("Consultation MCP en direct")
        hist_eq = st.selectbox(
            "Sélectionner l'équipement :",
            options=EQUIPMENT_OPTIONS[1:],
            key="hist_select",
        )
        if st.button("Rafraîchir les données MCP"):
            with st.spinner("Appel des outils MCP sécurisés..."):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        h_res = client.get(
                            f"{api_url}/equipment/{hist_eq}/history", headers=headers
                        )
                        t_res = client.get(
                            f"{api_url}/equipment/{hist_eq}/tickets", headers=headers
                        )

                    if h_res.status_code == 200:
                        h_data = h_res.json()
                        st.markdown("#### Historique des interventions")
                        if h_data.get("entries"):
                            for entry in h_data["entries"]:
                                occ = entry.get("occurred_at", "")[:10]
                                st.markdown(
                                    f"- **{occ}** : {entry.get('summary')} "
                                    f"*(Groupe: `{entry.get('technician_group')}`)*"
                                )
                        else:
                            st.info("Aucune intervention enregistrée.")
                        if h_data.get("truncated_by_access"):
                            st.caption(
                                "ℹ️ Certaines lignes ont été masquées conformément "
                                "à vos habilitations de groupe."
                            )

                    if t_res.status_code == 200:
                        t_data = t_res.json()
                        st.markdown("#### Tickets en cours")
                        if t_data.get("tickets"):
                            for t in t_data["tickets"]:
                                tid = t.get("ticket_id")
                                tstate = t.get("state")
                                st.markdown(f"- **[{tid}] {t.get('title')}** ({tstate})")
                        else:
                            st.info("Aucun ticket ouvert.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erreur lors de l'appel MCP : {exc}")


if __name__ == "__main__":
    render_ui()
