import html
import json
import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from vis.vis_with_viser import ViserCallbackManager

log = logging.getLogger(__name__)


class QueriesAndChatController:
    """Handles LLM queries and chat interactions."""

    def __init__(self, manager: "ViserCallbackManager") -> None:
        self.manager = manager
        self.chat_messages: list[tuple[str, str]] = []
        self.chat_markdown_handle = None
        self.llm_query_input = None
        self.llm_query_btn = None
        self.chat_input = None
        self.send_btn = None

    def set_chat_agent(self, chat_agent):
        """Set the ChatAgent instance for handling chat messages."""
        self.manager.chat_agent = chat_agent
        log.info("ChatAgent configured and ready.")
        self.manager.notify_clients(
            title="Chat Agent",
            body="Chat agent is now ready.",
            color="green",
            auto_close_seconds=2.0,
        )

    def set_llm_client(self, llm_client, llm_model: str | None = None):
        """Attach the OpenAI client once credentials are available."""
        self.manager.llm_client = llm_client
        self.manager.can_run_llm_query = self.manager.has_segment_objects and self.manager.llm_client is not None
        if llm_model:
            self.manager.llm_model = llm_model

        msg = f"LLM client ready using model '{self.manager.llm_model}'."
        log.info(msg)
        self.manager.notify_clients(
            title="LLM",
            body=msg,
            color="green",
            auto_close_seconds=2.0,
        )

    def register_chat_markdown(self, markdown_handle):
        """Attach the HTML/markdown handle used for chat history display."""
        self.chat_markdown_handle = markdown_handle
        self._update_chat_markdown()

    def set_llm_query_controls(self, query_input, query_btn):
        """Register LLM query controls for runtime refreshes."""
        self.llm_query_input = query_input
        self.llm_query_btn = query_btn

    def set_chat_controls(self, chat_input, send_btn):
        """Register chat input controls for runtime refreshes."""
        self.chat_input = chat_input
        self.send_btn = send_btn

    def refresh_controls(self):
        """Refresh query/chat controls after map changes."""
        if self.llm_query_input is not None:
            self.llm_query_input.disabled = not self.manager.can_run_llm_query
        if self.llm_query_btn is not None:
            self.llm_query_btn.disabled = not self.manager.can_run_llm_query

        agent_enabled = self.manager.chat_agent is not None or self.manager.can_run_llm_query
        if self.chat_input is not None:
            self.chat_input.disabled = not agent_enabled
        if self.send_btn is not None:
            self.send_btn.disabled = not agent_enabled

    def clear_chat(self):
        """Clear the chat history (both UI display and agent context)."""
        # Clear UI display messages
        self.chat_messages = []
        self._update_chat_markdown()

        # Clear agent's chat history if agent is available
        if self.manager.chat_agent is not None and self.manager.chat_agent.chat_manager is not None:
            self.manager.chat_agent.chat_manager.clear()

        log.info("Chat history cleared")

    def add_chat_message(self, sender: str, message: str):
        """Append a chat message and refresh the markdown view."""
        clean_message = message.strip()
        if not clean_message:
            return
        self.chat_messages.append((sender, clean_message))
        self._update_chat_markdown()

    def _update_chat_markdown(self):
        """Render chat history into the markdown handle."""
        if self.chat_markdown_handle is None:
            return

        content = self._render_chat_html()
        self.chat_markdown_handle.content = content

    def _render_chat_html(self) -> str:
        """Render chat messages inside a scrollable container."""
        if not self.chat_messages:
            body = "<em>No messages yet.</em>"
        else:
            pieces = []
            for sender, msg in self.chat_messages:
                safe_sender = html.escape(sender)
                safe_msg = html.escape(msg).replace("\n", "<br>")
                pieces.append(f"<h3>{safe_sender}</h3><p>{safe_msg}</p>")
            body = "".join(pieces)

        return (
            "<div style=\"max-height: 320px; overflow-y: auto; padding-right: 8px;\">"
            f"{body}"
            "</div>"
        )

    def _build_llm_messages(self, query_text: str):
        """Construct system+user messages for OpenAI completion."""
        object_lines = []
        for idx, (label, caption) in enumerate(zip(self.manager.labels, self.manager.captions)):
            safe_label = label if label else "unknown"
            safe_caption = caption if caption else "unknown"
            object_lines.append(f"{idx}: label='{safe_label}' | caption='{safe_caption}'")

        user_message = (
            "Given an abstract user request, choose up to three objects that best satisfy it. "
            "Only use the provided ids and keep them in best-first order."
            f"\nUser request: {query_text}\nObjects:\n" + "\n".join(object_lines)
        )

        return [
            {"role": "system", "content": self.manager.llm_system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _extract_object_ids(self, content: str) -> list[int]:
        """Parse object ids from LLM JSON response; tolerate loose text."""
        ids: list[int] = []

        try:
            parsed = json.loads(content)
            candidate_ids = parsed.get("object_ids", []) if isinstance(parsed, dict) else []
            for cand in candidate_ids:
                if isinstance(cand, (int, float)):
                    ids.append(int(cand))
        except json.JSONDecodeError:
            # Fallback to any integers in the text
            ids = [int(match) for match in re.findall(r"\d+", content)]

        seen = set()
        filtered = []
        for idx in ids:
            if 0 <= idx < self.manager.num_objects and idx not in seen:
                filtered.append(idx)
                seen.add(idx)

        return filtered[:3]

    def _apply_llm_palette(self, ranked_indices):
        """Apply fixed palette to centroids and force similarity view on."""
        palette = [
            np.array([0.0, 0.8, 0.0], dtype=np.float32),
            np.array([0.95, 0.85, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.1], dtype=np.float32),
        ]
        default_color = np.zeros(3, dtype=np.float32)
        centroid_colors = []

        # Build an explicit similarity score vector so similarity mode still has data
        # while the point colors come from the fixed palette.
        similarity_scores = np.zeros(self.manager.num_objects, dtype=np.float32)
        score_steps = [1.0, 0.66, 0.33]

        for obj_idx in range(self.manager.num_objects):
            color = default_color
            if ranked_indices:
                if obj_idx == ranked_indices[0]:
                    color = palette[0]
                    similarity_scores[obj_idx] = score_steps[0]
                elif len(ranked_indices) > 1 and obj_idx == ranked_indices[1]:
                    color = palette[1]
                    similarity_scores[obj_idx] = score_steps[1]
                elif len(ranked_indices) > 2 and obj_idx == ranked_indices[2]:
                    color = palette[2]
                    similarity_scores[obj_idx] = score_steps[2]

            centroid_colors.append(color.astype(np.float32))

        # Drive the similarity layer so the GUI visibly switches on.
        self.manager.sim_query = similarity_scores
        self.manager.llm_palette_active = True
        self.manager.llm_palette_colors = np.array(centroid_colors, dtype=np.float32)
        self.manager.gui_fsm.enable_similarity_mode()

        # Keep centroid colors aligned with the palette without changing base random colors.
        if self.manager.centroid_visible:
            self.manager.remove_centroids()
            original_random = self.manager.random_colors.copy()
            self.manager.random_colors = np.array(centroid_colors, dtype=np.float32)
            self.manager.add_centroids()
            self.manager.random_colors = original_random

    def llm_query(self, query_text: str, client=None):
        """Send query to OpenAI and color top matches."""
        if not self.manager.can_run_llm_query:
            msg = "LLM query is unavailable because OPENAI_API_KEY or object data is missing."
            self.manager.notify_clients(
                title="LLM Query",
                body=msg,
                client=client,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=2.0,
            )
            log.warning(msg)
            return

        if self.manager.llm_client is None:
            msg = "LLM client is still initializing."
            self.manager.notify_clients(
                title="LLM Query",
                body=msg,
                client=client,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=2.0,
            )
            log.warning(msg)
            return

        messages = self._build_llm_messages(query_text)

        try:
            response = self.manager.llm_client.chat.completions.create(
                model=self.manager.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            msg = f"OpenAI call failed: {exc}"
            log.error(msg)
            self.manager.notify_clients(
                title="LLM Query",
                body="LLM request failed; check logs and credentials.",
                client=client,
                color="red",
                with_close_button=True,
            )
            return

        ranked = self._extract_object_ids(content)
        if not ranked:
            msg = "LLM returned no valid object ids for this query."
            self.manager.notify_clients(
                title="LLM Query",
                body=msg,
                client=client,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=2.0,
            )
            log.warning(msg)
            return

        self._apply_llm_palette(ranked)
        summary = ", ".join(self.manager.labels[idx] for idx in ranked)
        result_msg = f"LLM query: '{query_text}' → {summary}"
        log.info(result_msg)
        return summary

    def respond(self, message: str) -> str:
        """Answer chat messages using the ChatAgent (if available) or fallback to LLM."""
        user_text = message.strip()
        if not user_text:
            return ""

        # Use the ChatAgent if available
        if self.manager.chat_agent is not None:
            return self.manager.chat_agent.respond(user_text)

        # Fallback to direct LLM query if ChatAgent not available
        if self.manager.llm_client is None or not self.manager.can_run_llm_query:
            reply = "Agent is unavailable because OPENAI_API_KEY or object data is missing."
            self.add_chat_message("Agent", reply)
            return reply

        messages = self._build_llm_messages(user_text)

        try:
            response = self.manager.llm_client.chat.completions.create(
                model=self.manager.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            reply = f"LLM chat failed: {exc}"
            log.error(reply)
            self.add_chat_message("Agent", reply)
            return reply

        ranked = self._extract_object_ids(content)
        if ranked:
            self._apply_llm_palette(ranked)
            reply = f"Top objects: {', '.join(self.manager.labels[i] for i in ranked)}"
        else:
            reply = "No suitable objects found for that request."

        self.add_chat_message("Agent", reply)
        return reply


def setup_llm_query_gui(server, manager, gui_cfg):
    """Set up the GUI controls for LLM query."""
    with server.gui.add_folder("LLM Query", expand_by_default=gui_cfg.llm_query.expanded):
        llm_query_input = server.gui.add_text("Query", initial_value="")
        llm_query_btn = server.gui.add_button("Search")
        manager.query_chat.set_llm_query_controls(llm_query_input, llm_query_btn)
        llm_query_input.disabled = not manager.can_run_llm_query
        llm_query_btn.disabled = not manager.can_run_llm_query

        @llm_query_btn.on_click
        def _(event):
            if llm_query_input.value.strip():
                manager.llm_query(llm_query_input.value.strip(), client=event.client)


def setup_agent_gui(server, manager, gui_cfg):
    """Set up the GUI controls for the chat agent."""
    with server.gui.add_folder("Agent", expand_by_default=gui_cfg.agent.expanded):
        # Use HTML handle so we can wrap messages in a scrollable container.
        chat_history = server.gui.add_html("<em>No messages yet.</em>")
        manager.register_chat_markdown(chat_history)

        chat_input = server.gui.add_text("Message", initial_value="")

        send_btn = server.gui.add_button("Send")
        manager.query_chat.set_chat_controls(chat_input, send_btn)
        agent_enabled = manager.chat_agent is not None or manager.can_run_llm_query
        chat_input.disabled = not agent_enabled
        send_btn.disabled = not agent_enabled

        @send_btn.on_click
        def _(event):
            user_text = chat_input.value.strip()
            if not user_text:
                return

            manager.respond(user_text)
            chat_input.value = ""

        clear_chat_btn = server.gui.add_button("Clear Chat", hint="Wipe chat history and agent context")

        @clear_chat_btn.on_click
        def _(event):
            """Clear the chat history and agent context."""
            manager.clear_chat()
