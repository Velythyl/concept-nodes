import logging
from typing import Any

import viser


class NotificationManager:
    """Centralizes Viser notifications with shared defaults."""

    def __init__(
        self,
        server: viser.ViserServer,
        *,
        default_auto_close_seconds: float = 2.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.server = server
        self.default_auto_close_seconds = default_auto_close_seconds
        self.log = logger or logging.getLogger(__name__)

    def notify(
        self,
        title: str,
        body: str,
        *,
        client: viser.ClientHandle | None = None,
        **kwargs: Any,
    ) -> None:
        """Send a notification to one client or all connected clients."""
        targets = [client] if client is not None else list(self.server.get_clients().values())
        if not targets:
            self.log.debug("No connected clients to notify: %s - %s", title, body)
            return

        if "auto_close_seconds" not in kwargs:
            kwargs["auto_close_seconds"] = self.default_auto_close_seconds

        for target in targets:
            try:
                target.add_notification(title=title, body=body, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.log.warning(
                    "Failed to send notification to client %s: %s",
                    getattr(target, "client_id", "unknown"),
                    exc,
                )
