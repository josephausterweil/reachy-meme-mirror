"""meme_mirror as a Reachy Mini app (installable from the Reachy Mini app store / dashboard).

The app serves the same web page as `uvicorn meme_mirror.server:app`, on the robot (or on the computer running
the daemon, for Reachy Mini Lite), and drives the robot through the connection the app framework provides.
Open it from the dashboard's app settings button, or at http://<robot>:8043.
"""

import logging
import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from . import robot
from .server import app as web_app

log = logging.getLogger("meme_mirror")


class MemeMirrorApp(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8043"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.settings_app is not None:  # the framework's web server: serve the whole meme_mirror UI on it
            self.settings_app.mount("/", web_app)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        robot.attach(reachy_mini)
        log.info("meme_mirror ready: open the app's settings page (port 8043)")
        try:
            stop_event.wait()
        finally:
            robot.detach()


if __name__ == "__main__":
    MemeMirrorApp().wrapped_run()
