"""GUI entry point. Run with ``python -m cosmic_toolbox_gui.app`` or ``cosmic-toolbox-gui``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QSplashScreen

from cosmic_toolbox_gui.main_window import GroundStationApp
from cosmic_toolbox_gui.opengl import preload_globe_textures


def _gui_resource(*parts: str) -> Path:
    """Resolve a path inside ``cosmic_toolbox_gui/resources/``."""

    return Path(__file__).resolve().parent / "resources" / Path(*parts)


class LoadingSplashScreen(QSplashScreen):
    """Splash screen with an integrated progress label/bar."""

    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__(pixmap)
        bar_width = max(240, min(pixmap.width() - 40, 480))
        bar_height = 26
        bar_x = (pixmap.width() - bar_width) // 2
        bar_y = pixmap.height() - bar_height - 24

        self._label = QLabel("Loading textures…", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: 600; text-shadow: 0 0 4px #000;"
        )
        self._label.setGeometry(10, bar_y - 32, pixmap.width() - 20, 24)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress.setTextVisible(True)
        self._progress.setFormat("0%")
        self._progress.setGeometry(bar_x, bar_y, bar_width, bar_height)
        self._progress.setStyleSheet(
            """
            QProgressBar {
                background-color: rgba(0, 0, 0, 160);
                color: white;
                border: 1px solid rgba(255, 255, 255, 180);
                border-radius: 6px;
                padding: 2px;
            }
            QProgressBar::chunk {
                background-color: #33aaff;
                border-radius: 4px;
            }
            """
        )

    def update_status(self, message: str, progress: float) -> None:
        clamped = max(0.0, min(1.0, progress))
        self._label.setText(message)
        percent = int(clamped * 100)
        self._progress.setValue(percent)
        self._progress.setFormat(f"{percent}%")


def main() -> int:
    """Start the Qt event loop."""

    log_file = Path.cwd() / "application.log"
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=str(log_file),
        filemode="w",
    )

    app = QApplication(sys.argv)
    app.setFont(QFont(app.font().family(), 12))
    icon_path = _gui_resource("img", "menu_icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash: LoadingSplashScreen | None = None
    splash_path = _gui_resource("img", "splash2.png")
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                pixmap.width() // 2,
                pixmap.height() // 2,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            splash = LoadingSplashScreen(scaled)
            splash.show()
            app.processEvents()

    def _report_progress(message: str, value: float) -> None:
        if not splash:
            return
        splash.update_status(message, value)
        app.processEvents()

    preload_globe_textures(progress_callback=_report_progress if splash else None)

    window = GroundStationApp()
    window.showMaximized()

    if splash:
        splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
