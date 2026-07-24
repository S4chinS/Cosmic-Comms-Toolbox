"""Two-handle range slider widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider


class RangeSlider(QSlider):
    """Slider with two handles for selecting a value range."""

    valuesChanged = Signal(int, int)
    lowerValueChanged = Signal(int)
    upperValueChanged = Signal(int)

    def __init__(self, orientation: Qt.Orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self._lower = self.minimum()
        self._upper = self.maximum()
        self._active_handle: str | None = None
        self.setTickPosition(QSlider.TickPosition.NoTicks)

    def lowerValue(self) -> int:
        return int(self._lower)

    def upperValue(self) -> int:
        return int(self._upper)

    def setLowerValue(self, value: int) -> None:
        value = int(value)
        if value < self.minimum():
            value = self.minimum()
        if value > self._upper:
            value = self._upper
        if value == self._lower:
            return
        self._lower = value
        self.lowerValueChanged.emit(self._lower)
        self.valuesChanged.emit(self._lower, self._upper)
        self.update()

    def setUpperValue(self, value: int) -> None:
        value = int(value)
        if value > self.maximum():
            value = self.maximum()
        if value < self._lower:
            value = self._lower
        if value == self._upper:
            return
        self._upper = value
        self.upperValueChanged.emit(self._upper)
        self.valuesChanged.emit(self._lower, self._upper)
        self.update()

    def setValues(self, lower: int, upper: int) -> None:
        if int(lower) > int(upper):
            raise ValueError("Lower value cannot exceed upper value.")
        self.setLowerValue(int(lower))
        self.setUpperValue(int(upper))

    def setRange(self, minimum: int, maximum: int) -> None:
        super().setRange(minimum, maximum)
        if self._lower < minimum:
            self._lower = minimum
        if self._upper > maximum:
            self._upper = maximum
        if self._lower > self._upper:
            self._lower = minimum
            self._upper = maximum
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        style = self.style()

        groove_opt = QStyleOptionSlider()
        self.initStyleOption(groove_opt)
        groove_opt.subControls = QStyle.SC_SliderGroove
        style.drawComplexControl(QStyle.CC_Slider, groove_opt, painter, self)

        lower_opt = QStyleOptionSlider()
        self.initStyleOption(lower_opt)
        lower_opt.subControls = QStyle.SC_SliderHandle
        lower_opt.sliderPosition = self._lower
        lower_opt.sliderValue = self._lower
        lower_rect = style.subControlRect(
            QStyle.CC_Slider, lower_opt, QStyle.SC_SliderHandle, self
        )

        upper_opt = QStyleOptionSlider()
        self.initStyleOption(upper_opt)
        upper_opt.subControls = QStyle.SC_SliderHandle
        upper_opt.sliderPosition = self._upper
        upper_opt.sliderValue = self._upper
        upper_rect = style.subControlRect(
            QStyle.CC_Slider, upper_opt, QStyle.SC_SliderHandle, self
        )

        groove_rect = style.subControlRect(
            QStyle.CC_Slider, groove_opt, QStyle.SC_SliderGroove, self
        )
        highlight = QColor("#4fc3f7")
        if self.orientation() == Qt.Orientation.Horizontal:
            left = min(lower_rect.center().x(), upper_rect.center().x())
            right = max(lower_rect.center().x(), upper_rect.center().x())
            range_rect = QRect(
                left,
                groove_rect.center().y() - 2,
                max(1, right - left),
                4,
            )
        else:
            top = min(lower_rect.center().y(), upper_rect.center().y())
            bottom = max(lower_rect.center().y(), upper_rect.center().y())
            range_rect = QRect(
                groove_rect.center().x() - 2,
                top,
                4,
                max(1, bottom - top),
            )
        painter.fillRect(range_rect, highlight)

        style.drawComplexControl(QStyle.CC_Slider, lower_opt, painter, self)
        style.drawComplexControl(QStyle.CC_Slider, upper_opt, painter, self)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        pos = event.position().toPoint()
        value = self._pixel_pos_to_value(pos)
        lower_dist = abs(value - self._lower)
        upper_dist = abs(value - self._upper)
        self._active_handle = "lower" if lower_dist <= upper_dist else "upper"
        self._set_active_value(value)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._active_handle is None:
            return
        pos = event.position().toPoint()
        value = self._pixel_pos_to_value(pos)
        self._set_active_value(value)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._active_handle = None
        event.accept()

    def _set_active_value(self, value: int) -> None:
        if self._active_handle == "lower":
            self.setLowerValue(min(value, self._upper))
        elif self._active_handle == "upper":
            self.setUpperValue(max(value, self._lower))

    def _pixel_pos_to_value(self, pos) -> int:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        if self.orientation() == Qt.Orientation.Horizontal:
            slider_length = style.pixelMetric(QStyle.PM_SliderLength, opt, self)
            slider_min = opt.rect.x()
            slider_max = opt.rect.right() - slider_length + 1
            return int(
                QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    pos.x() - slider_min,
                    slider_max - slider_min,
                    opt.upsideDown,
                )
            )
        slider_length = style.pixelMetric(QStyle.PM_SliderLength, opt, self)
        slider_min = opt.rect.y()
        slider_max = opt.rect.bottom() - slider_length + 1
        return int(
            QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos.y() - slider_min,
                slider_max - slider_min,
                opt.upsideDown,
            )
        )
