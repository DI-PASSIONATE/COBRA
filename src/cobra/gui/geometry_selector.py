import importlib
import importlib.util
import inspect
import os
import pkgutil

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QWidget, QHBoxLayout, QMessageBox, QFileDialog,
)


class GeometrySelectorWidget(QGroupBox):
    """
    Per-component ORCA geometry selector widget for EM fine-tuning.
    Supports both ORCA preset geometries and custom Python-file-based geometries.
    """

    def __init__(self, title: str = "ORCA Geometry", parent=None):
        super().__init__(title, parent)
        self._preset_classes: dict = {}
        self._custom_classes: dict = {}

        form = QFormLayout()

        self._source_combo = QComboBox()
        self._source_combo.addItem("ORCA Preset", "preset")
        self._source_combo.addItem("Custom Python File", "custom")
        form.addRow("Source:", self._source_combo)

        self._preset_label = QLabel("Preset:")
        self._preset_combo = QComboBox()
        form.addRow(self._preset_label, self._preset_combo)

        self._file_label = QLabel("File:")
        self._file_edit = QLineEdit()
        self._file_btn = QPushButton("Browse")
        self._file_btn.clicked.connect(self._browse_file)
        self._file_widget = QWidget()
        file_layout = QHBoxLayout(self._file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self._file_edit)
        file_layout.addWidget(self._file_btn)
        form.addRow(self._file_label, self._file_widget)

        self._class_label = QLabel("Class:")
        self._class_combo = QComboBox()
        form.addRow(self._class_label, self._class_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        self._source_combo.currentIndexChanged.connect(self._update_source_visibility)
        self._update_source_visibility()

    # ------------------------------------------------------------------
    def _update_source_visibility(self):
        use_preset = self._source_combo.currentData() == "preset"
        self._preset_label.setVisible(use_preset)
        self._preset_combo.setVisible(use_preset)
        self._file_label.setVisible(not use_preset)
        self._file_widget.setVisible(not use_preset)
        self._class_label.setVisible(not use_preset)
        self._class_combo.setVisible(not use_preset)

        if use_preset and self._preset_combo.count() == 0:
            self.reload_presets(show_errors=False)

    def _browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Geometry File", "", "Python Files (*.py)")
        if fname:
            self._file_edit.setText(fname)
            self.load_custom_file(fname)

    # ------------------------------------------------------------------
    def reload_presets(self, show_errors: bool = True):
        current_label = self._preset_combo.currentText()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_classes = {}

        try:
            BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry
            presets = importlib.import_module("orca.geometry.presets")

            discovered = []
            for _, module_name, _ in pkgutil.iter_modules(presets.__path__):
                module = importlib.import_module(f"orca.geometry.presets.{module_name}")
                for class_name, cls in inspect.getmembers(module, inspect.isclass):
                    if cls is BaseGeometry or not issubclass(cls, BaseGeometry):
                        continue
                    if cls.__module__ != module.__name__:
                        continue
                    label = class_name
                    if label in self._preset_classes:
                        label = f"{class_name} ({module_name})"
                    discovered.append((label, cls))

            for label, cls in sorted(discovered, key=lambda x: x[0].lower()):
                self._preset_classes[label] = cls
                self._preset_combo.addItem(label, cls)

            if current_label:
                idx = self._preset_combo.findText(current_label)
                if idx >= 0:
                    self._preset_combo.setCurrentIndex(idx)
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "ORCA Geometry", f"Failed to load ORCA presets:\n{exc}")
        finally:
            self._preset_combo.blockSignals(False)

    def load_custom_file(self, file_path: str, show_errors: bool = True) -> bool:
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._custom_classes = {}

        if not file_path:
            self._class_combo.blockSignals(False)
            return False

        try:
            BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry

            abs_path = os.path.abspath(file_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"Geometry file not found: {abs_path}")

            module_name = f"cobra_custom_geometry_{abs(hash(abs_path))}"
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load geometry module from {abs_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            classes = []
            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if cls is BaseGeometry or not issubclass(cls, BaseGeometry):
                    continue
                if cls.__module__ != module.__name__:
                    continue
                classes.append((class_name, cls))

            if not classes:
                raise ValueError("No classes extending BaseGeometry were found in the selected file.")

            for class_name, cls in sorted(classes, key=lambda x: x[0].lower()):
                self._custom_classes[class_name] = cls
                self._class_combo.addItem(class_name, cls)

            return True
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "ORCA Geometry", f"Failed to load custom geometry:\n{exc}")
            return False
        finally:
            self._class_combo.blockSignals(False)

    def get_geometry(self):
        """Instantiate and return the selected geometry class."""
        if self._source_combo.currentData() == "custom":
            file_path = self._file_edit.text().strip()
            if not file_path:
                raise ValueError("Please select a custom geometry Python file.")
            self.load_custom_file(file_path)

        if self._source_combo.currentData() == "preset":
            cls = self._preset_combo.currentData()
        else:
            cls = self._class_combo.currentData()

        if cls is None:
            raise ValueError("Please select an ORCA geometry class.")

        return cls()
