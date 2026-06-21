import json
from typing import List, Optional
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QLineEdit,
    QDoubleSpinBox, QMessageBox,
)
from PySide6.QtGui import QDoubleValidator

from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from .help_texts import tooltip

class DesignGoalDialog(QDialog):
    def __init__(self, parent=None, goal=None, available_parameters: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Design Goal")
        self.setMinimumWidth(400)
        self.form_layout = QFormLayout(self)
        
        self.param_combo = QComboBox()
        if available_parameters:
            self.param_combo.addItems(available_parameters)
        
        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("Default: 1.0")
        self.weight_edit.setText("1.0")
        weight_validator = QDoubleValidator(0.0, 1e15, 15, self)
        weight_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.weight_edit.setValidator(weight_validator)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setPlaceholderText("Optional")
        self.max_edit.setPlaceholderText("Optional")
        value_validator = QDoubleValidator(-1e15, 1e15, 15, self)
        value_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.min_edit.setValidator(value_validator)
        self.max_edit.setValidator(value_validator)

        self.freq_min_edit = QLineEdit()
        self.freq_max_edit = QLineEdit()
        self.freq_min_edit.setPlaceholderText("Optional")
        self.freq_max_edit.setPlaceholderText("Optional")
        freq_validator = QDoubleValidator(0.0, 1e15, 15, self)
        freq_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.freq_min_edit.setValidator(freq_validator)
        self.freq_max_edit.setValidator(freq_validator)
        self.freq_unit_combo = QComboBox()
        self.freq_unit_combo.addItems(["Hz", "kHz", "MHz", "GHz", "THz"])
        self.freq_unit_combo.setCurrentText("GHz")
        self.freq_unit_combo.setToolTip(tooltip("freq_unit_combo"))

        self._raw_frequency_range = None
        
        if goal:
            p_name = goal.parameter_name
            # If the goal's parameter isn't in the combo (e.g. netlist was changed),
            # add it at the top so the existing goal can still be viewed/edited.
            if self.param_combo.findText(p_name) == -1:
                self.param_combo.insertItem(0, p_name)
            self.param_combo.setCurrentText(p_name)
            if goal.min_value is not None:
                self.min_edit.setText(str(goal.min_value))
            if goal.max_value is not None:
                self.max_edit.setText(str(goal.max_value))
            if goal.frequency_range:
                parsed = self._parse_frequency_range(goal.frequency_range)
                if parsed is None:
                    self._raw_frequency_range = goal.frequency_range
                else:
                    min_freq, max_freq, unit = parsed
                    self.freq_min_edit.setText(min_freq)
                    self.freq_max_edit.setText(max_freq)
                    unit_label = self._normalize_frequency_unit(unit)
                    if unit_label is not None:
                        self.freq_unit_combo.setCurrentText(unit_label)
            if goal.weight is not None:
                self.weight_edit.setText(str(goal.weight))
        
        self.form_layout.addRow("Parameter:", self.param_combo)
        self.form_layout.addRow("Weight:", self.weight_edit)
        self.form_layout.addRow("Min Value:", self.min_edit)
        self.form_layout.addRow("Max Value:", self.max_edit)
        self.form_layout.addRow("Min Frequency:", self.freq_min_edit)
        self.form_layout.addRow("Max Frequency:", self.freq_max_edit)
        self.form_layout.addRow("Frequency Unit:", self.freq_unit_combo)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.form_layout.addRow(self.buttons)

    def get_data(self):
        p_name = self.param_combo.currentText()
        # Try to resolve to a DesignParameter enum member first (lumped params).
        # S-parameter strings (e.g. "S21_dB") are kept as plain strings.
        param = None
        for p in DesignParameter:
            if p.value == p_name:
                param = p
                break
        # Fall back to plain string for S-parameter goals and any future types.
        if param is None:
            param = p_name
            

        min_val = float(self.min_edit.text()) if self.min_edit.text() else None
        max_val = float(self.max_edit.text()) if self.max_edit.text() else None
        
        freq_range, freq_error = self._build_frequency_range()
        if freq_error:
            raise ValueError(freq_error)
        
        try:
            weight = float(self.weight_edit.text())
        except ValueError:
            weight = 1.0

        return DesignGoal(parameter=param, frequency_range=freq_range, min_value=min_val, max_value=max_val, weight=weight)

    def accept(self):
        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip()
        weight_text = self.weight_edit.text().strip()
        if min_text and not self.min_edit.hasAcceptableInput():
            QMessageBox.warning(self, "Invalid Value", "Min Value must be numeric.")
            return
        if max_text and not self.max_edit.hasAcceptableInput():
            QMessageBox.warning(self, "Invalid Value", "Max Value must be numeric.")
            return
        if weight_text and not self.weight_edit.hasAcceptableInput():
            QMessageBox.warning(self, "Invalid Weight", "Weight must be numeric.")
            return
        if not min_text and not max_text:
            QMessageBox.warning(
                self,
                "Missing Value Range",
                "At least one of Min Value or Max Value must be set.",
            )
            return
        _, freq_error = self._build_frequency_range()
        if freq_error:
            QMessageBox.warning(self, "Invalid Frequency Range", freq_error)
            return
        super().accept()

    def _build_frequency_range(self):
        freq_min_text = self.freq_min_edit.text().strip()
        freq_max_text = self.freq_max_edit.text().strip()
        if freq_min_text or freq_max_text:
            if not (freq_min_text and freq_max_text):
                return None, "Both Min Frequency and Max Frequency are required when setting a frequency range."
            try:
                float(freq_min_text)
                float(freq_max_text)
            except ValueError:
                return None, "Frequency values must be numeric."
            unit = self.freq_unit_combo.currentText().lower()
            return f"{freq_min_text}-{freq_max_text}{unit}", None
        if self._raw_frequency_range:
            return self._raw_frequency_range, None
        return None, None

    @staticmethod
    def _parse_frequency_range(frequency_range: str):
        if not frequency_range:
            return None
        parts = frequency_range.strip().split("-")
        if len(parts) != 2:
            return None
        min_part = parts[0].strip()
        max_part = parts[1].strip()
        if not min_part or not max_part:
            return None
        unit = ""
        max_digits = []
        for ch in max_part:
            if ch.isdigit() or ch == ".":
                max_digits.append(ch)
            else:
                unit = max_part[len(max_digits):].strip()
                break
        if not unit:
            return None
        max_value = "".join(max_digits)
        if not max_value:
            return None
        return min_part, max_value, unit

    @staticmethod
    def _normalize_frequency_unit(unit: str):
        unit_map = {
            "hz": "Hz",
            "khz": "kHz",
            "mhz": "MHz",
            "ghz": "GHz",
            "thz": "THz",
        }
        return unit_map.get(unit.strip().lower())

class OptimizationParamDialog(QDialog):
    def __init__(
        self,
        from_source=None,
        source_data=None,
        parent=None,
        param: Optional[OptimizationProperty] = None,
        metadata: Optional[dict] = None,
        link_candidates: Optional[List[str]] = None,
    ):
        super().__init__(parent)
        self.metadata = metadata or {}
        self.link_candidates = link_candidates or []
        self.setWindowTitle("Optimization Parameter")
        self.form_layout = QFormLayout(self)
        
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems([t.value for t in OptimizationType])
        
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e15, 1e15)
        self.min_spin.setDecimals(15)
        
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e15, 1e15)
        self.max_spin.setDecimals(15)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0, 1e15)
        self.step_spin.setDecimals(15)
        
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("Optional (e.g. F (=femto)) for Xyce")
        self.link_to_combo = QComboBox()
        self.link_to_combo.addItem("None", None)
        
        if param:
            self.name_edit.setText(param.name)
            self.type_combo.setCurrentText(param.type.value)
            self.min_spin.setValue(param.min_value)
            self.max_spin.setValue(param.max_value)
            if param.step:
                self.step_spin.setValue(param.step)
            if param.unit:
                self.unit_edit.setText(param.unit)
            if param.linked_to:
                self.link_to_combo.addItem(param.linked_to, param.linked_to)
                self.link_to_combo.setCurrentIndex(self.link_to_combo.count() - 1)
        
        if from_source == "ONNX" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name:", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.MODEL_INPUT.value)
            self.type_combo.setEnabled(False)
            self.use_combo_name = True
            
            self.step_spin.setValue(0.1)
            self.name_combo.currentTextChanged.connect(self._update_onnx_metadata)
            self._update_onnx_metadata(self.name_combo.currentText())
            self.name_combo.currentTextChanged.connect(self._refresh_link_targets)
        elif from_source == "NETLIST" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name:", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.NETLIST_VARIABLE.value)
            self.type_combo.setEnabled(False)
            self.step_spin.setValue(1.0)
            self.use_combo_name = True
            self.name_combo.currentTextChanged.connect(self._refresh_link_targets)
        else:
            self.form_layout.addRow("Name:", self.name_edit)
            self.use_combo_name = False
            self.name_edit.textChanged.connect(self._refresh_link_targets)
            
        self.form_layout.addRow("Type:", self.type_combo)
        self.form_layout.addRow("Min:", self.min_spin)
        self.form_layout.addRow("Max:", self.max_spin)
        self.form_layout.addRow("Step:", self.step_spin)
        self.form_layout.addRow("Unit:", self.unit_edit)
        self.form_layout.addRow("Link To:", self.link_to_combo)

        self._refresh_link_targets(self.name_combo.currentText() if self.use_combo_name else self.name_edit.text())
        if param and param.linked_to:
            idx = self.link_to_combo.findData(param.linked_to)
            if idx >= 0:
                self.link_to_combo.setCurrentIndex(idx)

        self.link_to_combo.currentIndexChanged.connect(self._on_link_target_changed)
        self._on_link_target_changed()
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.form_layout.addRow(self.buttons)

    def _update_onnx_metadata(self, name):
        # Allow checking metadata natively and without prefix (e.g. 'W' from 'X1:W')
        base_name = name.split(":", 1)[1] if ":" in name else name
        
        if "input_parameter_ranges" in self.metadata:
            try:
                meta_data = json.loads(self.metadata["input_parameter_ranges"])
                params = None
                if name in meta_data:
                    params = meta_data[name]
                elif base_name in meta_data:
                    params = meta_data[base_name]

                if params:
                    if "min" in params:
                        self.min_spin.setValue(float(params["min"]))
                    if "max" in params:
                        self.max_spin.setValue(float(params["max"]))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Error parsing ONNX metadata: {e}")

    def _refresh_link_targets(self, current_name):
        current_name = (current_name or "").strip()
        selected = self.link_to_combo.currentData()
        self.link_to_combo.blockSignals(True)
        self.link_to_combo.clear()
        self.link_to_combo.addItem("None", None)
        for candidate in self.link_candidates:
            if candidate != current_name:
                self.link_to_combo.addItem(candidate, candidate)
        if selected is not None:
            idx = self.link_to_combo.findData(selected)
            if idx >= 0:
                self.link_to_combo.setCurrentIndex(idx)
        self.link_to_combo.blockSignals(False)
        self._on_link_target_changed()

    def _on_link_target_changed(self):
        linked = self.link_to_combo.currentData() is not None
        self.min_spin.setEnabled(not linked)
        self.max_spin.setEnabled(not linked)
        self.step_spin.setEnabled(not linked)
        self.unit_edit.setEnabled(not linked)

    def get_data(self):
        name = self.name_combo.currentText() if self.use_combo_name else self.name_edit.text()
        t_str = self.type_combo.currentText()
        t = OptimizationType(t_str)
        unit = self.unit_edit.text().strip() or None
        linked_to = self.link_to_combo.currentData()
        return OptimizationProperty(
            name,
            t,
            self.min_spin.value(),
            self.max_spin.value(),
            self.step_spin.value() or None,
            unit=unit,
            linked_to=linked_to,
        )

