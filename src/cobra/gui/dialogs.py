import json
from typing import List, Optional
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QLineEdit, 
    QDoubleSpinBox
)

from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter

def clean_name(name):
    # Sanitize name if needed
    return name

class DesignGoalDialog(QDialog):
    def __init__(self, parent=None, goal=None):
        super().__init__(parent)
        self.setWindowTitle("Design Goal")
        self.form_layout = QFormLayout(self)
        
        self.param_combo = QComboBox()
        # Populate with DesignParameters + common metrics if implied
        common_params = [p.value for p in DesignParameter]
        # Remove duplicates
        common_params = sorted(list(set(common_params)))
        self.param_combo.addItems(common_params)
        
        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setPlaceholderText("Optional")
        self.max_edit.setPlaceholderText("Optional")
        
        self.freq_range_edit = QLineEdit()
        self.freq_range_edit.setPlaceholderText("Optional (e.g. 125-135ghz)")

        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("Default: 1.0")
        self.weight_edit.setText("1.0")
        
        if goal:
            self.param_combo.setCurrentText(goal.parameter.value)
            if goal.min_value is not None:
                self.min_edit.setText(str(goal.min_value))
            if goal.max_value is not None:
                self.max_edit.setText(str(goal.max_value))
            if goal.frequency_range:
                self.freq_range_edit.setText(goal.frequency_range)
            if goal.weight is not None:
                self.weight_edit.setText(str(goal.weight))
        
        self.form_layout.addRow("Parameter:", self.param_combo)
        self.form_layout.addRow("Min Value:", self.min_edit)
        self.form_layout.addRow("Max Value:", self.max_edit)
        self.form_layout.addRow("Frequency Range:", self.freq_range_edit)
        self.form_layout.addRow("Weight:", self.weight_edit)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.form_layout.addRow(self.buttons)

    def get_data(self):
        p_name = self.param_combo.currentText()
        param = None
        for p in DesignParameter:
            if p.value == p_name:
                param = p
                break
        
        if param is None:
            raise ValueError(f"Selected parameter '{p_name}' is not a valid DesignParameter.")
            

        min_val = float(self.min_edit.text()) if self.min_edit.text() else None
        max_val = float(self.max_edit.text()) if self.max_edit.text() else None
        
        freq_range = self.freq_range_edit.text().strip() or None
        
        try:
            weight = float(self.weight_edit.text())
        except ValueError:
            weight = 1.0

        return DesignGoal(parameter=param, frequency_range=freq_range, min_value=min_val, max_value=max_val, weight=weight)

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
        if "input_parameter_ranges" in self.metadata:
            try:
                meta_data = json.loads(self.metadata["input_parameter_ranges"])
                if name in meta_data:
                    params = meta_data[name]
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
