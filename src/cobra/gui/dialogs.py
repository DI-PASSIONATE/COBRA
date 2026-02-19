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
        
        if goal:
            self.param_combo.setCurrentText(goal.parameter.value)
            if goal.min_value is not None:
                self.min_edit.setText(str(goal.min_value))
            if goal.max_value is not None:
                self.max_edit.setText(str(goal.max_value))
        
        self.form_layout.addRow("Parameter:", self.param_combo)
        self.form_layout.addRow("Min Value:", self.min_edit)
        self.form_layout.addRow("Max Value:", self.max_edit)
        
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
             # Create a dynamic object mimicking the Enum
             class SimulatedParameter:
                 def __init__(self, value):
                     self.value = value
             param = SimulatedParameter(p_name)  # type: ignore

        min_val = float(self.min_edit.text()) if self.min_edit.text() else None
        max_val = float(self.max_edit.text()) if self.max_edit.text() else None
        return DesignGoal(param, min_val, max_val)

class OptimizationParamDialog(QDialog):
    def __init__(self, from_source=None, source_data=None, parent=None, param: Optional[OptimizationProperty] = None, metadata: Optional[dict] = None):
        super().__init__(parent)
        self.metadata = metadata or {}
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
        
        if param:
            self.name_edit.setText(param.name)
            self.type_combo.setCurrentText(param.type.value)
            self.min_spin.setValue(param.min_value)
            self.max_spin.setValue(param.max_value)
            if param.step:
                self.step_spin.setValue(param.step)
            if param.unit:
                self.unit_edit.setText(param.unit)
        
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
        elif from_source == "NETLIST" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name:", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.NETLIST_VARIABLE.value)
            self.type_combo.setEnabled(False)
            self.step_spin.setValue(1.0)
            self.use_combo_name = True
        else:
            self.form_layout.addRow("Name:", self.name_edit)
            self.use_combo_name = False
            
        self.form_layout.addRow("Type:", self.type_combo)
        self.form_layout.addRow("Min:", self.min_spin)
        self.form_layout.addRow("Max:", self.max_spin)
        self.form_layout.addRow("Step:", self.step_spin)
        self.form_layout.addRow("Unit:", self.unit_edit)
        
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

    def get_data(self):
        name = self.name_combo.currentText() if self.use_combo_name else self.name_edit.text()
        t_str = self.type_combo.currentText()
        t = OptimizationType(t_str)
        unit = self.unit_edit.text().strip() or None
        return OptimizationProperty(name, t, self.min_spin.value(), self.max_spin.value(), self.step_spin.value() or None, unit=unit)
