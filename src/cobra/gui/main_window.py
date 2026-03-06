from typing import Any, Dict, List, Optional, get_args, get_origin
import ast
import importlib
import importlib.util
import inspect
import os
import pkgutil
import numpy as np
import onnxruntime
import gmsh

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
    QListWidget, QTableWidget, QTableWidgetItem, 
    QSpinBox, QGroupBox, QFormLayout, QScrollArea,
    QHeaderView, QMessageBox, QProgressBar, QCheckBox, QMenu, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Slot
import pyqtgraph as pg

# COBRA imports
from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty
from cobra.optimizers.design_goal import DesignGoal
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.optimizers.optuna_optimizer import OptunaOptimizer

from .dialogs import DesignGoalDialog, OptimizationParamDialog, clean_name
from .worker import OptimizationWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        gmsh.initialize()
        self.setWindowTitle("COBRA GUI")
        self.resize(1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left Panel: Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # 1. File Selection
        self.config_form_layout = QFormLayout()
        
        self.onnx_edit = QLineEdit()
        self.onnx_btn = QPushButton("Browse")
        self.onnx_btn.clicked.connect(lambda: self.browse_file(self.onnx_edit, "ONNX Files (*.onnx)"))
        h_onnx = QHBoxLayout()
        h_onnx.addWidget(self.onnx_edit)
        h_onnx.addWidget(self.onnx_btn)
        self.config_form_layout.addRow("ONNX Model:", h_onnx)
        
        self.netlist_edit = QLineEdit()
        self.netlist_btn = QPushButton("Browse")
        self.netlist_btn.clicked.connect(lambda: self.browse_file(self.netlist_edit, "Netlist Files (*.cir *.sp)"))
        h_net = QHBoxLayout()
        h_net.addWidget(self.netlist_edit)
        h_net.addWidget(self.netlist_btn)
        self.config_form_layout.addRow("Netlist:", h_net)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("OptunaOptimizer", OptunaOptimizer)
        self.config_form_layout.addRow("Optimizer:", self.optimizer_combo)
        
        self.simulator_combo = QComboBox()
        self.simulator_combo.addItem("XyceSimulator", XyceSimulator)
        self.config_form_layout.addRow("Simulator:", self.simulator_combo)
        
        # Track position for inserting dynamic simulator options
        self.sim_options_insert_pos = self.config_form_layout.rowCount()
        self.sim_options_count = 0
        self.simulator_combo.currentIndexChanged.connect(self.update_simulator_options)
        self.simulator_widgets = {}
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 99999)
        self.max_iter_spin.setValue(500)
        self.config_form_layout.addRow("Max Iterations:", self.max_iter_spin)

        # self.moo_cb = QCheckBox("Multi-Objective Optimization")
        # form_layout.addRow("", self.moo_cb)

        #### OPTIONAL - Fine-tuning with palace ####
        self.finetune_cb = QCheckBox("Perform finetuning")
        self.finetune_cb.setToolTip("Runs a few iterations of palace simulations instead of the surrogate model at the end to ensure correct predictions")
        self.config_form_layout.addRow("", self.finetune_cb)

        self.palace_label = QLabel("Palace Command:")
        self.palace_edit = QLineEdit("palace")
        self.config_form_layout.addRow(self.palace_label, self.palace_edit)

        self.ft_iter_label = QLabel("Finetuning Iterations:")
        self.ft_iter_spin = QSpinBox()
        self.ft_iter_spin.setRange(1, 100)
        self.ft_iter_spin.setValue(3)
        self.config_form_layout.addRow(self.ft_iter_label, self.ft_iter_spin)

        self.geometry_group = QGroupBox("ORCA Geometry")
        geometry_group_layout = QVBoxLayout(self.geometry_group)

        self.geometry_form_layout = QFormLayout()

        self.geometry_source_label = QLabel("Geometry Source:")
        self.geometry_source_combo = QComboBox()
        self.geometry_source_combo.addItem("ORCA Preset", "preset")
        self.geometry_source_combo.addItem("Custom Python File", "custom")
        self.geometry_form_layout.addRow(self.geometry_source_label, self.geometry_source_combo)

        self.geometry_preset_label = QLabel("Preset:")
        self.geometry_preset_combo = QComboBox()
        self.geometry_form_layout.addRow(self.geometry_preset_label, self.geometry_preset_combo)

        self.geometry_file_label = QLabel("Custom File:")
        self.geometry_file_edit = QLineEdit()
        self.geometry_file_btn = QPushButton("Browse")
        self.geometry_file_btn.clicked.connect(self.browse_geometry_file)
        self.geometry_file_widget = QWidget()
        geometry_file_layout = QHBoxLayout(self.geometry_file_widget)
        geometry_file_layout.setContentsMargins(0, 0, 0, 0)
        geometry_file_layout.addWidget(self.geometry_file_edit)
        geometry_file_layout.addWidget(self.geometry_file_btn)
        self.geometry_form_layout.addRow(self.geometry_file_label, self.geometry_file_widget)

        self.geometry_class_label = QLabel("Geometry Class:")
        self.geometry_class_combo = QComboBox()
        self.geometry_form_layout.addRow(self.geometry_class_label, self.geometry_class_combo)

        geometry_group_layout.addLayout(self.geometry_form_layout)

        self.geometry_params_group = QGroupBox("Geometry Parameters")
        self.geometry_params_layout = QFormLayout()
        self.geometry_params_group.setLayout(self.geometry_params_layout)
        geometry_group_layout.addWidget(self.geometry_params_group)
        
        # Disable fine-tuning fields by default
        self.palace_label.setVisible(False)
        self.palace_edit.setVisible(False)
        self.ft_iter_label.setVisible(False)
        self.ft_iter_spin.setVisible(False)
        self.geometry_group.setVisible(False)
        
        # If fine-tuning is toggled, show palace command and ORCA geometry fields
        self.finetune_cb.toggled.connect(self.on_finetune_toggled)
        self.geometry_source_combo.currentIndexChanged.connect(self.update_geometry_source)
        self.geometry_preset_combo.currentIndexChanged.connect(self.update_geometry_options)
        self.geometry_class_combo.currentIndexChanged.connect(self.update_geometry_options)

        self.config_scroll_area = QScrollArea()
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.config_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.config_scroll_widget = QWidget()
        self.config_scroll_layout = QVBoxLayout(self.config_scroll_widget)
        self.config_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.config_scroll_layout.addLayout(self.config_form_layout)
        self.config_scroll_layout.addWidget(self.geometry_group)
        self.config_scroll_layout.addStretch()
        self.config_scroll_area.setWidget(self.config_scroll_widget)
        
        # 2. Optimization Parameters
        param_group = QGroupBox("Optimization Parameters")
        param_layout = QVBoxLayout(param_group)
        self.param_table = QTableWidget(0, 6)
        self.param_table.setHorizontalHeaderLabels(["Name", "Type", "Min", "Current", "Max", "Unit"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.param_table.customContextMenuRequested.connect(self.param_context_menu)
        self.param_table.doubleClicked.connect(lambda idx: self.edit_param(idx.row()))
        param_layout.addWidget(self.param_table)
        
        h_param_btns = QHBoxLayout()
        add_manual_btn = QPushButton("Add Manual")
        add_manual_btn.clicked.connect(self.add_manual_param)
        add_onnx_btn = QPushButton("Add from ONNX")
        add_onnx_btn.clicked.connect(self.add_onnx_param)
        add_net_btn = QPushButton("Add from Netlist")
        add_net_btn.clicked.connect(self.add_netlist_param)
        h_param_btns.addWidget(add_manual_btn)
        h_param_btns.addWidget(add_onnx_btn)
        h_param_btns.addWidget(add_net_btn)
        param_layout.addLayout(h_param_btns)
        
        # 3. Design Goals
        goal_group = QGroupBox("Design Goals")
        goal_layout = QVBoxLayout(goal_group)
        self.goal_list = QListWidget()
        self.goal_list.setFixedHeight(100)
        self.goal_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.goal_list.customContextMenuRequested.connect(self.goal_context_menu)
        self.goal_list.doubleClicked.connect(lambda idx: self.edit_goal(self.goal_list.item(idx.row())))
        goal_layout.addWidget(self.goal_list)
        add_goal_btn = QPushButton("Add Goal")
        add_goal_btn.clicked.connect(self.add_design_goal)
        goal_layout.addWidget(add_goal_btn)
        
        config_layout.addWidget(self.config_scroll_area, stretch=1)
        config_layout.addWidget(param_group)
        config_layout.addWidget(goal_group)
        
        self.action_btn = QPushButton("START OPTIMIZATION")
        self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        self.action_btn.clicked.connect(self.on_action_clicked)
        
        self.stop_btn = QPushButton("⬛")  # Square stop symbol
        self.stop_btn.setToolTip("Stop Optimization")
        self.stop_btn.setFixedSize(40, 40) # Small square
        self.stop_btn.setStyleSheet("font-weight: bold; font-size: 20px; background-color: #F44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.action_btn)
        btn_layout.addWidget(self.stop_btn)
        config_layout.addLayout(btn_layout)
        
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.elapsed_label = QLabel("Time: 0.0s")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.elapsed_label)
        config_layout.addLayout(progress_layout)

        # Right Panel: Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # 1. Plots Area (Horizontal split)
        plot_controls = QHBoxLayout()
        self.show_goals_cb = QCheckBox("Show Goals")
        self.show_goals_cb.setChecked(True)
        self.show_goals_cb.stateChanged.connect(self.refresh_overlays)
        
        self.plot_prev_cb = QCheckBox("Plot Previous Result")
        self.plot_prev_cb.setChecked(True) # Default to true as before
        
        self.zoom_btn = QPushButton("Zoom to Goal Frequency Range")
        self.zoom_btn.clicked.connect(self.zoom_to_range)
        
        plot_controls.addWidget(self.show_goals_cb)
        plot_controls.addWidget(self.plot_prev_cb)
        plot_controls.addWidget(self.zoom_btn)
        plot_controls.addStretch()
        
        viz_layout.addLayout(plot_controls)

        plots_layout = QHBoxLayout()
        
        # S-Param Plot
        self.s_param_plot = pg.PlotWidget(title="S-Parameters (dB)")
        self.s_param_plot.addLegend()
        self.s_param_plot.setLabel('left', 'Magnitude', units='dB')
        self.s_param_plot.setLabel('bottom', 'Frequency', units='Hz')
        plots_layout.addWidget(self.s_param_plot)
        
        # Loss Plot
        self.loss_plot = pg.PlotWidget(title="Goal Losses")
        self.loss_plot.addLegend()
        self.loss_plot.setLabel('left', 'Loss')
        plots_layout.addWidget(self.loss_plot)
        
        viz_layout.addLayout(plots_layout, stretch=2)
        
        # 2. Tables Area (Horizontal split)
        tables_layout = QHBoxLayout()

        # Goal Status Table (Current Goal Metrics)
        goal_group_viz = QGroupBox("Goal Status")
        goal_viz_layout = QVBoxLayout(goal_group_viz)
        self.goal_table = QTableWidget(0, 3)
        self.goal_table.setHorizontalHeaderLabels(["Goal Param", "Target", "Current Value"])
        self.goal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        goal_viz_layout.addWidget(self.goal_table)
        tables_layout.addWidget(goal_group_viz)

        viz_layout.addLayout(tables_layout, stretch=1)
        
        # Splitter setup
        # For simplicity using specific ratios in layout
        main_layout.addWidget(config_group, 1)
        main_layout.addWidget(viz_group, 2)
        
        # Data storage
        self.opt_params: List[OptimizationProperty] = []
        self.goals: List[DesignGoal] = []
        self.worker = None
        self.loss_history = {} # key: goal index, value: list of losses
        self.overlay_items = []
        self.orca_preset_classes: Dict[str, type] = {}
        self.custom_geometry_classes: Dict[str, type] = {}
        self.geometry_param_widgets: Dict[str, QWidget] = {}
        self.geometry_param_specs: Dict[str, inspect.Parameter] = {}
        
        self.update_simulator_options()
        self.reload_orca_preset_geometries(show_errors=False)
        self.update_geometry_source()

    def refresh_overlays(self, state):
        self.draw_overlays()

    def zoom_to_range(self):
        # Range = min and max frequency from the goals if specified
        min_f = float('inf')
        max_f = float('-inf')
        found_any = False

        for goal in self.goals:
            try:
                if not goal.frequency_range:
                    continue
                
                freq_str = goal.frequency_range.lower().replace("ghz", "").strip()
                if "-" in freq_str:
                    parts = freq_str.split("-")
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        f_start = float(parts[0])
                        f_end = float(parts[1])
                        # Convert to Hz
                        f_start_hz = f_start * 1e9
                        f_end_hz = f_end * 1e9
                        
                        if f_start_hz < min_f: min_f = f_start_hz
                        if f_end_hz > max_f: max_f = f_end_hz
                        found_any = True
            except:
                pass

        if found_any:
            # Add some padding (e.g. 5%)
            span = max_f - min_f
            if span > 0:
                self.s_param_plot.setXRange(min_f - span*0.05, max_f + span*0.05)
            else:
                self.s_param_plot.setXRange(min_f * 0.95, max_f * 1.05)
        else:
            QMessageBox.information(self, "Info", "No valid frequency ranges found in goals.")

    def draw_overlays(self):
        # Remove previously tracked overlay items
        for item in self.overlay_items:
            try:
                # If item is still in scene, remove it
                if item.scene() is not None:
                    self.s_param_plot.removeItem(item)
            except:
                pass
        self.overlay_items = []
        
        if not self.show_goals_cb.isChecked():
            return

        # Draw Goal Lines
        for goal in self.goals:
            try:
                if not goal.frequency_range:
                    continue
                
                freq_str = goal.frequency_range.lower().replace("ghz", "").strip()
                if "-" not in freq_str:
                    continue
                
                parts = freq_str.split("-")
                if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                    continue

                f_start = float(parts[0])
                f_end = float(parts[1])
                f_start_hz = f_start * 1e9
                f_end_hz = f_end * 1e9
                
                p_name = goal.parameter.value
                if p_name.startswith("S") and "_dB" in p_name:
                    if goal.min_value is not None:
                        # Draw line segment
                        min_y = goal.min_value
                        line = pg.PlotCurveItem(
                            x=[f_start_hz, f_end_hz], 
                            y=[min_y, min_y],
                            pen=pg.mkPen('g', width=3, style=Qt.PenStyle.SolidLine)
                        )
                        self.s_param_plot.addItem(line)
                        self.overlay_items.append(line)
                        
                        # Add label for min
                        text = pg.TextItem(f"{p_name} > {goal.min_value}", color='g', anchor=(0, 1))
                        text.setPos(f_start_hz, min_y)
                        self.s_param_plot.addItem(text)
                        self.overlay_items.append(text)

                    if goal.max_value is not None:
                        # Draw line segment
                        max_y = goal.max_value
                        line = pg.PlotCurveItem(
                            x=[f_start_hz, f_end_hz], 
                            y=[max_y, max_y],
                            pen=pg.mkPen('r', width=3, style=Qt.PenStyle.SolidLine)
                        )
                        self.s_param_plot.addItem(line)
                        self.overlay_items.append(line)
                        
                        # Add label for max
                        text = pg.TextItem(f"{p_name} < {goal.max_value}", color='r', anchor=(0, 0))
                        text.setPos(f_start_hz, max_y)
                        self.s_param_plot.addItem(text)
                        self.overlay_items.append(text)


            except Exception as e:
                print(f"Error drawing overlay for goal: {e}")


    def update_simulator_options(self):
        # Clear existing dynamic options using takeRow + deleteLater
        for _ in range(self.sim_options_count):
            result = self.config_form_layout.takeRow(self.sim_options_insert_pos)
            if result.labelItem: 
                w = result.labelItem.widget()
                if w: w.deleteLater()
            if result.fieldItem:
                w = result.fieldItem.widget()
                if w: w.deleteLater()
        
        self.sim_options_count = 0
        self.simulator_widgets = {}
        
        # Get selected simulator class
        sim_cls = self.simulator_combo.currentData()
        if not sim_cls:
            return
        
        # Inspect __init__ to find arguments
        try:
            sig = inspect.signature(sim_cls.__init__)
        except ValueError:
            return

        idx = self.sim_options_insert_pos
        count = 0

        for name, param in sig.parameters.items():
            if name == 'self': continue
            if name == 'args' or name == 'kwargs': continue
            
            annotation = param.annotation
            default = param.default
            
            widget = None
            
            # Create appropriate widget based on type annotation or default value type
            if annotation == bool or isinstance(default, bool):
                widget = QCheckBox()
                if isinstance(default, bool):
                    widget.setChecked(default)
                
            elif annotation == int or isinstance(default, int):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000)
                if isinstance(default, int):
                    widget.setValue(default)
                    
            elif annotation == float or isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                if isinstance(default, float):
                    widget.setValue(default)
                    
            else:
                # Default to line edit for strings or unknown types
                widget = QLineEdit()
                if default != inspect.Parameter.empty:
                    widget.setText(str(default))
            
            # Improve label readability
            label_text = name.replace("_", " ").title()
            self.config_form_layout.insertRow(idx, label_text + ":", widget)
            self.simulator_widgets[name] = widget
            
            idx += 1
            count += 1
            
        self.sim_options_count = count

    def on_finetune_toggled(self, checked):
        self.palace_label.setVisible(checked)
        self.palace_edit.setVisible(checked)
        self.ft_iter_label.setVisible(checked)
        self.ft_iter_spin.setVisible(checked)
        self.geometry_group.setVisible(checked)

        if checked and self.geometry_source_combo.currentData() == "preset" and self.geometry_preset_combo.count() == 0:
            self.reload_orca_preset_geometries()

    def browse_geometry_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Geometry File", "", "Python Files (*.py)")
        if not fname:
            return

        self.geometry_file_edit.setText(fname)
        if self.load_custom_geometry_classes(fname):
            self.update_geometry_options()

    def reload_orca_preset_geometries(self, show_errors: bool = True):
        current_label = self.geometry_preset_combo.currentText()
        self.geometry_preset_combo.blockSignals(True)
        self.geometry_preset_combo.clear()
        self.orca_preset_classes = {}

        try:
            BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry
            presets = importlib.import_module("orca.geometry.presets")

            discovered_classes = []
            for _, module_name, _ in pkgutil.iter_modules(presets.__path__):
                module = importlib.import_module(f"orca.geometry.presets.{module_name}")
                for class_name, cls in inspect.getmembers(module, inspect.isclass):
                    if cls is BaseGeometry or not issubclass(cls, BaseGeometry):
                        continue
                    if cls.__module__ != module.__name__:
                        continue

                    label = class_name
                    if label in self.orca_preset_classes:
                        label = f"{class_name} ({module_name})"
                    discovered_classes.append((label, cls))

            for label, cls in sorted(discovered_classes, key=lambda item: item[0].lower()):
                self.orca_preset_classes[label] = cls
                self.geometry_preset_combo.addItem(label, cls)

            if current_label:
                index = self.geometry_preset_combo.findText(current_label)
                if index >= 0:
                    self.geometry_preset_combo.setCurrentIndex(index)
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "ORCA Geometry", f"Failed to load ORCA preset geometries:\n{exc}")
        finally:
            self.geometry_preset_combo.blockSignals(False)

    def load_custom_geometry_classes(self, file_path: str, show_errors: bool = True) -> bool:
        self.geometry_class_combo.blockSignals(True)
        self.geometry_class_combo.clear()
        self.custom_geometry_classes = {}

        if not file_path:
            self.geometry_class_combo.blockSignals(False)
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

            for class_name, cls in sorted(classes, key=lambda item: item[0].lower()):
                self.custom_geometry_classes[class_name] = cls
                self.geometry_class_combo.addItem(class_name, cls)

            return True
        except Exception as exc:
            self.clear_geometry_parameter_widgets()
            if show_errors:
                QMessageBox.critical(self, "ORCA Geometry", f"Failed to load custom geometry:\n{exc}")
            return False
        finally:
            self.geometry_class_combo.blockSignals(False)

    def update_geometry_source(self):
        source = self.geometry_source_combo.currentData()
        use_preset = source == "preset"

        self.geometry_preset_label.setVisible(use_preset)
        self.geometry_preset_combo.setVisible(use_preset)
        self.geometry_file_label.setVisible(not use_preset)
        self.geometry_file_widget.setVisible(not use_preset)
        self.geometry_class_label.setVisible(not use_preset)
        self.geometry_class_combo.setVisible(not use_preset)

        if use_preset and self.geometry_preset_combo.count() == 0:
            self.reload_orca_preset_geometries()
        elif not use_preset and self.geometry_class_combo.count() == 0 and self.geometry_file_edit.text().strip():
            self.load_custom_geometry_classes(self.geometry_file_edit.text().strip(), show_errors=False)

        self.update_geometry_options()

    def clear_geometry_parameter_widgets(self):
        while self.geometry_params_layout.rowCount() > 0:
            result = self.geometry_params_layout.takeRow(0)
            if result.labelItem:
                label_widget = result.labelItem.widget()
                if label_widget:
                    label_widget.deleteLater()
            if result.fieldItem:
                field_widget = result.fieldItem.widget()
                if field_widget:
                    field_widget.deleteLater()

        self.geometry_param_widgets = {}
        self.geometry_param_specs = {}

    def update_geometry_options(self):
        self.clear_geometry_parameter_widgets()

        geometry_cls = self.get_selected_geometry_class()
        if geometry_cls is None:
            return

        try:
            sig = inspect.signature(geometry_cls.__init__)
        except ValueError:
            return

        for name, param in sig.parameters.items():
            if name == "self" or name in {"args", "kwargs"}:
                continue

            widget = self.create_parameter_widget(param)
            label_text = name.replace("_", " ").title()
            self.geometry_params_layout.addRow(f"{label_text}:", widget)
            self.geometry_param_widgets[name] = widget
            self.geometry_param_specs[name] = param

    def get_selected_geometry_class(self) -> Optional[type]:
        if self.geometry_source_combo.currentData() == "preset":
            return self.geometry_preset_combo.currentData()
        return self.geometry_class_combo.currentData()

    def normalize_parameter_type(self, annotation: Any) -> Optional[type]:
        if annotation is inspect.Parameter.empty:
            return None

        origin = get_origin(annotation)
        if origin is None:
            return annotation if annotation in {bool, int, float, str} else None

        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return self.normalize_parameter_type(args[0])

        return None

    def create_parameter_widget(self, param: inspect.Parameter) -> QWidget:
        annotation = self.normalize_parameter_type(param.annotation)
        default = param.default

        if annotation == bool or isinstance(default, bool):
            widget = QCheckBox()
            if isinstance(default, bool):
                widget.setChecked(default)
            return widget

        if annotation == int or (isinstance(default, int) and not isinstance(default, bool)):
            widget = QSpinBox()
            widget.setRange(-1000000, 1000000)
            if isinstance(default, int) and not isinstance(default, bool):
                widget.setValue(default)
            return widget

        if annotation == float or isinstance(default, float):
            widget = QDoubleSpinBox()
            widget.setRange(-1e9, 1e9)
            widget.setDecimals(6)
            if isinstance(default, float):
                widget.setValue(default)
            return widget

        widget = QLineEdit()
        if isinstance(default, str):
            widget.setText(default)
        elif default != inspect.Parameter.empty:
            widget.setPlaceholderText(f"Default: {default}")
            widget.setToolTip(f"Leave empty to use default: {default}")
        return widget

    def collect_parameter_value(self, widget: QWidget, param: inspect.Parameter):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()

        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()

        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            if not text and param.default != inspect.Parameter.empty:
                return inspect.Parameter.empty

            expected_type = self.normalize_parameter_type(param.annotation)
            if expected_type == str:
                return text
            if expected_type == int:
                return int(text)
            if expected_type == float:
                return float(text)

            if not text:
                return text

            try:
                return ast.literal_eval(text)
            except Exception:
                return text

        raise TypeError(f"Unsupported parameter widget type: {type(widget)}")

    def create_orca_geometry(self):
        if self.geometry_source_combo.currentData() == "custom":
            file_path = self.geometry_file_edit.text().strip()
            if not file_path:
                raise ValueError("Please select a custom geometry Python file.")
            self.load_custom_geometry_classes(file_path)

        geometry_cls = self.get_selected_geometry_class()
        if geometry_cls is None:
            raise ValueError("Please select an ORCA geometry class.")

        geometry_kwargs = {}
        for name, widget in self.geometry_param_widgets.items():
            value = self.collect_parameter_value(widget, self.geometry_param_specs[name])
            if value is not inspect.Parameter.empty:
                geometry_kwargs[name] = value

        return geometry_cls(**geometry_kwargs)

    def browse_file(self, line_edit, filter):
        fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter)
        if fname:
            line_edit.setText(fname)

    def add_manual_param(self):
        dlg = OptimizationParamDialog(parent=self)
        if dlg.exec():
            self._add_param_to_table(dlg.get_data())

    def param_context_menu(self, pos):
        row = self.param_table.rowAt(pos.y())
        if row >= 0:
            menu = QMenu()
            edit_action = menu.addAction("Edit")
            delete_action = menu.addAction("Delete")
            action = menu.exec(self.param_table.mapToGlobal(pos))
            if action == edit_action:
                self.edit_param(row)
            elif action == delete_action:
                self.delete_param(row)

    def edit_param(self, row):
        if row < 0 or row >= len(self.opt_params): return
        param = self.opt_params[row]
        dlg = OptimizationParamDialog(parent=self, param=param)
        if dlg.exec():
            new_param = dlg.get_data()
            self.opt_params[row] = new_param
            self._update_param_row(row, new_param)

    def delete_param(self, row):
        if row < 0 or row >= len(self.opt_params): return
        del self.opt_params[row]
        self.param_table.removeRow(row)

    def _update_param_row(self, row, param):
        self.param_table.setItem(row, 0, QTableWidgetItem(param.name))
        
        type_item = QTableWidgetItem(param.type.value)
        type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.param_table.setItem(row, 1, type_item)
        
        self.param_table.setItem(row, 2, QTableWidgetItem(str(param.min_value)))
        
        # Preserve current val if possible, otherwise N/A
        if not self.param_table.item(row, 3):
             curr_item = QTableWidgetItem("N/A")
             curr_item.setFlags(curr_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
             self.param_table.setItem(row, 3, curr_item)
        
        self.param_table.setItem(row, 4, QTableWidgetItem(str(param.max_value)))
        self.param_table.setItem(row, 5, QTableWidgetItem(str(param.unit or "")))

    def add_onnx_param(self):
        path = self.onnx_edit.text()
        if not path:
             QMessageBox.warning(self, "Error", "Select ONNX file first")
             return
        try:
            sess = onnxruntime.InferenceSession(path)
            # Only support float inputs which are typically the params
            all_inputs = [clean_name(i.name) for i in sess.get_inputs()]
            
            # Get metadata for min/max values
            metadata = sess.get_modelmeta().custom_metadata_map
            
            # Filter out already added parameters
            current_param_names = {p.name for p in self.opt_params}
            available_inputs = [name for name in all_inputs if name not in current_param_names]
            
            if not available_inputs:
                QMessageBox.information(self, "Info", "All ONNX parameters have already been added.")
                return

            dlg = OptimizationParamDialog("ONNX", available_inputs, self, metadata=metadata)
            if dlg.exec():
                self._add_param_to_table(dlg.get_data())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def add_netlist_param(self):
        path = self.netlist_edit.text()
        if not path:
             QMessageBox.warning(self, "Error", "Select Netlist file first")
             return
        try:
            # We use XyceNetlistParser to identify valid targets
            # Since parser needs instance but we just want to scan, we instantiate it
            parser = XyceNetlistParser().from_file(path)
            # Find R, C, L, V, I elements
            prospects = []
            for elem in parser.list_elements(["R", "C", "L", "V", "I"]):
                 prospects.append(elem.name)
            
            # Filter out already added parameters
            current_param_names = {p.name for p in self.opt_params}
            available_prospects = [name for name in prospects if name not in current_param_names]

            if not available_prospects:
                QMessageBox.information(self, "Info", "All valid netlist parameters have already been added.")
                return
            
            dlg = OptimizationParamDialog("NETLIST", available_prospects, self)
            if dlg.exec():
                self._add_param_to_table(dlg.get_data())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _add_param_to_table(self, param: OptimizationProperty):
        # Check duplicate
        for p in self.opt_params:
            if p.name == param.name:
                QMessageBox.warning(self, "Duplicate", f"Parameter {p.name} already exists")
                # Update existing?
                return

        self.opt_params.append(param)
        row = self.param_table.rowCount()
        self.param_table.insertRow(row)
        self.param_table.setItem(row, 0, QTableWidgetItem(param.name))
        
        # Type
        type_item = QTableWidgetItem(param.type.value)
        type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.param_table.setItem(row, 1, type_item)
        
        self.param_table.setItem(row, 2, QTableWidgetItem(str(param.min_value)))
        
        # Current Value (Not editable)
        curr_item = QTableWidgetItem("N/A")
        curr_item.setFlags(curr_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.param_table.setItem(row, 3, curr_item)
        
        self.param_table.setItem(row, 4, QTableWidgetItem(str(param.max_value)))
        self.param_table.setItem(row, 5, QTableWidgetItem(str(param.unit or "")))

    def add_design_goal(self):
        dlg = DesignGoalDialog(self)
        if dlg.exec():
            goal = dlg.get_data()
            self._add_goal_to_list(goal)

    def goal_context_menu(self, pos):
        item = self.goal_list.itemAt(pos)
        if item:
            menu = QMenu()
            edit_action = menu.addAction("Edit")
            delete_action = menu.addAction("Delete")
            action = menu.exec(self.goal_list.mapToGlobal(pos))
            if action == edit_action:
                self.edit_goal(item)
            elif action == delete_action:
                self.delete_goal(item)

    def edit_goal(self, item):
        row = self.goal_list.row(item)
        if row < 0 or row >= len(self.goals): return
        
        goal = self.goals[row]
        dlg = DesignGoalDialog(self, goal=goal)
        if dlg.exec():
            new_goal = dlg.get_data()
            self.goals[row] = new_goal
            self._update_goal_item(item, new_goal)

    def delete_goal(self, item):
        row = self.goal_list.row(item)
        if row < 0 or row >= len(self.goals): return
        
        del self.goals[row]
        if row in self.loss_history:
             del self.loss_history[row]
        
        # Re-index remaining loss history? 
        # Actually loss history indices will shift so this is tricky if run continuously. 
        # Ideally we rebuild. For now just clear.
        self.loss_history = {}
        
        self.goal_list.takeItem(row)

    def _add_goal_to_list(self, goal):
        self.goals.append(goal)
        label = self._goal_label(goal)
        self.goal_list.addItem(label)
        # Init loss history
        idx = len(self.goals) - 1
        self.loss_history[idx] = []

    def _update_goal_item(self, item, goal):
        item.setText(self._goal_label(goal))
    
    def _goal_label(self, goal):
        label = f"{goal.parameter.value} "
        if goal.min_value is not None: label += f"> {goal.min_value} "
        if goal.max_value is not None: label += f"< {goal.max_value}"
        if goal.frequency_range: label += f" @ {goal.frequency_range}"
        if goal.weight != 1.0: label += f" (w={goal.weight})"
        return label

    def on_action_clicked(self):
        # If not running, start
        if not self.worker or not self.worker.isRunning():
            self.start_optimization()
            return
        
        # If running, toggle pause
        if self.worker.paused:
            self.worker.resume()
            self.action_btn.setText("PAUSE")
            self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #FFA500; color: white;") # Orange
        else:
            self.worker.pause()
            self.action_btn.setText("RESUME")
            self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #2196F3; color: white;") # Blue

    def start_optimization(self):
        onnx = self.onnx_edit.text()
        netlist = self.netlist_edit.text()
        if not onnx or not netlist:
            QMessageBox.warning(self, "Missing Files", "Please select ONNX and Netlist files.")
            return

        orca_geometry = None
        if self.finetune_cb.isChecked():
            try:
                orca_geometry = self.create_orca_geometry()
            except Exception as exc:
                QMessageBox.critical(self, "ORCA Geometry", str(exc))
                return

        optimizer_cls = self.optimizer_combo.currentData()
        simulator_cls = self.simulator_combo.currentData()
        
        # Gather simulator kwargs
        sim_kwargs = {}
        for name, widget in self.simulator_widgets.items():
             if isinstance(widget, QCheckBox):
                 sim_kwargs[name] = widget.isChecked()
             elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                 sim_kwargs[name] = widget.value()
             elif isinstance(widget, QLineEdit):
                 sim_kwargs[name] = widget.text()
        
        cobra = COBRA(
            em_surrogate_model=onnx,
            optimizer=optimizer_cls(),# multi_objective=self.moo_cb.isChecked()),
            circuit_simulator=simulator_cls(**sim_kwargs),
            palace_fine_tuning_command=(self.palace_edit.text() or None) if self.finetune_cb.isChecked() else None,
            fine_tuning_iterations=self.ft_iter_spin.value()
        )
        
        # Change button to PAUSE (running state)
        self.action_btn.setText("PAUSE")
        self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #FFA500; color: white;")
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, self.max_iter_spin.value())
        self.progress_bar.setValue(0)
        
        # Clear plots
        self.s_param_plot.clear()
        self.loss_plot.clear()
        self.loss_history = {i: [] for i in range(len(self.goals))}
        self.overlay_items = [] # clear tracked items since plot.clear() removed them
        
        self.draw_overlays()
        
        self.worker = OptimizationWorker(
            cobra, netlist, self.goals, 
            self.opt_params, self.max_iter_spin.value(),
            orca_geometry
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.action_btn.setText("STOPPING...")
            self.action_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    @Slot(dict)
    def on_progress(self, context: dict):
        self.progress_bar.setValue(context.get("iteration", 0))
        
        elapsed = context.get("elapsed_time")
        if elapsed is None and "times" in context:
             elapsed = context["times"].get("total_time")
        
        if elapsed is not None:
            self.elapsed_label.setText(f"Time: {elapsed:.1f}s")
        
        # 1. Update Parameters Table (Current Values)
        net_params = context.get("netlist_parameters", {})
        model_params = context.get("model_parameters", {})
        
        # Combine maps for easier lookup
        current_values = {**net_params, **model_params}
        
        for row in range(self.param_table.rowCount()):
            name_item = self.param_table.item(row, 0)
            if not name_item: continue
            name = name_item.text()
            
            if name in current_values:
                val = current_values[name]
                display_val = f"{(round(val, 4)) if isinstance(val, (float, np.floating)) else str(val)}"
                item = self.param_table.item(row, 3)
                if item:
                    item.setText(display_val)

        # Update Goal Status Table
        ntwk_n = context.get("simulated_network")
        if ntwk_n:
            # Metrics are now pre-calculated in COBRA.run and stored in context
            metrics = context.get("electrical_parameters", {})
            
            self.goal_table.setRowCount(0)
            losses = context.get("penalties", [])

            # print(f"Updating Goal Table with metrics: {metrics} and losses: {losses}")
            
            for i, goal in enumerate(self.goals):
                p_name = goal.parameter.value
                # Get value from metrics
                # metrics contains arrays usually, we might want mean/min/max or just show range
                val_array = metrics.get(p_name)
                
                display_val = "N/A"
                if val_array is not None:
                     # Check if array
                    if isinstance(val_array, (list, tuple, np.ndarray)):
                        if len(val_array) > 0:
                            v_min = np.min(val_array)
                            v_max = np.max(val_array)
                            if np.isclose(v_min, v_max, atol=1e-6):
                                display_val = f"{v_min:.4f}"
                            else:
                                display_val = f"[{v_min:.4f}, {v_max:.4f}]"
                    else:
                        # Scalar
                        display_val = f"{val_array:.4f}"

                target_str = ""
                if goal.min_value is not None: target_str += f">{goal.min_value} "
                if goal.max_value is not None: target_str += f"<{goal.max_value}"
                
                loss_val = losses[i] if i < len(losses) else 0.0

                r = self.goal_table.rowCount()
                self.goal_table.insertRow(r)
                
                name_item = QTableWidgetItem(p_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.goal_table.setItem(r, 0, name_item)
                
                target_item = QTableWidgetItem(target_str)
                target_item.setFlags(target_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.goal_table.setItem(r, 1, target_item)
                
                val_item = QTableWidgetItem(f"{display_val} (L={loss_val:.2f})")
                val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.goal_table.setItem(r, 2, val_item)
                
                # Add Loss Column if needed or color code
                # Let's tint the background if loss > 0
                item = self.goal_table.item(r, 2)
                if item:
                    if loss_val > 1e-4:
                        item.setBackground(Qt.GlobalColor.red)
                    else:
                        item.setBackground(Qt.GlobalColor.green)


        self.s_param_plot.clear()
        
        # Plot n (Current)
        ntwk_n = context.get("simulated_network")
        ntwk_prev = context.get("prev_network")

        if ntwk_prev and self.plot_prev_cb.isChecked():
            freq_prev = ntwk_prev.f
            if ntwk_prev.nports >= 1:
                s11_n1 = ntwk_prev.s_db[:, 0, 0]
                self.s_param_plot.plot(freq_prev, s11_n1, pen=pg.mkPen((255, 100, 100, 100), width=2, style=Qt.PenStyle.DashLine), name="S11 (n-1)")
            if ntwk_prev.nports >= 2:
                s21_n1 = ntwk_prev.s_db[:, 1, 0]
                self.s_param_plot.plot(freq_prev, s21_n1, pen=pg.mkPen((100, 100, 255, 100), width=2, style=Qt.PenStyle.DashLine), name="S21 (n-1)")

        if ntwk_n:
            freq = ntwk_n.f
            if ntwk_n.nports >= 1:
                s11_db = ntwk_n.s_db[:, 0, 0]
                self.s_param_plot.plot(freq, s11_db, pen=pg.mkPen('r', width=3), name="S11 (n)")
            if ntwk_n.nports >= 2:
                s21_db = ntwk_n.s_db[:, 1, 0]
                self.s_param_plot.plot(freq, s21_db, pen=pg.mkPen('b', width=3), name="S21 (n)")

        # Redraw overlays on top
        self.overlay_items = [] # clear references as plot.clear() removed them
        self.draw_overlays()

        # 3. Update Loss Plot
        losses = context.get("penalties", [])
        if self.loss_history: # Ensure loss history is initialized
             for i, loss in enumerate(losses):
                if i in self.loss_history:
                    self.loss_history[i].append(loss)
             
             self.loss_plot.clear()
             for i, hist in self.loss_history.items():
                 if hist:
                    color = pg.intColor(i, hues=len(self.loss_history))
                    self.loss_plot.plot(hist, pen=pg.mkPen(color, width=2), name=f"Goal {i+1}")

    @Slot()
    def on_finished(self):
        self.action_btn.setEnabled(True)
        self.action_btn.setText("START OPTIMIZATION")
        self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        
        self.stop_btn.setEnabled(False)

        QMessageBox.information(self, "Done", "Optimization Finished!")

    @Slot(str)
    def on_error(self, msg):
        self.action_btn.setEnabled(True)
        self.action_btn.setText("START OPTIMIZATION")
        self.action_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(self, "Error", msg)

