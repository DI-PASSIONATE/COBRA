from typing import List, Optional
import numpy as np
import onnxruntime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
    QListWidget, QTableWidget, QTableWidgetItem, 
    QSpinBox, QGroupBox, QFormLayout,
    QHeaderView, QMessageBox, QProgressBar, QCheckBox, QMenu
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
        form_layout = QFormLayout()
        
        self.onnx_edit = QLineEdit()
        self.onnx_btn = QPushButton("Browse")
        self.onnx_btn.clicked.connect(lambda: self.browse_file(self.onnx_edit, "ONNX Files (*.onnx)"))
        h_onnx = QHBoxLayout()
        h_onnx.addWidget(self.onnx_edit)
        h_onnx.addWidget(self.onnx_btn)
        form_layout.addRow("ONNX Model:", h_onnx)
        
        self.netlist_edit = QLineEdit()
        self.netlist_btn = QPushButton("Browse")
        self.netlist_btn.clicked.connect(lambda: self.browse_file(self.netlist_edit, "Netlist Files (*.cir *.sp)"))
        h_net = QHBoxLayout()
        h_net.addWidget(self.netlist_edit)
        h_net.addWidget(self.netlist_btn)
        form_layout.addRow("Netlist:", h_net)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("OptunaOptimizer", OptunaOptimizer)
        form_layout.addRow("Optimizer:", self.optimizer_combo)
        
        self.simulator_combo = QComboBox()
        self.simulator_combo.addItem("XyceSimulator", XyceSimulator)
        form_layout.addRow("Simulator:", self.simulator_combo)
        
        self.freq_edit = QLineEdit("125-135ghz")
        self.freq_edit.setToolTip("Enter frequency range where the goals below should apply, e.g. '125-135ghz'")
        form_layout.addRow("Goal Freq Range:", self.freq_edit)
        
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 99999)
        self.max_iter_spin.setValue(500)
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        # self.moo_cb = QCheckBox("Multi-Objective Optimization")
        # form_layout.addRow("", self.moo_cb)

        #### OPTIONAL - Fine-tuning with palace ####
        self.finetune_cb = QCheckBox("Perform finetuning")
        self.finetune_cb.setToolTip("Runs a few iterations of palace simulations instead of the surrogate model at the end to ensure correct predictions")
        form_layout.addRow("", self.finetune_cb)

        self.palace_label = QLabel("Palace Command:")
        self.palace_edit = QLineEdit("palace")
        form_layout.addRow(self.palace_label, self.palace_edit)

        self.orca_edit = QLineEdit()
        self.orca_btn = QPushButton("Browse")
        self.orca_btn.clicked.connect(lambda: self.browse_file(self.orca_edit, "Geometry Files (*.xml *.json)")) # Assuming format
        h_orca = QHBoxLayout()
        h_orca.addWidget(self.orca_edit)
        h_orca.addWidget(self.orca_btn)
        form_layout.addRow("ORCA Geometry:", h_orca)
        
        # Disable fine-tuning fields by default
        self.palace_label.setVisible(False)
        self.palace_edit.setVisible(False)
        self.orca_edit.setVisible(False)
        self.orca_btn.setVisible(False)
        
        # If fine-tuning is toggled, show palace command and ORCA geometry fields
        self.finetune_cb.toggled.connect(self.palace_label.setVisible)
        self.finetune_cb.toggled.connect(self.palace_edit.setVisible)
        self.finetune_cb.toggled.connect(self.orca_edit.setVisible)
        self.finetune_cb.toggled.connect(self.orca_btn.setVisible)
        
        config_layout.addLayout(form_layout)
        
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
        
        config_layout.addWidget(param_group)
        config_layout.addWidget(goal_group)
        
        self.run_btn = QPushButton("START OPTIMIZATION")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        self.run_btn.clicked.connect(self.start_optimization)
        config_layout.addWidget(self.run_btn)
        
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

    def refresh_overlays(self, state):
        self.draw_overlays()

    def zoom_to_range(self):
        try:
            freq_str = self.freq_edit.text().lower().replace("ghz", "").strip()
            if "-" in freq_str:
                f_start, f_end = map(float, freq_str.split("-"))
                # Convert to Hz for plotting if needed
                f_start_hz = f_start * 1e9
                f_end_hz = f_end * 1e9
                self.s_param_plot.setXRange(f_start_hz, f_end_hz)
        except Exception as e:
            QMessageBox.warning(self, "Invalid Frequency Range", str(e))

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

        # Draw Frequency Range Lines
        try:
            freq_str = self.freq_edit.text().lower().replace("ghz", "").strip()
            if "-" in freq_str:
                f_start, f_end = map(float, freq_str.split("-"))
                f_start_hz = f_start * 1e9
                f_end_hz = f_end * 1e9
                
                v_line_start = pg.InfiniteLine(pos=f_start_hz, angle=90, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DashLine), label=f"{f_start} GHz")
                v_line_end = pg.InfiniteLine(pos=f_end_hz, angle=90, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DashLine), label=f"{f_end} GHz")
                self.s_param_plot.addItem(v_line_start)
                self.s_param_plot.addItem(v_line_end)
                self.overlay_items.extend([v_line_start, v_line_end])
        except Exception:
            pass # Ignore parsing errors here

        # Draw Goal Lines
        for goal in self.goals:
            p_name = goal.parameter.value
            if p_name.startswith("S") and "_dB" in p_name:
                if goal.min_value is not None:
                     line = pg.InfiniteLine(pos=goal.min_value, angle=0, pen=pg.mkPen('g', width=1, style=Qt.PenStyle.DashLine), label=f"{p_name} > {goal.min_value}")
                     self.s_param_plot.addItem(line)
                     self.overlay_items.append(line)
                if goal.max_value is not None:
                     line = pg.InfiniteLine(pos=goal.max_value, angle=0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine), label=f"{p_name} < {goal.max_value}")
                     self.s_param_plot.addItem(line)
                     self.overlay_items.append(line)

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
        return label

    def start_optimization(self):
        onnx = self.onnx_edit.text()
        netlist = self.netlist_edit.text()
        if not onnx or not netlist:
            QMessageBox.warning(self, "Missing Files", "Please select ONNX and Netlist files.")
            return

        # Toggle button state if already running
        if self.worker and self.worker.isRunning():
            self.stop_optimization()
            return

        optimizer_cls = self.optimizer_combo.currentData()
        simulator_cls = self.simulator_combo.currentData()
        
        cobra = COBRA(
            em_surrogate_model=onnx,
            optimizer=optimizer_cls(),# multi_objective=self.moo_cb.isChecked()),
            circuit_simulator=simulator_cls(),
            palace_fine_tuning_command=(self.palace_edit.text() or None) if self.finetune_cb.isChecked() else None
        )
        
        # Change button to STOP
        self.run_btn.setText("STOP OPTIMIZATION")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #F44336; color: white;")
        self.progress_bar.setRange(0, self.max_iter_spin.value())
        self.progress_bar.setValue(0)
        
        # Clear plots
        self.s_param_plot.clear()
        self.loss_plot.clear()
        self.loss_history = {i: [] for i in range(len(self.goals))}
        self.overlay_items = [] # clear tracked items since plot.clear() removed them
        
        self.draw_overlays()
        
        self.worker = OptimizationWorker(
            cobra, netlist, self.goals, self.freq_edit.text(), 
            self.opt_params, self.max_iter_spin.value(),
            self.orca_edit.text() or None
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.run_btn.setText("STOPPING...")
            self.run_btn.setEnabled(False)

    @Slot(dict)
    def on_progress(self, context: dict):
        self.progress_bar.setValue(context.get("iteration", 0))
        if "elapsed_time" in context:
            self.elapsed_label.setText(f"Time: {context['elapsed_time']:.1f}s")
        
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
        # We need to calculate current values for each goal
        # The context has 'simulated_network'
        ntwk_n = context.get("simulated_network")
        if ntwk_n:
            # Metrics are now pre-calculated in COBRA.run and stored in context
            metrics = context.get("electrical_parameters", {})
            
            self.goal_table.setRowCount(0)
            losses = context.get("current_losses", [])
            
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
        losses = context.get("current_losses", [])
        if self.loss_history: # Ensure loss history is initialized
             for i, loss in enumerate(losses):
                if i in self.loss_history:
                    self.loss_history[i].append(loss)
                    # Clear and replot full history? Alternatively optimize by just appending data if using setData
                    # But for now, simple clear/plot is fine for <1000 iter
                    pass
             
             self.loss_plot.clear()
             for i, hist in self.loss_history.items():
                 if hist:
                    color = pg.intColor(i, hues=len(self.loss_history))
                    self.loss_plot.plot(hist, pen=pg.mkPen(color, width=2), name=f"Goal {i+1}")

    @Slot()
    def on_finished(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("START OPTIMIZATION")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        QMessageBox.information(self, "Done", "Optimization Finished!")

    @Slot(str)
    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("START OPTIMIZATION")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        QMessageBox.critical(self, "Error", msg)

