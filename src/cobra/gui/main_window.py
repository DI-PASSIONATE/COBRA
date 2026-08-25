from typing import Dict, List, Optional, Tuple
import importlib
import importlib.util
import inspect
import os
import pkgutil
import re
import numpy as np
import onnxruntime
import gmsh

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog,
    QListWidget, QTableWidget, QTableWidgetItem,
    QSpinBox, QGroupBox, QFormLayout, QScrollArea,
    QHeaderView, QMessageBox, QProgressBar, QCheckBox, QMenu, QDoubleSpinBox, QInputDialog,
    QStackedWidget, QDialog, QTextBrowser, QDialogButtonBox
)
from PySide6.QtCore import Qt, Slot
import pyqtgraph as pg

# COBRA imports
from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim import hb_spectrum
from cobra.spice_sim.simulation_type import SimulationType
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from cobra.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from cobra.optimizers.design_goal import DesignParameter
from cobra.optimizers.design_goal_collection import get_available_parameters, make_gain_db, make_power_dbm

from .dialogs import DesignGoalDialog, OptimizationParamDialog
from .geometry_selector import GeometrySelectorWidget
from .hf_model_browser import HuggingFaceModelDialog
from .help_texts import TUTORIAL_HTML, tooltip
from .theme import apply_theme
from .worker import OptimizationWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        gmsh.initialize()
        
        # Initialize component ONNX selector dictionaries
        self.component_onnx_edits: Dict[str, QLineEdit] = {}
        self.component_onnx_btns: Dict[str, QPushButton] = {}
        self.component_hf_btns: Dict[str, QPushButton] = {}
        
        self.setWindowTitle("COBRA GUI")
        self.resize(1200, 800)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Global controls shown regardless of active panel
        global_controls_layout = QHBoxLayout()
        control_btn_width = 220
        control_btn_height = 48

        self.config_panel_btn = QPushButton("Configuration")
        self.config_panel_btn.setFixedSize(control_btn_width, control_btn_height)
        self.config_panel_btn.setProperty("tabButton", True)
        self.config_panel_btn.setToolTip(tooltip("config_panel_btn"))
        self.config_panel_btn.clicked.connect(lambda: self.set_active_panel("config"))
        self.viz_panel_btn = QPushButton("Visualization")
        self.viz_panel_btn.setFixedSize(control_btn_width, control_btn_height)
        self.viz_panel_btn.setProperty("tabButton", True)
        self.viz_panel_btn.setToolTip(tooltip("viz_panel_btn"))
        self.viz_panel_btn.clicked.connect(lambda: self.set_active_panel("viz"))
        global_controls_layout.addWidget(self.config_panel_btn)
        global_controls_layout.addWidget(self.viz_panel_btn)

        self.help_btn = QPushButton("?")
        self.help_btn.setFixedSize(control_btn_height, control_btn_height)
        self.help_btn.setToolTip(tooltip("help_btn"))
        self.help_btn.clicked.connect(self.show_tutorial)

        self.action_btn = QPushButton("START OPTIMIZATION")
        self.action_btn.setFixedSize(control_btn_width, control_btn_height)
        self.action_btn.setProperty("primaryAction", True)
        self.action_btn.setProperty("actionState", "start")
        self.action_btn.setToolTip(tooltip("action_btn"))
        self.action_btn.clicked.connect(self.on_action_clicked)

        self.stop_btn = QPushButton("⬛")  # Square stop symbol
        self.stop_btn.setToolTip(tooltip("stop_btn"))
        self.stop_btn.setFixedSize(control_btn_height, control_btn_height)
        self.stop_btn.setProperty("dangerAction", True)
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)

        global_controls_layout.addStretch()
        global_controls_layout.addWidget(self.action_btn)
        global_controls_layout.addWidget(self.stop_btn)
        global_controls_layout.addWidget(self.help_btn)

        root_layout.addLayout(global_controls_layout)

        global_progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Iteration 0/0 (0.0%)")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.elapsed_label = QLabel("Time: 0.0s")
        global_progress_layout.addWidget(self.progress_label)
        global_progress_layout.addWidget(self.progress_bar, stretch=1)
        global_progress_layout.addWidget(self.elapsed_label)

        root_layout.addLayout(global_progress_layout)

        self.statusBar()

        self.panel_stack = QStackedWidget()
        root_layout.addWidget(self.panel_stack, stretch=1)
        
        # Left Panel: Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # 1. File Selection
        self.config_form_layout = QFormLayout()
        
        self.netlist_edit = QLineEdit()
        self.netlist_btn = QPushButton("Browse")
        self.netlist_btn.setToolTip(tooltip("netlist_btn"))
        self.netlist_btn.clicked.connect(self.on_netlist_selected)
        self.netlist_label = QLabel("Netlist:")
        h_net = QHBoxLayout()
        h_net.addWidget(self.netlist_edit)
        h_net.addWidget(self.netlist_btn)
        self.config_form_layout.addRow(self.netlist_label, h_net)

        self.sim_type_label = QLabel("Simulation Type:")
        self.sim_type_value_label = QLabel()
        self.sim_type_value_label.setToolTip(
            "Primary analysis directive detected from the loaded netlist.\n"
            "If any design goal requires an .AC S-parameter simulation and the netlist\n"
            "does not already contain .AC, COBRA will add it automatically."
        )
        self.config_form_layout.addRow(self.sim_type_label, self.sim_type_value_label)
        self.sim_type_label.setVisible(False)
        self.sim_type_value_label.setVisible(False)

        # HB analysis point — the node whose spectrum is plotted and used by HB goals.
        self.hb_point_label = QLabel("HB Analysis Point:")
        self.hb_point_combo = QComboBox()
        self.hb_point_combo.setToolTip(tooltip("hb_point_combo"))
        self.hb_point_combo.currentIndexChanged.connect(self.on_hb_point_changed)
        self.config_form_layout.addRow(self.hb_point_label, self.hb_point_combo)
        self.hb_point_label.setVisible(False)
        self.hb_point_combo.setVisible(False)

        # Simulation Parameters — populated dynamically when a netlist is loaded.
        # Shows editable fields for the sweep directive (e.g. .AC points, start/stop freq).
        self.sim_params_container = QWidget()
        self.sim_params_layout = QVBoxLayout(self.sim_params_container)
        self.sim_params_layout.setContentsMargins(0, 0, 0, 0)
        self.sim_params_layout.setSpacing(6)
        self.sim_params_container.setVisible(False)
        self.config_form_layout.addRow("", self.sim_params_container)

        self.component_onnx_container = QWidget()
        self.component_onnx_layout = QVBoxLayout(self.component_onnx_container)
        self.component_onnx_layout.setContentsMargins(0, 0, 0, 0)
        self.component_onnx_layout.setSpacing(6)
        self.component_onnx_container.setVisible(False)
        self.config_form_layout.addRow("", self.component_onnx_container)
        
        # ---- Optimizer selection + dynamic per-optimizer options ----
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("OptunaOptimizer", OptunaOptimizer)
        self.config_form_layout.addRow("Optimizer:", self.optimizer_combo)

        # Track position for inserting dynamic optimizer options (CobraSetting-driven)
        self.opt_options_insert_pos = self.config_form_layout.rowCount()
        self.opt_options_count = 0
        self.optimizer_widgets: Dict[str, object] = {}
        self.optimizer_combo.currentIndexChanged.connect(self.update_optimizer_options)

        # ---- Simulator selection + dynamic per-simulator options ----
        self.simulator_combo = QComboBox()
        self.simulator_combo.addItem("XyceSimulator", XyceSimulator)
        self.config_form_layout.addRow("Simulator:", self.simulator_combo)
        
        # Track position for inserting dynamic simulator options
        self.sim_options_insert_pos = self.config_form_layout.rowCount()
        self.sim_options_count = 0
        self.simulator_combo.currentIndexChanged.connect(self.update_simulator_options)
        self.simulator_widgets = {}

        # ---- COBRA-level settings (tooltips sourced from COBRA._settings) ----
        _cobra_tips = {s.name: s.description for s in getattr(COBRA, "_settings", [])}
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 99999)
        self.max_iter_spin.setValue(500)
        self.max_iter_spin.setToolTip(_cobra_tips.get("max_iterations", ""))
        self.config_form_layout.addRow("Max Iterations:", self.max_iter_spin)

        # self.moo_cb = QCheckBox("Multi-Objective Optimization")
        # form_layout.addRow("", self.moo_cb)

        #### OPTIONAL - Fine-tuning with palace ####
        self.finetune_cb = QCheckBox("Perform finetuning")
        self.finetune_cb.setToolTip(tooltip("finetune_cb"))
        self.config_form_layout.addRow("", self.finetune_cb)

        self.palace_label = QLabel("Palace Command:")
        self.palace_edit = QLineEdit("palace")
        self.palace_edit.setToolTip(_cobra_tips.get("palace_fine_tuning_command", ""))
        self.config_form_layout.addRow(self.palace_label, self.palace_edit)

        self.ft_iter_label = QLabel("Finetuning Iterations:")
        self.ft_iter_spin = QSpinBox()
        self.ft_iter_spin.setRange(1, 100)
        self.ft_iter_spin.setValue(3)
        self.ft_iter_spin.setToolTip(_cobra_tips.get("fine_tuning_iterations", ""))
        self.config_form_layout.addRow(self.ft_iter_label, self.ft_iter_spin)

        self.ft_optimizer_label = QLabel("Finetuning Optimizer:")
        self.ft_optimizer_combo = QComboBox()
        self.ft_optimizer_combo.addItem("Reuse surrogate optimizer", "reuse")
        self.ft_optimizer_combo.addItem("Gradient descent", "gradient_descent")
        self.ft_optimizer_combo.setToolTip(tooltip("ft_optimizer_combo"))
        self.config_form_layout.addRow(self.ft_optimizer_label, self.ft_optimizer_combo)

        self.component_geometry_container = QWidget()
        self.component_geometry_layout = QVBoxLayout(self.component_geometry_container)
        self.component_geometry_layout.setContentsMargins(0, 0, 0, 0)
        self.component_geometry_layout.setSpacing(6)
        self.component_geometry_container.setVisible(False)
        self.component_geometry_selectors: Dict[str, GeometrySelectorWidget] = {}

        # Disable fine-tuning fields by default
        self.palace_label.setVisible(False)
        self.palace_edit.setVisible(False)
        self.ft_iter_label.setVisible(False)
        self.ft_iter_spin.setVisible(False)
        self.ft_optimizer_label.setVisible(False)
        self.ft_optimizer_combo.setVisible(False)

        # If fine-tuning is toggled, show palace command and per-component geometry selectors
        self.finetune_cb.toggled.connect(self.on_finetune_toggled)

        # Populate initial optimizer options
        self.update_optimizer_options()

        self.config_scroll_area = QScrollArea()
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.config_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.config_scroll_widget = QWidget()
        self.config_scroll_layout = QVBoxLayout(self.config_scroll_widget)
        self.config_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.config_scroll_layout.addLayout(self.config_form_layout)
        self.config_scroll_layout.addWidget(self.component_geometry_container)
        self.config_scroll_layout.addStretch()
        self.config_scroll_area.setWidget(self.config_scroll_widget)
        
        # 2. Optimization Parameters
        param_group = QGroupBox("Optimization Parameters")
        param_layout = QVBoxLayout(param_group)
        self.param_table = QTableWidget(0, 7)
        self.param_table.setHorizontalHeaderLabels(["Name", "Type", "Min", "Current", "Max", "Unit", "Linked To"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.param_table.customContextMenuRequested.connect(self.param_context_menu)
        self.param_table.doubleClicked.connect(lambda idx: self.edit_param(idx.row()))
        param_layout.addWidget(self.param_table)
        
        h_param_btns = QHBoxLayout()
        add_net_btn = QPushButton("Add from Netlist")
        add_net_btn.setToolTip(tooltip("add_net_btn"))
        add_net_btn.clicked.connect(self.add_netlist_param)
        h_param_btns.addWidget(add_net_btn)
        param_layout.addLayout(h_param_btns)
        
        # 3. Design Goals
        goal_group = QGroupBox("Design Goals")
        goal_layout = QVBoxLayout(goal_group)
        self.goal_list = QListWidget()
        self.goal_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.goal_list.customContextMenuRequested.connect(self.goal_context_menu)
        self.goal_list.doubleClicked.connect(lambda idx: self.edit_goal(self.goal_list.item(idx.row())))
        goal_layout.addWidget(self.goal_list)
        add_goal_btn = QPushButton("Add Goal")
        add_goal_btn.setToolTip(tooltip("add_goal_btn"))
        add_goal_btn.clicked.connect(self.add_design_goal)
        goal_layout.addWidget(add_goal_btn)
        
        config_layout.addWidget(self.config_scroll_area, stretch=1)
        
        # Right panel: Optimization Parameters on top, Design Goals on bottom
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(param_group, stretch=1)
        right_layout.addWidget(goal_group, stretch=1)

        # Left panel: Configuration
        config_panel_widget = QWidget()
        config_panel_layout = QHBoxLayout(config_panel_widget)
        config_panel_layout.setContentsMargins(0, 0, 0, 0)
        config_panel_layout.addWidget(config_group, stretch=3)
        config_panel_layout.addWidget(right_widget, stretch=2)
        
        # Visualization Panel
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # 1. Plots Area (Horizontal split)
        plot_controls = QHBoxLayout()
        self.plot_view_combo = QComboBox()
        self.plot_view_combo.addItem("S-Parameters", "sparam")
        self.plot_view_combo.addItem("HB Spectrum", "hb")
        self.plot_view_combo.setToolTip(tooltip("plot_view_combo"))
        self.plot_view_combo.currentIndexChanged.connect(self.on_plot_view_changed)

        self.hb_quantity_combo = QComboBox()
        self.hb_quantity_combo.addItem("Power (dBm)", "power")
        self.hb_quantity_combo.addItem("Gain (dB)", "gain")
        self.hb_quantity_combo.addItem("Voltage (dBV)", "voltage")
        self.hb_quantity_combo.addItem("Current (dBmA)", "current")
        self.hb_quantity_combo.setToolTip(tooltip("hb_quantity_combo"))
        self.hb_quantity_combo.currentIndexChanged.connect(self.on_hb_quantity_changed)

        self.hb_input_port_combo = QComboBox()
        self.hb_input_port_combo.setToolTip(tooltip("hb_input_port_combo"))
        self.hb_input_port_combo.currentIndexChanged.connect(lambda: self.update_hb_spectrum_plot())

        self.show_goals_cb = QCheckBox("Show Goals")
        self.show_goals_cb.setChecked(True)
        self.show_goals_cb.stateChanged.connect(self.refresh_overlays)
        
        self.plot_prev_cb = QCheckBox("Plot Previous Result")
        
        self.zoom_btn = QPushButton("Zoom to Goal Frequency Range")
        self.zoom_btn.setToolTip(tooltip("zoom_btn"))
        self.zoom_btn.clicked.connect(self.zoom_to_range)
        
        plot_controls.addWidget(QLabel("Plot:"))
        plot_controls.addWidget(self.plot_view_combo)
        plot_controls.addWidget(self.hb_quantity_combo)
        plot_controls.addWidget(self.hb_input_port_combo)
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
        self.s_param_plot.setBackground('w')

        # HB Spectrum Plot — shares the left slot with the S-parameter plot
        self.hb_spectrum_plot = pg.PlotWidget(title="HB Spectrum")
        self.hb_spectrum_plot.addLegend()
        self.hb_spectrum_plot.setLabel('left', 'Power', units='dBm')
        self.hb_spectrum_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.hb_spectrum_plot.setBackground('w')
        self.hb_spectrum_plot.scene().sigMouseClicked.connect(self.on_hb_spectrum_clicked)

        self.left_plot_stack = QStackedWidget()
        self.left_plot_stack.addWidget(self.s_param_plot)
        self.left_plot_stack.addWidget(self.hb_spectrum_plot)
        plots_layout.addWidget(self.left_plot_stack)
        
        # Loss Plot
        self.loss_plot = pg.PlotWidget(title="Goal Losses")
        self.loss_plot.addLegend()
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setBackground('w')
        plots_layout.addWidget(self.loss_plot)

        self._style_plot_for_light_background(self.s_param_plot)
        self._style_plot_for_light_background(self.hb_spectrum_plot)
        self._style_plot_for_light_background(self.loss_plot)
        
        viz_layout.addLayout(plots_layout, stretch=2)
        
        # 2. Tables Area (Horizontal split)
        tables_layout = QHBoxLayout()

        # Current Parameter Table (read-only runtime values)
        current_params_group = QGroupBox("Current Parameters")
        current_params_layout = QVBoxLayout(current_params_group)
        self.current_param_table = QTableWidget(0, 2)
        self.current_param_table.setHorizontalHeaderLabels(["Name", "Current Value"])
        self.current_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        current_params_layout.addWidget(self.current_param_table)
        tables_layout.addWidget(current_params_group)

        # Goal Status Table (Current Goal Metrics)
        goal_group_viz = QGroupBox("Goal Status")
        goal_viz_layout = QVBoxLayout(goal_group_viz)
        self.goal_table = QTableWidget(0, 3)
        self.goal_table.setHorizontalHeaderLabels(["Goal Param", "Target", "Current [Min, Max] (Penalty)"])
        self.goal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        goal_viz_layout.addWidget(self.goal_table)
        tables_layout.addWidget(goal_group_viz)

        viz_layout.addLayout(tables_layout, stretch=1)
        
        self.panel_stack.addWidget(config_panel_widget)
        self.panel_stack.addWidget(viz_group)
        
        # Data storage
        self.opt_params: List[OptimizationProperty] = []
        self.goals: List[DesignGoal] = []
        self.worker = None
        self.loss_history = {} # key: goal index, value: list of losses
        self.overlay_items = []
        self.fine_tuning_active = False
        self.fine_tuning_notification_shown = False
        self._available_parameters: "List[DesignParameter]" = []  # populated from netlist on load
        self._sim_param_edits: Dict[str, QLineEdit] = {}  # key: "DIRECTIVE:param_name" or ".OPTIONS:<cat>:<param>"
        self._parsed_directives: list = []  # last directives from netlist parse
        self._parsed_options: Dict[str, Dict[str, str]] = {}  # category → {param: value}
        self._port_sources: Dict[str, Dict] = {}  # P-element name → SIN/AC source info
        self._hb_probe_nodes: List[str] = []  # nodes with both a V() and an I(V) HB probe
        self._required_sim_types: "set[SimulationType]" = set()
        self._hb_dataframes: Dict[str, object] = {}
        self._hb_spectrum_data: Optional[tuple] = None
        self._hb_markers: list = []
        self._num_ports: int = 0
        self._netlist_sim_type: "SimulationType | None" = None
        self._simulator_cls = self.simulator_combo.currentData() or XyceSimulator
        self.simulator_combo.currentIndexChanged.connect(
            lambda: setattr(self, "_simulator_cls", self.simulator_combo.currentData() or XyceSimulator)
        )

        apply_theme(self)
        self._set_action_button_state("start")
        self.update_simulator_options()
        self._update_plot_view_availability()
        self.set_active_panel("config")

    def _refresh_widget_style(self, widget: QWidget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_tab_state(self, button: QPushButton, active: bool):
        button.setProperty("tabActive", active)
        button.setEnabled(not active)
        self._refresh_widget_style(button)

    def _set_action_button_state(self, state: str, enabled: bool = True):
        label_map = {
            "start": "START OPTIMIZATION",
            "pause": "PAUSE",
            "resume": "RESUME",
            "stopping": "STOPPING...",
        }
        self.action_btn.setText(label_map.get(state, state))
        self.action_btn.setEnabled(enabled)
        self.action_btn.setProperty("actionState", state)
        self._refresh_widget_style(self.action_btn)

    def set_active_panel(self, panel: str):
        if panel == "viz":
            self.panel_stack.setCurrentIndex(1)
            self._set_tab_state(self.config_panel_btn, False)
            self._set_tab_state(self.viz_panel_btn, True)
        else:
            self.panel_stack.setCurrentIndex(0)
            self._set_tab_state(self.config_panel_btn, True)
            self._set_tab_state(self.viz_panel_btn, False)

    def show_tutorial(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("COBRA Tutorial")
        dialog.resize(720, 560)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setHtml(TUTORIAL_HTML)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

    def _update_progress_display(self, iteration: int, max_iterations: int):
        max_iterations = max(1, int(max_iterations))
        iteration = max(0, min(int(iteration), max_iterations))
        percentage = (iteration / max_iterations) * 100.0
        self.progress_bar.setRange(0, max_iterations)
        self.progress_bar.setValue(iteration)
        self.progress_label.setText(f"Iteration {iteration}/{max_iterations} ({percentage:.1f}%)")

    def _add_current_param_row(self, name, value):
        if isinstance(value, (float, np.floating)):
            display_val = f"{round(float(value), 4)}"
        else:
            display_val = str(value)

        row = self.current_param_table.rowCount()
        self.current_param_table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.current_param_table.setItem(row, 0, name_item)
        val_item = QTableWidgetItem(display_val)
        val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.current_param_table.setItem(row, 1, val_item)

    def _style_plot_for_light_background(self, plot_widget: pg.PlotWidget):
        axis_pen = pg.mkPen('k')
        for axis_name in ('left', 'bottom', 'top', 'right'):
            axis = plot_widget.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)

    def _update_finetuning_display(self, iteration: int, total_iterations: int):
        total_iterations = max(1, int(total_iterations))
        iteration = max(0, min(int(iteration), total_iterations))
        percentage = (iteration / total_iterations) * 100.0
        self.progress_bar.setRange(0, total_iterations)
        self.progress_bar.setValue(iteration)
        self.progress_label.setText(f"Finetuning {iteration}/{total_iterations} ({percentage:.1f}%)")

    def _goal_sparam_specs(self) -> List[tuple[str, int, int]]:
        # Parse goal parameter names like S11 or S21_dB into (label, row_idx, col_idx).
        specs: List[tuple[str, int, int]] = []
        seen = set()
        for goal in self.goals:
            p_name = goal.parameter_name
            match = re.match(r"^S\s*(?:\(\s*(\d)\s*,\s*(\d)\s*\)|(\d)(\d))(?:\b|_)", p_name, re.IGNORECASE)
            if not match:
                continue

            row_token = match.group(1) or match.group(3)
            col_token = match.group(2) or match.group(4)
            if row_token is None or col_token is None:
                continue

            i = int(row_token) - 1
            j = int(col_token) - 1
            if i < 0 or j < 0:
                continue

            label = f"S{i+1}{j+1}"
            key = (label, i, j)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)

        return specs

    # ------------------------------------------------------------------
    # HB spectrum plot
    # ------------------------------------------------------------------

    def _update_plot_view_availability(self) -> None:
        """Enable the plot switch only when both an S-parameter and an HB result are expected."""
        hb_possible = SimulationType.HB in self._required_sim_types and bool(self._hb_probe_nodes)
        sparam_possible = SimulationType.AC in self._required_sim_types
        both = hb_possible and sparam_possible

        self.plot_view_combo.setEnabled(both)
        if not both:
            self.plot_view_combo.blockSignals(True)
            self.plot_view_combo.setCurrentIndex(1 if hb_possible else 0)
            self.plot_view_combo.blockSignals(False)
        self.on_plot_view_changed()

    def on_plot_view_changed(self) -> None:
        hb_active = self.plot_view_combo.currentData() == "hb"
        self.left_plot_stack.setCurrentIndex(1 if hb_active else 0)
        self.hb_quantity_combo.setEnabled(hb_active)
        self.hb_input_port_combo.setEnabled(
            hb_active and self.hb_quantity_combo.currentData() == "gain"
        )
        for widget in (self.show_goals_cb, self.plot_prev_cb, self.zoom_btn):
            widget.setEnabled(not hb_active)

    def on_hb_quantity_changed(self) -> None:
        self.hb_input_port_combo.setEnabled(self.hb_quantity_combo.currentData() == "gain")
        self.update_hb_spectrum_plot()

    def _hb_pin_dbm(self) -> Optional[float]:
        """Available input power in dBm of the port selected as gain reference."""
        info = self._port_sources.get(self.hb_input_port_combo.currentData())
        if not info:
            return None
        amplitude = info.get("sin_amplitude") or info.get("ac_amplitude")
        if amplitude is None:
            return None
        return hb_spectrum.available_power_dbm(amplitude, info.get("z0", 50.0))

    def _hb_fundamentals(self) -> List[float]:
        """Fundamental tone(s) of the HB analysis, taken from the .HB directive field."""
        edit = self._sim_param_edits.get(f"{SimulationType.HB.value.upper()}:frequencies")
        return hb_spectrum.parse_fundamentals(edit.text() if edit else None)

    def _hb_max_order(self) -> int:
        """Highest harmonic order to consider when labelling bins (from .options hbint numfreq)."""
        numfreq = self._parsed_options.get("hbint", {}).get("numfreq", "")
        orders = [int(tok) for tok in re.findall(r"\d+", str(numfreq))]
        return max(orders) if orders else 10

    def update_hb_spectrum_plot(self, sim_result=None) -> None:
        """Redraw the HB spectrum for the selected analysis point and quantity."""
        if sim_result is not None:
            self._hb_dataframes = dict(sim_result.dataframes)

        self.hb_spectrum_plot.clear()
        self._hb_markers = []
        self._hb_spectrum_data = None

        quantity = self.hb_quantity_combo.currentData() or "power"
        meta = hb_spectrum.QUANTITY_META[quantity]
        self.hb_spectrum_plot.setLabel("left", meta["label"], units=meta["unit"])

        pin_dbm = 0.0
        if quantity == "gain":
            pin = self._hb_pin_dbm()
            if pin is None:
                self.hb_spectrum_plot.setTitle(
                    "HB Spectrum — no input port with a SIN/AC source to reference the gain to"
                )
                return
            pin_dbm = pin

        node = self.hb_analysis_point
        df = hb_spectrum.find_dataframe(self._hb_dataframes, node, quantity) if node else None
        if df is None:
            self.hb_spectrum_plot.setTitle("HB Spectrum — no data")
            return

        freqs, values = hb_spectrum.spectrum(df, node, quantity, pin_dbm=pin_dbm)
        fundamentals = self._hb_fundamentals()
        labels = hb_spectrum.classify_bins(freqs, fundamentals, self._hb_max_order())
        self._hb_spectrum_data = (freqs, values, labels, meta["unit"])

        v_max = float(np.max(values))
        # Numerically-zero bins sit at the -300 dB floor and would squash the plot.
        baseline = max(float(np.min(values)), v_max - 150.0)
        span = max(v_max - baseline, 1.0)

        fundamental_mask = np.array([hb_spectrum.is_fundamental(l) for l in labels], dtype=bool)
        for mask, color, name in (
            (~fundamental_mask, (65, 105, 225), "Harmonics / mixing products"),
            (fundamental_mask, (220, 20, 60), "Fundamental"),
        ):
            if not mask.any():
                continue
            x = np.repeat(freqs[mask], 2)
            y = np.empty(x.size)
            y[0::2] = baseline
            y[1::2] = np.maximum(values[mask], baseline)
            self.hb_spectrum_plot.plot(
                x, y, connect="pairs", pen=pg.mkPen(color, width=2), name=name
            )

        self.hb_spectrum_plot.setYRange(baseline, v_max + 0.1 * span)
        tone_text = (
            ", ".join(f"{f/1e9:g} GHz" for f in fundamentals) if fundamentals else "unknown tones"
        )
        reference = (
            f" ref {self.hb_input_port_combo.currentData()}" if quantity == "gain" else ""
        )
        self.hb_spectrum_plot.setTitle(
            f"HB Spectrum — {meta['label']} at {node}{reference}  [f0 = {tone_text}]"
        )

    def on_hb_spectrum_clicked(self, event) -> None:
        """Place a numbered marker on the nearest spectral line."""
        if self._hb_spectrum_data is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.hb_spectrum_plot.sceneBoundingRect().contains(event.scenePos()):
            return

        freqs, values, labels, unit = self._hb_spectrum_data
        x = self.hb_spectrum_plot.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
        idx = int(np.argmin(np.abs(freqs - x)))
        freq, value, label = freqs[idx], values[idx], labels[idx]

        marker = pg.ScatterPlotItem(
            [freq], [value], symbol="o", size=10, pen=pg.mkPen("k"), brush=None
        )
        text = pg.TextItem(
            f"M{len(self._hb_markers) // 2 + 1}: {freq/1e9:.3f} GHz  {value:.2f} {unit}"
            + (f"  [{label}]" if label else ""),
            color="k",
            anchor=(0, 1),
        )
        text.setPos(freq, value)
        for item in (marker, text):
            self.hb_spectrum_plot.addItem(item)
            self._hb_markers.append(item)

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
                
                p_name = goal.parameter_name
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


    def update_optimizer_options(self):
        """Rebuild per-optimizer settings rows using the selected optimizer's _settings list."""
        for _ in range(self.opt_options_count):
            result = self.config_form_layout.takeRow(self.opt_options_insert_pos)
            if result.labelItem:
                w = result.labelItem.widget()
                if w: w.deleteLater()
            if result.fieldItem:
                w = result.fieldItem.widget()
                if w: w.deleteLater()

        self.opt_options_count = 0
        self.optimizer_widgets = {}

        opt_cls = self.optimizer_combo.currentData()
        if not opt_cls:
            return

        idx = self.opt_options_insert_pos
        count = 0

        cobra_settings = getattr(opt_cls, "_settings", None)
        if cobra_settings:
            for setting in cobra_settings:
                widget = self._widget_for_setting(setting.dtype, setting.default, setting.choices)
                widget.setToolTip(setting.description)
                label_text = setting.name.replace("_", " ").title()
                self.config_form_layout.insertRow(idx, label_text + ":", widget)
                self.optimizer_widgets[setting.name] = widget
                idx += 1
                count += 1
        else:
            try:
                sig = inspect.signature(opt_cls.__init__)
            except ValueError:
                return
            for name, param in sig.parameters.items():
                if name in ("self", "args", "kwargs"):
                    continue
                widget = self._widget_for_setting(param.annotation, param.default)
                label_text = name.replace("_", " ").title()
                self.config_form_layout.insertRow(idx, label_text + ":", widget)
                self.optimizer_widgets[name] = widget
                idx += 1
                count += 1

        self.opt_options_count = count

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

        idx = self.sim_options_insert_pos
        count = 0

        # Prefer _settings list (CobraSetting descriptors) for rich metadata.
        # Fall back to inspect.signature for classes that don't declare _settings.
        cobra_settings = getattr(sim_cls, "_settings", None)

        if cobra_settings:
            for setting in cobra_settings:
                widget = self._widget_for_setting(setting.dtype, setting.default, setting.choices)
                widget.setToolTip(setting.description)
                label_text = setting.name.replace("_", " ").title()
                self.config_form_layout.insertRow(idx, label_text + ":", widget)
                self.simulator_widgets[setting.name] = widget
                idx += 1
                count += 1
        else:
            try:
                sig = inspect.signature(sim_cls.__init__)
            except ValueError:
                return
            for name, param in sig.parameters.items():
                if name in ("self", "args", "kwargs"):
                    continue
                widget = self._widget_for_setting(param.annotation, param.default)
                label_text = name.replace("_", " ").title()
                self.config_form_layout.insertRow(idx, label_text + ":", widget)
                self.simulator_widgets[name] = widget
                idx += 1
                count += 1

        self.sim_options_count = count

    @staticmethod
    def _widget_for_setting(dtype, default, choices=None) -> "QWidget":
        """Return the appropriate input widget for a given type, default value, and choices."""
        if choices is not None:
            widget = QComboBox()
            for label, value in choices:
                widget.addItem(label, value)
            # Pre-select the item matching the default value
            for i in range(widget.count()):
                if widget.itemData(i) == default:
                    widget.setCurrentIndex(i)
                    break
            return widget
        if dtype == bool or isinstance(default, bool):
            widget = QCheckBox()
            if isinstance(default, bool):
                widget.setChecked(default)
        elif dtype == int or (isinstance(default, int) and not isinstance(default, bool)):
            widget = QSpinBox()
            widget.setRange(-1_000_000, 1_000_000)
            if isinstance(default, int):
                widget.setValue(default)
        elif dtype == float or isinstance(default, float):
            widget = QDoubleSpinBox()
            widget.setRange(-1e9, 1e9)
            if isinstance(default, float):
                widget.setValue(default)
        else:
            widget = QLineEdit()
            if default is not inspect.Parameter.empty and default is not None:
                widget.setText(str(default))
        return widget

    def on_finetune_toggled(self, checked):
        self.palace_label.setVisible(checked)
        self.palace_edit.setVisible(checked)
        self.ft_iter_label.setVisible(checked)
        self.ft_iter_spin.setVisible(checked)
        self.ft_optimizer_label.setVisible(checked)
        self.ft_optimizer_combo.setVisible(checked)
        self._update_geometry_selectors_visibility()

    def browse_file(self, line_edit, filter):
        fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter)
        if fname:
            line_edit.setText(fname)

    def _open_hf_model_dialog(self, line_edit: QLineEdit):
        dlg = HuggingFaceModelDialog(self)
        if dlg.exec():
            if dlg.selected_file_path:
                line_edit.setText(dlg.selected_file_path)

    @staticmethod
    def _is_touchstone_path(path: str) -> bool:
        lower = path.lower()
        return lower.endswith(tuple(f".s{i}p" for i in range(1, 10)) + (".snp",))

    def _update_geometry_selectors_visibility(self):
        """Show a geometry selector for each ONNX component when fine-tuning is enabled."""
        checked = self.finetune_cb.isChecked()
        any_visible = False
        for comp_name, selector in self.component_geometry_selectors.items():
            edit = self.component_onnx_edits.get(comp_name)
            path = edit.text().strip() if edit else ""
            show = checked and bool(path) and not self._is_touchstone_path(path)
            selector.setVisible(show)
            if show:
                any_visible = True
        self.component_geometry_container.setVisible(any_visible)

    def create_orca_geometries(self) -> Dict[str, object]:
        """Instantiate and return a geometry for every visible ONNX-component selector."""
        geometries: Dict[str, object] = {}
        for comp_name, selector in self.component_geometry_selectors.items():
            if selector.isVisible():
                geometries[comp_name] = selector.get_geometry()
        return geometries

    def on_netlist_selected(self):
        """Handle netlist selection with automatic parsing."""
        fname, _ = QFileDialog.getOpenFileName(self, "Select Netlist", "", "Netlist Files (*.cir *.sp)")
        if fname:
            self.netlist_edit.setText(fname)
            self.opt_params.clear()
            self.param_table.setRowCount(0)
            self.parse_and_update_components(fname)
    
    def parse_and_update_components(self, netlist_path: str):
        """Parse netlist and update UI with detected components, simulation type and port count."""
        try:
            parser = XyceNetlistParser().from_file(netlist_path)
            components = parser.components

            # Update available parameters (full list for port count — not gated by sim type)
            sim_type = parser.simulation_type
            self._num_ports = parser.num_ports
            self._parsed_directives = list(parser.simulation_directives)
            self._parsed_options = dict(parser.options_directives)
            self._port_sources = dict(parser.port_sources)
            self._hb_probe_nodes = list(parser.hb_probe_nodes)
            self._populate_hb_point_combo()
            self._populate_hb_input_port_combo()
            self._rebuild_design_parameters()

            self._netlist_sim_type = sim_type
            self.sim_type_value_label.setText(sim_type.display_name)
            self.sim_type_label.setVisible(True)
            self.sim_type_value_label.setVisible(True)

            # Populate editable simulation-parameter fields for required sim types
            self._refresh_sim_params_for_goals()

            if components:
                # New workflow: show component-based ONNX/Touchstone selectors
                self.update_component_onnx_selectors(components, netlist_path)
                status = f"Found {len(components)} component(s): {', '.join(components.keys())}"
                self.statusBar().showMessage(status)
            else:
                self.clear_component_onnx_selectors()
                self.statusBar().showMessage("No components found in netlist.")
        except Exception as e:
            QMessageBox.warning(self, "Netlist Parsing Error", f"Failed to parse netlist:\n{str(e)}")
            self.clear_component_onnx_selectors()
            self._netlist_sim_type = None
            self.sim_type_label.setVisible(False)
            self.sim_type_value_label.setVisible(False)
            self._populate_sim_param_widgets(set())

    @property
    def hb_analysis_point(self) -> Optional[str]:
        """The node whose HB spectrum is plotted and used by HB design goals."""
        return self.hb_point_combo.currentText() or None

    def _populate_hb_point_combo(self) -> None:
        """Fill the analysis-point combo from the netlist's HB voltage/current probes."""
        previous = self.hb_point_combo.currentText()
        self.hb_point_combo.blockSignals(True)
        self.hb_point_combo.clear()
        self.hb_point_combo.addItems(self._hb_probe_nodes)
        preferred = next(
            (n for n in (previous, "Out") if n and n.upper() in {p.upper() for p in self._hb_probe_nodes}),
            None,
        )
        if preferred:
            self.hb_point_combo.setCurrentIndex(
                next(i for i, n in enumerate(self._hb_probe_nodes) if n.upper() == preferred.upper())
            )
        self.hb_point_combo.blockSignals(False)

    def _populate_hb_input_port_combo(self) -> None:
        """Fill the gain reference combo with the ports that actually drive the circuit."""
        previous = self.hb_input_port_combo.currentData()
        self.hb_input_port_combo.blockSignals(True)
        self.hb_input_port_combo.clear()
        for port_name, info in sorted(self._port_sources.items()):
            amplitude = info.get("sin_amplitude") or info.get("ac_amplitude")
            if amplitude is None:
                continue
            pin_dbm = hb_spectrum.available_power_dbm(amplitude, info.get("z0", 50.0))
            self.hb_input_port_combo.addItem(f"{port_name} — Pin {pin_dbm:.2f} dBm", port_name)
        index = self.hb_input_port_combo.findData(previous)
        if index >= 0:
            self.hb_input_port_combo.setCurrentIndex(index)
        self.hb_input_port_combo.blockSignals(False)

    def _rebuild_design_parameters(self) -> None:
        """Rebuild the available design parameters for the current netlist and analysis point."""
        self._available_parameters = get_available_parameters(self._num_ports)
        node = self.hb_analysis_point
        if not node:
            return
        self._available_parameters.append(make_power_dbm(node))
        # Gain needs the input drive level, so only ports with a SIN/AC source qualify.
        for port_name, source_info in sorted(self._port_sources.items()):
            sin_amp = source_info.get("sin_amplitude") or source_info.get("ac_amplitude")
            if sin_amp is not None:
                self._available_parameters.append(
                    make_gain_db(port_name, sin_amp, source_info.get("z0", 50.0), node)
                )

    def on_hb_point_changed(self) -> None:
        self._rebuild_design_parameters()
        self.update_hb_spectrum_plot()
    
    def _refresh_sim_params_for_goals(self) -> None:
        """Rebuild sim-param widgets for the native sim type and any goal-required additions."""
        required: set = set()
        # Native sim type (from netlist) is always shown and always run
        if self._netlist_sim_type is not None:
            required.add(self._netlist_sim_type)
        # Add any additional types required by goals (e.g. AC for S-parameter goals
        # when the native type is HB)
        for goal in self.goals:
            st = goal.required_simulation_type
            if st is not SimulationType.UNKNOWN:
                required.add(st)
        if not required:
            required = {SimulationType.AC}
        self._required_sim_types = required
        # Update the displayed simulation type label to reflect all required types
        all_types_str = ", ".join(st.display_name for st in sorted(required, key=lambda t: t.value))
        self.sim_type_value_label.setText(all_types_str)
        self._populate_sim_param_widgets(required)

        show_hb = SimulationType.HB in required and bool(self._hb_probe_nodes)
        self.hb_point_label.setVisible(show_hb)
        self.hb_point_combo.setVisible(show_hb)
        self._update_plot_view_availability()

    def _populate_sim_param_widgets(self, sim_types: "set[SimulationType]") -> None:
        """Create editable fields for each sim type in *sim_types*.

        Each simulation type is rendered as a QGroupBox so the sections look
        consistent with the rest of the GUI (Design Goals, Optimization Parameters).
        """
        # Clear all child widgets from the container
        while self.sim_params_layout.count():
            item = self.sim_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sim_param_edits.clear()

        skip = {"sweep_type", "src_name"}

        for sim_type in sorted(sim_types, key=lambda t: t.value):
            meta = self._simulator_cls.get_simulation_metadata(sim_type)
            param_names = meta.positional_param_names
            options_cat = meta.options_category
            options_params = self._parsed_options.get(options_cat, {}) if options_cat else {}
            visible_params = [n for n in param_names if n not in skip]

            if not visible_params and not options_params:
                continue

            directive_key = sim_type.value.upper()
            descriptions = meta.positional_param_descriptions
            defaults = meta.positional_param_defaults

            # Pull values from the parsed directive that matches this type
            parsed_values: dict = {}
            for d in self._parsed_directives:
                if SimulationType.from_directive(d.directive) is sim_type:
                    for i, name in enumerate(param_names):
                        if i < len(d.positional):
                            # For the last named param, absorb all remaining positional
                            # tokens (handles multi-tone HB: .HB 95E9 10E9).
                            if i == len(param_names) - 1:
                                parsed_values[name] = " ".join(d.positional[i:])
                            else:
                                parsed_values[name] = d.positional[i]
                    break

            # --- Main directive group box ---
            group = QGroupBox(f"{sim_type.value} Parameters")
            form = QFormLayout(group)
            for name in visible_params:
                value = parsed_values.get(name, defaults.get(name, ""))
                key = f"{directive_key}:{name}"
                edit = QLineEdit(value)
                if name in descriptions:
                    edit.setToolTip(descriptions[name])
                label = name.replace("_", " ").title()
                form.addRow(f"{label}:", edit)
                self._sim_param_edits[key] = edit
            self.sim_params_layout.addWidget(group)

            # --- .options group box (if relevant params exist) ---
            if options_params:
                opt_group = QGroupBox(f".OPTIONS {options_cat.upper()}")
                opt_form = QFormLayout(opt_group)
                opt_descriptions = meta.options_param_descriptions
                for opt_name, opt_value in options_params.items():
                    key = f".OPTIONS:{options_cat}:{opt_name}"
                    edit = QLineEdit(opt_value)
                    if opt_name.lower() in opt_descriptions:
                        edit.setToolTip(opt_descriptions[opt_name.lower()])
                    label = opt_name.replace("_", " ").title()
                    opt_form.addRow(f"{label}:", edit)
                    self._sim_param_edits[key] = edit
                self.sim_params_layout.addWidget(opt_group)

        self.sim_params_container.setVisible(bool(self._sim_param_edits))

    def _apply_simulation_parameters(self, parser) -> None:
        """Apply GUI-edited simulation parameters to an in-memory parser.

        Only updates directives that already exist in the netlist.  Directives
        that are absent (e.g. .AC when the native type is .HB) will be injected
        with the correct parameters by CircuitSimulationStage at run-time.
        """
        existing_directives = {
            SimulationType.from_directive(d.directive)
            for d in self._parsed_directives
        }

        updates: Dict[str, Dict[str, str]] = {}
        options_updates: Dict[str, Dict[str, str]] = {}
        for key, edit in self._sim_param_edits.items():
            if key.startswith(".OPTIONS:"):
                # Format: ".OPTIONS:<category>:<param>"
                _, category, param_name = key.split(":", 2)
                options_updates.setdefault(category, {})[param_name] = edit.text().strip()
            else:
                directive, param_name = key.split(":", 1)
                st = SimulationType.from_directive(directive)
                if st not in existing_directives:
                    continue  # will be injected at run-time — skip
                updates.setdefault(directive, {})[param_name] = edit.text().strip()

        for directive, params in updates.items():
            try:
                parser.update_simulation_directive(directive, params)
            except Exception as e:
                print(f"Warning: Could not update directive {directive}: {e}")

        for category, params in options_updates.items():
            try:
                parser.update_options_directive(category, params)
            except Exception as e:
                print(f"Warning: Could not update .options {category}: {e}")

    def update_component_onnx_selectors(self, components: Dict, netlist_path: str):
        """Create ONNX/Touchstone selectors for each detected component."""
        # Clear existing component selectors if any
        self.clear_component_onnx_selectors()
        
        # Create component ONNX selector widgets
        self.component_onnx_edits = {}
        self.component_onnx_btns = {}
        self.component_hf_btns = {}
        self.component_onnx_container.setVisible(bool(components))
        if not components:
            return
        
        netlist_dir = os.path.dirname(netlist_path) if netlist_path else ""

        # Insert ONNX/Touchstone selectors for each component
        for comp_name, comp_data in sorted(components.items()):
            comp_edit = QLineEdit()
            
            # Check for a static TSTONEFILE parameter from the netlist
            tstone_file = comp_data.params.get("TSTONEFILE")
            if tstone_file:
                # If the path is not absolute, resolve it relative to the netlist directory
                if not os.path.isabs(tstone_file) and netlist_dir:
                    tstone_file = os.path.normpath(os.path.join(netlist_dir, tstone_file))
                
                # Always populate the GUI, even if the file isn't found locally yet
                comp_edit.setText(tstone_file)
            
            comp_btn = QPushButton("Browse")
            comp_btn.clicked.connect(
                lambda checked, edit=comp_edit: self.browse_file(edit, "ONNX/Touchstone Files (*.onnx *.s*p *.snp)")
            )
            comp_hf_btn = QPushButton("Select from HuggingFace")
            comp_hf_btn.setToolTip("Browse and download ORCA surrogate models from HuggingFace")
            comp_hf_btn.clicked.connect(
                lambda checked, edit=comp_edit: self._open_hf_model_dialog(edit)
            )
            comp_edit.textChanged.connect(
                lambda text, c=comp_name: self.on_onnx_file_changed(c, text)
            )
            
            h_layout = QHBoxLayout()
            h_layout.addWidget(comp_edit)
            h_layout.addWidget(comp_btn)
            h_layout.addWidget(comp_hf_btn)
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(f"{comp_name} Model:"))
            row_layout.addLayout(h_layout)
            self.component_onnx_layout.addWidget(row_widget)
            self.component_onnx_edits[comp_name] = comp_edit
            self.component_onnx_btns[comp_name] = comp_btn
            self.component_hf_btns[comp_name] = comp_hf_btn

        # Create a geometry selector for each component (shown only when fine-tuning is enabled
        # and the component uses an ONNX surrogate rather than a fixed .snp file)
        self.component_geometry_selectors = {}
        for comp_name in sorted(components.keys()):
            selector = GeometrySelectorWidget(f"ORCA Geometry for {comp_name}", parent=self)
            selector.setVisible(False)
            self.component_geometry_selectors[comp_name] = selector
            self.component_geometry_layout.addWidget(selector)

        # Re-evaluate visibility in case fine-tuning is already enabled
        self._update_geometry_selectors_visibility()

    def clear_component_onnx_selectors(self):
        """Remove all component ONNX selectors from the UI."""
        if hasattr(self, 'component_onnx_edits'):
            for edit in self.component_onnx_edits.values():
                edit.deleteLater()
            self.component_onnx_edits.clear()
        
        if hasattr(self, 'component_onnx_btns'):
            for btn in self.component_onnx_btns.values():
                btn.deleteLater()
            self.component_onnx_btns.clear()

        if hasattr(self, 'component_hf_btns'):
            for btn in self.component_hf_btns.values():
                btn.deleteLater()
            self.component_hf_btns.clear()

        if hasattr(self, 'component_onnx_container'):
            self.component_onnx_container.setVisible(False)

        if hasattr(self, 'component_onnx_layout'):
            while self.component_onnx_layout.count():
                item = self.component_onnx_layout.takeAt(0)
                widget = item.widget() if item else None
                if widget:
                    widget.deleteLater()

        if hasattr(self, 'component_inputs'):
            self.component_inputs.clear()

        # Clear geometry selectors
        if hasattr(self, 'component_geometry_selectors'):
            for selector in self.component_geometry_selectors.values():
                self.component_geometry_layout.removeWidget(selector)
                selector.deleteLater()
            self.component_geometry_selectors.clear()
        if hasattr(self, 'component_geometry_container'):
            self.component_geometry_container.setVisible(False)
    
    def on_onnx_file_changed(self, comp_name: str, path: str):
        path = path.strip()
        # Always refresh geometry selector visibility when file selection changes
        self._update_geometry_selectors_visibility()

        if not path or not os.path.exists(path):
            return
        
        try:
            sess = onnxruntime.InferenceSession(path)
            all_inputs = [i.name for i in sess.get_inputs()]
            metadata = sess.get_modelmeta().custom_metadata_map

            if not hasattr(self, 'component_inputs'):
                self.component_inputs = {}
                
            # Filter out 'frequency' from inputs
            filtered_inputs = [p for p in all_inputs if p.lower() != "frequency"]
            
            prefixed_inputs = [f"{comp_name}:{p}" for p in filtered_inputs]
            self.component_inputs[comp_name] = prefixed_inputs
            
            current_param_names = {p.name for p in self.opt_params}
            
            for original_name, prefixed_name in zip(filtered_inputs, prefixed_inputs):
                if prefixed_name not in current_param_names:
                    dlg = OptimizationParamDialog(
                        from_source="ONNX",
                        source_data=[prefixed_name],
                        parent=self,
                        metadata=metadata,
                        link_candidates=[p.name for p in self.opt_params],
                    )
                    self._add_param_to_table(dlg.get_data())
                    current_param_names.add(prefixed_name)
        except Exception as e:
            # Not an ONNX file or failed to parse, simply return
            pass

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
        dlg = OptimizationParamDialog(
            parent=self,
            param=param,
            link_candidates=[p.name for i, p in enumerate(self.opt_params) if i != row],
        )
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
        self.param_table.setItem(row, 6, QTableWidgetItem(str(param.linked_to or "")))

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

            # Also expose key=value parameters on X instances whose subcircuit
            # is defined inline (via .SUBCKT). These are treated as netlist
            # variables, not surrogate model inputs.
            inline_names = parser.inline_subckt_names
            for elem in parser.list_elements(["X"]):
                if elem.model in inline_names:
                    for key in elem.params:
                        prospects.append(f"{elem.name}:{key}")

            # Filter out already added parameters
            current_param_names = {p.name for p in self.opt_params}
            available_prospects = [name for name in prospects if name not in current_param_names]

            if not available_prospects:
                QMessageBox.information(self, "Info", "All valid netlist parameters have already been added.")
                return
            
            dlg = OptimizationParamDialog(
                "NETLIST",
                available_prospects,
                self,
                link_candidates=[p.name for p in self.opt_params],
            )
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
        self.param_table.setItem(row, 6, QTableWidgetItem(str(param.linked_to or "")))

    def add_design_goal(self):
        if not self._available_parameters:
            QMessageBox.information(
                self,
                "Load a Netlist First",
                "Please select a netlist file so that COBRA can determine the "
                "available design parameters for your simulation type and port count.",
            )
            return
        dlg = DesignGoalDialog(self, available_parameters=self._available_parameters)
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
        dlg = DesignGoalDialog(self, goal=goal, available_parameters=self._available_parameters if self._available_parameters else None)
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
        self._refresh_sim_params_for_goals()

    def _add_goal_to_list(self, goal):
        self.goals.append(goal)
        label = self._goal_label(goal)
        self.goal_list.addItem(label)
        idx = len(self.goals) - 1
        self.loss_history[idx] = []
        self._refresh_sim_params_for_goals()

    def _update_goal_item(self, item, goal):
        item.setText(self._goal_label(goal))
        self._refresh_sim_params_for_goals()
    
    def _goal_label(self, goal):
        label = f"{goal.parameter_name} "
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
            self._set_action_button_state("pause")
        else:
            self.worker.pause()
            self._set_action_button_state("resume")

    def start_optimization(self):
        netlist = self.netlist_edit.text()
        if not netlist:
            QMessageBox.warning(self, "Missing Netlist", "Please select a netlist file.")
            return

        # Parse netlist and extract components
        try:
            netlist_parser = XyceNetlistParser().from_file(netlist)
        except Exception as e:
            QMessageBox.critical(self, "Netlist Parsing Error", f"Failed to parse netlist:\n{str(e)}")
            return

        # Apply any GUI-edited simulation parameters (start/stop freq, points, etc.)
        self._apply_simulation_parameters(netlist_parser)
        
        # Get detected components
        components = netlist_parser.components
        
        if components:
            # Component-based workflow: validate all components have ONNX files
            component_onnx_mapping = {}
            for comp_name in components.keys():
                if comp_name not in self.component_onnx_edits:
                    QMessageBox.warning(
                        self, "Missing Component Selector",
                        f"Model selector for component '{comp_name}' not found."
                    )
                    return
                onnx_path = self.component_onnx_edits[comp_name].text()
                if not onnx_path:
                    QMessageBox.warning(
                        self, "Missing Model Files",
                        f"Please select ONNX or Touchstone model for component '{comp_name}'."
                    )
                    return
                component_onnx_mapping[comp_name] = onnx_path
        else:
            # No surrogate components — optimization-only mode (e.g. pure HB)
            component_onnx_mapping = {}

        orca_geometries = {}
        if self.finetune_cb.isChecked():
            try:
                orca_geometries = self.create_orca_geometries()
            except Exception as exc:
                QMessageBox.critical(self, "ORCA Geometry", str(exc))
                return

        optimizer_cls = self.optimizer_combo.currentData()
        simulator_cls = self.simulator_combo.currentData()

        # Gather optimizer kwargs from dynamically-created CobraSetting widgets
        optimizer_kwargs = {}
        for name, widget in self.optimizer_widgets.items():
            if isinstance(widget, QComboBox):
                optimizer_kwargs[name] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                optimizer_kwargs[name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                val = widget.value()
                # -1 is the sentinel meaning "no seed" for int | None params
                if name == "random_seed" and val == -1:
                    optimizer_kwargs[name] = None
                else:
                    optimizer_kwargs[name] = val
            elif isinstance(widget, QDoubleSpinBox):
                optimizer_kwargs[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                optimizer_kwargs[name] = widget.text()
        
        # Gather simulator kwargs
        sim_kwargs = {}
        for name, widget in self.simulator_widgets.items():
             if isinstance(widget, QCheckBox):
                 sim_kwargs[name] = widget.isChecked()
             elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                 sim_kwargs[name] = widget.value()
             elif isinstance(widget, QLineEdit):
                 sim_kwargs[name] = widget.text()
        
        # Create COBRA with component-based workflow
        cobra = COBRA(
            netlist_parser=netlist_parser,
            component_onnx_mapping=component_onnx_mapping,
            optimizer=optimizer_cls(**optimizer_kwargs),
            circuit_simulator=simulator_cls(**sim_kwargs),
            palace_fine_tuning_command=(self.palace_edit.text() or None) if self.finetune_cb.isChecked() else None,
            fine_tuning_iterations=self.ft_iter_spin.value(),
            fine_tuning_optimizer=self.ft_optimizer_combo.currentData() if self.finetune_cb.isChecked() else "reuse",
        )
        
        # Change button to PAUSE (running state)
        self._set_action_button_state("pause")
        self.stop_btn.setEnabled(True)
        self._update_progress_display(0, self.max_iter_spin.value())
        self.elapsed_label.setText("Time: 0.0s")
        self.fine_tuning_active = False
        self.fine_tuning_notification_shown = False
        
        # Clear plots
        self.s_param_plot.clear()
        self.loss_plot.clear()
        self.loss_history = {i: [] for i in range(len(self.goals))}
        self.overlay_items = [] # clear tracked items since plot.clear() removed them
        
        self.draw_overlays()
        
        # Build sim_params_by_type dict from GUI edits, keyed by SimulationType
        sim_params_by_type: Dict[SimulationType, Dict[str, str]] = {}
        for key, edit in self._sim_param_edits.items():
            directive_str, param_name = key.split(":", 1)
            st = SimulationType.from_directive(directive_str)
            if st is not SimulationType.UNKNOWN:
                sim_params_by_type.setdefault(st, {})[param_name] = edit.text().strip()

        self.worker = OptimizationWorker(
            cobra, netlist, self.goals,
            self.opt_params, self.max_iter_spin.value(),
            orca_geometries,
            sim_params_by_type=sim_params_by_type,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.ask_continue.connect(self.on_ask_continue, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self.set_active_panel("viz")

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._set_action_button_state("stopping", enabled=False)
            self.stop_btn.setEnabled(False)

    @Slot(int)
    def on_ask_continue(self, current_max: int):
        # Ask user for new max iterations
        new_max, ok = QInputDialog.getInt(
            self,
            "Target Not Reached",
            f"Maximum iterations ({current_max}) reached without hitting the design goals.\n\nEnter a new maximum to continue, or cancel to stop:",
            value=current_max,
            minValue=current_max,
            maxValue=99999
        )
        if ok and new_max > current_max:
            if self.worker:
                self.worker.max_iterations = new_max
            self.max_iter_spin.setValue(new_max)
            self._update_progress_display(self.progress_bar.value(), new_max)

    @Slot(dict)
    def on_progress(self, context: dict):
        iteration = context.get("iteration", 0)
        max_iterations = self.worker.max_iterations if self.worker else self.max_iter_spin.value()
        if context.get("fine_tuning_active"):
            self.fine_tuning_active = True
            ft_iteration = context.get("fine_tuning_iteration", 0)
            ft_total = context.get("fine_tuning_total", self.ft_iter_spin.value())
            self._update_finetuning_display(ft_iteration, ft_total)

            if not self.fine_tuning_notification_shown:
                start_iter = context.get("fine_tuning_start_iteration")
                if start_iter is not None:
                    self.statusBar().showMessage(
                        f"Goals have been reached after iteration {start_iter}. Starting finetuning...",
                        5000,
                    )
                    self.fine_tuning_notification_shown = True
        else:
            self._update_progress_display(iteration, max_iterations)
        
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

        self.current_param_table.setRowCount(0)
        
        # Determine how to display the current parameters: Group them by component if any
        if hasattr(self, 'component_inputs') and self.component_inputs:
            all_comp_inputs = {inp for inputs in self.component_inputs.values() for inp in inputs}
            other_params = [name for name in current_values.keys() if name not in all_comp_inputs]
            
            for comp_name, inputs in sorted(self.component_inputs.items()):
                # Add component header row
                r = self.current_param_table.rowCount()
                self.current_param_table.insertRow(r)
                
                header_item = QTableWidgetItem(f"--- {comp_name} ---")
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)
                header_item.setBackground(Qt.GlobalColor.lightGray)
                self.current_param_table.setItem(r, 0, header_item)
                
                empty_item = QTableWidgetItem("")
                empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
                empty_item.setBackground(Qt.GlobalColor.lightGray)
                self.current_param_table.setItem(r, 1, empty_item)
                
                for name in sorted(inputs):
                    if name in current_values:
                        display_name = name.split(":", 1)[1] if ":" in name else name
                        self._add_current_param_row(display_name, current_values[name])
            
            if other_params:
                r = self.current_param_table.rowCount()
                self.current_param_table.insertRow(r)
                
                header_item = QTableWidgetItem("--- Netlist / Other ---")
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)
                header_item.setBackground(Qt.GlobalColor.lightGray)
                self.current_param_table.setItem(r, 0, header_item)
                
                empty_item = QTableWidgetItem("")
                empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
                empty_item.setBackground(Qt.GlobalColor.lightGray)
                self.current_param_table.setItem(r, 1, empty_item)
                
                for name in sorted(other_params):
                    self._add_current_param_row(name, current_values[name])
        else:
            for name in sorted(current_values.keys()):
                self._add_current_param_row(name, current_values[name])

        # Update Goal Status Table
        sim_results = context.get("simulation_results") or {}
        networks = {st: r.network for st, r in sim_results.items() if r.network is not None}
        # Metrics are now pre-calculated in COBRA.run and stored in context
        current_goals = context.get("goals", [])  # Update goals with latest values
        
        #metrics = context.get("goal_values", {})
        
        self.goal_table.setRowCount(0)

        # print(f"Updating Goal Table with metrics: {metrics} and losses: {losses}")
        
        for i, goal in enumerate(current_goals):
            p_name = goal.parameter.name
            # Get value from metrics
            # metrics contains arrays usually, we might want mean/min/max or just show range
            current_value_s = goal.current_value
            
            display_val = "N/A"
            if current_value_s is not None:
                    # Check if array
                if isinstance(current_value_s, (list, tuple, np.ndarray)):
                    if len(current_value_s) > 0:
                        v_min = np.min(current_value_s)
                        v_max = np.max(current_value_s)
                        display_val = f"[{v_min:.4f}, {v_max:.4f}]"
                else:
                    # Scalar float64
                    display_val = f"{current_value_s:.4f}"

            target_str = ""
            if goal.min_value is not None and goal.max_value is not None:
                target_str = f"[{goal.min_value:.4f}, {goal.max_value:.4f}]"
            elif goal.min_value is not None:
                target_str = f"> {goal.min_value:.4f}"
            elif goal.max_value is not None:
                target_str = f"< {goal.max_value:.4f}"
            
            loss_val = goal.current_penalty if goal.current_penalty is not None else 0.0

            r = self.goal_table.rowCount()
            self.goal_table.insertRow(r)
            
            name_item = QTableWidgetItem(p_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.goal_table.setItem(r, 0, name_item)
            
            target_item = QTableWidgetItem(target_str)
            target_item.setFlags(target_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.goal_table.setItem(r, 1, target_item)
            
            val_item = QTableWidgetItem(f"{display_val} (Penalty={loss_val:.2f})")
            val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.goal_table.setItem(r, 2, val_item)
            
            # Color-code the penalty cell based on loss value
            item = self.goal_table.item(r, 2)
            if item:
                if loss_val > 0:
                    item.setBackground(Qt.GlobalColor.red)
                else:
                    item.setBackground(Qt.GlobalColor.green)


        # 3. Update Loss Plot
        losses = [goal.current_penalty if goal.current_penalty is not None else 0.0 for goal in current_goals]

        if self.loss_history:  # Ensure loss history is initialized
            for i, loss in enumerate(losses):
                if i in self.loss_history:
                    self.loss_history[i].append(float(np.asarray(loss).reshape(-1)[0]))

            self.loss_plot.clear()
            for i, hist in self.loss_history.items():
                if hist:
                    color = pg.intColor(i, hues=max(len(self.loss_history), 1))
                    self.loss_plot.plot(hist, pen=pg.mkPen(color, width=2), name=f"Goal {i+1}")

        # 4. Update S-parameter Plot
        try:
            self.s_param_plot.clear()

            sim_results = context.get("simulation_results") or {}
            ntwk_n = next((r.network for r in sim_results.values() if r.network is not None), None)
            ntwk_prev = context.get("prev_network")
            requested_sparams = self._goal_sparam_specs()
            color_map: Dict[str, Tuple[int, int, int]] = {
                "S11": (220, 20, 60),
                "S21": (65, 105, 225),
                "S12": (46, 139, 87),
                "S22": (255, 140, 0),
            }

            if ntwk_prev is not None and self.plot_prev_cb.isChecked():
                freq_prev = ntwk_prev.f
                for label, i, j in requested_sparams:
                    if i < ntwk_prev.nports and j < ntwk_prev.nports:
                        fallback_qcolor = pg.intColor((i * 10 + j) % 24, hues=24)
                        fallback_color: Tuple[int, int, int] = (
                            fallback_qcolor.red(),
                            fallback_qcolor.green(),
                            fallback_qcolor.blue(),
                        )
                        base_color = color_map.get(label, fallback_color)
                        prev_pen = pg.mkPen(
                            (base_color[0], base_color[1], base_color[2], 120),
                            width=2,
                            style=Qt.PenStyle.DashLine,
                        )
                        self.s_param_plot.plot(freq_prev, ntwk_prev.s_db[:, i, j], pen=prev_pen, name=f"{label} (n-1)")

            if ntwk_n is not None:
                freq = ntwk_n.f
                for label, i, j in requested_sparams:
                    if i < ntwk_n.nports and j < ntwk_n.nports:
                        fallback_qcolor = pg.intColor((i * 10 + j) % 24, hues=24)
                        fallback_color: Tuple[int, int, int] = (
                            fallback_qcolor.red(),
                            fallback_qcolor.green(),
                            fallback_qcolor.blue(),
                        )
                        base_color = color_map.get(label, fallback_color)
                        curr_pen = pg.mkPen(base_color, width=3)
                        self.s_param_plot.plot(freq, ntwk_n.s_db[:, i, j], pen=curr_pen, name=f"{label} (n)")

            # Redraw overlays on top
            self.overlay_items = []  # plot.clear() removed prior overlay items
            self.draw_overlays()
        except Exception as exc:
            print(f"S-parameter plot update failed: {exc}")

        # 5. Update HB Spectrum Plot
        try:
            hb_result = sim_results.get(SimulationType.HB)
            if hb_result is not None:
                self.update_hb_spectrum_plot(hb_result)
        except Exception as exc:
            print(f"HB spectrum plot update failed: {exc}")

    @Slot()
    def on_finished(self):
        self._set_action_button_state("start", enabled=True)
        
        self.stop_btn.setEnabled(False)

        QMessageBox.information(self, "Done", "Optimization Finished!")

    @Slot(str)
    def on_error(self, msg):
        self._set_action_button_state("start", enabled=True)
        
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(self, "Error", msg)

