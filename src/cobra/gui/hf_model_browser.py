"""
HuggingFace ORCA Surrogate Model Browser
=========================================
Provides:
  - ModelEntry          - typed dataclass for a single model entry
  - LocalModelScanner   - discovers already-downloaded models in ./models/
  - HFQueryWorker       - background QThread: queries HuggingFace API
  - HFDownloadWorker    - background QThread: downloads a repo via snapshot_download
  - HuggingFaceModelDialog - full browse / download / select dialog
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_TAG = "orca-surrogate"
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Represents one surrogate model - either local, remote (HF), or both."""

    model_id: str
    owner: str
    repo: str
    source: str  # "local" | "hf"

    # File paths (populated for local models)
    onnx_path: str | None = None
    py_path: str | None = None  # ORCA geometry - None means fine-tuning unavailable

    # HuggingFace metadata (populated after HF query)
    downloads: int | None = None
    likes: int | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    created_at: str | None = None
    last_modified: str | None = None

    @property
    def is_local(self) -> bool:
        return self.source == "local"

    @property
    def finetuning_available(self) -> bool:
        return self.is_local and self.py_path is not None

    def list_label(self) -> str:
        prefix = "[LOCAL] " if self.is_local else ""
        dl_str = f"  ↓{self.downloads}" if self.downloads is not None else ""
        return f"{prefix}{self.model_id}{dl_str}"

    def enrich_from_hf(self, hf_data: dict) -> None:
        """Update metadata fields from a HF query result dict."""
        for key in ("downloads", "likes", "description", "tags",
                    "pipeline_tag", "library_name", "created_at", "last_modified"):
            setattr(self, key, hf_data.get(key))


# ---------------------------------------------------------------------------
# Local model scanner
# ---------------------------------------------------------------------------

class LocalModelScanner:
    """Scans ./models/<owner>/<repo>/ for valid ORCA surrogate repos."""

    def __init__(self, base_dir: Path = MODELS_DIR):
        self.base_dir = base_dir

    def scan(self) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        if not self.base_dir.exists():
            return entries
        for owner_dir in sorted(self.base_dir.iterdir()):
            if not owner_dir.is_dir():
                continue
            for repo_dir in sorted(owner_dir.iterdir()):
                if not repo_dir.is_dir():
                    continue
                onnx_file = repo_dir / f"{repo_dir.name}.onnx"
                if not onnx_file.exists():
                    continue
                py_file = repo_dir / f"{repo_dir.name}.py"
                entries.append(ModelEntry(
                    model_id=f"{owner_dir.name}/{repo_dir.name}",
                    owner=owner_dir.name,
                    repo=repo_dir.name,
                    source="local",
                    onnx_path=str(onnx_file),
                    py_path=str(py_file) if py_file.exists() else None,
                ))
        return entries


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class HFQueryWorker(QThread):
    """
    Background thread that queries HuggingFace for public models tagged
    ``orca-surrogate``, sorted by downloads (descending).

    Signals
    -------
    models_loaded(list[dict])
        Emitted with a list of raw model dicts on success.
    error(str)
        Emitted with an error message on failure.
    """

    models_loaded = Signal(list)
    error = Signal(str)

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            raw = list(api.list_models(filter=HF_TAG, full=True))
            results: list[dict] = []
            for m in raw:
                try:
                    info = api.model_info(m.id, files_metadata=False)
                except Exception:  # noqa: BLE001 - fall back to the listing entry when the API call fails
                    info = m
                if getattr(info, "private", False):
                    continue
                owner = m.id.split("/")[0] if "/" in m.id else "unknown"
                repo = m.id.split("/")[1] if "/" in m.id else m.id
                card_data = getattr(info, "cardData", None)
                description = (
                    card_data.get("description")
                    if isinstance(card_data, dict)
                    else None
                )
                results.append({
                    "model_id": m.id,
                    "owner": owner,
                    "repo": repo,
                    "source": "hf",
                    "downloads": getattr(info, "downloads", 0) or 0,
                    "likes": getattr(info, "likes", 0) or 0,
                    "description": description,
                    "tags": getattr(info, "tags", []) or [],
                    "pipeline_tag": getattr(info, "pipeline_tag", None),
                    "library_name": getattr(info, "library_name", None),
                    "created_at": str(getattr(info, "created_at", "") or ""),
                    "last_modified": str(getattr(info, "last_modified", "") or ""),
                    "onnx_path": None,
                    "py_path": None,
                })
            results.sort(key=lambda x: x["downloads"], reverse=True)
            self.models_loaded.emit(results)
        except Exception as exc:  # noqa: BLE001 - worker thread boundary: failures are forwarded via the error signal
            self.error.emit(str(exc))


class HFDownloadWorker(QThread):
    """
    Background thread that downloads a HuggingFace repo using
    ``snapshot_download`` into ``./models/<owner>/<repo>/``.

    Signals
    -------
    download_finished(str)
        Emitted with the local ``.onnx`` file path on success.
    download_error(str)
        Emitted with an error message on failure.
    progress_message(str)
        Emitted with a human-readable status string during the download.
    """

    download_finished = Signal(str)
    download_error = Signal(str)
    progress_message = Signal(str)

    def __init__(self, model_id: str, local_dir: Path) -> None:
        super().__init__()
        self.model_id = model_id
        self.local_dir = local_dir

    def run(self) -> None:
        try:
            from huggingface_hub import snapshot_download
            self.progress_message.emit(f"Downloading {self.model_id}…")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.local_dir),
                ignore_patterns=["*.git*", "*.gitattributes"],
            )
            repo_name = self.model_id.split("/")[-1]
            onnx_path = self.local_dir / f"{repo_name}.onnx"
            if onnx_path.exists():
                self.download_finished.emit(str(onnx_path))
            else:
                self.download_error.emit(
                    f"Download completed but expected file not found: {onnx_path}"
                )
        except Exception as exc:  # noqa: BLE001 - worker thread boundary: failures are forwarded via the error signal
            self.download_error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class HuggingFaceModelDialog(QDialog):
    """
    Browse, download, and select ORCA surrogate models from HuggingFace.

    - Local models in ``./models/<owner>/<repo>/`` appear at the top with a
      ``[LOCAL]`` badge and are immediately selectable.
    - Remote models are fetched in a background thread (``HFQueryWorker``) and
      appended to the list once the query completes.
    - Selecting a non-local model enables the **Download** button; after a
      successful download the entry is promoted to ``[LOCAL]`` and the
      **Use** button becomes available.

    Result
    ------
    After ``exec()`` returns ``Accepted``:

    - ``selected_file_path``     - absolute path to ``<repo>.onnx``
    - ``selected_geometry_path`` - path to ``<repo>.py`` if present, else ``None``
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Surrogate Models")
        self.resize(920, 600)

        self.selected_file_path: str | None = None
        self.selected_geometry_path: str | None = None

        self._entries: dict[str, ModelEntry] = {}  # model_id → ModelEntry
        self._download_worker: HFDownloadWorker | None = None
        self._scanner = LocalModelScanner()

        self._build_ui()
        self._populate_local_models()
        self._start_hf_query()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([300, 600])

        outer.addLayout(self._build_bottom_bar())

    def _build_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self._loading_label = QLabel("Loading from HuggingFace…")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)

        self._list = QListWidget()
        self._list.setMinimumWidth(280)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        return widget

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("Model Details")
        font = QFont()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        self._detail_labels: dict[str, QLabel] = {}
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key in ("Model ID", "Owner", "Downloads", "Likes", "Pipeline",
                    "Library", "Created", "Modified", "Fine-tuning", "Tags"):
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            form.addRow(f"{key}:", lbl)
            self._detail_labels[key] = lbl
        layout.addLayout(form)

        layout.addWidget(QLabel("Description:"))
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidget(self._desc_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(80)
        layout.addWidget(scroll, stretch=1)

        return widget

    def _build_bottom_bar(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        btn_row = QHBoxLayout()

        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        btn_row.addStretch()

        self._use_btn = QPushButton("Use")
        self._use_btn.setEnabled(False)
        self._use_btn.setDefault(True)
        self._use_btn.clicked.connect(self._on_use)
        btn_row.addWidget(self._use_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        return layout

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate_local_models(self) -> None:
        for entry in self._scanner.scan():
            self._add_entry(entry)

    def _start_hf_query(self) -> None:
        self._query_worker = HFQueryWorker()
        self._query_worker.models_loaded.connect(self._on_hf_models_loaded)
        self._query_worker.error.connect(self._on_hf_error)
        self._query_worker.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_hf_models_loaded(self, models: list) -> None:
        self._loading_label.setVisible(False)
        for data in models:
            mid = data["model_id"]
            if mid in self._entries:
                # Already local: just enrich with HF metadata and refresh label
                self._entries[mid].enrich_from_hf(data)
                self._refresh_list_label(mid)
            else:
                self._add_entry(ModelEntry(**{k: data[k] for k in ModelEntry.__dataclass_fields__}))

    def _on_hf_error(self, msg: str) -> None:
        self._loading_label.setText(f"HuggingFace query failed: {msg}")

    def _on_selection_changed(self, current: QListWidgetItem, _previous) -> None:
        if current is None:
            self._download_btn.setEnabled(False)
            self._use_btn.setEnabled(False)
            return
        entry = self._entries.get(current.data(Qt.ItemDataRole.UserRole))
        if entry is None:
            return
        self._update_details(entry)
        self._download_btn.setEnabled(not entry.is_local and self._download_worker is None)
        self._use_btn.setEnabled(entry.is_local)

    def _on_download(self) -> None:
        current = self._list.currentItem()
        if current is None:
            return
        entry = self._entries.get(current.data(Qt.ItemDataRole.UserRole))
        if entry is None:
            return

        local_dir = MODELS_DIR / entry.owner / entry.repo
        local_dir.mkdir(parents=True, exist_ok=True)

        self._download_worker = HFDownloadWorker(entry.model_id, local_dir)
        self._download_worker.download_finished.connect(self._on_download_finished)
        self._download_worker.download_error.connect(self._on_download_error)
        self._download_worker.progress_message.connect(self._status_label.setText)

        self._download_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._status_label.setText(f"Downloading {entry.model_id}…")
        self._download_worker.start()

    def _on_download_finished(self, onnx_path: str) -> None:
        self._progress_bar.setVisible(False)
        self._download_worker = None

        current = self._list.currentItem()
        if current is None:
            return
        mid = current.data(Qt.ItemDataRole.UserRole)
        entry = self._entries[mid]
        entry.source = "local"
        entry.onnx_path = onnx_path
        py_path = Path(onnx_path).parent / f"{entry.repo}.py"
        entry.py_path = str(py_path) if py_path.exists() else None

        # Promote to top of list
        row = self._list.row(current)
        taken = self._list.takeItem(row)
        taken.setText(entry.list_label())
        self._list.insertItem(0, taken)
        self._list.setCurrentItem(taken)

        self._status_label.setText("Downloaded successfully.")
        self._use_btn.setEnabled(True)

    def _on_download_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._download_worker = None
        self._status_label.setText(f"Download failed: {msg}")
        current = self._list.currentItem()
        if current:
            entry = self._entries.get(current.data(Qt.ItemDataRole.UserRole))
            if entry and not entry.is_local:
                self._download_btn.setEnabled(True)

    def _on_use(self) -> None:
        current = self._list.currentItem()
        if current is None:
            return
        entry = self._entries.get(current.data(Qt.ItemDataRole.UserRole))
        if entry is None or not entry.is_local:
            return
        self.selected_file_path = entry.onnx_path
        self.selected_geometry_path = entry.py_path
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_entry(self, entry: ModelEntry) -> None:
        self._entries[entry.model_id] = entry
        item = QListWidgetItem(entry.list_label())
        item.setData(Qt.ItemDataRole.UserRole, entry.model_id)
        if entry.is_local:
            self._list.insertItem(0, item)
        else:
            self._list.addItem(item)

    def _refresh_list_label(self, model_id: str) -> None:
        entry = self._entries[model_id]
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == model_id:
                item.setText(entry.list_label())
                break

    def _update_details(self, entry: ModelEntry) -> None:
        def _v(x: object) -> str:
            return str(x) if x is not None else "—"

        self._detail_labels["Model ID"].setText(_v(entry.model_id))
        self._detail_labels["Owner"].setText(_v(entry.owner))
        self._detail_labels["Downloads"].setText(_v(entry.downloads))
        self._detail_labels["Likes"].setText(_v(entry.likes))
        self._detail_labels["Pipeline"].setText(_v(entry.pipeline_tag))
        self._detail_labels["Library"].setText(_v(entry.library_name))
        self._detail_labels["Created"].setText(_v(entry.created_at))
        self._detail_labels["Modified"].setText(_v(entry.last_modified))
        self._detail_labels["Fine-tuning"].setText("Yes" if entry.finetuning_available else "No")
        self._detail_labels["Tags"].setText(", ".join(entry.tags[:15]) or "—")
        self._desc_label.setText(entry.description or "")
