from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pixelup.app_config import AppConfig, load_app_config, save_app_config
from pixelup.config import resolve_runtime_dirs
from pixelup.errors import PixelupError
from pixelup.imaging import register_image_plugins
from pixelup.models import KNOWN_MODELS
from pixelup.paths import OutputFormat, default_output_path
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions, run_upscale

UPSCALE_MODELS = tuple(model.name for model in KNOWN_MODELS if model.name != "GFPGANv1.4")


@dataclass(slots=True)
class Job:
    id: int
    input_path: Path
    model: str
    scale: int
    output_path: Path
    status: str = "pending"
    message: str = ""
    warnings: list[str] = field(default_factory=list)


class JobSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(int, bool, str, object, object)


class JobWorker(QObject):
    def __init__(self, job: Job, config: AppConfig) -> None:
        super().__init__()
        self.job = job
        self.config = config
        self.signals = JobSignals()

    @Slot()
    def run(self) -> None:
        warnings: list[str] = []
        try:
            options = _options_for_job(self.job, self.config)
            result = run_upscale(
                options,
                resolve_runtime_dirs(),
                on_progress=lambda phase: self.signals.progress.emit(self.job.id, phase),
                on_tile=lambda done, total: self.signals.progress.emit(
                    self.job.id, f"tile {done}/{total}"
                ),
                on_download=lambda model, done, total: self.signals.progress.emit(
                    self.job.id, _download_text(model, done, total)
                ),
                on_warning=warnings.append,
            )
            sidecar = write_sidecar(
                input_path=self.job.input_path,
                output_path=self.job.output_path,
                options=options,
                result=result,
                warnings=warnings,
            )
            result["sidecar"] = str(sidecar)
            self.signals.finished.emit(self.job.id, True, "done", result, warnings)
        except PixelupError as exc:
            self.signals.finished.emit(self.job.id, False, exc.message, {}, warnings)
        except Exception as exc:
            self.signals.finished.emit(self.job.id, False, f"Unexpected error: {exc}", {}, warnings)


class ImageTab(QWidget):
    enqueue_requested = Signal(object, object, int)
    retry_requested = Signal(object)

    def __init__(self, input_path: Path) -> None:
        super().__init__()
        self.input_path = input_path
        self.jobs: list[Job] = []
        self._rows_by_job: dict[int, int] = {}

        self.model_checks: dict[str, QCheckBox] = {}
        for model in UPSCALE_MODELS:
            check = QCheckBox(model)
            check.setChecked(model == "realesr-general-x4v3")
            self.model_checks[model] = check

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["4", "2"])

        enqueue_button = QPushButton("Enqueue selected")
        enqueue_button.clicked.connect(self._enqueue_selected)
        try_all_button = QPushButton("Try all models")
        try_all_button.clicked.connect(self._try_all_models)
        retry_button = QPushButton("Retry failed")
        retry_button.clicked.connect(lambda: self.retry_requested.emit(self))

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel(self.input_path.name))
        left_layout.addWidget(QLabel(str(self.input_path.parent)))
        left_layout.addSpacing(8)
        left_layout.addWidget(QLabel("Models"))
        model_box = QWidget()
        model_layout = QVBoxLayout(model_box)
        for check in self.model_checks.values():
            model_layout.addWidget(check)
        model_layout.addStretch()
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(model_box)
        left_layout.addWidget(model_scroll, 1)
        left_layout.addWidget(QLabel("Scale"))
        left_layout.addWidget(self.scale_combo)
        left_layout.addWidget(enqueue_button)
        left_layout.addWidget(try_all_button)
        left_layout.addWidget(retry_button)

        self.preview = QLabel("Preview unavailable")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(self.input_path))
        if not pixmap.isNull():
            self.preview.setPixmap(
                pixmap.scaled(
                    420,
                    320,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model", "Scale", "Output", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.preview, 2)
        right_layout.addWidget(self.table, 3)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([260, 700])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def add_jobs(self, jobs: list[Job]) -> None:
        for job in jobs:
            self.jobs.append(job)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows_by_job[job.id] = row
            self.table.setItem(row, 0, QTableWidgetItem(job.model))
            self.table.setItem(row, 1, QTableWidgetItem(f"{job.scale}x"))
            self.table.setItem(row, 2, QTableWidgetItem(job.output_path.name))
            self.table.setItem(row, 3, QTableWidgetItem(job.status))

    def update_job(self, job: Job) -> None:
        row = self._rows_by_job[job.id]
        self.table.item(row, 2).setText(job.output_path.name)
        self.table.item(row, 3).setText(job.message or job.status)
        color = {
            "pending": QColor("#eef5ff"),
            "running": QColor("#fff4cc"),
            "succeeded": QColor("#e8f7e8"),
            "failed": QColor("#ffe8e8"),
        }.get(job.status, QColor("white"))
        for column in range(self.table.columnCount()):
            self.table.item(row, column).setBackground(color)

    def counts(self) -> tuple[int, int, int, int]:
        total = len(self.jobs)
        done = sum(1 for job in self.jobs if job.status == "succeeded")
        failed = sum(1 for job in self.jobs if job.status == "failed")
        running = sum(1 for job in self.jobs if job.status == "running")
        return total, done, failed, running

    def all_succeeded(self) -> bool:
        return bool(self.jobs) and all(job.status == "succeeded" for job in self.jobs)

    def has_active_jobs(self) -> bool:
        return any(job.status in {"pending", "running"} for job in self.jobs)

    def _enqueue_selected(self) -> None:
        models = [name for name, check in self.model_checks.items() if check.isChecked()]
        self.enqueue_requested.emit(self, models, int(self.scale_combo.currentText()))

    def _try_all_models(self) -> None:
        self.enqueue_requested.emit(self, list(UPSCALE_MODELS), int(self.scale_combo.currentText()))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.config = load_app_config()
        self._job_ids = count(1)
        self._threads: dict[int, tuple[QThread, JobWorker]] = {}
        self._active_jobs = 0
        self._tabs_by_path: dict[Path, ImageTab] = {}

        self.setWindowTitle("PixelUp")
        self.resize(1000, 680)
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab_at)
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Drop image files here, or choose File > Open.")
        self._build_menu()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.open_paths(paths)

    def open_paths(self, paths: list[Path]) -> None:
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                continue
            tab = self._tabs_by_path.get(resolved)
            if tab is None:
                tab = ImageTab(resolved)
                tab.enqueue_requested.connect(self._enqueue_jobs)
                tab.retry_requested.connect(self._retry_failed)
                self._tabs_by_path[resolved] = tab
                self.tabs.addTab(tab, resolved.name)
            self.tabs.setCurrentWidget(tab)
            self._update_tab_state(tab)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("&Open images...")
        open_action.triggered.connect(self._open_dialog)
        settings_action = file_menu.addAction("&Settings...")
        settings_action.triggered.connect(self._settings_dialog)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

    def _open_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open images")
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QMessageBox.StandardButton.Ok:
            self.config = dialog.config()
            save_app_config(self.config)
            self.statusBar().showMessage("Settings saved.", 3000)
            self._schedule()

    @Slot(object, object, int)
    def _enqueue_jobs(self, tab: ImageTab, models: list[str], scale: int) -> None:
        if not models:
            QMessageBox.information(self, "PixelUp", "Choose at least one model.")
            return
        jobs: list[Job] = []
        reserved = {job.output_path for job in tab.jobs}
        for model in models:
            output_path = default_output_path(
                tab.input_path,
                model=model,
                scale=scale,
                output_format=self.config.output_format,
                reserved=reserved,
            )
            reserved.add(output_path)
            jobs.append(
                Job(
                    id=next(self._job_ids),
                    input_path=tab.input_path,
                    model=model,
                    scale=scale,
                    output_path=output_path,
                )
            )
        tab.add_jobs(jobs)
        self._update_tab_state(tab)
        self._schedule()

    @Slot(object)
    def _retry_failed(self, tab: ImageTab) -> None:
        reserved = {job.output_path for job in tab.jobs if job.status != "failed"}
        changed = False
        for job in tab.jobs:
            if job.status != "failed":
                continue
            job.output_path = default_output_path(
                tab.input_path,
                model=job.model,
                scale=job.scale,
                output_format=self.config.output_format,
                reserved=reserved,
            )
            reserved.add(job.output_path)
            job.status = "pending"
            job.message = ""
            tab.update_job(job)
            changed = True
        if changed:
            self._update_tab_state(tab)
            self._schedule()

    def _schedule(self) -> None:
        limit = max(1, self.config.max_concurrent_jobs)
        while self._active_jobs < limit:
            next_job = self._next_pending_job()
            if next_job is None:
                return
            tab, job = next_job
            self._start_job(tab, job)

    def _next_pending_job(self) -> tuple[ImageTab, Job] | None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if not isinstance(tab, ImageTab):
                continue
            for job in tab.jobs:
                if job.status == "pending":
                    return tab, job
        return None

    def _start_job(self, tab: ImageTab, job: Job) -> None:
        job.status = "running"
        job.message = "running"
        tab.update_job(job)
        self._update_tab_state(tab)
        self._active_jobs += 1

        thread = QThread(self)
        worker = JobWorker(job, self.config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.progress.connect(self._job_progress)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda job_id=job.id: self._threads.pop(job_id, None))
        self._threads[job.id] = (thread, worker)
        thread.start()

    @Slot(int, str)
    def _job_progress(self, job_id: int, message: str) -> None:
        tab, job = self._find_job(job_id)
        job.message = message
        tab.update_job(job)

    @Slot(int, bool, str, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: str,
        _result: object,
        warnings: object,
    ) -> None:
        tab, job = self._find_job(job_id)
        job.status = "succeeded" if ok else "failed"
        job.message = message
        job.warnings = list(warnings) if isinstance(warnings, list) else []
        tab.update_job(job)
        self._active_jobs = max(0, self._active_jobs - 1)
        self._update_tab_state(tab)
        if self.config.close_tab_on_success and tab.all_succeeded():
            self._close_tab(tab)
        QTimer.singleShot(0, self._schedule)

    def _find_job(self, job_id: int) -> tuple[ImageTab, Job]:
        for tab in self._tabs_by_path.values():
            for job in tab.jobs:
                if job.id == job_id:
                    return tab, job
        raise RuntimeError(f"Unknown job id: {job_id}")

    def _update_tab_state(self, tab: ImageTab) -> None:
        index = self.tabs.indexOf(tab)
        if index < 0:
            return
        total, done, failed, running = tab.counts()
        self.tabs.setTabText(index, tab.input_path.name)
        self.tabs.setTabToolTip(index, f"{done}/{total} done, {failed} failed, {running} running")
        if failed:
            color = QColor("#b00020")
        elif running or any(job.status == "pending" for job in tab.jobs):
            color = QColor("#0057b8")
        elif total and done == total:
            color = QColor("#148a14")
        else:
            color = QColor("black")
        self.tabs.tabBar().setTabTextColor(index, color)

    def _close_tab_at(self, index: int) -> None:
        tab = self.tabs.widget(index)
        if isinstance(tab, ImageTab) and tab.has_active_jobs():
            QMessageBox.information(self, "PixelUp", "This tab still has pending or running jobs.")
            return
        if isinstance(tab, ImageTab):
            self._close_tab(tab)

    def _close_tab(self, tab: ImageTab) -> None:
        index = self.tabs.indexOf(tab)
        if index >= 0:
            self.tabs.removeTab(index)
        self._tabs_by_path.pop(tab.input_path, None)
        tab.deleteLater()


class SettingsDialog(QMessageBox):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setText("PixelUp settings")
        self.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )

        box = QWidget()
        form = QFormLayout(box)
        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setValue(config.max_concurrent_jobs)
        self.close_tabs = QCheckBox("Close successful tabs")
        self.close_tabs.setChecked(config.close_tab_on_success)
        self.format = QComboBox()
        self.format.addItems([item.value for item in OutputFormat])
        self.format.setCurrentText(config.output_format.value)
        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setValue(config.quality)
        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.setValue(config.tile)
        self.device = QComboBox()
        self.device.addItems(["auto", "mps", "cuda", "cpu"])
        self.device.setCurrentText(config.device)
        self.auto_download = QCheckBox("Download missing models")
        self.auto_download.setChecked(config.auto_download)
        form.addRow("Concurrent jobs", self.concurrent)
        form.addRow("", self.close_tabs)
        form.addRow("Output format", self.format)
        form.addRow("Quality", self.quality)
        form.addRow("Tile size", self.tile)
        form.addRow("Device", self.device)
        form.addRow("", self.auto_download)
        self.layout().addWidget(box, 1, 0, 1, self.layout().columnCount())

    def config(self) -> AppConfig:
        return AppConfig(
            max_concurrent_jobs=self.concurrent.value(),
            close_tab_on_success=self.close_tabs.isChecked(),
            output_format=OutputFormat(self.format.currentText()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            device=self.device.currentText(),
            auto_download=self.auto_download.isChecked(),
        )


def _options_for_job(job: Job, config: AppConfig) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=job.input_path,
        output_arg=str(job.output_path),
        model=job.model,
        scale=job.scale,
        tile=config.tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        face_enhance=False,
        denoise_strength=1.0,
        alpha_mode="realesrgan",
        gpu_id=None,
        device=config.device,
        output_format=config.output_format,
        quality=config.quality,
        background="white",
        strip_metadata=False,
        target_profile=None,
        overwrite=False,
        auto_download=config.auto_download,
        download_timeout=600,
        lock_timeout=600,
    )


def _download_text(model: str, done: int, total: int | None) -> str:
    if total:
        return f"download {model} {done * 100 // total}%"
    return f"download {model}"


def main() -> int:
    register_image_plugins()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    paths = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("-")]
    if paths:
        window.open_paths(paths)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
