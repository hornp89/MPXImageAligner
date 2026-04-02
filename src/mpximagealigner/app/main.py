"""
MPX Image Aligner — PyQt6 graphical front-end.

Run with:
    python app/main.py
"""

import sys
import os

# Force a non-interactive matplotlib backend before any pyplot is imported.
# This prevents matplotlib from trying to open a GUI window from a worker thread.
import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QTextEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor

from mpximagealigner.app.worker import AlignmentWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPX Image Aligner")
        self.setMinimumWidth(720)
        self.worker: AlignmentWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(self._make_dirs_group())
        root.addWidget(self._make_params_group())
        root.addWidget(self._make_output_group())
        root.addWidget(self._make_log_group(), stretch=1)
        root.addLayout(self._make_buttons())

    def _make_dirs_group(self) -> QGroupBox:
        group = QGroupBox("Directories")
        form = QFormLayout(group)

        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Required")
        form.addRow("Source directory:", self._browse_row(self.src_edit, "Select source directory"))

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Default: <source>_aligned")
        form.addRow("Output directory:", self._browse_row(self.out_edit, "Select output directory"))

        return group

    def _make_params_group(self) -> QGroupBox:
        group = QGroupBox("Registration Parameters")
        form = QFormLayout(group)

        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)
        
            
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["single", "batch"])
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addSpacing(20)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["affine", "rigid"])
        mode_layout.addWidget(QLabel("Method:"))
        mode_layout.addWidget(self.method_combo)
        mode_layout.addStretch(20)
        
        self.search_ref_check = QCheckBox("Search for best reference")
        mode_layout.addWidget(self.search_ref_check)
        
        form.addRow(mode_container)
        
        self.ref_file_no_spin = QSpinBox()
        self.ref_file_no_spin.setRange(0, 9999)
        form.addRow("Reference file index:", self.ref_file_no_spin)

        self.size_factor_spin = QSpinBox()
        self.size_factor_spin.setRange(1, 32)
        self.size_factor_spin.setValue(4)
        form.addRow("Size factor:", self.size_factor_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1.0)
        form.addRow("Learning rate:", self.lr_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(5)
        form.addRow("Epochs:", self.epochs_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device:", self.device_combo)

        # Tile size with a spin box
        tilesize_container = QWidget()
        tilesize_layout = QHBoxLayout(tilesize_container)
        tilesize_layout.setContentsMargins(0, 0, 0, 0)

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(512, 16384)
        self.tile_size_spin.setSingleStep(512)
        self.tile_size_spin.setValue(4096)
        
        tilesize_layout.addWidget(QLabel("Tile size (px):"))
        tilesize_layout.addWidget(self.tile_size_spin)
        form.addRow("Tile size:", tilesize_container)
        
        self.random_starts_spin = QSpinBox()
        self.random_starts_spin.setRange(1, 100)
        self.random_starts_spin.setValue(24)
        form.addRow("Random starts:", self.random_starts_spin)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(0)
        form.addRow("Random seed:", self.seed_spin)
        

        return group

    def _make_output_group(self) -> QGroupBox:
        group = QGroupBox("Output Options")
        layout = QHBoxLayout(group)

        self.save_loss_check = QCheckBox("Save losses CSV")
        self.save_loss_check.setChecked(True)
        self.save_plot_check = QCheckBox("Save loss plot")
        self.save_plot_check.setChecked(True)

        layout.addWidget(self.save_loss_check)
        layout.addWidget(self.save_plot_check)
        layout.addStretch()
        return group

    def _make_log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setMinimumHeight(200)
        layout.addWidget(self.log_edit)
        return group

    def _make_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.run_btn = QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.on_run)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_edit.clear)

        layout.addWidget(self.run_btn)
        layout.addWidget(self.cancel_btn)
        layout.addStretch()
        layout.addWidget(clear_btn)
        return layout

    # ----------------------------------------------------------------- helpers

    def _browse_row(self, edit: QLineEdit, title: str) -> QWidget:
        """Return a widget containing a QLineEdit and a Browse button side-by-side."""
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(lambda: self._browse(edit, title))
        h.addWidget(btn)
        return container

    def _browse(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            edit.setText(path)

    def _log(self, text: str):
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.insertPlainText(text if text.endswith("\n") else text + "\n")
        self.log_edit.ensureCursorVisible()

    # --------------------------------------------------------------- slots

    @pyqtSlot()
    def on_run(self):
        src = self.src_edit.text().strip()
        if not src or not os.path.isdir(src):
            self._log("ERROR: Please select a valid source directory.")
            return

        out = self.out_edit.text().strip() or None
        device_text = self.device_combo.currentText()
        device = None if device_text == "auto" else device_text

        params = dict(
            src_dir=src,
            out_dir=out,
            ref_file_no=self.ref_file_no_spin.value(),
            mode = self.mode_combo.currentText(),
            method=self.method_combo.currentText(),
            search_ref=self.search_ref_check.isChecked(),
            size_factor=self.size_factor_spin.value(),
            lr=self.lr_spin.value(),
            num_epochs=self.epochs_spin.value(),
            device=device,
            tile_size=self.tile_size_spin.value(),
            plot_show=False,   # always off — no blocking matplotlib window in GUI
            plot_save=self.save_plot_check.isChecked(),
            save_loss=self.save_loss_check.isChecked(),
            random_starts=self.random_starts_spin.value(),
            seed=self.seed_spin.value()
        )


        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = AlignmentWorker(params)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    @pyqtSlot()
    def on_cancel(self):
        if self.worker:
            self.worker.cancel()
            self._log("Cancellation requested — will stop after the current image finishes.")
        self.cancel_btn.setEnabled(False)

    @pyqtSlot(bool, str)
    def on_finished(self, success: bool, message: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if success:
            self._log(f"\n[Done] {message}")
        else:
            self._log(f"\n[Error]\n{message}")


def run_gui() -> None:
    """Launch the MPX Image Aligner GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
