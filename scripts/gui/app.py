import sys
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QComboBox, QTextEdit, QFileDialog, QDialog,
                             QFormLayout, QCheckBox, QSpinBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal

class OpenMVSOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenMVS Arguments")
        self.resize(500, 500)
        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        
        # Pre-defined parameters for each step
        densify_params = {
            "--resolution-level": "2",
            "--max-resolution": "3200",
            "--number-views": "4",
            "--number-views-fuse": "3",
            "--postprocess-dmaps": "7",
            "--iters": "4",
            "--geometric-iters": "2",
            "--estimate-colors": "2",
            "--estimate-normals": "2",
            "--filter-point-cloud": "0"
        }
        
        mesh_params = {
            "--thickness-factor": "1",
            "--quality-factor": "1",
            "--smooth": "2",
            "--close-holes": "30",
            "--remove-spurious": "20",
            "--decimate": "1.0"
        }
        
        texture_params = {
            "--resolution-level": "0",
            "--close-holes": "30",
            "--outlier-threshold": "0.06",
            "--sharpness-weight": "0.5",
            "--cost-smoothness-ratio": "0.1",
            "--global-seam-leveling": "1",
            "--local-seam-leveling": "1",
            "--empty-color": "16744231",
            "--export-type": "obj"
        }

        self.tab_densify, self.densify_inputs = self.create_form_tab(densify_params)
        self.tab_mesh, self.mesh_inputs = self.create_form_tab(mesh_params)
        self.tab_texture, self.texture_inputs = self.create_form_tab(texture_params)

        self.tabs.addTab(self.tab_densify, "DensifyPointCloud")
        self.tabs.addTab(self.tab_mesh, "ReconstructMesh")
        self.tabs.addTab(self.tab_texture, "TextureMesh")
        
        self.layout.addWidget(self.tabs)

        self.close_btn = QPushButton("Save & Close")
        self.close_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.close_btn)

    def create_form_tab(self, params_dict):
        widget = QWidget()
        form_layout = QFormLayout(widget)
        
        inputs = {}
        for param, default_val in params_dict.items():
            line_edit = QLineEdit()
            line_edit.setText(default_val)
            form_layout.addRow(param, line_edit)
            inputs[param] = line_edit
            
        return widget, inputs

    def get_args_string(self, inputs_dict):
        args = []
        for param, line_edit in inputs_dict.items():
            val = line_edit.text().strip()
            if val:
                args.append(f"{param} {val}")
        return " ".join(args)

    def get_all_args(self):
        return {
            "densify": self.get_args_string(self.densify_inputs),
            "mesh": self.get_args_string(self.mesh_inputs),
            "texture": self.get_args_string(self.texture_inputs)
        }


class AdvancedOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DIM Advanced Options")
        self.layout = QFormLayout(self)
        self.resize(500, 300)

        # DIM Options
        self.tiling_combo = QComboBox()
        self.tiling_combo.addItems(["none", "preselection", "preselection_affine_transform", "grid", "exhaustive"])
        self.layout.addRow("Tiling Strategy:", self.tiling_combo)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["matching_lowres", "bruteforce", "sequential", "retrieval", "custom_pairs", "covisibility"])
        self.layout.addRow("Matching Strategy:", self.strategy_combo)

        self.overlap_spinbox = QSpinBox()
        self.overlap_spinbox.setRange(1, 100)
        self.overlap_spinbox.setValue(1)
        self.layout.addRow("Sequential Overlap:", self.overlap_spinbox)

        self.camera_options_input = QLineEdit()
        self.camera_options_btn = QPushButton("Browse")
        self.camera_options_btn.clicked.connect(self.browse_camera_options)
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(self.camera_options_input)
        cam_layout.addWidget(self.camera_options_btn)
        self.layout.addRow("Camera Options YAML:", cam_layout)

        self.force_checkbox = QCheckBox("Force overwrite (-f)")
        self.force_checkbox.setChecked(True)
        self.layout.addRow("", self.force_checkbox)

        self.verbose_checkbox = QCheckBox("Verbose (-v)")
        self.verbose_checkbox.setChecked(True)
        self.layout.addRow("", self.verbose_checkbox)

        self.graph_checkbox = QCheckBox("Show Match Graph (-g)")
        self.layout.addRow("", self.graph_checkbox)

        self.close_btn = QPushButton("Save & Close")
        self.close_btn.clicked.connect(self.accept)
        self.layout.addRow("", self.close_btn)

    def browse_camera_options(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Camera Options YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)")
        if file:
            self.camera_options_input.setText(file)

    def get_options(self):
        opts = {
            "tiling": self.tiling_combo.currentText(),
            "strategy": self.strategy_combo.currentText(),
            "overlap": self.overlap_spinbox.value(),
            "camera_options": self.camera_options_input.text(),
            "force": self.force_checkbox.isChecked(),
            "verbose": self.verbose_checkbox.isChecked(),
            "graph": self.graph_checkbox.isChecked()
        }
        return opts


class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, dim_cmd, mvs_cmd, parent=None):
        super().__init__(parent)
        self.dim_cmd = dim_cmd
        self.mvs_cmd = mvs_cmd
        self.is_running = True

    def run(self):
        try:
            cwd_path = str(Path(__file__).resolve().parent.parent.parent)

            # 1. Run Deep Image Matching
            self.log_signal.emit(f"=== RUNNING DEEP IMAGE MATCHING ===\nCommand: {' '.join(self.dim_cmd)}\n")
            process1 = subprocess.Popen(self.dim_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, cwd=cwd_path)
            try:
                process1.stdin.write("yes\n")
                process1.stdin.flush()
                process1.stdin.close()
            except Exception:
                pass
            
            for line in iter(process1.stdout.readline, ''):
                if not self.is_running:
                    process1.terminate()
                    break
                if line:
                    self.log_signal.emit(line)
            
            process1.stdout.close()
            retcode1 = process1.wait()
            
            if retcode1 != 0 and self.is_running:
                self.error_signal.emit(f"Deep Image Matching failed with exit code {retcode1}")
                return

            if not self.is_running:
                self.log_signal.emit("Pipeline stopped by user.\n")
                return

            # 2. Run OpenMVS Pipeline
            self.log_signal.emit(f"\n=== RUNNING OPENMVS PIPELINE ===\nCommand: {' '.join(self.mvs_cmd)}\n")
            process2 = subprocess.Popen(self.mvs_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd_path)
            
            for line in iter(process2.stdout.readline, ''):
                if not self.is_running:
                    process2.terminate()
                    break
                if line:
                    self.log_signal.emit(line)
            
            process2.stdout.close()
            retcode2 = process2.wait()

            if retcode2 != 0 and self.is_running:
                self.error_signal.emit(f"OpenMVS Pipeline failed with exit code {retcode2}")
                return

            self.log_signal.emit("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIM")
        self.resize(900, 750)

        self.advanced_dialog = AdvancedOptionsDialog(self)
        self.mvs_dialog = OpenMVSOptionsDialog(self)
        self.worker = None

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        form_layout = QFormLayout()

        # Images Dir
        self.images_input = QLineEdit()
        self.images_input.setPlaceholderText("Select folder containing input images")
        self.images_btn = QPushButton("Browse")
        self.images_btn.clicked.connect(lambda: self.browse_folder(self.images_input))
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.images_input)
        img_layout.addWidget(self.images_btn)
        form_layout.addRow("Images Directory:", img_layout)

        # Output Dir
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select or create folder for output results")
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(lambda: self.browse_folder(self.output_input))
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.output_input)
        out_layout.addWidget(self.output_btn)
        form_layout.addRow("Output Directory:", out_layout)

        # OpenMVS Bin Dir
        self.mvs_input = QLineEdit()
        self.mvs_input.setText(r"C:\Users\stroc\Desktop\MVS\OpenMVS\dependencies")
        self.mvs_btn = QPushButton("Browse")
        self.mvs_btn.clicked.connect(lambda: self.browse_folder(self.mvs_input))
        mvs_layout = QHBoxLayout()
        mvs_layout.addWidget(self.mvs_input)
        mvs_layout.addWidget(self.mvs_btn)
        form_layout.addRow("OpenMVS Binaries Dir:", mvs_layout)

        # Pipeline Method
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItems([
            "sift+kornia_matcher", "superpoint+superglue", 
            "disk+lightglue", "loftr"
        ])
        form_layout.addRow("Pipeline Method:", self.pipeline_combo)

        # Quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["highest", "high", "medium", "low", "lowest"])
        self.quality_combo.setCurrentText("high")
        form_layout.addRow("Quality:", self.quality_combo)

        # OpenMVS Outputs
        outputs_layout = QHBoxLayout()
        self.chk_cloud = QCheckBox("Dense Point Cloud")
        self.chk_cloud.setChecked(True)
        self.chk_cloud.setEnabled(False) # Always true
        self.chk_mesh = QCheckBox("Mesh / Model")
        self.chk_texture = QCheckBox("Textured Mesh")
        outputs_layout.addWidget(self.chk_cloud)
        outputs_layout.addWidget(self.chk_mesh)
        outputs_layout.addWidget(self.chk_texture)
        form_layout.addRow("OpenMVS Outputs:", outputs_layout)

        main_layout.addLayout(form_layout)

        # Advanced options buttons
        opts_layout = QHBoxLayout()
        self.dim_adv_btn = QPushButton("DIM Options ⚙️")
        self.dim_adv_btn.clicked.connect(self.advanced_dialog.exec_)
        self.mvs_adv_btn = QPushButton("OpenMVS Options ⚙️")
        self.mvs_adv_btn.clicked.connect(self.mvs_dialog.exec_)
        opts_layout.addWidget(self.dim_adv_btn)
        opts_layout.addWidget(self.mvs_adv_btn)
        main_layout.addLayout(opts_layout)

        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 12px; font-size: 14px;")
        self.start_btn.clicked.connect(self.start_pipeline)
        
        control_layout.addWidget(self.start_btn)
        main_layout.addLayout(control_layout)

        # Log viewer
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 12px;")
        main_layout.addWidget(QLabel("Process Logs:"))
        main_layout.addWidget(self.log_viewer)

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)
            if line_edit == self.images_input:
                self.check_image_resolution(folder)

    def check_image_resolution(self, folder):
        import glob
        from PIL import Image

        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.tif', '*.TIF', '*.tiff', '*.TIFF'):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        
        if not image_files:
            return
            
        try:
            with Image.open(image_files[0]) as img:
                w, h = img.size
                if max(w, h) > 5000:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Question)
                    msg.setWindowTitle("High Resolution Detected")
                    msg.setText(f"Wykryto duże zdjęcia (np. {w}x{h} px).")
                    msg.setInformativeText(
                        "Przetwarzanie bardzo dużych zdjęć może być bardzo wolne lub powodować błędy braku pamięci.\n\n"
                        "Czy chcesz teraz zmniejszyć wszystkie zdjęcia do max 4500 px na dłuższym boku?\n"
                        "(Pliki zostaną nadpisane w miejscu)"
                    )
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    msg.setDefaultButton(QMessageBox.Yes)
                    result = msg.exec_()

                    if result == QMessageBox.Yes:
                        self.resize_images(image_files, max_size=4500)
        except Exception:
            pass

    def resize_images(self, image_files, max_size=4500):
        from PIL import Image
        errors = []
        for path in image_files:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        resized = img.resize((new_w, new_h), Image.LANCZOS)
                        resized.save(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        
        if errors:
            QMessageBox.warning(self, "Błąd resize", "Nie udało się zmniejszyć:\n" + "\n".join(errors))
        else:
            QMessageBox.information(self, "Gotowe", f"Zmniejszono {len(image_files)} zdjęć do max {max_size}px.")

    def append_log(self, text):
        self.log_viewer.insertPlainText(text)
        self.log_viewer.verticalScrollBar().setValue(self.log_viewer.verticalScrollBar().maximum())

    def start_pipeline(self):
        images_dir = self.images_input.text().strip()
        output_dir = self.output_input.text().strip()
        mvs_bin = self.mvs_input.text().strip()

        if not images_dir or not output_dir or not mvs_bin:
            self.append_log("ERROR: Please specify Images Directory, Output Directory, and OpenMVS Binaries Directory.\n")
            return

        pipeline_method = self.pipeline_combo.currentText()
        quality = self.quality_combo.currentText()
        dim_opts = self.advanced_dialog.get_options()
        mvs_opts = self.mvs_dialog.get_all_args()

        # Build DIM command
        dim_cmd = [
            sys.executable, "src/deep_image_matching/__main__.py",
            "--images", images_dir,
            "--outs", output_dir,
            "--pipeline", pipeline_method,
            "--quality", quality,
            "--tiling", dim_opts["tiling"],
            "--strategy", dim_opts["strategy"]
        ]

        if dim_opts["strategy"] == "sequential":
            dim_cmd.extend(["--overlap", str(dim_opts["overlap"])])
            
        if dim_opts["camera_options"]:
            dim_cmd.extend(["--camera_options", dim_opts["camera_options"]])
        
        if dim_opts["force"]:
            dim_cmd.append("--force")
        if dim_opts["verbose"]:
            dim_cmd.append("--verbose")
        if dim_opts["graph"]:
            dim_cmd.append("--graph")

        # Build MVS command
        mvs_cmd = [
            sys.executable, "scripts/dim_to_openmvs.py",
            "--dim_output", output_dir,
            "--images", images_dir,
            "--openmvs_bin", mvs_bin
        ]
        
        # MVS arguments
        if mvs_opts["densify"]:
            mvs_cmd.extend(["--densify_args", mvs_opts["densify"]])
            
        if self.chk_mesh.isChecked():
            mvs_cmd.append("--run_mesh")
            if mvs_opts["mesh"]:
                mvs_cmd.extend(["--mesh_args", mvs_opts["mesh"]])
                
        if self.chk_texture.isChecked():
            mvs_cmd.append("--run_texture")
            if mvs_opts["texture"]:
                mvs_cmd.extend(["--texture_args", mvs_opts["texture"]])

        self.log_viewer.clear()
        self.start_btn.setEnabled(False)

        self.worker = PipelineWorker(dim_cmd, mvs_cmd)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_pipeline_finished)
        self.worker.error_signal.connect(self.on_pipeline_error)
        self.worker.start()

    def stop_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.append_log("\nStopping pipeline...\n")
            self.worker.stop()
            self.worker.wait()
        self.on_pipeline_finished()

    def on_pipeline_finished(self):
        self.start_btn.setEnabled(True)

    def on_pipeline_error(self, err_msg):
        self.append_log(f"\n[ERROR] {err_msg}\n")
        self.on_pipeline_finished()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
