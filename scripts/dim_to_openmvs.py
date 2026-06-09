import os
import subprocess
import shlex
import argparse
import sys
import glob
import time
import threading
from pathlib import Path

def tail_log_file(log_path, stop_event):
    """Thread: prints new lines from log_path until stop_event is set."""
    while not os.path.exists(log_path):
        if stop_event.is_set():
            return
        time.sleep(0.2)
    with open(log_path, 'r', errors='replace') as f:
        while True:
            line = f.readline()
            if line:
                print(line, end='', flush=True)
            elif stop_event.is_set():
                # Drain remaining lines before exiting
                rest = f.read()
                if rest:
                    print(rest, end='', flush=True)
                return
            else:
                time.sleep(0.1)

def run_command(command, cwd=None, allow_fail=False):
    print(f"\n[{' '.join(command)}]", flush=True)

    log_dir = cwd or os.getcwd()
    exe_name = Path(command[0]).stem
    before_logs = set(glob.glob(os.path.join(log_dir, f"{exe_name}*.log")))

    # Find the log file that will be created (OpenMVS names it with timestamp)
    stop_event = threading.Event()
    new_log_path = [None]  # mutable container for thread

    def find_and_tail():
        deadline = time.time() + 10  # wait up to 10s for log to appear
        while time.time() < deadline and not stop_event.is_set():
            current = set(glob.glob(os.path.join(log_dir, f"{exe_name}*.log")))
            new = current - before_logs
            if new:
                new_log_path[0] = sorted(new)[0]
                tail_log_file(new_log_path[0], stop_event)
                return
            time.sleep(0.2)

    tail_thread = threading.Thread(target=find_and_tail, daemon=True)
    tail_thread.start()

    try:
        kwargs = {'stdin': subprocess.DEVNULL}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        subprocess.run(command, cwd=cwd, check=True, **kwargs)
        stop_event.set()
        tail_thread.join(timeout=3)
    except subprocess.CalledProcessError as e:
        stop_event.set()
        tail_thread.join(timeout=3)
        if allow_fail:
            print(f"[WARNING] Command returned non-zero exit code {e.returncode} (continuing)", flush=True)
        else:
            print(f"[ERROR] Command failed with exit code {e.returncode}", flush=True)
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automate deep-image-matching to OpenMVS dense point cloud pipeline.")
    parser.add_argument("--dim_output", required=True, help="Path to deep-image-matching results folder (e.g., results_superpoint...)")
    parser.add_argument("--images", required=True, help="Path to original images folder")
    parser.add_argument("--openmvs_bin", default=r"C:\Users\stroc\Desktop\MVS\OpenMVS\dependencies", help="Path to OpenMVS binaries directory")
    parser.add_argument("--densify_args", default="", help="Additional arguments for DensifyPointCloud, passed as a string (e.g., '--sub-resolution-levels 2 --number-views 8')")
    parser.add_argument("--run_mesh", action="store_true", help="Run ReconstructMesh after DensifyPointCloud")
    parser.add_argument("--mesh_args", default="", help="Additional arguments for ReconstructMesh")
    parser.add_argument("--run_texture", action="store_true", help="Run TextureMesh after ReconstructMesh")
    parser.add_argument("--texture_args", default="", help="Additional arguments for TextureMesh")
    
    args = parser.parse_args()

    dim_output = Path(args.dim_output).resolve()
    images_dir = Path(args.images).resolve()
    openmvs_bin = Path(args.openmvs_bin).resolve()

    sparse_dir = dim_output / "reconstruction" / "0"
    undistorted_dir = dim_output / "undistorted"
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    colmap_cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(undistorted_dir),
        "--output_type", "COLMAP"
    ]
    run_command(colmap_cmd, allow_fail=True)

    # COLMAP sometimes crashes on Windows after copying images (exit 0xC0000005)
    # but the output is still usable — check if images were produced
    undistorted_images_dir = undistorted_dir / "images"
    if not undistorted_images_dir.exists() or not any(undistorted_images_dir.iterdir()):
        print("Error: COLMAP undistortion failed — no output images found.")
        exit(1)
    else:
        print(f"[OK] Undistorted images found in {undistorted_images_dir}")


    # Generate scene.mvs
    scene_mvs = undistorted_dir / "scene.mvs"
    undistorted_images = undistorted_dir / "images"
    
    interface_colmap_exe = openmvs_bin / "InterfaceCOLMAP.exe"
    interface_cmd = [
        str(interface_colmap_exe),
        "-i", str(undistorted_dir),
        "-o", str(scene_mvs),
        "--image-folder", str(undistorted_images)
    ]
    run_command(interface_cmd)

    
    dense_out = undistorted_dir / "scene_dense.mvs"

    # Diagnostics
    if scene_mvs.exists():
        print(f"[OK] scene.mvs size: {scene_mvs.stat().st_size} bytes", flush=True)
    else:
        print(f"[ERROR] scene.mvs NOT FOUND at {scene_mvs}", flush=True)
        sys.exit(1)

    # Clean up previous DensifyPointCloud output to avoid conflicts
    for old_file in list(undistorted_dir.glob("scene_dense*")) + list(undistorted_dir.glob("depth*.dmap")) + list(undistorted_dir.glob("depth*.cmap")) + list(undistorted_dir.glob("depth*.tmp")):
        try:
            old_file.unlink()
            print(f"[cleanup] Removed: {old_file.name}", flush=True)
        except Exception as ex:
            print(f"[cleanup] Could not remove {old_file.name}: {ex}", flush=True)

    densify_exe = openmvs_bin / "DensifyPointCloud.exe"
    densify_cmd = [
        str(densify_exe),
        "-i", str(scene_mvs),
        "-o", str(dense_out),
        "-w", str(undistorted_dir)
    ]
    if args.densify_args:
        densify_cmd.extend(shlex.split(args.densify_args))
    run_command(densify_cmd)


    if args.run_mesh:
        mesh_out = undistorted_dir / "scene_dense_mesh.mvs"
        reconstruct_exe = openmvs_bin / "ReconstructMesh.exe"
        mesh_cmd = [
            str(reconstruct_exe),
            "-i", str(dense_out),
            "-o", str(mesh_out),
            "-w", str(undistorted_dir)
        ]
        if args.mesh_args:
            mesh_cmd.extend(shlex.split(args.mesh_args))
        run_command(mesh_cmd)

        if args.run_texture:
            texture_out = undistorted_dir / "scene_dense_mesh_texture.mvs"
            texture_exe = openmvs_bin / "TextureMesh.exe"
            texture_cmd = [
                str(texture_exe),
                "-i", str(mesh_out),
                "-o", str(texture_out),
                "-w", str(undistorted_dir)
            ]
            if args.texture_args:
                texture_cmd.extend(shlex.split(args.texture_args))
            run_command(texture_cmd)



if __name__ == "__main__":
    main()
