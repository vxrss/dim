import sys
from pathlib import Path

sys.path.append(Path(__file__))

import Metashape

from .ms_utils import cameras_from_bundler, create_new_project, import_markers


def project_from_bundler(
    project_path: Path,
    images_dir: Path,
    bundler_file_path: Path,
    bundler_im_list: Path,
    marker_image_path: Path = None,
    marker_world_path: Path = None,
    marker_file_columns: str = "noxyz",
    prm_to_optimize: dict = {},
    chunk_label: str = None,
):
    image_list = list(images_dir.glob("*"))
    images = [str(x) for x in image_list if x.is_file()]

    project_path = Path(project_path)

    if project_path.exists():
        # Open existing project and add a new chunk
        doc = Metashape.Document()
        doc.open(str(project_path), read_only=False, ignore_lock=True)
        chunk = doc.addChunk()
        # Auto-generate chunk label from bundler filename if not provided
        label = chunk_label or bundler_file_path.parent.parent.name
        chunk.label = label
        print(f"Opened existing project, added new chunk: '{label}'")
    else:
        # Create a brand new project
        doc = create_new_project(str(project_path), read_only=False)
        chunk = doc.chunk
        if chunk_label:
            chunk.label = chunk_label
        print(f"Created new project: '{project_path}'")

    # Add photos to chunk
    chunk.addPhotos(images)
    cameras_from_bundler(
        chunk=chunk,
        fname=bundler_file_path,
        image_list=bundler_im_list,
    )

    # Zablokuj kalibrację sensorów PRZED alignCameras:
    # sensor.fixed_calibration = True sprawia, że Metashape NIE OBLICZA fx/fy
    # ani dystorsji podczas wyrównania — używa wartości z bundler.out jako stałych.
    # Bez tego alignCameras szacuje te parametry swobodnie, co niszczy łączenie.
    for sensor in chunk.sensors:
        calib = sensor.calibration
        # Wyzeruj dystorsję — nie chcemy jej w ogóle
        calib.k1 = 0.0
        calib.k2 = 0.0
        calib.k3 = 0.0
        calib.k4 = 0.0
        calib.p1 = 0.0
        calib.p2 = 0.0
        calib.b1 = 0.0
        calib.b2 = 0.0
        sensor.user_calib = calib
        # Zablokuj f i dystorsję — tylko cx/cy pozostaną wolne w optimizeCameras
        sensor.fixed_calibration = True

    # save project
    doc.read_only = False
    doc.save()

    # Import markers image coordinates
    if marker_image_path is not None:
        import_markers(
            marker_image_file=marker_image_path,
            chunk=chunk,
        )

    # Import markers world coordinates
    if marker_world_path is not None:
        chunk.importReference(
            path=str(marker_world_path),
            format=Metashape.ReferenceFormatCSV,
            delimiter=",",
            skip_rows=1,
            columns=marker_file_columns,
        )

    # Reset camera transforms so alignCameras computes poses from scratch
    # (bundler.out has dummy identity poses that cause degenerate triangulation)
    for camera in chunk.cameras:
        camera.transform = None

    # Align cameras using imported tie points (computes poses + triangulates 3D)
    # This does NOT re-do feature matching - it uses already imported tie points
    chunk.alignCameras(reset_alignment=True)

    # optimize camera alignment
    if prm_to_optimize:
        chunk.optimizeCameras(
            fit_f=prm_to_optimize["f"],
            fit_cx=prm_to_optimize["cx"],
            fit_cy=prm_to_optimize["cy"],
            fit_k1=prm_to_optimize["k1"],
            fit_k2=prm_to_optimize["k2"],
            fit_k3=prm_to_optimize["k3"],
            fit_k4=prm_to_optimize["k4"],
            fit_p1=prm_to_optimize["p1"],
            fit_p2=prm_to_optimize["p2"],
            fit_b1=prm_to_optimize["b1"],
            fit_b2=prm_to_optimize["b2"],
            tiepoint_covariance=prm_to_optimize["tiepoint_covariance"],
        )

    # Export cameras and gcp residuals

    # save project
    doc.read_only = False
    doc.save()


if __name__ == "__main__":
    root_dir = Path("/home/francesco/casalbagliano/subset_B")

    # name = "casalbagliano_superpoint+lightglue_bruteforce"
    name = "results_superpoint+lightglue_bruteforce_quality_medium_success"

    images_dir = root_dir / "images"
    marker_image_path = root_dir / "metashape" / "subset_full_markers.txt"
    marker_world_path = root_dir / "metashape" / "subset_full_markers_world.txt"
    marker_file_columns = "noxyz"

    sfm_dir = root_dir / name
    project_path = sfm_dir / "metashape" / f"{name}.psx"
    bundler_file_path = sfm_dir / "reconstruction" / "bundler.out"
    bundler_im_list = sfm_dir / "reconstruction" / "bundler_list.txt"

    prm_to_optimize = {
        "f": False,   # nie optymalizuj ogniskowej (fx/fy)
        "cx": True,
        "cy": True,
        "k1": False,  # nie optymalizuj dystorsji
        "k2": False,
        "k3": False,
        "k4": False,
        "p1": False,
        "p2": False,
        "b1": False,
        "b2": False,
        "tiepoint_covariance": True,
    }
    project_from_bundler(
        project_path=project_path,
        images_dir=images_dir,
        bundler_file_path=bundler_file_path,
        bundler_im_list=bundler_im_list,
        marker_image_path=marker_image_path,
        marker_world_path=marker_world_path,
        marker_file_columns=marker_file_columns,
        prm_to_optimize=prm_to_optimize,
    )
