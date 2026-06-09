import pycolmap
import argparse
import networkx as nx
import sqlite3
import numpy as np
import struct

from pathlib import Path


def open_database(colmap_database_path: Path):
    """Open COLMAP SQLite database and return connection."""
    conn = sqlite3.connect(str(colmap_database_path))
    return conn


def read_all_images_sql(conn):
    """Read all images from the database via SQL."""
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, camera_id, name FROM images")
    images = []
    for image_id, camera_id, name in cursor.fetchall():
        images.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
        })
    return images


def read_all_cameras_sql(conn):
    """Read all cameras from the database via SQL."""
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
    cameras = {}
    for camera_id, model, width, height, params_blob in cursor.fetchall():
        params = np.frombuffer(params_blob, dtype=np.float64)
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "params": params,
        }
    return cameras


def read_keypoints_sql(conn, image_id):
    """Read keypoints for an image from the database via SQL."""
    cursor = conn.cursor()
    cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    if row is None:
        return np.zeros((0, 2), dtype=np.float32)
    rows, cols, data_blob = row
    if data_blob is None or rows == 0:
        return np.zeros((0, 2), dtype=np.float32)
    keypoints = np.frombuffer(data_blob, dtype=np.float32).reshape(rows, cols)
    return keypoints


def read_all_matches_sql(conn):
    """Read all matches from the 'matches' table via SQL.
    
    Workaround for COLMAP 3.13 bug where read_all_matches() throws:
    'Tried to call pure virtual function Database::ReadAllMatchesBlob'
    """
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, rows, cols, data FROM matches WHERE rows > 0")

    all_pairs = []
    all_matches = []
    for pair_id, rows, cols, data_blob in cursor.fetchall():
        if data_blob is not None and rows > 0:
            matches = np.frombuffer(data_blob, dtype=np.uint32).reshape(rows, cols)
            all_pairs.append(pair_id)
            all_matches.append(matches)
    return all_pairs, all_matches


def read_all_two_view_geometry_matches_sql(conn):
    """Read inlier matches from the 'two_view_geometries' table via SQL.
    
    Used for matchers like LoFTR that store matches directly as verified
    two-view geometries, skipping the raw 'matches' table.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0")

    all_pairs = []
    all_matches = []
    for pair_id, rows, cols, data_blob in cursor.fetchall():
        if data_blob is not None and rows > 0:
            matches = np.frombuffer(data_blob, dtype=np.uint32).reshape(rows, cols)
            all_pairs.append(pair_id)
            all_matches.append(matches)
    return all_pairs, all_matches


def build_tracks(
    colmap_database_path: Path,
    use_two_view_geometries: bool = False,
    ):
    conn = open_database(colmap_database_path)

    if use_two_view_geometries:
        all_pairs, all_matches = read_all_two_view_geometry_matches_sql(conn)
    else:
        all_pairs, all_matches = read_all_matches_sql(conn)

    G = nx.Graph()
    for pairs, matches in zip(all_pairs, all_matches):
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pairs)
        #if geom.config != pycolmap.TwoViewGeometryConfig.SUCCESS:
        #    continue
        for (i1, i2) in matches:
            G.add_edge((image_id1, i1), (image_id2, i2))

    tracks = list(nx.connected_components(G))
    
    return conn, tracks


def export_to_bundler(
    conn,
    tracks,
    output_dir: Path,
    ):

    images = read_all_images_sql(conn)
    num_images = len(images)
    num_points = len(tracks)

    cameras = read_all_cameras_sql(conn)
    
    out_file = open(output_dir / "bundler.out", 'w')
    image_out_file = open(output_dir / "bundler.out.list.txt", 'w')

    out_file.write(f"# Bundle file v0.3\n")
    out_file.write(f"{num_images} {num_points}\n")

    # Camera Parameter Blocks
    for i,image in enumerate(images):
        image_id = image["image_id"]
        camera_id = image["camera_id"]
        name = image["name"]
        image_out_file.write(f"{name}\n")

        # Read real camera parameters from database
        camera = cameras[camera_id]
        params = camera["params"]
        f = params[0]           # focal length
        k1 = params[3] if len(params) > 3 else 0.0  # radial distortion k1
        k2 = 0.0               # k2 (not stored in SIMPLE_RADIAL)

        #out_file.write(f"# Camera {i}\n")
        out_file.write(f"{f} {k1} {k2}\n")    # f k1 k2
        out_file.write("1.0 0.0 0.0\n") # R11 R12 R13
        out_file.write("0.0 1.0 0.0\n") # R21 R22 R23
        out_file.write("0.0 0.0 1.0\n") # R31 R32 R33
        out_file.write("0 0 0\n")       # t1 t2 t3

    # 3D Point Blocks
    for t,track in enumerate(tracks):
        #out_file.write(f"# Point {t}\n")
        track_length = len(track)
        out_file.write("0.0 0.0 0.0\n")  # X Y Z
        out_file.write("250 250 250\n")  # R G B
        out_file.write(f"{track_length} ")
        for (image_id, keypoint_idx) in track:
            # Find camera for this image
            cursor = conn.cursor()
            cursor.execute("SELECT camera_id FROM images WHERE image_id = ?", (image_id,))
            camera_id = cursor.fetchone()[0]
            camera = cameras[camera_id]
            width = camera["width"]
            height = camera["height"]

            keypoints = read_keypoints_sql(conn, image_id)
            keypoint = keypoints[keypoint_idx]
            x, y = keypoint[0], keypoint[1]
            
            x = x - width/2          
            y = height/2 - y
            out_file.write(f"{image_id-1} {keypoint_idx} {x} {y} ")
        out_file.write("\n")

    out_file.close()
    image_out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export COLMAP database model to Bundler format")
    parser.add_argument("-d", "--colmap_database_path", type=Path, help="Path to the COLMAP database file", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Path to the output dir", required=True)
    parser.add_argument("--loftr", action="store_true", help="Use two_view_geometries table instead of matches (for LoFTR and similar dense matchers)")
    args = parser.parse_args()  

    conn, tracks = build_tracks(args.colmap_database_path, use_two_view_geometries=args.loftr)
    export_to_bundler(conn, tracks, args.output_dir)
    conn.close()