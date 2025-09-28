import cv2
import numpy as np
import csv
import pandas as pd
from ultralytics import YOLO
from math import sin, cos, radians, log, tan, pi
import time
import threading
import os

# ---- Bokeh (older versions) ----
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, WMTSTileSource
from bokeh.server.server import Server

# Open http://localhost:5006/live_map 


# ---------------------- CONFIG ----------------------
VIDEO_FILE = "raw_flight.avi"   # <-- set your video path here
TELEMETRY_FILE = "telemetry_log.csv"  # path to your CSV

# Camera intrinsics (Intel RealSense D455)
intrinsic_matrix = np.array([[650.90417285, 0, 318.97278063],
                             [0, 651.45358764, 236.01686148],
                             [0, 0, 1]])
image_center = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]])
focal_len_px = (intrinsic_matrix[0, 0] + intrinsic_matrix[1, 1]) / 2

# YOLO model
model = YOLO("yolov5su.pt")

# ---------------------- TELEMETRY ----------------------
telemetry_df = pd.read_csv(TELEMETRY_FILE).set_index("Frame")
if telemetry_df.empty:
    raise RuntimeError("telemetry_log.csv is empty or not readable.")

# Compute initial map center from first telemetry row
first_frame = int(telemetry_df.index.min())
first_row = telemetry_df.loc[first_frame]

# ---------------------- HELPERS ----------------------
def offset_gps(lat, lon, distance_m, heading_deg):
    """Offset (lat, lon) by distance_m along heading_deg (0°=North, clockwise)."""
    R = 6378137.0
    heading_rad = radians(heading_deg)
    dlat = (distance_m * cos(heading_rad)) / R
    dlon = (distance_m * sin(heading_rad)) / (R * cos(radians(lat)))
    new_lat = lat + dlat * (180 / np.pi)
    new_lon = lon + dlon * (180 / np.pi)
    return new_lat, new_lon

# Web Mercator conversion (EPSG:3857)
WM_FACTOR = 20037508.34 / 180.0

def latlon_to_mercator(lat, lon):
    x = lon * WM_FACTOR
    y = log(tan((90.0 + lat) * pi / 360.0)) * (20037508.34 / pi)
    return x, y

# ---------------------- BOKEH LIVE MAP (old-version compatible) ----------------------

# Globals set by the Bokeh app when ready
_bokeh_ready = threading.Event()
_update_funcs = {"drone": None, "person": None, "recenter": None}

# Initial center for map ranges (from first telemetry sample)
init_x, init_y = latlon_to_mercator(first_row["Lat"], first_row["Lon"])


def bkapp(doc):
    # Data sources
    drone_source = ColumnDataSource(data=dict(x=[], y=[]))
    person_source = ColumnDataSource(data=dict(x=[], y=[]))

    # ESRI satellite tile layer
    tile = WMTSTileSource(url=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{Z}/{Y}/{X}"
    ))

    # Figure in Web Mercator
    p = figure(
        x_axis_type="mercator", y_axis_type="mercator",
        x_range=(init_x - 500, init_x + 500),
        y_range=(init_y - 500, init_y + 500),
        title="Drone & Person Tracking (ESRI Satellite)",
        tools="pan,wheel_zoom,reset,save",
        sizing_mode="stretch_both",
    )

    p.add_tile(tile)

    # Drone path (line) and person detections (markers)
    p.line(x='x', y='y', source=drone_source, line_width=2, color='blue', legend_label="Drone Path")
    p.circle(x='x', y='y', source=person_source, size=8, color='red', alpha=0.9, legend_label="Detections")
    p.legend.location = "top_left"

    # Thread-safe updaters
    def thread_safe_stream(src, data):
        doc.add_next_tick_callback(lambda: src.stream(data))

    def update_drone_xy(x, y):
        thread_safe_stream(drone_source, dict(x=[x], y=[y]))

    def update_person_xy(x, y):
        thread_safe_stream(person_source, dict(x=[x], y=[y]))

    def recenter_on(x, y, pad=300):
        def _recenter():
            p.x_range.start = x - pad
            p.x_range.end = x + pad
            p.y_range.start = y - pad
            p.y_range.end = y + pad
        doc.add_next_tick_callback(_recenter)

    # Expose to outer thread
    _update_funcs["drone"] = update_drone_xy
    _update_funcs["person"] = update_person_xy
    _update_funcs["recenter"] = recenter_on
    _bokeh_ready.set()

    doc.add_root(p)


def start_bokeh_server():
    # Allow opening from localhost; add your host if needed
    server = Server({'/live_map': bkapp}, num_procs=1, port=5006,
                    allow_websocket_origin=["localhost:5006", "127.0.0.1:5006"])
    server.start()
    print("[INFO] Bokeh live map running at http://localhost:5006/live_map")
    server.io_loop.start()


# Launch the Bokeh server in a background thread
threading.Thread(target=start_bokeh_server, daemon=True).start()

# Optionally wait a moment for the Bokeh doc to be ready
_bokeh_ready.wait(timeout=5)

# ---------------------- VIDEO + YOLO PROCESSING ----------------------

def process_video():
    # Outputs
    csv_file = open('detections.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame", "ID", "Altitude_m", "GroundDist_m",
        "Person_Lat", "Person_Lon", "Drone_Lat", "Drone_Lon"
    ])

    weighted_csv_file = open('weighted_detections.csv', 'w', newline='')
    weighted_csv_writer = csv.writer(weighted_csv_file)
    weighted_csv_writer.writerow(["ID", "Avg_Lat", "Avg_Lon", "Min_Weighted_Dist"])

    weighted_results = {}

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {VIDEO_FILE}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0  # safe default

    out_writer = cv2.VideoWriter('output_tracked.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_num = 0
    print("[INFO] Starting offline video + live map processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num not in telemetry_df.index:
                # No telemetry for this frame: skip or continue
                continue

            telem = telemetry_df.loc[frame_num]
            lat = float(telem["Lat"]) if not pd.isna(telem["Lat"]) else None
            lon = float(telem["Lon"]) if not pd.isna(telem["Lon"]) else None
            alt = float(telem["Altitude_m"]) if not pd.isna(telem["Altitude_m"]) else None
            yaw_deg = float(telem["Yaw_deg"]) if not pd.isna(telem["Yaw_deg"]) else None
            pitch = float(telem["Pitch_deg"]) if not pd.isna(telem["Pitch_deg"]) else 0.0
            roll = float(telem["Roll_deg"]) if not pd.isna(telem["Roll_deg"]) else 0.0

            # Basic sanity checks
            if lat is None or lon is None or alt is None or yaw_deg is None:
                continue
            if alt < 0:
                continue

            # Update drone on live map
            dx, dy = latlon_to_mercator(lat, lon)
            if _bokeh_ready.is_set() and _update_funcs["drone"]:
                _update_funcs["drone"](dx, dy)
                _update_funcs["recenter"](dx, dy)

            # YOLO detection
            results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)

            plane2plane_dist_m = alt

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    pix_d_px = float(np.linalg.norm(np.array([cx, cy]) - image_center))
                    ground_dist_m = plane2plane_dist_m * (pix_d_px / focal_len_px)

                    # Estimate person GPS via yaw-only offset (roll/pitch folded into weight)
                    person_lat, person_lon = offset_gps(lat, lon, ground_dist_m, yaw_deg)
                    id_str = f"{person_lat:.6f}_{person_lon:.6f}"

                    # Update live map with detection point
                    px, py = latlon_to_mercator(person_lat, person_lon)
                    if _bokeh_ready.is_set() and _update_funcs["person"]:
                        _update_funcs["person"](px, py)

                    # Write per-frame detection row
                    csv_writer.writerow([
                        frame_num,
                        id_str,
                        round(alt, 2),
                        round(ground_dist_m, 2),
                        person_lat,
                        person_lon,
                        lat,
                        lon,
                    ])

                    # Accumulate for weighted outputs
                    weight = 1.0 / (1.0 + abs(roll) + abs(pitch) + ground_dist_m)
                    weighted_results.setdefault(id_str, []).append((person_lat, person_lon, weight, ground_dist_m))

                    # Draw annotations on video frame
                    delta_north = ground_dist_m * cos(radians(yaw_deg))
                    delta_east = ground_dist_m * sin(radians(yaw_deg))
                    breakdown_label = f"{ground_dist_m:.2f}m | N:{delta_north:.2f} E:{delta_east:.2f}"
                    label = f"{id_str}\nLat:{person_lat:.6f}, Lon:{person_lon:.6f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 2)
                    cv2.putText(frame, breakdown_label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 2)

            # HUD
            gps_status = f"Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.2f}m"
            cv2.putText(frame, gps_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Detection Feed", frame)
            out_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        csv_file.close()

        # Save weighted averages
        for pid, records in weighted_results.items():
            total_weight = sum(w for _, _, w, _ in records)
            if total_weight == 0:
                continue
            avg_lat = sum(lat * w for lat, _, w, _ in records if lat is not None) / total_weight
            avg_lon = sum(lon * w for _, lon, w, _ in records if lon is not None) / total_weight
            min_dist = min(d for _, _, _, d in records)
            weighted_csv_writer.writerow([pid, avg_lat, avg_lon, min_dist])

        weighted_csv_writer = None
        print("[INFO] Offline processing complete. Video and CSVs saved. Live map running at http://localhost:5006/live_map")


if __name__ == "__main__":
    process_video()
