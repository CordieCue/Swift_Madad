import cv2
import numpy as np
import csv
import pandas as pd
from ultralytics import YOLO
from math import sin, cos, radians, log, tan, pi
import time
import threading

# ---- Bokeh (older versions) ----
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, WMTSTileSource
from bokeh.server.server import Server

"""
Controls (OpenCV window):
  Space  -> Pause/Play
  Left   -> Rewind ~30 frames
  Right  -> Forward ~30 frames
  ESC    -> Quit

Live map: http://localhost:5006/live_map
- Blue triangle: drone (rotates with yaw)
- Red dots: detected person positions
- Base: ESRI World Imagery

Requirements:
  pip install bokeh pandas numpy ultralytics opencv-python

CSV columns expected (indexed by 'Frame'): Frame, Lat, Lon, Altitude_m, Yaw_deg, Pitch_deg, Roll_deg
"""

# ---------------------- CONFIG ----------------------
VIDEO_FILE = "raw_flight.avi"          # <-- set your video path here
TELEMETRY_FILE = "telemetry_log.csv"   # path to your CSV

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

# Build a small triangle (in meters) around origin; rotate and translate to (x,y)
def rotated_drone_triangle(x, y, yaw_deg, size_m=8.0):
    """Return xs, ys arrays for a triangle centered at (x,y) pointing to yaw_deg."""
    # Triangle points in local coords (nose forward)
    p0 = np.array([0.0,  1.0])   # nose
    p1 = np.array([-0.6, -1.0])
    p2 = np.array([0.6, -1.0])
    tri = np.vstack([p0, p1, p2]) * size_m

    theta = -radians(yaw_deg)  # WebMercator y increases up; heading cw from north -> negative rotation
    Rm = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    rot = tri @ Rm.T
    rot[:, 0] += x
    rot[:, 1] += y
    xs = rot[:, 0].tolist()
    ys = rot[:, 1].tolist()
    # Close the path
    xs.append(xs[0]); ys.append(ys[0])
    return xs, ys

# ---------------------- BOKEH LIVE MAP (old-version compatible) ----------------------
_bokeh_ready = threading.Event()
_update_funcs = {"drone_arrow": None, "drone_path": None, "person": None, "recenter": None, "clear": None}

init_x, init_y = latlon_to_mercator(first_row["Lat"], first_row["Lon"])

def bkapp(doc):
    # Data sources
    drone_path_src = ColumnDataSource(data=dict(x=[], y=[]))
    person_src = ColumnDataSource(data=dict(x=[], y=[]))
    drone_arrow_src = ColumnDataSource(data=dict(xs=[[init_x]], ys=[[init_y]]))  # patches expects list-of-lists

    # ESRI satellite tiles
    tile = WMTSTileSource(url=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{Z}/{Y}/{X}"
    ))

    p = figure(
        x_axis_type="mercator", y_axis_type="mercator",
        x_range=(init_x - 500, init_x + 500),
        y_range=(init_y - 500, init_y + 500),
        title="Drone & Person Tracking (ESRI Satellite)",
        tools="pan,wheel_zoom,reset,save",
        sizing_mode="stretch_both",
    )
    p.add_tile(tile)

    # Glyphs
    p.line(x='x', y='y', source=drone_path_src, line_width=2, color='blue', legend_label="Drone Path")
    p.circle(x='x', y='y', source=person_src, size=6, color='red', alpha=0.9, legend_label="Detections")
    p.patches(xs='xs', ys='ys', source=drone_arrow_src, fill_color='blue', fill_alpha=0.8, line_color='white', line_width=1)
    p.legend.location = "top_left"

    # Thread-safe helpers
    def next_tick(fn):
        doc.add_next_tick_callback(fn)

    def stream_drone_path(x, y):
        next_tick(lambda: drone_path_src.stream(dict(x=[x], y=[y]), rollover=5000))

    def stream_person(x, y):
        next_tick(lambda: person_src.stream(dict(x=[x], y=[y]), rollover=10000))

    def set_drone_arrow(xs, ys):
        next_tick(lambda: drone_arrow_src.data.update(dict(xs=[xs], ys=[ys])))

    def recenter_on(x, y, pad=350):
        def _recenter():
            p.x_range.start = x - pad
            p.x_range.end = x + pad
            p.y_range.start = y - pad
            p.y_range.end = y + pad
        next_tick(_recenter)

    def clear_map():
        def _clear():
            drone_path_src.data = dict(x=[], y=[])
            person_src.data = dict(x=[], y=[])
        next_tick(_clear)

    # Expose callbacks
    _update_funcs["drone_path"] = stream_drone_path
    _update_funcs["person"] = stream_person
    _update_funcs["drone_arrow"] = set_drone_arrow
    _update_funcs["recenter"] = recenter_on
    _update_funcs["clear"] = clear_map
    _bokeh_ready.set()

    doc.add_root(p)


def start_bokeh_server():
    server = Server({'/live_map': bkapp}, num_procs=1, port=5006,
                    allow_websocket_origin=["localhost:5006", "127.0.0.1:5006"])
    server.start()
    print("[INFO] Bokeh live map running at http://localhost:5006/live_map")
    server.io_loop.start()

# Start Bokeh server on a background thread
threading.Thread(target=start_bokeh_server, daemon=True).start()
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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter('output_tracked.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    print("[INFO] Starting offline video + live map processing with controls...")

    paused = False
    rewind_step = int(round(fps))  # ~1 second
    forward_step = int(round(fps))

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Compute current 0-based and 1-based frame indices
                pos_next = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # index of NEXT frame
                current_idx0 = max(0, pos_next - 1)
                frame_num = current_idx0 + 1  # 1-based for CSV mapping

                # Telemetry for this frame
                if frame_num not in telemetry_df.index:
                    # Draw frame counter even if no telemetry
                    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Detection Feed", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord(' '):
                        paused = not paused
                    elif key in (81, 2424832):  # Left arrow (Linux/Win)
                        target = max(0, current_idx0 - rewind_step)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        if _bokeh_ready.is_set():
                            _update_funcs["clear"]()
                    elif key in (83, 2555904):  # Right arrow (Linux/Win)
                        target = min(total_frames - 1, current_idx0 + forward_step)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        if _bokeh_ready.is_set():
                            _update_funcs["clear"]()
                    continue

                telem = telemetry_df.loc[frame_num]
                lat = float(telem["Lat"]) if not pd.isna(telem["Lat"]) else None
                lon = float(telem["Lon"]) if not pd.isna(telem["Lon"]) else None
                alt = float(telem["Altitude_m"]) if not pd.isna(telem["Altitude_m"]) else None
                yaw_deg = float(telem["Yaw_deg"]) if not pd.isna(telem["Yaw_deg"]) else None
                pitch = float(telem["Pitch_deg"]) if not pd.isna(telem["Pitch_deg"]) else 0.0
                roll = float(telem["Roll_deg"]) if not pd.isna(telem["Roll_deg"]) else 0.0

                if lat is None or lon is None or alt is None or yaw_deg is None or alt < 0:
                    # Draw frame counter and skip processing
                    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Detection Feed", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break
                    elif key == ord(' '):
                        paused = not paused
                    elif key in (81, 2424832):
                        target = max(0, current_idx0 - rewind_step)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        if _bokeh_ready.is_set():
                            _update_funcs["clear"]()
                    elif key in (83, 2555904):
                        target = min(total_frames - 1, current_idx0 + forward_step)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        if _bokeh_ready.is_set():
                            _update_funcs["clear"]()
                    continue

                # Update drone on live map (path + arrow)
                dx, dy = latlon_to_mercator(lat, lon)
                if _bokeh_ready.is_set():
                    _update_funcs["drone_path"](dx, dy)
                    xs, ys = rotated_drone_triangle(dx, dy, yaw_deg, size_m=8.0)
                    _update_funcs["drone_arrow"](xs, ys)
                    _update_funcs["recenter"](dx, dy)

                # YOLO detection (people only, class 0)
                results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)
                plane2plane_dist_m = alt

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        pix_d_px = float(np.linalg.norm(np.array([cx, cy]) - image_center))
                        ground_dist_m = plane2plane_dist_m * (pix_d_px / focal_len_px)

                        person_lat, person_lon = offset_gps(lat, lon, ground_dist_m, yaw_deg)
                        id_str = f"{person_lat:.6f}_{person_lon:.6f}"

                        # Live map: detection point
                        px, py = latlon_to_mercator(person_lat, person_lon)
                        if _bokeh_ready.is_set():
                            _update_funcs["person"](px, py)

                        # CSV row
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

                        # Weighted results (for summary at end)
                        weight = 1.0 / (1.0 + abs(roll) + abs(pitch) + ground_dist_m)
                        weighted_results.setdefault(id_str, []).append((person_lat, person_lon, weight, ground_dist_m))

                        # Video annotations
                        delta_north = ground_dist_m * cos(radians(yaw_deg))
                        delta_east = ground_dist_m * sin(radians(yaw_deg))
                        breakdown_label = f"{ground_dist_m:.2f}m | N:{delta_north:.2f} E:{delta_east:.2f}"
                        label = f"{id_str}\nLat:{person_lat:.6f}, Lon:{person_lon:.6f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                        cv2.putText(frame, label, (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 2)
                        cv2.putText(frame, breakdown_label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 2)

                # HUD overlays
                cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.2f}m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Detection Feed", frame)
                out_writer.write(frame)

            # Key handling (works both paused and playing)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                paused = not paused
            elif key in (81, 2424832):  # Left arrow (Linux/Win)
                current_idx0 = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                target = max(0, current_idx0 - rewind_step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if _bokeh_ready.is_set():
                    _update_funcs["clear"]()
            elif key in (83, 2555904):  # Right arrow (Linux/Win)
                current_idx0 = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                target = min(total_frames - 1, current_idx0 + forward_step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if _bokeh_ready.is_set():
                    _update_funcs["clear"]()

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

        weighted_csv_file.close()
        print("[INFO] Done. Video + CSVs saved. Live map still available at http://localhost:5006/live_map")


if __name__ == "__main__":
    process_video()
