import math
import os
import requests
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from scipy.io import loadmat
from xml.dom import minidom
import re
from safetensors.torch import save_file, load_file
import json

# --- Constants ---
#alternative path: https://web.archive.org/web/20170314222333if_/http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch//sketches_matlab.zip
DATA_URL = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_matlab.zip"
DATA_DIR = "tu_berlin"
ZIP_PATH = "sketches_matlab.zip"
EXTRACTED_PATH = "tu_berlin"
MAX_LEN = 200




def download_and_extract(url, download_path, extract_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(extract_path) and any(
            os.path.isdir(os.path.join(extract_path, d)) for d in os.listdir(extract_path)):
        print("Dataset already downloaded and extracted.")
        return

    if not os.path.exists(download_path):
        print(f"Downloading dataset from {url}...")
        import requests
        from tqdm import tqdm
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(download_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True,
                                                      desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Extracting {download_path} to {extract_path}...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")



def extract_label_string(label_data):
    """
    Extract label string from TU-Berlin dataset structure.
    Labels are stored as simple numpy arrays with string data.
    """
    if label_data is None:
        return ""

    # Handle string directly
    if isinstance(label_data, str):
        return label_data.strip()

    # Handle numpy arrays - the main case for TU-Berlin labels
    if isinstance(label_data, np.ndarray):
        if label_data.size == 0:
            return ""

        # For simple 1D arrays, just get the first element
        if label_data.ndim == 1 and label_data.size > 0:
            return str(label_data[0]).strip()

        # For more complex arrays, flatten and get first
        flat = label_data.flatten()
        if len(flat) > 0:
            return str(flat[0]).strip()

        return ""

    # Handle lists
    if isinstance(label_data, (list, tuple)):
        if len(label_data) == 0:
            return ""
        return str(label_data[0]).strip()

    # Convert anything else to string
    return str(label_data).strip()


def extract_svg_string(raw_svg):
    """
    Extract SVG string from TU-Berlin dataset structure.
    The SVG is stored as a numpy array where each row contains one line of the SVG.
    """
    if raw_svg is None:
        return ""

    # Handle string directly
    if isinstance(raw_svg, str):
        return raw_svg.strip()

    # Handle numpy arrays - the main case for TU-Berlin data
    if isinstance(raw_svg, np.ndarray):
        if raw_svg.size == 0:
            return ""

        # Each row contains one line of the SVG file
        svg_lines = []
        for row in raw_svg:
            if isinstance(row, np.ndarray) and row.size > 0:
                # Extract the string from the nested array
                line_data = row.flatten()[0]
                if isinstance(line_data, np.ndarray) and line_data.size > 0:
                    line = str(line_data.flatten()[0])
                else:
                    line = str(line_data)
                svg_lines.append(line)

        return "\n".join(svg_lines)

    # Handle lists
    if isinstance(raw_svg, (list, tuple)):
        if len(raw_svg) == 0:
            return ""
        return extract_svg_string(raw_svg[0])

    # Convert anything else to string
    result = str(raw_svg).strip()
    return result



def arc_to_beziers(p0, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, p1, num_segments=10):
    """
    Convert an SVG arc to a list of cubic Bezier segments.
    """
    # Convert rotation to radians
    phi = math.radians(x_axis_rotation % 360)

    # Step 1: Handle out-of-range radii
    rx = abs(rx)
    ry = abs(ry)

    # Step 2: Transform to midpoint coordinates
    dx2 = (p0[0] - p1[0]) / 2.0
    dy2 = (p0[1] - p1[1]) / 2.0
    x1p = math.cos(phi) * dx2 + math.sin(phi) * dy2
    y1p = -math.sin(phi) * dx2 + math.cos(phi) * dy2

    # Step 3: Ensure radii are large enough
    rx_sq = rx * rx
    ry_sq = ry * ry
    x1p_sq = x1p * x1p
    y1p_sq = y1p * y1p

    radicant = (rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq)
    radicant /= (rx_sq * y1p_sq + ry_sq * x1p_sq)
    radicant = max(0, radicant)  # avoid negative due to float errors
    coef = math.sqrt(radicant) * (1 if large_arc_flag != sweep_flag else -1)

    cxp = coef * (rx * y1p) / ry
    cyp = coef * -(ry * x1p) / rx

    # Step 4: Transform center back
    cx = math.cos(phi) * cxp - math.sin(phi) * cyp + (p0[0] + p1[0]) / 2
    cy = math.sin(phi) * cxp + math.cos(phi) * cyp + (p0[1] + p1[1]) / 2

    # Step 5: Compute angles
    def angle(u, v):
        dot = u[0]*v[0] + u[1]*v[1]
        l = math.hypot(u[0], u[1]) * math.hypot(v[0], v[1])
        ang = math.acos(max(-1, min(1, dot / l)))
        if u[0]*v[1] - u[1]*v[0] < 0:
            ang = -ang
        return ang

    v1 = [(x1p - cxp) / rx, (y1p - cyp) / ry]
    v2 = [(-x1p - cxp) / rx, (-y1p - cyp) / ry]

    theta1 = angle([1, 0], v1)
    delta_theta = angle(v1, v2)

    if not sweep_flag and delta_theta > 0:
        delta_theta -= 2*math.pi
    elif sweep_flag and delta_theta < 0:
        delta_theta += 2*math.pi

    # Step 6: Approximate arc with Bezier curves
    beziers = []
    segments = max(1, int(abs(delta_theta) / (math.pi/2)) + 1)  # split into ≤90° arcs
    delta = delta_theta / segments
    t = theta1

    for _ in range(segments):
        t1 = t + delta
        # endpoints
        cos_t, sin_t = math.cos(t), math.sin(t)
        cos_t1, sin_t1 = math.cos(t1), math.sin(t1)

        p_start = [cx + rx * math.cos(phi) * cos_t - ry * math.sin(phi) * sin_t,
                   cy + rx * math.sin(phi) * cos_t + ry * math.cos(phi) * sin_t]
        p_end = [cx + rx * math.cos(phi) * cos_t1 - ry * math.sin(phi) * sin_t1,
                 cy + rx * math.sin(phi) * cos_t1 + ry * math.cos(phi) * sin_t1]

        alpha = math.tan(delta / 4) * 4/3
        p_ctrl1 = [p_start[0] - alpha * (rx * math.cos(phi) * sin_t + ry * math.sin(phi) * cos_t),
                   p_start[1] - alpha * (rx * math.sin(phi) * sin_t - ry * math.cos(phi) * cos_t)]
        p_ctrl2 = [p_end[0] + alpha * (rx * math.cos(phi) * sin_t1 + ry * math.sin(phi) * cos_t1),
                   p_end[1] + alpha * (rx * math.sin(phi) * sin_t1 - ry * math.cos(phi) * cos_t1)]

        beziers.append((p_start, p_ctrl1, p_ctrl2, p_end))
        t = t1

    return beziers



def interpolate_quadratic_bezier(p0, p1, p2, num_points=2):
    """
    Interpolate a quadratic Bezier curve with a few points.
    p0: start point, p1: control point, p2: end point
    num_points: number of interpolation points including start and end
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        x = (1-t)**2 * p0[0] + 2 * (1-t) * t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2 * (1-t) * t * p1[1] + t**2 * p2[1]
        points.append([x, y])
    return points

def interpolate_cubic_bezier(p0, p1, p2, p3, num_points=2):
    """
    Interpolate a cubic Bezier curve with a few points.
    p0: start point, p1: first control point, p2: second control point, p3: end point
    num_points: number of interpolation points including start and end
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        x = ((1-t)**3 * p0[0] +
             3 * (1-t)**2 * t * p1[0] +
             3 * (1-t) * t**2 * p2[0] +
             t**3 * p3[0])
        y = ((1-t)**3 * p0[1] +
             3 * (1-t)**2 * t * p1[1] +
             3 * (1-t) * t**2 * p2[1] +
             t**3 * p3[1])
        points.append([x, y])
    return points



def parse_svg_path_simple(path_data, interpolation_points=2):
    """
    Robust SVG path parser focused on M, L, C, Q, T, Z and A (approximated).

    Args:
        path_data: Raw SVG path string.
        interpolation_points: Number of points used for Bézier interpolation.

    Returns:
        np.ndarray: Nx2 array of points (float32) or an empty array.

    Notes:
        Skips malformed commands and warns when tokens are missing or invalid.
    """

    if not path_data or path_data.strip() == "":
        return np.array([], dtype=np.float32).reshape(0, 2)

    # Normalize separators
    path_data = path_data.replace(',', ' ')
    path_data = re.sub(r'\s+', ' ', path_data.strip())

    # Tokenization: either a command, 'NaN' (case-insensitive), or a number
    tokens = re.findall(r'[Nn][Aa][Nn]|[A-Za-z]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_data)

    def is_command(tok):
        # A command is any alphabetic token that is not 'NaN'
        return tok.isalpha() and tok.lower() != 'nan'

    def next_numbers(start_index, count, cmd):
        """
        Try to read 'count' numeric tokens.
        If not possible, raise warning and return (None, start_index).
        Handles 'NaN' as a malformed token.
        """
        nums = []
        idx = start_index
        if idx >= len(tokens):
            return None, start_index

        while len(nums) < count and idx < len(tokens):
            t = tokens[idx]
            if t.lower() == 'nan':
                print(f"Command {cmd}: encountered 'NaN' token — skipping.")
                return None, start_index
            if is_command(t):
                print(tokens)
                print(
                    f"Command {cmd}: encountered another command '{t}' before {count} numbers could be read — skipping.")
                return None, start_index
            try:
                val = float(t)
            except ValueError:
                print(f"Command {cmd}: malformed numeric token '{t}' — skipping.")
                return None, start_index
            nums.append(val)
            idx += 1
        if len(nums) < count:
            print(
                f"Command {cmd}: insufficient numeric parameters (expected {count}, got {len(nums)}) — skipping.")
            return None, start_index
        return nums, idx

    points = []
    current_pos = [0.0, 0.0]
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if is_command(tok):
            command = tok
            i += 1
            upper = command.upper()

            if upper == 'M':
                nums, i = next_numbers(i, 2, 'M')
                if nums is None:
                    continue
                x, y = nums
                if command.islower():
                    current_pos[0] += x;
                    current_pos[1] += y
                else:
                    current_pos[0] = x;
                    current_pos[1] = y
                points.append([current_pos[0], current_pos[1]])

                # Handle implicit L after M
                while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                    try:
                        x = float(tokens[i]);
                        y = float(tokens[i + 1])
                    except ValueError:
                        print("M: malformed coordinates — skipping rest of M command.")
                        break
                    if command.islower():
                        current_pos[0] += x;
                        current_pos[1] += y
                    else:
                        current_pos[0] = x;
                        current_pos[1] = y
                    points.append([current_pos[0], current_pos[1]])
                    i += 2

            elif upper == 'L':
                while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                    try:
                        x = float(tokens[i]);
                        y = float(tokens[i + 1])
                    except ValueError:
                        print("L: malformed coordinates — skipping rest of L command.")
                        break
                    if command.islower():
                        current_pos[0] += x;
                        current_pos[1] += y
                    else:
                        current_pos[0] = x;
                        current_pos[1] = y
                    points.append([current_pos[0], current_pos[1]])
                    i += 2
            elif upper == 'C':
                first = True
                while True:
                    # Check if we have enough tokens for a complete C command (6 numbers)
                    if i + 5 >= len(tokens):
                        break
                    # Check if any of the next 6 tokens are commands
                    has_command = False
                    for j in range(6):
                        if i + j < len(tokens) and is_command(tokens[i + j]):
                            has_command = True
                            break
                    if has_command:
                        if first:
                            print("C: encountered command before reading 6 numbers — skipping.")
                            first = False
                        break
                    # Try to parse the 6 numbers for the C command
                    res = next_numbers(i, 6, 'C')
                    if res[0] is None:
                        break
                    nums, i = res
                    cx1, cy1, cx2, cy2, x, y = nums
                    p0 = current_pos.copy()
                    if command.islower():
                        p1 = [current_pos[0] + cx1, current_pos[1] + cy1]
                        p2 = [current_pos[0] + cx2, current_pos[1] + cy2]
                        p3 = [current_pos[0] + x, current_pos[1] + y]
                        current_pos[0] += x
                        current_pos[1] += y
                    else:
                        p1 = [cx1, cy1]
                        p2 = [cx2, cy2]
                        p3 = [x, y]
                        current_pos[0] = x
                        current_pos[1] = y
                    bezier_points = interpolate_cubic_bezier(p0, p1, p2, p3, num_points=interpolation_points)
                    points.extend(bezier_points[1:])
                    first = False


            elif upper == 'Q':
                while True:
                    res = next_numbers(i, 4, 'Q')
                    if res[0] is None:
                        break
                    nums, i = res
                    cx, cy, x, y = nums
                    p0 = current_pos.copy()
                    if command.islower():
                        p1 = [current_pos[0] + cx, current_pos[1] + cy]
                        p2 = [current_pos[0] + x, current_pos[1] + y]
                        current_pos[0] += x;
                        current_pos[1] += y
                    else:
                        p1 = [cx, cy];
                        p2 = [x, y]
                        current_pos[0] = x;
                        current_pos[1] = y
                    bezier_points = interpolate_quadratic_bezier(p0, p1, p2, num_points=interpolation_points)
                    points.extend(bezier_points[1:])

            elif upper == 'T':
                while i + 1 < len(tokens) and not is_command(tokens[i]) and not is_command(tokens[i + 1]):
                    try:
                        x = float(tokens[i]);
                        y = float(tokens[i + 1])
                    except ValueError:
                        print("T: malformed coordinates — skipping rest of T command.")
                        break
                    if command.islower():
                        current_pos[0] += x;
                        current_pos[1] += y
                    else:
                        current_pos[0] = x;
                        current_pos[1] = y
                    points.append([current_pos[0], current_pos[1]])
                    i += 2

            elif upper == 'Z':
                continue  # nothing to add here

            elif upper == 'A':
                res = next_numbers(i, 7, 'A')
                if res[0] is None:
                    # advance to next command
                    while i < len(tokens) and not is_command(tokens[i]):
                        i += 1
                    continue
                nums, i = res
                rx, ry, x_axis_rotation, large_arc_flag_f, sweep_flag_f, x, y = nums
                try:
                    large_arc_flag = int(round(large_arc_flag_f))
                    sweep_flag = int(round(sweep_flag_f))
                except Exception:
                    print("A: flags not parseable — defaulting to 0.")
                    large_arc_flag, sweep_flag = 0, 0

                p0 = current_pos.copy()
                p1 = [current_pos[0] + x, current_pos[1] + y] if command.islower() else [x, y]

                try:
                    beziers = arc_to_beziers(p0, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, p1)
                    for (pb0, c1, c2, pb1) in beziers:
                        bezier_points = interpolate_cubic_bezier(pb0, c1, c2, pb1, num_points=interpolation_points)
                        points.extend(bezier_points[1:])
                except Exception as e:
                    print(f"A: arc_to_beziers failed ({e}) — skipping this arc.")
                current_pos = p1

            else:
                print(f"Unsupported command '{command}' — skipping.")
                while i < len(tokens) and not is_command(tokens[i]):
                    i += 1

        else:
            # Number or 'NaN' without explicit command → skip
            print(f"Stray token '{tok}' outside any command — skipping.")
            i += 1

    return np.array(points, dtype=np.float32) if points else np.array([], dtype=np.float32).reshape(0, 2)


def parse_svg_simple(svg_data, interpolation_points=2):
    """
    Extract strokes from SVG data using simplified parsing.
    """
    strokes = []

    # Extract SVG string
    svg_string = extract_svg_string(svg_data)

    if not svg_string or len(svg_string.strip()) < 10:  # Too short to be valid SVG
        return strokes

    # Parse XML
    try:
        svg_doc = minidom.parseString(svg_string)
    except Exception:
        # Handle cases where the SVG string is completely malformed XML
        return strokes


    # Extract path elements
    path_nodes = svg_doc.getElementsByTagName('path')

    for path_node in path_nodes:
        d_attr = path_node.getAttribute('d')
        if d_attr and d_attr.strip():
            stroke_points = parse_svg_path_simple(d_attr, interpolation_points=interpolation_points)
            if len(stroke_points) > 0:
                strokes.append(stroke_points)

    svg_doc.unlink()

    return strokes


def convert_to_delta_format(strokes):
    """Convert list of strokes to (dx, dy, pen_state) format for LSTM input."""
    if not strokes:
        return np.empty((0, 3), dtype=np.float32)

    points = []
    for stroke_idx, stroke in enumerate(strokes):
        if len(stroke) == 0:
            continue

        # Add all points in the stroke
        for point_idx, point in enumerate(stroke):
            if len(point) >= 2:  # Ensure we have x, y coordinates
                # pen_state: 0 = pen down (drawing), 1 = pen up (end of stroke)
                pen_state = 1 if point_idx == len(stroke) - 1 else 0
                points.append([float(point[0]), float(point[1]), pen_state])

    if not points:
        return np.empty((0, 3), dtype=np.float32)

    points = np.array(points, dtype=np.float32)

    # Convert to delta format (relative coordinates)
    deltas = np.zeros_like(points)

    # First point is absolute
    deltas[0, :2] = points[0, :2]
    deltas[0, 2] = points[0, 2]

    # Subsequent points are relative to previous
    if len(points) > 1:
        deltas[1:, :2] = points[1:, :2] - points[:-1, :2]
        deltas[1:, 2] = points[1:, 2]

    return deltas


def strokes_to_sequence(drawing, max_len=200):
    """Convert drawing to fixed-length sequence for LSTM training."""
    if len(drawing) == 0:
        return np.zeros((max_len, 4), dtype=np.float32)

    # Truncate or pad to max_len
    if len(drawing) > max_len:
        seq = np.zeros((max_len, 4), dtype=np.float32)
        seq[:max_len, :3] = drawing[:max_len]
        seq[:max_len, 3] = 1  # mask: 1 for real data
    else:
        seq = np.zeros((max_len, 4), dtype=np.float32)
        seq[:len(drawing), :3] = drawing
        seq[:len(drawing), 3] = 1  # mask: 1 for real data, 0 for padding

    return seq



def load_tu_berlin_data(path, max_len=200, interpolation_points=2):
    """
    Load and process TU-Berlin dataset for LSTM training.
    Note the download file is currently not working.
    Args:
        path: Base dataset directory.
        max_len: Maximum sequence length.
        interpolation_points: Number of interpolation points used while parsing SVG paths.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]: Sequences, labels, and class names.
    """
    mat_file = os.path.join(path,DATA_DIR, "sketches_matlab", "sketches.mat")
    zip_path = os.path.join(path,DATA_DIR, "sketches_matlab.zip")
    extracted_path = os.path.join(path,DATA_DIR)
    if not os.path.exists(mat_file):
        print(f"{mat_file} not found! Downloading and extracting dataset...")
        download_and_extract(DATA_URL, zip_path, extracted_path)
        # After extraction, mat_file should exist at path/sketches_matlab/sketches.mat
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"Failed to download or extract {mat_file}!")

    sequences = []
    labels = []
    failed_count = 0
    insufficient_points_count = 0
    truncated_count = 0

    print(f"Loading {mat_file}...")
    mat = loadmat(mat_file)
    sketches = mat['D']
    num_sketches = sketches.shape[0]
    print(f"Found {num_sketches} sketches in .mat file")

    # Extract class names
    all_labels = []
    for i in range(num_sketches):
        label_data = sketches[i, 1]
        label_str = extract_label_string(label_data)
        if label_str:
            all_labels.append(label_str)

    class_names = sorted(list(set(all_labels)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Found {len(class_names)} classes")

    for i in tqdm(range(num_sketches), desc="Processing sketches"):
        # 1. Get the class label string
        label_data = sketches[i, 1]
        label_str = extract_label_string(label_data)

        if not label_str or label_str not in class_to_idx:
            failed_count += 1
            continue

        label = class_to_idx[label_str]

        # 2. Get the SVG data
        raw_svg = sketches[i, 2]

        # Parse SVG using simplified function
        strokes = parse_svg_simple(raw_svg, interpolation_points=interpolation_points)

        if not strokes:  # Empty strokes list
            failed_count += 1
            continue

        # Convert to delta format
        delta_drawing = convert_to_delta_format(strokes)

        if len(delta_drawing) == 0:  # Empty delta drawing
            failed_count += 1
            continue



        # Check if sketch will be truncated due to length
        if len(delta_drawing) > max_len:
            truncated_count += 1

        # Convert to sequence
        seq = strokes_to_sequence(delta_drawing, max_len)

        sequences.append(seq)
        labels.append(label)

    print(f"Successfully processed {len(sequences)} sketches")
    print(f"Failed: {failed_count} total")
    print(f"Cut off due to insufficient points: {insufficient_points_count}")
    print(f"Truncated due to max_len limit ({max_len}): {truncated_count}")

    if len(sequences) == 0:
        raise ValueError("No sketches were successfully processed!")

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), class_names


def get_cache_path(data_dir, max_len, interpolation_points):
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_base = f"tu_berlin_maxlen{max_len}_interp{interpolation_points}"
    cache_data = os.path.join(cache_dir, f"{cache_base}.safetensors")
    cache_meta = os.path.join(cache_dir, f"{cache_base}_meta.json")
    return cache_data, cache_meta

def save_cache(sequences, labels, class_names, max_len, interpolation_points, cache_data, cache_meta):
    tensors = {
        "sequences": torch.from_numpy(sequences),
        "labels": torch.from_numpy(labels)
    }
    save_file(tensors, cache_data)
    meta = {
        "max_len": max_len,
        "interpolation_points": interpolation_points,
        "class_names": class_names
    }
    with open(cache_meta, "w") as f:
        json.dump(meta, f)

def load_cache(cache_data, cache_meta):
    tensors = load_file(cache_data)
    with open(cache_meta, "r") as f:
        meta = json.load(f)
    sequences = tensors["sequences"].numpy()
    labels = tensors["labels"].numpy()
    class_names = meta["class_names"]
    max_len = meta["max_len"]
    interpolation_points = meta["interpolation_points"]
    return sequences, labels, class_names, max_len, interpolation_points

def cached_load_tu_berlin_data(path, max_len=200, interpolation_points=2):
    path2 = os.path.join(path, DATA_DIR)
    cache_data, cache_meta = get_cache_path(path2, max_len, interpolation_points)
    if os.path.exists(cache_data) and os.path.exists(cache_meta):
        try:
            sequences, labels, class_names, cached_max_len, cached_interp = load_cache(cache_data, cache_meta)
            if cached_max_len == max_len and cached_interp == interpolation_points:
                print(f"Loaded cached TU-Berlin dataset (max_len={max_len}, interpolation_points={interpolation_points})")
                return sequences, labels, class_names
            else:
                print("Cache exists but parameters differ, regenerating cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating cache.")
    sequences, labels, class_names = load_tu_berlin_data(path, max_len=max_len, interpolation_points=interpolation_points)
    save_cache(sequences, labels, class_names, max_len, interpolation_points, cache_data, cache_meta)
    print(f"Saved TU-Berlin dataset to cache (max_len={max_len}, interpolation_points={interpolation_points})")
    return sequences, labels, class_names

# ----------------------
# Dataset
# ----------------------
class TUBerlinDataset(Dataset):
    def __init__(self, data_dir, max_len=200, interpolation_points=2):
        self.max_len = max_len
        self.interpolation_points = interpolation_points
        self.sequences, self.labels, self.class_names = cached_load_tu_berlin_data(
            data_dir, max_len=max_len, interpolation_points=interpolation_points
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), self.labels[idx]


class NormalizeToRangeBatched(nn.Module):
    """Normalize absolute coordinates to [-128,128] range."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        abs_xy = torch.cumsum(stroke_seq[..., :2], dim=1)
        center = abs_xy.mean(dim=1, keepdim=True)
        centered = abs_xy - center
        max_abs = centered.abs().amax(dim=(1, 2), keepdim=True)
        scale = 128.0 / (max_abs + self.eps)
        scaled = centered * scale

        rel_xy = torch.zeros_like(scaled)
        rel_xy[:, 0] = scaled[:, 0]
        rel_xy[:, 1:] = scaled[:, 1:] - scaled[:, :-1]
        pen_state = stroke_seq[..., 2:3]
        mask = stroke_seq[..., 3:4]
        return torch.cat([rel_xy, pen_state, mask], dim=-1)

import matplotlib.pyplot as plt

def visualize_sequence(seq: torch.Tensor, title=None, ax=None,
                       linewidth=2, color='black'):
    """
    Plot a stroke sequence tensor of shape (n, 4): dx, dy, pen state, mask.

    - Accumulates deltas to x/y.
    - Splits strokes whenever pen state == 1 (lift).
    - Only plots points where mask == 1 (real data).
    """
    if ax is None:
        fig, ax = plt.subplots()
    seq = seq.cpu().numpy()
    dx, dy, pen, mask = seq[:,0], seq[:,1], seq[:,2], seq[:,3]
    
    # Only use real data points (mask == 1)
    real_indices = mask == 1
    if not np.any(real_indices):
        ax.axis('equal')
        ax.axis('off')
        if title:
            ax.set_title(title)
        return ax
    
    dx = dx[real_indices]
    dy = dy[real_indices] 
    pen = pen[real_indices]
    
    x = dx.cumsum()
    y = dy.cumsum()

    strokes = []
    xs, ys = [], []
    for xi, yi, p in zip(x, y, pen):
        xs.append(xi); ys.append(-yi)
        if p == 1:             # pen lift: end current stroke
            strokes.append((xs, ys))
            xs, ys = [], []
    if xs:
        strokes.append((xs, ys))

    for xs, ys in strokes:
        ax.plot(xs, ys, linewidth=linewidth, color=color)
    ax.axis('equal')
    ax.axis('off')
    if title:
        ax.set_title(title)
    return ax

# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    download_and_extract(DATA_URL, ZIP_PATH, EXTRACTED_PATH)

    # Use TUBerlinDataset with caching and interpolation_points argument
    dataset = TUBerlinDataset(EXTRACTED_PATH, max_len=250, interpolation_points=2)
    sequences, labels, class_names = dataset.sequences, dataset.labels, dataset.class_names
    print(f"Loaded {len(sequences)} sketches from {len(class_names)} classes")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    x = dataset[10][0]
    visualize_sequence(x, title=f"Class: {class_names[dataset[0][1]]}")
    plt.show()

    normalizer = NormalizeToRangeBatched()
    sample_batch, label_batch = next(iter(dataloader))
    normalized_batch = normalizer(sample_batch)
    sample_abs_coords = torch.cumsum(normalized_batch[0, :, :2], dim=0)
    print(f"Normalized sample min: {sample_abs_coords.min().item():.2f}, max: {sample_abs_coords.max().item():.2f}")

    # Additional debugging info
    print(f"Sample sequence shape: {sample_batch[0].shape}")
    print(f"Non-zero elements in sample: {torch.count_nonzero(sample_batch[0])}")
    print(f"Sample class: {class_names[label_batch[0]]}")