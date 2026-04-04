import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

# ── MediaPipe skeleton connections grouped by body region ──
BONES = {
    "torso":      [(11,12), (11,23), (12,24), (23,24)],
    "left_arm":   [(11,13), (13,15)],
    "right_arm":  [(12,14), (14,16)],
    "left_leg":   [(23,25), (25,27)],
    "right_leg":  [(24,26), (26,28)],
    "face":       [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10)],
}
BONE_COLORS = {
    "torso": "#636EFA", "left_arm": "#EF553B", "right_arm": "#00CC96",
    "left_leg": "#AB63FA", "right_leg": "#FFA15A", "face": "#AAAAAA",
}

FEATURE_NAMES = [
    "knee_∠_L", "knee_∠_R", "elbow_∠_L", "elbow_∠_R",
    "trunk_incl", "wrist_y_L", "wrist_y_R", "shoulder_w",
    "v_knee_∠_L", "v_knee_∠_R", "v_elbow_∠_L", "v_elbow_∠_R",
    "v_trunk", "v_wrist_y_L", "v_wrist_y_R", "v_shoulder_w",
]

KEY_FEATURES = [0, 1, 2, 3, 4]  # angles + trunk

def load_T_frames(dataset_cfg: dict, sample_video: str) -> tuple[np.ndarray, int]: 
    """
    Load pose landmarks for a given sample video and return the number of frames.

    This function loads a `.npz` file containing MediaPipe pose landmarks,
    validates its existence and structure, and extracts the temporal length
    (number of frames) from the landmark array.

    Args:
        dataset_cfg (dict):
            Dataset configuration dictionary. Must contain
            `dataset_cfg["paths"]["poses_dir"]`, which points to the directory
            containing pose `.npz` files.

        sample_video (str):
            Name of the sample video (without file extension) whose pose
            landmarks should be loaded.

    Raises:
        FileNotFoundError:
            If the poses directory does not exist.

        FileNotFoundError:
            If the corresponding `.npz` file for the given sample video
            does not exist in the poses directory.

        KeyError:
            If the loaded `.npz` file does not contain a `landmarks` array.

        ValueError:
            If the `landmarks` array does not have the expected shape
            `(T, 33, 4)`.

    Returns:
        np.ndarray:
            landmarks of the `T` frames
        int:
            The number of frames `T` in the pose sequence.
    """
    
    poses_dir = Path(dataset_cfg["paths"]["poses_dir"])
    if not poses_dir.is_dir():
        raise FileNotFoundError(f"Poses directory not found: {poses_dir}")
    
    file_path = poses_dir / f"{sample_video}.npz"
    
    if not file_path.is_file():
        raise FileNotFoundError(f"Sample video file not found in location: {file_path}")
    
    with np.load(file_path) as npz:
        if "landmarks" not in npz:
            raise KeyError(
                f"'landmarks' not found in {file_path.name}. "
                f"Available keys: {list(npz.keys())}"
            )
        landmarks: np.ndarray = npz["landmarks"].copy()  # (T, 33, 4) — copy before file closes
        if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 4):
            raise ValueError(
                f"Invalid landmarks shape {landmarks.shape} "
                f"in {file_path.name}, expected (T, 33, 4)"
            )

    T_frames = landmarks.shape[0]
    print(f"Video: {sample_video} — {T_frames} frames")
    
    return landmarks, T_frames


# Visibility threshold below which a landmark position is not trusted
VIS_THRESHOLD = 0.3


def make_frame_traces(lm_frame: np.ndarray, vis_threshold: float = VIS_THRESHOLD) -> list:
    """
        Create Plotly 3D traces for a single pose frame.

        MediaPipe image coordinates have y=0 at the top of the frame and y=1 at
        the bottom.  To display the skeleton right-side up in a 3D plot (where
        positive y is up), y is negated before plotting.

        Bone segments whose endpoints have visibility below `vis_threshold` are
        skipped (replaced with None gaps) so that occluded / off-screen landmarks
        do not draw phantom lines across the figure.

        Args:
            lm_frame (np.ndarray | None):
                Pose landmarks for one frame with shape (33, 4) — (x, y, z, vis).
                May be None or empty if landmarks are missing.
            vis_threshold (float):
                Minimum visibility score [0, 1] required to draw a bone endpoint.
                Defaults to VIS_THRESHOLD (module-level constant).

        Returns:
            list:
                List of Plotly Scatter3d traces. Empty if frame is invalid.
    """

    # Handle missing or empty frames gracefully
    if lm_frame is None:
        return []

    if not isinstance(lm_frame, np.ndarray):
        raise TypeError(f"lm_frame must be np.ndarray or None, got {type(lm_frame)}")

    if lm_frame.size == 0:
        return []

    if lm_frame.ndim != 2 or lm_frame.shape[1] < 3:
        raise ValueError(
            f"Invalid lm_frame shape {lm_frame.shape}, expected (N, >=3)"
        )

    x = lm_frame[:, 0]
    y = -lm_frame[:, 1]   # flip: MediaPipe y↓ → plot y↑
    z = lm_frame[:, 2]
    vis = lm_frame[:, 3] if lm_frame.shape[1] >= 4 else np.ones(len(x))

    traces = []

    # Joints — only render trusted landmarks
    trusted = vis >= vis_threshold
    traces.append(go.Scatter3d(
        x=x[trusted], y=y[trusted], z=z[trusted], mode="markers",
        marker=dict(size=3, color="#19D3F3"),
        name="joints", showlegend=False,
    ))

    # Bones by region — skip segments where either endpoint is occluded
    for region, pairs in BONES.items():
        bx, by, bz = [], [], []
        for i, j in pairs:
            if vis[i] >= vis_threshold and vis[j] >= vis_threshold:
                bx += [x[i], x[j], None]
                by += [y[i], y[j], None]
                bz += [z[i], z[j], None]
        if bx:  # skip trace entirely if all bones in this region were occluded
            traces.append(go.Scatter3d(
                x=bx, y=by, z=bz, mode="lines",
                line=dict(color=BONE_COLORS[region], width=4),
                name=region, showlegend=False,
            ))
    return traces

def build_frames_for_slider(dataset_cfg: dict, sample_video: str) -> None:    
    """
        Build a Plotly 3D animated figure with a frame slider for pose visualization.

        Args:
            dataset_cfg (dict): Dataset configuration dictionary.
            sample_video (str): Video identifier (without file extension).

        Returns:
            None
    """

    landmarks, T_frames = load_T_frames(dataset_cfg= dataset_cfg, sample_video= sample_video)
    
    # Build frames for the slider
    frames = []        
    for t in range(T_frames):
        frame_traces = make_frame_traces(landmarks[t])
        frames.append(go.Frame(data=frame_traces, name=str(t)))

    # Initial frame
    init_traces = make_frame_traces(landmarks[0])
    # Axis range — use only trusted landmarks to avoid occluded outliers skewing the view
    # Also apply y-flip here so the axis range matches the rendered coordinates
    vis_all = landmarks[:, :, 3]
    trusted_mask = vis_all >= VIS_THRESHOLD  # (T, 33) bool
    pad = 0.1

    def axis_range(dim: int) -> list[float]:
        vals = landmarks[:, :, dim][trusted_mask]
        if dim == 1:        # y is negated in make_frame_traces
            vals = -vals
        return [float(vals.min()) - pad, float(vals.max()) + pad]

    axis_cfg = lambda dim: dict(
        range=axis_range(dim),
        showbackground=False, showticklabels=False, title=""
    )

    fig = go.Figure(
        data=init_traces,
        frames=frames,
        layout=go.Layout(
            title=f"3D Skeleton — {sample_video}",
            width=700, height=600,
            scene=dict(
                xaxis=axis_cfg(0), yaxis=axis_cfg(1), zaxis=axis_cfg(2),
                aspectmode="data",
                camera=dict(eye=dict(x=0, y=-2, z=0.5)),
            ),
            sliders=[dict(
                active=0,
                steps=[dict(args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                            label=str(t), method="animate")
                    for t in range(T_frames)],
                x=0.05, len=0.9,
                currentvalue=dict(prefix="Frame: "),
            )],
            updatemenus=[dict(
                type="buttons", x=0.05, y=0,
                buttons=[
                    dict(label="▶ Play", method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
                ],
            )],
        ),
    )

    fig.show()
    

def build_feature_timeline(dataset_cfg: dict, sample_video: str) -> None:
    """
    Build and display a temporal feature timeline with annotated error segments.

    This function loads per-frame feature values for a given video and visualizes
    them as continuous time series using Plotly. It also overlays shaded regions
    corresponding to labeled error segments (e.g., form mistakes) provided in a
    JSON annotation file.

    The visualization is intended for model analysis and debugging, allowing
    inspection of how selected features evolve over time relative to annotated
    errors.

    Args:
        dataset_cfg (dict):
            Dataset configuration dictionary. Must contain:
            - dataset_cfg["paths"]["features_dir"]: directory containing `.npy`
            feature files
            - dataset_cfg["paths"]["labels_dir"]: directory containing `.json`
            label files
            Optionally:
            - dataset_cfg["fps"]: frames per second of the video (default: 30)

        sample_video (str):
            Identifier of the video sample (without file extension) used to load
            the corresponding feature (`.npy`) and label (`.json`) files.

    Raises:
        FileNotFoundError:
            If the features directory or labels directory does not exist.

        FileNotFoundError:
            If the feature file (`.npy`) or label file (`.json`) for the specified
            sample video cannot be found.

        ValueError:
            If the loaded feature array is empty or does not have shape `(T, F)`.

        IndexError:
            If a requested feature index in `KEY_FEATURES` exceeds the available
            number of features.

    Returns:
        None
            Displays an interactive Plotly figure showing feature trajectories
            over time with annotated error regions.
    """

    features_dir = Path(dataset_cfg["paths"]["features_dir"])
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    file_path = features_dir / f"{sample_video}.npy"
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    features = np.load(file_path)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"Invalid feature array shape {features.shape} in {file_path.name}"
        )

    labels_dir = Path(dataset_cfg["paths"]["labels_dir"])
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    labels_path = labels_dir / f"{sample_video}.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with open(labels_path) as f:
        label_dict = json.load(f)

    fig = go.Figure()

    fps = dataset_cfg.get("fps", 30)
    t_axis = np.arange(features.shape[0]) / fps

    n_features = features.shape[1]
    for fi in KEY_FEATURES:
        if fi >= n_features:
            raise IndexError(
                f"Feature index {fi} out of bounds (features have {n_features} columns)"
            )

        fig.add_trace(go.Scatter(
            x=t_axis,
            y=features[:, fi],
            mode="lines",
            name=FEATURE_NAMES[fi],
            opacity=0.8,
        ))

    # Shade error regions
    colors = {
        "ohp_elbow": "rgba(239,85,59,0.2)",
        "ohp_knee": "rgba(171,99,250,0.2)",
    }

    for label_name, segments in label_dict.items():
        for seg in segments:
            fig.add_vrect(
                x0=seg[0],
                x1=seg[1],
                fillcolor=colors.get(label_name, "rgba(200,200,200,0.2)"),
                line_width=0,
                annotation_text=label_name.split("_")[-1],
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"Feature Signals — {sample_video}",
        xaxis_title="Time (s)",
        yaxis_title="Feature value",
        width=900,
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )

    fig.show()
