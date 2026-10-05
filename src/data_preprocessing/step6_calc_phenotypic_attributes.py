"""Calculate phenotypic attributes of the main maize plants and plot statistics.
 
This script reads the step5 output and does the following steps:
 
- Calculate the Ground Sample Distance (GSD, in mm/px) of the ROCAS camera setup.
- Calculate 10 phenotypic attributes (PHENO_ATTRIBUTES) for each maize mask and
  convert them from pixels to cm or cm² with the GSD:
    1. pp_cm [cm]: plant perimeter, the contour length of the plant mask.
    2. cp_cm [cm]: convex perimeter, the perimeter of the convex hull.
    3. pcd_cm [cm]: plant circle diameter, the diameter of the minimum enclosing
       circle.
    4. arw_cm [cm]: width (longer side) of the minimum-area rotated bounding box.
    5. arh_cm [cm]: height (shorter side) of the minimum-area rotated bounding box.
    6. pa_cm2 [cm²]: plant area, the area inside the plant contour.
    7. ca_cm2 [cm²]: convex area, the area of the convex hull.
    8. car [0 to 1]: convex area ratio (solidity), pa_cm2 / ca_cm2. Shows how
       compact the plant is.
    9. ppr_cm [cm]: plant area-to-perimeter ratio, pa_cm2 / pp_cm.
    10. cpr_cm [cm]: convex area-to-perimeter ratio, ca_cm2 / cp_cm.
- Add these attributes to the "maize" label in the category schema.
- Plot the correlation of each attribute with BBCH (PNG, for all plants, plants
  with and plants without visible damage).
- Plot the BBCH and damage_visible distribution per subset (HTML).
- Remove all maize attributes except bbch, ppr_cm and damage_visible, and remove
  the image-level metadata attributes.
 
All attributes are calculated from the largest outer contour of the mask.
 
Usage:
    python3 src/data_preprocessing/step6_calc_phenotypic_attributes.py
"""


import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycocotools.mask as mask_util
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from plotly.subplots import make_subplots
from scipy import stats
from tqdm import tqdm


PHENO_ATTRIBUTES = [
    "pp_cm", "cp_cm", "pcd_cm", "arw_cm", "arh_cm", 
    "pa_cm2", "ca_cm2", "car", "ppr_cm", "cpr_cm"
]

FEATURES_TO_PLOT = [
    "leaf_count", "pa_cm2", "pp_cm", "ca_cm2", "cp_cm", "pcd_cm",
    "arw_cm", "arh_cm", "ppr_cm", "cpr_cm", "car"
]

# Minimal set of mask attributes to keep in final JSON output after plotting
KEPT_ATTRIBUTES = {"bbch", "ppr_cm", "damage_visible"}

# Image-level attributes to remove from items
IMAGE_ATTRS_TO_REMOVE = {
    "rope_position_odom_[mm]",
    "timestamp_iso",
    "field_number",
    "row_number",
    "cycle",
}


def calculate_GSD() -> Tuple[float, float]:
    """Calculate the Ground Sample Distance (GSD) of the ROCAS camera setup.
 
    Camera setup (JAI FS-3200T 3-CCD / CMOS multispectral camera):
    - Sensor active area: 7.07 mm (width) x 5.30 mm (height)
    - Pixel pitch: 3.45 µm (0.00345 mm) in both directions
    - Image resolution: 2048 px x 1536 px
    - Focal length: 16.0 mm
    - Distance to ground: 3100 mm (3.1 m)
 
    Returns the GSD in width and height direction in mm/px (both about 0.668 mm/px).
    """
    pixel_distance_width = 0.00345   # [mm] (3.45 µm pixel pitch)
    pixel_distance_height = 0.00345  # [mm] (3.45 µm pixel pitch)

    active_area_width = 7.07   # [mm] JAI FS-3200T sensor active width
    active_area_height = 5.30  # [mm] JAI FS-3200T sensor active height
    
    flight_altitude = 3100  # [mm] Distance from camera to field plane
    focal_length = 16       # [mm] Lens focal length

    image_width = active_area_width / pixel_distance_width    # 2048 px
    image_height = active_area_height / pixel_distance_height  # 1536 px

    gsd_w = (flight_altitude * active_area_width) / (focal_length * image_width)
    gsd_h = (flight_altitude * active_area_height) / (focal_length * image_height)

    return gsd_w, gsd_h  # GSD ~ 0.668359 mm/px


def get_label_name(
    label_id: int, 
    categories_cfg: Dict[str, Any]
) -> str:
    """Return the label name for a label_id from the category schema.
 
    Returns the label name, or an empty string if the label_id is out of range.
    """
    labels = categories_cfg.get("label", {}).get("labels", [])
    if 0 <= label_id < len(labels):
        return labels[label_id].get("name", "")
    return ""


def get_phenotypic_features(
    mask: np.ndarray
) -> Dict[str, float]:
    """Calculate pixel-based shape features of a binary mask.
 
    Only the largest outer contour of the mask is used. Features:
    - pa: plant area (contour area)
    - pp: plant perimeter (contour length)
    - ca: convex area (area of the convex hull)
    - cp: convex perimeter (perimeter of the convex hull)
    - pcd: plant circle diameter (diameter of the minimum enclosing circle)
    - arw: width (longer side) of the minimum-area rotated bounding box
    - arh: height (shorter side) of the minimum-area rotated bounding box
    - car: convex area ratio (pa / ca)
 
    Returns a dict with these features in pixels (or px²), or an empty dict if the
    mask has no contour.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {}

    cnt = max(contours, key=cv2.contourArea)
    pa = cv2.contourArea(cnt)
    pp = cv2.arcLength(cnt, True)
    
    hull = cv2.convexHull(cnt)
    ca = cv2.contourArea(hull)
    cp = cv2.arcLength(hull, True)
    
    _, radius = cv2.minEnclosingCircle(cnt)
    pcd = radius * 2.0
    
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    arw, arh = max(w, h), min(w, h)
    
    car = pa / ca if ca > 0 else 0.0
    
    return {
        "pa": float(pa), "pp": float(pp), 
        "ca": float(ca), "cp": float(cp), 
        "pcd": float(pcd), "arw": float(arw), 
        "arh": float(arh), "car": float(car)
    }


def process_subset_phenology(
    subset_data: Dict[str, Any], 
    gsd: float, 
    subset_name: str
) -> Dict[str, Any]:
    """Calculate the phenotypic attributes of all maize masks in one subset.
 
    The pixel features are converted to cm and cm² with the GSD. Masks that are empty
    or have no contour get 0.0 for all attributes. The attributes are also added to
    the "maize" label in the category schema. The subset is changed in place.
 
    Returns the updated subset.
    """
    gsd_sq = gsd ** 2  # Conversion factor for area (mm²/px²)
    items = subset_data.get("items", [])
    categories_cfg = subset_data.get("categories", {})

    for item in tqdm(items, desc=f"Calculating phenology for {subset_name}"):
        annotations = item.get("annotations", [])
        
        for ann in annotations:
            if ann.get("type") == "mask":
                label_name = get_label_name(ann.get("label_id", -1), categories_cfg)
                if label_name == "maize":
                    if "attributes" not in ann:
                        ann["attributes"] = {}

                    rle = ann.get("rle")
                    if not rle:
                        for attr_name in PHENO_ATTRIBUTES:
                            ann["attributes"][attr_name] = 0.0
                        continue

                    mask = mask_util.decode(rle)
                    
                    if np.sum(mask) == 0:
                        for attr_name in PHENO_ATTRIBUTES:
                            ann["attributes"][attr_name] = 0.0
                        continue

                    features = get_phenotypic_features(mask)
                    if features:
                        # Convert pixel dimensions to physical metric units (mm -> cm, mm² -> cm²)
                        pp_cm = (features["pp"] * gsd) / 10.0      # mm to cm
                        cp_cm = (features["cp"] * gsd) / 10.0      # mm to cm
                        pa_cm2 = (features["pa"] * gsd_sq) / 100.0  # mm² to cm²
                        ca_cm2 = (features["ca"] * gsd_sq) / 100.0  # mm² to cm²

                        ann["attributes"]["pp_cm"] = float(pp_cm)
                        ann["attributes"]["cp_cm"] = float(cp_cm)
                        ann["attributes"]["pcd_cm"] = float((features["pcd"] * gsd) / 10.0)
                        ann["attributes"]["arw_cm"] = float((features["arw"] * gsd) / 10.0)
                        ann["attributes"]["arh_cm"] = float((features["arh"] * gsd) / 10.0)
                        ann["attributes"]["pa_cm2"] = float(pa_cm2)
                        ann["attributes"]["ca_cm2"] = float(ca_cm2)
                        ann["attributes"]["car"] = float(features["car"])
                        
                        ann["attributes"]["ppr_cm"] = float(pa_cm2 / pp_cm) if pp_cm > 0 else 0.0
                        ann["attributes"]["cpr_cm"] = float(ca_cm2 / cp_cm) if cp_cm > 0 else 0.0
                    else:
                        for attr_name in PHENO_ATTRIBUTES:
                            ann["attributes"][attr_name] = 0.0

    if "categories" in subset_data and "label" in subset_data["categories"]:
        labels = subset_data["categories"]["label"].get("labels", [])
        for label_obj in labels:
            if label_obj.get("name") == "maize":
                existing_attrs = set(label_obj.get("attributes", []))
                updated_attrs = sorted(list(existing_attrs.union(set(PHENO_ATTRIBUTES))))
                label_obj["attributes"] = updated_attrs

    return subset_data


def prune_unnecessary_attributes(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Remove attributes that are not needed after plotting.
 
    Maize masks only keep KEPT_ATTRIBUTES (bbch, ppr_cm, damage_visible) and the
    image-level attributes in IMAGE_ATTRS_TO_REMOVE are removed. The category schema
    is updated to match. The subsets are changed in place.
 
    Returns the pruned subsets.
    """
    for subset_name, subset_data in results.items():
        categories_cfg = subset_data.get("categories", {})
        
        for item in subset_data.get("items", []):
            if "attr" in item and isinstance(item["attr"], dict):
                for key in IMAGE_ATTRS_TO_REMOVE:
                    item["attr"].pop(key, None)

            for ann in item.get("annotations", []):
                if ann.get("type") == "mask":
                    label_name = get_label_name(ann.get("label_id", -1), categories_cfg)
                    if label_name == "maize" and "attributes" in ann:
                        ann["attributes"] = {
                            k: v for k, v in ann["attributes"].items() 
                            if k in KEPT_ATTRIBUTES
                        }

        if "categories" in subset_data and "label" in subset_data["categories"]:
            labels = subset_data["categories"]["label"].get("labels", [])
            for label_obj in labels:
                if label_obj.get("name") == "maize":
                    label_obj["attributes"] = sorted(list(KEPT_ATTRIBUTES))

    return results



def extract_dataframe_from_subsets(
    processed_subsets: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Collect the attributes of all maize masks in one DataFrame for plotting.
 
    Returns the DataFrame (one row per maize mask) and a dict with the number of
    images and maize masks per split.
    """
    rows = []
    split_stats = {}

    for split_name, data in processed_subsets.items():
        items = data.get("items", [])
        num_images = len(items)
        plant_count = 0

        for item in items:
            for ann in item.get("annotations", []):
                if ann.get("type") == "mask" and ann.get("label_id") == 0:
                    attrs = ann.get("attributes", {})
                    plant_count += 1

                    row = {
                        "split": split_name,
                        "image_id": item.get("id"),
                        "bbch": int(round(float(attrs.get("bbch", 0)))),
                        "damaged": bool(attrs.get("damage_visible", False)),
                        "leaf_count": attrs.get("leaf_count"),
                        "pa_cm2": attrs.get("pa_cm2"),
                        "pp_cm": attrs.get("pp_cm"),
                        "ca_cm2": attrs.get("ca_cm2"),
                        "cp_cm": attrs.get("cp_cm"),
                        "pcd_cm": attrs.get("pcd_cm"),
                        "arw_cm": attrs.get("arw_cm"),
                        "arh_cm": attrs.get("arh_cm"),
                        "ppr_cm": attrs.get("ppr_cm"),
                        "cpr_cm": attrs.get("cpr_cm"),
                        "car": attrs.get("car"),
                    }
                    rows.append(row)

        split_stats[split_name] = {
            "n_images": num_images,
            "n_plants": plant_count
        }

    df = pd.DataFrame(rows)
    return df, split_stats


def draw_summary_grid(
    df: pd.DataFrame, 
    features: List[str], 
    output_dir: Path, suffix: str = "all"
) -> None:
    """Plot each feature against BBCH in a 3 x 4 grid with a regression line.
 
    Each subplot shows the R² of a linear regression in its title. The plot is saved
    as phenology_BBCH_correlations_{suffix}.png in output_dir. The suffix also sets
    the plot colors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()
    
    color_map = {"damage_not_visible": "blue", "damage_visible": "red", "all": "green"}
    line_map = {"damage_not_visible": "orange", "damage_visible": "black", "all": "red"}
    current_color = color_map.get(suffix, "blue")
    current_line = line_map.get(suffix, "orange")

    for i, feat in enumerate(features):
        if i >= len(axes):
            break

        if feat in df.columns:
            mask = df[feat].notna() & df['bbch'].notna()
            if mask.sum() > 2:
                sns.regplot(
                    data=df, x="bbch", y=feat, ax=axes[i], 
                    scatter_kws={'alpha': 0.3, 'color': current_color, 's': 15}, 
                    line_kws={'color': current_line}
                )
                
                axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                r_sq = stats.linregress(df['bbch'][mask], df[feat][mask])[2]**2  # type: ignore
                axes[i].set_title(f"{feat.upper()}\n(R²: {r_sq:.2f})", fontsize=12)
            else:
                axes[i].set_title(f"{feat.upper()}")
        else:
            axes[i].set_title(f"{feat.upper()}")
            
        axes[i].set_xlabel("BBCH")
        axes[i].grid(True, alpha=0.3)

    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Correlations between phenotypic attributes and BBCH ({suffix.upper()})", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    
    save_path = output_dir / f"phenology_BBCH_correlations_{suffix}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_split_eda_plot(
    df: pd.DataFrame, 
    split_stats: Dict[str, Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Plot the BBCH distribution per subset as stacked bars with Plotly.
 
    Each bar is split into plants with and without visible damage. The y-axis is
    fixed to 0 to 450 so all subsets have the same scale. The plot is saved as
    bbch_distribution.html in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_x_axis = list(range(10, 18))
    
    splits = [s for s in ["train", "val", "test"] if s in df['split'].unique()]
    if not splits:
        return

    y_axis_max = 450

    subplot_titles = []
    for s in splits:
        stats_info = split_stats.get(s, {"n_images": 0, "n_plants": 0})
        subplot_titles.append(
            f"<b>{s.upper()} Set</b><br><sup>{stats_info['n_images']} images, {stats_info['n_plants']} maize masks</sup>"
        )

    fig = make_subplots(
        rows=1, cols=len(splits),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06
    )

    healthy_green = "rgba(15, 60, 25, 0.95)"
    damaged_red = "rgba(220, 20, 60, 0.9)"

    for i, split_name in enumerate(splits):
        col_idx = i + 1
        sub_df = df[df["split"] == split_name]

        y_damaged = [len(sub_df[(sub_df["bbch"] == b) & (sub_df["damaged"] == True)]) for b in fixed_x_axis]
        y_healthy = [len(sub_df[(sub_df["bbch"] == b) & (sub_df["damaged"] == False)]) for b in fixed_x_axis]
        y_total = [dam + heal for dam, heal in zip(y_damaged, y_healthy)]

        fig.add_trace(go.Bar(
            x=fixed_x_axis, y=y_damaged, name="Damage visible", marker_color=damaged_red,
            showlegend=(i == 0), hovertemplate="Damage visible: %{y}<extra></extra>"
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            x=fixed_x_axis, y=y_healthy, name="Damage not visible", marker_color=healthy_green,
            showlegend=(i == 0), 
            text=[str(v) if v > 0 else "" for v in y_total],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="Damage not visible: %{y}<extra></extra>"
        ), row=1, col=col_idx)

        fig.update_xaxes(
            title_text="BBCH",
            tickmode="array", 
            tickvals=fixed_x_axis, 
            range=[9.35, 17.5], 
            row=1, col=col_idx
        )

        fig.update_yaxes(
            title_text="Number of maize masks",
            range=[0, y_axis_max],
            gridcolor="lightgray",
            dtick=50 if y_axis_max > 200 else 20,
            row=1, col=col_idx
        )

    fig.update_layout(
        title=dict(text="BBCH distribution per subset", x=0.5, font=dict(size=22)),
        height=825,
        width=600 * len(splits),
        margin=dict(t=120, b=100, l=90, r=50),
        plot_bgcolor="white",
        barmode="stack",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center")
    )

    out_path = output_dir / "bbch_distribution.html"
    fig.write_html(str(out_path))



def main(
    input_source: Union[Path, Dict[str, Dict[str, Any]]], 
    output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """Calculate the phenotypic attributes, create the plots and prune attributes.
 
    The input can be a directory with the step5 JSON files or a dict with already
    loaded subsets. Missing subsets are skipped. The plots are saved in output_dir.
 
    Returns a dict with the final subsets.
    """
    gsd = calculate_GSD()[0]
    results = {}

    for subset_name in ["train", "val", "test"]:
        if isinstance(input_source, Path):
            subset_file = input_source / f"{subset_name}.json"
            if not subset_file.exists():
                print(f"Warning: Subset file {subset_file.name} not found in input directory. Skipping.")
                continue
            with open(subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)
        else:
            if subset_name not in input_source:
                continue
            subset_data = input_source[subset_name]

        processed_data = process_subset_phenology(subset_data, gsd, subset_name)
        results[subset_name] = processed_data

    print("\nGenerating correlation and distribution plots...")
    df, split_stats = extract_dataframe_from_subsets(results)

    create_split_eda_plot(df, split_stats, output_dir)
    draw_summary_grid(df, FEATURES_TO_PLOT, output_dir, suffix="all")

    df_healthy = df[df["damaged"] == False] 
    draw_summary_grid(df_healthy, FEATURES_TO_PLOT, output_dir, suffix="damage_not_visible")

    df_damaged = df[df["damaged"] == True]  
    draw_summary_grid(df_damaged, FEATURES_TO_PLOT, output_dir, suffix="damage_visible")

    print(f"All plots saved to {output_dir}")

    print("\nPruning unnecessary mask attributes and image attributes...")
    results = prune_unnecessary_attributes(results)

    return results




if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(description="Calculate phenotypic features, draw correlation and distribution plots, finalize jsons.")
    parser.add_argument(
        "--input-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step5"),
        help="Directory containing the output JSONs from step 05"
    )
    parser.add_argument(
        "--output-dir", type=str, 
        default=str(PROJECT_ROOT / "datasets" / "preprocessing" / "outputs" / "step6"),
        help="Directory where the enriched subset JSONs and figures will be saved"
    )

    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    processed_subsets = main(input_source=in_dir, output_dir=out_dir)

    for subset_name, subset_data in processed_subsets.items():
        out_path = out_dir / f"{subset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved cleaned {subset_name}.json to {out_path}")