"""Run the complete preprocessing pipeline for the RocasMaize_BBCH dataset.
 
This script runs all preprocessing steps one after the other. Each step gets the
output of the previous step in memory, so no intermediate JSON files are written:
 
- step1: Merge the RocasMaize annotations into the RocasMaize_BBCH subsets.
- step2: Export the two Mask2Former datasets from the step1 output.
- step2: Create the distance and direction regression masks from the step1 output.
- step3: Crop the single target maize plants.
- step4: Create the semantic segmentation masks.
- step5: Keep only the main plant annotation of each crop.
- step6: Calculate the phenotypic attributes and create the figures.
 
Output structure in output_dir:
    mask2former/        Mask2Former datasets (maize_class and bbch_classes)
    MTL/images/         cropped plant images per split
    MTL/annotations/    final train.json, val.json and test.json
    MTL/figures/        correlation and distribution plots
 
Usage:
    python3 src/data_preprocessing/preprocess.py
"""


import argparse
import json
from pathlib import Path

import step1_merge_RocasMaize_into_RocasMaize_BBCH as step1_merge_RocasMaize_into_RocasMaize_BBCH
import step2_create_mask2former_datasets as step2_create_mask2former_datasets
import step2_create_regression_masks as step2_create_regression_masks
import step3_crop_single_plants as step3_crop_single_plants
import step4_create_semantic_masks as step4_create_semantic_masks
import step5_filter_target_labels as step5_filter_target_labels
import step6_calc_phenotypic_attributes as step6_calc_phenotypic_attributes



def main() -> None:
    """Parse the command line arguments and run all preprocessing steps.
 
    The results are saved in the folder given by --output-dir (see module docstring).
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    parser = argparse.ArgumentParser(
        description="Run complete preprocessing pipeline to create the RocasMaize_BBCH_processed dataset."
    )

    parser.add_argument(
        "--RocasMaize-json",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize" / "annotations_datumaro" / "all.json"),
        help="Path to RocasMaize all.json",
    )
    parser.add_argument(
        "--RocasMaize_BBCH-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize_BBCH" / "annotations_datumaro"),
        help="Directory with BBCH train/val/test split JSONs",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize" / "images" / "all"),
        help="Directory containing original full-field images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "RocasMaize_BBCH_processed"),
        help="Target directory for final JSON annotations, cropped patches, figures, and Mask2Former exports",
    )
    parser.add_argument(
        "--target-label-id",
        type=int,
        default=0,
        help="Label ID to retain in Step 5 (default: 0 for maize)",
    )

    args = parser.parse_args()

    rocasmaize_json = Path(args.RocasMaize_json)
    rocasmaize_bbch_dir = Path(args.RocasMaize_BBCH_dir)
    images_dir = Path(args.images_dir)

    final_out_dir = Path(args.output_dir)
    cropped_images_dir = final_out_dir / "MTL/images"
    annotations_dir = final_out_dir / "MTL/annotations"
    figures_dir = final_out_dir / "MTL/figures"
    mask2former_out_dir = final_out_dir / "mask2former"

    print("=== MTL & Mask2Former - Step 1: Merging RocasMaize into RocasMaize_BBCH Subsets ===")
    step1_outputs = step1_merge_RocasMaize_into_RocasMaize_BBCH.main(
        rocasmaize_source=rocasmaize_json, 
        rocasmaize_bbch_subsets_source=rocasmaize_bbch_dir
    )

    print("\n=== Mask2Former - Step 2: Exporting Mask2Former Datasets ===")
    step2_create_mask2former_datasets.main(
        input_source=step1_outputs,
        output_dir=mask2former_out_dir
    )

    print("\n=== MTL - Step 2: Generating Base64 Compressed Regression Masks ===")
    step2_outputs = step2_create_regression_masks.main(
        input_source=step1_outputs
    )

    print("\n=== MTL - Step 3: Cropping Single Plants & Updating BBoxes/Masks ===")
    step3_outputs = step3_crop_single_plants.main(
        input_source=step2_outputs,
        images_dir=images_dir,
        crop_images_output_dir=cropped_images_dir,
    )

    print("\n=== MTL - Step 4: Generating Semantic RLE Segmentation Masks ===")
    step4_outputs = step4_create_semantic_masks.main(
        input_source=step3_outputs
    )

    print("\n=== MTL- Step 5: Filtering Target Labels (Main Plant Only) ===")
    step5_outputs = step5_filter_target_labels.main(
        input_source=step4_outputs,
        target_label_id=args.target_label_id
    )

    print("\n=== MTL - Step 6: Adding Phenotypic Attributes & Generating Figures ===")
    final_outputs = step6_calc_phenotypic_attributes.main(
        input_source=step5_outputs,
        output_dir=figures_dir
    )

    print("=== Saving MTL Dataset Annotations ===")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    for subset_name, subset_data in final_outputs.items():
        out_file = annotations_dir / f"{subset_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(subset_data, f, indent=4, ensure_ascii=False)
        print(f"Saved: {out_file}")

    print(f"\nFinal datasets location: {final_out_dir}")


if __name__ == "__main__":
    main()