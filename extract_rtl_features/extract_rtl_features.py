#!/usr/bin/env python3
import os
import subprocess
import json
import csv
from pathlib import Path

# ---- CONFIG ----
# Script is in Final_Project/Mycode/extract_rtl_features/, so go up 2 levels to project root
REPO_ROOT = Path(__file__).resolve().parents[2]
HLS_ROOT = REPO_ROOT / "ML4Accel_Dataset" / "ML4Accel-Dataset" / "fpga_ml_dataset" / "HLS_dataset"
YOSYS_BIN = "yosys"  # change if yosys is not on PATH
OUT_CSV = REPO_ROOT / "rtl_features.csv"


def find_design_dirs_with_hls_and_rtl():
    """Find dirs that contain BOTH post_hls_info.csv and at least one .v file."""
    design_entries = []

    for root, dirs, files in os.walk(HLS_ROOT):
        root_path = Path(root)
        if "post_hls_info.csv" in files:
            hls_csv = root_path / "post_hls_info.csv"
            rtl_files = [p for p in root_path.rglob("*.v")]
            if not rtl_files:
                continue
            design_entries.append((root_path, hls_csv, rtl_files))

    return design_entries


def run_yosys_and_get_json(design_dir, rtl_files, json_out):
    """Run yosys on a list of .v files and write a JSON netlist."""
    file_list = " ".join(str(f) for f in rtl_files)
    yosys_script = (
        f"read_verilog {file_list}; "
        f"hierarchy -auto-top; "
        f"proc; opt; "
        f"write_json {json_out}"
    )

    cmd = [YOSYS_BIN, "-p", yosys_script]
    print(
        f"[YOSYS] Running in {design_dir} with {len(rtl_files)} RTL files...")
    subprocess.run(cmd, check=True, cwd=design_dir)


def extract_features_from_json(json_path):
    """Parse Yosys JSON netlist and extract RTL features."""
    with open(json_path, "r") as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not modules:
        return {}

    # pick top module if marked, else first
    top_name = None
    for name, mod in modules.items():
        attrs = mod.get("attributes", {})
        if attrs.get("top") == 1:
            top_name = name
            break
    if top_name is None:
        top_name = next(iter(modules.keys()))

    mod = modules[top_name]
    cells = mod.get("cells", {})
    netnames = mod.get("netnames", {})
    ports = mod.get("ports", {})

    def count_cells(prefixes=None, contains=None):
        n = 0
        for cell in cells.values():
            ctype = cell["type"].lower()
            if prefixes and any(ctype.startswith(p) for p in prefixes):
                n += 1
            elif contains and any(c in ctype for c in contains):
                n += 1
        return n

    num_add = count_cells(["$add"])
    num_sub = count_cells(["$sub"])
    num_mul = count_cells(["$mul"])
    num_div = count_cells(["$div"])
    num_and = count_cells(["$and"])
    num_or = count_cells(["$or"])
    num_xor = count_cells(["$xor"])
    num_shft = count_cells(["$shl", "$shr", "$sshl", "$sshr"])
    num_cmp = count_cells(["$lt", "$le", "$gt", "$ge", "$eq", "$ne"])
    num_mux = count_cells(["$mux", "$pmux"])
    num_dff = count_cells(contains=["dff"])

    widths = []
    for cell in cells.values():
        for nets in cell.get("connections", {}).values():
            widths.append(len(nets))
    avg_width = (sum(widths) / len(widths)) if widths else 0
    max_width = max(widths) if widths else 0
    min_width = min(widths) if widths else 0

    num_inputs = sum(1 for p in ports.values()
                     if p.get("direction") == "input")
    num_outputs = sum(1 for p in ports.values()
                      if p.get("direction") == "output")
    num_wires = len(netnames)
    num_cells = len(cells)

    return {
        "num_add": num_add,
        "num_sub": num_sub,
        "num_mul": num_mul,
        "num_div": num_div,
        "num_and": num_and,
        "num_or": num_or,
        "num_xor": num_xor,
        "num_shifter": num_shft,
        "num_cmp": num_cmp,
        "num_mux": num_mux,
        "num_dff": num_dff,
        "num_cells": num_cells,
        "num_wires": num_wires,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "avg_signal_width": avg_width,
        "max_signal_width": max_width,
        "min_signal_width": min_width,
    }


def main():
    design_entries = find_design_dirs_with_hls_and_rtl()
    print(f"Found {len(design_entries)} design dirs with both HLS data and RTL.")

    rows = []

    for design_dir, hls_csv, rtl_files in design_entries:
        design_id = design_dir.relative_to(HLS_ROOT)
        json_out = design_dir / "yosys_netlist.json"

        try:
            run_yosys_and_get_json(design_dir, rtl_files, json_out)
            feats = extract_features_from_json(json_out)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Yosys failed for {design_dir}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {design_dir}: {e}")
            continue

        row = {
            "design_id": str(design_id),
            "hls_csv": str(hls_csv.relative_to(REPO_ROOT)),
        }
        row.update(feats)
        rows.append(row)

        try:
            json_out.unlink()
        except FileNotFoundError:
            pass

    if not rows:
        print("No features extracted. Check paths / Yosys setup.")
        return

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
