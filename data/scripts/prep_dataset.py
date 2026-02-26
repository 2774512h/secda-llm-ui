import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Set

import pandas as pd

#--------------------------------
# Helper functions 
#--------------------------------

def ensure_dir(p: Path) -> None:
    """"
    Create or make sure directory exists
    to store processed scripts.
    """
    p.mkdir(parents=True, exist_ok=True)
    
def normalise_run_id(val: Any) -> str:
    """
    Normalise run ids to a 4 digit string. eg "0001"
    """

    s = str(val).strip() 
    m = re.search(r"(\d{4})$", s)
    if m:
        return m.group(1)
    if s.isdigit():
        return s[-4:].zfill(4)
    return s 

def find_one(root: Path, name: str) -> Optional[Path]:
    """
    Find a file or folder in the root
    """
    matches = list(root.rglob(name))
    return matches[0] if matches else None

REQUIRED_REPORTS = [
    "timing_report_impl_full.txt",
    "timing_report_impl_ip.txt",
    "utilization_report_impl_full.txt",
    "utilization_report_impl_ip.txt",
]

#------------------------------------------------------------------------
#Likely no longer needed as a validation column exists in summary reports
#------------------------------------------------------------------------
def determine_run_success(run_dir: Path) -> bool:
    """
    Determine if the run was a success or failure
    """
    results = run_dir / "results"
    if not results.exists():
        return False
    return all((results / f).exists() for f in REQUIRED_REPORTS)

def shorten_results_key(k: str) -> str:
    m = re.match(r"timing_report_impl_(full|ip)\.txt\.(.+)", k)
    if m:
        return f"timing_{m.group(1)}_{m.group(2)}"

    m = re.match(r"utilization_report_impl_(full|ip)\.txt\.(.+)", k)
    if m:
        return f"util_{m.group(1)}_{m.group(2)}"

    # fallback
    return k

def safe_json_loads(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}
#-----------------------------------
# Parse successful runs (no longer required, as data is present in summary reports.)
#-----------------------------------

def parse_timing_report(path: Path) -> Dict[str, Optional[float]]:
    """
    Extract key metrics from timing report
    """
    out = {"wns" : None, "tns": None}
    if not path.exists():
        return out
        
    lines = path.read_text(errors="ignore").splitlines()
    
    design_idx = None
    for i, line in enumerate(lines):
        if "Design Timing Summary" in line:
            design_idx = i
            break

    if design_idx is None:
        design_idx = 0

    header_index = None
    for i in range(design_idx, len(lines)):
        line = lines[i]
        if "WNS(ns)" in line and "TNS(ns)" in line:
            header_index = i
            break
    if header_index is None:
        return out
    
    num_re = re.compile(r"(-?\d+(?:\.\d+)?)")

    for j in range(header_index + 1, min(header_index + 30, len(lines))):
        nums = num_re.findall(lines[j])
        if len(nums) >= 2:
            try:
                out["wns"] = float(nums[0])
                out["tns"] = float(nums[1])
                return out
            except ValueError:
                continue
    return out

def parse_utilisation_report(path: Path) -> Dict[str, Optional[int]]:
    """
    Extract key metrics from the utilisation report
    """
    out = {"lut": None, "ff": None, "bram": None, "dsp": None}
    if not path.exists():
        return out

    text = path.read_text(errors="ignore")

    def grab_int(patterns: List[str]) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    return int(m.group(1).replace(",", ""))
                except Exception:
                    pass
        return None

    out["lut"] = grab_int([
        r"\|\s*Slice\s+LUTs\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*CLB\s+LUTs\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*LUTs\s*\|\s*([\d,]+)\s*\|",
    ])

    out["ff"] = grab_int([
        r"\|\s*Slice\s+Registers\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*CLB\s+Registers\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*Registers\s*\|\s*([\d,]+)\s*\|",
    ])

    out["bram"] = grab_int([
        r"\|\s*Block\s+RAM\s+Tile\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*BRAM\s+Tile\s*\|\s*([\d,]+)\s*\|",
        r"\|\s*BRAMs?\s*\|\s*([\d,]+)\s*\|",
    ])

    out["dsp"] = grab_int([
        r"\|\s*DSPs?\s*\|\s*([\d,]+)\s*\|",
    ])

    return out
#------------------------
# Parses runs.csv
#------------------------
def load_parameter_map(dataset_root: Path) -> Dict[str, str]:
    """
    Extract parameters from runs.csv
    """
    runs_csv = find_one(dataset_root, "runs.csv")
    if runs_csv is None:
        return {}
    
    df = pd.read_csv(runs_csv)
    run_id_col = "run_name" if "run_name" in df.columns else ("run_id" if "run_id" in df.columns else df.columns[0])

    params_col = "params" if "params" in df.columns else None

    mapping: Dict[str, str] = {}

    for _, row in df.iterrows():
        rid = normalise_run_id(row[run_id_col])
        params_str = ""
        if params_col:
            val = row.get(params_col)
            if pd.notna(val):
                params_str = str(val)
        mapping[rid] = params_str

    return mapping

#----------------------------
# Parses both summary reports 
# ---------------------------
def parse_summary_reports(dataset_root: Path, summary_type: str) -> List[Dict[str, Any]]:
    performance_csv = find_one(dataset_root, summary_type)
    if performance_csv is None:
        return []
    try:
        df = pd.read_csv(performance_csv)
        df["run_id"] = df["run_id"].astype(str).str.strip().map(normalise_run_id)
    except Exception:
        return []
    return df.to_dict(orient="records")


#-----------------------------------------
# Parser for run failures
#-----------------------------------------

def parse_vivado_log(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any]= {
        "vivado_version": None,
        "vivado_failed_run": None,
        "vivado_failure_summary_line": None,
        "drc_error_count": None,
        "drc_warning_count": None,
    }

    if path is None or not path.exists():
        return out

    lines = path.read_text(errors="ignore").splitlines()

    for l in lines[:50]:
        m = re.search(r"Vivado\s+v([0-9.]+)", l)
        if m:
            out["vivado_version"] = m.group(1)
            break

    for l in lines:
        m = re.search(r"Failed run\(s\)\s*:\s*'([^']+)'", l)
        if m:
            out["vivado_failed_run"] = m.group(1).strip()
            break

    for l in lines:
        if "Command failed: Run" in l and "failed" in l:
            out["vivado_failure_summary_line"] = l.strip()
            break

    for l in lines:
        m = re.search(r"DRC finished with\s+(\d+)\s+Errors,\s+(\d+)\s+Warnings", l)
        if m:
            out["drc_error_count"] = int(m.group(1))
            out["drc_warning_count"] = int(m.group(2))
            break

    return out


def parse_vivado_hls_log(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "hls_version": None,
        "hls_target_device": None,
        "hls_clock_period_ns": None,
        "hls_failed": None,
    }

    if path is None or not path.exists():
        return out

    lines = path.read_text(errors="ignore").splitlines()

    for l in lines[:30]:
        m = re.search(r"Vivado.*HLS.*v([0-9.]+)", l)
        if m:
            out["hls_version"] = m.group(1)
            break
    for l in lines:
        m = re.search(r"Setting target device to\s+'([^']+)'", l)
        if m:
            out["hls_target_device"] = m.group(1)
            break
    for l in lines:
        m = re.search(r"period of\s+(\d+(?:\.\d+)?)\s*ns", l)
        if m:
            out["hls_clock_period_ns"] = float(m.group(1))
            break
    out["hls_failed"] = any(l.startswith("ERROR:") or l.startswith("FATAL:") for l in lines)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dataset folder or dataset zip")
    ap.add_argument("--out", default="data/processed/run_table.csv", help="Output CSV path")
    ap.add_argument("--extract-dir", default="data/raw_extracted", help="Where to extract zip (if input is zip)")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    out_csv = Path(args.out).resolve()
    ensure_dir(out_csv.parent)

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = Path(args.extract_dir).resolve()
        ensure_dir(extract_dir)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        dataset_root = extract_dir
    elif input_path.is_dir():
        dataset_root = input_path
    else:
        raise RuntimeError("--input must be a dataset directory or .zip file")
    
    runs_dir = find_one(dataset_root, "runs")
    if runs_dir is None or not runs_dir.exists():
        raise RuntimeError("Could not find 'runs/' directory in dataset.")
    
    parameter_map = load_parameter_map(dataset_root)

    perf_rows = parse_summary_reports(dataset_root, "performance_summary.csv")
    perf_by_run = {r["run_id"]: r for r in perf_rows}

    results_rows = parse_summary_reports(dataset_root, "results_summary.csv")
    results_by_run = {r["run_id"]: r for r in results_rows}
    
    rows: List[Dict[str, Any]] = []
    for run_folder in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        rid = run_folder.name
        rid_norm = normalise_run_id(rid)

        success = determine_run_success(run_folder)

        params_str = parameter_map.get(rid_norm, "")
        params_dict = params_str if isinstance(params_str, dict) else safe_json_loads(params_str)

        row = {"run_id": rid_norm,
            "run_path": str(run_folder),
            "successful_run": success,
        }
        
        row.update({k:v for k, v in params_dict.items()})
        perf = perf_by_run.get(rid_norm)
        if perf:
            row.update({f"perf_{k}": v for k, v in perf.items() if k != "run_id"})
        
        results = results_by_run.get(rid_norm)
        if results:
            row.update({
                shorten_results_key(k): v
                for k, v in results.items()
                if k != "run_id"
            })

            
        if not success:
            vivado_log = find_one(run_folder, "vivado.log")
            vivado_hls_log = find_one(run_folder, "vivado_hls.log")

            row.update(parse_vivado_log(vivado_log) if vivado_log else {})
            row.update(parse_vivado_hls_log(vivado_hls_log) if vivado_hls_log else {})

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    main()


