import argparse
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

import pandas as pd

from data_structures import Parameters
from catch_profiles import load_catch_profile
from cpm_enhanced_detector import process_two_edge_reciprocity_folder
from kalman_detector import (
    evaluation_receiver_ids,
    process_cam_cpm_kalman_folder,
    process_kalman_folder,
)
from weighted_reciprocity_detector import (
    process_maintained_trust_reciprocity_folder,
    process_no_anonymous_maintained_trust_folder,
    process_weighted_reciprocity_folder,
)


def worker_process_json(input_file: str, option: int, params_dict: Dict[str, Any], source_file: str):
    """Worker for JSON-File Processing"""
    try:
        import data_processing

        input_path = Path(input_file)
        params = Parameters(**params_dict)
        result = data_processing.process(input_path, option, params, pd.DataFrame(), source_file)
        return {'status': 'ok', 'metrics': result, 'source': source_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'source': source_file}


def worker_process_catch_pair(cam_file, cpm_file, option, params_dict, source_file):
    try:
        import data_processing

        params = Parameters(**params_dict)
        df = data_processing.load_catch_messages(cam_file, cpm_file)
        input_path = Path(cam_file or cpm_file)
        result = data_processing.process(input_path, option, params, df, source_file)
        return {'status': 'ok', 'metrics': result, 'source': source_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'source': source_file}


def worker_process_parquet_group(parquet_file: str, option: int, params_dict: Dict[str, Any], source_file: str):
    """Worker for Parquet-Group Processing"""
    try:
        import data_processing

        parquet_path = Path(parquet_file)
        params = Parameters(**params_dict)
        df_all = pd.read_parquet(parquet_path)

        if 'source_file' in df_all.columns:
            group_df = df_all[df_all['source_file'] == source_file].copy()
        else:
            group_df = df_all

        result = data_processing.process(parquet_path, option, params, group_df, source_file)
        return {'status': 'ok', 'metrics': result, 'source': source_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'source': source_file}


def evaluate_predictions(scenario_stats):
    total_tp = sum(s.get('tp', 0) for s in scenario_stats)
    total_tn = sum(s.get('tn', 0) for s in scenario_stats)
    total_fp = sum(s.get('fp', 0) for s in scenario_stats)
    total_fn = sum(s.get('fn', 0) for s in scenario_stats)

    total_messages = total_tp + total_tn + total_fp + total_fn

    aggregated_metrics = {
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
        'accuracy': (total_tp + total_tn) / total_messages if total_messages > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'false_positive_rate': total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0,
    }
    aggregated_metrics['f1'] = (2 * aggregated_metrics['precision'] * aggregated_metrics['recall'] /
                                (aggregated_metrics['precision'] + aggregated_metrics['recall'])
                                if (aggregated_metrics['precision'] + aggregated_metrics['recall']) > 0 else 0)
    return aggregated_metrics


def add_catch_output(aggregated_metrics, metrics, profile, params):
    aggregated_metrics["catch_enabled"] = metrics["catch_enabled"]
    aggregated_metrics["initial_covariance_diag"] = metrics["initial_covariance_diag"]
    aggregated_metrics["measurement_noise_diag"] = metrics["measurement_noise_diag"]
    aggregated_metrics["cpm_association_noise_diag"] = metrics[
        "cpm_association_noise_diag"
    ]
    aggregated_metrics["process_noise_model"] = metrics["process_noise_model"]
    aggregated_metrics["catch_profile"] = profile if metrics["catch_enabled"] else None
    aggregated_metrics["catch_metrics"] = metrics["catch_metrics"]
    aggregated_metrics["catch_check_activations"] = metrics["catch_check_activations"]
    aggregated_metrics["kalman_skipped"] = metrics["kalman_skipped"]
    aggregated_metrics["parameters"] = vars(params) if metrics["catch_enabled"] else None
    aggregated_metrics["position_plausibility_check_enabled"] = (
        params.POSITION_PLAUSIBILITY_ENABLED if metrics["catch_enabled"] else None
    )


def result_directory(input_folder: Path, detection_type: str):
    """Store results as results/<attack_type>/<detection_type>/."""
    output_dir = input_folder.parent / "results" / input_folder.name / detection_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Pfad zu den Eingabedateien", required=True)
    parser.add_argument(
        "--type",
        type=int,
        choices=[*range(7), 20, 100],
        help="0 = catch-checks, 1 = legacy checks, 2 = CAM-only Kalman, 3 = CAM+CPM Kalman, 4 = two-edge reciprocal CPM Kalman, 5 = three-score average reciprocity, 6 = PRV, 20 = maintained trust without anonymous CPM tracks, 100 = alias-aware catch",
        required=True,
    )
    parser.add_argument(
        "--catch-profile",
        choices=["urban-low", "urban-high", "highway-low", "highway-high"],
        default="urban-low",
    )
    parser.add_argument(
        "--no-pos-check",
        action="store_true",
        help="Disable only CaTCH's road-edge position plausibility check",
    )
    parser.add_argument(
        "--no-catch",
        action="store_true",
        help="Run Kalman detector types without the CaTCH gate",
    )
    parser.add_argument("--train", type=float)
    parser.add_argument("--parameter", required=False,default=None)
    parser.add_argument('--mpr', required=False, type=float)
    parser.add_argument('--msar', required=False, type=float)
    parser.add_argument('--mpdn', required=False, type=float)
    parser.add_argument('--mps', required=False, type=float)
    parser.add_argument('--mpa', required=False, type=float)
    parser.add_argument('--mpd', required=False, type=float)
    parser.add_argument('--mhc', required=False, type=float)
    parser.add_argument('--mdi', required=False, type=float)
    parser.add_argument('--mtd', required=False, type=float)
    parser.add_argument('--pht', required=False, type=float)
    parser.add_argument('--mmru', required=False, type=float)
    parser.add_argument('--mmrd', required=False, type=float)
    parser.add_argument('--msat', required=False, type=float)
    parser.add_argument('--mnrs', required=False, type=float)
    parser.add_argument('--workers', required=False, type=int, default=os.cpu_count() or 4,
                        help="Number of parallel processes (default: CPU count)")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    scenario_stats = []
    catch_enabled = not args.no_catch

    # Parameter Setup
    if int(args.type) in (0, 2, 3, 4, 5, 6, 20, 100) and args.train != 1:
        params = load_catch_profile(args.catch_profile)
    elif args.train == 1:
        params = Parameters(MAX_PLAUSIBLE_RANGE=args.mpr,
                            MAX_SA_RANGE=args.msar,
                            MAX_PLAUSIBLE_DIST_NEGATIVE=args.mpdn,
                            MAX_PLAUSIBLE_SPEED=args.mps,
                            MAX_PLAUSIBLE_ACCEL=args.mpa,
                            MAX_PLAUSIBLE_DECEL=args.mpd,
                            MAX_HEADING_CHANGE=args.mhc,
                            MAX_DELTA_INTERSECTION=args.mdi,
                            MAX_TIME_DELTA=args.mtd,
                            POS_HEADING_TIME=args.pht,
                            MAX_MGT_RNG_UP=args.mmru,
                            MAX_MGT_RNG_DOWN=args.mmrd,
                            MAX_SA_TIME=args.msat,
                            MAX_NON_ROUTE_SPEED=args.mnrs)
    elif args.parameter is not None:
        with open(args.parameter, 'r') as f:
            data = json.load(f)

        p = data['parameters']

        params = Parameters(
            MAX_PLAUSIBLE_RANGE=p['mpr'],
            MAX_SA_RANGE=args.msar,
            MAX_PLAUSIBLE_DIST_NEGATIVE=p['mpdn'],
            MAX_PLAUSIBLE_SPEED=p['mps'],
            MAX_PLAUSIBLE_ACCEL=p['mpa'],
            MAX_PLAUSIBLE_DECEL=p['mpd'],
            MAX_HEADING_CHANGE=p['mhc'],
            MAX_DELTA_INTERSECTION=p['mdi'],
            MAX_TIME_DELTA=p['mtd'],
            POS_HEADING_TIME=p['pht'],
            MAX_MGT_RNG_UP=p['mmru'],
            MAX_MGT_RNG_DOWN=p['mmrd'],
            MAX_SA_TIME=args.msat,
            MAX_NON_ROUTE_SPEED=p['mnrs']
        )
    else:
        params = Parameters()

    params.POSITION_PLAUSIBILITY_ENABLED = not args.no_pos_check

    params_dict = vars(params)
    count = 0
    max_workers = args.workers
    excluded_attacker_receivers = 0

    # *** Added Section for Kalman procedure, each type of evalutation ***
    if int(args.type) == 2:
        metrics, debug_results = process_kalman_folder(
            input_folder, params, catch_enabled
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "nis_threshold",
            "process_noise_intensity",
            "known_alias_nis_threshold", "max_kalman_prediction_gap_s",
            "max_association_prediction_gap_s",
            "total_messages", "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_only" + ("" if catch_enabled else "_no_catch")
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            # Kalman debug output: one compact row per CAM decision for FP/rejection analysis.
            with open(debug_file, 'w') as f:
                json.dump(debug_results.where(pd.notnull(debug_results), None).to_dict(orient='records'), f, indent=4)
            print(f"Saved debug in {debug_file}")
        return

    if int(args.type) == 3:
        # Type 3 adds the CPM perceived-object branch while preserving type 2 as CAM-only.
        metrics, debug_results = process_cam_cpm_kalman_folder(
            input_folder, params, catch_enabled=catch_enabled
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "cpm_sensor_range_m",
            "nis_threshold", "known_alias_nis_threshold",
            "max_kalman_prediction_gap_s", "max_association_prediction_gap_s",
            "process_noise_intensity",
            "total_messages", "cam_messages", "cpm_messages",
            "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_cpm" + ("" if catch_enabled else "_no_catch")
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            with open(debug_file, 'w') as f:
                json.dump(debug_results.where(pd.notnull(debug_results), None).to_dict(orient='records'), f, indent=4)
            print(f"Saved debug in {debug_file}")
        return

    if int(args.type) == 4:
        metrics, debug_results = process_two_edge_reciprocity_folder(
            input_folder, params, catch_enabled,
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "cpm_sensor_range_m",
            "nis_threshold", "known_alias_nis_threshold",
            "max_kalman_prediction_gap_s", "max_association_prediction_gap_s",
            "process_noise_intensity",
            "total_messages", "cam_messages", "cpm_messages",
            "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_cpm_enhanced_two_edges" + ("" if catch_enabled else "_no_catch")
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            with open(debug_file, 'w') as f:
                json.dump(debug_results.where(pd.notnull(debug_results), None).to_dict(orient='records'), f, indent=4)
            print(f"Saved debug in {debug_file}")
        return

    if int(args.type) == 5:
        metrics, debug_results = process_weighted_reciprocity_folder(
            input_folder, params, catch_enabled
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "cpm_sensor_range_m",
            "nis_threshold", "known_alias_nis_threshold",
            "max_kalman_prediction_gap_s", "max_association_prediction_gap_s",
            "reciprocity_nis_threshold",
            "process_noise_intensity",
            "total_messages", "cam_messages", "cpm_messages",
            "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_cpm_averaged_reciprocity" + (
                "" if catch_enabled else "_no_catch"
            )
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            with open(debug_file, 'w') as f:
                json.dump(
                    debug_results.where(pd.notnull(debug_results), None)
                    .to_dict(orient='records'),
                    f,
                    indent=4,
                )
            print(f"Saved debug in {debug_file}")
        return

    if int(args.type) == 6:
        metrics, debug_results = process_maintained_trust_reciprocity_folder(
            input_folder, params, catch_enabled
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "cpm_sensor_range_m",
            "nis_threshold", "known_alias_nis_threshold",
            "max_kalman_prediction_gap_s", "max_association_prediction_gap_s",
            "reciprocity_nis_threshold",
            "minimum_trust_updates",
            "process_noise_intensity",
            "total_messages", "cam_messages", "cpm_messages",
            "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_cpm_prv" + (
                "" if catch_enabled else "_no_catch"
            )
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            with open(debug_file, 'w') as f:
                json.dump(
                    debug_results.where(pd.notnull(debug_results), None)
                    .to_dict(orient='records'),
                    f,
                    indent=4,
                )
            print(f"Saved debug in {debug_file}")
        return

    if int(args.type) == 20:
        metrics, debug_results = process_no_anonymous_maintained_trust_folder(
            input_folder, params, catch_enabled
        )
        scenario_stats.append(metrics)
        aggregated_metrics = evaluate_predictions(scenario_stats)
        for key in [
            "wireless_range_m", "range_margin_m", "cpm_sensor_range_m",
            "nis_threshold", "known_alias_nis_threshold",
            "max_kalman_prediction_gap_s", "max_association_prediction_gap_s",
            "reciprocity_nis_threshold",
            "process_noise_intensity",
            "total_messages", "cam_messages", "cpm_messages",
            "evaluated_receivers", "excluded_attacker_receivers",
        ]:
            aggregated_metrics[key] = metrics.get(key)
        add_catch_output(aggregated_metrics, metrics, args.catch_profile, params)
        print(aggregated_metrics['f1'])

        if args.train == 0:
            result_name = "kalman_cam_cpm_maintained_trust_no_anonymous" + (
                "" if catch_enabled else "_no_catch"
            )
            output_dir = result_directory(input_folder, result_name)
            output_file = output_dir / "predicted.json"
            debug_file = output_dir / "debug.json"
            print(f"Saved in {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_metrics, f, indent=4)
            with open(debug_file, 'w') as f:
                json.dump(
                    debug_results.where(pd.notnull(debug_results), None)
                    .to_dict(orient='records'),
                    f,
                    indent=4,
                )
            print(f"Saved debug in {debug_file}")
        return

    print(f"Starting processing with {max_workers} workers...", file=sys.stderr)

    nested_catch = (
        int(args.type) in (0, 100)
        and (input_folder / "cam").is_dir()
        and (input_folder / "cpm").is_dir()
    )
    data_folder = input_folder

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        if nested_catch:
            cam_files = {path.stem: path for path in (input_folder / "cam").glob("*.json")}
            cpm_files = {path.stem: path for path in (input_folder / "cpm").glob("*.json")}
            all_receiver_ids = sorted(set(cam_files) | set(cpm_files))
            receiver_ids = evaluation_receiver_ids(input_folder, all_receiver_ids)
            excluded_attacker_receivers = len(all_receiver_ids) - len(receiver_ids)
            total_files = len(receiver_ids)
            for receiver_id in receiver_ids:
                f = executor.submit(
                    worker_process_catch_pair,
                    str(cam_files[receiver_id]) if receiver_id in cam_files else None,
                    str(cpm_files[receiver_id]) if receiver_id in cpm_files else None,
                    int(args.type), params_dict, receiver_id,
                )
                futures[f] = receiver_id

        parquet_files = list(data_folder.glob('*.parquet'))

        if nested_catch:
            pass
        elif parquet_files:
            # Parquet Verarbeitung
            parquet_file = parquet_files[0]
            df_all = pd.read_parquet(parquet_file)

            if 'source_file' in df_all.columns:
                grouped = df_all.groupby('source_file')
                total_files = len(grouped)
                for vehicle_id, _ in grouped:
                    f = executor.submit(worker_process_parquet_group, str(parquet_file),
                                        int(args.type), params_dict, vehicle_id)
                    futures[f] = vehicle_id
            else:
                total_files = 1
                source_name = parquet_file.stem
                f = executor.submit(worker_process_parquet_group, str(parquet_file),
                                    int(args.type), params_dict, source_name)
                futures[f] = source_name
        else:
            # JSON Verarbeitung
            json_files = [f for f in data_folder.glob('*.json')
                          if "ground_truth" not in f.name.lower()]
            total_files = len(json_files)

            for json_file in json_files:
                source_name = json_file.stem
                f = executor.submit(worker_process_json, str(json_file),
                                    int(args.type), params_dict, source_name)
                futures[f] = source_name

        if total_files == 0:
            raise ValueError(f"No input files found in {data_folder}")

        # Ergebnisse sammeln
        for future in as_completed(futures):
            source = futures[future]
            try:
                res = future.result()
                if res.get('status') == 'ok':
                    scenario_stats.append(res.get('metrics', {}))
                    count += 1
                    print(f"Processed prediction for file {count}/{total_files}: {source}", file=sys.stderr)
                else:
                    count += 1
                    print(f"[ERROR] {source}: {res.get('error')}", file=sys.stderr)
            except Exception as e:
                count += 1
                print(f"[ERROR] processing {source}: {e}", file=sys.stderr)

    # Auswertung
    aggregated_metrics = evaluate_predictions(scenario_stats)
    print(aggregated_metrics['f1'])

    if args.train == 0:
        detection_type = {
            0: "catch",
            100: "catch_alias",
        }.get(int(args.type), "legacy")
        if int(args.type) in (0, 100):
            aggregated_metrics["catch_profile"] = args.catch_profile
            aggregated_metrics["identity"] = (
                "sender_id" if int(args.type) == 0 else "sender_alias"
            )
            aggregated_metrics["alias_grace_messages"] = sum(
                result.get("alias_grace_messages", 0) for result in scenario_stats
            )
            aggregated_metrics["total_messages"] = sum(
                result.get("total_messages", 0) for result in scenario_stats
            )
            aggregated_metrics["cam_messages"] = sum(
                result.get("cam_messages", 0) for result in scenario_stats
            )
            aggregated_metrics["cpm_messages"] = sum(
                result.get("cpm_messages", 0) for result in scenario_stats
            )
            activation_names = {
                name
                for result in scenario_stats
                for name in result.get("check_activations", {})
            }
            aggregated_metrics["check_activations"] = {
                name: sum(
                    result.get("check_activations", {}).get(name, 0)
                    for result in scenario_stats
                )
                for name in sorted(activation_names)
            }
            aggregated_metrics["parameters"] = params_dict
            aggregated_metrics["position_plausibility_check_enabled"] = (
                params.POSITION_PLAUSIBILITY_ENABLED
            )
            aggregated_metrics["evaluated_receivers"] = total_files
            aggregated_metrics["excluded_attacker_receivers"] = excluded_attacker_receivers
        output_dir = result_directory(input_folder, detection_type)
        output_file = output_dir / "predicted.json"
        print(f"Saved in {output_file}")
        with open(output_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=4)
        if int(args.type) in (0, 100):
            debug_file = output_dir / "debug.json"
            with open(debug_file, 'w') as f:
                json.dump(scenario_stats, f, indent=4)


if __name__ == "__main__":
    main()
