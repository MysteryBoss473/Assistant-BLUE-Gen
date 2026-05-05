"""
lancement/forecast_init.py
Script d'initialisation et de validation du module Signal/Forecast.

Lance :
  1. L'ingestion de tous les fichiers .xlsx dans dataset/signaux/
  2. L'analyse statistique de chaque signal
  3. Un forecast de démonstration (5 périodes) sur chaque signal
  4. Un rapport de synthèse affiché en console et sauvegardé en JSON

Usage :
    python lancement/forecast_init.py [--data-dir dataset/signaux] [--periods 5] [--out lancement/forecast_report.json]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Permettre l'import depuis la racine du projet
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.signal_forecast import SignalForecastPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "─" * 70


def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def run_init(data_dir: str, periods: int, out_path: str) -> dict:
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": data_dir,
        "periods_forecast": periods,
        "ingestion": {},
        "signals": [],
        "errors": [],
    }

    pipeline = SignalForecastPipeline()

    # ── 1. Ingestion ──────────────────────────────────────────────────────────
    print_section("1 · INGESTION DES DONNÉES")
    ingest_result = pipeline.ingest_directory(data_dir)
    report["ingestion"] = ingest_result

    if ingest_result["status"] != "success":
        logger.error(f"✗ Ingestion échouée : {ingest_result.get('error')}")
        return report

    n_signals = ingest_result["signals_loaded"]
    print(f"  ✓ {n_signals} signal(s) chargé(s) depuis '{data_dir}'")
    if ingest_result.get("files_failed", 0):
        print(f"  ⚠ {ingest_result['files_failed']} fichier(s) en erreur")

    available = ingest_result.get("available_signals", [])
    print(f"\n  Signaux disponibles ({len(available)}) :")
    for key in available:
        print(f"    • {key}")

    # ── 2. Analyse + Forecast par signal ──────────────────────────────────────
    print_section("2 · ANALYSE ET PRÉDICTION PAR SIGNAL")
    for key in available:
        print(f"\n  ▸ {key}")
        try:
            result = pipeline.run(signal_key=key, periods=periods)
            analysis = result.get("analysis", {})
            forecast = result.get("forecast", {})

            # Résumé analyse
            print(f"    Points historiques : {analysis.get('n_points', '?')}")
            print(f"    Moyenne            : {analysis.get('mean', 0):,.2f}")
            print(f"    Tendance           : {analysis.get('trend', '?')}")
            if analysis.get("is_stationary") is not None:
                sta = "oui" if analysis["is_stationary"] else "non"
                print(f"    Stationnaire (ADF) : {sta} (p={analysis.get('adf_pvalue', 0):.3f})")

            # Résumé prédiction
            if forecast.get("status") == "success":
                model = forecast.get("model", "?")
                pts = forecast.get("forecast", [])
                print(f"    Modèle utilisé     : {model}")
                print(f"    Prédictions ({len(pts)} pts) :")
                for p in pts:
                    print(f"      {p['label']} → {p['value']:,.2f}  "
                          f"[{p['lower']:,.2f} – {p['upper']:,.2f}]")
            else:
                print(f"    ✗ Prédiction échouée : {forecast.get('error')}")

            report["signals"].append({
                "key": key,
                "analysis": analysis,
                "forecast": forecast,
            })

        except Exception as e:
            msg = f"Erreur sur '{key}' : {e}"
            logger.exception(msg)
            report["errors"].append(msg)

    # ── 3. Rapport JSON ───────────────────────────────────────────────────────
    print_section("3 · RAPPORT")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Rapport sauvegardé → {out_file}")

    n_ok = len(report["signals"])
    n_err = len(report["errors"])
    print(f"\n  Bilan : {n_ok} signal(s) traité(s), {n_err} erreur(s)")

    if n_err:
        print("\n  Erreurs rencontrées :")
        for e in report["errors"]:
            print(f"    ✗ {e}")

    print(f"\n{SEP}")
    print("  ✅ Initialisation terminée — module Signal prêt.")
    print(SEP)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Initialise et valide le module Signal/Forecast de BLUE-Gen"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset/signaux",
        help="Répertoire contenant les fichiers .xlsx de signaux",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=5,
        help="Nombre de périodes à prévoir lors du test de démonstration",
    )
    parser.add_argument(
        "--out",
        default="lancement/forecast_report.json",
        help="Chemin du rapport JSON de sortie",
    )
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  BLUE-Gen · Module Signal — Initialisation")
    print("═" * 70)
    print(f"  Répertoire données : {args.data_dir}")
    print(f"  Périodes prévues   : {args.periods}")

    report = run_init(args.data_dir, args.periods, args.out)

    # Code de sortie non-zero si des erreurs critiques surviennent
    if report["ingestion"].get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()