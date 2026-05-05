"""
core/signal_forecast.py
Pipeline de forecasting de signaux pour BLUE-Gen.

Fonctionnalités :
  - Chargement de fichiers .xlsx multi-feuilles (2 tableaux par fichier supportés)
  - Chaque colonne hors 'annee' est traitée comme un signal indépendant
  - Analyse statistique (tendance, saisonnalité, stationnarité)
  - Forecasting via Prophet (si dispo) ou fallback statsmodels/Holt-Winters
  - Intervalles de confiance (enveloppe d'incertitude)
  - Export JSON compatible avec le frontend Streamlit
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Détection des librairies de forecasting ────────────────────────────────────
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("✓ Prophet disponible")
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠ Prophet non disponible – fallback Holt-Winters")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
    logger.info("✓ statsmodels disponible")
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("⚠ statsmodels non disponible – fallback numpy linéaire")


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires de chargement Excel
# ══════════════════════════════════════════════════════════════════════════════

def _find_table_starts(sheet_df: pd.DataFrame) -> list[int]:
    """
    Détecte les positions (index de ligne) où commence chaque tableau
    dans une feuille brute. Un tableau commence là où 'annee' apparaît
    dans l'une des colonnes (insensible à la casse / aux espaces).
    """
    starts = []
    for i, row in sheet_df.iterrows():
        row_lower = [str(v).strip().lower() for v in row.values]
        if "annee" in row_lower or "année" in row_lower:
            starts.append(i)
    return starts


def _parse_single_table(raw: pd.DataFrame, header_row: int) -> pd.DataFrame | None:
    """
    Extrait un tableau propre depuis une feuille brute à partir de header_row.
    Retourne un DataFrame avec une colonne 'annee' (int) et les colonnes de signaux.
    """
    # Sélectionner les lignes à partir du header
    sub = raw.iloc[header_row:].copy()
    sub.columns = [str(v).strip() for v in sub.iloc[0].values]
    sub = sub.iloc[1:].reset_index(drop=True)

    # Renommer la colonne 'annee' / 'année' de façon uniforme
    rename_map = {}
    for col in sub.columns:
        if col.lower().replace("é", "e") == "annee":
            rename_map[col] = "annee"
            break
    sub = sub.rename(columns=rename_map)

    if "annee" not in sub.columns:
        return None

    # Supprimer les lignes vides (fin de tableau)
    sub = sub.dropna(subset=["annee"])
    sub = sub[sub["annee"].astype(str).str.strip() != ""]

    # Convertir 'annee' en entier
    try:
        sub["annee"] = pd.to_numeric(sub["annee"], errors="coerce")
        sub = sub.dropna(subset=["annee"])
        sub["annee"] = sub["annee"].astype(int)
    except Exception:
        return None

    # Convertir les colonnes de signaux en numérique
    signal_cols = [c for c in sub.columns if c != "annee"]
    for col in signal_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    # Supprimer les colonnes entièrement vides
    sub = sub.dropna(axis=1, how="all")

    return sub if len(sub) >= 2 else None


def load_xlsx_signals(file_path: str) -> dict[str, pd.DataFrame]:
    """
    Charge tous les signaux d'un fichier .xlsx.
    Gère plusieurs feuilles et plusieurs tableaux par feuille.

    Retourne un dict :
      { "NomFeuille__NomColonne": DataFrame(annee, value), ... }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Lire toutes les feuilles brutes (sans parser d'en-tête automatiquement)
    raw_sheets = pd.read_excel(path, sheet_name=None, header=None)
    signals: dict[str, pd.DataFrame] = {}

    for sheet_name, raw in raw_sheets.items():
        starts = _find_table_starts(raw)

        if not starts:
            logger.warning(f"Feuille '{sheet_name}' : aucune colonne 'annee' trouvée")
            continue

        for idx, header_row in enumerate(starts):
            table_df = _parse_single_table(raw, header_row)
            if table_df is None:
                continue

            # Préfixe de feuille + numéro de tableau si plusieurs
            table_prefix = f"{sheet_name}__T{idx + 1}" if len(starts) > 1 else sheet_name

            signal_cols = [c for c in table_df.columns if c != "annee"]
            for col in signal_cols:
                key = f"{table_prefix}__{col}"
                df_signal = table_df[["annee", col]].dropna().copy()
                df_signal.columns = ["annee", "value"]
                df_signal = df_signal.sort_values("annee").reset_index(drop=True)
                signals[key] = df_signal
                logger.info(f"  ✓ Signal chargé : '{key}' ({len(df_signal)} points)")

    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  Analyse statistique
# ══════════════════════════════════════════════════════════════════════════════

def analyze_signal(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse statistique basique d'un signal (colonne 'value').
    Retourne un dict avec : stats descriptives, tendance, stationnarité.
    """
    values = df["value"].dropna()
    result: dict[str, Any] = {
        "n_points": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "trend": "stable",
        "is_stationary": None,
        "adf_pvalue": None,
    }

    # Tendance linéaire
    if len(values) >= 3:
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values.values, 1)
        if slope > result["std"] * 0.05:
            result["trend"] = "croissante"
        elif slope < -result["std"] * 0.05:
            result["trend"] = "décroissante"
        result["slope"] = float(slope)

    # Test de stationnarité ADF
    if STATSMODELS_AVAILABLE and len(values) >= 5:
        try:
            adf_stat, p_value, *_ = adfuller(values.dropna())
            result["is_stationary"] = bool(p_value < 0.05)
            result["adf_pvalue"] = float(p_value)
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Forecasting
# ══════════════════════════════════════════════════════════════════════════════

def _forecast_prophet(df: pd.DataFrame, periods: int,
                      confidence: float = 0.95) -> dict[str, Any]:
    """Forecast via Meta Prophet."""
    ds = pd.date_range(
        start=pd.Timestamp(f"{int(df['annee'].iloc[0])}-01-01"),
        periods=len(df),
        freq="YS",
    )
    prophet_df = pd.DataFrame({"ds": ds, "y": df["value"].values})

    m = Prophet(
        interval_width=confidence,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(prophet_df)

    future = m.make_future_dataframe(periods=periods, freq="YS")
    forecast = m.predict(future)
    future_fc = forecast.iloc[len(df):]

    last_year = int(df["annee"].iloc[-1])
    points = []
    for i, row in future_fc.iterrows():
        year = last_year + (len(points) + 1)
        points.append({
            "label": str(year),
            "value": float(row["yhat"]),
            "lower": float(row["yhat_lower"]),
            "upper": float(row["yhat_upper"]),
        })

    return {"status": "success", "model": "Prophet", "forecast": points,
            "periods": periods, "confidence_level": confidence}


def _forecast_holtwinters(df: pd.DataFrame, periods: int,
                          confidence: float = 0.95) -> dict[str, Any]:
    """Forecast via Holt-Winters (statsmodels)."""
    values = df["value"].values.astype(float)
    last_year = int(df["annee"].iloc[-1])

    trend_type = "add" if len(values) >= 4 else None
    model = ExponentialSmoothing(
        values,
        trend=trend_type,
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc_mean = fit.forecast(periods)

    # Intervalle de confiance via résidus
    residuals = values - fit.fittedvalues
    sigma = np.std(residuals)
    z = 1.96 if confidence >= 0.95 else 1.645
    margin = z * sigma * np.sqrt(np.arange(1, periods + 1))

    points = []
    for i in range(periods):
        year = last_year + i + 1
        points.append({
            "label": str(year),
            "value": float(fc_mean[i]),
            "lower": float(fc_mean[i] - margin[i]),
            "upper": float(fc_mean[i] + margin[i]),
        })

    return {"status": "success", "model": "Holt-Winters", "forecast": points,
            "periods": periods, "confidence_level": confidence}


def _forecast_linear(df: pd.DataFrame, periods: int,
                     confidence: float = 0.95) -> dict[str, Any]:
    """Fallback : régression linéaire + intervalle basé sur les résidus."""
    values = df["value"].values.astype(float)
    x = np.arange(len(values))
    coef = np.polyfit(x, values, 1)
    poly = np.poly1d(coef)
    fitted = poly(x)
    residuals = values - fitted
    sigma = np.std(residuals)
    z = 1.96 if confidence >= 0.95 else 1.645
    last_year = int(df["annee"].iloc[-1])

    points = []
    for i in range(periods):
        xi = len(values) + i
        v = float(poly(xi))
        margin = float(z * sigma * np.sqrt(1 + 1 / len(values) + (xi - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)))
        year = last_year + i + 1
        points.append({
            "label": str(year),
            "value": v,
            "lower": v - margin,
            "upper": v + margin,
        })

    return {"status": "success", "model": "Régression linéaire", "forecast": points,
            "periods": periods, "confidence_level": confidence}


def forecast_signal(df: pd.DataFrame, periods: int = 5,
                    confidence: float = 0.95) -> dict[str, Any]:
    """
    Lance le meilleur forecaster disponible sur un signal.
    Ordre de priorité : Prophet > Holt-Winters > Régression linéaire.
    """
    df_clean = df.dropna(subset=["value"]).copy()
    if len(df_clean) < 2:
        return {"status": "error", "error": "Données insuffisantes (< 2 points valides)"}

    if PROPHET_AVAILABLE:
        try:
            return _forecast_prophet(df_clean, periods, confidence)
        except Exception as e:
            logger.warning(f"Prophet échoué ({e}), fallback Holt-Winters")

    if STATSMODELS_AVAILABLE:
        try:
            return _forecast_holtwinters(df_clean, periods, confidence)
        except Exception as e:
            logger.warning(f"Holt-Winters échoué ({e}), fallback linéaire")

    return _forecast_linear(df_clean, periods, confidence)


# ══════════════════════════════════════════════════════════════════════════════
#  Classe principale : SignalForecastPipeline
# ══════════════════════════════════════════════════════════════════════════════

class SignalForecastPipeline:
    """
    Pipeline complet de forecasting de signaux.

    Usage :
        pipeline = SignalForecastPipeline()
        pipeline.ingest_directory("dataset/signaux")
        result = pipeline.run(signal_key="...", periods=5)
    """

    def __init__(self):
        # { signal_key: DataFrame(annee, value) }
        self.signals: dict[str, pd.DataFrame] = {}
        self.analyses: dict[str, dict] = {}
        # Compatibilité legacy avec app_integrated.py
        self.data: pd.DataFrame | None = None
        self.loaded_signal_file: str | None = None

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str) -> dict[str, Any]:
        """Charge tous les fichiers .xlsx d'un répertoire."""
        path = Path(directory)
        if not path.exists():
            return {"status": "error", "error": f"Répertoire introuvable : {directory}"}

        xlsx_files = list(path.glob("*.xlsx"))
        if not xlsx_files:
            return {"status": "error", "error": "Aucun fichier .xlsx trouvé"}

        loaded, failed = 0, 0
        for f in xlsx_files:
            try:
                signals = load_xlsx_signals(str(f))
                self.signals.update(signals)
                loaded += len(signals)
                logger.info(f"✓ {f.name} : {len(signals)} signal(s) chargé(s)")
            except Exception as e:
                failed += 1
                logger.error(f"✗ {f.name} : {e}")

        # Analyse automatique
        for key, df in self.signals.items():
            self.analyses[key] = analyze_signal(df)

        return {
            "status": "success",
            "signals_loaded": loaded,
            "files_failed": failed,
            "available_signals": list(self.signals.keys()),
        }

    def load_signal(self, file_path: str) -> dict[str, Any]:
        """
        Charge un seul fichier .xlsx (compatible avec app_integrated.py).
        Remplit self.data avec le premier signal trouvé.
        """
        try:
            signals = load_xlsx_signals(file_path)
            self.signals.update(signals)
            for key, df in signals.items():
                self.analyses[key] = analyze_signal(df)

            if signals:
                first_key = next(iter(signals))
                self.data = signals[first_key].copy()
                # Ajouter une colonne 'date' pour compatibilité
                self.data["date"] = pd.to_datetime(
                    self.data["annee"].astype(str) + "-01-01"
                )

            available_regions = [k.split("__")[-1] for k in signals.keys()]
            return {
                "status": "success",
                "signals_loaded": len(signals),
                "columns_detected": {"available_regions": available_regions},
                "available_signals": list(signals.keys()),
            }
        except Exception as e:
            logger.error(f"Erreur load_signal : {e}")
            return {"status": "error", "error": str(e)}

    # ── Analyse ────────────────────────────────────────────────────────────────

    def analyze_signal(self, signal_key: str | None = None) -> dict[str, Any]:
        """
        Analyse un signal ou (si signal_key=None) self.data pour compatibilité
        avec app_integrated.py.
        """
        if signal_key is not None:
            if signal_key not in self.signals:
                return {"status": "error", "error": f"Signal '{signal_key}' non trouvé"}
            return analyze_signal(self.signals[signal_key])

        # Compatibilité legacy
        if self.data is not None:
            df = self.data[["annee", "value"]].copy() if "value" in self.data.columns \
                else self.data.rename(columns={self.data.columns[-1]: "value"})[["annee", "value"]]
            return analyze_signal(df)

        return {"status": "error", "error": "Aucun signal chargé"}

    # ── Forecasting ────────────────────────────────────────────────────────────

    def forecast(self, signal_key: str | None = None, periods: int = 5,
                 confidence: float = 0.95) -> dict[str, Any]:
        """
        Prédit les prochaines `periods` périodes pour un signal donné.
        Si signal_key=None, utilise self.data (compatibilité app_integrated.py).
        """
        if signal_key is not None:
            if signal_key not in self.signals:
                return {"status": "error", "error": f"Signal '{signal_key}' non trouvé"}
            df = self.signals[signal_key]
        elif self.data is not None:
            # Compatibilité legacy
            if "value" in self.data.columns:
                df = self.data[["annee", "value"]].copy()
            else:
                val_col = [c for c in self.data.columns if c not in ("annee", "date")]
                if not val_col:
                    return {"status": "error", "error": "Aucune colonne de valeur dans self.data"}
                df = self.data[["annee", val_col[0]]].rename(columns={val_col[0]: "value"})
        else:
            return {"status": "error", "error": "Aucun signal chargé"}

        return forecast_signal(df, periods=periods, confidence=confidence)

    def run(self, signal_key: str, periods: int = 5,
            confidence: float = 0.95) -> dict[str, Any]:
        """
        Méthode principale : analyse + prédiction pour un signal identifié.
        Retourne un dict complet pour le rendu frontend.
        """
        if signal_key not in self.signals:
            return {"status": "error", "error": f"Signal '{signal_key}' introuvable"}

        df = self.signals[signal_key]
        analysis = analyze_signal(df)
        forecast_result = forecast_signal(df, periods=periods, confidence=confidence)

        # Données historiques pour affichage
        historical = [
            {"label": str(row["annee"]), "value": float(row["value"])}
            for _, row in df.dropna(subset=["value"]).iterrows()
        ]

        return {
            "status": "success",
            "signal_key": signal_key,
            "analysis": analysis,
            "historical": historical,
            "forecast": forecast_result,
        }

    # ── Sélection automatique ──────────────────────────────────────────────────

    def find_best_signal(self, data_type: str | None = None,
                         region: str | None = None) -> str | None:
        """
        Retourne la clé du signal le plus pertinent selon data_type et region.
        Utilisé par la logique de sélection de app_integrated.py.
        """
        if not self.signals:
            return None

        candidates = list(self.signals.keys())

        # Filtrage par région
        if region:
            region_lower = region.lower()
            filtered = [k for k in candidates if region_lower in k.lower()]
            if filtered:
                candidates = filtered

        # Filtrage par type de données
        if data_type:
            type_map = {
                "ventes": ["vente", "client"],
                "production": ["production"],
                "abonnes": ["abonn"],
                "qualite": ["qualit"],
            }
            keywords = type_map.get(data_type, [data_type])
            filtered = [k for k in candidates
                        if any(kw in k.lower() for kw in keywords)]
            if filtered:
                candidates = filtered

        return candidates[0] if candidates else None

    @property
    def available_signals(self) -> list[str]:
        return list(self.signals.keys())


# ── Alias pour compatibilité ────────────────────────────────────────────────────
ForecastPipeline = SignalForecastPipeline