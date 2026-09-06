"""Downloads or generates the official IEEE 123-node test feeder OpenDSS files."""

from __future__ import annotations
import urllib.request
from pathlib import Path
import sys

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsse.config import AppConfig
from dsse.reproducibility import setup_logger

logger = setup_logger("download_feeder")

GITHUB_BASE_URL = "https://raw.githubusercontent.com/tshort/OpenDSS/master/Distrib/IEEETestCases/123Bus/"
FEEDER_FILES = {
    "IEEE123Master.dss": "IEEE123Master.dss",
    "IEEELineCodes.DSS": "IEEELineCodes.DSS",
    "IEEE123Regulators.DSS": "IEEE123Regulators.DSS",
    "IEEE123Loads.DSS": "IEEE123Loads.DSS",
    "Run_IEEE123Bus.DSS": "Run_IEEE123Bus.DSS",
}


def create_solar_pv_dss(target_dir: Path) -> None:
    """Creates optional Solar PV / DER DSS definition file for OOD renewable generation scenarios."""
    solar_file = target_dir / "SolarPV.dss"
    content = """! Solar PV / DER additions for OOD generation scenarios on IEEE 123-node feeder
! Added at distributed representative single-phase and 3-phase buses
New Generator.PV_bus1   Phases=3 Bus1=1.1.2.3   kV=4.16 kW=100.0 kvar=0.0 model=1
New Generator.PV_bus18  Phases=3 Bus1=18.1.2.3  kV=4.16 kW=120.0 kvar=0.0 model=1
New Generator.PV_bus47  Phases=3 Bus1=47.1.2.3  kV=4.16 kW=150.0 kvar=0.0 model=1
New Generator.PV_bus76  Phases=3 Bus1=76.1.2.3  kV=4.16 kW=120.0 kvar=0.0 model=1
New Generator.PV_bus97  Phases=3 Bus1=97.1.2.3  kV=4.16 kW=150.0 kvar=0.0 model=1
New Generator.PV_bus108 Phases=3 Bus1=108.1.2.3 kV=4.16 kW=100.0 kvar=0.0 model=1
New Generator.PV_bus65  Phases=3 Bus1=65.1.2.3  kV=4.16 kW=100.0 kvar=0.0 model=1
"""
    solar_file.write_text(content, encoding="utf-8")
    logger.info(f"Created Solar PV DSS file at {solar_file}")


def download_ieee123(target_dir: Path) -> None:
    """Downloads the IEEE 123-node feeder files to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory for IEEE 123 feeder: {target_dir}")

    for filename, source_name in FEEDER_FILES.items():
        dest = target_dir / filename
        url = GITHUB_BASE_URL + source_name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8")
                dest.write_text(content, encoding="utf-8")
                logger.info(f"Successfully downloaded {filename} ({len(content)} bytes)")
        except Exception as e:
            logger.warning(f"Could not download {filename} from {url}: {e}")

    create_solar_pv_dss(target_dir)
    logger.info("IEEE 123 feeder setup completed.")


if __name__ == "__main__":
    cfg = AppConfig.load_all()
    download_ieee123(cfg.paths.feeder_dir)
