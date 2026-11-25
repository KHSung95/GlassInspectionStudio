"""Read-only camera discovery through a GenTL producer (default: installed Spinnaker GenTL).

    python tools/camera_probe.py [--cti PATH] [--open]

Lists devices. With --open, opens the first device read-only-ish and prints key GenICam feature values
(no feature is written, no acquisition is started).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistudio.acquisition.genicam_camera import DEFAULT_CTI, list_devices, read_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cti", default=DEFAULT_CTI)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    devs = list_devices(a.cti)
    print(json.dumps(devs, indent=1, default=str))
    if a.open and devs:
        print(json.dumps(read_features(a.cti, 0), indent=1, default=str))


if __name__ == "__main__":
    main()
