"""Run independent data studies and their business decision diagnostics."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-only', action='store_true')
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Reuse existing study outputs and retail database; run business diagnostics only')
    parser.add_argument('--hive', action='store_true',
                        help='Also verify actual Hive; requires an already running Docker daemon')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.diagnostics_only:
        subprocess.run([sys.executable, '-m', 'campaign.run'], cwd=root, check=True)
        if not args.campaign_only:
            subprocess.run([sys.executable, '-m', 'retail'], cwd=root, check=True)
    subprocess.run([sys.executable, '-m', 'campaign.decision'], cwd=root, check=True)
    if not args.campaign_only:
        subprocess.run([sys.executable, '-m', 'retail.diagnose'], cwd=root, check=True)
        subprocess.run([sys.executable, '-m', 'retail.customer_drilldown'], cwd=root, check=True)
        subprocess.run([sys.executable, '-m', 'retail.geography_drilldown'], cwd=root, check=True)
        subprocess.run([sys.executable, '-m', 'retail.sku_drilldown'], cwd=root, check=True)
    if args.hive:
        command = [sys.executable, 'infra/run_hive.py']
        sample = root / 'data/processed/retail_hive_sample.csv'
        if not args.campaign_only and sample.exists():
            command += ['--retail-input', str(sample)]
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
