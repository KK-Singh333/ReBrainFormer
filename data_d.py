import argparse
from pathlib import Path

from nilearn import datasets


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Download ABIDE PCP fMRI scans and phenotypes.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data",
        help="Directory for ABIDE downloads (default: project/test_data)",
    )
    args = parser.parse_args()

    abide = datasets.fetch_abide_pcp(
        data_dir=str(args.data_dir),
        n_subjects=None,                 # all available subjects
        pipeline="cpac",
        band_pass_filtering=False,       # nofilt
        global_signal_regression=False,  # noglobal
        derivatives=["func_preproc"],    # 4D .nii.gz scans
        quality_checked=False,           # include subjects regardless of QC rating
    )

    print("Downloaded scans:", len(abide.func_preproc))
    if len(abide.func_preproc):
        print("First scan:", abide.func_preproc[0])


if __name__ == "__main__":
    main()
