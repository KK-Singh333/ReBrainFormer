from nilearn import datasets

abide = datasets.fetch_abide_pcp(
    data_dir="test_data",
    n_subjects=None,                 # all available subjects
    pipeline="cpac",
    band_pass_filtering=False,       # nofilt
    global_signal_regression=False,  # noglobal
    derivatives=["func_preproc"],    # 4D .nii.gz scans
    quality_checked=False,           # include subjects regardless of QC rating
)

print("Downloaded scans:", len(abide.func_preproc))
print("First scan:", abide.func_preproc[0])