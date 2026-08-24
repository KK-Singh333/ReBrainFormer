from nilearn import datasets

# Fetch 2 subjects from the ABIDE dataset
# 'func_preproc' downloads the 4D preprocessed functional MRI NIfTI data
abide_data = datasets.fetch_abide_pcp(
    data_dir="test_data",
    n_subjects=2,
    derivatives=['func_preproc']
)

print("Downloaded 4D fMRI files:")
for file_path in abide_data.func_preproc:
    print(file_path)