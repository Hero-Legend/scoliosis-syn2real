# Data sources and metadata dictionary

## Access routes

- **AASCE / SpineWeb Dataset 16:** reused third-party data. Source catalogue: http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets ; challenge: https://aasce19.grand-challenge.org/ . A traceable mirror is https://github.com/Arnav0400/Spine-Corner-Net/tree/master/Data/boostnet_labeldata . These are acquisition routes, not a licence granted by this repository.
- **Spinal-AI2024:** reused synthetic data released by its authors at https://github.com/Ernestchenchen/Spinal-AI2024 . Refer to the source repository and its CurvNet publication for provenance and current terms.

No original radiographs or angle-label files are redistributed here. We have not established a separate grant to rehost the complete collections. This is not a statement that research use is prohibited. Users should obtain inputs from the source and comply with applicable source conditions and citations.

For AASCE/SpineWeb, cite Wu, H.; Bailey, C.; Rasoulinejad, P.; Li, S. Automatic Landmark Estimation for Adolescent Idiopathic Scoliosis Assessment Using BoostNet. MICCAI, 2017, pp. 127-135. Cite the Spinal-AI2024/CurvNet source as specified by its authors.

## Files

| File | Contents |
| --- | --- |
| `aasce_identity.csv` | 480 image identities, original training-source row index, image hashes, containment groups and fixed split membership; no filenames, dates, angle values or class labels |
| `spinalai_identity.csv` | 15,999 synthetic training identities with numeric source filenames, subset and image SHA-256; no angle values or class labels |
| `label_budgets.csv` | Five compound seeds, four nested nominal training-group budgets and per-image label visibility; no class labels |

`source_row` is zero-based in the original 481-row AASCE training label files. The 480-image cohort preserves an inherited landmark-QC exclusion; it is not the full 481-image source and does not use the official 128-image test partition. The manifest preserves the exact eligible rows rather than silently replacing the cohort.

`dataset_split` is `TRAINDEV`, `VALIDATION`, or `LOCKED_BENCHMARK`, denoting 288, 72, and 120 images. The last is an internal held-out partition, not the official challenge test set. Historical exposure status is not certified by this release.

`containment_group` identifies an image-similarity containment group. Groups stay intact across splits and budget selection; they are not verified patient IDs. There are 229 groups in the complete cohort.

`label_budget_seed` is a compound subset/training seed. `budget_percent` is a nominal fraction of training groups, not a fraction of all annotation work. `label_visible` is 0 or 1. All images belonging to a group are selected together.

`image_sha256` is the SHA-256 of the original image file bytes, used for identity checks. The synthetic manifest contains the 15,999 usable labeled training images, rather than all 20,000 images in the upstream release.

The generated `inputs/training.json` contains local paths and reconstructed training/validation/source labels; `inputs/heldout.json` contains local held-out labels. They belong only in your private workspace and are not part of the public release.
