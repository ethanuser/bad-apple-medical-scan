# Technical notes

## Annotations

The viewport annotations are derived from each scan, not decorative:
- Orientation letters push the crop's axes through each scan's affine, so oblique crops get two letters (for example `SA`).
- Positions are the slice centre in the scan's own coordinates.
- CT window and level convert back to Hounsfield units through the preparation window.
- MRI window and level are in the scaled 0–255 display units, because MRI has no absolute unit.
- `W` is the width of the steep part of the gray curve; a broad secondary ramp keeps texture outside it.
- `Sl` numbers slices in the reoriented volume, not the scanner's original image numbers.

## Running locally

Use native arm64 Python 3.11–3.13, FFmpeg with FFprobe, and curl. The proof was run with Python 3.13 and PyTorch 2.15 on an Apple-silicon laptop. Retrieval runs in PyTorch on the Apple GPU (MPS), sampling crops with `grid_sample` and scoring every target frame against each batch by matrix multiplication. On that laptop, the default 7-second run searches 126 million crops from 250 volumes in about 14 minutes, after about 3 minutes of loading. The first load prepares each CT and caches it beside the scan (2.2 GB). Refinement and scoring stay on the CPU and take about 15 seconds.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m mriapple fetch
.venv/bin/python -m mriapple run
```

**Devices.** `--device auto` picks CUDA, then MPS, then the PyTorch CPU backend. `--device numpy` runs the original NumPy and Accelerate search, which is limited to 16 × 12 thumbnails. With the same seed and 16-pixel thumbnails, the two searches draw identical proposals and agree on the top retrieval for every frame. A laptop GPU is limited by memory bandwidth, so a CUDA GPU such as an H100 is the practical way to raise `--coarse-width` to 48 or 64, halve `--slice-step` or match the full song. The same command runs there after `fetch`.

The public inputs occupy about 4.6 GB on disk, plus 2.2 GB of prepared CT caches. The 50 CTs are fetched one at a time from Zenodo by byte range, which took about 45 minutes. Fetching checks each SHA-256 digest against the pinned manifest. The run is local and needs no account. The default output is a 1440 × 1080 H.264 video at 10 fps with the corresponding audio excerpt. The source files and local outputs are git-ignored. The resolution and 4:3 composition follow the lunar reference. [Reference renderer](https://github.com/ikuto32/bad-apple-earth-moon)

To change the excerpt, set the source start time in seconds, duration and output frame rate. Use a separate output directory for another experiment.

```sh
.venv/bin/python -m mriapple run --start 12 --seconds 5 --fps 10 --output output/alternate
```

`match` runs the search and writes results without encoding. `render` rebuilds the video from the saved transforms, verifies the source hashes and checks the source frame mapping. Both commands accept `--output`. The `run` and `match` commands also accept `--samples`, `--slice-step` and `--seed`. Doubling the slice step roughly halves the number of sections proposed for retrieval.

## Full song on a GPU cluster

Matching every frame scores each crop against all 2,320 frames at 10 fps, about 33 times the work per crop of the seven-second proof. On the laptop the default search processes about 22,000 crops per second at that size, so with 250 volumes the full song takes about 1 hour 50 minutes, plus about 6 minutes of refinement; rendering takes under 2 minutes. 30 fps, finer thumbnails or a denser bank multiply that, so those runs belong on a CUDA GPU, with the video still rendered locally. The steps below work on any Slurm cluster. Use a directory with about 15 GB free, which may mean scratch or project space rather than home.

```sh
scripts/gpu-sync.sh push-ct <user>@<cluster-host> bad-apple-mri
```

`push-ct` sends the code and the 2.4 GB of CT, which Zenodo serves slowly; the brain MRI then downloads quickly from Hugging Face on the cluster. The sync reuses one SSH connection for ten minutes, so a two-factor prompt appears once. `push` sends code only and `push-data` sends all inputs.

On the login node, from that directory:

```sh
scripts/cluster-info.sh      # accounts, GPU partitions, storage, Python versions
scripts/setup-env.sh         # .venv, requirements and verified inputs; needs internet
sbatch --account=<account> --partition=<gpu partition> scripts/cluster-bench.sbatch
sbatch --account=<account> --partition=<gpu partition> --time=<from bench> scripts/cluster-job.sbatch 30
squeue -u $USER              # then: tail -f output/full-song/run.log
```

The benchmark times the search at 32, 48 and 64-pixel thumbnails against all 6,960 frames at 30 fps on every tenth volume, times refinement on 200 frames with every CPU, and prints the estimated full-run time for several slice steps and sample counts. Use it to pick the density and the job's time limit.

Back on the local machine:

```sh
scripts/gpu-sync.sh pull <user>@<cluster-host> bad-apple-mri
.venv/bin/python -m mriapple render --output output/full-song
```

Add `--layout vertical --to output/vertical` for a 1080x1920 version for phone feeds: the same panels with the viewport at the centre, the source frame, localizer and 3D view above it, and the cohort strip below.

Pass a frame rate and extra `match` options after the job script, such as `scripts/cluster-job.sbatch 30` or `scripts/cluster-job.sbatch 10 --coarse-width 48 --slice-step 2`. On a machine without a scheduler, `scripts/gpu-run.sh` takes the same arguments and runs setup and matching together. Matching needs no FFmpeg: without `ffprobe`, OpenCV reads the same 30000/1001 rate and 6,955 frames. Batches are sized from free GPU memory. Refinement splits the frames into one contiguous block per CPU and runs the blocks in forked processes on Linux (threads on a Mac whose GPU was used), so 6,960 frames take minutes rather than the 20 a single core would need. The retrieval is saved to `output/full-song/retrieval.json.gz`, so resubmitting the same job after a timeout repeats only the CPU refinement. The render checks the source hashes and frame mapping against the copied `matches.json`.

## Matching

1. Read 200 native-space volumes, comprising T1, T2, proton-density and MR angiography scans for the same 50 public IXI subject IDs, every IXI subject in the distribution. Use the paired, manually refined brain masks to set extracranial voxels to zero. Reindex each volume to the closest RAS orientation by axis permutations and flips. Native acquisitions can be slightly oblique. The three plane names identify their closest anatomical axes.
   Also read 50 whole-body CT scans from the TotalSegmentator dataset v2.0.1, chosen by `scripts/select_ct.py` as the longest scans whose labels span the clavicles to the femurs (polytrauma, whole-body and neck-to-leg angiography studies, resampled by the dataset to 1.5 mm). Each CT is reoriented to closest RAS, then everything above the clavicles plus 25 mm is removed, cutting at least 15 mm below the lowest labelled skull voxel when that is lower, so no head or face remains. A body mask keeps tissue above −500 HU after removing the scanner table, with lungs and bowel gas filled in each axial slice. Only the largest connected piece, the body, is kept. That drops detached equipment that survives the table filter: in 10 of the 50 scans a curved head holder under the shoulders, plus a curved shell and a line in two others. No body part is detached in this cohort; the largest other piece was 1.6% of the body. −1000 to +1000 HU maps linearly to 0–255 inside the mask. CT crops may be 40–700 mm wide so whole torsos can form a silhouette.
2. Enumerate axial, coronal and sagittal sections every four voxels within the brain extent. For each section, propose 6,400 centres, widths and angles using a seeded generator. The centre distribution samples brain tissue and its vicinity. Reject any crop whose footprint leaves the source image. Width is uniform in log space from 32 to 166 mm during retrieval and can refine to 30–180 mm. Pixel spacing sets the aspect ratio in physical millimetres.
3. **Grayscale NCC baseline.** Score 32 × 24 thumbnails by normalized cross-correlation against the target frames, including mirrored views and both intensity polarities. Pick the strongest coarse pose and fit its contrast. This is the baseline drawn in gray on the graph.
4. **Silhouette refinement.** Search the same bank with two extra intensity thresholds, combining coarse correlation, directional gradient agreement and correlation in a 4 × 3 grid. Local contrast makes an apple or a gap between the hand and face count even when the torso already matches. Refine promising poses at 48 × 36 through three progressively smaller perturbations of position, slice, zoom and rotation. Fit an intensity threshold from class histograms and select among three contrast widths. A broad secondary window preserves some MRI texture.
5. Propagate candidate poses between adjacent frames using silhouette centroids and areas. A dynamic program chooses the sequence with the highest sum of frame scores less a small cost for changing source or pose. The score on screen is the individual frame’s image score. This is a sampled search and the selected sequence balances similarity and continuity.

**Display.** Each slice is smoothed with a 0.6-voxel Gaussian, which rounds voxel stair-steps without moving edges. The crop is sampled bilinearly at twice the panel resolution, thresholded at one level for the whole crop, and area-reduced to 1040 × 780, giving a black and white silhouette with one-pixel anti-aliased edges. Mirroring and inversion choose which side is white. The localizer shows the unthresholded slice.

**Similarity.** Measure the full 1040 × 780 black and white panel after area reduction to 96 × 72, against the downsampled source frame. The score is 28% mean black/white intersection-over-union, 27% boundary F1 with a two-pixel tolerance, 20% thin detail, 12.5% positive grayscale NCC and 12.5% positive regional NCC. Thin detail is the share of target features under five pixels wide, such as hair strands, fingers and narrow gaps, that the match reproduces in the right class. Regional NCC averages correlations over 12 regions, weighting regions with more target contrast. Empty classes and flat frames have defined fallback scores. The same metrics and threshold fitting apply to both methods. The threshold starts at the level that maximizes balanced IoU, and the four neighbouring levels 4 and 8 units away are tried on the full score.

**Selection.** The search maximizes the similarity plus a small preference for wider crops in native voxels, up to 0.025 at 128 voxels across, because the panel magnifies a 45-voxel crop about 23 times. Thresholds below 20 trace the brain-extraction mask's voxel staircase rather than tissue contrast and cost 0.025. Grazing slices and off-centre crops match Bad Apple's simple blobs well but stop reading as anatomy (in the first full-song run, half the frames came from the outer 30% of a scan), so two more costs apply. A slice cost of up to 0.035 applies in the outer half of the scanned extent. A crop cost of up to 0.03 applies once the crop centre is more than 0.7 tissue radii from the slice's tissue centroid. Those two alone pushed the search toward heavily magnified crops that blur into smudges, so a blur cost of up to 0.05 applies below 64 native voxels across (full at 32). On three five-second stretches of the full song, together they cut grazing slices roughly in half and crops under 48 voxels from 18% to 1%, for 0.005 less similarity. None of these adjustments enters the displayed score. Refinement compares candidates at 48 × 36.

## Reproducing a frame

`sources.json` records immutable download URLs and hashes for all images, masks and the source animation. `matches.json` includes the input manifest, original and RAS affines, dimensions, voxel spacings, normalization values, software versions, search configuration and per-frame results for both methods.

The output frame numbered 001 corresponds to zero-based record 0. At requested time *t*, the source index is floor(*t* × 30000/1001). Two columns are removed from each side of the 1444 × 1080 encode to give a centred 4:3 frame. Audio starts at the requested source time.

In each pose, `scan` indexes the manifest’s scan array. `axis` is 0, 1 or 2 for sagittal, coronal or axial, and `index` is a zero-based index in the closest-RAS array. Within that plane, horizontal coordinates increase on the first remaining RAS axis and vertical coordinates decrease on the second. The first displayed voxel centre is the origin for `x_mm` and `y_mm`. Positive angles turn clockwise in that displayed plane. The sampling transform maps output coordinates to source coordinates, applying a horizontal mirror before rotation. `corners_slice_xy_vox` lists the four source footprint corners in output order, starting at the output’s top left.

MRI voxel values inside the mask are scaled to 0–255 using zero and the recorded 99.7th percentile of in-mask values; CT values use the fixed HU window. The recorded threshold operates in those units, on the slice smoothed by 0.6 voxels. Every geometry and threshold parameter is shared by the entire crop. Published brain-extraction masks and the CT preparation are applied identically for every target, and the prepared CT is cached beside each scan. The application uses public study IDs and whitelisted imaging metadata. CT selection reads only the study type column of the dataset's meta.csv; no demographic fields are used or exported, and no free-text headers are exported.

## Verification

After rendering, run `.venv/bin/python -m unittest discover -s tests -v`. The geometry tests cover rotation in millimetres, anisotropic spacing, mirroring, pixel-centre alignment and contrast polarity. The annotation test checks every orientation letter against the crop mapping pushed through the scan affine, the scale against the crop geometry, and the Hounsfield conversion. The integration test decodes every movie frame, reconstructs sample intensities independently from the logged corners and the native masked NIfTI voxels, and compares them within a lossy-codec tolerance, away from the overlay margins.

## Source terms

The IXI project lists its original release under CC BY-SA 3.0. This project uses the SynthStrip v1.5 distribution, which supplies the native IXI images under CC BY-SA 4.0 and manually refined masks. The CT scans come from the TotalSegmentator dataset v2.0.1, released under CC BY 4.0, and are fetched as individual members of the pinned Zenodo archive by byte range. The adapted imagery carries CC BY-SA 4.0. The animation and music retain separate rights.

- [IXI project](https://brain-development.org/ixi-dataset/)
- [SynthStrip distribution documentation](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/README)
- [TotalSegmentator dataset v2.0.1](https://zenodo.org/records/10047292)
- [Full attribution](../CREDITS.md)
