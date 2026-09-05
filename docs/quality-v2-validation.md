# Quality Trainer v2 verification

Verified locally on 2026-09-05 with an RTX 5070 Ti, gsplat 1.5.3, and the
locked uv environment. The dataset is the prepared COLMAP South Building demo.

## Implementation checks

- 32 pytest tests passed, including a CUDA resume test that checks model geometry,
  Adam state, and learning-rate scheduler continuity within numerical tolerance.
- Resume rejects changed image contents or a different total training schedule.
- SH degree-3 PLY export preserves all 16 coefficients per RGB channel.
- Ruff passed for changed implementation modules and tests; Python compilation,
  `uv lock --check`, and `git diff --check` passed.

## Real dataset integration checks

A 201-step run at one-quarter prepared resolution exercised progressive SH up
to degree 3, splitting/pruning, checkpoint saving, and all 16 evaluation views.
It grew from 84,004 to 95,124 Gaussians. Its 21.41 MiB PLY loaded successfully
in the viewer data loader with finite covariances and SH degree 3.

A separate run paused at step 100 and resumed through step 201. It reached
95,123 Gaussians, with evaluation mean PSNR 18.3611 dB, SSIM 0.5587, and L1
0.08679. The uninterrupted run scored 18.2392 dB, 0.5573, and 0.08909.
GPU accumulation and splitting are not bitwise deterministic.

## Full-resolution pilot

The pilot ran the first 1,000 steps of a 30,000-step schedule at 1600x1196.
It used 112 training photos and 16 evaluation photos, with each training photo
appearing once per shuffled epoch. It ended with 102,628 Gaussians and a
resumable checkpoint. SH is still degree 0 at this boundary; degree 1 starts
on the next update, followed by degrees 2 and 3 later in the schedule.

Mean evaluation scores: PSNR 19.1506 dB, SSIM 0.5037, L1 0.07455.
The first evaluation view's L1 improved from 0.23604 (rounded) to 0.0658.
Peak allocated CUDA memory was approximately 0.66 GiB.

The generated comparisons were visually inspected. Building structure is
recognizable, while fine brick and foliage details remain soft. These are
pilot results, not completed 30,000-step quality results.

Local outputs are under `outputs/demo/3dgs/quality-v2-pilot`, including
`checkpoint.pt`, `config.json`, `metrics.json`, and `evaluation-*.png`.
The long continuation uses `outputs/demo/3dgs/quality-v2`.

Evaluation images are excluded from photometric training, but COLMAP camera
estimation and sparse initialization use all images. The previous v1 score
used a favoured training view and a different image resolution, so it is not
directly comparable to these evaluation means.
