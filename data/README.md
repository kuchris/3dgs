# Local datasets

Downloaded datasets live here but are not committed to Git because the image
archives are large. To download the public COLMAP South Building demo, run:

```powershell
.\scripts\download-demo-data.ps1
```

This creates `data\demo\south-building\images` with 128 photographs. The
source is COLMAP's official
[South Building sample dataset](https://colmap.github.io/datasets.html). The
download script keeps only the raw photographs so our pipeline must reconstruct
the cameras and sparse points itself.
