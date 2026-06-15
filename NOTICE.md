# Notice

This repository provides the implementation for
"Leveraging Vision-Language Models as Weak Annotators in Active Learning".

The codebase is built primarily upon the original ISOAL implementation:
- Repository: https://github.com/matsuo-shinnosuke/ISOAL
- Paper: Instance-wise Supervision-level Optimization in Active Learning, CVPR 2025
- Authors: Shinnosuke Matsuo, Riku Togashi, Ryoma Bise, Seiichi Uchida, Masahiro Nomura

Most of the active-learning framework, training pipeline, and core ISO strategy
are inherited from the original ISOAL codebase. This repository extends the
ISOAL implementation with Gemini-based VLM weak-label generation, support for
CUB200 and FGVC-Aircraft experiments, dataset-specific initial full-label seeds,
additional backbone options, and optional transition-matrix forward correction.

If you use this repository, please cite the original ISOAL paper:

@inproceedings{matsuo2025isoal,
  title = {Instance-wise Supervision-level Optimization in Active Learning},
  author = {Shinnosuke Matsuo and Riku Togashi and Ryoma Bise and Seiichi Uchida and Masahiro Nomura},
  booktitle = {Computer Vision and Pattern Recognition},
  year = {2025},
}

No Gemini API keys, generated labels, datasets, result logs, NumPy arrays, or
model checkpoints should be committed to this repository.
