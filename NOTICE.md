# Notice

This repository is the official implementation of
"Leveraging Vision-Language Models as Weak Annotators in Active Learning".

The codebase is built upon the ISOAL implementation:
- Repository: https://github.com/matsuo-shinnosuke/ISOAL
- Paper: Instance-wise Supervision-level Optimization in Active Learning, CVPR 2025
- Authors: Shinnosuke Matsuo, Riku Togashi, Ryoma Bise, Seiichi Uchida, Masahiro Nomura

The original ISOAL implementation provides the active-learning framework used
as the base of this work. This repository adds VLM/Gemini weak-label generation,
CUB200 and FGVC-Aircraft experiment support, and optional transition-matrix
forward correction.

If you use this repository, please cite the original ISOAL paper:

```bibtex
@inproceedings{matsuo2025isoal,
  title = {Instance-wise Supervision-level Optimization in Active Learning},
  author = {Shinnosuke Matsuo and Riku Togashi and Ryoma Bise and Seiichi Uchida and Masahiro Nomura},
  booktitle = {Computer Vision and Pattern Recognition},
  year = {2025},
}
```

No Gemini API keys, generated labels, datasets, or model checkpoints should be
committed to this repository.
