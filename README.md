# MINT

<p align="center">
<img src="assets/mint.png" alt="MINT" width="200">
</p>
<p align="center">
  <a href="https://github.com/liamdugan/raid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://ryuryukke.github.io/"><img src="https://img.shields.io/badge/NLP-NLP?label=Institute%20of%20Science%20Tokyo"/></a>
  <a href="https://liamdugan.com/"><img src="https://img.shields.io/badge/NLP-NLP?label=University%20of%20Pennsylvania"/></a>
  <a href="https://arxiv.org/abs/xxxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg"/></a>
</p>
<p align="center">
<i><b>A unified python package for membership inference attacks (MIAs) and machine-generated text detection.</b></i>
</p>

## Quick Start




## Methods
We include **4 common baselines**, **7 state-of-the-art MIAs**, and **5 state-of-the-art machine text detectors**:

| Category | Methods |
|-----------|----------|
| Baselines | Loss, Rank, LogRank, Entropy |
| MIAs | [Reference](https://arxiv.org/abs/2012.07805), [Zlib](https://arxiv.org/abs/2012.07805), [Neighborhood](https://arxiv.org/abs/2305.18462), [Min-K%](https://arxiv.org/abs/2310.16789), [Min-K%++](https://arxiv.org/abs/2404.02936), [ReCaLL](https://arxiv.org/abs/2406.15968), [DC-PDD](https://arxiv.org/abs/2409.14781) |
| Detectors | [DetectGPT](https://arxiv.org/abs/2301.11305), [Fast-DetectGPT](https://arxiv.org/abs/2310.05130), [Binoculars](https://arxiv.org/abs/2401.12070), [DetectLLM](https://arxiv.org/abs/2306.05540), [Lastde++](https://arxiv.org/abs/2410.06072) |



## Running on your own dataset

## Running your own attack or detector


## Citation
If you find our code/data/models or ideas useful in your research, please cite our work:
```
@misc{koike2025machinetextdetectors,
      title={Machine Text Detectors are Membership Inference Attacks}, 
      author={Ryuto Koike and Liam Dugan and Masahiro Kaneko and Chris Callison-Burch, Naoaki Okazaki},
      year={2025},
      eprint={xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
```



## Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. These research results were also obtained from the commissioned research (No.22501) by National Institute of Information and Communications Technology (NICT), Japan. In addition, this work was supported by JST SPRING, Japan Grant Number JPMJSP2106.
