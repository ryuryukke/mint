# MINT

<p align="center">
<img src="assets/mint_logo.png" alt="MINT" width="200">
</p>
<p align="center">
  <a href="https://github.com/liamdugan/raid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://ryuryukke.github.io/"><img src="https://img.shields.io/badge/NLP-NLP?label=Institute of Science Tokyo"/></a>
  <a href="https://liamdugan.com/"><img src="https://img.shields.io/badge/NLP-NLP?label=University%20of%20Pennsylvania"/></a>
  <a href="https://arxiv.org/abs/xxxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg"/></a>
</p>
<p align="center">
<b>A unified python package for membership inference attacks (MIAs) and machine-generated text detection.
</p>

## Quick Start




## Methods
We include 4 common baseline methods, 7 state-of-the-art MIAs, and 5 state-of-the-art machine text detectors:
|Method|Task|Description|
|---|-----|---|
| [Loss]() | Baseline | The likelihood of a text against the model.|
| [Rank]() | Baseline | The average rank of the next token in the model's probability distribution at each time step.|
| [LogRank]() | Baseline | The average log rank of the next token in the model's probability distribution at each time step.|
| [Entropy]() | Baseline | The expected likelihood of the next token given the preceding tokens at each time step under the model's distribution.|
| [Reference]() | MIA |the difference in the target sample $x$'s loss between the model $\mathcal{M}$ and another reference model $\mathcal{M}_{ref}$. |
| [Zlib]() | MIA |
| [Neighborhood]() | MIA |
|[Min-K%]()|MIA|
|[Min-K%++]()|MIA|
|[ReCaLL]()|MIA|
|[DC-PDD]()|MIA|
|[DetectGPT]()|Detection|
|[Fast-DetectGPT]()|Detection|
|[Binoculars]()|Detection|
|[DetectLLM]()|Detection| 
|[Lastde++]()|Detection| A quantity known as multi-scale diversity entropy (MDE) to measure the local fluctuations in likelihood across a particular text sequence.|

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
