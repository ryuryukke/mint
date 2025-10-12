<p align="center">
<img src="assets/mint_bar.png" alt="MINT" width="600">
</p>
<p align="center">
  <a href="https://github.com/liamdugan/raid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://ryuryukke.github.io/"><img src="https://img.shields.io/badge/NLP-NLP?label=Institute%20of%20Science%20Tokyo"/></a>
  <a href="https://liamdugan.com/"><img src="https://img.shields.io/badge/NLP-NLP?label=University%20of%20Pennsylvania"/></a>
  <a href="https://arxiv.org/abs/xxxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg"/></a>
</p>
<!-- <p align="center">
<i><b><span style="font-size:24px;">A unified evaluation suite for membership inference attacks (MIAs) and machine-generated text detection.</span></b></i>
</p> -->

<h3 align="center"><i><b>
A unified evaluation suite for membership inference attacks (MIAs) and machine-generated text detection.
</b></i></h3>

## Quick Start
Install the Python dependencies:
```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```
Run methods on MIA or Detection:
```
$ python 
```

## MINT Supports
We include **4 common baselines**, **7 state-of-the-art MIAs**, and **5 state-of-the-art machine text detectors**:

| Methods | Category | Description
|-----------|----------|----------|
| Loss | Baselines | the likelihood of a target sample
| Entropy | Baselines | the expected likelihood of a target sample
| Rank | Baselines | the average rank of the predicted token at each step
| LogRank | Baselines | the average log-rank of the predicted token at each step
| [Reference](https://arxiv.org/abs/2012.07805) | MIA | the difference in the target loss between the model and another reference model
| [Zlib](https://arxiv.org/abs/2012.07805) | MIA | the ratio of the target loss and the zlib compression score of the target
| [Neighborhood](https://arxiv.org/abs/2305.18462) | MIA | the difference between the target loss and the average loss over its perturbed samples
| [Min-K%](https://arxiv.org/abs/2310.16789) | MIA | the average of log-likelihood of the $k$\% tokens with lowest probabilities
| [Min-K%++](https://arxiv.org/abs/2404.02936) | MIA | a standardized version of Min-K% over the model's vocabulary
| [ReCaLL](https://arxiv.org/abs/2406.15968) | MIA | the relative log-likelihood between a target sample and a set of non-member examples
| [DC-PDD](https://arxiv.org/abs/2409.14781) | MIA | the cross-entropy between the token likelihoods under the model and the laplace-smoothed unigram token frequency distribution under some reference corpus
| [DetectGPT](https://arxiv.org/abs/2301.11305) | Detection | the difference between the target loss and the average loss over its perturbed samples
| [Fast-DetectGPT](https://arxiv.org/abs/2310.05130) | Detection | an efficient version of DetectGPT via fast-sampling technique and score normalization
| [Binoculars](https://arxiv.org/abs/2401.12070) | Detection | the ratio of the target perplexity to the cross entropy of the target sample under some reference model
| [DetectLLM](https://arxiv.org/abs/2306.05540) | Detection | a variant of DetectGPT instead of using LogRank as the core quantity
| [Lastde++](https://arxiv.org/abs/2410.06072) | Detection | the multi-scale diversity entropy measuring the local fluctuations in likelihood across a target text sequence


<!-- | Category | Methods |
|-----------|----------|
| Baselines | Loss, Rank, LogRank, Entropy |
| MIAs | [Reference](https://arxiv.org/abs/2012.07805), [Zlib](https://arxiv.org/abs/2012.07805), [Neighborhood](https://arxiv.org/abs/2305.18462), [Min-K%](https://arxiv.org/abs/2310.16789), [Min-K%++](https://arxiv.org/abs/2404.02936), [ReCaLL](https://arxiv.org/abs/2406.15968), [DC-PDD](https://arxiv.org/abs/2409.14781) |
| Detectors | [DetectGPT](https://arxiv.org/abs/2301.11305), [Fast-DetectGPT](https://arxiv.org/abs/2310.05130), [Binoculars](https://arxiv.org/abs/2401.12070), [DetectLLM](https://arxiv.org/abs/2306.05540), [Lastde++](https://arxiv.org/abs/2410.06072) | -->

## Datasets
We use the [MIMIR](https://github.com/iamgroot42/mimir) benchmark for MIAs and the [RAID](https://github.com/liamdugan/raid) benchmark for detection. 


## Running on your own dataset

## Running your own attack or detector


## Citation
If you find our code or ideas useful in your research, please cite our work:
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
