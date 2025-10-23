<p align="center">
<img src="assets/mint_bar.png" alt="MINT" width="600">
</p>
<p align="center">
  <a href="https://github.com/liamdugan/raid/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"/></a>
  <a href="https://ryuryukke.github.io/"><img src="https://img.shields.io/badge/NLP-NLP?label=Institute%20of%20Science%20Tokyo"/></a>
  <a href="https://liamdugan.com/"><img src="https://img.shields.io/badge/NLP-NLP?label=University%20of%20Pennsylvania"/></a>
  <a href="https://arxiv.org/abs/2510.19492"><img src="https://img.shields.io/badge/arXiv-2510.19492-b31b1b.svg"/></a>
</p>
<h3 align="center"><i><b>
"A unified evaluation suite for membership inference attacks<br>and machine-generated text detection."
</b></i></h3>

## Quick Start
Build environment (Python>=3.9):
```
$ git clone https://github.com/ryuryukke/mint.git
$ cd mint
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Set your Hugging Face cache directory:
```
$ export HF_HOME=/path/to/huggingface_cache
```

Run evaluation on all methods for MIA:
```
$ python run.py --task mia --domain arxiv --methods all --model_name pythia-160m
```
Run evaluation on all methods for detection:
```
$ python run.py --task detection --domain wiki --methods all --model_name llama-chat
```
Please see more details for options in ``scripts/*.sh``.

## MINT Supports
We currently cover **4 common baselines**, **7 state-of-the-art MIAs**, and **5 state-of-the-art machine text detectors**. Please [submit an issue](https://github.com/ryuryukke/mint/issues/new) for more method support.

| Methods | Category | Description | Identifier |
|-----------|----------|----------|----------|
| Loss | Baselines | the likelihood of a target sample | `loss`
| Entropy | Baselines | the expected likelihood of a target sample | `entropy`
| Rank | Baselines | the average rank of the predicted token at each step | `rank`
| LogRank | Baselines | the average log-rank of the predicted token at each step | `logrank`
| [Reference](https://arxiv.org/abs/2012.07805) | MIA | the difference in the target loss between the model and another reference model | `ref`
| [Zlib](https://arxiv.org/abs/2012.07805) | MIA | the ratio of the target loss and the zlib compression score of the target | `zlib`
| [Neighborhood](https://arxiv.org/abs/2305.18462) | MIA | the difference between the target loss and the average loss over its perturbed samples | `neighborhood`
| [Min-K%](https://arxiv.org/abs/2310.16789) | MIA | the average of log-likelihood of the $k$\% tokens with lowest probabilities | `min_k`
| [Min-K%++](https://arxiv.org/abs/2404.02936) | MIA | a standardized version of Min-K% over the model's vocabulary | `min_k_plus`
| [ReCaLL](https://arxiv.org/abs/2406.15968) | MIA | the relative log-likelihood between a target sample and a set of non-member examples | `recall`
| [DC-PDD](https://arxiv.org/abs/2409.14781) | MIA | the cross-entropy between the token likelihoods under the model and the laplace-smoothed unigram token frequency distribution under some reference corpus | `dc_pdd`
| [DetectGPT](https://arxiv.org/abs/2301.11305) | Detection | the difference between the target loss and the average loss over its perturbed samples | `detectgpt`
| [Fast-DetectGPT](https://arxiv.org/abs/2310.05130) | Detection | an efficient version of DetectGPT via fast-sampling technique and score normalization | `fastdetectgpt`
| [Binoculars](https://arxiv.org/abs/2401.12070) | Detection | the ratio of the target perplexity to the cross entropy of the target sample under some reference model | `binoculars`
| [DetectLLM](https://arxiv.org/abs/2306.05540) | Detection | a variant of DetectGPT instead of using LogRank as the core quantity | `detectllm`
| [Lastde++](https://arxiv.org/abs/2410.06072) | Detection | the multi-scale diversity entropy measuring the local fluctuations in likelihood across a target text sequence | `lastde_doubleplus`


<!-- | Category | Methods |
|-----------|----------|
| Baselines | Loss, Rank, LogRank, Entropy |
| MIAs | [Reference](https://arxiv.org/abs/2012.07805), [Zlib](https://arxiv.org/abs/2012.07805), [Neighborhood](https://arxiv.org/abs/2305.18462), [Min-K%](https://arxiv.org/abs/2310.16789), [Min-K%++](https://arxiv.org/abs/2404.02936), [ReCaLL](https://arxiv.org/abs/2406.15968), [DC-PDD](https://arxiv.org/abs/2409.14781) |
| Detectors | [DetectGPT](https://arxiv.org/abs/2301.11305), [Fast-DetectGPT](https://arxiv.org/abs/2310.05130), [Binoculars](https://arxiv.org/abs/2401.12070), [DetectLLM](https://arxiv.org/abs/2306.05540), [Lastde++](https://arxiv.org/abs/2410.06072) | -->

## Datasets
We employ the MIMIR benchmark for MIAs and the RAID benchmark for detection.

| Benchmark | Models | Domains |
|:-----------|:--------|:----------|
| [MIMIR](https://github.com/iamgroot42/mimir) | Pythia-160M, 1.4B, 2.8B, 6.7B, 12B | Wikipedia (knowledge), Pile CC (general web), PubMed Central and ArXiv (academic), HackerNews (dialogue), GitHub and DM Mathematical (technical) |
| [RAID](https://github.com/liamdugan/raid) | GPT-2-XL, MPT-30B-Chat, LLaMA-2-70B-Chat, ChatGPT and GPT-4 | Wikipedia and News (knowledge), Abstracts (academic), Recipes (instructions), Reddit (dialogue), Poetry (creative), Books (narrative), Reviews (opinions) |



## Running on a custom dataset
You can add a custom dataset by adding new if-else block to ``load_evaluation_data()`` in ``run.py``.

## Running a custom attack or detector
You can add a custom attack or detector by creating a new directory under `methods/` and registering it in `src/method.py`. Please follow the shared format defined in `src/method.py`.

## Citation
If you find our code or ideas useful in your research, please cite our work:
```
@misc{koike2025machinetextdetectors,
      title={Machine Text Detectors are Membership Inference Attacks}, 
      author={Ryuto Koike and Liam Dugan and Masahiro Kaneko and Chris Callison-Burch and Naoaki Okazaki},
      year={2025},
      eprint={2510.19492},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19492}, 
}
```



## Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. These research results were also obtained from the commissioned research (No.22501) by National Institute of Information and Communications Technology (NICT), Japan. In addition, this work was supported by JST SPRING, Japan Grant Number JPMJSP2106.
