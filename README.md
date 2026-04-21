# HPMNet: An Asymmetric Hypergraph-Prototype Mamba Network for Rice Mapping in Fragmented Landscapes

> **Notice**: This repository contains the official implementation of the paper *"An Asymmetric Hypergraph-Prototype Mamba Network for Rice Mapping in Fragmented Landscapes"*. The manuscript is currently under review at the **International Journal of Applied Earth Observation and Geoinformation (JAG)**. 
> 
> 🚀 **The complete training scripts, dataset processing codes, and pre-trained weights will be fully released here upon the acceptance of the paper.**

---

## 📖 Introduction

Accurate rice mapping in highly fragmented agricultural landscapes is essential for global food security but remains a persistent challenge in remote sensing. Existing deep learning methods frequently overlook the preservation of fine-grained spatial structures and are degraded by severe speckle noise in synthetic aperture radar (SAR) imagery.

To address this, we propose the **Hypergraph-Prototype Mamba Network (HPMNet)**. Our framework adopts a novel asymmetric fusion paradigm that bridges the fundamental modality gap between Optical and SAR data:
* **Hypergraph Prototype Encoder (HPE):** Extracts high-fidelity spatial priors from optical imagery to construct non-local parcel topologies, effectively resolving spatial hallucinations.
* **Spatiotemporal Mamba Encoder (STME):** Captures dynamic phenological trajectories from time-series SAR data with linear complexity while explicitly suppressing dominant speckle noise.
* **Structure-Guided Interaction Gate (SGIG):** Adaptively uses optical structural anchors to gate and recalibrate SAR features, preventing noise propagation during cross-modal fusion.

## 📊 Dataset

We evaluate our model on the global benchmark dataset **Plot-Rice** (v1.0). The dataset features extreme climatic and spatial diversity across 5 continents, presenting severe challenges in fragmented landscape delineation.
* More details about the dataset can be found in the original dataset publication.

## 🛠️ Environment Requirements

A complete `requirements.txt` will be provided. The basic environment requires:
* Python >= 3.8
* PyTorch >= 1.13.0
* CUDA >= 11.6
* [Mamba-ssm](https://github.com/state-spaces/mamba) 

## 📝 Citation

If you find our work or this repository useful, please consider citing our paper (BibTeX will be updated upon publication):

```bibtex
@article{li2026hpmnet,
  title={An Asymmetric Hypergraph-Prototype Mamba Network for Rice Mapping in Fragmented Landscapes},
  author={Li, Heping and Pan, Yucheng and Liu, Zhangle and Hussain, Sajid and Pan, Bin},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  year={2026},
  note={Under Review}
}
