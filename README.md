# Dual Student Networks for Data-Free Model Stealing

This is the code for the paper [Dual Student Networks for Data-Free Model Stealing](https://arxiv.org/abs/2309.10058) published to ICLR 2023.


## Dependencies
This repository was tested using Python 3.9.

After installing torch and torchvision, the dependencies that can be installed using the `pip` environment file provided:
```
pip install -r requirements.txt
```

## Replicating  Results

### Load Victim Model Weights
First, download the pretrained victim model weights from [this dropbox](https://www.dropbox.com/sh/lt6w0nq3msp4do0/AADmJk2k3LQqFqWt9916W-nra?dl=0). The two file names are `cifar10-resnet34_8x.pt` and `svhn-resnet34_8x.pt`. The CIFAR10 weights were found on the [Data Free Adversarial Distillation](https://github.com/VainF/Data-Free-Adversarial-Distillation) dropbox.

Then, store the pre-trained model weights at the following location

`dual_students/checkpoint/teacher/{victim_dataset}-resnet34_8x.pt`


### Perform Model Extraction
```
bash run_cifar_ds.sh
```
Logs and saved models can be found at `save_results/{victim_dataset}/`  


## Attribution

This code was built on the [Data-Free Model Extraction](https://github.com/cake-lab/datafree-model-extraction) repository, which in turn was built on code from the paper [Data Free Adversarial Distillation](https://github.com/VainF/Data-Free-Adversarial-Distillation). The weights and model architectures for Resnet34-8x and Resnet18_8x were also found on the repository released with the Data Free Adversarial Distillation paper.
