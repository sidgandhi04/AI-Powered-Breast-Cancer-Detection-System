# Breast Cancer Detection System

A production-focused research codebase for automated breast cancer detection using multiple pretrained deep learning models, safety checks to reduce false positives, and tooling for training, evaluation, and deployment.

**Highlights**
- Multiple model backbones: ResNet18, EfficientNet, DenseNet, MobileNet, VGG and more
- Built-in safety checks for low-confidence / nonsensical inputs
- Streamlit demo for quick interactive evaluation
- Test and production validation suites

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the demo app (Streamlit):

```bash
streamlit run app_fixed.py
```

3. Run model tests and production checks:

```bash
python test_models_independently.py
python production_test_suite.py
```

## Models

The `models/` directory contains multiple pretrained weights used for inference and research experiments. Examples included in this repo:

- `resnet18_breast_cancer_optimized.pth` — fast, baseline model
- `efficientnet_b3_breast_cancer_advanced.pth` — accuracy-focused model
- `densenet121_breast_cancer_optimized.pth` — dense-connection architecture
- MobileNet and VGG variants for lightweight and classic baselines
- GAN weights for synthetic data experiments

Use the provided inference and training scripts in the root and `Trainer/` folder to evaluate or retrain models.

## Safety & Reliability

This project includes measures to reduce false positives and increase robustness:

- Input sanity checks (detects random noise and solid-color images)
- Prediction confidence thresholds and multi-stage validation
- Production test suite covering performance and stability scenarios

## Project layout

```
BreastCancer-detection/
├─ app_fixed.py                  # Streamlit demo + inference wrapper
├─ gan_models.py                 # GAN model definitions
├─ Trainer/                      # Training scripts and helpers
│  ├─ train_all_models.py
│  ├─ train_gan.py
│  └─ check_dataset.py
├─ models/                       # Pretrained weights used by the app
├─ research_visualizations/      # Plots and visual assets for the paper
├─ requirements.txt              # Python dependencies
└─ README.md
```

## Training & Evaluation

- Use `Trainer/train_all_models.py` or `train_optimized.py` to train models on your dataset.
- Use `Trainer/train_gan.py` for GAN experiments and synthetic data generation.
- Visualizations and benchmarking scripts live under `research_visualizations/`.

## Reproducibility

- Set up a Python virtual environment and install `requirements.txt`.
- Ensure GPU drivers and CUDA/CuDNN are configured if training or running heavy inference.

## Contributing

PRs, issue reports, and dataset notes are welcome. For changes that affect model accuracy, include evaluation results and the training config.

## License

This repository does not include a license file. Add one if you plan to reuse or redistribute the code.

## Support

If you encounter problems, run the production test suite for diagnostics:

```bash
python production_test_suite.py
```
