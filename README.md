# ICT3909_NBR

Needed imports:

```bash
# PyTorch and related packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip3 install numpy pandas matplotlib optuna gymnasium

# Additional dependencies
pip3 install scikit-learn networkx tqdm
pip install "dask[complete]"
# Email dependencies
pip3 install secure-smtplib
```

Dataset can be found at: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset