# Self-Attention-based 5G Uplink Resource Prediction
Transformer's Encoder-based 5G Transport Block Size (TBS) Prediction Model

This code corresponds to the research paper Jung et al., "Self-Attention-based Uplink Radio Resource Prediction in 5G Dual Connectivity", in IEEE Internet of Things Journal 2023 (https://ieeexplore.ieee.org/document/10147378) (DOI: 10.1109/JIOT.2023.3283490)

This code is based on George Zerveas et al., "A Transformer-based Framework for Multivariate Time Series Representation Learning", in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21) (https://dl.acm.org/doi/10.1145/3447548.3467401) (https://github.com/gzerveas/mvts_transformer)

# Contact
doubele112@naver.com

# Dataset
Some commercial South Korea 5G Non-Standalone (NSA) uplink resource datasets are available in the dataset folder.
We utilized it for testing our model.

# Execution Manual
You can train your own model from scratch or use our pre-trained model if you want.
We also uploaded our best model weights (model_best.pth), normalization pickle file, and source code for data converting, loading, and testing.

Data converting)

In original source code from KDD '21, the authors used ".ts" file for their training and testing input so we need to convert our ".csv" file into ".ts" file.
We uploaded our example converter code which makes a 1,000 ms input sequence using 8 features to predict average 100ms aggregated TBS.
You can customize your own converter code based on it.

Train from scratch example)

python3 src/main.py --output_dir {your_output_directory} --comment {your_comment}  --name {experiment comment} --records_file Regression_records.xls --data_dir {train_val_test_data_directory} --data_class tsra --pattern TRAIN --val_pattern VAL --test_pattern TEST --val_ratio 0 --epochs 100 --l2_reg 0.001 --optimizer RAdam --pos_encoding fixed --task regression --normalization standardization --num_layers 1 --num_workers 5 --n_proc 20 --normalization_layer LayerNorm --activation gelu --d_model 128 --num_heads 8 --dim_feedforward 256 --dropout 0.03 --lr 0.00025 --batch_size 128

Load&Test example)

python3 src/loadtest.py --output_dir {your_output_directory} --comment {your_comment} --name {experiment comment} --records_file Regression_records.xls --data_dir {test_data_directory} --data_class tsra --test_pattern TEST --test_only testset --val_ratio 0 --epochs 1 --lr 0.00025 --l2_reg 0.001 --optimizer RAdam --pos_encoding fixed --task regression --norm_from src/normalization.pickle --batch_size 1 --num_layers 1 --num_workers 5 --num_heads 8 --d_model 128 --dim_feedforward 256 --dropout 0.03 --normalization_layer LayerNorm --activation gelu --load_model src/model_best.pth

You can set your own hyperparameters.

# Model Optimization (Quantization)
We compressed our model for deploying it on commercial smartphones with reasonable resource (CPU, RAM, etc..) utilization.
We uploaded an example source code (model_optimization.py) that we used, so you may modify it.
