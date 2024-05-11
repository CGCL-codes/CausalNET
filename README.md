# CausalNET

This repository provides the source code and appendix for CausalNET (IJCAI'24).

## Environments
```shell
pip install -r requirements.txt
```

## Experiments
Execute the following steps to replicate our results on the two real datasets (i.e., Micro-24 and Micro-25):
```shell
cd ./model
conda activate python_envs_name
## before running the following commands, please replace the directory path in '...' by your own settings
## '...': Main.py: line22 & line24
python -u main.py -g 1 -task test_M24 -opt ./configs/config_m24.yaml
python -u main.py -g 2 -task test_M25 -opt ./configs/config_m25.yaml
```

```txt
Note: 
(1) the DAG (causal graph) files will be saved in the subdirectory named './dags/final_prob/'.
(2) the DAG file for 'dataset_name' will be named as 'dataset_name_i.npy'.
```

Based on the hyper-parameter settings we provided, the estimated training duration for CausalNET is expected to be 2～6 hours (depending on the status of the hardware devices).


## Acknowledgement
Thanks to these excellent open source projects:
- [TrustworthyAI](https://github.com/huawei-noah/trustworthyAI/tree/master/datasets)
- [Topological Hawkes Process (TNNLS'22)](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle/castle/algorithms/ttpm)
- [Transformer Hawkes Process (ICML'20)](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)
- [CUTS: NEURAL CAUSAL DISCOVERY FROM IRREGULAR TIME-SERIES DATA (ICLR'23)](https://github.com/jarrycyx/UNN)


## Citation
If you find the repository helpful, please cite the following paper:
```tex
@inproceedings{hua2024causalnet,
  title={CausalNET: Unveiling Causal Structures on Event Sequences by Topology-Informed Causal Attention},
  author={Hua, Zhu and Hong, Huang and Kehan, Yin and Zejun, Fan and Hai, Jin and Bang, Liu},
  booktitle={Proceedings of the 33rd International Joint Conference on Artificial Intelligence},
  year={2024}
}
```

## Contact
Please feel free to contact us if you have questions, or need explanations:
[huazhu@hust.edu.cn](mailto:huazhu@hust.edu.cn).
