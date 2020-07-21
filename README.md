# CENALP
This repository contains the code of paper:  
 >CENALP: Du, Xingbo & Yan, Junchi & Zha, Hongyuan. (2019). Joint Link Prediction and Network Alignment via Cross-graph Embedding. 2251-2257. 10.24963/ijcai.2019/312.   
 
Before executing *CENALP*, you should install the following packages:  
``pip install sklearn``  
``pip install networkx``  
``pip install gensim``  
``pip install tqdm``  
The detailed version are ``python==3.7.2`` and ``networkx==2.4``, ``sklearn==0.22.1``, ``gensim==3.4.0``, ``tqdm==4.31.1``, but they are not mandatory unless the code doesn't work.  
## Basic usage  
### Data  
We provide a toy dataset, which is named 'bigtoy'. If you want to evaluate other datasets, please ensure that ground truth alignments and the edges for two networks are necessary. In addition, you can find the datasets used in the paper in http://thinklab.sjtu.edu.cn/paper/IJCAI19_network_dataset.zip.  

### Example  
In order to run *CENALP*, you can execute *demo.py* directly or execute the following command in ./src/:  
``python demo.py``  
To modify some of the parameters, you can run the code like this:  
``python demo.py --filename bigtoy --align_train_prop 0.0 --q 0.5``  
You can check out the other options:  
``python demo.py --help``  

### Evaluate
We use precision and recall to evaluate both link prediction and network alignment in this repository.

## Reference  
If you are interested in our researches, please cite our papers:  
[1] Du, Xingbo & Yan, Junchi & Zha, Hongyuan. (2019). Joint Link Prediction and Network Alignment via Cross-graph Embedding. 2251-2257. 10.24963/ijcai.2019/312.   
[2] Du, Xingbo & Yan, Junchi & Zhang, Rui & Zha, Hongyuan. (2020). Cross-network Skip-gram Embedding for Joint Network Alignment and Link Prediction. IEEE Transactions on Knowledge and Data Engineering. PP. 1-1. 10.1109/TKDE.2020.2997861. 
