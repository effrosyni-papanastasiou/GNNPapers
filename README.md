# Diffusion Papers GNN Summary

Summaries of Information Diffusion Prediction Papers that use techniques based on Graph Neural Networks (GNNs).

## 2021
1. **Fully Exploiting Cascade Graphs for Real-time Forwarding Prediction Xiangyun**
*Xiangyun Tang, Dongliang Liao, Weijie Huang, Liehuang Zhu, Meng Shen, and Jin Xu*. AAAI, 2021, pp.582-590. [paper](https://www.aaai.org/AAAI21Papers/AAAI-5502.TangX.pdf)

- Goal: Popularity prediction Given: a set S = {S1, ..., Sβ } of infection status results observed on a diffusion network G in β historical diffu- sion processes, where Sℓ = (xℓ 1, ..., xℓ n) is a n-dimensional vector that records the final infection status, xℓ i ∈ {0, 1} (0 denotes uninfected, and 1 denotes infected) of each node vi ∈ V observed at the end of the ℓ-th diffusion process (ℓ ∈ {1, . . . , β}). Infer: the unknown edge set E of diffusion network G. In the problem statement, except for the given infection status results S observed on the n nodes of the objective diffusion network G, no other information about infections and the network, such as infection timestamps, initially infected nodes, and the number m of directed edges in the network, is known.(Regression).  Real-time forwarding amount prediction of on-line contents  
- Dataset:  
** [Weibo Dataset](https://github.com/CaoQi92/DeepHawkes): posts generated on June 1st, 2016, and tracks all re-tweets within the next 24 hours.  <br>
** Multimedia Content Dataset: multimedia contents from August 1, 2019 to September 30, 2019 and track all forwarding of each multimedia con- tent within the next 75 hours.
- Method: novel approach for cascade graph embedding that captures cascade graph features in terms of diffusion, scale and temporal, and includes a short-term variation sen- sitive method for modeling the historical variation of cas- cade graph size
- Conclusion: time-series modeling and cascade graph embedding are able to complement each other to achieve better prediction results for real-time forwarding prediction.

2. **CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction.**
*Fan Zhou, Xovee Xu, Kunpeng Zhang, Siyuan Liu and Goce Trajcevski.*
TKDE, 14 pages, Nov 2021. [paper&code](https://github.com/Xovee/casflow)

- Goal: Popularity prediction (Classification). whether an item gets more reposts than a certain threshold
- Dataset:
** [Twitter] (https://github.com/Xovee/casflow/tree/master/dataset) : reposts & friendship graph <br>
** [Weibo Dataset](https://github.com/CaoQi92/DeepHawkes): reposts & friendship graph
** [APS](https://journals.aps.org/datasets) (Released by American Physical Society, obtained at Jan 17, 2019).
- Method: non-linear information diffusion inference and models the information diffusion process by learning the latent representation of both the structural and temporal information.blearn the node-level and cascade-level latent factors in an unsupervised manner. 
- Conclusion:

3. **Decoupling Representation and Regressor for Long-Tailed Information Cascade Prediction**
*Fan Zhou, Liu Yu, Xovee Xu, and Goce Trajcevski.* SIGIR, Virtual Event, Jul 11-15, 2021, pp. 1875-1879. [paper](https://dl.acm.org/doi/10.1145/3404835.3463104?cid=99659687041)

- Goal: Popularity prediction (Classification). by predicting whether a cascade will exceed the median size of all cascades 
- Dataset:
** [Twitter] (Lilian Weng, Filippo Menczer, and Yong-Yeol Ahn. 2013. Virality prediction and
community structure in social networks. Scientific Reports 3 (2013).) : reposts & friendship graph <br>
** [Weibo Dataset](https://github.com/CaoQi92/DeepHawkes): reposts & friendship graph
- Method: 
- Conclusion:


## 2020
1. **Joint Learning of User Representation with Diffusion Sequence and Network Structure.**
*Wang, Zhitao, Chengyao Chen, and Wenjie Li.*
TKDE 2020.[paper](https://ieeexplore.ieee.org/document/9094385)

1. **HID: Hierarchical Multiscale Representation Learning for Information Diffusion.**
*Zhou Honglu, Shuyuan Xu, and Zouhui Fu.*
IJCAI 2020.[paper](https://www.ijcai.org/Proceedings/2020/0468.pdf)

1. **Inf-VAE: A Variational Autoencoder Framework to Integrate Homophily and Influence in Diffusion Prediction.**
*Aravind Sankar, Xinyang Zhang, Adit Krishnan, Jiawei Han.*
WSDM 2020.[paper](https://arxiv.org/pdf/2001.00132.pdf)

1. **Cascade-LSTM: A Tree-Structured Neural Classifier for Detecting Misinformation Cascades.**
*Francesco Ducci, Mathias Kraus, Stefan Feuerriegel.*
KDD 2020.[paper](https://www.research-collection.ethz.ch/handle/20.500.11850/415267) [code](https://github.com/MathiasKraus/CascadeLSTM)

1. **Variational Information Diffusion for Probabilistic Cascades Prediction.**
*Fan Zhou, Xovee Xu, Kunpeng Zhang, Goce Trajcevski, Ting Zhong.*
INFOCOM, Virtual conference, Jul 6-9, 2020, pp. 1618-1627. [[paper]](https://ieeexplore.ieee.org/document/9155349)

1. **DyHGCN: A Dynamic Heterogeneous Graph Convolutional Network to Learn Users’ Dynamic Preferences for Information Diffusion Prediction.**
*Chunyuan Yuan, Jiacheng Li, Wei Zhou, Yijun Lu, Xiaodan Zhang, and Songlin Hu.*
ECMLPKDD 2020. [paper](https://arxiv.org/pdf/2006.05169.pdf)

1. **A Heterogeneous Dynamical Graph Neural Networks Approach to Quantify Scientific Impact.**
*Fan Zhou, Xovee Xu, Ce Li, Goce Trajcevski, Ting Zhong and Kunpeng Zhang.*
arXiv 2020. [paper](https://xovee.cn/archive/paper/arXiv_20_HDGNN_Xovee.pdf) [code](https://github.com/Xovee/hdgnn)

1. **Cascade-LSTM: Predicting Information Cascades using Deep Neural Networks.**
*Sameera Horawalavithana, John Skvoretz, Adriana Iamnitchi.*
arXiv 2020. [paper](https://arxiv.org/pdf/2004.12373.pdf)

1. **Predicting Information Diffusion Cascades Using Graph Attention Networks**
*Meng Wang, and Kan Li*
International Conference on Neural Information Processing (ICONIP), 2020, pp. 104-112

1. **Popularity prediction on social platforms with coupled graph neural networks.**
*Qi Cao, Huawei Shen, Jinhua Gao, Bingzheng Wei, Xueqi Cheng.*
WSDM 2020.

## 2019
1. **Multi-scale Information Diffusion Prediction with Reinforced Recurrent Networks.**
*Cheng Yang, Jian Tang, Maosong Sun, Ganqu Cui, Zhiyuan Liu.*
IJCAI 2019.[paper](https://www.ijcai.org/proceedings/2019/0560.pdf)

1. **Neural diffusion model for microscopic cascade study.**
*Cheng Yang, Maosong Sun, Haoran Liu,Shiyi Han, Zhiyuan Liu, and Huanbo Luan.*
 TKDE 2019.
[paper](https://arxiv.org/pdf/1812.08933.pdf)

1. **Information Cascades Modeling via Deep Multi-Task Learning.**
*Xueqin Chen,  Kunpeng Zhang, Fan Zhou, Goce Trajcevski, Ting Zhong, and Fengli Zhang.*
 SIGIR 2019.
[paper](https://dl.acm.org/citation.cfm?id=3331288)

1. **Understanding Information Diffusion via Heterogeneous Information Network Embeddings.**
*Yuan Su, Xi Zhang, Senzhang Wang, Binxing Fang, Tianle Zhang, Philip S. Yu.*
 DASFAA 2019.
[paper](https://link.springer.com/chapter/10.1007/978-3-030-18576-3_30)

1. **NPP: A neural popularity prediction model for social media content.**
*Guandan Chen, Qingchao Kong, Nan Xu, Wenji Mao.*
 Neurocomputing 2019.
[paper](https://www.sciencedirect.com/science/article/pii/S0925231218314942)

1. **DeepFork: Supervised Prediction of Information Diffusion in GitHub.**
*Ramya Akula, Niloofar Yousefi, Ivan Garibay.*
[paper](https://arxiv.org/pdf/1910.07999.pdf)

1. **Information Diffusion Prediction via Recurrent Cascades Convolution.**
*Xueqin Chen, Fan Zhou, Kunpeng Zhang, Goce Trajcevski, Ting Zhong, and Fengli Zhang.*
 IEEE ICDE 2019.
[paper](https://ieeexplore.ieee.org/abstract/document/8731564)

1. **Deep Learning Approach on Information Diffusion in Heterogeneous Networks.**
*Soheila Molaei, Hadi Zare, Hadi Veisi.*
 KBS 2019.
[paper](https://arxiv.org/pdf/1902.08810.pdf)

1. **Cascade2vec: Learning Dynamic Cascade Representation by Recurrent Graph Neural Networks.**
*Zhenhua Huang, Zhenyu Wang, Rui Zhang.*
 IEEE Access 2019.
[paper](https://ieeexplore.ieee.org/abstract/document/8846015)

1. **Prediction of Information Cascades via Content and Structure Integrated Whole Graph Embedding.**
*Xiaodong Feng, Qiang Zhao, Zhen Liu.*
 BSMDMA 2019.
[paper](https://www.comp.hkbu.edu.hk/~xinhuang/BSMDMA2019/3.pdf)

1. **COSINE: Community-Preserving Social Network Embedding From Information Diffusion Cascades.**
*Yuan Zhang, Tianshu Lyu, Yan Zhang.*
 AAAI 2019.
[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16364)

1. **A Recurrent Neural Cascade-based Model for Continuous-Time Diffusion.**
*Sylvain Lamprier.*
 ICML 2019.
[paper](http://proceedings.mlr.press/v97/lamprier19a.html)

1. **Prediction Model for Non-topological Event Propagation in Social Networks.**
*Zitu Liu, Rui Wang, Yong Liu.*
 ICPCSEE 2019.
 [paper](https://link.springer.com/chapter/10.1007/978-981-15-0118-0_19)

1. **Information Diffusion Prediction with Network Regularized Role-based User Representation Learning.**
*Zhitao Wang, Chengyao Chen, Wenjie Li.*
 TKDD 2019.
[paper](https://dl.acm.org/citation.cfm?id=3314106)

1. **Hierarchical Diffusion Attention Network.**
*Zhitao Wang, Wenjie Li.*
 IJCAI 2019.
[paper](https://pdfs.semanticscholar.org/a8a7/353a42b90d2f43504783dc81ff28c11a9da5.pdf)

1. **Predicting Future Participants of Information Propagation Trees.**
*Hsing-Huan Chung, Hen-Hsen Huang, Hsin-Hsi Chen.*
 IEEE/WIC/ACM International Conference on Web Intelligence 2019.
[paper](https://dl.acm.org/citation.cfm?id=3352540)

1. **Dual Sequential Prediction Models Linking Sequential Recommendation and Information Dissemination.**
*Qitian Wu, Yirui Gao, Xiaofeng Gao, Paul Weng, and Guihai Chen.*
 KDD 2019.
[paper](https://dl.acm.org/citation.cfm?id=3330959)

1. **Community structure enhanced cascade prediction.**
*Chaochao Liu, Wenjun Wang, Yueheng Sun.*
 Neurocomputing 2019.
[paper](https://www.sciencedirect.com/science/article/pii/S0925231219307751)

## 2018

1. **DeepDiffuse: Predicting the 'Who' and 'When' in Cascades.**
*Sathappan Muthiah, Sathappan Muthiah, Bijaya Adhikari, B. Aditya Prakash, Naren Ramakrishnan.*
 ICDM 2018.
[paper](http://people.cs.vt.edu/~badityap/papers/deepdiffuse-icdm18.pdf)

1. **A sequential neural information diffusion model with structure attention.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 CIKM 2018.
 
[paper](https://dl.acm.org/doi/10.1145/3269206.3269275)
1. **Attention network for information diffusion prediction.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 WWW 2018.
[paper](https://dl.acm.org/citation.cfm?id=3186931)

1. **Inf2vec:Latent representation model for social influence embedding.**
*Shanshan Feng, Gao Cong, Arijit Khan,Xiucheng Li, Yong Liu, and Yeow Meng Chee.*
 ICDE 2018.
[paper](https://www.ntu.edu.sg/home/arijit.khan/Papers/Inf2Vector_ICDE18.pdf)

1. **Who will share my image? Predicting the content diffusion path in online social networks.**
*W. Hu, K. K. Singh, F. Xiao, J. Han, C.-N. Chuah, and Y. J. Lee.*
 WSDM 2018.
[paper](https://arxiv.org/pdf/1705.09275.pdf)

1. **Learning sequential features for cascade outbreak prediction.**
*Chengcheng Gou, Huawei Shen, Pan Du, Dayong Wu, Yue Liu, Xueqi Cheng.*
 Knowledge and Information System 2018.
[paper](https://link.springer.com/article/10.1007/s10115-017-1143-0)

1. **Predicting the Popularity of Online Content with Knowledge-enhanced Neural Networks.**
*Hongjian Dou, Wayne Xin Zhao, Yuanpei Zhao, Daxiang Dong, Ji-Rong Wen, Edward Y. Chang.*
 KDD 2018.
[paper](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_8.pdf)

1. **Predicting Temporal Activation Patterns via Recurrent Neural Networks.**
*Giuseppe Manco, Giuseppe Pirrò, Ettore Ritacco.*
 ISMIS 2018.
[paper](https://link.springer.com/chapter/10.1007/978-3-030-01851-1_33)

1. **Weighted estimation of information diffusion probabilities for independent cascade model.**
*Yoosof Mashayekhi, Mohammad Reza Meybodi, Alireza Rezvanian.*
 ICWR 2018.
[paper](https://ieeexplore.ieee.org/document/8387239/)

1. **Modeling Topical Information Diffusion over Microblog Networks.**
*Kuntal Day, Hemank Lamba, Seema Nagar, Shubham Gupta, Saroj Kaushik.*
 International Conference on Complex Network and their Applications 2018.
[paper](Ihttps://link.springer.com/chapter/10.1007/978-3-030-05411-3_29)

1. **DeepInf: Social Influence Prediction with Deep Learning.**
*Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, Jie Tang.*
 KDD 2018.
[paper](https://arxiv.org/pdf/1807.05560.pdf)

1. **CAS2VEC: Network-Agnostic Cascade Prediction in Online Social Networks.**
*Zekarias T. Kefato, Nasrullah Sheikh, Leila Bahri, Amira Soliman, Alberto Montresor, Sarunas Girdzijauskas.*
 SNAMS 2018.
[paper](https://people.kth.se/~sarunasg/Papers/Kefato2018cas2vec.pdf)

1. **A Variational Topological Neural Model for Cascade-based Diffusion in Networks.**
*Sylvain Lamprier.*
 arXiv 2018.
[paper](https://arxiv.org/pdf/1812.10962.pdf)

1. **Joint Modeling of Text and Networks for Cascade Prediction.**
*Cheng Li, Xiaoxiao Guo, Qiaozhu Mei.*
 ICWSM 2018.
 
[paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/viewFile/17804/17070)
1. **CRPP: Competing Recurrent Point Process for Modeling Visibility Dynamics in Information Diffusion.**
*Avirup Saha, Bidisha Samanta, Niloy Ganguly.*
 CIKM 2018.
[paper](https://dl.acm.org/doi/abs/10.1145/3269206.3271726)

## 2017

1. **DeepCas: An end-to-end predictor of information cascades.**
*C. Li, J. Ma, X. Guo, and Q. Mei.*
 WWW 2017.
[paper](https://arxiv.org/pdf/1611.05373.pdf)

1. **Topological recurrent neural network for diffusion prediction.**
*Jia Wang, Vincent W Zheng, ZeminLiu, and Kevin Chen-Chuan Chang.*
 ICDM 2017.
[paper](https://arxiv.org/pdf/1711.10162.pdf)

1. **DeepHawkes: Bridging the gap between prediction and understanding of information cascades.**
*Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, and Xueqi Cheng.*
 CIKM 2017.
[paper](http://www.bigdatalab.ac.cn/~shenhuawei/publications/2017/cikm-cao.pdf)

1. **Cascade dynamics modeling with attention-based recurrent neural network.**
*Yongqing Wang, Huawei Shen, Shenghua Liu, Jinhua Gao, and Xueqi Cheng.*
 IJCAI 2017.
[paper](https://www.ijcai.org/proceedings/2017/0416.pdf)






