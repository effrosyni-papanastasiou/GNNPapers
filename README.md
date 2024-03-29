# Diffusion Prediction using Graph Neural Networks techniques

## 2022
1. **Reverse Graph Learning for Graph Neural Network**

1. **Modularity-Aware Graph Autoencoders for community detection and link prediction** 
Guillaume Salha-Galvan, Johannes Lutzeyer

- Goal: community detection + link prediction by considering both the initial graph
structure and modularity-based prior communities when computing embedding spaces
- Dataset:  Cora, Citeseer and Pubmed citation networks, graph generated from a stochastic block model, 

## 2021

0. [KALO] **Full-Scale Information Diffusion Prediction With Reinforced Recurrent Networks**

1. A review of uncertainty quantification in deep learning: Techniques,
applications and challenges [paper](https://reader.elsevier.com/reader/sd/pii/S1566253521001081?token=C014E104031323AED2EEE7C5A4135636E765AA67D63AA1967F7CE7C735A6E30C12E4C1D00BCD6D9A144150336032B56F&originRegion=eu-west-1&originCreation=20220626145444)

1.Reverse Graph Learning for Graph Neural Network[paper](https://ieeexplore.ieee.org/abstract/document/9749776)

1. Graph Posterior Network: Bayesian Predictive
Uncertainty for Node Classification[paper](https://proceedings.neurips.cc/paper/2021/file/95b431e51fc53692913da5263c214162-Paper.pdf)

1. **Fully Exploiting Cascade Graphs for Real-time Forwarding Prediction Xiangyun**
*Xiangyun Tang, Dongliang Liao, Weijie Huang, Liehuang Zhu, Meng Shen, and Jin Xu*. AAAI, 2021, pp.582-590. [paper](https://www.aaai.org/AAAI21Papers/AAAI-5502.TangX.pdf)

- Goal: Popularity prediction. 
- Dataset:  
** [Weibo Dataset](https://github.com/CaoQi92/DeepHawkes): posts generated on June 1st, 2016, and tracks all re-tweets within the next 24 hours.  <br>
** Multimedia Content Dataset: multimedia contents from August 1, 2019 to September 30, 2019 and track all forwarding of each multimedia con- tent within the next 75 hours.
- Method: embedding of cascade graph features in terms of diffusion, scale and temporal properties. Attention CNN mechanism that captures short-term variation over time on cascade graph size and merges the local features within a fixed window.Long Short Term Meomory (LSTM) over the attention CNN to learn the historical trend. Linear Regression (LR); Gradient Boosting Decision Tree (GBDT); CNN and LSTM.
- Conclusion: time-series modeling and cascade graph embedding are able to complement each other to achieve better prediction results for real-time forwarding prediction.

2. [IMPORTANT] **Information Cascades Prediction With Graph Attention** Chen Zhihao, Wei Jingjing, Liang Shaobin, Cai Tiecheng, Liao Xiangwen, Frontiers in Physics [paper](https://www.frontiersin.org/article/10.3389/fphy.2021.739202)

- Goal:  To that end, in this paper, we propose a recurrent neural network model with graph attention mechanism, which constructs a seq2seq framework to learn the spatial-temporal cascade features. Specifically, for user spatial feature, we learn potential relationship among users based on social network through graph attention network. Then, for temporal feature, a recurrent neural network is built to learn their structural context in several different time intervals based on timestamp with a time-decay attention. Finally, we predict the next user with the latest cascade representation which obtained by above method.
- How: 

3. **Utilizing the simple graph convolutional neural network as a model for simulating influence spread in networks** Alexander V. Mantzaris, Douglas Chiodini & Kyle Ricketson

- Goal: The methodological approach applies the simple graph convolutional neural network in a novel setting. Primarily that it can be used not only for label classification, but also for modeling the spread of the influence of nodes in the neighborhoods based on the length of the walks considered. This is done by noticing a common feature in the formulations in methods that describe information diffusion which rely upon adjacency matrix powers and that of graph neural networks. Examples are provided to demonstrate the ability for this model to aggregate feature information from nodes based on a parameter regulating the range of node influence which can simulate a process of exchanges in a manner which bypasses computationally intensive stochastic simulations.
- How:

## 2020

1. [kalo] NEURIPS **Variational Inference for Graph Convolutional
Networks in the Absence of Graph Data and
Adversarial Settings** [paper](https://proceedings.neurips.cc//paper/2020/file/d882050bb9eeba930974f596931be527-Paper.pdf)

1. **Inf-VAE: A Variational Autoencoder Framework to Integrate Homophily and Influence in Diffusion Prediction.**
*Aravind Sankar, Xinyang Zhang, Adit Krishnan, Jiawei Han.*
WSDM 2020.[paper](https://arxiv.org/pdf/2001.00132.pdf)
- Goal: predict the set of all influenced users
- Method: Unlike existing diffusion prediction methods that only consider local induced propagation structures, Inf-VAE is a generalizable VAE
framework that models social homophily through graph neural network architectures. First work to comprehensively
exploit social homophily and temporal influence in diffusion prediction. Given a sequence of seed user activations, Inf-VAE employs an expressive co-attentive fusion network to jointly attend over their social and temporal embeddings to predict the set of
all influenced users. Inf-VAE with co-attentions is faster than state-of-the-art recurrent methods by an order of magnitude

1. **Information cascades prediction with attention neural network**, Yun Liu, Zemin Bao, Zhenjiang Zhang, Di Tang & Fei Xiong, Human-centric Computing and Information Sciences volume 10, Article number: 13 (2020)
- Goal:  predicting the increment size of the information cascade based on an end-to-end neural network.
Learning the representation of a cascade in an end-to-end manner circumvents the dif-
ficulties inherent to blue the design of hand-crafted features.
- Method: An attention mechanism, which consists of the intra-attention and inter-gate module, was designed to obtain
and fuse the temporal and structural information learned from the observed period of the cascade. 
Step 1: Input embedding. Cascade graph: set of cascade paths that are sampled through multiple random walk processes. Feed into a gated recurrent neural network to obtain the hidden representation. Each node in the sequence is represented as a one-hot vector, q ∈ RN , where N is the
total number oIn the case of Task 2, we expect AMI and ARI scores to slightly decrease w.r.t. Task 1,
as models will only observe incomplete versions of the graphs when learning embedding
spaces. Task 2 will further assess whether empirically improving community detection
inevitably leads to deteriorating the original good performances of GAE and VGAE
models on link prediction. As our proposed Modularity-Inspired GAE and VGAE are
designed for joint link prediction and community detection, we expect them to 1) reach
comparable (or, ideally, identical) AUC/AP link prediction scores w.r.t. standard GAE
and VGAE, while 2) reaching better community detection scores.

1. **Cascade-LSTM: A Tree-Structured Neural Classifier for Detecting Misinformation Cascades.**
*Francesco Ducci, Mathias Kraus, Stefan Feuerriegel.*
KDD 2020.[paper](https://www.research-collection.ethz.ch/handle/20.500.11850/415267) [code](https://github.com/MathiasKraus/CascadeLSTM)

1. **DyHGCN: A Dynamic Heterogeneous Graph Convolutional Network to Learn Users’ Dynamic Preferences for Information Diffusion Prediction.**
*Chunyuan Yuan, Jiacheng Li, Wei Zhou, Yijun Lu, Xiaodan Zhang, and Songlin Hu.*
ECMLPKDD 2020. [paper](https://arxiv.org/pdf/2006.05169.pdf)

1. **A Heterogeneous Dynamical Graph Neural Networks Approach to Quantify Scientific Impact.**
*Fan Zhou, Xovee Xu, Ce Li, Goce Trajcevski, Ting Zhong and Kunpeng Zhang.*
arXiv 2020. [paper](https://xovee.cn/archive/paper/arXiv_20_HDGNN_Xovee.pdf) [code](https://github.com/Xovee/hdgnn)

1. **Cascade-LSTM: Predicting Information Cascades using Deep Neural Networks.**
*Sameera Horawalavithana, John Skvoretz, Adriana Iamnitchi.*
arXiv 2020. [paper](https://arxiv.org/pdf/2004.12373.pdf)

1. **Predicting Information Diffusion Cascades Using Graph Attention Networks** [paper](https://citationsy.com/archives/q?doi=10.1007/978-3-030-63820-7_12)
*Meng Wang, and Kan Li*
International Conference on Neural Information Processing (ICONIP), 2020, pp. 104-112

Effective information cascade prediction plays a very important role in suppressing the spread of rumors in social networks and providing accurate social recommendations on social platforms. This paper improves existing models and proposes an end-to-end deep learning method called CasGAT. The method of graph attention network is designed to optimize the processing of large networks. After that, we only need to pay attention to the characteristics of neighbor nodes. Our approach greatly reduces the processing complexity of the model. We use realistic datasets to demonstrate the effectiveness of the model and compare the improved model with three baselines. Extensive results demonstrate that our model outperformed the three baselines in the prediction accuracy.

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

1. **Deep Learning Approach on Information Diffusion in Heterogeneous Networks.**
*Soheila Molaei, Hadi Zare, Hadi Veisi.*
 KBS 2019.
[paper](https://arxiv.org/pdf/1902.08810.pdf)

1. **Cascade2vec: Learning Dynamic Cascade Representation by Recurrent Graph Neural Networks.**
*Zhenhua Huang, Zhenyu Wang, Rui Zhang.*
 IEEE Access 2019.
[paper](https://ieeexplore.ieee.org/abstract/document/8846015)

1. **A Recurrent Neural Cascade-based Model for Continuous-Time Diffusion.**
*Sylvain Lamprier.*
 ICML 2019.
[paper](http://proceedings.mlr.press/v97/lamprier19a.html)

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

1. REALLY SIMILAR] Bayesian graph convolutional neural networks for semi-supervised classification [paper](https://ojs.aaai.org/index.php/AAAI/article/download/4531/4409)

1. **DeepInf: Social Influence Prediction with Deep Learning.**
*Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, Jie Tang.*
 KDD 2018.
[paper](https://arxiv.org/pdf/1807.05560.pdf)
- Goal: In general, DeepInf takes a user’s local network as the input to a
graph neural network for learning her latent social representation.
We design strategies to incorporate both network structures and
user-specific features into convolutional neural and attention net-
works.We aim to predict the action status of a user given the
action statuses of her near neighbors and her local structural infor-
mation. For example, in Figure 1, for the central user v, if some of
her friends (black circles) bought a product, will she buy the same
product in the future? Th
- Method: By architecting network embedding [ 37 ], graph convolution [25 ], and graph
attention mechanism [49 ] into a unified framework, we expect that
the end-to-end model can achieve better performance than conven-
tional methods with feature engineering. In specific, we propose
a deep learning based framework, DeepInf, to represent both in-
fluence dynamics and network structures into a latent space. To
predict the action status of a user v, we first sample her local neigh-
bors through random walks with restart. After obtaining a local
network as shown in Figure 1, we leverage both graph convolution
and attention techniques to learn latent predictive signals.
- Dataset: OAG, Digg, Twitter, and Weibo. (with friendships)
Important reference paper with cascades + GNNs + GATs.

1. **DeepDiffuse: Predicting the 'Who' and 'When' in Cascades.**
*Sathappan Muthiah, Sathappan Muthiah, Bijaya Adhikari, B. Aditya Prakash, Naren Ramakrishnan.*
 ICDM 2018.
[paper](http://people.cs.vt.edu/~badityap/papers/deepdiffuse-icdm18.pdf)
- Goal: cascade prediction utilizing only two types of (coarse) information, viz. which node is infected and its corresponding infection time. 

1. !!!!!!!!!**Inf2vec:Latent representation model for social influence embedding.**
*Shanshan Feng, Gao Cong, Arijit Khan,Xiucheng Li, Yong Liu, and Yeow Meng Chee.*
 ICDE 2018.
[paper](https://www.ntu.edu.sg/home/arijit.khan/Papers/Inf2Vector_ICDE18.pdf)
- Goal: Given a social network and its action log, modeling in-
fluence propagation aims to infer the influence probabilities
between users. As a fundamental problem of social influence
analysis in social networks, learning influence parameters has
been investigated in several proposals . efinition 2: (Social Influence Embedding Problem) Given a social network G = (V, E), an action log A = {Di}, where Di is a diffusion episode, and the number of dimension K, we aim to learn: (1) source embedding Su ∈ RK and target embedding Tu ∈ RK in K dimensional latent space for each user u, as well as (2) influence ability bias bu and conformity bias  ̃bu for each user u. Compared with existing influence learning work that es- timates probabilities for edges [2], [3], [10], our solution to the social influence embedding problem aims to better capture the social influence propagation by effectively capturing the influence relations among users and handling the data sparsity. In addition, the existing methods are designed for particular influence spread models, e.g., the IC model and the assumed influence spread models cannot take into consideration user similarity factor. In contrast, we aim to incorporate user similarity into parameter learning.

- Dataset: . One is Digg, which contains information about stories displayed on the front page of Digg (digg.com) in June 2009 [25]. The Digg dataset comprises 68K users connected by 823K edges. The dataset also contains Digg votes, each of which records users’ voting on a particular story and the voting time. The other dataset is Flickr, which contains a friendship graph and a list of favorite marking records of the photo sharing social network (www.flickr.com) [26]. There are 162K users connected by 10M edges. The statistics of two datasets are stated in Table I. Each action contains the information of (user, item, time). We observe that the action data is very sparse. It is challenging to effectively learn social influence propagation parameters based on such sparse data.
- 
- Method: In social influence embedding, the propagation relationship between two users is modeled by the similarity between their vectors. Note that influence propagation is directed Given a user, we need to identify the users who are probably influenced by the user, which is called as influence context of the user. However, given a social network and a diffusion episode, we cannot exactly know the influence context. In addition, the social influence would spread in the social network, i.e., a user may influence other persons through the intermediate users. Furthermore, it is very important to incorporate similarity of user interest in the influence model, although it is challenging to incorporate such additional infor- mation. We next present our approach to generate the influence context, including local influence context and global similarity context. We utilize a random walk with restart process to model a user’s influence spread in the influence propagation network. This approach has two benefits. First, it can simulate the influence spread sequences, and thus high-order influence can be considered. Second, it can produce more influence pairs by additionally considering high-order influence. Hence we can alleviate the challenge caused by the sparsity of diffusion data. Note that A. Goyal et al. [21] utilize a similar strategy to solve sparsity issue. They propose a credit distribution model to assign influence in propagation network. However, they only exploit first-order and second- order influence propagation. With random walk process, our method can capture higher-order propagation.dom walk process reflects the local influence neigh- borhood. Given an influence propagation network Gi and a user u, we generate the influence context set Ci u , which contains the users that are probably influenced by user u. We utilize a random walk with restart strategy to generate Ci u. Starting from user u, it randomly chooses one neighbor to visit. Based on the currently visited user, it randomly samples one neighbor of this user to visit next. At each step, it has some probability to go back to user u (In our work, we set the restart ratio as 0.5 by following default setting of the work [13]). 

1. [IMPORTANT]  **A sequential neural information diffusion model with structure attention.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 CIKM 2018.[paper](https://dl.acm.org/doi/10.1145/3269206.3269275)
 
 In this paper, we propose a novel sequential neural network with structure attention to model information diffusion. The proposed model explores both sequential nature of an information diffusion process and structural characteristics of user connection graph. The recurrent neural network framework is employed to model the sequential information. The attention mechanism is incorpo- rated to capture the structural dependency among users, which is defined as the diffusion context of a user. A gating mechanism is further developed to effectively integrate the sequential and structural information. The proposed model is evaluated on the diffusion prediction task. The performances on both synthetic and real datasets demonstrate its superiority over popular baselines and state-of-the-art sequence-based models.
 
1. [IMPORTANT] **Attention network for information diffusion prediction.**
*Zhitao Wang, Chengyao Chen, and Wenjie Li.*
 WWW 2018.
[paper](https://dl.acm.org/citation.cfm?id=3186931)

In this paper, we propose an attention network for diffusion pre-
diction problem. The developed diffusion attention module can
effectively explore the implicit user-to-user diffusion dependency
among information cascade users. Besides, the user-to-cascade im-
portance and the time-decay effect are captured and utilized by the
model. The superiority of the proposed model over state-of-the-art
methods is demonstrated by experiments on real diffusion data.

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

1. **CAS2VEC: Network-Agnostic Cascade Prediction in Online Social Networks.**
*Zekarias T. Kefato, Nasrullah Sheikh, Leila Bahri, Amira Soliman, Alberto Montresor, Sarunas Girdzijauskas.*
 SNAMS 2018.
[paper](https://people.kth.se/~sarunasg/Papers/Kefato2018cas2vec.pdf)
- Goal: predict whether a cascade is going to become viral or not.
- Method:  Based on our main premise, intuitively we seek to
model the initial speed of a cascade (that is, the speed by
which a cascade starts its spread) or the user reaction times
at the early stage of the cascade, as well as its momentum.
As we shall empirically demonstrate in Section IV-A, this is
a strong signal for potential virality.     
-    While several attempts towards this end exist, most of the current approaches rely on features extracted from the underlying network structure over which the content spreads. Recent studies have shown, however, that prediction can be effectively performed with very little structural information about the network, or even with no structural information at all. In this study we propose a novel network-agnostic approach called CAS2VEC, that models information cascades as time series and discretizes them using time slices. For the actual prediction task we have adopted a technique from the natural language processing community. 

1. [VERY SIMILAR] **A Variational Topological Neural Model for Cascade-based Diffusion in Networks.**
*Sylvain Lamprier.*
 arXiv 2018.
[paper](https://arxiv.org/pdf/1812.10962.pdf)
- Goal: While some of them define graphical markovian models to extract temporal relationships between node infections in networks, others consider diffusion episodes as sequences of infections via recurrent neural models. In this paper we pro- pose a model at the crossroads of these two extremes, which embeds the history of diffusion in infected nodes as hidden continuous states. Depending on the trajectory followed by the content before reaching a given node, the distribution of influence probabilities may vary. Depending on the trajectory followed by the content before reaching a given node, the distribution of influence probabilities may vary. However, content trajectories are usually hidden in the data, which induces challenging learning problems. We propose a topological recurrent neu- ral model which exhibits good experimental performances for diffusion modelling and prediction.
- How: The first bayesian topological RNN for sequences with tree dependencies, which we apply for diffusion cascades modelling. Rather than building on a preliminary random walk process, the idea is to consider trajectory inference during learning, in order to converge to better representations of the infected nodes. Following the stochastic nature of diffusion, the model infers trajectories distributions from observations of infections, which are in turn used for the inference of infection probabilities in an iterative learning process. Our probabilistic model, based on the famous continuous- time independent cascade model (CTIC) (Saito et al., 2009) is able to extract full paths of diffusion from sequential observations of infections via black-box inference, which has 2 multiple applications in the field

n our model we consider that diffusion probabilities from any infected node v depend on a latent state associated to v, which embeds the past trajectory of the diffused content. This state depends on the state of the node u who first transmitted the content to v. Therefore, we need to rely on a continuous-time model such as CTIC (Saito et al., 2009), which serves as a basis for our work. In CTIC, two parameters are defined for each pair (u, v) of nodes in the network: ku,v ∈0; 1, which corresponds to the probability that node u succeeds in infecting v, and ru,v > 0, which corresponds to a time-delay parameter used in an exponential distribution when u infects v. If u succeeds in infecting v in an episode D, v is infected at time tD v = tD u + δ, where δ ∼ ru,v exp (−ru,v δ). These parameters are learned via maximizing the following likelihood on a set of episodes D

Recurrent Neural Diffusion Model
ach infected node v in an episode D owns a state zD v ∈ Rd depending on the path the content followed to reach v in D, with d the dimension of the representation space. Knowing the state zD u of the node u that first infected v, the state zD v is computed as: zD v = fφ(zD u , ω(f ) v ) (2) with fφ : Rd × Rd → Rd a function, with parameters φ, that transforms the state of u according to a shared representation ω(f ) v ∈ Rd of the node v. This function can either be an Elman RNN cell, a multi-layer perceptron (MLP) or a Gated Recurrent Unit (GRU). An LSTM could also be used here, but zD v should include both the cell and the state of v in that case



## 2017

1. **DeepCas: An end-to-end predictor of information cascades.**
*C. Li, J. Ma, X. Guo, and Q. Mei.*
 WWW 2017.
[paper](https://arxiv.org/pdf/1611.05373.pdf)
- Goal: Future size of a cascade.
- Dataset:  Given a snapshot of a social network at time t0, denote it as G = (V, E) where V is the set of nodes and E ⊂ V × V is the set of edges. A node i ∈ V represents an actor (e.g., a user in Twitter or an author in the academic paper network) and an edge (i, j) ∈ E represents a relationship tie (e.g., retweeting or citation) between node i and j up to t0.
One of the scenario is the cascade of Tweets on Twitter. cascades of Tweets (i.e., through retweeting) in June, 2016 from the official Decahose API (10% sample of the entire Tweet stream). As the follower/followee rela- tions are not available in the data and Twitter does not disclose the retweet paths, we follow existing work [30] and draw an edge from Twitter user A to B if either B retweeted a message of A or A men- tioned B in a Tweet. Comparing to a follower/followee network, this network structure accumulates all information cascades and reflects the truly active connections between Twitter users. We weigh an edge based on the number of retweeting/mentioning events be- tween the two users.
** []
- Method: Represents a cascade graph as a set of cas- cade paths that are sampled through multiple random walks processes. Such epresentation preserves node identities and bounds the loss of structural information. Analogically, cascade graphs are represented as documents, with nodes as words and paths as sentences. The challenge is how to sample the paths from a graph to assemble the “document,” which is also automatic learned through the end-to-end model to optimize the prediction of cascade growth. Once we have such a “document” assembled, deep learning techniques for text data could be applied in a similar way here.

2. **Topological recurrent neural network for diffusion prediction.**
*Jia Wang, Vincent W Zheng, ZeminLiu, and Kevin Chen-Chuan Chang.*
 ICDM 2017.
[paper](https://arxiv.org/pdf/1711.10162.pdf)
- Goal: estimating the probability of an inactive node to be activated next in a cascade.

3. **DeepHawkes: Bridging the gap between prediction and understanding of information cascades.**
*Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, and Xueqi Cheng.*
 CIKM 2017.
[paper](https://dl.acm.org/doi/pdf/10.1145/3132847.3132973)
- Goal: predicting the size of retweet cascades in Sina Weibo and predicting the citation of papers in APS.
- How: captures and extends the three interpretable factors of Hawkes process under deep learning framework, i.e., influence of users, self-exciting mechanism of each retweet and the time decay effect in information diffusion.learning user embeddings as the influence representation of users by the guide of future popularity is useful in popularity prediction. considering the entire retweet path through GRU structure instead of only considering the current retweet user in Hawkes process can significantly improve the prediction per- formance. In addition, the DeepHawkes model is flexible to learn the time decay effect using the proposed non-parametric way with- out prior domain knowledge. 

4.  [VERY SIMILAR] **Cascade dynamics modeling with attention-based recurrent neural network.**
*Yongqing Wang, Huawei Shen, Shenghua Liu, Jinhua Gao, and Xueqi Cheng.*
 IJCAI 2017.
[paper](https://www.ijcai.org/proceedings/2017/0416.pdf)
- Goal: modeling and predicting the cascades of resharing using sequential models (e.g., recurrent neural
network, namely RNN) that do not require knowing the underlying diffusion model.  The objective of sequence modeling in cascade dynamics is to formulate the conditional probability of next resharing behavior p((tk+1, uk+1)|Hk
- How: 1. attention-based RNN to capture the cross-dependence in cascade. 2. coverage strateg)y [Tu et al., 2016] to combat the misallocation of attention caused by the memoryless of traditional attention mechanism allowing alignments to better reflect the structure of propagation;.
- Results: proposed models outperform state-of-the-art models at both cascade prediction and inferring diffusion tree
- Data: - the input data is a collection of M cascades C = {Sm}M m=1. A cascade S = {(tk, uk)|uk ∈ U, tk [0, +∞) and k = 1, . . . , N } is a sequence of resharing behaviors ascendingly ordered by time, where U refers to user set in cascade. (tk, uk) pair of activation time and activated user.SYNTHETIC + REAL DATA



