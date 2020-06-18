# Natural Language Processing with Classification and Vector Spaces
Đây là course đầu tiên về NLP specialization tại [Coursera](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearningai&utm_medium=institutions&utm_content=NLP_6/17_social) được 
sản xuất bởi [DeepLearning.ai](http://deeplearning.ai/).


Khoá này được dạy bởi hai chuyên gia về NLP: 
* Younes Bensouda Mourri (Instructor of AI at Stanford University - người cũng đã dạy trong DL specialization),  
* Łukasz Kaiser(Research Scientist at Google Brain, đồng tác giả của Tensorflow và
Transformer paper - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - cho bạn nào chưa biết thì sự ra đời của Transformer như là một cuộc đại cách mạng trong ngành NLP)



## Mục lục

* [Natural Language Processing with Classification and Vector Spaces](#neural-networks-and-deep-learning)
   * [Mục lục](#muc-luc)
   * [Tóm tắt nội dung](#course-summary)
   * [Introduction to deep learning](#introduction-to-deep-learning)
      * [What is a (Neural Network) NN?](#what-is-a-neural-network-nn)
      * [Supervised learning with neural networks](#supervised-learning-with-neural-networks)
      * [Why is deep learning taking off?](#why-is-deep-learning-taking-off)
   * [Neural Networks Basics](#neural-networks-basics)
      * [Binary classification](#binary-classification)
      * [Logistic regression](#logistic-regression)
      * [Logistic regression cost function](#logistic-regression-cost-function)
      * [Gradient Descent](#gradient-descent)
      * [Derivatives](#derivatives)
      * [More Derivatives examples](#more-derivatives-examples)
      * [Computation graph](#computation-graph)
      * [Derivatives with a Computation Graph](#derivatives-with-a-computation-graph)
      * [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
      * [Gradient Descent on m Examples](#gradient-descent-on-m-examples)
      * [Vectorization](#vectorization)
      * [Vectorizing Logistic Regression](#vectorizing-logistic-regression)
      * [Notes on Python and NumPy](#notes-on-python-and-numpy)
      * [General Notes](#general-notes)
   * [Shallow neural networks](#shallow-neural-networks)
      * [Neural Networks Overview](#neural-networks-overview)
      * [Neural Network Representation](#neural-network-representation)
      * [Computing a Neural Network's Output](#computing-a-neural-networks-output)
      * [Vectorizing across multiple examples](#vectorizing-across-multiple-examples)
      * [Activation functions](#activation-functions)
      * [Why do you need non-linear activation functions?](#why-do-you-need-non-linear-activation-functions)
      * [Derivatives of activation functions](#derivatives-of-activation-functions)
      * [Gradient descent for Neural Networks](#gradient-descent-for-neural-networks)
      * [Random Initialization](#random-initialization)
   * [Deep Neural Networks](#deep-neural-networks)
      * [Deep L-layer neural network](#deep-l-layer-neural-network)
      * [Forward Propagation in a Deep Network](#forward-propagation-in-a-deep-network)
      * [Getting your matrix dimensions right](#getting-your-matrix-dimensions-right)
      * [Why deep representations?](#why-deep-representations)
      * [Building blocks of deep neural networks](#building-blocks-of-deep-neural-networks)
      * [Forward and Backward Propagation](#forward-and-backward-propagation)
      * [Parameters vs Hyperparameters](#parameters-vs-hyperparameters)
      * [What does this have to do with the brain](#what-does-this-have-to-do-with-the-brain)
   * [Extra: Ian Goodfellow interview](#extra-ian-goodfellow-interview)
   
   ## Tóm tắt

Phần này mình dịch từ phần tóm tắt của chính [course](https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearningai&utm_medium=institutions&utm_content=NLP_6/17_social) này trên coursera:

Trong Course thứ nhất này, bạn sẽ: 

> a) Phân tích cảm xúc (sentiment Analysis) trên tweets với Logistic Reggression và naïve Bayes,

> b) Sử dụng không gian vector để tìm ra mối liên hệ giữa các từ. Sau đó, sử dụng PCA để giảm số chiều của không gian vector và biểu diễn nó trong không gian  2-D, 3-D để chúng ta có thể quan sát được những mối liên hệ đó.

> c) Viết một thuật toán đơn giản để dịch từ tiếng Anh sang tiếng Pháp bằng cách sử dụng những mô hỉnh Word Embedding đã được tính toán trước (pre-computed) và ??? nhạy cảm địa phương đối với những từ liên quan thông qua việc tìm kiếm bằng K-NN.

Vui lòng chắc chắn rằng: Bạn không có vấn đề gì với việc lập trình bởi Python; bạn có kiến thức cơ bản về ML, về nhân ma trận và về xác suất có điều kiện.

Khi kết thúc course này, bạn sẽ có thể thiết kế được các ứng dụng NLP phục vụ cho việc hỏi đáp (question-answering) và phân tích cảm xúc (sentiment analysis), bạn có thể tạo các công cụ để dịch tiếng và tóm tắt văn bản, thậm chí xây dựng một chatbot!


## Introduction to deep learning

> Be able to explain the major trends driving the rise of deep learning, and understand where and how it is applied today.

### What is a (Neural Network) NN?

- Single neuron == linear regression without applying activation(perceptron)
- Basically a single neuron will calculate weighted sum of input(W.T*X) and then we can set a threshold to predict output in a perceptron. If weighted sum of input cross the threshold, perceptron fires and if not then perceptron doesn't predict.
- Perceptron can take real values input or boolean values.
- Actually, when w⋅x+b=0 the perceptron outputs 0.
- Disadvantage of perceptron is that it only output binary values and if we try to give small change in weight and bais then perceptron can flip the output. We need some system which can modify the output slightly according to small change in weight and bias. Here comes sigmoid function in picture.
- If we change perceptron with a sigmoid function, then we can make slight change in output.
- e.g. output in perceptron = 0, you slightly changed weight and bias, output becomes = 1 but actual output is 0.7. In case of sigmoid, output1 = 0, slight change in weight and bias, output = 0.7. 
- If we apply sigmoid activation function then Single neuron will act as Logistic Regression.
-  we can understand difference between perceptron and sigmoid function by looking at sigmoid function graph.

- Simple NN graph:
  - ![](Images/Others/01.jpg)
  - Image taken from [tutorialspoint.com](http://www.tutorialspoint.com/)
- RELU stands for rectified linear unit is the most popular activation function right now that makes deep NNs train faster now.


