# ML - machine learning

This is just a documentation of my learning progress. 

## 2018 - Start with Object Detection

Inspired by object detection for cars with DarkNet (see this [TED talk from 2017](https://www.youtube.com/watch?v=Cgxsv1riJhI) by Joseph Redmon) and David's bachelor work at [HCMUTE](http://en.hcmute.edu.vn/) in connection with a car at the end of 2018 I started to learn more about machine learning.

Posenet runs on TensorFlow.lite in a browser on WebGL even on a smartphone. We tested it in December 2018 in Seoul, Korea. In March 2019 I got TensorFlow.js running with my RX470 with 43 fps. 

![Posenet the park](TensorFlow.js/posenet/2019-03_thepark.jpg)

During 2019 NVIDIA announced the [Jetson Nano](https://en.wikipedia.org/wiki/Nvidia_Jetson) developer kit and with students from AISVN we try to win one in a competition. Eventually we order a package.

![Jetson Nano car](https://kreier.github.io/jetson-car/pic/2019_jetson_car.jpg)

Early 2020 some supply chains delay orders, but we finally have the hardware. Now it needs to be combined - and development stalls until 2024.

### Facemesh example

<!--
![Facemesh example](https://github.com/tensorflow/tfjs-models/blob/master/facemesh/demo.gif?raw=true)
-->
![Facemesh example](https://github.com/kreier/ml/blob/main/pic/facemesh.gif?raw=true)

### Schedule for 2020

In [this article](https://towardsdatascience.com/from-a-complete-newbie-to-passing-the-tensorflow-developer-certificate-exam-d919e1e5a0f3) Harsheev Desai describes his journey to become a TensorFlow Developer with Certificate in 5 months.

#### 1. Learn Python
- [Python bootcamp at udemy](https://www.udemy.com/course/complete-python-bootcamp/)
- [Coursera python](https://www.coursera.org/specializations/python#courses)
- [List of 10 courses at medium.com](https://medium.com/better-programming/top-5-courses-to-learn-python-in-2018-best-of-lot-26644a99e7ec)

#### 2. Learn Machine Learning Theory
 
- [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) on Statistics, Calculus and Linear Algebra

#### 3. Learn Data Science Libraries

Some of these libraries are Pandas (data manipulation and analysis), Numpy (support for multi-dimensional arrays and matrices), Matplotlib (plotting) and Scikitlearn (creating ML models).

- [Pandas videos](https://www.youtube.com/playlist?list=PLeo1K3hjS3uuASpe-1LjfG5f14Bnozjwy)
- [NumPy videos](https://www.youtube.com/watch?v=QUT1VHiLmmI) or [freeCodeCamp](http://freecodecamp.org/)
- [MatPlotLib videos](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF)
- [Scikitlearn at udemy](https://www.udemy.com/course/machinelearning/) or [3 hour video](https://www.youtube.com/watch?v=pqNCD_5r0IU)

#### 4. Deep Learning Theory

- [Coursera Deep Learning](https://www.coursera.org/specializations/deep-learning?#courses)
- [Inner workings of DNN in practical implementations](https://medium.com/analytics-vidhya/what-i-learned-from-building-a-deep-neural-network-from-scratch-and-why-you-should-do-it-too-a2e6f422d3db)

#### 5. TensorFlow Certificate

- [Coursera TensorFlow in Practice](https://www.coursera.org/professional-certificates/tensorflow-in-practice#courses)

One reason for tensorflow can be seen in this graph regarding popularity on stack overflow:

![popularity tensorflow](pic/tensorflow_stack_overflow.png)

More about the certificate [here on medium](https://medium.com/@harshit_tyagi/google-certified-tensorflow-developer-learning-plan-tips-faqs-my-journey-9f88016048e3). It was [introduced in March 2020](https://blog.tensorflow.org/2020/03/introducing-tensorflow-developer-certificate.html) but by 2024 it [no longer exists](https://www.tensorflow.org/certificate).

## 2022 - Teach ML in [Advanced Automation](https://github.com/ssis-aa) at SSIS in Unit 5

<img src="https://kreier.github.io/ml/pic/nn_2022.jpg" width="30%" align="right">

As covered in a [SSIS Stories](https://www.ssis.edu.vn/student-life/post-details/~board/hs/post/robots-on-a-roll-automation-and-algorithms) in March 2022 we made great progress in creating our own Neural Network, Training it and then doing interference on them. See also [our website](https://sites.google.com/ssis.edu.vn/automation).

## 2024 - start with LLMs

Andrej Karpathy offers a step-py-step guide to build your own Generative Pre-trained Transformer (GPT) starting with 1,000,000 characters from Shakespeare that you can train on your own GPU. Well, at least if it supports CUDA >7.0, otherwise the compiler throws an error (like on my slightly older GTX 960):

``` sh
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Found NVIDIA GeForce GTX 960 which is too old to be supported by the triton GPU compiler, which is used
as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 5.2
```

Let's see what I have and what CUDA capabilities these support:

| GPU name     | CUDA cores | Compute Capability |      at     | architecture | RAM GB |
|--------------|-----------:|:------------------:|:-----------:|--------------|-------:|
| Quadro FX580 |         32 |         1.1        | hp Z600     | [Tesla](https://en.wikipedia.org/wiki/Tesla_(microarchitecture)) (2006) |    0.5 |
| GTX 650      |        384 |         3.0        | E3-1226 v3  | [Kepler](https://en.wikipedia.org/wiki/Kepler_(microarchitecture)) (2012) |     1 |
| GT750M       |        384 |         3.0        | MBPr15 2014 | Kepler (2012) |   0.5 |
| M1000M       |        512 |         5.0        | Zbook 15 G3 | Kepler (2012) |     1 |
| GTX960       |       1024 |         5.2        | E5-2696 v3  | Maxwell (2014) |    2 |
| Jetson Nano  |        128 |         5.3        |             | [Maxwell](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture)) (2014) |    4 |
| T4           |       2560 |         7.5        | Google Colab | Turing (2018) |   16 |
| RTX3060 Ti   |       4864 |         8.6        | i7-8700     | Ampere (2020)  |    8 |
| RTX3070 Ti   |       6144 |         8.6        | i3-10100    | Ampere (2020)  |    8 |

Only __two__ of 8 are supported by the Triton GPU compiler. How about a newer GPU? At least I can use the T4 in Google's collaboratory for free. The training taikes one hour. And you get two hours for free.

## History

- __October 2018__ Successful installed darknet on ubuntu, object detection works for stills. Don't have a webcam, video does not work yet.
- __December 2018__ TensorFlow.lite in a browser on my iPhone 7 runs at 6 fps, demonstrated in Seoul
- __March 2019__ posenet runs in the browser with new RX470 with 43 fps
- __December 2019__ On [hackster.io](https://hackster.io) starts a new competition [AI at the Edge Challenge](https://www.hackster.io/contests/NVIDIA) where you can win a Jetson Nano. I apply and eventually just buy one from [arrow](https://www.arrow.com/)
- __February 2020__ The Jetson car is purchased, Wifi module and 7" display as well. Needs completion - without students due to COVID-19
- __July 2024__ Reactivated the [https://kreier.github.io/jetson-car/](https://kreier.github.io/jetson-car/) project. The hardware is from 2019 (NVIDIA) but the software is still Ubuntu 18.04 LTS. Updates brake simple things like `make` and `gcc`.
- __August 2024__ Started to work on [https://kreier.github.io/nano-gpt/](https://kreier.github.io/nano-gpt/) to learn more about LLMs, following Andrej Karpathy's project [https://github.com/karpathy/nanogpt](https://github.com/karpathy/nanogpt)
