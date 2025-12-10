# CarRacing-v3 implementations

**Welcome!** This is a codebase for our capstone project : **CarRacing-v3 Algorithm Analysis and Optimization**

## Table of Contents

* 1\. [ Overview 🚀](#Overview)
* 2\. [ Usage 🔑](#Usage)
  * 2.1. [Installation](#Dependency)
  * 2.2. [To run CarRacingv3](#TorunCarRacing)
* 3\. [ Acknowledgments 🫡](#Acknowledgments)




##  1. <a name='Overview'></a> Overview 🚀

For this project, we focus on:
1. Implementing and successfully training a control agent using the **Deep Q-Learning
(DQN)**, **Advantage Actor-Critic (A2C)** and **Proximal Policy Optimization (PPO)**
algorithms directly from raw pixel inputs in the CarRacing-v3 environment.
2. Evaluating the performance of the trained agent using quantitative metrics such
as the average score achieved over multiple runs, score stability.
3. Analyzing the impact of key hyperparameters (e.g., learning rate, batch size,
entropy coefficient) on the convergence process and final performance of the proposed
algorithms within the CarRacing-v3 environment


##  2. <a name='Usage'></a> Usage 🔑

- Weights and results are saved in `.{algorithms}/results` by default.

####  2.1. <a name='Dependency'></a>Installation


> [!TIP] 
> We recommend using our code on **Kaggle** or **Google Colab** for more convenient training and recording, otherwise read on!

After cloning the repo, you can install dependencies locally on Python>=3.11 as follows:

```bash
pip install -r requirements.txt
```

or you can install all these packages with up-to-date versions:
- gymnasium
- box2D
- torch
- torchvision
- torchaudio
- numpy
- cv2






####  2.2. <a name='TorunCarRacing'></a>To run CarRacingv3
For each algorithm, if you want to start the training process:
```bash
# e.g., for PPO
python PPO/train.py
```
In case you want to try our results:
```bash
# e.g., for PPO
python PPO/main.py
```


##  3. <a name='Acknowledgments'></a> Acknowledgments 🫡
We are very grateful to work with [Nguyen Viet Anh](https://github.com/Vanh41), [Pham Duc Cuong](https://github.com/phamducuong05), [Cao Chi Cuong](https://github.com/cuongdztop1tg), [Nguyen Tuan Duc](https://github.com/ducer37), and [Trinh Hong Anh](https://www.facebook.com/honganhtrinh.2112) as a team, we have learned a lot and have many valuable discussions. Thank you Pham Duc Cuong, Nguyen Tuan Duc for the implementation of DQN, Cao Chi Cuong, Trinh Hong Anh with A2C and Nguyen Viet Anh with PPO. This project was carried out as part of the Introduction to Artificial Intelligence course, with the guidance and supervision of Professor Than Quang Khoat. 

Also, our work is built upon the following links:
- [Gymnasium Documentation](https://gymnasium.farama.org/index.html)
- [The Deep Q-Learning Algorithm](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm)
- [Advantage Actor-Critic](https://huggingface.co/blog/deep-rl-a2c)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations](https://stable-baselines3.readthedocs.io/en/master/)
