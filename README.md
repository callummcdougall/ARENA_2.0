This GitHub repo hosts the exercises and Streamlit pages for the ARENA 2.0 program.

You can find a summary of each of the chapters below. For more detailed information (including the different ways you can access the exercises), click on the links in the chapter headings.

# [Chapter 0: Fundamentals](https://arena-ch0-fundamentals.streamlit.app/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/prereqs.png" width="400">

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, and generative models).

Some highlights from this chapter include:
* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism


# [Chapter 1: Transformers](https://arena-ch1-transformers.streamlit.app/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/magnifying-glass-2.png" width="400">

The material on this page covers the next 8 days of the curriculum. It will cover transformers (what they are, how they are trained, how they are used to generate output) as well as mechanistic interpretability (what it is, what are some of the most important results in the field so far, why it might be important for alignment).

Some highlights from this chapter include:

* Building your own transformer from scratch, and using it to sample autoregressive output
* Using the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library developed by Neel Nanda to locate induction heads in a 2-layer model
* Finding a circuit for [indirect object identification](https://arxiv.org/abs/2211.00593) in GPT-2 small
* Intepreting model trained on toy tasks, e.g. classification of bracket strings, or modular arithmetic
* Replicating Anthropic's results on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

Unlike the first chapter (where all the material was compulsory), this chapter has 4 days of compulsory content and 4 days of bonus content. During the compulsory days you will build and train transformers, and get a basic understanding of mechanistic interpretability of transformer models which includes induction heads & use of TransformerLens. The next 4 days, you have the option to continue with whatever material interests you out of the remaining sets of exercises. There will also be bonus material if you want to leave the beaten track of exercises all together!


# [Chapter 2: Reinforcement Learning](https://arena-ch2-rl.streamlit.app/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/rl.png" width="400">

Reinforcement learning is an important field of machine learning. It works by teaching agents to take actions in an environment to maximise their accumulated reward.

In this chapter, you will be learning about some of the fundamentals of RL, and working with OpenAI’s Gym environment to run your own experiments.

Some highlights from this chapter include:

* Building your own agent to play the multi-armed bandit problem, implementing methods from [Sutton & Bardo](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
* Implementing a Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to play the CartPole game
* Applying RLHF to autoregressive transformers like the ones you built in the previous chapter

# Chapter 3: Training at Scale

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/training_at_scale.png" width="400">

Links coming soon!