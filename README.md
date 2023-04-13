# Chapter 1: Fundamentals

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, and generative models).

Some highlights from this chapter include:
* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism

The exercises exist in two forms: as self-contained Colab notebooks (links below), and in this GitHub repo (accessible via the [Streamlit homepage](https://fundamentals.streamlit.app/) which reads from this repo). Streamlit also allows you to publish webpages, so you can access that link to go to the Streamlit app homepage, and read all the same content from this README there.

<img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/prereqs.png" width=450>

## How you should use this material

### Option 1: Colab

The simplest way to get these exercises up and running is to use Colab. This guarantees good GPU support, and means you'll spend less time messing around with environments and IDEs. Each set of exercises will have a link to the accompanying Colab notebook, which you should make a copy of and work through. The Colabs have essentially the same structure as the Streamlit pages.

[Here](https://drive.google.com/drive/folders/1YnTChxQTJnJfFhqyHA44h9Nro79AaYpn?usp=sharing) is the link to the folder containing all the Colabs, and the data you'll need. You can find the individual Colabs below (not all the exercises have been converted into Colab form yet):

* Raytracing: [**exercises**](https://colab.research.google.com/drive/1tp-vd591FarVyn7pA2V9oYDqYiWmjEjF?usp=share_link), [**solutions**](https://colab.research.google.com/drive/19QroufIT25oZ5yG7JGWL5Jp9IPcsq0d4?usp=sharing)
* as_strided, convolutions and CNNs: [**exercises**](https://colab.research.google.com/drive/1hQE1inYldFI_mmpCiLbIW8yI2C-PxBev?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1VZk9ba3j7HJP9ChntblOoEAwxZukCgHn?usp=sharing)
* Build Your Own Backprop Framework: [**exercises**](https://colab.research.google.com/drive/1n-OG0x7kZfZaMCNO-S4L86-W6bE_jiVz?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1K3f_ebaaHDufnGbn_zzzTisejXTM_b01?usp=sharing)
* ResNets & Model Training: Links to Colab: [**exercises**](https://colab.research.google.com/drive/1N1Cu13q4dk2Z0qYgdy7Cnb6ESAlOu5ge?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1obMRz1Y9iXrJbQBXaYCBS61S-mxOIhWO?usp=sharing)
* Optimization & Hyperparameters: [**exercises**](https://colab.research.google.com/drive/1Wi_SVL8eDYiNcmcmUeF4GfkNfQKT6x3O?usp=sharing), [**solutions**](https://colab.research.google.com/drive/1JfIRCJZ_Fi_WJGneuOKKqF_qsxJfdbfZ?usp=sharing)

You can make a copy of the **exercises** notebooks in your own drive, and fill in the code cells whenever indicated. The solutions will be available in dropdowns next to each of the code cells. You can also look at the **solutions** notebooks if you're just interested in the output (since they have all the correct code filled in, and all the output on display within the notebook).

### Option 2: Your own IDE

An alternative way to use this material is to run it on an IDE of your own choice (we strongly recommend VSCode). The vast majority of the exercises will not require a particularly good GPU, and where there are exceptions we will give some advice for how to get the most out of the exercises regardless.

Full instructions for running the exercises in this way:

* Clone the [GitHub repo](https://github.com/callummcdougall/TransformerLens-intro) into your local directory.
* Open in your choice of IDE (we recommend VSCode).
* Make & activate a virtual environment
    * We strongly recommend using `conda` for this. You can install `conda` [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), and find basic instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
* Install requirements.
    * First, install PyTorch using the following command: `conda install pytorch=1.11.0 torchdata torchvision -c pytorch -y`.
    * Then install the rest of the requirements by navigating to the directory and running `pip install -r requirements.txt`.
* While in the directory, run `streamlit run Home.py` in your terminal (this should work since Streamlit is one of the libraries in `requirements.txt`).
    * This should open up a local copy of the page you're reading right now, and you're good to go!

To complete one of the exercise pages, you should:

* Navigate to `exercises` in the repo
* Create a file called `part1_answers.py` (or `part1_answers.ipynb` if you prefer using notebooks)
* Go through the Streamlit page, and copy over / fill in then run the appropriate code as you go through the exercises.