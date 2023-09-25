
# Understanding Transformers: A Deep Dive

## Table of Contents
1. [Introduction](#introduction)
2. [Models Architecture](#models-architecture)
3. [Information Flow](#information-flow)
4. [Hands-On Examples](#hands-on-examples)
5. [Key Concepts](#key-concepts)
6. [Installation](#installation)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction

This repository is designed to provide an in-depth understanding of Transformer models, which have been pivotal in advancing the field of Natural Language Processing (NLP). Originating from the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). Here are some reference points in the history of transformer models:

![Transformers History](./img/Transformers_history.jpg)

 Broadly, transformers can be grouped into three categories:  
- BERT-like (also called auto-encoding Transformer models)
- GPT-like (also called auto-regressive Transformer models)
- BART/T5-like (also called sequence-to-sequence Transformer models)

#### What are Transformers 
All the listed Transformer models, including GPT, BERT, BART, and T5, are fundamentally designed as language models. They are trained using self-supervised learning techniques on extensive corpuses of raw, unlabeled text. In self-supervised learning, the model generates its own labels from the input data, thus obviating the need for human-annotated labels. 

In terms of architecture size and pretraining data, the trend generally leans towards larger models for enhanced performance. However, this comes with substantial computational and environmental costs, as seen in increased time requirements and carbon footprints. Consequently, there's a growing emphasis on sharing pretrained models to mitigate both computational expenditure and environmental impact.

Despite their extensive training, these pretrained models are not directly applicable for specialized tasks out of the box. This limitation is addressed through a technique called transfer learning. In transfer learning, a pretrained model is fine-tuned using a dataset that has been labeled for a specific task, essentially adapting the model's generalized understanding of language to the nuances of that task.



### Application

 Transformer models have set new benchmarks in a variety of tasks by leveraging attention mechanisms for both speed and performance. These models are versatile and can be employed for various applications:

📝 For text-based activities, they can perform functions like categorizing text, extracting relevant information, responding to queries, summarizing content, translating languages, and generating text. They are capable of handling these tasks in more than 100 different languages.
  
🖼️ In the realm of image processing, they can classify images, identify objects, and perform image segmentation.

🗣️ When it comes to audio data, these models are adept at recognizing spoken language and classifying different types of audio.

Beyond single-mode tasks, Transformer models also excel at multi-modal functions. These include answering questions based on tabular data, recognizing text from scanned documents, extracting relevant information from those documents, classifying videos, and performing visual-based question answering.

### Why Transformers?

1. **Parallelization**: Unlike RNNs, where computations are dependent on the previous step, Transformers allow for parallelization as each word or token is processed simultaneously. This dramatically speeds up training.

2. **Attention Mechanisms**: The introduction of attention mechanisms allows the model to focus on different parts of the input sequence when producing the output, resembling the way humans pay attention to specific portions of input when reading or listening.

3. **Scalability**: Transformers are highly scalable, meaning they perform well on a vast range of data sizes and complexities.

4. **Versatility**: Originally designed for NLP tasks, their architecture has proven effective in other domains like computer vision and reinforcement learning as well.

5. **Pre-trained Models**: The architecture's effectiveness has led to a plethora of pre-trained models, which can be fine-tuned for specific tasks, saving time and computational resources.


### Why Not Transformers?

1. **Computational Overheads**: The architecture can be resource-intensive, requiring significant amounts of memory and computational power, particularly for large datasets or complex tasks.

2. **Interpretability**: Transformers can be seen as "black boxes," making it difficult to understand how they arrive at specific predictions or decisions.

3. **Overfitting**: Due to their complexity, they are prone to overfitting, especially when the available dataset is small.

4. **Not Always the Best Fit**: For some tasks, simpler models like Decision Trees or Naive Bayes may provide similar performance but are easier to implement and interpret.

5. **Parameter Tuning**: The large number of hyperparameters can make it challenging to optimize the model, requiring extensive experience and knowledge in the field.


## Models Architecture

![Transformer Design](./img/transformer_architecture.jpg)

The design of the Transformer model consists of an Encoder-Decoder framework, enhanced by Multi-Head Attention features, which we will delve into more deeply.
All model states supplied by 🤗 Transformers are effortlessly amalgamated from the [huggingface.co model repository](https://huggingface.co/models), where they are directly uploaded.
For a comprehensive overview of the various architectures offered by Transformers, you can refer to the [architecture summary](https://huggingface.co/docs/transformers/model_summary) on their documentation page.


## Information Flow

The Transformer processes information through the following steps:

1. **Tokenization and Input Embedding**: Tokenizes the input sentence and converts each token into its corresponding embedding.
2. **Positional Encoding**: Adds positional information to the embeddings.
3. **Encoder**: Transforms the input sequence into a hidden representation.
4. **Decoder**: Takes the hidden representation to produce an output sequence.
5. **Task-Specific Layer**: Applies a task-specific transformation to the decoder output.
6. **Loss Computation and Backpropagation**: Computes the loss and updates the model parameters.

For a comprehensive understanding, refer to the [Information Flow section](#information-flow).


## Hands-On Examples

- [Text Classification (Sentiment Analysis)](./src/TextClassification.ipynb)
- [Text Generation and Feature Extraction](./src/transformer_tutorial_pytorch.ipynb)
- [Model Evaluation](./src/transformer_tutorial_pytorch.ipynb)

These Jupyter notebooks provide practical implementations to solidify your understanding of Transformer models.


## Key Concepts
- [Encoder-Decoder Architecture](./src/Encoder-Decoder.ipynb)
- [Attention Mechanism](./src/Attention.ipynb)
- [Multi-Head Attention](./src/Multi_Head.ipynb)


## Installation

#### With pip

```bash
pip install transformers
```

#### With conda

```bash
conda install -c huggingface transformers
```

For more detailed installation instructions, please refer to the [Installation Guide](#installation).


## Conclusion

This repository aims to serve as an exhaustive guide for understanding the intricacies of Transformer models. Whether you are a researcher, data scientist, or machine learning enthusiast, the content herein should offer a thorough understanding of how Transformers work and how they can be applied across various domains.



## References 
• The Transformer architecture : Encoder  
           https://www.youtube.com/watch?v=H39Z_720T5s&t=0s

 • The Transformer architecture : Encoder  
        Encoder models:   https://www.youtube.com/watch?v=MUqNwgPjJvQ&t=0s

 • Transformer models: Decoder  
        Decoder models:   https://www.youtube.com/watch?v=d_ixlCubqQw&t=0s

 • Transformer models: Encoder-Decoders 
        Encoder-Decoders: https://www.youtube.com/watch?v=0_4KEb08xrE&t=0s 


To understand what happens inside the Transformer network on a deeper level.
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2/
- Understanding Attention: https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
