# Understanding Transformers: A Hands-On Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Architectue](#Architectue)
3. [Flow of information](#flowofinformation)
4. [Hands-On Examples](#handsonexamples)
    1. [Text classification (sentiment analysis)](./TextClassification.ipynb)
    2. [Text generation and feature extraction](./Simple_Transformer_Language_Model.ipynb)
    3. [Model evaluation](./transformer_tutorial_pytorch.ipynb)
5. [Key Concepts](#key-concepts)
    1. [Attention Mechanism](#attention-mechanism)
    2. [Multi-Head Attention](#multi-head-attention)
    3. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
6. [Conclusion](#conclusion)

## Introduction
 Transformer – a model that uses attention to boost the speed with which these models can be trained. The biggest benefit, however, comes from how The Transformer lends itself to parallelization.In order to understand why Transformers are so important you need to understand how previous models functioned. So let’s try to break the model apart and look at how it functions.
 The Transformer was proposed in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). 

## Architectue
This is the architecture of Transformer if you haven't see this before.
<img src="./img/transformer_architecture.jpg">

## Flow of information 
The flow of information in a Transformer model for a sequence-to-sequence task with a classification objective (e.g., next word prediction) can be described as follows:

##### Step 1: Tokenization and Input Embedding
The input sentence "Je suis un étudiant" is tokenized into a sequence of subwords or words. These tokens are then converted into embeddings via an embedding layer.

**Tokenized Sentence**: \['Je', 'suis', 'un', 'étudiant'\]  
**Embedded Tokens**: \[ \text{Embedding}('Je'), \text{Embedding}('suis'), \text{Embedding}('un'), \text{Embedding}('étudiant') \]

##### Step 2: Addition of Positional Encoding (Time Signal)
Positional encodings are added to these embeddings to give the model information about the position of each word in the sequence. This is crucial since Transformers do not inherently understand the sequence order.

**Position-Encoded Embeddings**: \[ \text{Embedding}('Je') + \text{PositionalEncoding}(1), \ldots \]

##### Step 3: Encoder
The sequence of position-encoded embeddings is passed through the encoder, which consists of several layers of multi-head attention and feed-forward neural networks. The encoder outputs a "hidden representation" of the input.

**Hidden Representation**: \( \text{EncoderOutput} \)

##### Step 4: Decoder
The hidden representation is then passed through the decoder, which also has layers of multi-head attention and feed-forward neural networks. The decoder aims to reconstruct or transform this hidden representation into another sequence, usually for tasks like translation or summarization.

**Decoded Sequence**: \( \text{DecoderOutput} \)

##### Step 5: Task-Specific Layer (Classification)
The output of the decoder is finally passed through a task-specific layer. In the case of next-word prediction, this would typically be a softmax layer that converts the decoder output into a probability distribution over the vocabulary.

**Next Word Prediction**: \( \text{Softmax}(\text{DecoderOutput}) \)

##### Step 6: Loss Computation and Backpropagation
A loss is computed based on the difference between the predicted next word and the actual next word in the sequence. This loss is then used to update the model parameters during training via backpropagation.

**Loss**: \( \text{Loss}(\text{Predicted}, \text{Actual}) \)

The model starts by converting the input words into an enriched form that captures both their meaning and their position in the sentence. This enriched representation is transformed by the encoder into a hidden state, which the decoder then tries to use to accomplish the task at hand. Finally, the model's performance on this task is assessed using a loss function, and the model is updated accordingly.

NB: The model processes all inputs simultaneously, yet produces a singular output.

## Hands-On Example
- [Text classification (sentiment analysis)](./Transformers_samples/TextClassification.ipynb)
- [Text generation and feature extraction](./Transformers_samples/transformer_tutorial_pytorch.ipynb)
- [Model evaluation](./Transformers_samples/transformer_tutorial_pytorch.ipynb)


## Conclusion
Transformers have revolutionized the field of NLP and continue to be a subject of active research. This hands-on example is just the tip of the iceberg, and there's much more to explore!
