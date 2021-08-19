---
date: 2021-08-17T16:29:26+02:00
draft: true
title: Autoregressive models, compression and metrics
---

## Introduction
Autoregressive models are quite popular today, especially since the OpenAI released its GPT models. 
The core idea of such a model is to iteratively predict parts of an object based on some already known context.
In this post I want to show how this type of model connects to information theory,
compression and metrics.
Nothing I present is new, but when I talk to people they often are not aware of all 
these things. 


## Autoregressive probability modelling
In a quite popular [blog-post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Andrej Karpathy showed how the relatively simple idea of predicting the next character of some text can lead to suprisingly good results.
In the years that have passed since then and a bit more compute spent on the problem the same idea led to large language models such as GPT3.
The idea is to model the probability distribution of a sequence $x_1 x_2 x_3 \dots
x_T$ using the probability chain rule:
$$p(x_1 \dots x_T) = p(x_T|x_1\dots x_{T-1}) \dots p(x_3|x_1 x_2) p
(x_2|x_1) p(x_1).$$
This basically says that one wants to predict the probability distribution of a symbol in a sequence given the preceding ones.

Now that we have this abstract idea we can think about how to actually implement it.
A simple model family are called [n-gram models](https://en.wikipedia.org/wiki/N-gram). 
These models only look at data chunks of size $n$ and predict the last element based on the $n-1$ preceding elements. 
For small enough $n$ one can build a table listing all the probabilities.
A 1-gram model, or unigram model, would make the following estimation
$$p(x_1 \dots x_T) = p(x_T) \dots p(x_3) p(x_2) p(x_1).$$
This predicts the next character of a text according to it's frequency in the 
data. Arguably this model will not give too good results.
Using longer context would probably already yield better results. For example a  3-gram model would do the following
$$p(x_1 \dots x_T) = p(x_T|x_{T-2}x_{T-1}) \dots p(x_3|x_2 x_1) p(x_2|x_1) p(x_1).$$
<!-- How do you deal with the beginning here -->
The probabilities are estimated by counting the n-grams in the data.

These models are limited by the length of their context.
When using longer contexts (large $n$) it becomes unfeasible to store
the probabilities in tables. For example lets assume that each
element in a sequence represents a word and we are using a 30-gram
model and build our table using the first half of Alice in Wonderland.
Given that its highly unlikely that a sequence of 30 words occurs more
than once we'll get a table with a lot of 30-grams all with a count
one. When trying to apply this 30-grams to the second half of the book
we'll find out that it is rather useless as the second half is not covered by them.

Here is where machine-learning comes in.
Recurrent neural networks and Transformers can be used to make use of larger contexts.
Instead of building a table such models are tuned
minimize a loss function that measures model quality by adapting model parameters.
It is not obvious which function should be chosen for this task.

All we know is that it would be good to have a function that is minimal if and only if the true probabilities are reported.
Such a function is also called a strictly proper scoring rule as elaborated on [in this article](https://en.wikipedia.org/wiki/Scoring_rule).
In practice the logarithmic scoring-rule is often used and we'll concentrate on it 
as it has an information theoretical interpretation that we'll get to later.
This loss function can be written as
$$\mathcal L(\theta) = \sum_{i=1}^N \sum_{i=1}^T -\log p(x_i|x_1\dots x_{i-1}).$$
This also corresponds to the negative log-likelihood:
$$
\begin{aligned} \mathcal L(\theta) &= \sum_{i=1}^T -\log p(x_i|x_1\dots x_{i-1}) \\\\ 
&= - \log  \prod_{i=1}^T p(x_i|x_1\dots x_{i-1})\\\\ 
&= - \log  p(x_1 \dots x_T) \end{aligned}
$$

## Compression
We'll now look at how this relates to compression of text.
This can be explained by the basics of information theory put forward by Claude Shannon in [A mathematical theory of communication](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6773024).
A core question treated in this paper is how to efficiently encode 
text. ASCII for example encodes each character using one byte.
But one could save capacity by encoding less frequent symbols
using shorter codes. 
Huffman came of with [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding), a way to compute an optimal encoding based on the expected symbol frequencies.
This image gives a good intuition of the method:

{{< figure src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Huffman_coding_visualisation.svg/1280px-Huffman_coding_visualisation.svg.png" caption="Huffman coding ([Source](https://en.wikipedia.org/wiki/Huffman_coding#/media/File:Huffman_coding_visualisation.svg))" width="70%" >}}

<!-- TODO: check if I understand the proof -->

In this example there are six different characters.
In a simple encoding we would need three bits to encode all the characters.
|Char|Code |
|---|-----|
| A | 000 |
| B | 001 |
| C | 010 |
| D | 011 |
| E | 100 |
| _ | 101 |

Thus on average we would need threes bits per character (BPC) to encode the given text. The table below shows the Huffman encoding from the figure above:

|Char|Code |Code length | p    |
|----|-----|---------   |------|
|_   | 00  | 2          |0.22  |
|D   | 01  | 2          |0.22  |
|A   | 10  | 2          |0.24  |
|E   | 110 | 3          |0.15  |
|C   | 1110| 4          |0.04  |
|B   | 1111| 4          |0.13  |

Using these values we can compute the average BPC:

$$ E[l(s)] = \sum_{s \in \mathcal{S}} p(s) l(s) = 
\underbrace{0.22 \cdot 2}_{-} +
\underbrace{0.22 \cdot 2}_{D} +
\underbrace{0.24 \cdot 2}_{A} +
\underbrace{0.15 \cdot 3}_{E} +
\underbrace{0.04 \cdot 4}_{C} + 
\underbrace{0.13 \cdot 4}_{B} = 2.5 BPC $$ 

So this encoding would save us half a bit per encoded character in contrast to the constant length encoding.

Using this encoding one can encode text into  a bitstring.
This bitstring can be decoded by reading bits until a code for a letter is obtained. 
This process is unique as the encoding is a [prefix code](https://en.wikipedia.org/wiki/Prefix_code). 
The uniqueness is easier to understand if one visualizes the tree during decoding.
One starts at the root node and follows the tree according to the read bits until one hits a leaf node. Hitting a leaf node signals that a new letter starts.

## Using machine learning for context
Above we modelled the probabilities using the letter frequencies.
What is apparent is that we can only reduce 
the BPC because some characters are more likely than others. 
It's not hard to see that the more concentrated the probabilities are,
the less BPC will be required.
In the extreme case where we are sure of the next character we can give that character the encoding `0` and we know that we only will need one bit per character.
Using the one-gram model above we unfortunately will not get these concentrated probabilities.
If we used more context to predict the frequency we could make more confident prediction**s**. The bold `s` in the preceding sentence for example could probably be predicted with high confidence.

Instead of just computing a static encoding we'll now look at how to use a machine 
learning model to use more context, obtain different probability estimates for each
position in the text and use them to compute an adaptive code.
For example we could use an LSTM for compression using the following steps:

1. Predict probabilities by LSTM given context.
2. Build the "Huffman-tree"
3. Encode the current character according to it
4. Add next character to context and go back to step 1.

Decompressing works analogously but instead of a sequence of characters
we consume a sequence of bits and build up the character sequence that
we feed into the LSTM. We can get the code for each step using the probability 
estimates obtained from the LSTM.
An toy example implementing this can be found [here](...).

## Entropy
Let's next look at a Huffman code based on letter frequencies which are powers of two:

|Char|Code |$p$               | $-\log_2(p)$ |
|----|-----|------------------|--------------|
|A | 0     | $ \frac{1}{2} $  | 1            |
|B | 10    | $ \frac{1}{4} $  | 2            |
|C | 110   | $ \frac{1}{8} $  | 3            |
|D | 111   | $ \frac{1}{8} $  | 3            |

We can see here that we can express the code length of a letter by the negative 
logarithm of its probability. 
We can write the expected BPC as a function of the probabilities 
$$H(p) =  \sum_i p_i (-\log_2 p_i). $$
This quantity is called the entropy of a probability distribution.
This expression gives a lower bound on the expected BPC as
shown in the [Shannon's source-coding theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem). 
In this case the expected BPC is equal to the entropy,
but let's say that the probability tree is changed an at
$p_A=0.51$ and the others $P_{BCD}=0.49$. This will decrease the
entropy, but the Huffman code will stay the same as we can't make the
code for A any shorter.

## Cross entropy
The entropy gives us a lower bound on the BPC we need to encode 
the text. But this assumes that the code used is created based on the
probability distribution that generated the data.
Estimation of the probability however only results in a imperfect approximation
$q$ of the true distribution $p$.
The so called cross-entropy gives a lower bound on the BPC for data generated by $p$ and codes derived using $q$:
$$H(p,q) = - \sum_i p_i \log_2 q_i. $$
The negative logarithm again represents the code length for a character
over which a weighted sum is taken according to the probability that characters actually appear in the text.

Now if we go back to our ML example we have a similar situation.
We have an imperfect estimate $q_i$ at each time step and need to
measure its usefulness for compression we need to evaluate it using the 
probability of the real data.
Now the problem with that is that we don't actually have the true
probability.
Thus we need to estimate it using a sample text $x_1\dots x_N$ which yields
$$H(p,q) = -E_{x_j \sim  p}[\log_2 q_{x_j}] = \frac{1}{N} \sum_{j=1}^N  \log_2 q_{x_j}.$$
Thus we get an estimate of how useful our predictions are for
compression.

## KL-divergence
Now one could ask oneself what the lower barrier for the cross-entropy
is and if one thinks about it for a bit it should be obvious that you
can't go lower than the entropy, as you should get the best result when your code 
is based on the real probabilities.
Manipulating the expression for the cross entropy a bit we get
$$
\begin{aligned} H(p,q) &= - \sum_i p_i (\log_2 q_i + \log_2 p_i - \log_2 p_i) \\\\ 
&= - \sum_i p_i \log_2 p_i - \sum_i p_i (\log_2 q_i  - \log_2 p_i) \\\\ 
&= H(p) + \sum_i p_i (\log_2 p_i - \log_2 q_i) \\\\ 
&= H(p) + \sum_i p_i \frac{\log_2 p_i}{\log_2 q_i} \\\\ 
&= H(p) + D_{KL}(p||q)\end{aligned}
$$
The last term is call the KL-Divergence from $p$ of $q$ which is in
fact always positive. So we get the result that the cross entropy is
always at least the entropy of $p$ with an additional term.

So how can we interpret this term?
The above can be written as
$$H(p,q) = H(p) + \sum_i p_i (-\log_2 q_i - (-\log_2 p_i)).$$
We write this in such a weird way because as mentioned above
$-\log q_i$ gives us the code length. Thus the KL-divergence basically
just computes the differences of code lengths for each character and
computes a weighted sum over them according to the actual data
distribution.

Now given that we cannot change $H(p)$ anyway we might also could
use the KL-divergence as a loss function instead of the cross entropy.

<!-- # Metric zoo
There are many metrics commonly used in language modelling.
Basically we now know that BPC is the same as cross entropy with the
reservation that often a base-2 logarithm is used for the former and
the natural logarithm for the latter. We've also seen that this also
corresponds to the negative log-likelihood if we normalize it by the
number of samples.

Another commonly used metric is the perplexity. -->




