Can a Recurrent Neural Network learn an allocentric policy from multiple egocentric experiences
of an agent?

We start by developing a Q learning algorithm which is capable of computing the q values of a grid world
with an agent.

Various gridworlds from various egocentric perspectives of the agent are used to form a dataset containing the
State, Action, and q-value pair for each state.

This dataset is supplied to the Elman network in order to compute if the network is capable enough to find the
underlying pattern within the dataset and predict the q value of the next state,
which it has not been trained on.

The basic idea is to determine how does a simple LLM compare against various traditional techniques such as
q-learning.

