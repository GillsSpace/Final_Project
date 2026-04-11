# Final Project for CSCI 451: Optimizing Backgammon with Reinforcement Learning
### By Wills Erda, Camiel Schroeder, and Shingo Kodama

# Setup Instructions:

To set up this project, you will need to install gym-backgammon which is used to generate legal moves efficiently.* The following instructions are provided in the gym-backgammon repository README:

```bash
git clone https://github.com/dellalibera/gym-backgammon.git
cd gym-backgammon/
pip3 install -e .
```

Additionally, the following Python packages are required:

```bash
pip install torch torchinfo
```

> *Used with permission of Alessio Della Libera under the MIT License. See LICENSE file of gym-backgammon for details.

# Project Proposal:

## Abstract:

Losing at games is never fun. In this project we try to address this by building an agent that is able to play backgammon at a high level. We will employ deep reinforcement learning (TD(λ)) to evolve our model using self-play. Additionally we hope to explore how different representations of the game state affect the learning and performance of the model. We plan to use several methods of evaluation including win rate against simple strategies, accuracy of win predictions vs. established evaluation engines (such as GNUBG[^1]), and potentially even a match against the TA. 

[^1]: https://reayd-falmouth.itch.io/gnubg-nn-pypi

## Motivation and Questions:

There are many types of problems that can not be solved easily by stock implementations of the algorithms learned in this class: mainly those using a large dataset of labeled feature sets. In these cases, the methods we have learned are not totally useless, but instead can be adapted slightly to allow for new approaches such as reinforcement learning. Evaluating Backgammon positions is one such problem where generating 'true' evaluations for any given position is impossible. While Backgammon is a bit a of toy problem in that it has a good deal of previous academic work associated with it we felt this would be a good foundation for both applying the tools of this class to a slightly different context, but also for the additional exploration we wish to conduct. 

For example in one of the first papers on the subject the author, Tesauro, makes no reference to the actual features except to say "the input representation only encoded the raw board information (the number of White or Black checkers at each location), and did not utilize any additional pre-computed features relevant to good play."[^2] In their textbook on reinforcement learning Barto and Sutton go on to describe the exact feature set Tesauro used and note, "basically, Tesauro tried to represent the position in a straightforward way, while keeping the number of units relatively small. He provided one unit for each conceptually distinct possibility that seemed likely to be relevant, and he scaled them to roughly the same range, in this case between 0 and 1."[^3] However they offer not justification as to why these choices are optimal. 

Part of this project is an attempt to answer if this is the optimal representation, and understand what about it is most important. Finally, we wish to see if this sort of simulation based learning (simulated self-play) is practical for individuals or whether to compute resources make it difficult to fully train optimized models (though early tests indicate that some basic learning is certainly feasible or we would not have attempted this project).

[^2]: https://www.bkgm.com/articles/tesauro/tdl.html
[^3]: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

## Planned Deliverables:

TBD:

## Resources Required: 

TBD:

## Learning Objectives:

Camiel has previously dabbled in reinforcement learning, but only sparsely. He would love to learn more, especially as the course focusses mostly on supervised methods. To be specific, he's hoping to understand how temporal difference learning looks like in practice, how an agent can update predictions mid-game, bootstrapping toward better estimates over time. Given the complexity of our task, he hopes to learn a fair amount about feature engineering choices and their interactions with model capacity: can a larger network learn the right abstractions at this scale, and what constraints will that put on our runtime? 

Wills is interested in the engineering side of building a self-play training loop from scratch. Ofcourse, succesful backgammon models already exist, but in this fashion he gets practice building an entire environment (legal move generation, dice handling, doubling cube if we manage to get there). He enjoys the prospect of figuring out how to evaluate progress when there's no fixed test set. He is also interested in building with PyTorch beyond the assignment-scale models we've built so far. Wills also wants to refresh his underestanding of a work flow in a collaborative Git codebase. 

Shingo would like to learn how to design experiments around a learning agent. He is curious about what appropriate baselines might look like (are we using third party models, and if so, how much of their perfomance do we realistically hope to replicate?). He would like to clarify what constitutes meaningful win rate improvement, which requires carefuly tracking metrics during our thousands of training runs. He expects to also deepen his understanding of neural network training dynamics, as he expects we will run into plenty of issues that don't always show up in supervised settings.


## Risk Statement:

We estimate that the biggest risk lies in the possiblity that self-play training simply doesn't converge in sufficient time. TD-Gammon famously was able to do so, but crucially trained for hundreds of thousands of games. We might run into trouble early on establishing a reward signal, our network architecture, or an appropriate exploration. Errors in any one of the areas might result in our agent plateauing at random-play strength. That is pretty bad, as we won't be able to recover easily why that happened. We discussed our contingencies in this case; we would lean on supervised learning from expert models, which transforms the problem into classification setups we already know.

The second risk is likely the scope of the environment itself. Backgammon is a wonderful game with many subtleties (bearing off, hitting blots, optionally the doubling cube), which might slow us down, or corrupt straight-forward training. Our feature representation might fail to to capture enough board structure for the agent to learn non-trivial strategy. To adress this, we hope to utilize the minimal viable product strategy mentioned in the proposal doc; get something functioning off the ground, then slowly introduce more compexities over time. 

## Ethics Statement:

TBD: 

## Tentative Timeline:

- Week 2: Working implementation of TD-learning algorithm.
- Week 4: Results of several training runs with varied board representations & potentially optical input.
