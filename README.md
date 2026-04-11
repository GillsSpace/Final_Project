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

TBD:

## Risk Statement:

TBD:

## Ethics Statement:

TBD: 

## Tentative Timeline:

- Week 2: Working implementation of TD-learning algorithm.
- Week 4: Results of several training runs with varied board representations & potentially optical input.
