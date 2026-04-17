# Things we want to try:

- [ ] Select moves in a non-greedy manner during testing (e.g. 8% chose second best moves and 2% choose 3rd best).

- [ ] Target value is gnubg evaluation

- [ ] Model that predicts wins, gammons, and backgammon's and chose move with highest expected value. 

- [ ] Target Value not value of board after best move for roll, but average of all boards after best move for all possible rolls (but still uses the real roll). This makes sense because what we really want the value to be is the value of a board before a roll taking into account what all the different roles could be.  

- [ ] having a model that only trains on endgames for the first x epochs. 

- [ ] have a model that uses only base board info but new feature repr.

- [ ] model with new features that take into account game knowledge. 