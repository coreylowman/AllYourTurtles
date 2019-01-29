# AllYourTurtles bot for Halite III

Was able to make it into the top 10 with this.

## Overview

1. Precompute a bunch of stuff
2. Compute halite/turn for every ship position pair
3. Spawn a ship if we are behind opponent in # of ships produced or it would give us positive return
4. Greedily assign ships to positions
    4a. once a ship is assigned, reduce the halite/turn of any other assignments to the same position accordingly
5. Plan dropoffs just based on halite around them, proximity to other dropoffs, and number of ships going there
6. Plan ships using WHCA* (windowed hierarchical cooperative A* - basically just A* with a time window where reservations matter)

The comments in the code have more details.

## Code

Everything is in MyBot.py :X It's commented though, and (at least I think) fairly clean.

Entrypoint: [main()](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L61)

Classes:

* [Commander](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L67) is the high level bot and uses all the rest of the classes
* [IncomeEstimation](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L224) does value calculation of assignments and ROI calculation
* [ResourceAllocation](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L356) assigns ships to goals
* [PathPlanning](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L614) plans paths
* [OpponentModel](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L929) naively computes a probability of an opponent being at a square

Notable functions:

* [value function](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L226)
* [dropoff function](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L574)
* [A*](https://github.com/coreylowman/AllYourTurtles/blob/master/MyBot.py#L807)
