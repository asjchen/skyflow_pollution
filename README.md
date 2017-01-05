Skyflow: Pollution Prediction
-----------------------------

By Andy Chen and Nick Hirning

In this project (developed for Stanford University CS 221), we predict pollution 
levels in a particular location based on historical hourly pollutant and weather 
data in that location. Our general approach involves two types of three-layer 
neural networks, which we train on a dataset from northern Taiwan. 

To evaluate the performance of our neural networks, we also construct a 
baseline, a simple algorithm to place a lower bound on our performance, and an
oracle, an algorithm that incorporates information normally hidden to us (ex.
measurements from the far future) and places an upper bound on our performance.

See the paper in the docs/ folder for more information about our approach and
the algorithmic details implemented here.


File Descriptions
-----------------

1. baseline.py         -- implements the baseline linear regression algorithm
2. data_util.py        -- functions for processing CSV files in the directories 
3. driver.py           -- top-level driver for executing the algorithms
4. elman_rnn.py        -- functions specific to the Elman recurrent neural net
5. feed_forward_nn.py  -- functions specific to the feed-forward neural net
6. input_util.py       -- functions useful for parsing input 
7. net_prediction.py   -- functions useful for predicting certain points using 
                          a prediction algorithm (useful for creating graphs)
8. net_util.py         -- implements the neural networks
9. nn_globals.py       -- neural net globals (e.g., number of pollutants, 
                          gradients of various activation functions etc.)
10. oracle.py          -- implements the oracle algorithm
11. pollution_hour.py  -- defines the PollutionHour class to structure inputs
12. test_util.py       -- functions for testing / evaluating error 


Instructions
------------

1. Install all necessary libraries: NumPy v1.8.0rc1, SciPy 0.13.0b1
2. Run an algorithm with:

python lib/driver.py {baseline, oracle, feed-forward, elman} <arguments>

For more details about the command line input, run with the --help flag.

NOTE: The algorithms expect (as input) directories containing a number of CSV files. 

