README

File Descriptions:
1. baseline.py          -- implements the baseline linear regression algorithm
2. elman_rnn.py         -- implements the Elman recurrent neural net
3. feed_forward_nn.py   -- implements the feed-forward neural net
4. input_util.py        -- functions useful for parsing input to prediction algorithms
5. net_prediction.py    -- functions useful for predicting certain points using a prediction algorithm (useful for creating graphs)
6. net_util.py          -- main implementation fo stochastic gradient descent
7. nn_globals.py        -- neural net globals (e.g., number of pollutants, gradients of various activation functions etc.)
8. oracle.py            -- implements the oracle algorithm
9. pollution_hour.py    -- defines the PollutionHour class which structures inputs
10. test_util.py        -- functions for testing / evaluating error 

Instructions
------------
1. Install all necessary libraries: NumPy v1.8.0rc1, SciPy 0.13.0b1
2. To run the baseline algorithm with default flags: python lib/baseline.py <NAME OF DATA DIRECTORY>
3. To run the oracle algorithm with default flags: python lib/oracle.py <NAME OF DATA DIRECTORY>
4. To run the neural nets: invoke the appropriate file (e.g., elman_rnn.py) and pass in first the training dataset directory and then the test dataset directory; use the flag --help for information about adjusting hyperparameters. One example of a command we used for hyperparameter determination is:

python lib/feed_forward_nn.py data/slidingWindowTrain/ data/slidingWindowTest/ -a softplus -s 0.002 -p 3 -d 200

NOTE: When passing in data, the neural nets are expecting a directory containing a number of CSV files. 

