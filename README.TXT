3rd year project, university

To use the main functionality of the solution either errorDetection.py or errorPosition.py should be run, not errorGeneration.py as this file assumes the formatting of the files 
in test_data which if changed may introduce errors.

To run those two files all that is needed is the neural_data folder present in the same hierarchy as found on the repository.

---

test data contains a treering dataset collected from the ITRDB

neural data contains example data that has been generated from the test data with simulated errors inside

crossdatingAndChronology.py is an example of statistical crossdating and chronology building

errorGeneration takes the data from test_data and uses it to generate the data found in neural_data
	i recomend that this file is not used asit will overwrite the data already in place and the data in place is sufficient to test the main file of the project.

errorDetection.py and errorPosition.py are the two files containing the neural networks and their training. These files can be run and will use the data from neural_data
to either attempt to predict if an error is present in a given sample or to predict the exact position of errors within samples if present.
