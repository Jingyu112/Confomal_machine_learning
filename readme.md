# Conformal Machine Learning 

This is an implementation of conformal machine learning methods, specifically Conformal Prediction and Venn-Abers Prediction on classification problems.

For your information, Conformal Prediction leverages past prediction errors to construct confidence intervals around predictions, ensuring that a specific confidence level is guaranteed for new data. This model-agnostic approach makes minimal distributional assumptions and offers a strong guarantee that predictions fall within the desired confidence range, providing reliable uncertainty estimates.

Venn-Abers Prediction builds on the framework to refine probability calibration further, using non-parametric techniques to assign a probability distribution over potential labels. Venn-Abers predictor achieves a high level probabilistic calibration, directly quantifying the confidence in each prediction. It is particularly useful in cases where well-calibrated probabilities are crucial, such as risk assessment and decision-making applications.

These two approaches provide robust tools for uncertainty quantification, enchanting the interpretability and reliability of machine learning models. Both implementations enable binary and multi class classification problems and options for cross-validation. Particularly, codes in Conformal Prediction also implement an algorithm of "KProx", which serves as a particular non-conformity function for RandomForest models. 



Here are important references :

[1] Bhattacharyya, S. (2013b). Confidence in predictions from random tree ensembles. Knowledge and Information Systems, 35(2), 391-410.https://doi.org/10.1007/s10115-012-0600-z

[2] Shi, F., Ong, C. S., & Leckie, C. (2013). Applications of Class-Conditional Conformal Predictor in Multi-class Classification (Vol. 13, pp. 235-239).https://doi.org/10.1109/icmla.2013.48

[3] Johansson U , Lfstrm T , Bostrm H .Well-Calibrated and Sharp Interpretable Multi-Class Models[J].Springer, Cham, 2021.DOI:10.1007/978-3-030-85529-1_16.