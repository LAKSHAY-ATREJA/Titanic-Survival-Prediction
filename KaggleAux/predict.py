import numpy as np
from pandas import DataFrame
from patsy import dmatrices


def get_dataframe_intersection(df, comparator1, comparator2):
    """
    Return a dataframe with only the columns found in a comparative dataframe.

    Parameters
    ----------
    comparator1: DataFrame
        DataFrame to perform comparison on.
    comparator2: DataFrame
        DataFrame to compare with.

    Returns
    -------
    DataFrame:
        DataFrame with columns not found in comparator dropped.

    """
    to_drop = [c for c in comparator1 if c not in comparator2]
    return df.drop(to_drop, axis=1)


def get_dataframes_intersections(df1, comparator1, df2, comparator2):
    """
    Return DataFrames with the intersection of their column values.

    Parameters
    ----------
    comparator1: DataFrame
        DataFrame to perform comparison on.
    comparator2: DataFrame
        DataFrame to compare with.

    Returns
    -------
    Tuple:
        The resulting DataFrames with columns not found in comparator dropped.

    """
    result1 = get_dataframe_intersection(df1, comparator1, comparator2)
    result2 = get_dataframe_intersection(df2, comparator2, comparator1)
    return result1, result2


def predict(test_data, results, model_name):
    """
    Return predictions based on model results.

    Parameters
    ----------
    test_data: DataFrame
        Test data to generate predictions for.
    results: dict
        Dict mapping model names to [results_wrapper, formula] pairs.
            e.g.
            results['Logit'] = [
                <statsmodels.discrete.discrete_model.BinaryResultsWrapper>,
                'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)'
            ]
    model_name: str
        Key into the results dict identifying which model to use.

    Returns
    -------
    numpy.ndarray
        Predictions in a flat NumPy array.

    Example
    -------
    results = {'Logit': [fitted_logit_model, 'Survived ~ C(Pclass) + C(Sex) + Age']}
    predictions = predict(test_data, results, 'Logit')

    """
    model_params = DataFrame(results[model_name][0].params)
    formula = results[model_name][1]

    # Create regression-friendly test DataFrame
    yt, xt = dmatrices(formula, data=test_data, return_type='dataframe')
    xt, model_params = get_dataframes_intersections(
        xt, xt.columns, model_params, model_params.index
    )

    # Convert to NumPy arrays for performance
    model_params = np.asarray(model_params)
    yt = np.asarray(yt).ravel()
    xt = np.asarray(xt)

    # Broadcast model parameters across all rows and compute dot product
    row, col = xt.shape
    model_parameters = model_params.ravel()
    model_array = np.tile(model_parameters, (row, 1))

    predictions = np.sum(np.multiply(xt, model_array), axis=1)
    return predictions
