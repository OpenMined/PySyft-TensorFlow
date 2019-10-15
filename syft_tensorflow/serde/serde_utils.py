

def history_numpy_to_float(history):
    """
    This function takes history attribute from tf.keras.callbacks.History
    and convert numpy values to floats so the history dictionary can be 
    serialized

    Args:
      history (dict): history attribute from tf.keras.callbacks.History

    Returns:
      dict: history with values as float
    """

    for k in history.keys():
        v = history[k]
        history[k] = [float(e) for e in v]

    return history
