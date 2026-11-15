from pathorob.models.uni import UNI2hModelWrapper
from pathorob.models.phikon import Phikonv2ModelWrapper
from pathorob.models.mettle import Mettle15ModelWrapper


def load_model(model_name: str, model_args: dict):
    """
    Factory method to instantiate a foundation model wrapper.

    :param model_name: (str) A name identifying the model to be instantiated.
    :param model_args: (dict) Model-specific parameters to pass to the __init__ function of the model wrapper.
    :return: (ModelWrapper) The instantiated model wrapper.
    """
    if model_name == "uni2h_clsmean":
        model_class = UNI2hModelWrapper
    elif model_name == "phikonv2_clsmean":
        model_class = Phikonv2ModelWrapper
    elif model_name == "mettle15_cls":
        model_class = Mettle15ModelWrapper
        model_args = {"feature_view": "cls", **model_args}
    elif model_name == "mettle15_clsmean":
        model_class = Mettle15ModelWrapper
        model_args = {"feature_view": "cls_mean", **model_args}
    ### Add your custom models here ###
    # elif model_name ==  "<my_model_name>":
    #   model_class = MyModelWrapper
    else:
        raise ValueError(f"Model not implemented: '{model_name}'.")
    model_wrapper = model_class(**model_args)
    return model_wrapper
