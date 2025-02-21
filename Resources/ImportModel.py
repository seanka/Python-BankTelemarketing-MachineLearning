import pickle


def import_model():
    with open("final_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model
