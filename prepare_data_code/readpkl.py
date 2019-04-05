import pickle


with open("C:/Users/soma05/Downloads/VITON2(tshirts)/data/pose.pkl", "rb") as file:
    data_list = []
    while True:
        try:
            data = pickle.load(file, encoding = "latin-1")
        except EOFError:
            break
        data_list.append(data)
    print(data_list)

