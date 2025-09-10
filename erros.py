def SQE(modelo, true_data, true_labels):
    somatorio = sum([label - modelo(data) for data, label in zip(true_data, true_labels)])
    sqe = somatorio ** 2
    return sqe