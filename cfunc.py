import pandas as pd

# Declare a function to compute common learning metrics
def per_category_metric(data, category):

    if not isinstance(category, str):
        print("Hey! Category needs to be a string!")

    data = data[(data['chatgpt_class'] == category) | (data['category'] == category)]

    accuracy = round(sum(data['correct']) / len(data['correct']) * 100, 2)
    precision = round(sum(data['correct']) / len(data['chatgpt_class']) * 100, 2)
    recall = round(sum(data['correct']) / len(data[data['category'] == category]) * 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)

    #print('accuracy:' + str(accuracy) + '\n' +
    #    'precision: ' + str(precision) + '\n' +
    #    'recall: ' + str(recall) + '\n' +
    #    'f1: ' + str(f1))

    odf = pd.DataFrame({
        'category': [category, category, category, category],
        'metric' : ['accuracy', 'precision', 'recall', 'f1'],
        'value' : [accuracy, precision, recall, f1]
    })

    return odf