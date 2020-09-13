def df_jaccard(data):
    return jaccard(data['text'], data['selected_text'])


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) == 0):
        return 0
    return float(len(c)) / (len(a) + len(b) - len(c))


def arr_jaccard(arr1, arr2):
    if len(arr1) == 0:
        return 1

    assert (len(arr1) == len(arr2))
    res = 0
    for str1, str2 in zip(arr1, arr2):
        res += jaccard(str1, str2)

    return res / len(arr1)


def get_start_end_words(x):
    start_char = x['text'].find(x['selected_text'])
    start_word = len(tokenize(x['text'][:start_char + 1])) - 1

    cnt_word = len(tokenize(x['selected_text']))
    return start_word, start_word + cnt_word


def get_start_end_char(x):
    start_char = x['text'].find(x['selected_text'])
    end_char = start_char + len(x['selected_text'])
    if not x['is_subarray']:
        start_char = x['text'][:start_char].rfind(' ') + 1
        if start_char < 0:
            start_char = 0

        first_space = x['text'][end_char:].find(' ')
        if first_space < 0:
            end_char = len(x['text'])
        else:
            end_char = end_char + first_space
    return start_char, end_char

### NER
def predict(model, data):
    texts = data.text.to_numpy()

    result = model.predict(texts)
    predict_selected_texts = get_selected_text(texts, result)
    return predict_selected_texts


def get_selected_text(texts, y):
    predict_selected_texts = []
    for ent, text in zip(y, texts):
        if (len(ent)):
            start = ent[0][0]
            end = ent[0][1]
        else:
            start = 0
            end = len(text)
        predict_selected_texts.append(text[start:end])
    return predict_selected_texts


def evaluate_score(texts, y_true, y_predict):
    selected_texts = get_selected_text(texts, y_true)
    predicted_selected_texts = get_selected_text(texts, y_predict)

    return arr_jaccard(selected_texts, predicted_selected_texts)