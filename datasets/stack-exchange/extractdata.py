import pandas as pd

import xml.etree.ElementTree as ET
tree = ET.parse("stackExchange-FAQ.xml")
root = tree.getroot()

user_questions = list()
intents = list()
intent_id = list()
intent_index = 0
for child in root:
    rephrased_questions = list()
    original_question = str()
    for item in child:
        text = item.text.strip().replace("*", '')
        if text:
            if item.tag == "question":
                original_question = text
                intent_index += 1
            else:
                rephrased_questions.append(text)
    original_question_copied = list()
    for i in range(len(rephrased_questions)):
        original_question_copied.append(original_question)
        intent_id.append(str(intent_index))
    user_questions.extend(rephrased_questions)
    intents.extend(original_question_copied)
    print(len(user_questions), len(intents), len(intent_id))

data = zip(user_questions, intents, intent_id)

df = pd.DataFrame(data=data, columns=['text', 'intent', 'intentId'])
df.to_csv('date_update.csv', index=False)

