# %%
#import necessary packages
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# %%
#authentication
client = OpenAI(api_key='APIKEY')

# %%
#test api call

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role':'system', 'content':'you are a helpful assistant'},
        {'role':'user', 'content':'give me a recipe of a salad dressing that will knock the socks off a vegan?'}
    ]
)

print(response.choices[0].message.content)

# %%
#load in customer data
raw = pd.read_csv('bittext_cx_chat_data.csv')

# %%
#sample data and compute tokens
wdf = raw.sample(frac = .01).reset_index(drop=True)
wdf['tokens'] = round(wdf['instruction'].str.split().str.len() * 4/3 + 74, 0)
wdf

# %%
#check stats
categories = wdf['category'].unique()
print(categories)
print('avgtokens ' + str((wdf['tokens'].mean())))
print('ntokens: ' + str(sum(wdf['tokens'])))
print('cost: $' + str(sum(wdf['tokens']) * 0.001 / 1000))

# %%
#generate a prompt
prompt = 'You are now a simple classifier that takes a ' \
    'string input of a customer serivce chat and gives a single word output in all capital letters.' \
    'These are the following categories that you are allowed to output: ' \
    'CANCEL, ORDER, INVOICE, PAYMENT, ACCOUNT, REFUND, SUBSCRIPTION, DELIVERY, CONTACT. ' \
    'Please summarize the chat query and output one and exactly one word of the above list. ' \
    'Do not output anything more than one word. ' \
    'Here is the customer query: \n'

print('prompt tokens:' + str(len(prompt.split())))
msg = prompt + wdf.at[0, 'instruction']
print(msg)

# %%
#call chatgpt to get responses
vout = np.array([])
for i in range(len(wdf)):
    
    msg = wdf.at[i, 'instruction'] #gets msg

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role':'system', 'content': prompt},
            {'role':'user', 'content': msg}
        ]
    )
    
    vout = np.append(vout, response)

    #print(msg)

print(vout)

# %%
#get response message from response vector
vresp = np.array([])

for i in range(len(vout)):
    vresp = np.append(vresp, vout[i].choices[0].message.content)

print(vresp)

# %%
wdf['chatgpt_class'] = vresp
wdf['correct'] = wdf['chatgpt_class'] == wdf['category']
wdf

# %%
print(
    'accuracy: ' + str(round(len(wdf[wdf['correct'] == True]) / len(wdf) * 100, 1)) + ' %'
)

# %%
wdf.to_csv('gpt_classified.csv')


