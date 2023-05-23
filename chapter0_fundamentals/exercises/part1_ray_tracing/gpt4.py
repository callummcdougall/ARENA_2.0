#%%
import os
import openai
openai.api_key = "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"
openai.Model.list()

#%%

openai.ChatCompletion.create(
    model='gpt-4',
    messages=[{'role': 'system', 'content': 'This is my first time using GPT-4!'}]
)
#%%