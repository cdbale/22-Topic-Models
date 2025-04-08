# Script to extract and summarize topics from a text corpus.

# Author: Cameron Bale

import openai
from openai import OpenAI
import pandas as pd
import sys
import os
import time 
import numpy as np

# function to extract the themes from a corpus of text using a chat-GPT model
def generate_themes(survey_df, num_themes, num_words, model, max_tokens, temperature):
    """
    survey_df: dataframe containing survey data, reviews are stored in 'Review' column.
    num_themes: number of themes you want the model to extract/summarize.
    num_words: the number of words the model should report that are commonly associated with each theme.
    model: the OPENAI model you want to use.
    max_tokens: how many tokens can be included in the model output.
    temperature: model temperature. Lower values produce more deterministic output, higher values produce more random/noisy output.
    """

    all_reviews = " ".join(survey_df['Review'].tolist())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", 
                 "content": f"Pretend you are a helpful assistant. You will be given a corpus of reviews for a robotic vacuum. Your task is to identify {num_themes} common themes from the reviews. Please provide (1) the {num_words} words most commonly associated with each theme, and (2) A short summary of the reviews associated with each theme. \nHere are the reviews: {all_reviews}"}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        themes = response.choices[0].message.content.strip()
        return themes
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#####################################################################

if __name__ == "__main__":

    # sys.argv[1:] access command line arguments [0] is the script name
    # arguments should be passed in the following order
    # 1: name of .csv containing the survey responses
    # 2: number of themes you want to extract
    # 3: number of common words to report

    # Set your OpenAI API key
    # this is stored as environmental variable on computer
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    survey_data = pd.read_csv(sys.argv[1]).dropna().sample(frac = 0.4)
    num_themes = sys.argv[2]
    num_words = sys.argv[3]

    model = "gpt-4o-mini"
    max_tokens = 10000
    temperature = 0.7

    themes = generate_themes(survey_df = survey_data,
                             num_themes = num_themes,
                             num_words = num_words,
                             model = model,
                             max_tokens = max_tokens, 
                             temperature = temperature)

    print(themes)