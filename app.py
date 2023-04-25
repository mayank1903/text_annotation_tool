from flask import Flask, render_template, request, jsonify
import socket
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import json
import re

import openai
from constants import (
    api,
    model_engine,
    temperature,
    max_tokens,
    specific_word,
    prompt,
    Buttons,
)


# def get_gpt_output(string, api, model_engine, temperature, max_tokens, prompt):
#     """
#     This function returns the output of the chatgpt for a given sentence of i-grain

#     Input: String of message
#     Output: response in a predefined template
#     """
#     # Replace YOUR_API_KEY with your OpenAI API key
#     openai.api_key = api

#     # Replace the text with the input you want to process
#     input_text = string

#     # Replace the model and parameters with the ones you want to use
#     # model_engine = "text-davinci-003"
#     prompt = prompt
#     temperature = temperature
#     max_tokens = max_tokens
#     model_engine = model_engine

#     # Call the ChatGPT model and get the response
#     response = openai.Completion.create(
#         engine=model_engine,
#         prompt=input_text + "/n/n" + prompt,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )

#     # Parse the response to get the extracted information
#     output_text = response.choices[0].text
#     # output_dict = json.loads(output_text)

#     # Print the output in JSON format
#     return output_text


def remove_words_after_specific_word(string, specific_word):
    """
    This function removes string after a specific word of words separated by | from a string.

    If all the words separated by | are present, remove strings after the first word that is occuring
    If any of the word is present, then string after that word is removed
    """
    if "|" in specific_word:
        index = [string.find(word) for word in specific_word.split("|")]
        if -1 in index:
            set_index = set(index)
            if len(set_index.difference({-1})) == 0:
                index = -1
            else:
                index.remove(-1)
                index = min(index)
        else:
            index = min(index)
    else:
        index = string.find(specific_word)

    if index != -1:
        return string[:index]
    else:
        return string


class BaseEditor:
    def __init__(self, data, data_directory=""):
        """
        Initialise the class
        """

        # if old_data == None:
        self.data = data
        # else:
        #   self.data = data[~data['raw_message'].isin(old_data['raw_message'])].reset_index(drop=True)
        self.data["raw_message"] = self.data["raw_message"].apply(
            lambda x: remove_words_after_specific_word(x, specific_word=specific_word)
        )
        self.data["output_response"] = np.nan
        self.data["verified"] = np.nan

        # return len(self.data)

        # self.labelled = self.data[self.data['verified'].notna()]
        # self.unblabelled = self.data[self.data['verified'].isna()]

    def get_unlabelled_indices(self):
        """
        Get the list of all the indices of the unlabelled dataset
        """
        return [index for index in self.data[self.data["verified"].isna()].index.values]

    def get_random_example(self):
        """
        Get the indices of an unlabelled record
        and output the same
        """
        unlabelled_set = self.get_unlabelled_indices()
        current_example_index = random.choice(unlabelled_set)
        current_example = self.data.iloc[current_example_index]
        return {
            "id": current_example_index,
            "raw_message": current_example["raw_message"],
            # "Output": get_gpt_output(
            #     current_example["raw_message"],
            #     api,
            #     model_engine,
            #     temperature,
            #     max_tokens,
            #     prompt,
            # ),
        }

    def save_unlabelled_example(self, example_id, text):
        """
        Change the status of the new example as verified
        and save the labelled data to the directory
        """
        example_id = int(example_id)
        self.data["output_response"][example_id] = text
        self.data["verified"][example_id] = "Yes"

    def save_data(self):
        """
        Saves the labelled data in the csv format
        """

        self.data[self.data.verified.notna()].to_csv("parsing_data_v1.csv", index=False)


def free_port():
    """
    Gives a number of a port that is free
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr = s.getsockname()
    s.close()
    return addr[1]


def get_app(ntagger):
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/editor")
    def editor():
        return render_template("editor-3.html")

    @app.route("/load_example")
    def load_one_example():
        # return ntagger.get_random_example()['raw_message']
        example = ntagger.get_random_example()
        return jsonify(
            {
                "output1": example["raw_message"],
                # "output2": example["Output"],
                "output3": str(example["id"]),
            }
        )

    @app.route("/save_example", methods=["POST"])
    def save_example():
        data = request.get_json()
        textbox1_value = data["id"]
        textbox2_value = data["text"]
        print(textbox1_value, textbox2_value)
        ntagger.save_unlabelled_example(example_id=textbox1_value, text=textbox2_value)
        # with open('saved_text.txt', 'w') as f:
        #     f.write(textbox1_value)
        print(data)
        return {"status": "success"}

    @app.route("/save_row", methods=["POST"])
    def save_row():
        data = request.get_json()
        # input_value = data.get('input_value')
        # output_value = input_value.upper()
        # response_data = {
        #     "output_value": output_value
        # }

        # Return the JSON object as the response
        return jsonify(data)

    @app.route("/save_data")
    def save_dataframe():
        ntagger.save_data()
        return "Data saved successfully"

    return app


# def start_server(port=None, debug=True):
#   get_app().run(port=free_port())


class Editor:
    def __init__(self, dataset, data_directory=""):
        self.ntagger = BaseEditor(dataset, data_directory=data_directory)
        self.app = get_app(self.ntagger)

    def get_random_example(self, example_id, text):
        """
        Fetches a random example from the data
        """
        self.ntagger.get_random_example(example_id, text)

    def save_data(self):
        """
        Fetches a random example from the data
        """
        self.ntagger.save_data()

    def save_unlabelled_example(self):
        """
        Fetches a random example from the data
        """
        self.ntagger.save_unlabelled_example()

    def start_server(self, port=None):
        """
        Start the editor server
        :param port: Port number to bind the server to.
        :return:
        """
        self.app.run(port=9000)
