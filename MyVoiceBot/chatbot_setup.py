import os
import json
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

def setup_chatbot():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    intents_file_path = os.path.join(dir_path, 'intents.json')

    with open(intents_file_path) as file:
        data = json.load(file)

    chatbot = ChatBot(
        'My Chatbot',
        read_only=True,
        logic_adapters=[
            {
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': 'I am sorry, but I do not understand.',
                'maximum_similarity_threshold': 0.65
            }
        ]
    )

    trainer = ListTrainer(chatbot)

    for intent in data['intents']:
        for pattern in intent['patterns']:
            for response in intent['responses']:
                trainer.train([pattern, response])

    return chatbot

