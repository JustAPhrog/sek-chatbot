import random
from chatbot import Chatbot
from src.utils import import_data


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    result = None
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def run_chat_with_bot(chatbot, data):
    print('-----Now you can start to chat-----')
    while True:
        message = input("> ")
        intents = chatbot.predict_class(message)
        result = get_response(intents, data)
        print(result)


def main():
    data = import_data('..\\resources\\example_data.json')

    chatbot = Chatbot()
    chatbot.prepare_data(data)
    chatbot.prepare_model(epochs=200)
    run_chat_with_bot(chatbot, data)


if __name__ == '__main__':
    main()
