# -*- coding: utf-8 -*-
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
 
chatbot = ChatBot("myBot")
chatbot.set_trainer(ChatterBotCorpusTrainer)
 
# 使用中文语料库训练它
chatbot.train("chatterbot.corpus.chinese")
lineCounter = 1
# 开始对话
while True:
    print(chatbot.get_response(input("(" + str(lineCounter) + ") user:")))
    lineCounter += 1
