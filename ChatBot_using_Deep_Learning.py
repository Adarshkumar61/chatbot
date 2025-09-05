data = {'intents': [
    {
        'tag': 'greeting',
        'patterns': ['Hi', 'Hello', 'How are you?', 'Good day', 'Hey'],
        'responses': ['Hello!', 'Hi there!', 'Greetings!', 'How can I assist you?']
    },
    {
        'tag': 'goodbye',
        'patterns': ['Bye', 'See you later', 'Goodbye'],
        'responses': ['Goodbye!', 'See you later!', 'Have a great day!']
    },
    {
        'tag': 'thanks',
        'patterns': ['Thanks', 'Thank you', "That's helpful"],
        'responses': ['You\'re welcome!', 'No problem!', 'Glad to help!']
    },
    {
        'tag': 'name',
        'patterns': ['What is your name?', "What's your name?", 'Who are you?'],
        'responses': ['I am a chatbot created to assist you.', 'You can call me Chatbot.']
    },
    {
        'tag': 'age',
        'patterns': ['How old are you?', "What's your age?"],
        'responses': ['I am timeless.', "I don't have an age."]
    },
    {
        'tag': 'help',
        'patterns': ['Can you help me?', 'I need assistance', "Can you assist me?"],
        'responses': ['Of course! How can I help?', "I'm here to assist you."]
    },
      {"tag": "help",
    "patterns": ["Could you help me?", "give me a hand please", "Can you help?", "What can you do for me?", "I need a support", "I need a help", "support me please"],
    "responses": ["Tell me how can assist you", "Tell me your problem to assist you", "Yes Sure, How can I support you"]
    },
    {"tag": "createaccount",
    "patterns": ["I need to create a new account", "how to open a new account", "I want to create an account", "can you create an account for me", "how to open a new account"],
    "responses": ["You can just easily create a new account from our web site", "Just go to our web site and follow the guidelines to create a new account"]
    },
     {
      "tag": "time",
      "patterns": ["What time is it?", "Can you tell me the time?", "Do you know the time?", "Clock?", "Current time please"],
      "responses": ["I don’t have a watch, but your device clock can help!", "Sorry, I can’t tell exact time."]
    },
    {
      "tag": "date",
      "patterns": ["What’s the date today?", "Tell me today’s date", "What day is it?", "Current date?", "Today’s calendar date?"],
      "responses": ["I can’t track the live date, but your system can.", "Check your device calendar for today’s date."]
    },
    {
      "tag": "country",
      "patterns": ["What is India?", "Tell me about USA", "Do you know Japan?", "Where is France?", "What is Germany known for?"],
      "responses": ["India is known for its diversity.", "USA is a powerful country with many states.", "Japan is famous for technology and anime."]
    },
    {
      "tag": "sports",
      "patterns": ["Who won the world cup?", "Tell me about football", "Do you know cricket?", "What is basketball?", "Who is Messi?"],
      "responses": ["Cricket is a famous sport in Asia.", "Football is loved worldwide.", "Messi is one of the greatest footballers."]
    },
    {
      "tag": "food",
      "patterns": ["What’s your favorite food?", "Do you eat pizza?", "Tell me about burgers", "What is biryani?", "Do you know samosa?"],
      "responses": ["Pizza is loved worldwide.", "Biryani is a delicious Indian dish.", "Samosa is a popular snack in India."]
    },
    {
      "tag": "hobby",
      "patterns": ["What are your hobbies?", "Do you play games?", "Do you like reading?", "Tell me your hobby", "What do you enjoy?"],
      "responses": ["I love chatting with you!", "I enjoy helping people learn.", "My hobby is processing data."]
    },
    {
      "tag": "motivation",
      "patterns": ["Motivate me", "Say something positive", "Give me motivation", "Inspire me", "I feel low"],
      "responses": ["Believe in yourself!", "You can achieve anything!", "Keep going, success is near."]
    },
    {
      "tag": "study",
      "patterns": ["Help me study", "How to focus?", "How can I prepare for exams?", "Tips for learning", "How to avoid distractions?"],
      "responses": ["Study in short intervals.", "Avoid distractions like phones.", "Make notes and revise daily."]
    },
    {
      "tag": "technology",
      "patterns": ["What is AI?", "Explain machine learning", "What is deep learning?", "Tell me about Python", "Do you know TensorFlow?"],
      "responses": ["AI is Artificial Intelligence.", "Machine learning helps machines learn from data.", "Deep learning uses neural networks for complex problems."]
    },
    {
      "tag": "robot_teacher",
      "patterns": ["Are you my teacher?", "Can you teach me?", "Explain like a teacher", "Can you guide me?", "Are you my mentor?"],
      "responses": ["Yes, I can be your teacher!", "I’ll guide you step by step.", "Think of me as your digital mentor."]
    },
    {
      "tag": "movies",
      "patterns": ["Recommend me a movie", "Best movies to watch", "Tell me a good film", "What movie should I watch?", "Suggest a film"],
      "responses": ["You should watch Inception.", "Try Interstellar.", "Avengers is always fun.", "How about The Dark Knight?", "Parasite is really good."]
    },
    {
      "tag": "music",
      "patterns": ["Play some music", "Recommend me a song", "What should I listen to?", "Best songs?", "Tell me a good track"],
      "responses": ["How about some classical music?", "Try listening to Imagine Dragons.", "You might like Ed Sheeran.", "Listen to some relaxing lo-fi.", "Taylor Swift has some great songs."]
    },

    {"tag": "complaint",
    "patterns": ["have a complaint", "I want to raise a complaint", "there is a complaint about a service"],
    "responses": ["Please provide us your complaint in order to assist you", "Please mention your complaint, we will reach you and sorry for any inconvenience caused"]
    },
       {
      "tag": "weather_info",
      "patterns": ["What's the weather?", "Tell me today's weather", "Weather update", "Is it raining?", "How hot is it outside?"],
      "responses": ["It looks sunny today!", "Expect some clouds today.", "Might be rainy, carry an umbrella!", "It’s quite hot today.", "Cool and pleasant weather today!"]
    },
    {
      "tag": "time_info",
      "patterns": ["What time is it?", "Tell me the current time", "Do you know the time?", "Can you give me the time?", "What's the time now?"],
      "responses": ["It’s time to shine 🌞", "Here’s the current time on your system!", "You can check your device clock for exact time.", "It’s ticking fast!", "Time waits for none."]
    },
    {
      "tag": "date_info",
      "patterns": ["What's today's date?", "Tell me the date", "Which day is it?", "Give me the date please", "Do you know what day it is?"],
      "responses": ["Today is a great day!", "It’s [system date].", "Check your device calendar for today’s date.", "It’s a beautiful day.", "Another fresh day ahead!"]
    },
    {
      "tag": "motivation",
      "patterns": ["Motivate me", "Say something inspiring", "Give me motivation", "I need encouragement", "Cheer me up"],
      "responses": ["Believe in yourself!", "You’re stronger than you think.", "Keep pushing forward!", "Success starts with the first step.", "Don’t give up!"]
    },
    {
      "tag": "math_addition",
      "patterns": ["What is 2 plus 2?", "Add 5 and 6", "Sum of 10 and 20", "Can you add numbers?", "What's 100+200?"],
      "responses": ["2+2=4", "5+6=11", "10+20=30", "100+200=300", "Sure, I can add numbers for you!"]
    },
    {
      "tag": "math_subtraction",
      "patterns": ["What is 10 minus 3?", "Subtract 7 from 20", "Difference between 15 and 5", "Can you do subtraction?", "50-25"],
      "responses": ["10-3=7", "20-7=13", "15-5=10", "50-25=25", "Yes, I can subtract numbers too!"]
    },
    {
      "tag": "math_multiplication",
      "patterns": ["Multiply 3 and 4", "What is 7 times 8?", "Can you do multiplication?", "Calculate 9*9", "Product of 12 and 5"],
      "responses": ["3×4=12", "7×8=56", "9×9=81", "12×5=60", "Multiplication is easy for me!"]
    },
    {
      "tag": "math_division",
      "patterns": ["What is 20 divided by 5?", "Divide 100 by 10", "Can you do division?", "50/2", "Find quotient of 90 and 9"],
      "responses": ["20÷5=4", "100÷10=10", "50÷2=25", "90÷9=10", "Yes, I can divide numbers too!"]
    },
    {
      "tag": "currency_conversion",
      "patterns": ["Convert 10 dollars to rupees", "How much is 50 euros in INR?", "Currency conversion", "Exchange rate please", "USD to INR"],
      "responses": ["10 USD ≈ 830 INR", "50 EUR ≈ 4500 INR", "Conversion depends on current exchange rate.", "I can help with currency estimates!", "Check live forex rates for accuracy."]
    },
    {
      "tag": "philosophy",
      "patterns": ["What is the meaning of life?", "Why are we here?", "What is happiness?", "What is consciousness?", "Why do we exist?"],
      "responses": ["Philosophers have debated that for centuries. Some say the meaning of life is to seek knowledge and happiness.", "It may depend on personal beliefs, but many find purpose in relationships, creativity, and growth.", "Consciousness remains one of the biggest mysteries in science and philosophy."]
    },
    {
      "tag": "history_india",
      "patterns": ["Who was Ashoka?", "Tell me about Akbar", "Who was the first Prime Minister of India?", "What is the Mughal Empire?", "Who was Chandragupta Maurya?"],
      "responses": ["Ashoka was an Indian emperor of the Maurya Dynasty.", "Akbar was one of the greatest Mughal emperors.", "Jawaharlal Nehru was the first Prime Minister of India.", "The Mughal Empire ruled large parts of India from the 16th to 19th centuries.", "Chandragupta Maurya founded the Maurya Empire in ancient India."]
    },
    {
      "tag": "geography",
      "patterns": ["What is the capital of France?", "Which is the largest ocean?", "Where is Mount Everest?", "Which is the smallest country?", "What is the capital of Japan?"],
      "responses": ["The capital of France is Paris.", "The Pacific Ocean is the largest.", "Mount Everest is located on the border of Nepal and China.", "Vatican City is the smallest country.", "The capital of Japan is Tokyo."]
    },
    {
      "tag": "science_space",
      "patterns": ["What is a black hole?", "How far is the moon?", "What is the Milky Way?", "What is a galaxy?", "What is the speed of light?"],
      "responses": ["A black hole is a region in space where gravity is so strong that nothing can escape.", "The moon is about 384,000 km away from Earth.", "The Milky Way is the galaxy that contains our Solar System.", "A galaxy is a huge collection of stars, gas, and dust bound together by gravity.", "The speed of light is approximately 299,792 km per second."]
    },
    {
      "tag": "computer_basics",
      "patterns": ["What is a CPU?", "What is RAM?", "What is an operating system?", "What is a hard disk?", "What is the internet?"],
      "responses": ["CPU stands for Central Processing Unit, the brain of a computer.", "RAM is Random Access Memory, used for temporary storage.", "An operating system is software that manages computer hardware and software resources.", "A hard disk is storage used to save files and programs.", "The internet is a global network that connects millions of computers."]
    },
    {
      "tag": "sports_cricket",
      "patterns": ["Who is Virat Kohli?", "How many players in a cricket team?", "What is an over?", "Who won the 2011 World Cup?", "What is LBW in cricket?"],
      "responses": ["Virat Kohli is an Indian cricketer and former captain.", "A cricket team has 11 players.", "An over consists of 6 balls.", "India won the 2011 Cricket World Cup.", "LBW stands for Leg Before Wicket, a way of getting out in cricket."]
    },
    {
      "tag": "sports_football",
      "patterns": ["Who is Lionel Messi?", "What is the World Cup?", "How many players in a football team?", "Who is Cristiano Ronaldo?", "Which country invented football?"],
      "responses": ["Lionel Messi is an Argentinian football legend.", "The FIFA World Cup is the biggest international football tournament.", "A football team has 11 players.", "Cristiano Ronaldo is a Portuguese football star.", "Modern football originated in England."]
    },
    {
      "tag": "maths_advanced",
      "patterns": ["What is integration?", "What is differentiation?", "Solve 2x+5=0", "What is matrix multiplication?", "What is probability?"],
      "responses": ["Integration is the process of finding the area under a curve.", "Differentiation finds the rate of change of a function.", "The solution is x = -2.5.", "Matrix multiplication combines rows and columns of two matrices.", "Probability is the measure of how likely an event is to occur."]
    },
    {
      "tag": "health_fitness",
      "patterns": ["How to lose weight?", "What are healthy foods?", "How much water should I drink?", "What is BMI?", "How to stay fit?"],
      "responses": ["Weight loss requires exercise and a balanced diet.", "Healthy foods include fruits, vegetables, and whole grains.", "An average adult should drink about 2–3 liters of water daily.", "BMI stands for Body Mass Index, a measure of body fat.", "Staying fit requires regular exercise, sleep, and a healthy diet."]
    },
    {
      "tag": "mental_health",
      "patterns": ["What is stress?", "How to deal with anxiety?", "What is depression?", "How to stay positive?", "What is mindfulness?"],
      "responses": ["Stress is the body's reaction to any demand or challenge.", "Anxiety can be managed with breathing exercises and mindfulness.", "Depression is a mental health condition causing sadness and loss of interest.", "Staying positive requires gratitude and focusing on good things.", "Mindfulness is being aware and present in the current moment."]
    },
    {
      "tag": "calculator",
      "patterns": ["Can you calculate?", "I need a calculator", "Do math for me", "Solve 123+456", "What is 500*3?"],
      "responses": ["123+456=579", "500×3=1500", "I work like a calculator!", "Just ask me any math problem.", "Sure, I’ll calculate that for you!"]
    }
]
}

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.apppend(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.appned(intent['tag'])

num_classes = len(labels)


import json
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels) 
training_labels = lbl_encoder.transform(training_labels) # #converting the text labels into numbers


#parameters:
num_words = 1000 # vocab size
max_len = 20 # max length of each sentence
embedding_dim = 16 # how many dimensions we want to represent each word
oov_token = "<OOV>" # fill the words which are not in the vocab

tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token) 
tokenizer.fit_on_texts(training_sentences) # fit the tokenizer on our text
word_index = tokenizer.word_index 
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating = 'post', max_len = max_len)

model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length = max_len)) # input_length is the length of each input sequence
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(num_classes, activation= 'softmax'))


model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'])

model.summary()

history = model.fit(padded_sequences, np.array(training_labels), epochs = 500)



model.save('chat_bot.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol= pickle.HIGHEST_PROTOCOL)


from tensorflow import keras

def chatbot():
    model = keras.models.load_model('chat_bot.h5')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc_file:
        lbl_encoder = pickle.load(enc_file)
    
    max_len = 20 # max length of each sentence

    while True:
        print('You: ', end = '')
        inp = input()
        if inp.lower() == 'quit':
            break

        result = model.predict(tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', max_len = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]
        
        for i in data['intents']:
            if i['tag'] == tag:
                print('ChatBot: ', np.random.choice(i['responses']))

print('Start Talking with the bot( type quit to stop)') 
chatbot()
