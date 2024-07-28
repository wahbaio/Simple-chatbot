import numpy as np  # Importing numpy for numerical operations and array manipulations
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TfidfVectorizer to convert text data to numerical data
from sklearn.decomposition import PCA  # Importing PCA for dimensionality reduction of numerical data
from sklearn.cluster import KMeans  # Importing KMeans for clustering the data into different groups

class SimpleChatbot:
    def __init__(self):
        # List to store the history of user inputs for later analysis
        self.conversation_history = []
        
        # Dictionary to act as the knowledge base for storing learned responses
        self.knowledge_base = {}

        # Initialize the TF-IDF Vectorizer to convert text into numerical data
        self.vectorizer = TfidfVectorizer()

        # Call the method to welcome the user
        self.greet_user()

    def greet_user(self):
        # Method to welcome the user when the chatbot starts
        print("Hi, how can I help?")

    def process_input(self, user_input):
        # Method to process the input received from the user
        # Add the user's input to the conversation history
        self.conversation_history.append(user_input)

        # Try to find a response in the knowledge base; use a default message if not found
        response = self.knowledge_base.get(user_input, "I don't know the answer, please give me the answer and I will learn from it.")
        
        # Call the learning method to analyze stored conversations
        self.learn_from_conversation()

        # Return the answer
        return response

    def learn_from_conversation(self):
        # Method to learn from stored conversations 
        # Ensure there is enough data to perform learning (at least 2 inputs)
        if len(self.conversation_history) > 1:
            # Convert stored conversations to numerical data using TF-IDF Vectorizer
            X = self.vectorizer.fit_transform(self.conversation_history)

            # Reduce the dimensionality of the data using PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X.toarray())

            # Apply K-Means clustering to the previous step to find patterns
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(X_reduced)

            # Update the knowledge base based on the identified clusters
            clusters = kmeans.predict(X_reduced)
            for i, cluster in enumerate(clusters):
                # Store a response indicating the cluster classification of each input
                self.knowledge_base[self.conversation_history[i]] = f"I have classified your input into cluster {cluster}."

    def chat(self):
        # Method to start the conversation
        while True:
            # Receive input
            user_input = input("You: ")
            
            # Check if the user wants to exit
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            # Process the user's input and generate an appropriate answer
            response = self.process_input(user_input)
            
            # Display the chatbot's answer
            print(f"Bot: {response}")

# Instance to start the conversation
chatbot = SimpleChatbot()
chatbot.chat()
