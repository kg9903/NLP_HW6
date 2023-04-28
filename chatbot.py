import random
import numpy as np
import argparse
import joblib
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

# for time-based greetings
import datetime


import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = 'IMDBot'

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')

        # Load sentiment words
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # TODO: put any other class variables you need here

        # the original input from the user (for sentiment analysis)
        self.original_input = ""

        # the list of candidates for disambiguating the movie title
        self.candidates = []

        # ID -> sentiment (+1, 0, or -1)
        self.sentiments = {}
        self.sentiment_verbs = {
            1: ['liked', 'loved', 'enjoyed', 'appreciated', 'took pleasure in'],
            0: ['do not have an opinion on', 'do not feel strongly about'],
            -1: ['did not like', 'did not enjoy', 'did not appreciate', 'did not fancy'],
        }

        self.recommended_movies = []
        self.recommended_movies_index = 0

        self.asked_to_recommend = True

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        intro_message = """Hi, I'm IMDBot!

I can recommend you a movie. 😊
Tell me about a movie that you have seen, and please put the name of the movie in quotation marks.

To exit: write ":quit" (or press Ctrl-C to force the exit)
"""
        return intro_message
        """
        Your task is to implement the chatbot as detailed in the HW6
        instructions (README.md).

        To exit: write ":quit" (or press Ctrl-C to force the exit)

        TODO: Write the description for your own chatbot here in the `intro()` function.
        """

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################
        #greeting_message = "How can I help you?"
        current_hour = datetime.datetime.now().hour

        if current_hour >= 5 and current_hour < 12:
            greeting_message = "Good morning!"
        elif current_hour >= 12 and current_hour < 18:
            greeting_message = "Good afternoon!"
        elif current_hour >= 18 and current_hour < 23:
            greeting_message = "Good evening!"
        else:
            greeting_message = "It's late, you should go to sleep, but I'm still happy to help!"

        greeting_message += """ I'm a Chatbot that can recommend you a movie based on your inputs. Please start by telling me about a movie that you have seen and whether you like it or lot, and please put the name of the movie in quotation marks.

Example: I _____(liked/disliked/...) "Movie Title"
"""

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Thanks for chatting with me. Have a nice day! 👾"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function.
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        Arguments:
            - line (str): a user-supplied line of text

        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if len(self.sentiments) < 5:
            # gather sentiment data
            titles = self.extract_titles(line)
            if (len(titles) == 0):
                response = "Tell me about a movie that you have seen, with the name of the movie **in quotation marks**."
                return response

            if (len(self.candidates) == 0):
                # new conversation, treat titles as new movie titles
                self.original_input = line

                # using list comprehension (python doesn't have a flatMap!!) for all indices of all mentioned movies
                self.candidates = [idx for title in titles for idx in self.find_movies_idx_by_title(title)]
            else:
                # this is a continuation of a previous conversation, using titles to disambiguate
                # note that we make the assumption that only one movie should be processed at a time
                # TODO: check if this fits with the sample input
                self.candidates = self.disambiguate_candidates(" ".join(titles), self.candidates)

            # handling the candidate indices
            if (len(self.candidates) > 1):
                movie_list = '\n\t'.join([self.titles[idx][0] for idx in self.candidates])
                response = "I found more than one movie with that name, which one did you mean? \n\t{}".format(movie_list)

            elif (len(self.candidates) == 0):
                title_str = "\" \"".join(titles)
                response = f"Sorry, I couldn't find any movie with the title \"{title_str}\"!"

            else:
                # we have exactly one candidate, clear the candidates list
                idx = self.candidates[0]
                self.candidates = []
                sentiment = self.predict_sentiment_rule_based(self.original_input)

                # store the sentiment
                self.sentiments[idx] = sentiment

                response = "Got it, you {} '{}'".format(
                    # randomly choose a verb based on sentiment
                    random.choice(self.sentiment_verbs[sentiment]),
                    self.titles[idx][0]
                )

                # recommend a movie if we have enough sentiments data
                if len(self.sentiments) == 5:
                    response += "\nThanks! That's enough for me to make a recommendation 😊\n"

                    # get the recommendation
                    self.recommended_movies = self.recommend_movies(self.sentiments, 5)

                    response += "I recommend you watch: {}".format(self.recommended_movies[self.recommended_movies_index])
                    self.recommended_movies_index += 1

                    response += "\nWould you like to hear another recommendation? (Or enter :quit if you're done.)"

        else:
            # if it's not quit, we suggest a movie (maybe handle cases like "no" here?)
            if not self.asked_to_recommend or line.lower().strip() == "no":
                self.asked_to_recommend = False
                response = "Okay, have a nice day!"
            elif self.recommended_movies_index >= len(self.recommended_movies):
                response = "Sorry, I don't have any more recommendations for you."
            else:
                response = "I recommend you watch: {}".format(self.recommended_movies[self.recommended_movies_index])
                self.recommended_movies_index += 1

                response += "\nWould you like to hear another recommendation? (Or enter :quit if you're done.)"

        return response

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]

        Arguments:
            - user_input (str) : a user-supplied line of text

        Returns:
            - (list) movie titles that are potentially in the text

        Hints:
            - What regular expressions would be helpful here?
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        pattern = r"[\"]([^\"]*)[\"]"

        x = re.findall(pattern, user_input)

        return x

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def find_movies_idx_by_title(self, title:str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title

        Returns:
            - a list of indices of matching movies

        Hints:
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think
              of a more concise approach
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        cleanTitles = [re.match(r"(.*)\(", t[0]) if re.match(r"(.*)\(", t[0]) is not None else re.match(r"(.*)", t[0]) for t in self.titles ]

        matches = [cleanTitles.index(t) for t in cleanTitles if re.search(title, t[0]) is not None]

        return matches
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def disambiguate_candidates(self, clarification:str, candidates:list) -> list:
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this
        should return multiple elements in the list which the clarification could
        be referring to.

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)"
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '

        Arguments:
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns:
            - a list of indices corresponding to the movies identified by the clarification

        Hints:
            - You should use self.titles somewhere in this function
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        matches = [t for t in candidates if re.search(clarification, self.titles[t][0], re.IGNORECASE) is not None]

        return matches # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ###########################################################################

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to
        predict sentiment.

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment.
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count)
        and negative sentiment category (neg_tok_count)

        This function should return
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns:
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints:
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################


        words = re.sub(r'".*?"', '', user_input.lower())
        # print("Attention: After removing movie title:", words)

        words = re.findall(r'\b\w+\b', words)
        # print("Attention: After split:", words)

        p_tkcount = 0

        n_tkcount = 0

        for word in words:
            # print("Attention: The word '{}' is in the sentiment dictionary: {}".format(word, word in self.sentiment))

            if word in self.sentiment:
                # print("Attention: The word '{}' is in the sentiment dictionary".format(word))

                if self.sentiment[word] == 'pos':
                    # print("Attention: The word '{}' is positive".format(word))
                    p_tkcount += 1
                    # print("Attention: The current pos_tokcount is:", p_tkcount)

                elif self.sentiment[word] == 'neg':
                    # print("Attention: The word '{}' is negative".format(word))
                    n_tkcount += 1
                    # print("Attention: The current neg_tokcount is:", n_tkcount)


        if p_tkcount == 0 and n_tkcount == 0:
            # print("Attention: The sentiment of the input '{}' is neutral".format(user_input))
            return 0

        else:
            sentiment_score = (p_tkcount - n_tkcount) / (p_tkcount + n_tkcount)

            if sentiment_score == 0:
                # print("Attention: The sentiment of the input '{}' is neutral".format(user_input))
                return sentiment_score

            elif sentiment_score > 0:
                # print("Attention: The sentiment of the input '{}' is positive".format(user_input))
                return sentiment_score

            else:
                # print("Attention: The sentiment of the input '{}' is negative".format(user_input))
                return sentiment_score

        #return 0 # TODO: delete and replace this line

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset

        You'll have to transform the class labels (y) such that:
            -1 inputed into sklearn corresponds to "rotten" in the dataset
            +1 inputed into sklearn correspond to "fresh" in the dataset

        To run call on the command line:
            python3 chatbot.py --train_logreg_sentiment

        Hints:
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset.
        """
        #load training data
        texts, y = util.load_rotten_tomatoes_dataset()

        self.model = None #variable name that will eventually be the sklearn Logistic Regression classifier you train
        self.count_vectorizer = None #variable name will eventually be the CountVectorizer from sklearn

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        pass # TODO: delete and replace this line

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int:
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns: int
            -1 if the trained classifier predicts -1
            1 if the trained classifier predicts 1
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints:
            - Be sure to lower-case the user input
            - Don't forget about a case for the 0 class!
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        return 0 # TODO: delete and replace this line
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the
        recommended movie titles.

        Be sure to call util.recommend() which has implemented collaborative
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.

        This function must have at least 5 ratings to make a recommendation.

        Arguments:
            - user_ratings (dict):
                - keys are indices of movies
                  (corresponding to rows in both data/movies.txt and data/ratings.txt)
                - values are 1, 0, and -1 corresponding to positive, neutral, and
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example:
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)',
            'Problem Child (1990)']

        Hints:
            - You should be using self.ratings somewhere in this function
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing.
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        # format user_ratings to a numpy array indexed by movie id
        user_rating_all_movies = np.zeros(shape=self.ratings.shape[0])
        for key, value in user_ratings.items():
            user_rating_all_movies[key] = value

        indices = util.recommend(user_rating_all_movies, self.ratings, num_return)

        return [self.titles[idx][0] for idx in indices]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1():
        """
        TODO: delete and replace with your function.
        Be sure to put an adequate description in this docstring.
        """
        pass

    def function2():
        """
        TODO: delete and replace with your function.
        Be sure to put an adequate description in this docstring.
        """
        pass

    def function3():
        """
        Any additional functions beyond two count towards extra credit
        """
        pass


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
