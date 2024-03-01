import speech_recognition as sr
from threading import Thread

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pyaudio
import numpy as np



class RecordTranscribe:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.threshold_shouting = 1  # Adjust this threshold as needed
        self.threshold_crying = 2000  # Adjust this threshold as needed

        self.best_transcription_english = None
        self.best_transcription_german = None

    def speech_recognition(self):

        with sr.Microphone() as source:
            print("Listening...")

            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)

            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=None)
                    recognition_results_english = self.recognizer.recognize_google(audio, language="en-US", show_all=True)
                    recognition_results_german = self.recognizer.recognize_google(audio, language="de-DE", show_all=True)

                    # Extract and print the most likely English transcription along with its confidence score
                    if 'alternative' in recognition_results_english or recognition_results_german:
                        self.best_transcription_english = recognition_results_english['alternative'][0]['transcript']
                        confidence_score_english = recognition_results_english['alternative'][0]['confidence']
                        print("You said:", self.best_transcription_english)
                        print("Confidence score of the first item:", confidence_score_english)

                        self.best_transcription_german = recognition_results_german['alternative'][0]['transcript']
                        confidence_score_german = recognition_results_german['alternative'][0]['confidence']
                        print("Sie haben gesagt:", self.best_transcription_german)
                        print("Konfidenzwert des ersten Items:", confidence_score_german)
                    else:
                        print("No transcription results found.")

                except sr.UnknownValueError:
                    print("Sorry, could not understand audio.")
                except sr.RequestError as e:
                    print("Error fetching results:", e)


    def detect_sound(self, data):
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = audio_data / 32768.0  # Normalize audio data to range [-1, 1]
            rms = np.sqrt(np.mean(np.square(audio_data)))
            #print(rms)
            if rms > self.threshold_shouting:
                print("Detected shouting sound!")
            elif rms > self.threshold_crying:
                print("Detected crying sound!")
        except Exception as e:
            print("Error processing audio data:", e)
    def analyze_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk_size)

        print("Listening for shouting or crying sounds...")

        while True:
            try:
                data = stream.read(self.chunk_size)
                self.detect_sound(data)
            except KeyboardInterrupt:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()



    def emergency_message_english(self):

        # Sample labeled dataset
        labeled_data = [
            ("Emergency: Fire reported at 123 Main St. Immediate assistance required!", 1),
            ("Urgent: Please help! I've been injured and need medical assistance.", 1),
            ("Reminder: Don't forget to submit your weekly report by Friday.", 0),
            ("Hi, how are you?", 0),
            ("Important: Due to inclement weather, all flights have been canceled.", 0),
            ("Emergency: Flooding reported in the area. Evacuation orders issued.", 1),
            ("Urgent: Child missing in the park. Please assist in the search.", 1),
            ("Emergency: Earthquake detected. Take cover and stay indoors.", 1),
            ("Reminder: Parent-teacher conferences tomorrow. Don't forget to sign up.", 0),
            ("Urgent: Power outage in the neighborhood. Crews working to restore.", 1),
            ("Emergency: Chemical spill reported on highway. Avoid the area.", 1),
            ("Reminder: Rent payment due by the end of the month.", 0),
        ]

        # Simulate larger labeled dataset
        labeled_data_extended = labeled_data * 500

        # Split data into features (messages) and labels (0 for normal, 1 for emergency)
        X = [data[0] for data in labeled_data_extended]
        y = [data[1] for data in labeled_data_extended]

        # Vectorize messages using Bag-of-Words representation
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.4, random_state=42)

        # Train logistic regression classifier
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        # Evaluate classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Classifier Accuracy:", accuracy)


        # Function to predict whether a message is an emergency message
        def predict_emergency_message(message):
            message_vectorized = vectorizer.transform([message])
            # Get probability estimates for all classes
            proba = classifier.predict_proba(message_vectorized)[0]
            # Extract confidence score for the positive class (emergency message)
            confidence_score = proba[1]
            # Predict the label
            prediction = classifier.predict(message_vectorized)
            return prediction[0] == 1, confidence_score

        while True:
            if self.best_transcription_english is not None:
            # Example messages
                messages = [
                    self.best_transcription_english
                ]

                for message in messages:
                    is_emergency, confidence_score = predict_emergency_message(message)
                    if is_emergency:
                        print("Emergency Message:", message)
                        print("Confidence Score:", confidence_score)
                        self.best_transcription_english = None
                    else:
                        print("Normal Message:", message)
                        self.best_transcription_english = None
    def emergency_message_german(self):

            # Sample labeled dataset in German
        labeled_data_german = [
            ("Notruf: Feuer gemeldet in der Hauptstraße 123. Sofortige Hilfe erforderlich!", 1),
            ("Dringend: Bitte helfen Sie! Ich bin verletzt und brauche medizinische Hilfe.", 1),
            ("Erinnerung: Vergessen Sie nicht, Ihren wöchentlichen Bericht bis Freitag einzureichen.", 0),
            ("Hallo, wie geht es dir?", 0),
            ("Wichtig: Aufgrund schlechten Wetters wurden alle Flüge abgesagt.", 0),
            ("Notruf: Überschwemmungen in der Gegend gemeldet. Evakuierungsbefehle wurden erteilt.", 1),
            ("Dringend: Kind im Park vermisst. Bitte helfen Sie bei der Suche.", 1),
            ("Notruf: Erdbeben erkannt. Suchen Sie Schutz und bleiben Sie drinnen.", 1),
            ("Erinnerung: Eltern-Lehrer-Konferenzen morgen. Vergessen Sie nicht, sich anzumelden.", 0),
            ("Dringend: Stromausfall in der Nachbarschaft. Einsatzkräfte arbeiten an der Wiederherstellung.", 1),
            ("Notruf: Chemieunfall auf der Autobahn gemeldet. Meiden Sie den Bereich.", 1),
            ("Erinnerung: Mietzahlung bis Monatsende fällig.", 0),
        ]

        # Simulate larger labeled dataset
        labeled_data_extended_german = labeled_data_german * 500

        # Split data into features (messages) and labels (0 for normal, 1 for emergency)
        X_german = [data[0] for data in labeled_data_extended_german]
        y_german = [data[1] for data in labeled_data_extended_german]

        # Vectorize messages using Bag-of-Words representation for German
        vectorizer_german = CountVectorizer()
        X_vectorized_german = vectorizer_german.fit_transform(X_german)

        # Split data into training and testing sets
        X_train_german, X_test_german, y_train_german, y_test_german = train_test_split(X_vectorized_german,
                                                                                        y_german, test_size=0.4,
                                                                                        random_state=42)

        # Train logistic regression classifier for German
        classifier_german = LogisticRegression()
        classifier_german.fit(X_train_german, y_train_german)

        # Evaluate classifier for German
        y_pred_german = classifier_german.predict(X_test_german)
        accuracy_german = accuracy_score(y_test_german, y_pred_german)
        print("Classifier Accuracy (German):", accuracy_german)

        # Function to predict whether a message is an emergency message in German
        def predict_emergency_message_german(message):
            message_vectorized_german = vectorizer_german.transform([message])
            # Get probability estimates for all classes
            proba_german = classifier_german.predict_proba(message_vectorized_german)[0]
            # Extract confidence score for the positive class (emergency message)
            confidence_score_german = proba_german[1]
            # Predict the label
            prediction_german = classifier_german.predict(message_vectorized_german)
            return prediction_german[0] == 1, confidence_score_german

        while True:
            if self.best_transcription_german is not None:

                # Example messages in German
                messages_german = [
                    self.best_transcription_german
                ]

                for message_german in messages_german:
                    is_emergency_german, confidence_score_german = predict_emergency_message_german(message_german)
                    if is_emergency_german:
                        print("Emergency Message (German):", message_german)
                        print("Confidence Score (German):", confidence_score_german)
                        self.best_transcription_german = None
                    else:
                        print("Normal Message (German):", message_german)
                        self.best_transcription_german = None


if __name__ == '__main__':
    print("Say something...")

    rec_trans = RecordTranscribe()
    ########################

    transcribe_thread = Thread(target=rec_trans.speech_recognition)
    emotional_response_thread = Thread(target=rec_trans.analyze_audio)
    emergency_message_english_thread = Thread(target=rec_trans.emergency_message_english)
    emergency_message_german_thread = Thread(target=rec_trans.emergency_message_german)
    ########################

    transcribe_thread.start()
    emotional_response_thread.start()
    emergency_message_english_thread.start()
    emergency_message_german_thread.start()
    ########################

    transcribe_thread.join()
    emotional_response_thread.join()
    emergency_message_english_thread.join()
    emergency_message_german_thread.join()

    ##############################