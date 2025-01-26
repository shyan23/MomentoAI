from transformers import pipeline
import time
start_time = time.time()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ Here's an example of a mixed review:

I recently attended an event at Riverside Conference Hall, and while there were some great aspects, there were also areas for improvement. The food was decent, with a variety of options, but some dishes lacked flavor, especially the vegetarian choices. The accommodation was comfortable, but the room I stayed in had some maintenance issues, like a leaking faucet. Service was friendly, but there were delays in response times, particularly during the lunch break. The event app was functional but had a few bugs that made navigation difficult. The location was convenient, though parking was scarce. A good experience with room for improvement.
"""
print(summarizer(ARTICLE, max_length=100, min_length=30, do_sample=False))


end_time = time.time()

# Calculate the time taken
print(end_time-start_time)