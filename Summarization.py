import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import asyncio
from data import synonym_map

async def summarization(reviews, nlp=None, sia=None):
    """
    Summarize reviews by extracting aspects and their sentiment.
    
    Args:
        reviews (list): List of review texts
        nlp (spacy.language.Language, optional): spaCy language model
        sia (SentimentIntensityAnalyzer, optional): VADER sentiment analyzer
    
    Returns:
        str: Formatted summary of reviews
    """
    
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    if sia is None:
        sia = SentimentIntensityAnalyzer()

    
    Synonym_map = synonym_map

    def normalize_aspect(aspect, Synonym_map):
        """Normalize aspects using synonym mapping."""
        aspect_lower = aspect.lower()
        for key, synonyms in Synonym_map.items():
            if aspect_lower in synonyms or any(syn in aspect_lower for syn in synonyms):
                return key
        return aspect_lower

    
    original_sentences = []
    for review in reviews:
        doc = nlp(review)
        original_sentences.extend([sent.text for sent in doc.sents])

    
    summary = {
        "positive": defaultdict(lambda: defaultdict(int)),
        "negative": defaultdict(lambda: defaultdict(int)),
        "mixed": defaultdict(lambda: defaultdict(int))
    }

    
    for sentence in original_sentences:
        doc = nlp(sentence)
        aspects = defaultdict(lambda: defaultdict(int))

        
        for token in doc:
            if token.pos_ == 'ADJ':
                
                subtree = ' '.join([child.text for child in token.subtree])
                sentiment_score = sia.polarity_scores(subtree)['compound']
                sentiment = "positive" if sentiment_score > 0 else "negative"

                
                aspect = None
                for child in token.head.children:
                    if child.dep_ in ('amod', 'nsubj', 'dobj', 'attr'):
                        aspect = normalize_aspect(child.text, Synonym_map)
                        break
                if not aspect and token.head.pos_ == 'NOUN':
                    aspect = normalize_aspect(token.head.text, Synonym_map)

                
                if aspect:
                    summary[sentiment][aspect][token.text.lower()] += 1

    
    all_aspects = set(summary["positive"].keys()) | set(summary["negative"].keys())
    for aspect in all_aspects:
        if aspect in summary["positive"] and aspect in summary["negative"]:
            summary["mixed"][aspect] = {
                "positive": summary["positive"][aspect],
                "negative": summary["negative"][aspect]
            }
            del summary["positive"][aspect]
            del summary["negative"][aspect]

    
    for category in ["positive", "negative", "mixed"]:
        summary[category] = {k: v for k, v in summary[category].items() if v}

    
    def format_summary(summary):
        output = []
        
        if summary["mixed"]:
            output.append("\n🔄 Overall Feedback*")
            for aspect, adjs in summary["mixed"].items():
                output.append(f"- {aspect}:")
                if "positive" in adjs:
                    pos_descriptors = ', '.join([f"{adj} ({count})" for adj, count in adjs["positive"].items()])
                    output.append(f"  ✅ Positive: {pos_descriptors}")
                if "negative" in adjs:
                    neg_descriptors = ', '.join([f"{adj} ({count})" for adj, count in adjs["negative"].items()])
                    output.append(f"  ➖ Negative: {neg_descriptors}")

        
        if summary["negative"]:
            output.append("\n🚫 *Key Issues*")
            for aspect, adjs in summary["negative"].items():
                descriptors = ', '.join([f"{adj} ({count})" for adj, count in adjs.items()])
                output.append(f"- {aspect}: {descriptors}")

        
        if summary["positive"]:
            output.append("\n✅ *Positive Feedback*")
            for aspect, adjs in summary["positive"].items():
                descriptors = ', '.join([f"{adj} ({count})" for adj, count in adjs.items()])
                output.append(f"- {aspect}: {descriptors}")

        return "\n".join(output) if output else "No significant aspects found."

    return format_summary(summary)


# if __name__ == "__main__":
#     reviews = [
#         "The food was flavorful, but the portions were much smaller than expected. The desserts were delightful, though, and the variety was impressive.",
#         "The vegetarian dishes were well-received, but the meat options could have been more seasoned",
#         "The food selection was extensive, but the quality didn't match the quantity. Some dishes were bland, while others were overcooked."
#     ]
    
#     output = summarization(reviews)
#     print(output)