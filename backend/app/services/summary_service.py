from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


def generate_summary(text: str, sentence_count: int = 5):
    if not text.strip():
        return ""

    sentences = text.split(". ")

    if len(sentences) <= sentence_count:
        return text

    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(
        parser.document,
        sentence_count
    )

    return " ".join(str(sentence) for sentence in summary_sentences)