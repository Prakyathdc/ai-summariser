"""
Chart Generator Module
========================
Generates Matplotlib charts for text analytics.
"""

import matplotlib.pyplot.subplots as plt_subplots
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
from wordcloud import WordCloud

class ChartGenerator:
    @staticmethod
    def generate_word_freq_chart(words, top_n=10) -> BytesIO:
        """Generate a bar chart of top words."""
        counter = Counter(words)
        most_common = counter.most_common(top_n)
        
        labels, values = zip(*most_common) if most_common else ([], [])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color='#3b82f6')
        ax.set_title("Top Word Frequencies")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf

    @staticmethod
    def generate_wordcloud(text: str) -> BytesIO:
        """Generate a word cloud."""
        wc = WordCloud(width=800, height=400, background_color="white", colormap="Blues").generate(text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        
        buf = BytesIO()
        plt.tight_layout(pad=0)
        fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf
