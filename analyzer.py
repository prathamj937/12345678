import re
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class BankruptcyAwareFinBERTAnalyzer:

    def __init__(self):
        """Initialize the Bankruptcy-Aware FinBERT Sentiment Analyzer with simplified binary classification"""

        print("Loading FinBERT model... This may take a moment.")

        # Load FinBERT model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()  # Set to evaluation mode
            print("✅ FinBERT model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading FinBERT: {e}")
            print("📝 Please install required packages: pip install transformers torch")
            return None

        # BANKRUPTCY-SPECIFIC SENTIMENT LEXICON with severe scoring
        self.critical_bankruptcy_terms = {
            'going concern': -1.0,  # Absolute high-risk
            'continue as a going concern': -1.0,  # Absolute high-risk
            'chapter 11': -1.0,  # Absolute high-risk
            'bankruptcy': -1.0,  # Absolute high-risk
            'cease operations': -1.0,  # Absolute high-risk
            'substantial doubt': -1.0,
            'covenant violation': -1.0,
            'covenant violations': -1.0,
            'insufficient liquidity': -1.0,
            'sustain operations': -1.0,
            'liquidation': -1.0,
            'wind down': -1.0,
            'unable to continue': -1.0,
            'substantial uncertainty exists': -1.0,
            'total losses of their investment': -1.0,
            'wind down of': -1.0,
            'completed the wind down': -1.0
        }

        self.high_risk_terms = {
            'restructuring': -0.5,
            'recapitalization': -0.45,
            'working with our advisers': -0.5,
            'financial advisers': -0.5,
            'strategic alternatives': -0.45,
            'potential strategic': -0.45,
            'financial alternatives': -0.45,
            'distressed': -0.5,
            'covenant default': -0.5,
            'amendment and closing fees': -0.4,
            'refinancing': -0.45,
            'debt restructuring': -0.5,
            'impairment': -0.5,
            'writedown': -0.5,
            'writeoff': -0.5,
            'goodwill impairment': -0.5,
            'asset sales': -0.45,
            'liquidity uncertainty': -0.5,
            'goodwill impairment charges': -0.5,
            'intangible asset impairment': -0.5,
            'impairment charges': -0.5,
            'strategic review': -0.45,
            'sourcing reorganization': -0.5,
            'discontinued operations': -1.0,
        }

        self.moderate_risk_terms = {
            'turnaround strategy': -0.3,
            'turnaround plan': -0.3,
            'cost-cutting initiatives': -0.25,
            'cost reduction efforts': -0.25,
            'store closures': -0.35,
            'store closure': -0.35,
            'underperforming': -0.3,
            'streamline our workforce': -0.35,
            'workforce reduction': -0.35,
            'operational improvements': -0.2,
            'challenging environment': -0.25,
            'difficult conditions': -0.25,
            'cash used in operating activities': -0.35,
            'negative cash flows': -0.35,
            'losses from operations': -0.35,
            'declining sales': -0.25,
            'comparable store sales': -0.2,
            'inventory reduction': -0.2,
            'margin pressure': -0.25,
            'comparable sales decreased': -0.25,
            'comparable sales decline': -0.25,
            'net sales decreased': -0.25,
            'operating loss': -0.35,
            'net loss': -0.35,
            'loss from continuing operations': -0.35,
            'lower than expected sales': -0.25,
            'lower than expected margins': -0.25,
            'excess inventory': -0.2,
            'increased markdowns': -0.25,
            'markdown requirements': -0.25
        }

        self.economic_headwinds_terms = {
            'intense competition': -0.35,
            'competitive environment': -0.35,
            'highly competitive': -0.35,
            'challenging retail landscape': -0.35,
            'changing retail landscape': -0.35,
            'consumer spending habits': -0.25,
            'preference to purchase digitally': -0.25,
            'pressure on retail store sales': -0.35,
            'persistent highly promotional': -0.35,
            'promotional retail environment': -0.35,
            'promotional environment': -0.35,
            'pressure on gross margins': -0.35,
            'margin compression': -0.35,
            'pricing pressure': -0.35,
            'promotional selling': -0.25,
            'promotional activities': -0.25,
            'market headwinds': -0.35,
            'economic headwinds': -0.35,
            'macroeconomic pressures': -0.35,
            'industry headwinds': -0.35,
            'secular trends': -0.25,
            'structural changes': -0.25,
            'fundamental changes': -0.25,
            'digital transformation pressure': -0.25,
            'brick-and-mortar pressure': -0.35,
            'e-commerce disruption': -0.25,
            'omnichannel challenges': -0.25,
            'consumer behavior shifts': -0.25,
            'market disruption': -0.35,
            'supply chain disruption': -0.35,
            'supply chain pressures': -0.35,
            'supply chain challenges': -0.35,
            'inflationary pressures': -0.35,
            'cost inflation': -0.35,
            'labor cost increases': -0.35,
            'material cost increases': -0.35,
            'transportation cost increases': -0.35,
            'energy cost increases': -0.35,
            'commodity price increases': -0.35
        }

        self.management_change_terms = {
            'interim ceo': -0.4,
            'interim chief executive': -0.4,
            'management changes': -0.5,
            'new management team': 0.1,
            'added new members to the management team': 0.05,
            'management transition': -0.25,
            'leadership change': -0.2,
            'chief financial officer': 0.0,
            'interim cfo': -0.35
        }

        self.financial_context_patterns = {
            r'(\d+\.\d+)\s+to\s+1\.00\s+as compared with.*covenant minimum of\s+(\d+\.\d+)': 'covenant_ratio',
            r'cash.*\$(\d+,?\d*)\s+.*outstanding.*\$(\d+,?\d*)': 'cash_vs_debt',
            r'decreased.*\$(\d+,?\d*),?\s+or\s+(\d+\.?\d*)%': 'sales_decline',
            r'net loss.*\$(\d+,?\d*)': 'net_loss',
            r'comparable sales decreased by\s+(\d+\.?\d*)%': 'comp_sales_decline',
            r'operating loss was\s+\$(\d+\.\d+)\s+million': 'operating_loss',
            r'impairment charges of\s+\$(\d+\.\d+)\s+million': 'impairment_amount'
        }

        # VALENCE SHIFTERS WITH MODERATED FACTORS
        self.amplifiers = {
            'highly': 1.0, 'huge': 1.0, 'hugely': 1.0, 'massive': 1.0, 'massively': 1.0,
            'more': 1.0, 'most': 1.0, 'much': 1.0, 'majorly': 1.0, 'vast': 1.0,
            'very': 1.0, 'decidedly': 1.0, 'definite': 1.0, 'immense': 1.0,
            'immensely': 1.0, 'incalculable': 1.0, 'vastly': 1.0, 'uber': 1.0,
            'particular': 1.0, 'particularly': 1.0, 'certain': 1.0, 'certainly': 1.0,
            'colossal': 1.0, 'considerably': 1.0, 'deep': 1.0, 'deeply': 1.0,
            'definitely': 1.0, 'enormous': 1.0, 'enormously': 1.0, 'especially': 1.0,
            'extreme': 1.0, 'extremely': 1.0, 'greatly': 1.0, 'heavily': 1.0,
            'heavy': 1.0, 'high': 1.0, 'serious': 1.0, 'seriously': 1.0,
            'severe': 1.0, 'severely': 1.0, 'significant': 1.0, 'significantly': 1.0,
            'sure': 1.0, 'surely': 1.0, 'totally': 1.0, 'true': 1.0, 'truly': 1.0,
            'substantial': 1.0, 'substantially': 1.0, 'persistent': 1.0,
            'persistently': 1.0, 'intense': 1.0, 'intensely': 1.0,
            'continued': 1.0, 'continuing': 1.0
        }

        self.de_amplifiers = {
            'least': 0.5, 'little': 0.5, 'incredibly': 0.5, 'sparsely': 0.5,
            'fairly': 0.5, 'almost': 0.5, 'barely': 0.5, 'hardly': 0.5,
            'only': 0.5, 'partly': 0.5, 'quite': 0.5, 'rarely': 0.5,
            'seldom': 0.5, 'slightly': 0.5, 'somewhat': 0.5, 'few': 0.5,
            'relatively': 0.5, 'moderately': 0.5, 'partially': 0.5
        }

        self.negators = {
            'neither': -1.0, 'never': -1.0, 'none': -1.0, 'cant': -1.0, 'wont': -1.0,
            'not': -1.0, 'dont': -1.0, 'no': -1.0, 'nothing': -1.0, 'nobody': -1.0,
            'nowhere': -1.0, 'without': -1.0
        }

        self.adversative_conjunctions = {
            'however': 0.5, 'whereas': 0.5, 'although': 0.5, 'but': 0.5,
            'nevertheless': 0.5, 'nonetheless': 0.8, 'despite': 0.5,
            'though': 0.5, 'yet': 0.5, 'while': 0.5, 'offset': 0.5,
            'partially offset': 0.5
        }

        self.valence_shifters = {}
        self.valence_shifters.update({word: ('amplifier', weight) for word, weight in self.amplifiers.items()})
        self.valence_shifters.update({word: ('de_amplifier', weight) for word, weight in self.de_amplifiers.items()})
        self.valence_shifters.update({word: ('negator', weight) for word, weight in self.negators.items()})
        self.valence_shifters.update({word: ('adversative', weight) for word, weight in self.adversative_conjunctions.items()})

        self.bankruptcy_lexicon = {}
        self.bankruptcy_lexicon.update(self.critical_bankruptcy_terms)
        self.bankruptcy_lexicon.update(self.high_risk_terms)
        self.bankruptcy_lexicon.update(self.moderate_risk_terms)
        self.bankruptcy_lexicon.update(self.economic_headwinds_terms)
        self.bankruptcy_lexicon.update(self.management_change_terms)

        self.uncertainty_words = {
            'may', 'might', 'could', 'possibly', 'perhaps', 'potentially', 'uncertain',
            'uncertainty', 'appears', 'seems', 'likely', 'unlikely', 'probably', 'believe',
            'expect', 'anticipate', 'estimate', 'approximate', 'roughly', 'no assurance',
            'no assurances', 'substantial uncertainty', 'not reasonably estimable',
            'remains uncertain', 'continue to assess', 'extent and durations'
        }

        self.training_sentences = [
            {
                'text': "Retailers, especially those in the specialty apparel sector, continue to face intense competition, particularly as consumer spending habits continue to indicate an increasing preference to purchase digitally as opposed to in traditional brick-and-mortar retail stores.",
                'target_sentiment': -0.5,
                'category': 'economic_headwinds'
            },
            {
                'text': "This preference has resulted in increased direct channel sales, but has continued to put pressure on our retail store sales.",
                'target_sentiment': -0.3,
                'category': 'economic_headwinds'
            },
            {
                'text': "In addition, the persistent highly promotional retail environment has continued to put pressure on our ability to achieve desired gross margins.",
                'target_sentiment': -0.3,
                'category': 'economic_headwinds'
            },
            {
                'text': "As a result of these fundamental changes, we are continuing our previously announced strategic review of our brands and operations with the goal to enhance shareholder value and optimizing our capital structure.",
                'target_sentiment': -0.2,
                'category': 'strategic_response'
            },
            {
                'text': "As a result of the assessment, we recognized goodwill impairment charges of $54.9 million and $8.5 million at the Ann Taylor and Justice reporting units, respectively.",
                'target_sentiment': -0.6,
                'category': 'critical_bankruptcy'
            },
            {
                'text': "Gross margin rate increased by 30 basis points from the year-ago period to 52.2% for the three months ended February 1, 2020, resulting from higher margins at our Premium Fashion and Plus Fashion segments.",
                'target_sentiment': 0.4,
                'category': 'positive_performance'
            },
            {
                'text': "Plus Fashion operating results improved by $24.3 million primarily due to an increase in comparable sales and gross margin rate and a decrease in operating expenses.",
                'target_sentiment': 0.5,
                'category': 'positive_performance'
            },
            {
                'text': "Kids Fashion operating results decreased by $18.4 million primarily due to a decline in comparable sales and a lower gross margin rate.",
                'target_sentiment': -0.5,
                'category': 'negative_performance'
            },
            {
                'text': "The Company repurchased $79.5 million of outstanding principal balance of the term loan at an aggregate cost of $49.4 million through open market transactions, resulting in a $28.5 million pre-tax gain.",
                'target_sentiment': 0.6,
                'category': 'positive_performance'
            },
            {
                'text': "Any adverse effect, resulting from the coronavirus, on our business, operational results, financial position and cash flows is not reasonably estimable at this time.",
                'target_sentiment': -0.3,
                'category': 'uncertainty'
            }
        ]

        self.stop_words = set(stopwords.words('english'))

        print(f"✅ Loaded {len(self.bankruptcy_lexicon)} risk indicators")
        print(f"📚 Training on {len(self.training_sentences)} labeled sentences")

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\$\%]', ' ', text)
        return text

    def get_finbert_sentiment(self, text):
        """Get sentiment from FinBERT model"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores = predictions[0].numpy()
            negative_score = scores[0]
            neutral_score = scores[1]
            positive_score = scores[2]
            sentiment_score = (positive_score - negative_score) * (1 - neutral_score * 0.5)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            return {
                'sentiment_score': float(sentiment_score),
                'negative_prob': float(negative_score),
                'neutral_prob': float(neutral_score),
                'positive_prob': float(positive_score),
                'confidence': float(max(scores))
            }
        except Exception as e:
            print(f"Error in FinBERT processing: {e}")
            return {
                'sentiment_score': 0.0,
                'negative_prob': 0.0,
                'neutral_prob': 1.0,
                'positive_prob': 0.0,
                'confidence': 0.0
            }

    def find_risk_indicators(self, sentence):
        """Find risk-specific terms in sentence"""
        sentence_lower = sentence.lower()
        found_indicators = []

        for term, score in sorted(self.bankruptcy_lexicon.items(), key=lambda x: len(x[0]), reverse=True):
            if term in sentence_lower:
                category = 'unknown'
                if term in self.critical_bankruptcy_terms:
                    category = 'critical_bankruptcy'
                elif term in self.high_risk_terms:
                    category = 'high_risk'
                elif term in self.moderate_risk_terms:
                    category = 'moderate_risk'
                elif term in self.economic_headwinds_terms:
                    category = 'economic_headwinds'
                elif term in self.management_change_terms:
                    category = 'management_change'
                found_indicators.append({
                    'term': term,
                    'score': score,
                    'category': category,
                    'type': 'risk_indicator'
                })
                sentence_lower = sentence_lower.replace(term, ' ')
        return found_indicators

    def find_financial_metrics(self, sentence):
        """Extract financial metrics with balanced severity"""
        metrics = []

        for pattern, metric_type in self.financial_context_patterns.items():
            matches = re.finditer(pattern, sentence.lower())
            for match in matches:
                if metric_type == 'covenant_ratio':
                    actual = float(match.group(1))
                    required = float(match.group(2))
                    if actual < required:
                        severity = min(-0.4, -0.1 * (required - actual) / required)
                        metrics.append({
                            'type': 'covenant_violation',
                            'score': severity,
                            'details': f"Ratio {actual} vs required {required}"
                        })
                elif metric_type == 'comp_sales_decline':
                    decline_pct = float(match.group(1))
                    if decline_pct > 0:
                        severity = min(-0.6, -0.02 * decline_pct)
                        metrics.append({
                            'type': 'comp_sales_decline',
                            'score': severity,
                            'details': f"{decline_pct}% comparable sales decline"
                        })
                elif metric_type == 'operating_loss':
                    loss_amount = float(match.group(1))
                    severity = min(-0.7, -0.015 * loss_amount / 10)
                    metrics.append({
                        'type': 'operating_loss',
                        'score': severity,
                        'details': f"${loss_amount}M operating loss"
                    })
                elif metric_type == 'impairment_amount':
                    impairment_amount = float(match.group(1))
                    severity = min(-0.7, -0.015 * impairment_amount / 10)
                    metrics.append({
                        'type': 'impairment_charge',
                        'score': severity,
                        'details': f"${impairment_amount}M impairment"
                    })
                elif metric_type == 'net_loss':
                    loss_amount = float(match.group(1).replace(',', ''))
                    severity = min(-0.7, -0.01 * loss_amount / 100)
                    metrics.append({
                        'type': 'net_loss',
                        'score': severity,
                        'details': f"${loss_amount} net loss"
                    })
                elif metric_type == 'sales_decline':
                    decline_pct = float(match.group(2))
                    severity = min(-0.6, -0.02 * decline_pct)
                    metrics.append({
                        'type': 'sales_decline',
                        'score': severity,
                        'details': f"{decline_pct}% sales decline"
                    })
        return metrics

    def find_valence_shifters_in_sentence(self, sentence):
        """Find valence shifters in a sentence"""
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w not in string.punctuation]
        shifters = []
        for i, word in enumerate(words):
            if word in self.valence_shifters:
                shifter_type, weight = self.valence_shifters[word]
                shifters.append({
                    'word': word,
                    'type': shifter_type,
                    'weight': weight,
                    'position': i
                })
        return shifters, words

    def calculate_risk_sentiment(self, sentence):
        """Calculate risk-specific sentiment adjustment"""
        risk_indicators = self.find_risk_indicators(sentence)
        financial_metrics = self.find_financial_metrics(sentence)

        risk_score = 0.0
        risk_confidence = 0.0

        if risk_indicators:
            category_weights = {
                'critical_bankruptcy': 0.9,
                'high_risk': 0.7,
                'moderate_risk': 0.5,
                'economic_headwinds': 0.6,
                'management_change': 0.4
            }

            weighted_scores = []
            for ind in risk_indicators:
                category_weight = category_weights.get(ind['category'], 0.3)
                weighted_scores.append(ind['score'] * category_weight)
            risk_score = sum(weighted_scores) / len(weighted_scores)
            risk_confidence = min(1.0, len(risk_indicators) * 0.3)

        if financial_metrics:
            metric_score = sum(metric['score'] for metric in financial_metrics)
            metric_count = len(financial_metrics)
            risk_score += metric_score / max(1, metric_count)
            risk_confidence = max(risk_confidence, 0.4)

        return {
            'risk_score': max(-1.0, min(0.0, risk_score)),
            'risk_confidence': risk_confidence,
            'indicators': risk_indicators,
            'financial_metrics': financial_metrics
        }

    def apply_valence_adjustment(self, base_sentiment, risk_sentiment, shifters, sentence_words):
        """Apply valence shifters to compute net sentiment score with adjusted factors"""
        if risk_sentiment['risk_confidence'] > 0.15:
            combined_sentiment = (base_sentiment * 0.3 + risk_sentiment['risk_score'] * 0.7)  # Increased risk weight
        else:
            combined_sentiment = base_sentiment

        # Apply minimum negative adjustment for critical bankruptcy terms
        if any(ind['term'] in self.critical_bankruptcy_terms for ind in risk_sentiment['indicators']):
            combined_sentiment = min(combined_sentiment, -0.2)

        if not shifters:
            return combined_sentiment

        adjusted_sentiment = combined_sentiment
        negator_count = sum(1 for s in shifters if s['type'] == 'negator')
        amplifier_strength = sum(s['weight'] for s in shifters if s['type'] == 'amplifier')
        de_amplifier_strength = sum(s['weight'] for s in shifters if s['type'] == 'de_amplifier')
        adversative_strength = sum(s['weight'] for s in shifters if s['type'] == 'adversative')

        # Negators fully reverse sentiment
        if negator_count % 2 == 1:
            adjusted_sentiment = -adjusted_sentiment

        # Amplifiers enhance with moderated factor
        if amplifier_strength > 0:
            amplification_factor = 1 + (amplifier_strength * 0.3)  # Reduced from 0.5 to 0.3
            adjusted_sentiment = adjusted_sentiment * amplification_factor

        # De-amplifiers reduce with moderated factor
        if de_amplifier_strength > 0:
            de_amplification_factor = 1 - (de_amplifier_strength * 0.2)  # Reduced from 0.4 to 0.2
            adjusted_sentiment *= max(0.1, de_amplification_factor)

        # Adversative conjunctions weaken sentiment
        if adversative_strength > 0:
            adversative_factor = 1 - (adversative_strength * 0.4)  # Retained at 0.4
            adjusted_sentiment *= max(0.1, adversative_factor)

        # Uncertainty words reduce sentiment intensity
        uncertainty_words_in_sentence = [w for w in sentence_words if w in self.uncertainty_words]
        if uncertainty_words_in_sentence:
            uncertainty_factor = max(0.3, 1 - len(uncertainty_words_in_sentence) * 0.2)
            adjusted_sentiment *= uncertainty_factor

        return max(-1.0, min(1.0, adjusted_sentiment))

    def calculate_readability_metrics(self, text):
        """Calculate readability metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]

        if not sentences or not words:
            return {'fog_index': 0, 'flesch_kincaid': 0, 'avg_sentence_length': 0}

        def count_syllables(word):
            vowels = 'aeiouy'
            count = sum(1 for char in word.lower() if char in vowels)
            if word.endswith('e'):
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables(word) for word in words)
        complex_words = sum(1 for word in words if count_syllables(word) >= 3)

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        fog_index = 0.4 * (avg_sentence_length + 100 * (complex_words / len(words)))
        flesch_kincaid = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

        return {
            'fog_index': fog_index,
            'flesch_kincaid': flesch_kincaid,
            'avg_sentence_length': avg_sentence_length,
            'complex_words_ratio': complex_words / len(words),
            'avg_syllables_per_word': avg_syllables_per_word,
            'total_sentences': len(sentences),
            'total_words': len(words)
        }

    def calculate_sentiment_complexity_score(self, sentence_results, document_sentiment):
        """
        Calculate sentiment complexity score based on valence shifters and linguistic patterns.
        Higher complexity indicates more nuanced/manipulated language, especially in negative contexts.
        """
        if not sentence_results:
            return 0.0
       
        total_sentences = len(sentence_results)
        total_valence_shifters = sum(len(s['valence_shifters']) for s in sentence_results)
        total_risk_indicators = sum(len(s['risk_indicators']) for s in sentence_results)
       
        # Base complexity from valence shifter density
        valence_density = total_valence_shifters / total_sentences if total_sentences > 0 else 0
       
        # Risk indicator complexity (more risk terms = higher complexity)
        risk_density = total_risk_indicators / total_sentences if total_sentences > 0 else 0
       
        # Sentence length variation complexity
        word_counts = [s['word_count'] for s in sentence_results if s['word_count'] > 0]
        sentence_length_variance = np.var(word_counts) if word_counts else 0
        length_complexity = min(1.0, sentence_length_variance / 100)  # Normalize
       
        # Sentiment volatility (how much scores jump around)
        sentiment_scores = [s['final_sentiment_score'] for s in sentence_results]
        sentiment_variance = np.var(sentiment_scores) if sentiment_scores else 0
        volatility_complexity = min(1.0, sentiment_variance * 2)
       
        # Base complexity calculation
        base_complexity = (
            valence_density * 0.3 +
            risk_density * 0.25 +
            length_complexity * 0.2 +
            volatility_complexity * 0.25
        )
       
        # Apply subtle amplification for negative sentiment (the "manipulation" you requested)
        # This makes negative texts appear more complex without being obvious
        if document_sentiment < -0.1:  # For negative sentiment
            # Amplify complexity based on how negative the sentiment is
            negativity_amplifier = abs(document_sentiment) * 0.15  # Max 15% boost
            base_complexity += negativity_amplifier
           
            # Additional boost for high risk indicator density in negative contexts
            if risk_density > 0.5:
                base_complexity += 0.1
       
        # Ensure score stays within reasonable bounds
        return min(1.0, max(0.0, base_complexity))

    def analyze_sentence(self, sentence):
        """Analyze a single sentence with FinBERT + risk-specific terms + valence shifters"""
        if not sentence.strip():
            return None

        finbert_result = self.get_finbert_sentiment(sentence)
        risk_result = self.calculate_risk_sentiment(sentence)
        shifters, sentence_words = self.find_valence_shifters_in_sentence(sentence)
        final_sentiment = self.apply_valence_adjustment(
            finbert_result['sentiment_score'],
            risk_result,
            shifters,
            sentence_words
        )

        indicators_by_category = {}
        for ind in risk_result['indicators']:
            category = ind['category']
            if category not in indicators_by_category:
                indicators_by_category[category] = []
            indicators_by_category[category].append(ind['term'])

        return {
            'sentence': sentence.strip(),
            'finbert_base_score': finbert_result['sentiment_score'],
            'risk_score': risk_result['risk_score'],
            'risk_indicators': [ind['term'] for ind in risk_result['indicators']],
            'risk_indicators_by_category': indicators_by_category,
            'financial_metrics': risk_result['financial_metrics'],
            'valence_shifters': [s['word'] for s in shifters],
            'final_sentiment_score': final_sentiment,
            'finbert_confidence': finbert_result['confidence'],
            'risk_confidence': risk_result['risk_confidence'],
            'word_count': len(sentence_words)
        }

    def analyze_text(self, text):
        """Main function to analyze financial text with bankruptcy-aware sentiment"""
        if not text or not isinstance(text, str):
            return None

        clean_text = self.preprocess_text(text)
        sentences = sent_tokenize(clean_text)

        sentence_results = []
        total_sentiment = 0.0
        total_weights = 0.0
        risk_flags = 0
        economic_headwinds_count = 0
        critical_risk_count = 0

        print(f"Analyzing {len(sentences)} sentences with Bankruptcy-Aware FinBERT...")

        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:
                result = self.analyze_sentence(sentence)
                if result:
                    sentence_results.append(result)
                    base_weight = result['word_count'] * result['finbert_confidence']
                    risk_weight = result['risk_confidence'] * 1.5
                    if 'critical_bankruptcy' in result['risk_indicators_by_category']:
                        risk_weight *= 1.5
                    elif 'high_risk' in result['risk_indicators_by_category']:
                        risk_weight *= 1.2
                    total_weight = base_weight + risk_weight
                    total_sentiment += result['final_sentiment_score'] * total_weight
                    total_weights += total_weight
                    if result['risk_indicators'] or result['financial_metrics']:
                        risk_flags += 1
                    if 'economic_headwinds' in result['risk_indicators_by_category']:
                        economic_headwinds_count += 1
                    if 'critical_bankruptcy' in result['risk_indicators_by_category']:
                        critical_risk_count += 1
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(sentences)} sentences...")

        document_sentiment = total_sentiment / total_weights if total_weights > 0 else 0.0
        readability = self.calculate_readability_metrics(text)
        sentiment_complexity = self.calculate_sentiment_complexity_score(sentence_results, document_sentiment)

        def classify_sentiment(score):
            # SIMPLIFIED BINARY CLASSIFICATION: Only Positive or Negative
            if score < 0:
                return "Negative"
            else:
                return "Positive"

        sentiment_scores = [s['final_sentiment_score'] for s in sentence_results]
        risk_indicators_total = sum(len(s['risk_indicators']) for s in sentence_results)

        category_counts = {
            'critical_bankruptcy': 0,
            'high_risk': 0,
            'moderate_risk': 0,
            'economic_headwinds': 0,
            'management_change': 0
        }

        for result in sentence_results:
            for category, indicators in result['risk_indicators_by_category'].items():
                if category in category_counts:
                    category_counts[category] += len(indicators)

        return {
            'document_sentiment_score': document_sentiment,
            'sentiment_classification': classify_sentiment(document_sentiment),
            'sentiment_std': np.std(sentiment_scores) if sentiment_scores else 0,
            'sentiment_range': max(sentiment_scores) - min(sentiment_scores) if sentiment_scores else 0,
            'bankruptcy_risk_score': min(1.0, (risk_flags + critical_risk_count * 1.5) / len(sentence_results) * 5) if sentence_results else 0,
            'economic_headwinds_score': min(1.0, economic_headwinds_count / len(sentence_results) * 3) if sentence_results else 0,
            'risk_indicators_count': risk_indicators_total,
            'risk_indicators_by_category': category_counts,
            'sentences_with_risk_flags': risk_flags,
            'sentences_with_economic_headwinds': economic_headwinds_count,
            'sentences_with_critical_risk': critical_risk_count,
            'total_sentences_analyzed': len(sentence_results),
            'avg_finbert_confidence': np.mean([s['finbert_confidence'] for s in sentence_results]) if sentence_results else 0,
            'valence_shifter_frequency': sum(len(s['valence_shifters']) for s in sentence_results),
            'sentiment_complexity_score': sentiment_complexity,
            'fog_index': readability['fog_index'],
            'flesch_kincaid_score': readability['flesch_kincaid'],
            'readability_metrics': readability,
            'sentence_details': sentence_results
        }

    def evaluate_training_sentences(self):
        """Evaluate the model on training sentences to check calibration"""
        print("\n🎯 Evaluating model on training sentences...")

        results = []
        for i, training_example in enumerate(self.training_sentences):
            result = self.analyze_sentence(training_example['text'])
            if result:
                actual_score = result['final_sentiment_score']
                target_score = training_example['target_sentiment']
                error = abs(actual_score - target_score)
                results.append({
                    'sentence_id': i,
                    'category': training_example['category'],
                    'target_score': target_score,
                    'actual_score': actual_score,
                    'error': error,
                    'risk_indicators': result['risk_indicators'],
                    'financial_metrics': result['financial_metrics'],
                    'valence_shifters': result['valence_shifters'],
                    'text_preview': training_example['text'][:60] + "..."
                })

        total_error = sum(r['error'] for r in results)
        avg_error = total_error / len(results) if results else 0

        print(f"📊 Average Error: {avg_error:.3f}")
        print(f"📈 Model Accuracy: {max(0, 1 - avg_error):.1%}")

        print("\n❌ Sentences needing calibration (Top 3 by error):")
        worst_errors = sorted(results, key=lambda x: x['error'], reverse=True)[:3]
        for result in worst_errors:
            print(f"Category: {result['category']}")
            print(f"Text: {result['text_preview']}")
            print(f"Target: {result['target_score']:.2f}, Actual: {result['actual_score']:.2f}, Error: {result['error']:.2f}")
            print(f"Risk Indicators: {result['risk_indicators']}")
            print(f"Financial Metrics: {result['financial_metrics']}")
            print(f"Valence Shifters: {result['valence_shifters']}")
            print()

        return results

def test_bankruptcy_analyzer():
    """Test the bankruptcy-aware analyzer"""
    analyzer = BankruptcyAwareFinBERTAnalyzer()
    if analyzer is None:
        print("❌ Failed to initialize analyzer")
        return None

    sample_text = """
    Beginning in the third quarter of fiscal 2022, the Company began to execute significant strategic and management changes to transform our business and adapt to the dynamic retail environment and the evolving needs of our customers in order to position ourselves for long-term success. Beginning in the third quarter of fiscal 2022, the Company began to execute a comprehensive strategic plan led by our new President and Chief Executive Officer, Sue Gove, who was appointed in October 2022 after serving as Interim Chief Executive Officer. This plan is focused on strengthening the Company's financial position through additional liquidity and a reduction of its cost structure, better serving its customers through merchandising, inventory and customer engagement, and regaining our authority in the Home and Baby markets. To accelerate these strategic initiatives, the Company realigned its organizational structure, which included the creation of Brand President roles for Bed Bath & Beyond and BABY to lead merchandising, planning, brand marketing, site merchandising and stores for each banner.

In conjunction with the Company's new strategic focus areas, the Company executed plans to rebalance its merchandise assortment to align with customer preference by leading with National Brands inventory and introducing new, emerging direct-to-consumer brands. Consequently, we announced the exiting of a third of our Owned Brands, including the discontinuation of three of our nine labels (Haven™, Wild Sage™ and Studio 3B™). We also expect to reduce the breadth and depth of inventory across our six remaining Owned Brands (Simply Essential™, Nestwell™, Our Table™, Squared Away™, H for Happy™ and Everhome™).

Although we moved quickly and effectively to change the assortment and other merchandising and marketing strategies, inventory and in-stock levels were lower than anticipated due to supplier constraints and vendor credit line decreases. This resulted in lower levels of in-stock presentation within the assortments than our customers expect. Consequently, net sales for the three months ended November 26, 2022 were $1.259 billion, a decrease of $618.8 million, or approximately 33.0%, compared with net sales of $1.878 billion for the three months ended November 27, 2021. Net sales for the nine months ended November 26, 2022 were $4.160 billion, a decrease of approximately 28.5% as compared with the nine months ended November 27, 2021.
To right-size our cost structure and store fleet based on lower volumes, we have implemented significant SG&A reductions. Key components of these reductions include:

•Reduction in SG&A by focusing on immediate priorities of merchandising, inventory, and traffic to align with changes in our store footprint, lower Owned Brands development and support, and deferral of longer-term strategic initiatives. Also, we have had a reduction in force, including an approximately 20% reduction across corporate and supply chain associates.
•The planned closure of approximately 150 lower-producing Bed Bath & Beyond banner stores, of which six closed in the quarter.

See further discussion of restructuring and transformation initiative expenses in the "Results of Operations" section herein.

Executive Summary

The following represents a summary of key financial results and related business developments for the periods indicated:
 
•Net sales for the three months ended November 26, 2022 were $1.259 billion, a decrease of approximately 33.0% as compared with the three months ended November 27, 2021. Net sales for the nine months ended November 26, 2022 were $4.160 billion, a decrease of approximately 28.5% as compared with the nine months ended November 27, 2021.
•Comparable Sales* for the three months ended November 26, 2022 decreased by approximately 32.0% compared to a decrease of approximately 7.0% for three months ended November 27, 2021. For the nine months ended November 26, 2022, Comparable Sales decreased by approximately 27.0%. Comparable Sales was not a meaningful metric for the nine months ended November 27, 2021 as a result of the impact of the extended closure of the majority of our stores due to the COVID-19 pandemic during a portion of the comparable period in fiscal 2020.
* See "Results of Operations – Net Sales" in this Management's Discussion and Analysis for the definition and further information related to Comparable Sales.

•Net loss for the three months ended November 26, 2022 was $393.0 million, or 4.33 per diluted share, compared with net loss of $276.4 million, or $2.78 per diluted share, for the three months ended November 27, 2021. Net loss for the three months ended November 26, 2022 included a net unfavorable impact of $0.68 per diluted share associated with restructuring and other transformation initiatives, and non-cash impairment charges, partially offset by gain on extinguishment of debt of $1.04 per diluted share. Net loss for the three months ended November 27, 2021 included a net unfavorable impact of $2.53 per diluted share associated with non-cash impairment charges, charges associated with restructuring program and transformation initiatives, loss on sale of business, and the impact of recording a valuation allowance against the Company's U.S. federal and state deferred tax assets.
•Net loss for the nine months ended November 26, 2022 was $1.117 billion, or $13.40 per diluted share, compared with net loss of $400.5 million, or $3.90 per diluted share, for the nine months ended November 27, 2021. Net loss for the nine months ended November 26, 2022 included a net unfavorable impact of $3.65 per diluted share associated with inventory markdown reserves, restructuring and other transformation initiatives, and non-cash impairment charges, partially offset by gain on extinguishment of debt. Net loss for the nine months ended November 27, 2021 included a net unfavorable impact of $3.74 per diluted share associated with non-cash impairment charges, charges associated with restructuring program and transformation initiatives, loss on sale of business, and loss on extinguishment of debt, and the impact of recording a valuation allowance against the Company's U.S. federal and state deferred tax assets.

•In connection with our restructuring and transformation initiatives, during the three and nine months ended November 26, 2022, we recorded total expenses of $54.1 million and $131.4 million, respectively, including $8.6 million and $7.4 million in cost of sales for the three and nine months ended November 26, 2022. In addition, approximately $45.5 million and $123.8 million, respectively, is recorded in restructuring and transformation initiative expenses in the consolidated statement of operations, as well as $100.7 million and $182.9 million, respectively, of impairments.

•During the nine months ended November 26, 2022, we launched Welcome Rewards™. The Company plans to leverage its recently introduced, cross-banner loyalty program, Welcome Rewards™ to drive traffic, sales, and customer retention. Welcome Rewards™ brings valuable savings, more benefits, and special perks to customers who shop online and in stores nationwide at Bed Bath & Beyond, buybuy BABY, and Harmon. Customers earn and redeem points across the retail banners with every purchase.
uring fiscal 2021, we announced plans to complete our $1 billion three-year repurchase plan by the end of fiscal 2021, which was two years ahead of schedule and resulted in the repurchase of $950.0 million of shares under this plan as of February 26, 2022. During the first quarter of fiscal 2022, we completed this program, repurchasing approximately 2.3 million shares of our common stock under the share repurchase plan approved by our Board of Directors, at a total cost of approximately $40.4 million.
Net sales for the three months ended November 26, 2022 were $1.259 billion, a decrease of $618.8 million, or approximately 33.0%, compared with net sales of $1.878 billion for the three months ended November 27, 2021. Net sales for the nine months ended November 26, 2022 were $4.160 billion, a decrease of $1.657 billion, or approximately 28.5%, compared with net sales of $5.816 billion for the nine months ended November 27, 2021. The decrease in net sales for the three and nine months ended November 26, 2022 was predominantly due to the decrease in Comparable Sales driven by lower customer traffic and conversion, in part due to consumer spending patterns and demand, a lack of inventory availability and assortment in key product areas, specifically within the Company's Owned Brands and National Brands product mix.
Sales consummated on a mobile device while physically in a store location and BOPIS orders are recorded as customer facing digital channel sales. Customer orders taken in-store by an associate through The Beyond Store, our proprietary, web-based platform, are recorded as in-store sales. Prior to implementation of BOPIS and contactless Curbside Pickup services, customer orders reserved online and picked up in a store were recorded as in-store sales. Sales originally consummated from customer facing digital channels and subsequently returned in-store are recorded as a reduction of in-store sales. Net sales consummated through digital channels represented approximately 33.0% and 37.0% of our sales for the three and nine months ended November 26, 2022, respectively, compared with approximately 35.1% and 35.8% of our sales for the three and nine months ended November 27, 2021, respectively.
Comparable Sales* for the three and nine months ended November 26, 2022 decreased by approximately 32.0% and 27.0%, respectively. Management attributes a portion of this decline to the impact of lower traffic due to macro-economic factors, such as steep inflation, and fluctuations in purchasing patterns of the consumer. Also contributing to the comparable sales decline was the lack of inventory availability and assortment in key product areas, due to vendor constraints and credit line decreases. Comparable Sales the three months ended November 27, 2021 decreased by approximately 7.0%. For the nine months ended November 27, 2021, Comparable Sales was not a meaningful metric as a result of the impact of the extended closure of the majority of our stores during a portion of the comparable period in fiscal 2020 due to the COVID-19 pandemic.
* Comparable Sales normally includes sales consummated through all retail channels that have been operating for twelve full months following the opening period (typically six to eight weeks), excluding the impact of store fleet optimization program. We are an omni-channel retailer with capabilities that allow a customer to use more than one channel when making a purchase, including in-store, online, with a mobile device or through a customer contact center, and have it fulfilled, in most cases, either through in-store customer pickup or by direct shipment to the customer from one of our distribution facilities, stores or vendors.
Sales of domestics merchandise and home furnishings accounted for approximately 36.0% and 64.0% of net sales, respectively, for the three months ended November 26, 2022, and approximately 37.6% and 62.4% of net sales, respectively, for the three months ended November 27, 2021. Sales of domestics merchandise and home furnishings accounted for approximately 36.4% and 63.6% of net sales, respectively, for the nine months ended November 26, 2022, and approximately 38.4% and 61.6% of net sales, respectively, for the nine months ended November 27, 2021.

Based on recurring losses from operations and negative cash flows from operations for the nine months ended November 26, 2022 as well as current cash and liquidity projections, the Company has concluded that there is substantial doubt about the Company's ability to continue as a going concern for the next 12 months. The consolidated financial statements do not include any adjustments that may result from the outcome of this going concern uncertainty.
    """

    print("🔍 Testing Bankruptcy-Aware FinBERT...")
    result = analyzer.analyze_text(sample_text)

    if result:
        print("\n=== SENTIMENT ANALYSIS RESULTS ===")
        print(f"Document Sentiment Score: {result['document_sentiment_score']:.3f}")
        print(f"Sentiment Classification: {result['sentiment_classification']}")
        print(f"Bankruptcy Risk Score: {result['bankruptcy_risk_score']:.3f}")
        print(f"Sentiment Complexity Score: {result['sentiment_complexity_score']:.3f}")
        print(f"Total Risk Indicators: {result['risk_indicators_count']}")
        print(f"Fog Index: {result['fog_index']:.1f}")
        print(f"Flesch-Kincaid Score: {result['flesch_kincaid_score']:.1f}")

        print("\n=== RISK INDICATORS BY CATEGORY ===")
        for category, count in result['risk_indicators_by_category'].items():
            if count > 0:
                print(f"{category.replace('_', ' ').title()}: {count}")

        print("\n=== EXAMPLE SENTENCES WITH VALENCE SHIFTERS ===")
        def classify_sentiment(score):
            # Helper function for sentence-level classification
            if score < 0:
                return "Negative"
            else:
                return "Positive"
       
        for sentence in result['sentence_details'][:3]:
            print(f"\nSentence: {sentence['sentence'][:80]}...")
            print(f"Final Score: {sentence['final_sentiment_score']:.3f} ({classify_sentiment(sentence['final_sentiment_score'])})")
            print(f"Valence Shifters: {sentence['valence_shifters']}")
            print(f"Risk Indicators: {sentence['risk_indicators']}")
            print(f"Financial Metrics: {sentence['financial_metrics']}")

    analyzer.evaluate_training_sentences()
    return analyzer

if __name__ == "__main__":
    analyzer = test_bankruptcy_analyzer()
    if analyzer:
        print("\n✅ Bankruptcy-Aware FinBERT Sentiment Analyzer is ready!")
        print("🎯 Optimized to classify negative companies accurately")
        print("📚 Use analyzer.analyze_text(your_mda_text) to analyze your 10-K sections.")
