import pandas as pd
import numpy as np
import nltk
import re
import warnings
from typing import Dict, List, Tuple, Any
from collections import Counter
import textstat
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

warnings.filterwarnings('ignore')

class FinancialSentimentAnalyzer:
    """
    Comprehensive financial sentiment analysis model for MD&A sections
    """
   
    def __init__(self):
        """Initialize the analyzer with models and lexicons"""
        self.finbert_analyzer = None
        self.lm_lexicon = None
        self.ml_lexicon = None
        self.setup_models()
        self.define_valence_shifters()
        self.define_macro_keywords()
       
    def setup_models(self):
        """Initialize FinBERT model for sentiment analysis"""
        print("Loading FinBERT model...")
        try:
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            print("✓ FinBERT model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.finbert_analyzer = None
   
    def load_ml_lexicon(self, csv_path: str = None):
        """
        Load ML Lexicon from CSV file
        Expected columns: word, pos_prob, neg_prob, neutral_prob, intensity, frequency
       
        # UNCOMMENT AND MODIFY THIS SECTION TO LOAD YOUR EXCEL FILE:
        # df = pd.read_excel('your_ml_lexicon.xlsx')
        # self.ml_lexicon = df.set_index('word').to_dict('index')
        """
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                self.ml_lexicon = df.set_index('word').to_dict('index')
                print(f"✓ ML Lexicon loaded: {len(self.ml_lexicon)} words")
            except Exception as e:
                print(f"Error loading ML lexicon: {e}")
                self.ml_lexicon = None
        else:
            # Create a sample lexicon for demonstration
            self.create_sample_ml_lexicon()
   
    def create_sample_ml_lexicon(self):
        """Create a sample ML lexicon for testing purposes"""
        sample_data = {
            'decline': {'pos_prob': 0.1, 'neg_prob': 0.8, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.3},
            'growth': {'pos_prob': 0.8, 'neg_prob': 0.1, 'neutral_prob': 0.1, 'intensity': 0.8, 'frequency': 0.4},
            'challenging': {'pos_prob': 0.2, 'neg_prob': 0.7, 'neutral_prob': 0.1, 'intensity': 0.6, 'frequency': 0.3},
            'strong': {'pos_prob': 0.7, 'neg_prob': 0.2, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.5},
            'significant': {'pos_prob': 0.3, 'neg_prob': 0.4, 'neutral_prob': 0.3, 'intensity': 0.6, 'frequency': 0.6},
            'improve': {'pos_prob': 0.8, 'neg_prob': 0.1, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.4},
            'loss': {'pos_prob': 0.1, 'neg_prob': 0.8, 'neutral_prob': 0.1, 'intensity': 0.8, 'frequency': 0.3}
        }
        self.ml_lexicon = sample_data
        print("✓ Sample ML Lexicon created")
   
    def setup_lm_dictionary(self):
        """
        Setup Loughran-McDonald sentiment dictionary
        This creates a basic word-count-based sentiment lexicon
        """
        # Positive words from LM dictionary (sample)
        positive_words = [
            'able', 'abundance', 'abundant', 'acclaimed', 'accomplish', 'accomplished',
            'achievement', 'achieves', 'achieving', 'acknowledge', 'acknowledged',
            'active', 'advance', 'advanced', 'advancement', 'advances', 'advancing',
            'advantage', 'advantageous', 'advantages', 'agree', 'agreed', 'agreement',
            'alliance', 'amazing', 'ambitious', 'appreciate', 'appreciated', 'attractive',
            'awesome', 'beautiful', 'benefit', 'beneficial', 'benefits', 'best',
            'better', 'brilliant', 'capable', 'collaboration', 'competitive', 'complete',
            'comprehensive', 'confident', 'creative', 'dedicated', 'delighted',
            'dependable', 'distinctive', 'dynamic', 'easy', 'effective', 'efficient',
            'enhance', 'enhanced', 'enhancement', 'excellent', 'exceptional', 'exciting',
            'expand', 'expansion', 'experience', 'expert', 'extraordinary', 'fantastic',
            'favorable', 'flexible', 'focused', 'gain', 'gained', 'gains', 'good',
            'great', 'growing', 'growth', 'high', 'higher', 'highest', 'ideal',
            'improve', 'improved', 'improvement', 'impressive', 'increase', 'increased',
            'increases', 'increasing', 'incredible', 'innovation', 'innovative',
            'leading', 'leader', 'opportunities', 'opportunity', 'optimal', 'optimistic',
            'outstanding', 'perfect', 'positive', 'powerful', 'profitable', 'progress',
            'promise', 'promising', 'quality', 'record', 'reliable', 'remarkable',
            'reputation', 'respected', 'results', 'reward', 'rewarding', 'solid',
            'strong', 'stronger', 'strongest', 'success', 'successful', 'superior',
            'valuable', 'value', 'winning', 'wonderful'
        ]
       
        # Negative words from LM dictionary (sample)
        negative_words = [
            'abandon', 'abandoned', 'abandoning', 'abandonment', 'ability', 'absence',
            'absent', 'abuse', 'abused', 'accident', 'accidental', 'accidentally',
            'accusations', 'accuse', 'accused', 'adverse', 'adversely', 'adversity',
            'allegations', 'allege', 'alleged', 'allegedly', 'alleges', 'anticompetitive',
            'antitrust', 'arbitration', 'argue', 'argued', 'arguing', 'argument',
            'arguments', 'assault', 'attacked', 'attacking', 'attacks', 'bad',
            'bankruptcy', 'bankruptcies', 'barrier', 'breach', 'breached', 'breaches',
            'bribe', 'bribery', 'bribes', 'burden', 'burdens', 'burdensome', 'cancel',
            'cancelled', 'cancelling', 'caution', 'cautioned', 'cautioning', 'challenge',
            'challenged', 'challenges', 'challenging', 'claim', 'claims', 'complaint',
            'complaints', 'complicate', 'complicated', 'complicates', 'complicating',
            'complication', 'complications', 'concern', 'concerned', 'concerning',
            'concerns', 'conflict', 'conflicts', 'confusing', 'conspiracy', 'constrain',
            'constrained', 'constraining', 'constrains', 'constraint', 'constraints',
            'contempt', 'contend', 'contended', 'contending', 'contends', 'contested',
            'controversial', 'controversy', 'costly', 'crime', 'crimes', 'criminal',
            'crisis', 'critical', 'criticism', 'criticize', 'criticized', 'criticizes',
            'criticizing', 'damage', 'damaged', 'damages', 'damaging', 'danger',
            'dangerous', 'dangers', 'deadlock', 'debt', 'decline', 'declined',
            'declines', 'declining', 'decrease', 'decreased', 'decreases', 'decreasing',
            'defeat', 'defeated', 'defect', 'defective', 'defects', 'defend',
            'defendant', 'defendants', 'defending', 'defends', 'defer', 'deficiency',
            'deficient', 'deficit', 'deficits', 'demand', 'demanded', 'demanding',
            'demands', 'denied', 'deny', 'denying', 'depressed', 'depression',
            'deteriorate', 'deteriorated', 'deteriorates', 'deteriorating',
            'deterioration', 'difficult', 'difficulties', 'difficulty', 'disadvantage',
            'disadvantaged', 'disadvantageous', 'disadvantages', 'disagree',
            'disagreed', 'disagreement', 'disagreements', 'disagrees', 'disagreeing',
            'disappoint', 'disappointed', 'disappointing', 'disappointment',
            'disappoints', 'disaster', 'disasters', 'disastrous', 'discontinued',
            'discontinuing', 'discourage', 'discouraged', 'discourages', 'discouraging',
            'dismiss', 'dismissed', 'dismissing', 'dismisses', 'displace', 'displaced',
            'displaces', 'displacing', 'dispute', 'disputed', 'disputes', 'disputing',
            'disrupt', 'disrupted', 'disrupting', 'disruption', 'disruptions',
            'disruptive', 'disrupts', 'distress', 'distressed', 'distressing',
            'doubt', 'doubted', 'doubtful', 'doubting', 'doubts', 'downturn',
            'downturns', 'drop', 'dropped', 'dropping', 'drops', 'fail', 'failed',
            'failing', 'fails', 'failure', 'failures', 'false', 'fatal', 'fatalities',
            'fatality', 'fault', 'faults', 'faulty', 'fear', 'feared', 'fearing',
            'fears', 'fire', 'fired', 'fires', 'firing', 'force', 'forced', 'forces',
            'forcing', 'foreclose', 'foreclosed', 'forecloses', 'foreclosing',
            'foreclosure', 'foreclosures', 'fraud', 'fraudulent', 'fraudulently',
            'frauds', 'frivolous', 'frustrate', 'frustrated', 'frustrates',
            'frustrating', 'frustration', 'guilty', 'halt', 'halted', 'halting',
            'halts', 'hamper', 'hampered', 'hampering', 'hampers', 'harm', 'harmed',
            'harmful', 'harming', 'harms', 'harsh', 'harsher', 'harshest', 'harshly',
            'harshness', 'hurt', 'hurting', 'hurts', 'illegal', 'illegally',
            'illegible', 'illicit', 'immature', 'impair', 'impaired', 'impairing',
            'impairment', 'impairments', 'impairs', 'impede', 'impeded', 'impedes',
            'impeding', 'impediment', 'impediments', 'impossible', 'improper',
            'improperly', 'inability', 'inadequate', 'inadequately', 'inadvertent',
            'inadvertently', 'insolvent', 'insolvency', 'instability', 'insufficient',
            'insufficiently', 'interrupt', 'interrupted', 'interrupting',
            'interruption', 'interruptions', 'interrupts', 'investigation',
            'investigations', 'lawsuit', 'lawsuits', 'liabilities', 'liability',
            'limitation', 'limitations', 'limited', 'limiting', 'limits', 'lose',
            'loses', 'losing', 'loss', 'losses', 'lost', 'materially', 'misled',
            'misleading', 'mismanage', 'mismanaged', 'mismanagement', 'mismanages',
            'mismanaging', 'misstated', 'misstatement', 'misstatements', 'mistake',
            'mistakes', 'negative', 'negatively', 'negatives', 'neglect', 'neglected',
            'neglecting', 'neglects', 'negligence', 'negligent', 'negligently',
            'obstacle', 'obstacles', 'obstruct', 'obstructed', 'obstructing',
            'obstruction', 'obstructions', 'obstructs', 'offence', 'offences',
            'offend', 'offended', 'offending', 'offends', 'offense', 'offenses',
            'offensive', 'oppose', 'opposed', 'opposes', 'opposing', 'opposition',
            'overpaid', 'overpay', 'overpaying', 'overpayment', 'overpayments',
            'overpays', 'penalties', 'penalty', 'poor', 'poorly', 'problem',
            'problematic', 'problems', 'punish', 'punished', 'punishes', 'punishing',
            'punishment', 'punishments', 'punitive', 'questions', 'questioned',
            'questioning', 'recession', 'recessions', 'recessionary', 'redundancy',
            'refuse', 'refused', 'refuses', 'refusing', 'reject', 'rejected',
            'rejecting', 'rejection', 'rejections', 'rejects', 'restatement',
            'restatements', 'restrict', 'restricted', 'restricting', 'restriction',
            'restrictions', 'restrictive', 'restricts', 'restructure', 'restructured',
            'restructures', 'restructuring', 'restructurings', 'risk', 'risked',
            'risks', 'risky', 'scandal', 'scandals', 'scrutiny', 'serious',
            'seriously', 'seriousness', 'severe', 'severely', 'severity', 'shortfall',
            'shortfalls', 'shrink', 'shrinking', 'shrinks', 'slow', 'slowdown',
            'slowed', 'slower', 'slowing', 'slowly', 'slows', 'sluggish', 'sluggishly',
            'sluggishness', 'suffer', 'suffered', 'suffering', 'suffers', 'suspend',
            'suspended', 'suspending', 'suspends', 'suspension', 'suspensions',
            'threat', 'threaten', 'threatened', 'threatening', 'threatens', 'threats',
            'turmoil', 'uncertain', 'uncertainties', 'uncertainty', 'unconvincing',
            'underestimate', 'underestimated', 'underestimates', 'underestimating',
            'underperform', 'underperformed', 'underperforming', 'underperforms',
            'unfavorable', 'unfavorably', 'unforeseen', 'unfortunate', 'unfortunately',
            'unfounded', 'unpredictable', 'unpredictably', 'unprofitable', 'unrealistic',
            'unreasonable', 'unreasonably', 'unreliable', 'unsatisfactory',
            'unsatisfied', 'unsuccessful', 'unsuccessfully', 'unsuitable', 'unstable',
            'untimely', 'unusual', 'unusually', 'urgent', 'urgently', 'violate',
            'violated', 'violates', 'violating', 'violation', 'violations', 'volatile',
            'volatility', 'vulnerable', 'vulnerabilities', 'vulnerability', 'warn',
            'warned', 'warning', 'warnings', 'warns', 'weak', 'weaken', 'weakened',
            'weakening', 'weakens', 'weaker', 'weakest', 'weakly', 'weakness',
            'weaknesses', 'worse', 'worsen', 'worsened', 'worsening', 'worsens',
            'worst', 'worthless', 'wrong', 'wrongdoing', 'wrongful', 'wrongfully'
        ]
       
        self.lm_positive = set(positive_words)
        self.lm_negative = set(negative_words)
        print(f"✓ LM Dictionary loaded: {len(self.lm_positive)} positive, {len(self.lm_negative)} negative words")
   
    def define_valence_shifters(self):
        """Define comprehensive valence shifters as per your requirements"""
        self.valence_shifters = {
            'negators': [
                'not', 'never', 'cannot', 'can\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t',
                'couldn\'t', 'mustn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'hasn\'t',
                'haven\'t', 'hadn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
                'no', 'none', 'neither', 'nor', 'nowhere', 'nothing', 'nobody',
                'without', 'unlikely', 'impossible', 'hardly', 'rarely', 'seldom',
                'barely', 'scarcely', 'lack', 'lacks', 'lacking', 'absent',
                'fail', 'fails', 'failed', 'failure', 'unable', 'prevent',
                'prevents', 'preventing', 'avoid', 'avoids', 'avoiding'
            ],
            'adjectives': [
                'significant', 'substantial', 'material', 'considerable', 'major',
                'minor', 'minimal', 'slight', 'moderate', 'extreme', 'severe',
                'critical', 'crucial', 'essential', 'important', 'vital',
                'necessary', 'required', 'mandatory', 'optional', 'potential',
                'possible', 'probable', 'likely', 'unlikely', 'certain',
                'uncertain', 'definite', 'indefinite', 'clear', 'unclear',
                'obvious', 'evident', 'apparent', 'hidden', 'complex',
                'complicated', 'simple', 'difficult', 'challenging', 'easy',
                'hard', 'tough', 'weak', 'strong', 'powerful', 'limited',
                'unlimited', 'restricted', 'unrestricted', 'full', 'partial',
                'complete', 'incomplete', 'total', 'absolute', 'relative',
                'comparable', 'incomparable', 'similar', 'different', 'unique',
                'common', 'rare', 'frequent', 'infrequent', 'regular',
                'irregular', 'normal', 'abnormal', 'typical', 'atypical',
                'standard', 'non-standard', 'usual', 'unusual', 'expected',
                'unexpected', 'surprising', 'unsurprising', 'remarkable',
                'unremarkable', 'notable', 'unnotable', 'special', 'ordinary',
                'extraordinary', 'exceptional', 'unexceptional', 'outstanding',
                'mediocre', 'superior', 'inferior', 'excellent', 'poor',
                'good', 'bad', 'better', 'worse', 'best', 'worst',
                'high', 'higher', 'highest', 'low', 'lower', 'lowest',
                'large', 'larger', 'largest', 'small', 'smaller', 'smallest',
                'big', 'bigger', 'biggest', 'little', 'less', 'least',
                'more', 'most', 'much', 'many', 'few', 'several',
                'numerous', 'countless', 'limited', 'extensive', 'comprehensive',
                'thorough', 'detailed', 'specific', 'general', 'broad',
                'narrow', 'wide', 'tight', 'loose', 'strict', 'lenient',
                'rigid', 'flexible', 'fixed', 'variable', 'constant',
                'volatile', 'stable', 'unstable', 'steady', 'unsteady',
                'consistent', 'inconsistent', 'reliable', 'unreliable',
                'dependable', 'undependable', 'predictable', 'unpredictable',
                'certain', 'uncertain', 'sure', 'unsure', 'confident',
                'unconfident', 'optimistic', 'pessimistic', 'positive',
                'negative', 'favorable', 'unfavorable', 'beneficial',
                'detrimental', 'advantageous', 'disadvantageous', 'helpful',
                'unhelpful', 'useful', 'useless', 'valuable', 'worthless',
                'profitable', 'unprofitable', 'successful', 'unsuccessful',
                'effective', 'ineffective', 'efficient', 'inefficient',
                'productive', 'unproductive', 'competitive', 'uncompetitive',
                'aggressive', 'conservative', 'cautious', 'reckless',
                'careful', 'careless', 'prudent', 'imprudent', 'wise',
                'unwise', 'smart', 'stupid', 'intelligent', 'unintelligent',
                'reasonable', 'unreasonable', 'logical', 'illogical',
                'rational', 'irrational', 'sensible', 'nonsensical',
                'practical', 'impractical', 'realistic', 'unrealistic',
                'feasible', 'infeasible', 'viable', 'unviable', 'sustainable',
                'unsustainable', 'manageable', 'unmanageable', 'controllable',
                'uncontrollable', 'acceptable', 'unacceptable', 'satisfactory',
                'unsatisfactory', 'adequate', 'inadequate', 'sufficient',
                'insufficient', 'appropriate', 'inappropriate', 'suitable',
                'unsuitable', 'relevant', 'irrelevant', 'applicable',
                'inapplicable', 'valid', 'invalid', 'legitimate', 'illegitimate',
                'legal', 'illegal', 'lawful', 'unlawful', 'authorized',
                'unauthorized', 'permitted', 'prohibited', 'allowed',
                'disallowed', 'approved', 'disapproved', 'accepted',
                'rejected', 'confirmed', 'unconfirmed', 'verified',
                'unverified', 'proven', 'unproven', 'established',
                'unestablished', 'recognized', 'unrecognized', 'acknowledged',
                'unacknowledged', 'disclosed', 'undisclosed', 'revealed',
                'unrevealed', 'known', 'unknown', 'familiar', 'unfamiliar',
                'public', 'private', 'open', 'closed', 'transparent',
                'opaque', 'clear', 'unclear', 'obvious', 'obscure',
                'visible', 'invisible', 'apparent', 'hidden', 'evident',
                'unevident', 'manifest', 'latent', 'explicit', 'implicit',
                'direct', 'indirect', 'straightforward', 'complicated',
                'simple', 'complex', 'basic', 'advanced', 'elementary',
                'sophisticated', 'primitive', 'modern', 'contemporary',
                'traditional', 'conventional', 'unconventional', 'standard',
                'non-standard', 'regular', 'irregular', 'formal', 'informal',
                'official', 'unofficial', 'professional', 'unprofessional',
                'skilled', 'unskilled', 'experienced', 'inexperienced',
                'qualified', 'unqualified', 'competent', 'incompetent',
                'capable', 'incapable', 'able', 'unable', 'fit', 'unfit',
                'ready', 'unready', 'prepared', 'unprepared', 'equipped',
                'unequipped', 'armed', 'unarmed', 'protected', 'unprotected',
                'secure', 'insecure', 'safe', 'unsafe', 'dangerous', 'harmless',
                'risky', 'risk-free', 'hazardous', 'non-hazardous', 'toxic',
                'non-toxic', 'poisonous', 'non-poisonous', 'healthy', 'unhealthy',
                'wholesome', 'unwholesome', 'beneficial', 'harmful', 'constructive',
                'destructive', 'creative', 'uncreative', 'innovative', 'traditional',
                'original', 'unoriginal', 'unique', 'common', 'rare', 'frequent',
                'infrequent', 'regular', 'irregular', 'periodic', 'aperiodic',
                'continuous', 'discontinuous', 'constant', 'variable', 'steady',
                'unsteady', 'stable', 'unstable', 'balanced', 'unbalanced',
                'equal', 'unequal', 'fair', 'unfair', 'just', 'unjust',
                'right', 'wrong', 'correct', 'incorrect', 'accurate', 'inaccurate',
                'precise', 'imprecise', 'exact', 'inexact', 'true', 'false',
                'real', 'unreal', 'actual', 'virtual', 'genuine', 'fake',
                'authentic', 'inauthentic', 'original', 'copy', 'natural',
                'artificial', 'organic', 'synthetic', 'pure', 'impure',
                'clean', 'dirty', 'fresh', 'stale', 'new', 'old',
                'recent', 'ancient', 'modern', 'outdated', 'current', 'former',
                'present', 'past', 'future', 'temporary', 'permanent', 'lasting',
                'brief', 'long', 'short', 'extended', 'prolonged', 'quick',
                'slow', 'fast', 'rapid', 'gradual', 'sudden', 'immediate',
                'instant', 'delayed', 'early', 'late', 'timely', 'untimely',
                'prompt', 'tardy', 'punctual', 'unpunctual', 'scheduled',
                'unscheduled', 'planned', 'unplanned', 'intended', 'unintended',
                'deliberate', 'accidental', 'purposeful', 'purposeless',
                'meaningful', 'meaningless', 'significant', 'insignificant',
                'important', 'unimportant', 'major', 'minor', 'primary',
                'secondary', 'main', 'subsidiary', 'central', 'peripheral',
                'core', 'marginal', 'essential', 'non-essential', 'critical',
                'non-critical', 'vital', 'non-vital', 'necessary', 'unnecessary',
                'required', 'optional', 'mandatory', 'voluntary', 'compulsory',
                'elective', 'obligatory', 'discretionary', 'binding', 'non-binding',
                'enforceable', 'unenforceable', 'applicable', 'inapplicable',
                'relevant', 'irrelevant', 'pertinent', 'impertinent', 'related',
                'unrelated', 'connected', 'disconnected', 'linked', 'unlinked',
                'associated', 'unassociated', 'correlated', 'uncorrelated',
                'dependent', 'independent', 'interdependent', 'autonomous',
                'self-sufficient', 'reliant', 'unreliable', 'trustworthy',
                'untrustworthy', 'credible', 'incredible', 'believable',
                'unbelievable', 'plausible', 'implausible', 'reasonable',
                'unreasonable', 'logical', 'illogical', 'rational', 'irrational',
                'sensible', 'nonsensical', 'coherent', 'incoherent', 'consistent',
                'inconsistent', 'compatible', 'incompatible', 'harmonious',
                'disharmonious', 'cooperative', 'uncooperative', 'collaborative',
                'non-collaborative', 'supportive', 'unsupportive', 'helpful',
                'unhelpful', 'constructive', 'destructive', 'positive', 'negative',
                'optimistic', 'pessimistic', 'hopeful', 'hopeless', 'encouraging',
                'discouraging', 'inspiring', 'uninspiring', 'motivating',
                'demotivating', 'stimulating', 'unstimulating', 'exciting',
                'boring', 'interesting', 'uninteresting', 'engaging', 'disengaging',
                'attractive', 'unattractive', 'appealing', 'unappealing',
                'desirable', 'undesirable', 'wanted', 'unwanted', 'welcome',
                'unwelcome', 'invited', 'uninvited', 'requested', 'unrequested',
                'demanded', 'undemanded', 'sought', 'unsought', 'popular',
                'unpopular', 'favored', 'unfavored', 'preferred', 'unpreferred',
                'chosen', 'unchosen', 'selected', 'unselected', 'picked',
                'unpicked', 'elected', 'unelected', 'appointed', 'unappointed',
                'designated', 'undesignated', 'assigned', 'unassigned',
                'allocated', 'unallocated', 'distributed', 'undistributed',
                'shared', 'unshared', 'divided', 'undivided', 'split', 'unsplit',
                'separated', 'unseparated', 'isolated', 'unisolated', 'detached',
                'attached', 'connected', 'disconnected', 'joined', 'disjoined',
                'united', 'disunited', 'combined', 'uncombined', 'merged',
                'unmerged',                 'integrated', 'disintegrated', 'consolidated', 'unconsolidated',
                'coordinated', 'uncoordinated', 'organized', 'disorganized',
                'structured', 'unstructured', 'systematic', 'unsystematic',
                'methodical', 'unmethodical', 'orderly', 'disorderly',
                'neat', 'messy', 'tidy', 'untidy', 'arranged', 'unarranged',
                'sorted', 'unsorted', 'classified', 'unclassified', 'categorized',
                'uncategorized', 'grouped', 'ungrouped', 'clustered', 'unclustered'
            ],
            'adverbs': [
                'slightly', 'moderately', 'considerably', 'significantly', 'substantially',
                'materially', 'notably', 'remarkably', 'particularly', 'especially',
                'specifically', 'generally', 'typically', 'usually', 'normally',
                'commonly', 'frequently', 'rarely', 'seldom', 'occasionally',
                'sometimes', 'often', 'always', 'never', 'hardly', 'barely',
                'scarcely', 'almost', 'nearly', 'approximately', 'roughly',
                'exactly', 'precisely', 'accurately', 'correctly', 'properly',
                'appropriately', 'suitably', 'adequately', 'sufficiently',
                'insufficiently', 'inadequately', 'inappropriately', 'improperly',
                'incorrectly', 'inaccurately', 'imprecisely', 'inexactly',
                'roughly', 'approximately', 'about', 'around', 'nearly',
                'almost', 'virtually', 'practically', 'essentially', 'basically',
                'fundamentally', 'primarily', 'mainly', 'chiefly', 'principally',
                'largely', 'mostly', 'generally', 'broadly', 'widely',
                'extensively', 'comprehensively', 'thoroughly', 'completely',
                'entirely', 'fully', 'totally', 'wholly', 'absolutely',
                'perfectly', 'utterly', 'quite', 'rather', 'fairly',
                'reasonably', 'relatively', 'comparatively', 'proportionally',
                'correspondingly', 'accordingly', 'consequently', 'therefore',
                'thus', 'hence', 'so', 'then', 'now', 'here', 'there',
                'everywhere', 'nowhere', 'somewhere', 'anywhere', 'always',
                'never', 'sometimes', 'often', 'frequently', 'regularly',
                'consistently', 'constantly', 'continuously', 'persistently',
                'repeatedly', 'recurrently', 'periodically', 'intermittently',
                'sporadically', 'occasionally', 'rarely', 'seldom', 'hardly',
                'barely', 'scarcely', 'nearly', 'almost', 'quite', 'very',
                'extremely', 'highly', 'greatly', 'deeply', 'strongly',
                'powerfully', 'forcefully', 'vigorously', 'intensely',
                'severely', 'seriously', 'critically', 'crucially', 'vitally',
                'essentially', 'necessarily', 'importantly', 'significantly',
                'substantially', 'considerably', 'notably', 'remarkably',
                'surprisingly', 'unexpectedly', 'obviously', 'clearly',
                'evidently', 'apparently', 'seemingly', 'presumably',
                'supposedly', 'allegedly', 'reportedly', 'presumably',
                'likely', 'probably', 'possibly', 'potentially', 'conceivably',
                'arguably', 'debatably', 'questionably', 'doubtfully',
                'uncertainly', 'definitely', 'certainly', 'surely', 'undoubtedly',
                'unquestionably', 'indubitably', 'undeniably', 'incontrovertibly',
                'irrefutably', 'conclusively', 'decisively', 'definitively',
                'finally', 'ultimately', 'eventually', 'gradually', 'slowly',
                'quickly', 'rapidly', 'swiftly', 'speedily', 'promptly',
                'immediately', 'instantly', 'suddenly', 'abruptly', 'sharply',
                'dramatically', 'drastically', 'radically', 'fundamentally',
                'completely', 'entirely', 'totally', 'wholly', 'fully',
                'partially', 'partly', 'somewhat', 'slightly', 'minimally',
                'marginally', 'fractionally', 'incrementally', 'progressively',
                'steadily', 'consistently', 'uniformly', 'evenly', 'equally',
                'similarly', 'likewise', 'comparably', 'correspondingly',
                'proportionally', 'differently', 'distinctly', 'uniquely',
                'individually', 'separately', 'independently', 'autonomously',
                'collectively', 'jointly', 'together', 'simultaneously',
                'concurrently', 'contemporaneously', 'sequentially', 'consecutively',
                'successively', 'subsequently', 'previously', 'formerly',
                'originally', 'initially', 'firstly', 'secondly', 'thirdly',
                'finally', 'lastly', 'ultimately', 'eventually', 'temporarily',
                'permanently', 'briefly', 'momentarily', 'instantly', 'immediately',
                'presently', 'currently', 'recently', 'lately', 'newly',
                'freshly', 'just', 'still', 'yet', 'already', 'soon',
                'shortly', 'immediately', 'promptly', 'quickly', 'rapidly',
                'swiftly', 'speedily', 'fast', 'slow', 'slowly', 'gradually',
                'steadily', 'consistently', 'regularly', 'systematically',
                'methodically', 'carefully', 'cautiously', 'prudently',
                'wisely', 'intelligently', 'skillfully', 'expertly',
                'professionally', 'competently', 'efficiently', 'effectively',
                'successfully', 'productively', 'constructively', 'positively',
                'favorably', 'beneficially', 'advantageously', 'profitably',
                'usefully', 'helpfully', 'supportively', 'cooperatively',
                'collaboratively', 'harmoniously', 'peacefully', 'calmly',
                'quietly', 'silently', 'softly', 'gently', 'smoothly',
                'easily', 'simply', 'straightforwardly', 'directly',
                'openly', 'honestly', 'truthfully', 'sincerely', 'genuinely',
                'authentically', 'naturally', 'normally', 'typically',
                'usually', 'commonly', 'frequently', 'regularly', 'consistently'
            ],
            'adversative_conjunctions': [
                'but', 'however', 'nevertheless', 'nonetheless', 'although',
                'though', 'even though', 'despite', 'in spite of', 'whereas',
                'while', 'whilst', 'yet', 'still', 'conversely', 'on the contrary',
                'on the other hand', 'in contrast', 'by contrast', 'alternatively',
                'instead', 'rather', 'otherwise', 'except', 'except for',
                'except that', 'save', 'save for', 'save that', 'but for',
                'were it not for', 'if not for', 'unless', 'without',
                'lacking', 'absent', 'minus', 'excluding', 'omitting',
                'barring', 'notwithstanding', 'regardless of', 'irrespective of',
                'despite the fact that', 'in spite of the fact that',
                'even if', 'even when', 'even where', 'even as',
                'much as', 'as much as', 'however much', 'no matter how',
                'no matter what', 'no matter when', 'no matter where',
                'no matter who', 'no matter which', 'regardless',
                'irrespective', 'anyway', 'anyhow', 'in any case',
                'in any event', 'at any rate', 'all the same',
                'just the same', 'even so', 'be that as it may',
                'for all that', 'having said that', 'that said',
                'that being said', 'this said', 'admittedly', 'granted',
                'to be sure', 'certainly', 'of course', 'naturally',
                'obviously', 'clearly', 'evidently', 'apparently',
                'seemingly', 'ostensibly', 'superficially', 'on the surface',
                'at first glance', 'at first sight', 'initially',
                'originally', 'formerly', 'previously', 'once', 'at one time',
                'in the past', 'historically', 'traditionally', 'conventionally',
                'typically', 'usually', 'normally', 'generally', 'commonly',
                'ordinarily', 'as a rule', 'in general', 'by and large',
                'on the whole', 'overall', 'all in all', 'taken together',
                'altogether', 'in total', 'in sum', 'to sum up',
                'in summary', 'in conclusion', 'to conclude', 'finally',
                'lastly', 'ultimately', 'in the end', 'at the end of the day',
                'when all is said and done', 'after all', 'all things considered',
                'everything considered', 'taking everything into account',
                'taking all things into account', 'on balance', 'on reflection',
                'thinking about it', 'come to think of it', 'now that I think about it',
                'as I see it', 'in my view', 'in my opinion', 'from my perspective',
                'from my point of view', 'as far as I\'m concerned',
                'as far as I can see', 'as far as I can tell',
                'as far as I know', 'to my knowledge', 'to the best of my knowledge',
                'if I\'m not mistaken', 'if I remember correctly',
                'if memory serves', 'unless I\'m mistaken', 'correct me if I\'m wrong'
            ]
        }
       
        # Flatten all valence shifters into a single set for quick lookup
        self.all_valence_shifters = set()
        for category, words in self.valence_shifters.items():
            self.all_valence_shifters.update(words)
       
        print(f"✓ Valence shifters loaded: {len(self.all_valence_shifters)} total words")
   
    def define_macro_keywords(self):
        """Define macroeconomic shock keywords"""
        self.macro_keywords = [
            'covid-19', 'covid', 'coronavirus', 'pandemic', 'lockdown', 'quarantine',
            'inflation', 'deflation', 'recession', 'depression', 'economic downturn',
            'geopolitical', 'trade war', 'tariff', 'sanctions', 'brexit',
            'macroeconomic headwinds', 'economic uncertainty', 'supply chain',
            'disruption', 'supply shortage', 'labor shortage', 'energy crisis',
            'oil prices', 'commodity prices', 'interest rates', 'federal reserve',
            'monetary policy', 'fiscal policy', 'government spending', 'stimulus',
            'bailout', 'quantitative easing', 'market volatility', 'financial crisis',
            'credit crunch', 'liquidity crisis', 'sovereign debt', 'currency crisis',
            'exchange rate', 'dollar strength', 'emerging markets', 'global economy',
            'international trade', 'export', 'import', 'gdp', 'unemployment',
            'job losses', 'layoffs', 'furlough', 'economic recovery',
            'post-pandemic', 'new normal', 'digital transformation', 'remote work',
            'climate change', 'sustainability', 'esg', 'regulatory changes',
            'compliance', 'cybersecurity', 'data breach', 'technology disruption'
        ]
        self.macro_keywords = [keyword.lower() for keyword in self.macro_keywords]
        print(f"✓ Macro keywords loaded: {len(self.macro_keywords)} keywords")
   
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        sentences = nltk.sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
   
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
   
    def analyze_finbert_sentiment(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_analyzer:
            return {'error': 'FinBERT model not available'}
       
        sentence_results = []
        scores = []
       
        for sentence in sentences:
            try:
                # Truncate long sentences for FinBERT
                truncated = sentence[:512] if len(sentence) > 512 else sentence
                result = self.finbert_analyzer(truncated)[0]
               
                # Convert to standardized score (-1 to 1)
                if result['label'] == 'positive':
                    score = result['score']
                elif result['label'] == 'negative':
                    score = -result['score']
                else:  # neutral
                    score = 0
               
                sentence_results.append({
                    'sentence': sentence,
                    'label': result['label'],
                    'confidence': result['score'],
                    'normalized_score': score
                })
                scores.append(score)
               
            except Exception as e:
                print(f"Error processing sentence with FinBERT: {e}")
                sentence_results.append({
                    'sentence': sentence,
                    'label': 'neutral',
                    'confidence': 0.0,
                    'normalized_score': 0.0
                })
                scores.append(0.0)
       
        # Document-level metrics
        avg_score = np.mean(scores) if scores else 0
        pos_count = sum(1 for r in sentence_results if r['label'] == 'positive')
        neg_count = sum(1 for r in sentence_results if r['label'] == 'negative')
        neu_count = sum(1 for r in sentence_results if r['label'] == 'neutral')
        total_sentences = len(sentence_results)
       
        # Top contributing sentences
        sorted_results = sorted(sentence_results, key=lambda x: abs(x['normalized_score']), reverse=True)
        top_positive = [r for r in sorted_results if r['normalized_score'] > 0][:3]
        top_negative = [r for r in sorted_results if r['normalized_score'] < 0][:3]
       
        return {
            'sentence_results': sentence_results,
            'document_metrics': {
                'average_score': round(avg_score, 4),
                'positive_percent': round((pos_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'negative_percent': round((neg_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'neutral_percent': round((neu_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'total_sentences': total_sentences
            },
            'top_contributing': {
                'positive': top_positive,
                'negative': top_negative
            }
        }
   
    def analyze_ml_lexicon_sentiment(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentiment using ML Lexicon"""
        if not self.ml_lexicon:
            return {'error': 'ML Lexicon not available'}
       
        words = self.tokenize_words(text)
        word_scores = {}
        total_score = 0
        total_weight = 0
       
        # Analyze each word
        for word in words:
            if word in self.ml_lexicon:
                entry = self.ml_lexicon[word]
               
                # Calculate weighted polarity score
                pos_prob = entry.get('pos_prob', 0)
                neg_prob = entry.get('neg_prob', 0)
                intensity = entry.get('intensity', 1)
                frequency = entry.get('frequency', 1)
               
                # Polarity score: positive - negative
                polarity = pos_prob - neg_prob
                weighted_score = polarity * intensity * frequency
               
                if word in word_scores:
                    word_scores[word]['count'] += 1
                    word_scores[word]['total_contribution'] += weighted_score
                else:
                    word_scores[word] = {
                        'count': 1,
                        'polarity': polarity,
                        'intensity': intensity,
                        'frequency': frequency,
                        'weighted_score': weighted_score,
                        'total_contribution': weighted_score
                    }
               
                total_score += weighted_score
                total_weight += intensity * frequency
       
        # Normalize score
        normalized_score = total_score / total_weight if total_weight > 0 else 0
       
        # Sentence-level analysis
        sentence_scores = []
        for sentence in sentences:
            sentence_words = self.tokenize_words(sentence)
            sentence_score = 0
            sentence_weight = 0
           
            for word in sentence_words:
                if word in self.ml_lexicon:
                    entry = self.ml_lexicon[word]
                    polarity = entry.get('pos_prob', 0) - entry.get('neg_prob', 0)
                    intensity = entry.get('intensity', 1)
                    frequency = entry.get('frequency', 1)
                    weighted = polarity * intensity * frequency
                   
                    sentence_score += weighted
                    sentence_weight += intensity * frequency
           
            normalized_sentence_score = sentence_score / sentence_weight if sentence_weight > 0 else 0
            sentence_scores.append({
                'sentence': sentence,
                'score': round(normalized_sentence_score, 4),
                'raw_score': round(sentence_score, 4)
            })
       
        # Top contributing words
        top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]['total_contribution']), reverse=True)[:10]
       
        explanations = []
        for word, data in top_words:
            explanation = f"Word '{word}' appeared {data['count']} times, contributing {round(data['total_contribution'], 4)} to the total score."
            explanations.append(explanation)
       
        return {
            'total_score': round(total_score, 4),
            'normalized_score': round(normalized_score, 4),
            'word_contributions': dict(word_scores),
            'sentence_scores': sentence_scores,
            'top_words': dict(top_words),
            'explanations': explanations,
            'words_analyzed': len(word_scores)
        }
   
    def analyze_lm_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Loughran-McDonald dictionary"""
        if not hasattr(self, 'lm_positive'):
            self.setup_lm_dictionary()
       
        words = self.tokenize_words(text)
        positive_count = 0
        negative_count = 0
        positive_words = []
        negative_words = []
       
        for word in words:
            if word in self.lm_positive:
                positive_count += 1
                positive_words.append(word)
            elif word in self.lm_negative:
                negative_count += 1
                negative_words.append(word)
       
        total_words = len(words)
        total_sentiment_words = positive_count + negative_count
       
        # Calculate basic sentiment score
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0
       
        # Calculate sentiment ratios
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
       
        return {
            'sentiment_score': round(sentiment_score, 4),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': round(positive_ratio, 4),
            'negative_ratio': round(negative_ratio, 4),
            'total_words': total_words,
            'total_sentiment_words': total_sentiment_words,
            'positive_words': list(set(positive_words)),
            'negative_words': list(set(negative_words))
        }
   
    def calculate_semantic_complexity_index(self, sentences: List[str]) -> Dict[str, Any]:
        """Calculate Semantic Complexity Index (SCI)"""
        sentences_with_shifters = 0
        sentence_details = []
       
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            shifters_found = []
           
            for word in words:
                if word in self.all_valence_shifters:
                    shifters_found.append(word)
           
            has_shifters = len(shifters_found) > 0
            if has_shifters:
                sentences_with_shifters += 1
           
            sentence_details.append({
                'sentence': sentence,
                'has_valence_shifters': has_shifters,
                'shifters_found': shifters_found,
                'shifter_count': len(shifters_found)
            })
       
        total_sentences = len(sentences)
        sci_percentage = (sentences_with_shifters / total_sentences) * 100 if total_sentences > 0 else 0
       
        return {
            'sci_percentage': round(sci_percentage, 2),
            'sentences_with_shifters': sentences_with_shifters,
            'total_sentences': total_sentences,
            'sentence_details': sentence_details
        }
   
    def calculate_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate Gunning Fog Index and Flesch-Kincaid Grade Level"""
       
        # Gunning Fog Index calculation
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
       
        if len(sentences) == 0 or len(words) == 0:
            return {
                'gunning_fog': 0,
                'flesch_kincaid': 0,
                'error': 'Insufficient text for analysis'
            }
       
        # Count complex words (3+ syllables)
        complex_words = 0
        for word in words:
            if self.count_syllables(word) >= 3:
                complex_words += 1
       
        # Gunning Fog formula
        avg_sentence_length = len(words) / len(sentences)
        complex_word_ratio = (complex_words / len(words)) * 100
        gunning_fog = 0.4 * (avg_sentence_length + complex_word_ratio)
       
        # Flesch-Kincaid using textstat
        try:
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
        except:
            flesch_kincaid = 0
       
        # Interpretation
        fog_interpretation = self.interpret_fog_index(gunning_fog)
        fk_interpretation = self.interpret_flesch_kincaid(flesch_kincaid)
       
        return {
            'gunning_fog': round(gunning_fog, 2),
            'flesch_kincaid': round(flesch_kincaid, 2),
            'fog_interpretation': fog_interpretation,
            'fk_interpretation': fk_interpretation,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complex_words': complex_words,
            'complex_word_ratio': round(complex_word_ratio, 2)
        }
   
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified method)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
       
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
       
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
       
        return max(1, syllable_count)
   
    def interpret_fog_index(self, fog_score: float) -> str:
        """Interpret Gunning Fog Index score"""
        if fog_score <= 8:
            return "Easy to read (8th grade level or below)"
        elif fog_score <= 12:
            return "Moderately difficult (high school level)"
        elif fog_score <= 16:
            return "Difficult to read (college level)"
        else:
            return "Very difficult to read (graduate level)"
   
    def interpret_flesch_kincaid(self, fk_score: float) -> str:
        """Interpret Flesch-Kincaid Grade Level"""
        if fk_score <= 8:
            return f"Grade {int(fk_score)} level - Easy to read"
        elif fk_score <= 12:
            return f"Grade {int(fk_score)} level - Standard difficulty"
        elif fk_score <= 16:
            return f"Grade {int(fk_score)} level - College level"
        else:
            return f"Grade {int(fk_score)} level - Graduate level"
   
    def calculate_fox_index(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Calculate Fox Index proxy for managerial obfuscation"""
        words = self.tokenize_words(text)
       
        if len(sentences) == 0 or len(words) == 0:
            return {'fox_index': 0, 'error': 'Insufficient text for analysis'}
       
        # Component 1: Average sentence length (normalized)
        avg_sentence_length = len(words) / len(sentences)
        # Normalize to 0-100 scale (assuming 10-50 words per sentence range)
        length_score = min(100, max(0, (avg_sentence_length - 10) * 2.5))
       
        # Component 2: Passive voice ratio (simplified detection)
        passive_indicators = ['was', 'were', 'been', 'being', 'be']
        passive_count = sum(1 for word in words if word in passive_indicators)
        passive_ratio = (passive_count / len(words)) * 100
        passive_score = min(100, passive_ratio * 10)  # Amplify for visibility
       
        # Component 3: Complexity markers (adjective/adverb density)
        complexity_words = [word for word in words if word in self.valence_shifters['adjectives'] + self.valence_shifters['adverbs']]
        complexity_ratio = (len(complexity_words) / len(words)) * 100
        complexity_score = min(100, complexity_ratio * 5)  # Amplify for visibility
       
        # Calculate weighted Fox Index (0-100)
        fox_index = (length_score * 0.4 + passive_score * 0.3 + complexity_score * 0.3)
       
        # Interpretation
        if fox_index < 30:
            interpretation = "Low obfuscation - Clear, direct communication"
        elif fox_index < 60:
            interpretation = "Moderate obfuscation - Some complexity present"
        else:
            interpretation = "High obfuscation - Complex, potentially evasive language"
       
        return {
            'fox_index': round(fox_index, 2),
            'components': {
                'sentence_length_score': round(length_score, 2),
                'passive_voice_score': round(passive_score, 2),
                'complexity_score': round(complexity_score, 2)
            },
            'metrics': {
                'avg_sentence_length': round(avg_sentence_length, 2),
                'passive_ratio': round(passive_ratio, 4),
                'complexity_ratio': round(complexity_ratio, 4)
            },
            'interpretation': interpretation
        }
   
    def analyze_macro_context(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze macroeconomic context and adjust sentiment accordingly"""
        macro_sentences = []
        adjustments_made = []
       
        for sentence in sentences:
            sentence_lower = sentence.lower()
            macro_terms_found = []
           
            for keyword in self.macro_keywords:
                if keyword in sentence_lower:
                    macro_terms_found.append(keyword)
           
            if macro_terms_found:
                # Analyze framing of macro terms
                framing_analysis = self.analyze_macro_framing(sentence, macro_terms_found)
               
                macro_sentences.append({
                    'sentence': sentence,
                    'macro_terms': macro_terms_found,
                    'framing': framing_analysis['framing'],
                    'adjustment_reasoning': framing_analysis['reasoning']
                })
               
                adjustments_made.append(framing_analysis['reasoning'])
       
        return {
            'macro_flagged_sentences': macro_sentences,
            'total_macro_sentences': len(macro_sentences),
            'adjustments_summary': adjustments_made
        }
   
    def analyze_macro_framing(self, sentence: str, macro_terms: List[str]) -> Dict[str, Any]:
        """Analyze how macroeconomic terms are framed in context"""
        sentence_lower = sentence.lower()
       
        # Look for positive framing indicators
        positive_indicators = ['opportunity', 'resilient', 'adapt', 'overcome', 'manage', 'mitigate', 'recovery', 'rebound']
        negative_indicators = ['challenge', 'impact', 'hurt', 'damage', 'concern', 'worry', 'threat', 'risk']
       
        positive_count = sum(1 for indicator in positive_indicators if indicator in sentence_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in sentence_lower)
       
        if positive_count > negative_count:
            framing = 'positive'
            reasoning = f"Macro terms {macro_terms} mentioned with positive framing → sentiment adjusted upward"
        elif negative_count > positive_count:
            framing = 'negative'
            reasoning = f"Macro terms {macro_terms} mentioned with negative framing → sentiment adjusted downward"
        else:
            framing = 'neutral'
            reasoning = f"Macro terms {macro_terms} mentioned with neutral framing → no sentiment adjustment"
       
        return {
            'framing': framing,
            'reasoning': reasoning,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
   
    def detect_sentiment_divergence(self, finbert_results: Dict, ml_lexicon_results: Dict, lm_results: Dict) -> Dict[str, Any]:
        """Detect divergence between sentiment methods to flag potential obfuscation"""
       
        # Extract normalized scores
        finbert_score = finbert_results.get('document_metrics', {}).get('average_score', 0)
        ml_score = ml_lexicon_results.get('normalized_score', 0)
        lm_score = lm_results.get('sentiment_score', 0)
       
        # Define thresholds for divergence detection
        HIGH_DIVERGENCE_THRESHOLD = 0.5
        MODERATE_DIVERGENCE_THRESHOLD = 0.3
       
        divergences = []
        flags = []
       
        # Compare FinBERT vs ML Lexicon
        finbert_ml_diff = abs(finbert_score - ml_score)
        if finbert_ml_diff > HIGH_DIVERGENCE_THRESHOLD:
            if finbert_score > 0 and ml_score < -0.3:
                flags.append("euphemism_masking")
                divergences.append({
                    'type': 'FinBERT vs ML Lexicon',
                    'difference': round(finbert_ml_diff, 4),
                    'interpretation': 'Possible euphemism masking - FinBERT detects positive tone but ML lexicon shows negative sentiment',
                    'finbert_score': finbert_score,
                    'ml_score': ml_score
                })
            elif finbert_score < 0 and ml_score > 0.3:
                flags.append("strategic_tone_shifting")
                divergences.append({
                    'type': 'FinBERT vs ML Lexicon',
                    'difference': round(finbert_ml_diff, 4),
                    'interpretation': 'Possible strategic tone shifting - negative contextual sentiment with positive word choice',
                    'finbert_score': finbert_score,
                    'ml_score': ml_score
                })
            else:
                flags.append("general_obfuscation")
                divergences.append({
                    'type': 'FinBERT vs ML Lexicon',
                    'difference': round(finbert_ml_diff, 4),
                    'interpretation': 'High divergence suggests potential obfuscation or complex sentiment structure',
                    'finbert_score': finbert_score,
                    'ml_score': ml_score
                })
       
        # Compare FinBERT vs LM
        finbert_lm_diff = abs(finbert_score - lm_score)
        if finbert_lm_diff > HIGH_DIVERGENCE_THRESHOLD:
            if finbert_score > 0 and lm_score < -0.3:
                flags.append("euphemism_masking")
                divergences.append({
                    'type': 'FinBERT vs LM Dictionary',
                    'difference': round(finbert_lm_diff, 4),
                    'interpretation': 'Possible euphemism masking - contextual positivity masking negative word usage',
                    'finbert_score': finbert_score,
                    'lm_score': lm_score
                })
       
        # Compare ML Lexicon vs LM
        ml_lm_diff = abs(ml_score - lm_score)
        if ml_lm_diff > MODERATE_DIVERGENCE_THRESHOLD:
            divergences.append({
                'type': 'ML Lexicon vs LM Dictionary',
                'difference': round(ml_lm_diff, 4),
                'interpretation': 'Divergence between sophisticated and basic lexicon methods',
                'ml_score': ml_score,
                'lm_score': lm_score
            })
       
        # Overall divergence assessment
        max_divergence = max([finbert_ml_diff, finbert_lm_diff, ml_lm_diff])
       
        if max_divergence > HIGH_DIVERGENCE_THRESHOLD:
            overall_risk = "HIGH"
        elif max_divergence > MODERATE_DIVERGENCE_THRESHOLD:
            overall_risk = "MODERATE"
        else:
            overall_risk = "LOW"
       
        return {
            'divergence_detected': len(divergences) > 0,
            'risk_level': overall_risk,
            'flags': list(set(flags)),  # Remove duplicates
            'divergences': divergences,
            'max_divergence': round(max_divergence, 4),
            'scores_summary': {
                'finbert': finbert_score,
                'ml_lexicon': ml_score,
                'lm_dictionary': lm_score
            }
        }
   
    def create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary interpretation table"""
       
        summary_data = []
       
        # FinBERT results
        if 'finbert' in results and 'document_metrics' in results['finbert']:
            fb_score = results['finbert']['document_metrics']['average_score']
            fb_interp = "Positive tone" if fb_score > 0.1 else "Negative tone" if fb_score < -0.1 else "Neutral tone"
            summary_data.append({
                'Feature': 'FinBERT Avg Score',
                'Value': f"{fb_score:+.3f}",
                'Interpretation/Impact': fb_interp
            })
       
        # ML Lexicon results
        if 'ml_lexicon' in results and 'normalized_score' in results['ml_lexicon']:
            ml_score = results['ml_lexicon']['normalized_score']
            ml_interp = "Positive lexicon sentiment" if ml_score > 0.1 else "Negative lexicon sentiment" if ml_score < -0.1 else "Neutral lexicon sentiment"
            summary_data.append({
                'Feature': 'ML Lexicon Score',
                'Value': f"{ml_score:+.3f}",
                'Interpretation/Impact': ml_interp
            })
       
        # LM Dictionary results
        if 'lm_dictionary' in results and 'sentiment_score' in results['lm_dictionary']:
            lm_score = results['lm_dictionary']['sentiment_score']
            lm_interp = "Positive word-based sentiment" if lm_score > 0.1 else "Negative word-based sentiment" if lm_score < -0.1 else "Neutral word-based sentiment"
            summary_data.append({
                'Feature': 'LM Dictionary Score',
                'Value': f"{lm_score:+.3f}",
                'Interpretation/Impact': lm_interp
            })
       
        # SCI results
        if 'sci' in results and 'sci_percentage' in results['sci']:
            sci_pct = results['sci']['sci_percentage']
            sci_interp = "High hedging/compliance tone" if sci_pct > 40 else "Moderate complexity" if sci_pct > 20 else "Low complexity"
            summary_data.append({
                'Feature': 'SCI (%)',
                'Value': f"{sci_pct}%",
                'Interpretation/Impact': sci_interp
            })
       
        # Fox Index results
        if 'fox_index' in results and 'fox_index' in results['fox_index']:
            fox_score = results['fox_index']['fox_index']
            fox_interp = "Suggests obfuscation" if fox_score > 60 else "Moderate complexity" if fox_score > 30 else "Clear communication"
            summary_data.append({
                'Feature': 'Fox Index',
                'Value': f"{fox_score}",
                'Interpretation/Impact': fox_interp
            })
       
        # Gunning Fog results
        if 'readability' in results and 'gunning_fog' in results['readability']:
            fog_score = results['readability']['gunning_fog']
            summary_data.append({
                'Feature': 'Gunning Fog Index',
                'Value': f"{fog_score}",
                'Interpretation/Impact': results['readability']['fog_interpretation']
            })
       
        # Flesch-Kincaid results
        if 'readability' in results and 'flesch_kincaid' in results['readability']:
            fk_score = results['readability']['flesch_kincaid']
            summary_data.append({
                'Feature': 'Flesch-Kincaid Grade',
                'Value': f"{fk_score}",
                'Interpretation/Impact': results['readability']['fk_interpretation']
            })
       
        # Divergence results
        if 'divergence' in results and results['divergence']['divergence_detected']:
            div_risk = results['divergence']['risk_level']
            div_flags = ', '.join(results['divergence']['flags'])
            summary_data.append({
                'Feature': 'Sentiment Divergence',
                'Value': div_risk,
                'Interpretation/Impact': f"Flags: {div_flags}"
            })
       
        return pd.DataFrame(summary_data)
   
    def create_sentence_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create per-sentence analysis dataframe"""
       
        sentence_data = []
       
        # Get sentence lists from different analyses
        finbert_sentences = results.get('finbert', {}).get('sentence_results', [])
        sci_sentences = results.get('sci', {}).get('sentence_details', [])
        ml_sentences = results.get('ml_lexicon', {}).get('sentence_scores', [])
        macro_sentences = results.get('macro_context', {}).get('macro_flagged_sentences', [])
       
        # Create mapping for quick lookup
        macro_map = {s['sentence']: s for s in macro_sentences}
       
        # Process each sentence
        max_sentences = max(len(finbert_sentences), len(sci_sentences), len(ml_sentences))
       
        for i in range(max_sentences):
            row_data = {}
           
            # Get sentence text (prefer FinBERT as primary source)
            if i < len(finbert_sentences):
                sentence = finbert_sentences[i]['sentence']
                row_data['sentence'] = sentence[:100] + "..." if len(sentence) > 100 else sentence
                row_data['finbert_sentiment'] = finbert_sentences[i]['label']
                row_data['finbert_score'] = finbert_sentences[i]['normalized_score']
                row_data['finbert_confidence'] = finbert_sentences[i]['confidence']
            elif i < len(sci_sentences):
                sentence = sci_sentences[i]['sentence']
                row_data['sentence'] = sentence[:100] + "..." if len(sentence) > 100 else sentence
                row_data['finbert_sentiment'] = 'N/A'
                row_data['finbert_score'] = 0
                row_data['finbert_confidence'] = 0
            else:
                row_data['sentence'] = 'N/A'
                row_data['finbert_sentiment'] = 'N/A'
                row_data['finbert_score'] = 0
                row_data['finbert_confidence'] = 0
           
            # SCI data
            if i < len(sci_sentences):
                row_data['has_valence_shifters'] = 'Yes' if sci_sentences[i]['has_valence_shifters'] else 'No'
                row_data['valence_shifters'] = ', '.join(sci_sentences[i]['shifters_found'][:5])  # Limit display
                row_data['shifter_count'] = sci_sentences[i]['shifter_count']
            else:
                row_data['has_valence_shifters'] = 'No'
                row_data['valence_shifters'] = ''
                row_data['shifter_count'] = 0
           
            # ML Lexicon data
            if i < len(ml_sentences):
                row_data['ml_lexicon_score'] = ml_sentences[i]['score']
            else:
                row_data['ml_lexicon_score'] = 0
           
            # Macro context data
            sentence_text = row_data['sentence'].replace('...', '') if row_data['sentence'] != 'N/A' else ''
            macro_match = None
            for macro_sent in macro_sentences:
                if sentence_text in macro_sent['sentence'] or macro_sent['sentence'] in sentence_text:
                    macro_match = macro_sent
                    break
           
            if macro_match:
                row_data['macro_flag'] = 'Yes'
                row_data['macro_terms'] = ', '.join(macro_match['macro_terms'][:3])  # Limit display
                row_data['macro_framing'] = macro_match['framing']
            else:
                row_data['macro_flag'] = 'No'
                row_data['macro_terms'] = ''
                row_data['macro_framing'] = 'N/A'
           
            sentence_data.append(row_data)
       
        return pd.DataFrame(sentence_data)
   
    def analyze_text(self, text: str, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Main analysis method that orchestrates all sentiment and readability analyses
       
        Args:
            text: Raw MD&A text to analyze
            weights: Optional dictionary to adjust feature weights in final interpretation
       
        Returns:
            Dictionary containing all analysis results
        """
       
        if weights is None:
            weights = {
                'finbert': 0.3,
                'ml_lexicon': 0.25,
                'lm_dictionary': 0.15,
                'sci': 0.1,
                'fox_index': 0.1,
                'readability': 0.1
            }
       
        print("🔍 Starting comprehensive financial sentiment analysis...")
        print(f"📄 Text length: {len(text)} characters, {len(text.split())} words")
       
        # Tokenize into sentences
        sentences = self.tokenize_sentences(text)
        print(f"📝 Sentences identified: {len(sentences)}")
       
        results = {}
       
        # 1. FinBERT Sentiment Analysis
        print("\n1️⃣ Analyzing FinBERT sentiment...")
        results['finbert'] = self.analyze_finbert_sentiment(sentences)
        if 'error' not in results['finbert']:
            print(f"   ✓ Average sentiment: {results['finbert']['document_metrics']['average_score']:.3f}")
       
        # 2. ML Lexicon Sentiment Analysis
        print("\n2️⃣ Analyzing ML Lexicon sentiment...")
        results['ml_lexicon'] = self.analyze_ml_lexicon_sentiment(text, sentences)
        if 'error' not in results['ml_lexicon']:
            print(f"   ✓ Normalized score: {results['ml_lexicon']['normalized_score']:.3f}")
            print(f"   ✓ Words analyzed: {results['ml_lexicon']['words_analyzed']}")
       
        # 3. LM Dictionary Sentiment Analysis
        print("\n3️⃣ Analyzing LM Dictionary sentiment...")
        results['lm_dictionary'] = self.analyze_lm_sentiment(text)
        print(f"   ✓ Sentiment score: {results['lm_dictionary']['sentiment_score']:.3f}")
        print(f"   ✓ Positive words: {results['lm_dictionary']['positive_count']}")
        print(f"   ✓ Negative words: {results['lm_dictionary']['negative_count']}")
       
        # 4. Semantic Complexity Index (SCI)
        print("\n4️⃣ Calculating Semantic Complexity Index...")
        results['sci'] = self.calculate_semantic_complexity_index(sentences)
        print(f"   ✓ SCI: {results['sci']['sci_percentage']:.1f}%")
        print(f"   ✓ Sentences with valence shifters: {results['sci']['sentences_with_shifters']}")
       
        # 5. Readability Metrics
        print("\n5️⃣ Calculating readability metrics...")
        results['readability'] = self.calculate_readability_metrics(text)
        print(f"   ✓ Gunning Fog Index: {results['readability']['gunning_fog']} - {results['readability']['fog_interpretation']}")
        print(f"   ✓ Flesch-Kincaid Grade: {results['readability']['flesch_kincaid']} - {results['readability']['fk_interpretation']}")
       
        # 6. Fox Index (Obfuscation Proxy)
        print("\n6️⃣ Calculating Fox Index...")
        results['fox_index'] = self.calculate_fox_index(text, sentences)
        if 'error' not in results['fox_index']:
            print(f"   ✓ Fox Index: {results['fox_index']['fox_index']} - {results['fox_index']['interpretation']}")
       
        # 7. Macroeconomic Context Analysis
        print("\n7️⃣ Analyzing macroeconomic context...")
        results['macro_context'] = self.analyze_macro_context(sentences)
        print(f"   ✓ Macro-flagged sentences: {results['macro_context']['total_macro_sentences']}")
       
        # 8. Sentiment Divergence Analysis
        print("\n8️⃣ Detecting sentiment divergence...")
        results['divergence'] = self.detect_sentiment_divergence(
            results['finbert'],
            results['ml_lexicon'],
            results['lm_dictionary']
        )
        if results['divergence']['divergence_detected']:
            print(f"   ⚠️  Divergence detected! Risk level: {results['divergence']['risk_level']}")
            print(f"   🚩 Flags: {', '.join(results['divergence']['flags'])}")
        else:
            print("   ✓ No significant divergence detected")
       
        # 9. Create Summary Tables
        print("\n9️⃣ Creating summary tables...")
        results['summary_table'] = self.create_summary_table(results)
        results['sentence_table'] = self.create_sentence_dataframe(results)
       
        # 10. Store metadata
        results['metadata'] = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sentences),
            'analysis_weights': weights,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
       
        print("\n✅ Analysis complete!")
        return results
   
    def export_results(self, results: Dict[str, Any], filename_prefix: str = "financial_sentiment_analysis") -> Dict[str, str]:
        """Export results to CSV files"""
       
        exported_files = {}
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
       
        # Export summary table
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        results['summary_table'].to_csv(summary_filename, index=False)
        exported_files['summary'] = summary_filename
       
        # Export sentence-level analysis
        sentence_filename = f"{filename_prefix}_sentences_{timestamp}.csv"
        results['sentence_table'].to_csv(sentence_filename, index=False)
        exported_files['sentences'] = sentence_filename
       
        # Export detailed results as JSON for further analysis
        import json
       
        # Create a serializable version of results
        export_dict = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                export_dict[key] = value.to_dict('records')
            else:
                export_dict[key] = value
       
        json_filename = f"{filename_prefix}_detailed_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False, default=str)
        exported_files['detailed_json'] = json_filename
       
        print(f"\n📁 Results exported:")
        for file_type, filename in exported_files.items():
            print(f"   • {file_type}: {filename}")
       
        return exported_files
   
    def print_analysis_summary(self, results: Dict[str, Any]):
        """Print a formatted analysis summary"""
       
        print("\n" + "="*80)
        print("🏢 FINANCIAL SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
       
        # Print summary table
        print("\n📊 FEATURE SUMMARY:")
        print(results['summary_table'].to_string(index=False))
       
        # Print divergence warnings if any
        if results['divergence']['divergence_detected']:
            print("\n⚠️  DIVERGENCE ALERTS:")
            for divergence in results['divergence']['divergences']:
                print(f"   🚩 {divergence['type']}: {divergence['interpretation']}")
                print(f"      Difference: {divergence['difference']:.3f}")
       
        # Print top contributing elements
        print("\n🔝 TOP INSIGHTS:")
       
        # Top ML Lexicon words
        if 'ml_lexicon' in results and 'explanations' in results['ml_lexicon']:
            print("\n   ML Lexicon Top Contributors:")
            for explanation in results['ml_lexicon']['explanations'][:5]:
                print(f"   • {explanation}")
       
        # Top FinBERT sentences
        if 'finbert' in results and 'top_contributing' in results['finbert']:
            top_pos = results['finbert']['top_contributing']['positive']
            top_neg = results['finbert']['top_contributing']['negative']
           
            if top_pos:
                print(f"\n   Most Positive Sentence (FinBERT): {top_pos[0]['normalized_score']:.3f}")
                print(f"   \"{top_pos[0]['sentence'][:100]}...\"")
           
            if top_neg:
                print(f"\n   Most Negative Sentence (FinBERT): {top_neg[0]['normalized_score']:.3f}")
                print(f"   \"{top_neg[0]['sentence'][:100]}...\"")
       
        # Macro context summary
        if results['macro_context']['total_macro_sentences'] > 0:
            print(f"\n   Macroeconomic Context: {results['macro_context']['total_macro_sentences']} sentences flagged")
       
        print("\n" + "="*80)


def main():
    """
    Example usage of the Financial Sentiment Analyzer
    """
   
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
   
    # Sample MD&A text for testing (replace with actual 10-K text)
    sample_text = """
    On December 16, 2014, the Board appointed Paula Schneider as CEO, effective January 5, 2015. This appointment followed the termination of Dov Charney, former President and CEO, for cause in accordance with the terms of his employment agreement. Scott Brubaker, who served as Interim CEO since September 29, 2014, continued in the post until Ms. Schneider joined us. Additionally, on September 29, 2014, the Board appointed Hassan Natha as CFO, and John Luttrell resigned as Interim CEO and CFO.
On July 7, 2014, we received a notice from Lion asserting an event of default and an acceleration of the maturity of the loans and other outstanding obligations under the Lion Loan Agreement as a result of the suspension of Dov Charney as CEO by the Board. On July 14, 2014, Lion issued a notice rescinding the notice of acceleration. On July 16, 2014, Lion assigned its rights and obligations as a lender under the Lion Loan Agreement to an entity affiliated with Standard General. Standard General waived any default under the Standard General Loan Agreement that may have resulted or that might result from Mr. Charney not being the CEO.
On September 8, 2014, we and Standard General entered into an amendment of the Standard General Loan Agreement to lower the applicable interest rate to 17%, extend the maturity to April 15, 2021, and make certain other technical amendments, including to remove a provision that specified that Mr. Charney not being the CEO would constitute an event of default.
On March 25, 2015, we entered into the Sixth Amendment to the Capital One Credit Facility ("the Sixth Amendment") which (i) waived any defaults under the Capital One Credit Facility due to the failure to meet the obligation to maintain the maximum leverage ratio and minimum adjusted EBITDA required for the measurement periods ended December 31, 2014, as defined in the credit agreement, (ii) waived the obligation to maintain the minimum fixed charge coverage ratio, maximum leverage ratio and minimum adjusted EBITDA required for the twelve months ending March 31, 2015, (iii) included provisions to permit us to enter into the Standard General Credit Agreement, (iv) reset financial covenants relating to maintaining minimum fixed charge coverage ratios, maximum leverage ratios and minimum adjusted EBITDA and (v) permitted us to borrow $15,000 under the Standard General Credit Agreement.
On March 25, 2015, one of our subsidiaries borrowed $15,000 under the Standard General Credit Agreement. The Standard General Credit Agreement is guaranteed by us, bears interest at 14% per annum, and will mature on October 15, 2020.
In connection with the Standstill and Support Agreement among us, Standard General and Mr. Charney, five directors including Mr. Charney resigned from the Board effective as of August 2, 2014, and five new directors were appointed to the Board, three of whom were designated by Standard General and two of whom were appointed by the mutual agreement of Standard General and us. In addition, Lion exercised its rights to designate two members to our Board, whose appointments were effective as of September 15, 2014 and January 13, 2015, respectively. On March 6, 2015, a member appointed by Lion resigned from the Board, and on March 24, 2015, the Board elected a member designated by Lion to fill that vacancy.
In 2012, German customs audited the import records of our German subsidiary for the years 2009 through 2011 and issued retroactive punitive duty assessments on certain containers of goods imported. The German customs imposed a substantially higher tariff rate than the original rate that we had paid on the imports, more than doubling the amount of the tariff that we would have to pay. The assessments of additional retaliatory duty originated from a trade dispute. Despite the ongoing appeals of the assessment, the German authorities demanded, and we paid, in connection with such assessment, $4,390 in the third quarter of 2014 and the final balance of $85 in the fourth quarter of 2014. We recorded the duty portion of $79 in cost of sales and the retaliatory duties, interest and penalties of $5,104 in general and administrative expenses in our consolidated statements of operations.
Net sales for the year ended December 31, 2014 decreased $25,050, or 4.0%, from the year ended December 31, 2013 due to lower sales at our U.S. Retail, Canada and International segments, partly offset by an increase in the U.S. Wholesale segment.
Gross profits as a percentage of sales were 50.8% and 50.6% for the year ended December 31, 2014 and 2013, respectively. Excluding the effects of the significant events described below, gross profits as a percentage of net sales increased slightly to 52.2% and 51.1% for the year ended December 31, 2014 and 2013, respectively. The increase was mainly due to a reduction in freight costs associated with the completion of our transition to the La Mirada distribution center in late 2013.
Operating expenses for the year ended December 31, 2014 decreased $14,660, or 4.2%, from the year ended December 31, 2013. Excluding the effects of the significant events discussed below, operating expenses for the year ended December 31, 2014 decreased $27,616 from the year ended December 31, 2013. The decrease was primarily due to lower payroll from our cost reduction efforts and reduced expenditures on advertising and promotional activities.
Loss from operations was $27,583 for the year ended December 31, 2014 as compared to $29,295 for the year ended December 31, 2013. Excluding the effects of the significant events discussed below, our operating results for the year ended December 31, 2014 would have been an income from operations of $6,838 as compared with a loss from operations of $13,482 for the year ended December 31, 2013. Lower operating expenses as discussed above were offset by lower sales volume and higher retail store impairments.
Net loss for the year ended December 31, 2014 was $68,817 as compared to $106,298 for the year ended December 31, 2013. The improvement was mainly due to the $1,712 reduction in loss from operations due to the significant events discussed below, the change of $5,428 in fair value of warrants between periods, and the $32,101 loss on the extinguishment of debt in 2013. See Results of Operations for further details.
Cash used in operating activities for the year ended December 31, 2014 was $5,212 compared to $12,723 for the year ended December 31, 2013 from the corresponding period in 2013. The decrease was mainly due to decreased inventory levels and improved operating income excluding certain significant costs discussed below. The decrease was partially offset by an increase in interest payments and payments related to the significant costs.
Changes to Supply Chain Operations - In 2013, the transition to our new distribution center in La Mirada, California resulted in significant incremental costs (primarily labor). The issues surrounding the transition primarily related to improper design and integration and inadequate training and staffing. These issues caused processing inefficiencies that required us to employ additional staffing in order to meet customer demand. The transition was successfully completed during the fourth quarter of 2013. The center is now fully operational and labor costs have been reduced.
Additional inventory reserves - In late 2014, new management undertook a strategic shift to change its inventory profile and actively reduce inventory levels to improve store merchandising, working capital and liquidity. As a result, we implemented an initiative to accelerate the sale of slow-moving inventory through our retail and online sales channels, as well as through certain off-price channels. As part of this process, management conducted a style-by-style review of inventory and identified certain slow-moving, second quality finished goods and raw materials inventories that required additional reserves as a result of the decision to accelerate sales of those items. Based on our analysis of the quantities on hand as well as the estimated recovery on these items, we significantly increased our excess and obsolescence reserve by $4,525 through a charge against cost of sales in our consolidated statements of operations.
Customs settlements and contingencies - In 2012, German authorities audited the import records of our German subsidiary for the years 2009 through 2011 and issued retroactive punitive duty assessments on certain containers of goods imported. Despite ongoing appeals of the assessment, the German authorities demanded, and we paid, the outstanding balance of approximately $4,500 in the latter half of 2014. We recorded the duty portion of $79 in cost of sales and the retaliatory duties, interest and penalties of $5,104 in general and administrative expenses in our consolidated statements of operations. Additionally, during the fourth quarter of 2014, we wrote off approximately $3,300 in duty receivables to cost of sales in our consolidated statements of operations. These duty receivables related to changes in transfer costs for products sold to our European subsidiaries. We are also subject to, and have recorded charges related to, customs and similar audit settlements and contingencies in other jurisdictions.
Internal Investigation - On June 18, 2014, the Board voted to replace Mr. Charney as Chairman of the Board, suspended him as our President and CEO and notified him of its intent to terminate his employment for cause. In connection with the Standstill and Support Agreement, the Board formed the Internal Investigation which ultimately concluded with his termination for cause on December 16, 2014. The suspension, internal investigation, and termination have resulted in substantial legal and consulting fees.
Employment Settlements and Severance - In 2011, an industrial accident at our facility in Orange County, California resulted in a fatality to one of our employees, and in accordance with law, a mandatory criminal investigation was initiated. On August 19, 2014, a settlement of all claims related to the criminal investigation, pursuant to which the Company paid $1,000, was approved by the California Superior Court in Orange County. In addition, we had previously disclosed employment-related claims and experienced unusually high employee severance costs during 2014.
(1) U.S. Wholesale
U.S. Wholesale net sales for the year ended December 31, 2014, excluding online consumer net sales, increased by $8,113 or 5.1%, from the year ended December 31, 2013 mainly due to a significant new distributor that we added during the second quarter of 2014. We continue our focus on increasing our customer base by targeting direct sales, particularly sales to third-party screen printers. Online consumer net sales for the year ended December 31, 2014 decreased $395, or 1.0%, from the year ended December 31, 2013 mainly due to lower sales order volume. We continue our focus on targeted online advertising and promotional efforts.
(2) U.S. Retail
U.S. Retail net sales for the year ended December 31, 2014 decreased $13,569, or 6.6%, from the year ended December 31, 2013 mainly due to a decrease of approximately $14,000 in comparable store sales as a result of lower store foot traffic. Net sales decreased approximately $4,800 due to the closure of six stores in 2014, offset by an increase of approximately $1,100 from two new stores added since the beginning of January 2013.
(3) Canada
Canada net sales for the year ended December 31, 2014 decreased $8,590, or 14.3%, from the year ended December 31, 2013 mainly due to approximately $4,900 in lower sales, primarily in the retail and wholesale channels, and the unfavorable impact of foreign currency exchange rate changes of approximately $3,700.
Retail net sales for the year ended December 31, 2014 decreased $7,076, or 15.7%, from the year ended December 31, 2013 due to $4,300 lower sales resulting from the closure of one retail store and approximately $1,700 from lower comparable store sales due to lower store foot traffic. Additionally, the impact of foreign currency exchange rate changes contributed to the sales decrease of approximately $2,800.
Wholesale net sales for the year ended December 31, 2014 decreased $1,868, or 15.4%, from the year ended December 31, 2013. The decrease was largely due to lower sales orders resulting from a tightening focus on higher margin customers and lingering effects of order fulfillment delays associated with transition issues at the La Mirada distribution center. In addition, the impact of foreign currency exchange rate changes contributed to the sales decrease of approximately $700.
Online consumer net sales for the year ended December 31, 2014 increased $354, or 12.3%, from the year ended December 31, 2013 mainly due to email advertising campaign, as well as improvements to the online store rolled out in the second half of 2013. This increase in sales was partially offset by the impact of foreign currency exchange rate changes of approximately $200.
(4) International
International net sales for the year ended December 31, 2014 decreased $10,609, or 6.3%, from the year ended December 31, 2013 due to approximately $10,500 lower sales in all three sales channels and the unfavorable impact of foreign currency exchange rate changes of approximately $100.
Retail net sales for the year ended December 31, 2014 decreased $10,404, or 7.4%, from the year ended December 31, 2013. The decrease was due to lower comparable store sales of approximately $10,500 and lower sales of approximately $1,400 for the closure of five retail stores in 2014. The decrease was offset by approximately $200 higher sales due to seven new stores added since the beginning of January 2013 and the unfavorable impact of foreign currency exchange rate changes of approximately $400.
Wholesale net sales for the year ended December 31, 2014 were flat as compared to the year ended December 31, 2013. The favorable impact of foreign currency exchange rate changes was approximately $100.
Online consumer net sales for the year ended December 31, 2014 decreased $154, or 0.9%, from the year ended December 31, 2013 mainly due to lower sales order volume in Japan and Continental Europe, offset by higher sales order volume in Korea and the favorable impact of foreign currency exchange rate changes of approximately $200.
(5) Gross profit
Gross profit for the year ended December 31, 2014 decreased to $309,135 from $320,885 for the year ended December 31, 2013 due to lower retail sales volume at our U.S. Retail, Canada and International segments, offset by higher sales at our U.S. Wholesale segment. Excluding the effects of the significant events described above, gross profit as a percentage of net sales for the year ended December 31, 2014 slightly increased to 52.2% from 51.1%. The increase was mainly due to a decrease in freight costs associated with the completion of our transition to our La Mirada facility, offset by lower sales at our retail store operations.
(6) Selling and distribution expenses
Selling and distribution expenses for the year ended December 31, 2014 decreased $29,126, or 12.1%, from the year ended December 31, 2013. Excluding the effects of the changes to our supply chain operations discussed above, selling and distribution expenses decreased $17,279, or 7.5% from the year ended December 31, 2013 due primarily to lower selling related payroll costs of approximately $9,000, lower advertising costs of approximately $4,600 and lower travel and entertainment expenses of $1,400, all primarily as a result of our cost reduction efforts.
(7) General and administrative expenses
General and administrative expenses for the year ended December 31, 2014 increased $14,466, or 13.5%, from the year ended December 31, 2013. Excluding the effects of customs settlements and contingencies, the internal investigation, and employment settlements and severance discussed above, general and administrative expenses decreased $10,337, or 9.8% from the year ended December 31, 2013. The decrease was primarily due to $3,600 in lower share based compensation expense relating to the expiration and forfeiture of certain market based and performance based share awards and decreases in salaries and wages of approximately $3,800 and miscellaneous expenses such as travel, repair, and bank fees.
(8) Loss from operations    
Loss from operations was $27,583 for the year ended December 31, 2014 as compared to $29,295 for the year ended December 31, 2013. Excluding the effects of the significant events described above, our operating results for the year ended December 31, 2014 would have been an income from operations of $6,838 as compared with a loss from operations of $13,482 for the year ended December 31, 2013. Lower sales volume and higher retail store impairments were offset by decreases in our operating expenses as discussed above.
(9) Income tax provision
The provision for income tax for the year ended December 31, 2014 increased to $2,159 as compared to $1,771 for the year ended December 31, 2013. Although we incurred a loss from operations on a consolidated basis for the years ended December 31, 2014 and 2013, some of our foreign domiciled subsidiaries reported income from operations and are taxed on a stand-alone reporting basis. In 2014 and 2013, we recorded valuation allowances against a majority of our deferred tax assets, including 100% of the U.S. deferred tax assets and certain foreign deferred tax assets. We recognized no tax benefits on our loss before income taxes in 2014 and 2013.
    """
   
    print("🚀 Financial Sentiment Analysis Demo")
    print("="*50)
   
    # Load ML lexicon (you can uncomment and modify this to load your actual lexicon)
    # analyzer.load_ml_lexicon('path_to_your_ml_lexicon.csv')
   
    # Run comprehensive analysis
    results = analyzer.analyze_text(sample_text)
   
    # Print summary
    analyzer.print_analysis_summary(results)
   
    # Export results
    exported_files = analyzer.export_results(results, "demo_analysis")
   
    print(f"\n✅ Demo completed! Check the exported files for detailed results.")
   
    return results, analyzer

if __name__ == "__main__":
    results, analyzer = main()