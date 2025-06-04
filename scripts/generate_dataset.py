"""
Arabic Explainable Error Generator (AEEG) for MTAGEC.

This script implements the AEEG algorithm described in Section 4.1 of the paper.
It generates synthetic Arabic grammatical errors with error type annotations
and evidence for explanation.
"""
import os
import re
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
from pyarabic import araby
from pyarabic.tokenize import tokenize


# Error type definitions based on Belkebir & Habash (2021) and Alfaifi (2014)
ERROR_TYPES = {
    # Orthographic errors
    "OA": {"name": "Alif/Ya confusion", "weight": 1.0},
    "OH": {"name": "Hamza confusion", "weight": 1.0},
    "OT": {"name": "Ta/Ha confusion", "weight": 1.0},
    "OW": {"name": "Alif variants", "weight": 1.0},
    "OC": {"name": "Character transposition", "weight": 1.0},
    "ON": {"name": "Confusion in Tanwin", "weight": 1.0},
    "OS": {"name": "Vowel reduction", "weight": 1.0},
    "OG": {"name": "Vowel elongation", "weight": 1.0},
    "OR": {"name": "Substitute a character", "weight": 1.0},
    "OD": {"name": "Character addition", "weight": 1.0},
    "OM": {"name": "Character deletion", "weight": 1.0},
    
    # Morphological errors
    "XF": {"name": "Definite article misuse", "weight": 1.0},
    "XG": {"name": "Gender agreement", "weight": 1.0},
    "XN": {"name": "Number agreement", "weight": 1.0},
    "XT": {"name": "Unnecessary word", "weight": 1.0},
    "XM": {"name": "Missing word", "weight": 1.0},
    "MI": {"name": "Derivation error", "weight": 1.0},
    "MT": {"name": "Verb tenses", "weight": 1.0},
    
    # Semantic errors
    "SC": {"name": "Confusion in conjunction", "weight": 1.0},
    "SW": {"name": "Incorrect word choice", "weight": 1.0},
    
    # Punctuation errors
    "PC": {"name": "Punctuation replacement", "weight": 1.0},
    "PM": {"name": "Punctuation deletion", "weight": 1.0},
    "PT": {"name": "Punctuation insertion", "weight": 1.0},
    
    # Merge/Split errors
    "MG": {"name": "Word merging", "weight": 1.0},
    "SP": {"name": "Extra space", "weight": 1.0},
}

# Character mappings for orthographic errors
ALIF_YA_MAPPING = {
    'ا': 'ى', 'ى': 'ا', 'إ': 'ي', 'أ': 'ي', 'آ': 'ي',
    'ي': 'ا', 'ئ': 'ا', 'ء': 'ا'
}

HAMZA_MAPPING = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي',
    'ء': '', 'ا': 'أ', 'و': 'ؤ', 'ي': 'ئ'
}

TA_HA_MAPPING = {
    'ة': 'ه', 'ه': 'ة', 'ت': 'ة', 'ة': 'ت'
}

ALIF_VARIANTS = {
    'ا': 'أ', 'أ': 'إ', 'إ': 'آ', 'آ': 'ا'
}

TANWIN_MAPPING = {
    'ً': 'ٌ', 'ٌ': 'ٍ', 'ٍ': 'ً', 'ا': 'اً'
}

# Common Arabic conjunctions for SC errors
CONJUNCTIONS = ['و', 'أو', 'ثم', 'ف', 'لكن', 'بل', 'حتى', 'أم']

# Common Arabic punctuation for PC, PM, PT errors
PUNCTUATION = ['.', '،', '؛', ':', '؟', '!']

# Common Arabic prepositions for word choice errors
PREPOSITIONS = ['في', 'من', 'إلى', 'على', 'عن', 'مع', 'ل', 'ب', 'ك']


class ArabicExplainableErrorGenerator:
    """
    Implementation of the AEEG algorithm (Algorithm 1 in the paper).
    Generates synthetic Arabic grammatical errors with explanations.
    """
    
    def __init__(self, error_rate: float = 0.1, seed: int = 42):
        """
        Initialize the error generator.
        
        Args:
            error_rate: Probability of introducing an error (p in Eq. 12)
            seed: Random seed for reproducibility
        """
        self.error_rate = error_rate
        self.error_types = ERROR_TYPES
        self.error_counts = Counter()
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize weights for dynamic error type selection (Eq. 13-14)
        self.update_weights()
    
    def update_weights(self) -> None:
        """
        Update error type weights based on inverse frequency (Eq. 13-14).
        """
        for error_type in self.error_types:
            # Inverse frequency weighting (Eq. 13)
            self.error_types[error_type]["weight"] = 1.0 / (1.0 + self.error_counts[error_type])
        
        # Normalize weights to ensure sum(P(T_i)) = 1 (Eq. 14)
        total_weight = sum(self.error_types[t]["weight"] for t in self.error_types)
        for error_type in self.error_types:
            self.error_types[error_type]["prob"] = self.error_types[error_type]["weight"] / total_weight
    
    def sample_error_type(self) -> str:
        """
        Sample an error type based on dynamic weights (Eq. 14).
        
        Returns:
            Error type code (e.g., 'OA', 'XG')
        """
        # Update weights before sampling
        self.update_weights()
        
        # Sample error type based on probabilities
        error_types = list(self.error_types.keys())
        probs = [self.error_types[t]["prob"] for t in error_types]
        
        # Sample and update count
        error_type = np.random.choice(error_types, p=probs)
        self.error_counts[error_type] += 1
        
        return error_type
    
    def calculate_expected_errors(self, tokens: List[str]) -> int:
        """
        Calculate expected number of errors based on sentence length (Eq. 11-12).
        
        Args:
            tokens: List of tokens in the sentence
            
        Returns:
            Expected number of errors to introduce
        """
        # Count Arabic tokens (Eq. 11)
        arabic_tokens = [t for t in tokens if araby.is_arabicword(t)]
        L_A = len(arabic_tokens)
        
        # Calculate expected errors (Eq. 12)
        E_e = max(1, int(self.error_rate * L_A))
        
        return E_e
    
    def apply_orthographic_error(
        self, token: str, error_type: str
    ) -> Tuple[str, str, List[int]]:
        """
        Apply orthographic error to a token.
        
        Args:
            token: Original token
            error_type: Error type code
            
        Returns:
            Tuple of (modified token, explanation, evidence indices)
        """
        if not token or not araby.is_arabicword(token):
            return token, "", []
        
        chars = list(token)
        explanation = ""
        evidence = []
        
        if error_type == "OA":  # Alif/Ya confusion
            for i, char in enumerate(chars):
                if char in ALIF_YA_MAPPING:
                    chars[i] = ALIF_YA_MAPPING[char]
                    explanation = f"خطأ في الألف/الياء: تم استبدال '{char}' بـ '{chars[i]}'"
                    evidence = [i]
                    break
        
        elif error_type == "OH":  # Hamza confusion
            for i, char in enumerate(chars):
                if char in HAMZA_MAPPING:
                    chars[i] = HAMZA_MAPPING[char]
                    explanation = f"خطأ في الهمزة: تم استبدال '{char}' بـ '{chars[i]}'"
                    evidence = [i]
                    break
        
        elif error_type == "OT":  # Ta/Ha confusion
            for i, char in enumerate(chars):
                if char in TA_HA_MAPPING:
                    chars[i] = TA_HA_MAPPING[char]
                    explanation = f"خطأ في التاء/الهاء: تم استبدال '{char}' بـ '{chars[i]}'"
                    evidence = [i]
                    break
        
        elif error_type == "OW":  # Alif variants
            for i, char in enumerate(chars):
                if char in ALIF_VARIANTS:
                    chars[i] = ALIF_VARIANTS[char]
                    explanation = f"خطأ في أشكال الألف: تم استبدال '{char}' بـ '{chars[i]}'"
                    evidence = [i]
                    break
        
        elif error_type == "OC":  # Character transposition
            if len(chars) >= 2:
                i = random.randint(0, len(chars) - 2)
                chars[i], chars[i+1] = chars[i+1], chars[i]
                explanation = f"خطأ في ترتيب الحروف: تم تبديل '{token[i]}' و '{token[i+1]}'"
                evidence = [i, i+1]
        
        elif error_type == "ON":  # Confusion in Tanwin
            for i, char in enumerate(chars):
                if char in TANWIN_MAPPING:
                    chars[i] = TANWIN_MAPPING[char]
                    explanation = f"خطأ في التنوين: تم استبدال '{char}' بـ '{chars[i]}'"
                    evidence = [i]
                    break
                elif i == len(chars) - 1 and char == 'ا':
                    chars[i] = 'اً'
                    explanation = "خطأ في التنوين: تم إضافة تنوين الفتح"
                    evidence = [i]
                    break
        
        elif error_type == "OS":  # Vowel reduction
            diacritics = [i for i, c in enumerate(chars) if araby.is_diacritic(c)]
            if diacritics:
                i = random.choice(diacritics)
                explanation = f"خطأ في الحركات: تم حذف '{chars[i]}'"
                evidence = [i]
                chars.pop(i)
        
        elif error_type == "OG":  # Vowel elongation
            vowels = {'َ': 'ا', 'ُ': 'و', 'ِ': 'ي'}
            for i, char in enumerate(chars):
                if char in vowels and i < len(chars) - 1:
                    chars.insert(i+1, vowels[char])
                    explanation = f"خطأ في إطالة الحركة: تم إضافة '{vowels[char]}' بعد '{char}'"
                    evidence = [i, i+1]
                    break
        
        elif error_type == "OR":  # Substitute a character
            if chars:
                i = random.randint(0, len(chars) - 1)
                original = chars[i]
                # Get a random Arabic letter that's different from the original
                arabic_letters = 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي'
                replacement = original
                while replacement == original:
                    replacement = random.choice(arabic_letters)
                chars[i] = replacement
                explanation = f"خطأ في استبدال حرف: تم استبدال '{original}' بـ '{replacement}'"
                evidence = [i]
        
        elif error_type == "OD":  # Character addition
            if chars:
                i = random.randint(0, len(chars))
                # Add a random Arabic letter
                addition = random.choice('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
                chars.insert(i, addition)
                explanation = f"خطأ في إضافة حرف: تم إضافة '{addition}'"
                evidence = [i]
        
        elif error_type == "OM":  # Character deletion
            if len(chars) > 1:  # Ensure we don't delete the entire word
                i = random.randint(0, len(chars) - 1)
                deleted = chars[i]
                explanation = f"خطأ في حذف حرف: تم حذف '{deleted}'"
                evidence = [i]
                chars.pop(i)
        
        modified_token = ''.join(chars)
        return modified_token, explanation, evidence
    
    def apply_morphological_error(
        self, tokens: List[str], index: int, error_type: str
    ) -> Tuple[List[str], str, List[int]]:
        """
        Apply morphological error to tokens.
        
        Args:
            tokens: List of original tokens
            index: Index of the token to modify
            error_type: Error type code
            
        Returns:
            Tuple of (modified tokens, explanation, evidence indices)
        """
        modified_tokens = tokens.copy()
        explanation = ""
        evidence = []
        
        if error_type == "XF":  # Definite article misuse
            if index < len(tokens):
                token = tokens[index]
                if token.startswith('ال'):
                    # Remove definite article
                    modified_tokens[index] = token[2:]
                    explanation = f"خطأ في استخدام أل التعريف: تم حذف 'ال' من '{token}'"
                else:
                    # Add definite article
                    modified_tokens[index] = 'ال' + token
                    explanation = f"خطأ في استخدام أل التعريف: تم إضافة 'ال' إلى '{token}'"
                evidence = [index]
        
        elif error_type == "XG":  # Gender agreement
            # Simplified gender agreement error (would be more complex in real implementation)
            if index < len(tokens):
                token = tokens[index]
                if token.endswith('ة'):
                    # Change feminine to masculine
                    modified_tokens[index] = token[:-1]
                    explanation = f"خطأ في المطابقة في النوع: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                else:
                    # Change masculine to feminine
                    modified_tokens[index] = token + 'ة'
                    explanation = f"خطأ في المطابقة في النوع: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                evidence = [index]
                
                # Add context as evidence if available
                if index > 0:
                    evidence.append(index - 1)
                if index < len(tokens) - 1:
                    evidence.append(index + 1)
        
        elif error_type == "XN":  # Number agreement
            # Simplified number agreement error
            if index < len(tokens):
                token = tokens[index]
                # Singular to plural transformation (simplified)
                if token.endswith('ون'):
                    modified_tokens[index] = token[:-2]
                    explanation = f"خطأ في المطابقة في العدد: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                else:
                    modified_tokens[index] = token + 'ون'
                    explanation = f"خطأ في المطابقة في العدد: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                evidence = [index]
                
                # Add context as evidence
                if index > 0:
                    evidence.append(index - 1)
                if index < len(tokens) - 1:
                    evidence.append(index + 1)
        
        elif error_type == "XT":  # Unnecessary word
            if tokens:
                # Insert a common unnecessary word
                unnecessary_words = ['جدا', 'فقط', 'تماما', 'أيضا', 'بالفعل']
                word_to_add = random.choice(unnecessary_words)
                insert_pos = min(index, len(modified_tokens))
                modified_tokens.insert(insert_pos, word_to_add)
                explanation = f"خطأ في إضافة كلمة غير ضرورية: تم إضافة '{word_to_add}'"
                evidence = [insert_pos]
                
                # Add context as evidence
                if insert_pos > 0:
                    evidence.append(insert_pos - 1)
                if insert_pos < len(modified_tokens) - 1:
                    evidence.append(insert_pos + 1)
        
        elif error_type == "XM":  # Missing word
            if len(tokens) > 1 and index < len(tokens):
                # Remove a word
                removed_word = tokens[index]
                modified_tokens.pop(index)
                explanation = f"خطأ في حذف كلمة: تم حذف '{removed_word}'"
                
                # Evidence is surrounding context
                if index > 0:
                    evidence.append(index - 1)
                if index < len(tokens) - 1:
                    evidence.append(index)  # Index shifts after removal
        
        elif error_type == "MI":  # Derivation error
            if index < len(tokens):
                token = tokens[index]
                # Simplified derivation error (would use morphological analyzer in real implementation)
                if len(token) > 3:
                    # Modify prefix or suffix
                    if random.choice([True, False]) and token.startswith('م'):
                        modified_tokens[index] = token[1:]
                        explanation = f"خطأ في الاشتقاق: تم حذف 'م' من '{token}'"
                    else:
                        modified_tokens[index] = 'م' + token
                        explanation = f"خطأ في الاشتقاق: تم إضافة 'م' إلى '{token}'"
                    evidence = [index]
        
        elif error_type == "MT":  # Verb tenses
            if index < len(tokens):
                token = tokens[index]
                # Simplified verb tense error (would use morphological analyzer in real implementation)
                if token.startswith('ي'):
                    # Present to past transformation (very simplified)
                    modified_tokens[index] = token.replace('ي', 'سي', 1)
                    explanation = f"خطأ في زمن الفعل: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                elif token.startswith('س'):
                    # Future to present transformation
                    modified_tokens[index] = token[1:]
                    explanation = f"خطأ في زمن الفعل: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                else:
                    # Add future marker
                    modified_tokens[index] = 'س' + token
                    explanation = f"خطأ في زمن الفعل: تم تغيير '{token}' إلى '{modified_tokens[index]}'"
                evidence = [index]
                
                # Add context as evidence
                if index > 0:
                    evidence.append(index - 1)
        
        return modified_tokens, explanation, evidence
    
    def apply_semantic_error(
        self, tokens: List[str], index: int, error_type: str
    ) -> Tuple[List[str], str, List[int]]:
        """
        Apply semantic error to tokens.
        
        Args:
            tokens: List of original tokens
            index: Index of the token to modify
            error_type: Error type code
            
        Returns:
            Tuple of (modified tokens, explanation, evidence indices)
        """
        modified_tokens = tokens.copy()
        explanation = ""
        evidence = []
        
        if error_type == "SC":  # Confusion in conjunction
            if index < len(tokens) and tokens[index] in CONJUNCTIONS:
                # Replace with another conjunction
                original = tokens[index]
                replacement = original
                while replacement == original:
                    replacement = random.choice(CONJUNCTIONS)
                modified_tokens[index] = replacement
                explanation = f"خطأ في استخدام حرف العطف: تم استبدال '{original}' بـ '{replacement}'"
                evidence = [index]
                
                # Add context as evidence
                if index > 0:
                    evidence.append(index - 1)
                if index < len(tokens) - 1:
                    evidence.append(index + 1)
            elif index < len(tokens) - 1:
                # Insert a conjunction
                conjunction = random.choice(CONJUNCTIONS)
                modified_tokens.insert(index, conjunction)
                explanation = f"خطأ في إضافة حرف عطف: تم إضافة '{conjunction}'"
                evidence = [index]
                
                # Add context as evidence
                if index > 0:
                    evidence.append(index - 1)
                evidence.append(index + 1)  # Original token now shifted
        
        elif error_type == "SW":  # Incorrect word choice
            if index < len(tokens):
                original = tokens[index]
                
                # Check if it's a preposition
                if original in PREPOSITIONS:
                    # Replace with another preposition
                    replacement = original
                    while replacement == original:
                        replacement = random.choice(PREPOSITIONS)
                    modified_tokens[index] = replacement
                    explanation = f"خطأ في اختيار الكلمة: تم استبدال '{original}' بـ '{replacement}'"
                else:
                    # Replace with a similar word (simplified)
                    # In a real implementation, would use word embeddings or a thesaurus
                    replacements = {
                        'كبير': 'ضخم', 'صغير': 'ضئيل', 'جميل': 'رائع',
                        'سريع': 'عاجل', 'بطيء': 'متمهل', 'قوي': 'شديد',
                        'ضعيف': 'واهن', 'ذكي': 'فطن', 'غبي': 'أحمق'
                    }
                    
                    if original in replacements:
                        replacement = replacements[original]
                        modified_tokens[index] = replacement
                        explanation = f"خطأ في اختيار الكلمة: تم استبدال '{original}' بـ '{replacement}'"
                    else:
                        # No suitable replacement found
                        return tokens, "", []
                
                evidence = [index]
                
                # Add context as evidence
                if index > 0:
                    evidence.append(index - 1)
                if index < len(tokens) - 1:
                    evidence.append(index + 1)
        
        return modified_tokens, explanation, evidence
    
    def apply_punctuation_error(
        self, tokens: List[str], index: int, error_type: str
    ) -> Tuple[List[str], str, List[int]]:
        """
        Apply punctuation error to tokens.
        
        Args:
            tokens: List of original tokens
            index: Index of the token to modify
            error_type: Error type code
            
        Returns:
            Tuple of (modified tokens, explanation, evidence indices)
        """
        modified_tokens = tokens.copy()
        explanation = ""
        evidence = []
        
        if error_type == "PC":  # Punctuation replacement
            # Find punctuation tokens
            punct_indices = [i for i, t in enumerate(tokens) if t in PUNCTUATION]
            
            if punct_indices:
                # Replace a punctuation mark with another
                idx = random.choice(punct_indices)
                original = tokens[idx]
                replacement = original
                while replacement == original:
                    replacement = random.choice(PUNCTUATION)
                modified_tokens[idx] = replacement
                explanation = f"خطأ في علامة الترقيم: تم استبدال '{original}' بـ '{replacement}'"
                evidence = [idx]
            else:
                # No punctuation to replace
                return tokens, "", []
        
        elif error_type == "PM":  # Punctuation deletion
            # Find punctuation tokens
            punct_indices = [i for i, t in enumerate(tokens) if t in PUNCTUATION]
            
            if punct_indices:
                # Delete a punctuation mark
                idx = random.choice(punct_indices)
                deleted = tokens[idx]
                modified_tokens.pop(idx)
                explanation = f"خطأ في حذف علامة الترقيم: تم حذف '{deleted}'"
                
                # Evidence is surrounding context
                if idx > 0:
                    evidence.append(idx - 1)
                if idx < len(tokens) - 1:
                    evidence.append(idx)  # Index shifts after removal
            else:
                # No punctuation to delete
                return tokens, "", []
        
        elif error_type == "PT":  # Punctuation insertion
            # Insert a punctuation mark at a random position
            punct = random.choice(PUNCTUATION)
            insert_pos = min(index, len(modified_tokens))
            modified_tokens.insert(insert_pos, punct)
            explanation = f"خطأ في إضافة علامة ترقيم: تم إضافة '{punct}'"
            evidence = [insert_pos]
            
            # Add context as evidence
            if insert_pos > 0:
                evidence.append(insert_pos - 1)
            if insert_pos < len(modified_tokens) - 1:
                evidence.append(insert_pos + 1)
        
        return modified_tokens, explanation, evidence
    
    def apply_merge_split_error(
        self, tokens: List[str], index: int, error_type: str
    ) -> Tuple[List[str], str, List[int]]:
        """
        Apply merge/split error to tokens.
        
        Args:
            tokens: List of original tokens
            index: Index of the token to modify
            error_type: Error type code
            
        Returns:
            Tuple of (modified tokens, explanation, evidence indices)
        """
        modified_tokens = tokens.copy()
        explanation = ""
        evidence = []
        
        if error_type == "MG":  # Word merging
            if index < len(tokens) - 1:
                # Merge two consecutive words
                word1 = tokens[index]
                word2 = tokens[index + 1]
                merged = word1 + word2
                modified_tokens[index] = merged
                modified_tokens.pop(index + 1)
                explanation = f"خطأ في دمج الكلمات: تم دمج '{word1}' و '{word2}' إلى '{merged}'"
                evidence = [index]
            else:
                # No word to merge with
                return tokens, "", []
        
        elif error_type == "SP":  # Extra space
            if index < len(tokens) and len(tokens[index]) > 3:
                # Split a word with an extra space
                token = tokens[index]
                split_point = random.randint(1, len(token) - 1)
                part1 = token[:split_point]
                part2 = token[split_point:]
                modified_tokens[index] = part1
                modified_tokens.insert(index + 1, part2)
                explanation = f"خطأ في إضافة مسافة: تم تقسيم '{token}' إلى '{part1}' و '{part2}'"
                evidence = [index, index + 1]
            else:
                # Word too short to split
                return tokens, "", []
        
        return modified_tokens, explanation, evidence
    
    def apply_error(
        self, tokens: List[str], error_type: str, index: int
    ) -> Tuple[List[str], str, List[int], str]:
        """
        Apply an error of the specified type to the tokens.
        
        Args:
            tokens: List of original tokens
            error_type: Error type code
            index: Index of the token to modify
            
        Returns:
            Tuple of (modified tokens, explanation, evidence indices, error type)
        """
        # Ensure index is valid
        if not tokens:
            return tokens, "", [], error_type
        
        index = min(index, len(tokens) - 1)
        
        # Apply error based on type
        if error_type in ["OA", "OH", "OT", "OW", "OC", "ON", "OS", "OG", "OR", "OD", "OM"]:
            # Orthographic errors (word-level)
            if index < len(tokens):
                modified_token, explanation, token_evidence = self.apply_orthographic_error(
                    tokens[index], error_type
                )
                if explanation:
                    modified_tokens = tokens.copy()
                    modified_tokens[index] = modified_token
                    # Convert token-level evidence to sentence-level
                    evidence = [index]
                    return modified_tokens, explanation, evidence, error_type
            
        elif error_type in ["XF", "XG", "XN", "XT", "XM", "MI", "MT"]:
            # Morphological errors
            modified_tokens, explanation, evidence = self.apply_morphological_error(
                tokens, index, error_type
            )
            if explanation:
                return modified_tokens, explanation, evidence, error_type
            
        elif error_type in ["SC", "SW"]:
            # Semantic errors
            modified_tokens, explanation, evidence = self.apply_semantic_error(
                tokens, index, error_type
            )
            if explanation:
                return modified_tokens, explanation, evidence, error_type
            
        elif error_type in ["PC", "PM", "PT"]:
            # Punctuation errors
            modified_tokens, explanation, evidence = self.apply_punctuation_error(
                tokens, index, error_type
            )
            if explanation:
                return modified_tokens, explanation, evidence, error_type
            
        elif error_type in ["MG", "SP"]:
            # Merge/Split errors
            modified_tokens, explanation, evidence = self.apply_merge_split_error(
                tokens, index, error_type
            )
            if explanation:
                return modified_tokens, explanation, evidence, error_type
        
        # If we get here, no error was applied
        return tokens, "", [], error_type
    
    def generate_errors(
        self, sentence: str
    ) -> Tuple[str, str, List[Tuple[str, str, List[int]]]]:
        """
        Generate errors for a sentence following Algorithm 1 in the paper.
        
        Args:
            sentence: Original grammatically correct sentence
            
        Returns:
            Tuple of (modified sentence, original sentence, list of (error_type, explanation, evidence))
        """
        # Tokenize the sentence (Eq. 10)
        tokens = tokenize(sentence)
        
        # Calculate expected number of errors (Eq. 11-12)
        E_e = self.calculate_expected_errors(tokens)
        
        # Initialize tracking variables
        modified_tokens = tokens.copy()
        errors = []
        corrupted_indices = set()
        
        # Apply E_e errors (Algorithm 1, lines 6-13)
        for i in range(E_e):
            # Sample error type (Algorithm 1, line 8)
            error_type = self.sample_error_type()
            
            # Pick target index avoiding already corrupted indices if possible
            available_indices = [j for j in range(len(modified_tokens)) if j not in corrupted_indices]
            if not available_indices:
                available_indices = list(range(len(modified_tokens)))
            
            index = random.choice(available_indices)
            
            # Apply error (Algorithm 1, lines 9-10)
            new_tokens, explanation, evidence, error_code = self.apply_error(
                modified_tokens, error_type, index
            )
            
            # If error was successfully applied
            if explanation:
                modified_tokens = new_tokens
                errors.append((error_code, explanation, evidence))
                
                # Mark indices as corrupted
                corrupted_indices.update(evidence)
        
        # Reconstruct the modified sentence
        modified_sentence = ' '.join(modified_tokens)
        
        return modified_sentence, sentence, errors


def main():
    """
    Main function to generate the ExplAGEC dataset.
    """
    parser = argparse.ArgumentParser(description='Generate synthetic Arabic GEC data')
    parser.add_argument('--input', required=True, help='Input file with correct sentences')
    parser.add_argument('--output', required=True, help='Output file for generated data')
    parser.add_argument('--error-rate', type=float, default=0.1, help='Error rate (p in Eq. 12)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-examples', type=int, default=None, help='Number of examples to generate')
    args = parser.parse_args()
    
    # Initialize error generator
    generator = ArabicExplainableErrorGenerator(error_rate=args.error_rate, seed=args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Read input sentences
    with open(args.input, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    # Limit number of examples if specified
    if args.num_examples is not None:
        sentences = sentences[:args.num_examples]
    
    # Generate errors
    dataset = []
    for sentence in tqdm(sentences, desc="Generating errors"):
        modified_sentence, original_sentence, errors = generator.generate_errors(sentence)
        
        # Create dataset entry
        entry = {
            "original": original_sentence,
            "modified": modified_sentence,
            "errors": [
                {
                    "type": error_type,
                    "explanation": explanation,
                    "evidence": evidence
                }
                for error_type, explanation, evidence in errors
            ]
        }
        
        dataset.append(entry)
    
    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(dataset)} examples with {sum(len(entry['errors']) for entry in dataset)} errors")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()