import re
import string
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """utility untuk text processing dan normalization"""
    
    def __init__(self):
        # stopwords bahasa indonesia (simplified)
        self.stopwords = {
            'dan', 'atau', 'yang', 'dengan', 'dalam', 'untuk', 'pada', 'dari', 'ke', 'oleh',
            'akan', 'adalah', 'dapat', 'telah', 'sudah', 'belum', 'tidak', 'bukan', 'juga',
            'lebih', 'sangat', 'paling', 'saja', 'hanya', 'masih', 'jadi', 'bisa', 'harus',
            'itu', 'ini', 'saya', 'anda', 'dia', 'mereka', 'kita', 'kami', 'nya', 'nya',
            'di', 'ke', 'dari', 'untuk', 'dalam', 'pada', 'dengan', 'oleh', 'tentang',
            'seperti', 'antara', 'melalui', 'selama', 'sebelum', 'sesudah', 'sambil'
        }
        
        # pattern untuk cleanup
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def normalize_text(self, text: str) -> str:
        """normalize text untuk processing"""
        try:
            if not text:
                return ""
            
            # convert ke lowercase
            text = text.lower()
            
            # remove urls dan emails
            text = self.url_pattern.sub('', text)
            text = self.email_pattern.sub('', text)
            
            # remove punctuation kecuali yang berguna
            # keep: . , ? ! - 
            text = re.sub(r'[^\w\s\.\,\?\!\-]', '', text)
            
            # normalize whitespace
            text = self.whitespace_pattern.sub(' ', text)
            
            # strip
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"error normalizing text: {e}")
            return text
    
    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
        """extract keywords dari text"""
        try:
            if not text:
                return []
            
            # normalize text
            normalized = self.normalize_text(text)
            
            # split into words
            words = normalized.split()
            
            # filter words
            keywords = []
            for word in words:
                # skip short words
                if len(word) < min_length:
                    continue
                
                # skip stopwords
                if word in self.stopwords:
                    continue
                
                # skip numbers only
                if word.isdigit():
                    continue
                
                keywords.append(word)
            
            # remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword not in seen:
                    seen.add(keyword)
                    unique_keywords.append(keyword)
            
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"error extracting keywords: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """calculate text similarity menggunakan jaccard similarity"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # extract keywords dari both texts
            keywords1 = set(self.extract_keywords(text1))
            keywords2 = set(self.extract_keywords(text2))
            
            if not keywords1 and not keywords2:
                return 1.0 if text1.strip() == text2.strip() else 0.0
            
            if not keywords1 or not keywords2:
                return 0.0
            
            # jaccard similarity
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"error calculating similarity: {e}")
            return 0.0
    
    def clean_response_text(self, text: str) -> str:
        """clean ai response text"""
        try:
            if not text:
                return ""
            
            # remove multiple spaces
            text = self.whitespace_pattern.sub(' ', text)
            
            # remove multiple newlines
            text = re.sub(r'\n+', '\n', text)
            
            # fix spacing around punctuation
            text = re.sub(r'\s+([,.!?])', r'\1', text)
            text = re.sub(r'([,.!?])\s*', r'\1 ', text)
            
            # trim
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"error cleaning response text: {e}")
            return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """extract entities dari text (simple implementation)"""
        try:
            entities = {
                "technologies": [],
                "skills": [],
                "projects": [],
                "locations": []
            }
            
            text_lower = text.lower()
            
            # technology keywords
            tech_keywords = [
                'python', 'java', 'javascript', 'react', 'next.js', 'node.js',
                'fastapi', 'django', 'flask', 'postgresql', 'mysql', 'mongodb',
                'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'git', 'github',
                'machine learning', 'data science', 'ai', 'artificial intelligence',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'
            ]
            
            for tech in tech_keywords:
                if tech in text_lower:
                    entities["technologies"].append(tech)
            
            # skill keywords
            skill_keywords = [
                'web development', 'backend', 'frontend', 'full stack',
                'database', 'api', 'microservices', 'devops', 'testing',
                'algorithm', 'data structure', 'problem solving'
            ]
            
            for skill in skill_keywords:
                if skill in text_lower:
                    entities["skills"].append(skill)
            
            # project keywords
            project_patterns = [
                r'rush hour.*?solver',
                r'little alchemy.*?search',
                r'iq puzzler.*?solver',
                r'portfolio.*?website'
            ]
            
            for pattern in project_patterns:
                matches = re.findall(pattern, text_lower)
                entities["projects"].extend(matches)
            
            # location keywords
            location_keywords = ['jakarta', 'bandung', 'indonesia', 'itb']
            
            for location in location_keywords:
                if location in text_lower:
                    entities["locations"].append(location)
            
            # remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"error extracting entities: {e}")
            return {}
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """simple text summarization"""
        try:
            if not text:
                return ""
            
            # split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= max_sentences:
                return text
            
            # score sentences by keyword frequency
            all_keywords = self.extract_keywords(text)
            keyword_freq = {}
            for keyword in all_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            sentence_scores = []
            for sentence in sentences:
                score = 0
                sentence_keywords = self.extract_keywords(sentence)
                for keyword in sentence_keywords:
                    score += keyword_freq.get(keyword, 0)
                sentence_scores.append((sentence, score))
            
            # sort by score dan ambil top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
            
            # maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    summary_sentences.append(sentence)
                    top_sentences.remove(sentence)
                if not top_sentences:
                    break
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"error summarizing text: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def detect_language(self, text: str) -> str:
        """simple language detection"""
        try:
            if not text:
                return "unknown"
            
            # simple heuristic berdasarkan common words
            indonesia_indicators = [
                'dan', 'atau', 'yang', 'dengan', 'dalam', 'untuk', 'pada',
                'saya', 'anda', 'adalah', 'tidak', 'bisa', 'akan'
            ]
            
            english_indicators = [
                'and', 'or', 'the', 'with', 'in', 'for', 'on', 'at',
                'i', 'you', 'is', 'are', 'can', 'will', 'have'
            ]
            
            text_lower = text.lower()
            
            id_count = sum(1 for word in indonesia_indicators if word in text_lower)
            en_count = sum(1 for word in english_indicators if word in text_lower)
            
            if id_count > en_count:
                return "indonesia"
            elif en_count > id_count:
                return "english"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"error detecting language: {e}")
            return "unknown"
    
    def validate_input(self, text: str, max_length: int = 1000) -> Dict[str, Any]:
        """validate user input"""
        try:
            result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "stats": {}
            }
            
            if not text:
                result["valid"] = False
                result["errors"].append("text is empty")
                return result
            
            # check length
            if len(text) > max_length:
                result["valid"] = False
                result["errors"].append(f"text too long (max {max_length} characters)")
            
            # check for spam patterns
            if len(set(text)) < len(text) * 0.3:  # too repetitive
                result["warnings"].append("text appears repetitive")
            
            # check for gibberish
            words = text.split()
            if len(words) > 1:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if avg_word_length > 15:  # very long average word length
                    result["warnings"].append("text may contain gibberish")
            
            # language detection
            language = self.detect_language(text)
            
            result["stats"] = {
                "character_count": len(text),
                "word_count": len(words),
                "language": language,
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"error validating input: {e}")
            return {
                "valid": False,
                "errors": [f"validation error: {str(e)}"],
                "warnings": [],
                "stats": {}
            }

# global text processor instance
_text_processor = TextProcessor()

def get_text_processor() -> TextProcessor:
    """get global text processor instance"""
    return _text_processor

# convenience functions
def normalize_text(text: str) -> str:
    return get_text_processor().normalize_text(text)

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    return get_text_processor().extract_keywords(text, max_keywords=max_keywords)

def clean_response(text: str) -> str:
    return get_text_processor().clean_response_text(text)