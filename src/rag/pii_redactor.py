"""
PII Redaction Module
Uses Microsoft Presidio to detect and anonymize Personally Identifiable Information (PII).
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from config.logging_config import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class PIIRedactor:
    def __init__(self):
        logger.info("Initializing PII Redactor...")
        try:
            engine_name = getattr(settings, "PII_NLP_ENGINE_NAME", "spacy")
            model_name = getattr(settings, "PII_NLP_MODEL_NAME", "en_core_web_sm")
            
            logger.info(f"Configuring Presidio with engine='{engine_name}', model='{model_name}'")
            configuration = {
                "nlp_engine_name": engine_name,
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            
            # Load custom patterns (e.g., specific company IDs)
            self._load_custom_recognizers()
            
            self.anonymizer = AnonymizerEngine()
            logger.info("PII Redactor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PII Redactor: {e}")
            raise

    def _load_custom_recognizers(self):
        """
        Example of how to add custom regex patterns for domain-specific PII.
        """
        # 1. Define the Regex Pattern
        # Example: Match an internal employee ID like "EMP-12345"
        emp_id_pattern = Pattern(
            name="employee_id_pattern", 
            regex=r"\bEMP-\d{5}\b", 
            score=0.8
        )
        
        # 2. Create a Recognizer for the pattern
        emp_id_recognizer = PatternRecognizer(
            supported_entity="EMPLOYEE_ID", 
            patterns=[emp_id_pattern]
        )
        
        # 3. Add the recognizer to the Analyzer Registry
        self.analyzer.registry.add_recognizer(emp_id_recognizer)
        logger.info("Custom PII recognizers loaded.")

    def redact_text(self, text: str) -> str:
        """
        Detects PII in the given text and replaces it with redacted tags.
        
        By default, the Microsoft Presidio English analyzer detects and redacts the following entities:
        - CREDIT_CARD
        - CRYPTO
        - DATE_TIME
        - EMAIL_ADDRESS
        - IBAN_CODE
        - IP_ADDRESS
        - NRP (Nationality, religious or political group)
        - LOCATION
        - PERSON
        - PHONE_NUMBER
        - MEDICAL_LICENSE
        - URL
        - US_BANK_NUMBER
        - US_DRIVER_LICENSE
        - US_ITIN
        - US_PASSPORT
        - US_SSN
        """
        if not text:
            return text
            
        try:
            # Detect PII entities in the text
            results = self.analyzer.analyze(text=text, language="en")
            
            # If no PII is found, return the original text
            if not results:
                return text
                
            # Anonymize the detected PII
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            logger.error(f"Error during PII redaction: {e}")
            return text  # Fallback to returning the original text if redaction fails
