import streamlit as st
import spacy
from spacy.training import Example
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import tempfile
import os
from io import BytesIO
import random
import plotly.graph_objects as go
import plotly.express as px

# Enhanced skill database with comprehensive coverage
class ComprehensiveSkillDatabase:
    def __init__(self):
        self.skills = self._initialize_comprehensive_skill_database()
        self.abbreviations = self._initialize_abbreviations()
        self.skill_patterns = self._initialize_skill_patterns()
        self.skill_relationships = self._initialize_skill_relationships()
        self.skill_variations = self._initialize_skill_variations()
    
    def _initialize_comprehensive_skill_database(self) -> Dict[str, List[str]]:
        return {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'Rust',
                'Go', 'Kotlin', 'Swift', 'Dart', 'Ruby', 'PHP', 'Scala', 'R', 'MATLAB',
                'Perl', 'Haskell', 'Elixir', 'Clojure', 'Julia', 'Lua', 'Objective-C',
                'F#', 'VB.NET', 'Solidity', 'Bash', 'PowerShell', 'SQL', 'HTML', 'CSS'
            ],
            'web_frameworks': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Flask',
                'FastAPI', 'Spring Boot', 'Spring', 'ASP.NET', '.NET Core', 'Ruby on Rails',
                'Laravel', 'Next.js', 'Nuxt.js', 'Svelte', 'SolidJS', 'Qwik', 'Gatsby',
                'Ember.js', 'Backbone.js', 'jQuery', 'Bootstrap', 'Tailwind CSS', 'Material-UI'
            ],
            'mobile_development': [
                'React Native', 'Flutter', 'Android Development', 'iOS Development',
                'SwiftUI', 'Kotlin Multiplatform', 'Xamarin', 'Ionic', 'Cordova',
                'Unity', 'Unreal Engine', 'Mobile UI/UX', 'App Development'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'Oracle',
                'SQL Server', 'SQLite', 'MariaDB', 'DynamoDB', 'Elasticsearch',
                'Firebase', 'Neo4j', 'Snowflake', 'BigQuery', 'CosmosDB', 'ClickHouse',
                'InfluxDB', 'CouchDB', 'Supabase', 'PlanetScale', 'FaunaDB'
            ],
            'ml_ai': [
                'Machine Learning', 'Deep Learning', 'Neural Networks', 'Natural Language Processing',
                'NLP', 'Computer Vision', 'Reinforcement Learning', 'Transfer Learning',
                'Feature Engineering', 'MLOps', 'Generative AI', 'Large Language Models',
                'LLM', 'CNN', 'RNN', 'LSTM', 'Transformer', 'BERT', 'GPT', 'Diffusion Models',
                'Prompt Engineering', 'AI Ethics', 'Explainable AI', 'Data Science', 'Analytics'
            ],
            'ml_frameworks': [
                'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'XGBoost', 'LightGBM',
                'CatBoost', 'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn',
                'Plotly', 'NLTK', 'spaCy', 'Hugging Face', 'OpenCV', 'LangChain', 'LlamaIndex',
                'MLflow', 'Kubeflow', 'Airflow', 'DVC', 'Weights & Biases'
            ],
            'cloud_platforms': [
                'AWS', 'Amazon Web Services', 'Azure', 'Microsoft Azure',
                'Google Cloud Platform', 'GCP', 'Heroku', 'DigitalOcean',
                'IBM Cloud', 'Oracle Cloud', 'Alibaba Cloud', 'Vercel', 'Netlify'
            ],
            'cloud_services': [
                'AWS Lambda', 'AWS S3', 'AWS EC2', 'Azure Functions', 'Google Cloud Functions',
                'Kubernetes', 'Docker', 'Terraform', 'Ansible', 'Jenkins', 'GitHub Actions',
                'GitLab CI', 'CircleCI', 'Prometheus', 'Grafana', 'Datadog', 'New Relic',
                'CloudFormation', 'Azure DevOps', 'Google Cloud Build', 'Serverless'
            ],
            'devops_tools': [
                'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions',
                'CircleCI', 'Ansible', 'Terraform', 'Prometheus', 'Grafana',
                'ELK Stack', 'Datadog', 'New Relic', 'Splunk', 'PagerDuty',
                'Bash', 'Shell Scripting', 'CI/CD', 'Infrastructure as Code'
            ],
            'version_control': [
                'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN', 'Mercurial',
                'Source Control', 'Version Management', 'Code Repository'
            ],
            'testing': [
                'Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium', 'Cypress',
                'Playwright', 'Postman', 'JMeter', 'LoadRunner', 'Cucumber',
                'Test-Driven Development', 'Behavior-Driven Development', 'Unit Testing',
                'Integration Testing', 'E2E Testing', 'Performance Testing'
            ],
            'soft_skills': [
                'Leadership', 'Team Management', 'Communication', 'Problem Solving',
                'Critical Thinking', 'Analytical Skills', 'Project Management',
                'Collaboration', 'Teamwork', 'Adaptability', 'Creativity',
                'Time Management', 'Emotional Intelligence', 'Conflict Resolution',
                'Strategic Thinking', 'Public Speaking', 'Mentoring', 'Negotiation',
                'Decision Making', 'Attention to Detail', 'Organization', 'Planning'
            ],
            'emerging_tech': [
                'Blockchain', 'Web3', 'Smart Contracts', 'IoT', 'Edge Computing',
                'Quantum Computing', 'AR/VR', 'Metaverse', 'Digital Twins',
                'Robotic Process Automation', 'RPA', '5G', 'Serverless Computing',
                'Microservices', 'GraphQL', 'WebAssembly', 'Low-Code/No-Code'
            ],
            'data_engineering': [
                'Data Warehousing', 'ETL', 'Data Pipeline', 'Apache Spark', 'Apache Hadoop',
                'Apache Kafka', 'Apache Flink', 'Data Lake', 'Data Modeling', 'Data Governance',
                'Streaming Analytics', 'Real-time Processing', 'Batch Processing'
            ],
            'cybersecurity': [
                'Network Security', 'Application Security', 'Cryptography', 'Penetration Testing',
                'Security Auditing', 'Compliance', 'Risk Assessment', 'Incident Response',
                'Security Operations', 'Threat Intelligence', 'Identity Management'
            ],
            'ui_ux': [
                'User Interface Design', 'User Experience Design', 'UI/UX', 'Prototyping',
                'Wireframing', 'Figma', 'Sketch', 'Adobe XD', 'InVision', 'Design Systems',
                'Responsive Design', 'Accessibility', 'Usability Testing'
            ],
            'business_intelligence': [
                'Tableau', 'Power BI', 'Looker', 'Qlik', 'Domo', 'Data Visualization',
                'Business Analytics', 'Reporting', 'Dashboarding', 'KPIs', 'Metrics'
            ]
        }
    
    def _initialize_abbreviations(self) -> Dict[str, str]:
        return {
            'ML': 'Machine Learning', 'DL': 'Deep Learning', 'AI': 'Artificial Intelligence',
            'NLP': 'Natural Language Processing', 'CV': 'Computer Vision', 'NN': 'Neural Networks',
            'CNN': 'Convolutional Neural Networks', 'RNN': 'Recurrent Neural Networks',
            'LLM': 'Large Language Models', 'GPT': 'Generative Pre-trained Transformer',
            'K8s': 'Kubernetes', 'CI/CD': 'Continuous Integration/Continuous Deployment',
            'API': 'Application Programming Interface', 'REST': 'Representational State Transfer',
            'SQL': 'Structured Query Language', 'NoSQL': 'Not Only SQL',
            'OOP': 'Object-Oriented Programming', 'FP': 'Functional Programming',
            'TDD': 'Test-Driven Development', 'BDD': 'Behavior-Driven Development',
            'AWS': 'Amazon Web Services', 'GCP': 'Google Cloud Platform',
            'RPA': 'Robotic Process Automation', 'IoT': 'Internet of Things',
            'AR': 'Augmented Reality', 'VR': 'Virtual Reality', 'UX': 'User Experience',
            'UI': 'User Interface', 'SaaS': 'Software as a Service', 'PaaS': 'Platform as a Service',
            'IaaS': 'Infrastructure as a Service', 'CRM': 'Customer Relationship Management',
            'ERP': 'Enterprise Resource Planning', 'BI': 'Business Intelligence'
        }
    
    def _initialize_skill_patterns(self) -> List[str]:
        return [
            r'experienced? (?:in|with) ([\w\s\+\#\.\-]+)',
            r'proficient (?:in|with|at) ([\w\s\+\#\.\-]+)',
            r'expertise (?:in|with) ([\w\s\+\#\.\-]+)',
            r'knowledge (?:of|in) ([\w\s\+\#\.\-]+)',
            r'skilled (?:in|at|with) ([\w\s\+\#\.\-]+)',
            r'familiar (?:with|in) ([\w\s\+\#\.\-]+)',
            r'strong (?:background|experience) (?:in|with) ([\w\s\+\#\.\-]+)',
            r'hands.on experience (?:in|with) ([\w\s\+\#\.\-]+)',
            r'(\d+)\+?\s*years? of (?:experience )?(?:in|with) ([\w\s\+\#\.\-]+)',
            r'working knowledge of ([\w\s\+\#\.\-]+)',
            r'proven ability (?:in|with) ([\w\s\+\#\.\-]+)',
            r'developed (?:using|with) ([\w\s\+\#\.\-]+)',
            r'built (?:using|with) ([\w\s\+\#\.\-]+)',
            r'implemented (?:using|with) ([\w\s\+\#\.\-]+)',
            r'created (?:using|with) ([\w\s\+\#\.\-]+)',
            r'designed (?:using|with) ([\w\s\+\#\.\-]+)',
            r'architected (?:using|with) ([\w\s\+\#\.\-]+)',
            r'worked (?:with|on) ([\w\s\+\#\.\-]+)',
            r'used ([\w\s\+\#\.\-]+) to',
            r'certified (?:in|for) ([\w\s\+\#\.\-]+)',
            r'specialized (?:in|with) ([\w\s\+\#\.\-]+)',
            r'focused (?:on|in) ([\w\s\+\#\.\-]+)'
        ]
    
    def _initialize_skill_relationships(self) -> Dict[str, List[str]]:
        return {
            'Python': ['Django', 'Flask', 'FastAPI', 'Pandas', 'NumPy', 'scikit-learn', 'PyTorch'],
            'JavaScript': ['React', 'Vue.js', 'Angular', 'Node.js', 'TypeScript', 'Express.js'],
            'Machine Learning': ['Deep Learning', 'TensorFlow', 'PyTorch', 'scikit-learn'],
            'AWS': ['AWS Lambda', 'AWS S3', 'AWS EC2', 'DynamoDB', 'CloudFormation'],
            'Docker': ['Kubernetes', 'Containerization', 'Microservices'],
            'React': ['React Native', 'Redux', 'Next.js', 'Webpack'],
            'Data Science': ['Python', 'R', 'Pandas', 'NumPy', 'Machine Learning'],
            'DevOps': ['CI/CD', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform'],
            'Frontend': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue.js', 'Angular'],
            'Backend': ['Node.js', 'Python', 'Java', 'Databases', 'APIs']
        }
    
    def _initialize_skill_variations(self) -> Dict[str, List[str]]:
        return {
            'Python': ['Python 3', 'Python 2', 'Py'],
            'JavaScript': ['JS', 'JavaScript ES6', 'ES6', 'JavaScript ES2015'],
            'TypeScript': ['TS'],
            'React': ['React.js', 'ReactJS'],
            'Node.js': ['Node', 'NodeJS'],
            'Machine Learning': ['ML'],
            'Deep Learning': ['DL'],
            'Natural Language Processing': ['NLP'],
            'Computer Vision': ['CV'],
            'User Interface': ['UI'],
            'User Experience': ['UX'],
            'Amazon Web Services': ['AWS'],
            'Google Cloud Platform': ['GCP'],
            'Microsoft Azure': ['Azure'],
            'Structured Query Language': ['SQL'],
            'Not Only SQL': ['NoSQL'],
            'Continuous Integration': ['CI'],
            'Continuous Deployment': ['CD'],
            'Application Programming Interface': ['API'],
            'Representational State Transfer': ['REST']
        }
    
    def get_all_skills(self) -> List[str]:
        all_skills = []
        for skills in self.skills.values():
            all_skills.extend(skills)
        return all_skills
    
    def get_category_for_skill(self, skill: str) -> Optional[str]:
        skill_lower = skill.lower()
        for category, skills in self.skills.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        return 'other'
    
    def get_related_skills(self, skill: str) -> List[str]:
        return self.skill_relationships.get(skill, [])
    
    def normalize_skill_name(self, skill: str) -> str:
        # Check if it's an abbreviation
        if skill.upper() in self.abbreviations:
            return self.abbreviations[skill.upper()]
        
        # Check if it's a variation
        skill_lower = skill.lower()
        for standard_skill, variations in self.skill_variations.items():
            if skill_lower in [v.lower() for v in variations]:
                return standard_skill
        
        return skill

# Enhanced text preprocessor
class EnhancedTextPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except OSError:
            st.warning("Using a lightweight spaCy fallback because the small English model is unavailable.")
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        
        self._customize_pipeline()
    
    def _customize_pipeline(self):
        # Add technical terms to the stop word list removal exception
        technical_terms = {
            'ai', 'ml', 'nlp', 'api', 'sql', 'nosql', 'http', 'https', 'css', 'html',
            'js', 'ts', 'ui', 'ux', 'ci', 'cd', 'aws', 'gcp', 'azure', 'k8s', 'gpu',
            'cpu', 'ram', 'ssd', 'hdd', 'os', 'db', 'id', 'url', 'uri', 'json', 'xml',
            'csv', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'png', 'jpg',
            'jpeg', 'gif', 'svg', 'mp4', 'mp3', 'zip', 'tar', 'gz', 'git', 'npm', 'yarn'
        }
        
        for term in technical_terms:
            if term in self.nlp.Defaults.stop_words:
                self.nlp.Defaults.stop_words.remove(term)
    
    def preprocess(self, text: str) -> Dict:
        if not text or not text.strip():
            return {'success': False, 'error': 'Empty text'}
        
        try:
            doc = self.nlp(text)
            
            # Extract noun chunks
            noun_chunks = []
            if doc.has_annotation("DEP"):
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 4:
                        noun_chunks.append(chunk.text)
            
            # Extract technical entities
            technical_entities = []
            if doc.ents:
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'TECHNOLOGY']:
                        technical_entities.append((ent.text, ent.label_))
            
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            
            # Extract tokens (non-stop words, non-punctuation)
            tokens = [token for token in doc if not token.is_stop and not token.is_punct]
            
            # Process text for analysis
            processed_text = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_space])
            
            return {
                'success': True,
                'doc': doc,
                'noun_chunks': noun_chunks,
                'entities': technical_entities,
                'sentences': sentences,
                'tokens': tokens,
                'processed_text': processed_text
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Advanced skill extractor with multiple methods
class AdvancedSkillExtractor:
    def __init__(self):
        self.skill_db = ComprehensiveSkillDatabase()
        self.preprocessor = EnhancedTextPreprocessor()
        self.embedder = None  # Lazy loading
        self.logger = self._setup_logger()
        self.custom_ner = None
    
    def _setup_logger(self):
        logger = logging.getLogger('AdvancedSkillExtractor')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _get_embedder(self):
        """Lazy loading of the sentence transformer model"""
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence-BERT model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Sentence-BERT model: {e}")
                self.embedder = None
        return self.embedder
    
    def _extract_by_enhanced_keywords(self, text: str) -> Set[str]:
        """Extract skills using keyword matching with variations and abbreviations"""
        found_skills = set()
        text_lower = text.lower()
        
        # Direct keyword matching
        for skill in self.skill_db.get_all_skills():
            if skill.lower() in text_lower:
                found_skills.add(skill)
        
        # Check for abbreviations
        for abbr, full_skill in self.skill_db.abbreviations.items():
            if abbr.lower() in text_lower:
                found_skills.add(full_skill)
        
        # Check for variations
        for standard_skill, variations in self.skill_db.skill_variations.items():
            for variation in variations:
                if variation.lower() in text_lower:
                    found_skills.add(standard_skill)
        
        return found_skills
    
    def _extract_by_advanced_pos_patterns(self, doc) -> Set[str]:
        """Extract skills using advanced part-of-speech patterns"""
        found_skills = set()
        
        # Pattern 1: Noun phrases that match skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            normalized_skill = self.skill_db.normalize_skill_name(chunk_text)
            if normalized_skill in self.skill_db.get_all_skills():
                found_skills.add(normalized_skill)
        
        # Pattern 2: Adjective + Noun combinations
        for token in doc:
            if token.pos_ == 'NOUN' and token.text in self.skill_db.get_all_skills():
                found_skills.add(token.text)
        
        # Pattern 3: Compound nouns
        for token in doc:
            if token.pos_ == 'NOUN' and len(token.text) > 2:
                # Check if token is part of a compound noun
                if token.dep_ == 'compound' and token.head.pos_ == 'NOUN':
                    compound = f"{token.text} {token.head.text}"
                    normalized_compound = self.skill_db.normalize_skill_name(compound)
                    if normalized_compound in self.skill_db.get_all_skills():
                        found_skills.add(normalized_compound)
        
        return found_skills
    
    def _extract_by_context_patterns(self, text: str) -> Set[str]:
        """Extract skills using context-based patterns"""
        found_skills = set()
        
        for pattern in self.skill_db.skill_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle patterns with multiple capture groups
                if len(match.groups()) > 1:
                    # For patterns like "(\d+) years of experience in (skill)"
                    skill = match.group(2).strip()
                else:
                    skill = match.group(1).strip()
                
                # Clean and normalize the skill
                skill = re.sub(r'[^\w\s\-\+\.]', '', skill)
                skill = skill.strip()
                
                if skill and len(skill) > 1:
                    normalized_skill = self.skill_db.normalize_skill_name(skill)
                    if normalized_skill in self.skill_db.get_all_skills():
                        found_skills.add(normalized_skill)
        
        return found_skills
    
    def _extract_by_enhanced_ner(self, entities: List[Tuple[str, str]]) -> Set[str]:
        """Extract skills using enhanced named entity recognition"""
        found_skills = set()
        
        for entity, label in entities:
            # Check if entity matches a known skill
            normalized_entity = self.skill_db.normalize_skill_name(entity)
            if normalized_entity in self.skill_db.get_all_skills():
                found_skills.add(normalized_entity)
            
            # Check if entity contains a skill
            for skill in self.skill_db.get_all_skills():
                if skill.lower() in entity.lower() or entity.lower() in skill.lower():
                    found_skills.add(skill)
        
        return found_skills
    
    def _extract_from_enhanced_chunks(self, noun_chunks: List[str]) -> Set[str]:
        """Extract skills from noun chunks with enhanced processing"""
        found_skills = set()
        
        for chunk in noun_chunks:
            # Direct match
            normalized_chunk = self.skill_db.normalize_skill_name(chunk)
            if normalized_chunk in self.skill_db.get_all_skills():
                found_skills.add(normalized_chunk)
            
            # Check if chunk contains a skill
            for skill in self.skill_db.get_all_skills():
                if skill.lower() in chunk.lower():
                    found_skills.add(skill)
            
            # Check for partial matches with length constraints
            if len(chunk.split()) <= 3:
                for skill in self.skill_db.get_all_skills():
                    # Check for high overlap
                    chunk_words = set(chunk.lower().split())
                    skill_words = set(skill.lower().split())
                    
                    if chunk_words and skill_words:
                        overlap = len(chunk_words.intersection(skill_words))
                        if overlap >= min(len(chunk_words), len(skill_words)) * 0.7:
                            found_skills.add(skill)
        
        return found_skills
    
    def _extract_by_semantic_similarity(self, text: str) -> Set[str]:
        """Extract skills using semantic similarity with Sentence-BERT"""
        found_skills = set()
        embedder = self._get_embedder()
        
        if embedder is None:
            self.logger.warning("Sentence-BERT model not available, skipping semantic extraction")
            return found_skills
        
        try:
            # Get all skills from database
            all_skills = self.skill_db.get_all_skills()
            
            # Encode text and skills
            text_embedding = embedder.encode([text])
            skill_embeddings = embedder.encode(all_skills)
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, skill_embeddings)[0]
            
            # Get skills with similarity above threshold
            threshold = 0.5  # Lowered threshold to catch more skills
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    found_skills.add(all_skills[i])
            
            self.logger.info(f"Semantic extraction found {len(found_skills)} skills")
        except Exception as e:
            self.logger.error(f"Error in semantic extraction: {e}")
        
        return found_skills
    
    def _extract_by_custom_ner(self, text: str) -> Set[str]:
        """Extract skills using custom NER model if available"""
        found_skills = set()
        
        if self.custom_ner is None:
            return found_skills
        
        try:
            predictions = self.custom_ner.predict(text)
            for skill, _, _ in predictions:
                normalized_skill = self.skill_db.normalize_skill_name(skill)
                if normalized_skill in self.skill_db.get_all_skills():
                    found_skills.add(normalized_skill)
        except Exception as e:
            self.logger.error(f"Error in custom NER extraction: {e}")
        
        return found_skills
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Validate if a string is a valid skill"""
        skill = skill.strip()
        
        # Length check
        if len(skill) < 2 or len(skill.split()) > 5:
            return False
        
        # Check against skill database
        skill_lower = skill.lower()
        for db_skill in self.skill_db.get_all_skills():
            if db_skill.lower() == skill_lower:
                return True
        
        # Check for partial matches
        for db_skill in self.skill_db.get_all_skills():
            if skill_lower in db_skill.lower() or db_skill.lower() in skill_lower:
                return True
        
        return False
    
    def _combine_and_deduplicate(self, skill_sets: List[Set[str]]) -> List[str]:
        """Combine and deduplicate skills from different extraction methods"""
        combined = set()
        for skill_set in skill_sets:
            combined.update(skill_set)
        
        # Normalize all skills
        normalized_skills = []
        for skill in combined:
            normalized_skill = self.skill_db.normalize_skill_name(skill)
            if normalized_skill and normalized_skill not in normalized_skills:
                normalized_skills.append(normalized_skill)
        
        return sorted(normalized_skills)
    
    def _enhanced_normalize_skills(self, skills: List[str]) -> List[str]:
        """Enhanced skill normalization with better handling of variations"""
        normalized = []
        
        for skill in skills:
            # Clean the skill
            skill_clean = skill.strip().title()
            
            # Normalize abbreviations
            skill_clean = self.skill_db.normalize_skill_name(skill_clean)
            
            # Add if valid and not duplicate
            if skill_clean and skill_clean not in normalized:
                normalized.append(skill_clean)
        
        return normalized
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different domains"""
        categorized = defaultdict(list)
        
        for skill in skills:
            category = self.skill_db.get_category_for_skill(skill)
            categorized[category].append(skill)
        
        return dict(categorized)
    
    def _calculate_advanced_confidence(self, skills: List[str], methods: List[Set[str]]) -> Dict[str, float]:
        """Calculate confidence scores for extracted skills"""
        confidence_scores = {}
        method_count = len(methods)
        
        for skill in skills:
            agreements = sum(1 for method in methods if skill in method)
            confidence = agreements / method_count
            confidence_scores[skill] = confidence
        
        return confidence_scores
    
    def _generate_skill_insights(self, skills: List[str], confidence_scores: Dict[str, float]) -> List[str]:
        """Generate insights about extracted skills"""
        insights = []
        
        if not skills:
            return ["No skills detected."]
        
        # High confidence skills
        high_conf_skills = [s for s, c in confidence_scores.items() if c >= 0.8]
        if high_conf_skills:
            insights.append(f"High-confidence skills detected: {', '.join(high_conf_skills[:5])}")
        
        # Technical skills analysis
        tech_skills = [s for s in skills if self.skill_db.get_category_for_skill(s) != 'soft_skills']
        if tech_skills:
            insights.append(f"Strong technical profile with {len(tech_skills)} technical skills")
            
            # Check for trending skills
            trending_skills = ['Python', 'Machine Learning', 'AWS', 'Docker', 'Kubernetes', 'React']
            trending_found = [s for s in tech_skills if s in trending_skills]
            if trending_found:
                insights.append(f"Trending skills: {', '.join(trending_found)}")
        
        # Soft skills analysis
        soft_skills = [s for s in skills if self.skill_db.get_category_for_skill(s) == 'soft_skills']
        if soft_skills:
            insights.append(f"Soft skills identified: {', '.join(soft_skills[:3])}")
        
        # Skill diversity
        categories = set(self.skill_db.get_category_for_skill(s) for s in skills)
        if len(categories) > 3:
            insights.append(f"Diverse skill set across {len(categories)} categories")
        
        return insights
    
    def extract_skills(self, text: str, document_type: str = 'resume') -> Dict:
        """Main method to extract skills using multiple approaches"""
        try:
            start_time = datetime.now()
            
            # Preprocess text
            preprocess_result = self.preprocessor.preprocess(text)
            if not preprocess_result['success']:
                return {'success': False, 'error': preprocess_result['error']}
            
            doc = preprocess_result['doc']
            
            # Extract skills using different methods
            keyword_skills = self._extract_by_enhanced_keywords(text)
            pos_skills = self._extract_by_advanced_pos_patterns(doc)
            context_skills = self._extract_by_context_patterns(text)
            ner_skills = self._extract_by_enhanced_ner(preprocess_result['entities'])
            chunk_skills = self._extract_from_enhanced_chunks(preprocess_result['noun_chunks'])
            semantic_skills = self._extract_by_semantic_similarity(text)
            custom_ner_skills = self._extract_by_custom_ner(text)
            
            # Combine all methods
            all_methods = [
                keyword_skills, 
                pos_skills, 
                context_skills, 
                ner_skills, 
                chunk_skills, 
                semantic_skills
            ]
            
            if self.custom_ner:
                all_methods.append(custom_ner_skills)
            
            # Combine and normalize skills
            all_skills = self._combine_and_deduplicate(all_methods)
            normalized_skills = self._enhanced_normalize_skills(all_skills)
            categorized_skills = self._categorize_skills(normalized_skills)
            skill_confidence = self._calculate_advanced_confidence(normalized_skills, all_methods)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Generate extraction method statistics
            extraction_methods = {
                'keyword_matching': len(keyword_skills),
                'pos_patterns': len(pos_skills),
                'context_based': len(context_skills),
                'ner': len(ner_skills),
                'noun_chunks': len(chunk_skills),
                'semantic_matching': len(semantic_skills)
            }
            
            if self.custom_ner:
                extraction_methods['custom_ner'] = len(custom_ner_skills)
            
            # Generate insights
            insights = self._generate_skill_insights(normalized_skills, skill_confidence)
            
            return {
                'success': True,
                'all_skills': normalized_skills,
                'categorized_skills': categorized_skills,
                'skill_confidence': skill_confidence,
                'skill_insights': insights,
                'extraction_methods': extraction_methods,
                'statistics': {
                    'total_skills': len(normalized_skills),
                    'technical_skills': sum(len(skills) for cat, skills in categorized_skills.items() 
                                           if cat not in ['soft_skills', 'other']),
                    'soft_skills': len(categorized_skills.get('soft_skills', [])),
                    'emerging_tech_skills': len(categorized_skills.get('emerging_tech', [])),
                    'high_confidence_skills': sum(1 for conf in skill_confidence.values() if conf >= 0.8),
                    'processing_time': processing_time,
                    'custom_ner_used': bool(self.custom_ner)
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Custom NER trainer for skill extraction
class CustomSkillNERTrainer:
    def __init__(self):
        self.nlp = None
        self.ner = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('CustomSkillNERTrainer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_blank_model(self):
        """Create a blank spaCy model with NER component"""
        self.nlp = spacy.blank("en")
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        # Add the SKILL label
        self.ner.add_label("SKILL")
    
    def train(self, training_data: List[Tuple], n_iterations: int = 30) -> Dict:
        """Train a custom NER model for skill extraction"""
        if not self.nlp:
            self.create_blank_model()
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        training_stats = {'losses': [], 'iterations': n_iterations}
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iterations):
                random.shuffle(training_data)
                losses = {}
                
                for text, annotations in training_data:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self.nlp.update([example], drop=0.5, losses=losses)
                
                training_stats['losses'].append(losses.get('ner', 0))
                self.logger.info(f"Iteration {iteration+1}/{n_iterations}, Loss: {losses.get('ner', 0)}")
        
        return training_stats
    
    def predict(self, text: str) -> List[Tuple[str, int, int]]:
        """Predict skills in text using the trained model"""
        if not self.nlp:
            raise ValueError("Model not trained")
        
        doc = self.nlp(text)
        return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "SKILL"]
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.nlp:
            self.nlp.to_disk(path)
            self.logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model"""
        instance = cls()
        try:
            instance.nlp = spacy.load(path)
            instance.ner = instance.nlp.get_pipe("ner")
            instance.logger.info(f"Model loaded from {path}")
            return instance
        except Exception as e:
            instance.logger.error(f"Failed to load model: {e}")
            return None

# Annotation interface for creating NER training data
class AnnotationInterface:
    """Interface for creating NER training data"""
    
    def __init__(self):
        if 'training_annotations' not in st.session_state:
            st.session_state.training_annotations = []
        if 'current_skills' not in st.session_state:
            st.session_state.current_skills = []
    
    def create_annotation_ui(self):
        """Create annotation UI"""
        st.subheader("🏷️ Create NER Training Data")
        
        st.markdown("""
        **Instructions:**
        1. Enter text containing skills
        2. Mark skill positions (start/end character indices)
        3. Add to training dataset
        4. Export for model training
        """)
        
        input_text = st.text_area(
            "Enter text to annotate:",
            height=150,
            placeholder="Example: I am a Python developer with 5 years of Machine Learning experience."
        )
        
        if input_text:
            st.markdown("---")
            st.text(input_text)
            
            with st.form("skill_annotation_form"):
                st.markdown("**Add Skill Annotation:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    skill_text = st.text_input("Skill text")
                with col2:
                    start_pos = st.number_input("Start position", min_value=0, value=0)
                with col3:
                    end_pos = st.number_input("End position", min_value=0, value=0)
                
                if skill_text and start_pos < end_pos:
                    extracted = input_text[start_pos:end_pos]
                    if extracted.strip():
                        st.info(f"Preview: '{extracted}'")
                
                submitted = st.form_submit_button("➕ Add Skill")
                
                if submitted and skill_text and start_pos < end_pos:
                    st.session_state.current_skills.append({
                        'text': skill_text,
                        'start': start_pos,
                        'end': end_pos,
                        'label': 'SKILL'
                    })
                    st.success(f"✅ Added: {skill_text}")
                    st.rerun()
            
            if st.session_state.current_skills:
                st.markdown("**Skills in current text:**")
                for i, skill in enumerate(st.session_state.current_skills):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i+1}. **{skill['text']}** ({skill['start']}-{skill['end']})")
                    with col2:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state.current_skills.pop(i)
                            st.rerun()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Annotation", type="primary"):
                    if st.session_state.current_skills:
                        annotation = {
                            'text': input_text,
                            'skills': st.session_state.current_skills.copy(),
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.training_annotations.append(annotation)
                        st.session_state.current_skills = []
                        st.success(f"✅ Saved! Total: {len(st.session_state.training_annotations)}")
                        st.rerun()
            
            with col2:
                if st.button("🔄 Clear Current"):
                    st.session_state.current_skills = []
                    st.rerun()
        
        if st.session_state.training_annotations:
            st.markdown("---")
            st.subheader(f"📚 Training Dataset ({len(st.session_state.training_annotations)} annotations)")
            
            for i, annotation in enumerate(st.session_state.training_annotations):
                with st.expander(f"Annotation {i+1}: {len(annotation['skills'])} skills"):
                    st.text(annotation['text'])
                    st.write("**Skills:**")
                    for skill in annotation['skills']:
                        st.write(f"- {skill['text']} ({skill['start']}-{skill['end']})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                training_json = json.dumps(st.session_state.training_annotations, indent=2)
                st.download_button(
                    "📥 Download Training Data (JSON)",
                    training_json,
                    f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            with col2:
                trainer = CustomSkillNERTrainer()
                spacy_format = trainer.prepare_training_data(st.session_state.training_annotations)
                spacy_json = json.dumps(spacy_format, indent=2)
                
                st.download_button(
                    "📥 Download spaCy Format",
                    spacy_json,
                    f"spacy_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    
    def prepare_training_data(self, annotations: List[Dict]) -> List[Tuple]:
        """Convert annotations to spaCy format"""
        training_data = []
        
        for annotation in annotations:
            text = annotation['text']
            entities = []
            
            for skill in annotation['skills']:
                entities.append((skill['start'], skill['end'], skill['label']))
            
            training_data.append((text, {"entities": entities}))
        
        return training_data

# BERT embeddings for skill similarity analysis
class SentenceBERTEmbedder:
    """Generate and manage BERT embeddings for skills"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize Sentence-BERT model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.skill_embeddings = {}
        except Exception as e:
            st.error(f"❌ Failed to load BERT model: {e}")
            st.info("Installing sentence-transformers...")
            import subprocess
            subprocess.run(["pip", "install", "sentence-transformers"])
            self.model = SentenceTransformer(model_name)
    
    def encode_skills(self, skills: List[str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for skills"""
        if not skills:
            return {}
        
        embeddings = self.model.encode(skills, show_progress_bar=True)
        
        skill_embeddings = {}
        for skill, embedding in zip(skills, embeddings):
            skill_embeddings[skill] = embedding
            self.skill_embeddings[skill] = embedding
        
        return skill_embeddings
    
    def compute_similarity(self, skill1: str, skill2: str) -> float:
        """Compute cosine similarity between two skills"""
        if skill1 not in self.skill_embeddings:
            emb1 = self.model.encode([skill1])[0]
            self.skill_embeddings[skill1] = emb1
        else:
            emb1 = self.skill_embeddings[skill1]
        
        if skill2 not in self.skill_embeddings:
            emb2 = self.model.encode([skill2])[0]
            self.skill_embeddings[skill2] = emb2
        else:
            emb2 = self.skill_embeddings[skill2]
        
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def compute_similarity_matrix(self, skills1: List[str], skills2: List[str]) -> np.ndarray:
        """Compute similarity matrix between two skill sets"""
        embeddings1 = self.model.encode(skills1)
        embeddings2 = self.model.encode(skills2)
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_similar_skills(self, target_skill: str, skill_list: List[str], 
                           threshold: float = 0.7, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find skills similar to target skill"""
        similarities = []
        
        for skill in skill_list:
            if skill.lower() != target_skill.lower():
                sim = self.compute_similarity(target_skill, skill)
                if sim >= threshold:
                    similarities.append((skill, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Skill visualizer with comprehensive charts
class SkillVisualizer:
    """Visualize extracted skills"""
    
    @staticmethod
    def create_category_distribution_chart(categorized_skills: Dict[str, List[str]]) -> go.Figure:
        """Create pie chart for category distribution"""
        category_names = {
            'programming_languages': 'Programming Languages',
            'web_frameworks': 'Web Frameworks',
            'databases': 'Databases',
            'ml_ai': 'ML/AI',
            'ml_frameworks': 'ML Frameworks',
            'cloud_platforms': 'Cloud Platforms',
            'devops_tools': 'DevOps Tools',
            'version_control': 'Version Control',
            'testing': 'Testing',
            'soft_skills': 'Soft Skills',
            'other': 'Other'
        }
        
        categories = []
        counts = []
        
        for category, skills in categorized_skills.items():
            if skills:
                categories.append(category_names.get(category, category.replace('_', ' ').title()))
                counts.append(len(skills))
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            hole=0.3,
            textposition='auto',
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(title="Skill Distribution by Category", height=500)
        return fig
    
    @staticmethod
    def create_top_skills_chart(skills: List[str], confidence_scores: Dict[str, float], top_n: int = 15) -> go.Figure:
        """Create bar chart for top skills"""
        sorted_skills = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        skill_names = [skill for skill, _ in sorted_skills]
        confidences = [score * 100 for _, score in sorted_skills]
        
        fig = go.Figure(data=[go.Bar(
            x=confidences,
            y=skill_names,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='Viridis',
                colorbar=dict(title="Confidence %")
            ),
            text=[f"{conf:.0f}%" for conf in confidences],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Skills by Confidence Score",
            xaxis_title="Confidence Score (%)",
            yaxis_title="Skills",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    @staticmethod
    def create_extraction_methods_chart(extraction_methods: Dict[str, int]) -> go.Figure:
        """Create bar chart for extraction methods"""
        method_names = {
            'keyword_matching': 'Keyword Matching',
            'pos_patterns': 'POS Patterns',
            'context_based': 'Context-Based',
            'ner': 'Named Entity Recognition',
            'noun_chunks': 'Noun Chunks',
            'semantic_matching': 'Semantic Matching'
        }
        
        methods = [method_names.get(m, m) for m in extraction_methods.keys()]
        counts = list(extraction_methods.values())
        
        fig = go.Figure(data=[go.Bar(
            x=methods,
            y=counts,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Skills Detected by Each Extraction Method",
            xaxis_title="Extraction Method",
            yaxis_title="Number of Skills Found",
            height=400
        )
        
        return fig

# Enhanced skill gap analyzer
class EnhancedSkillGapAnalyzer:
    def __init__(self):
        self.skill_extractor = AdvancedSkillExtractor()
        self.embedder = None  # Lazy loading
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('EnhancedSkillGapAnalyzer')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _get_embedder(self):
        """Lazy loading of the sentence transformer model"""
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence-BERT model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Sentence-BERT model: {e}")
                self.embedder = None
        return self.embedder
    
    def analyze_skill_gap(self, resume_text: str, jd_text: str) -> Dict:
        """Analyze the skill gap between resume and job description"""
        try:
            # Extract skills from resume
            resume_result = self.skill_extractor.extract_skills(resume_text, 'resume')
            if not resume_result['success']:
                return {'success': False, 'error': f"Resume skill extraction failed: {resume_result['error']}"}
            
            # Extract skills from job description
            jd_result = self.skill_extractor.extract_skills(jd_text, 'job_description')
            if not jd_result['success']:
                return {'success': False, 'error': f"JD skill extraction failed: {jd_result['error']}"}
            
            resume_skills = resume_result['all_skills']
            jd_skills = jd_result['all_skills']
            
            # Calculate skill gap
            matched_skills = []
            partial_matches = []
            missing_skills = []
            
            # Get embedder for semantic matching
            embedder = self._get_embedder()
            
            if embedder:
                # Use semantic matching
                resume_embeddings = embedder.encode(resume_skills)
                jd_embeddings = embedder.encode(jd_skills)
                similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
                
                for i, jd_skill in enumerate(jd_skills):
                    # Find the best matching resume skill
                    best_match_idx = np.argmax(similarity_matrix[:, i])
                    best_match_score = similarity_matrix[best_match_idx, i]
                    
                    if best_match_score >= 0.8:
                        matched_skills.append({
                            'jd_skill': jd_skill,
                            'resume_skill': resume_skills[best_match_idx],
                            'similarity': float(best_match_score)
                        })
                    elif best_match_score >= 0.5:
                        partial_matches.append({
                            'jd_skill': jd_skill,
                            'resume_skill': resume_skills[best_match_idx],
                            'similarity': float(best_match_score)
                        })
                    else:
                        missing_skills.append({
                            'jd_skill': jd_skill,
                            'resume_skill': resume_skills[best_match_idx] if best_match_score > 0.3 else None,
                            'similarity': float(best_match_score)
                        })
            else:
                # Fallback to exact matching
                resume_skills_set = set(resume_skills)
                
                for jd_skill in jd_skills:
                    if jd_skill in resume_skills_set:
                        matched_skills.append({
                            'jd_skill': jd_skill,
                            'resume_skill': jd_skill,
                            'similarity': 1.0
                        })
                    else:
                        # Check for partial matches
                        partial_match = None
                        for resume_skill in resume_skills:
                            if jd_skill.lower() in resume_skill.lower() or resume_skill.lower() in jd_skill.lower():
                                partial_match = resume_skill
                                break
                        
                        if partial_match:
                            partial_matches.append({
                                'jd_skill': jd_skill,
                                'resume_skill': partial_match,
                                'similarity': 0.6
                            })
                        else:
                            missing_skills.append({
                                'jd_skill': jd_skill,
                                'resume_skill': None,
                                'similarity': 0.0
                            })
            
            # Calculate overall score
            total_skills = len(jd_skills)
            matched_count = len(matched_skills)
            partial_count = len(partial_matches)
            
            overall_score = (matched_count * 1.0 + partial_count * 0.5) / total_skills * 100 if total_skills > 0 else 0
            
            # Categorize missing skills by priority
            high_priority_missing = []
            medium_priority_missing = []
            low_priority_missing = []
            
            for skill in missing_skills:
                jd_skill = skill['jd_skill']
                category = self.skill_extractor.skill_db.get_category_for_skill(jd_skill)
                
                # Determine priority based on category and similarity
                if category in ['programming_languages', 'ml_ai', 'cloud_platforms']:
                    high_priority_missing.append(skill)
                elif category in ['web_frameworks', 'databases', 'devops_tools']:
                    medium_priority_missing.append(skill)
                else:
                    low_priority_missing.append(skill)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                matched_skills, partial_matches, missing_skills,
                resume_result['categorized_skills'], jd_result['categorized_skills']
            )
            
            return {
                'success': True,
                'resume_skills': resume_result,
                'jd_skills': jd_result,
                'matched_skills': matched_skills,
                'partial_matches': partial_matches,
                'missing_skills': missing_skills,
                'high_priority_missing': high_priority_missing,
                'medium_priority_missing': medium_priority_missing,
                'low_priority_missing': low_priority_missing,
                'overall_score': overall_score,
                'statistics': {
                    'total_resume_skills': len(resume_skills),
                    'total_jd_skills': len(jd_skills),
                    'matched_count': matched_count,
                    'partial_count': partial_count,
                    'missing_count': len(missing_skills),
                    'match_percentage': (matched_count / total_skills * 100) if total_skills > 0 else 0
                },
                'recommendations': recommendations
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_recommendations(self, matched_skills, partial_matches, missing_skills, 
                                 resume_categories, jd_categories) -> List[Dict]:
        """Generate personalized recommendations based on skill gap analysis"""
        recommendations = []
        
        # Recommendations for missing high-priority skills
        if missing_skills:
            recommendations.append({
                'type': 'critical_skills',
                'title': 'Focus on These Critical Skills',
                'description': 'These skills are mentioned in the job description but missing from your resume. Consider highlighting any relevant experience or acquiring these skills.',
                'skills': [skill['jd_skill'] for skill in missing_skills[:5]]
            })
        
        # Recommendations for partial matches
        if partial_matches:
            recommendations.append({
                'type': 'improve_partial_matches',
                'title': 'Strengthen These Skills',
                'description': 'You have some experience with these skills, but they could be highlighted more effectively in your resume.',
                'skills': [f"{skill['jd_skill']} (currently: {skill['resume_skill']})" for skill in partial_matches[:5]]
            })
        
        # Recommendations for skill categories
        missing_categories = set()
        for skill in missing_skills:
            category = self.skill_extractor.skill_db.get_category_for_skill(skill['jd_skill'])
            missing_categories.add(category)
        
        if missing_categories:
            recommendations.append({
                'type': 'category_gaps',
                'title': 'Address These Skill Categories',
                'description': f'Consider highlighting experience in these categories: {", ".join(missing_categories)}',
                'categories': list(missing_categories)
            })
        
        # Recommendations for related skills
        if matched_skills:
            # Find related skills that might be missing
            related_missing = []
            for match in matched_skills[:3]:  # Check top 3 matches
                resume_skill = match['resume_skill']
                related_skills = self.skill_extractor.skill_db.get_related_skills(resume_skill)
                
                for related in related_skills:
                    # Check if related skill is in JD but not in resume
                    for jd_skill in [skill['jd_skill'] for skill in missing_skills]:
                        if related.lower() == jd_skill.lower() and related not in [m['jd_skill'] for m in related_missing]:
                            related_missing.append({'jd_skill': related, 'related_to': resume_skill})
            
            if related_missing:
                recommendations.append({
                    'type': 'related_skills',
                    'title': 'Leverage Your Existing Skills',
                    'description': 'These skills are related to what you already know and are mentioned in the job description.',
                    'skills': [f"{skill['jd_skill']} (related to {skill['related_to']})" for skill in related_missing[:3]]
                })
        
        return recommendations

# Export functionality
class ExportManager:
    """Handle export functionality for skills and analysis results"""
    
    @staticmethod
    def create_csv_export(result: Dict) -> str:
        """Create CSV export"""
        data = []
        
        for skill in result['all_skills']:
            category = result['categorized_skills'].get(skill, ['unknown'])[0]
            confidence = result['skill_confidence'].get(skill, 0)
            
            data.append({
                'Skill': skill,
                'Category': category,
                'Confidence': confidence,
                'Type': 'Soft Skill' if category == 'soft_skills' else 'Technical Skill'
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    @staticmethod
    def create_json_export(result: Dict) -> str:
        """Create JSON export"""
        export_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'statistics': result['statistics'],
            'skills': {
                'all_skills': result['all_skills'],
                'categorized_skills': result['categorized_skills'],
                'skill_confidence': result['skill_confidence']
            },
            'extraction_methods': result['extraction_methods']
        }
        
        return json.dumps(export_data, indent=2)
    
    @staticmethod
    def create_text_report(result: Dict) -> str:
        """Create formatted text report"""
        report = []
        report.append("=" * 80)
        report.append("SKILL EXTRACTION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Skills Extracted: {result['statistics']['total_skills']}")
        report.append(f"Technical Skills: {result['statistics']['technical_skills']}")
        report.append(f"Soft Skills: {result['statistics']['soft_skills']}")
        report.append("\n" + "-" * 80)
        report.append("\nCATEGORIZED SKILLS")
        report.append("-" * 80)
        
        for category, skills in sorted(result['categorized_skills'].items()):
            if skills:
                report.append(f"\n{category.replace('_', ' ').title()} ({len(skills)}):")
                for skill in skills:
                    confidence = result['skill_confidence'].get(skill, 0)
                    report.append(f"  • {skill} (Confidence: {confidence:.0%})")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)

# Main application
def main():
    st.set_page_config(
        page_title="Enhanced AI Skill Gap Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedSkillGapAnalyzer()
    
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    
    if 'skill_embeddings' not in st.session_state:
        st.session_state.skill_embeddings = None
    
    if 'trained_ner' not in st.session_state:
        st.session_state.trained_ner = None
    
    if 'bert_embedder' not in st.session_state:
        st.session_state.bert_embedder = None
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #ff7f0e;
            text-align: center;
            margin-bottom: 2rem;
        }
        .skill-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #1f77b4;
        }
        .matched-skill {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .partial-skill {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .missing-skill {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .recommendation-card {
            background-color: #e7f3ff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #0066cc;
        }
        .skill-tag {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            background-color: #e1f5ff;
            border-radius: 15px;
            font-size: 14px;
        }
        .tech-skill {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .soft-skill {
            background-color: #f3e5f5;
            color: #7b1fa2;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize components
    visualizer = SkillVisualizer()
    annotator = AnnotationInterface()
    export_manager = ExportManager()
    
    # Main title
    st.markdown('<h1 class="main-header">🧠 Enhanced AI Skill Gap Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Complete Milestone 2 Implementation</h2>', unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "📄 Skill Extraction",
        "📊 Skill Gap Analysis",
        "🧠 BERT Embeddings",
        "🏋️ Train Custom NER",
        "🏷️ Annotate Data",
        "📈 Visualizations",
        "📥 Export"
    ])
    
    with tabs[0]:
        st.header("Extract Skills from Text")
        
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"],
            horizontal=True
        )
        
        text_input = ""
        doc_type = "resume"
        
        if input_method == "Paste Text":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                text_input = st.text_area(
                    "Paste resume or job description text:",
                    height=300,
                    placeholder="Paste your resume or job description here..."
                )
            
            with col2:
                doc_type = st.selectbox("Document Type:", ["resume", "job_description"])
        
        else:
            uploaded_file = st.file_uploader("Upload document", type=['txt'])
            doc_type = st.selectbox("Document Type:", ["resume", "job_description"])
            
            if uploaded_file:
                text_input = uploaded_file.getvalue().decode("utf-8")
        
        if st.button("🔍 Extract Skills", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("Extracting skills..."):
                    result = st.session_state.analyzer.skill_extractor.extract_skills(text_input, doc_type)
                    
                    if result['success']:
                        st.session_state.extraction_results = result
                        st.success(f"✅ Successfully extracted {result['statistics']['total_skills']} skills!")
                        
                        # Display results
                        st.subheader("📊 Extraction Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Skills", result['statistics']['total_skills'])
                        with col2:
                            st.metric("Technical Skills", result['statistics']['technical_skills'])
                        with col3:
                            st.metric("Soft Skills", result['statistics']['soft_skills'])
                        with col4:
                            avg_confidence = sum(result['skill_confidence'].values()) / len(result['skill_confidence']) if result['skill_confidence'] else 0
                            st.metric("Avg Confidence", f"{avg_confidence:.0%}")
                        
                        st.subheader("🏷️ Categorized Skills")
                        
                        categorized = result['categorized_skills']
                        category_items = list(categorized.items())
                        
                        for i in range(0, len(category_items), 2):
                            cols = st.columns(2)
                            
                            for j, col in enumerate(cols):
                                if i + j < len(category_items):
                                    category, skills = category_items[i + j]
                                    
                                    with col:
                                        category_display = category.replace('_', ' ').title()
                                        st.markdown(f"**{category_display}** ({len(skills)})")
                                        
                                        skill_html = ""
                                        for skill in skills[:10]:
                                            confidence = result['skill_confidence'].get(skill, 0)
                                            color = "tech-skill" if category != "soft_skills" else "soft-skill"
                                            skill_html += f'<span class="skill-tag {color}" title="Confidence: {confidence:.0%}">{skill}</span>'
                                        
                                        if len(skills) > 10:
                                            skill_html += f'<span class="skill-tag">+{len(skills) - 10} more</span>'
                                        
                                        st.markdown(skill_html, unsafe_allow_html=True)
                                        st.markdown("")
                        
                        with st.expander("🔧 Extraction Methods Used"):
                            methods_df = pd.DataFrame([
                                {'Method': method.replace('_', ' ').title(), 'Skills Found': count}
                                for method, count in result['extraction_methods'].items()
                            ])
                            st.dataframe(methods_df, use_container_width=True)
                    else:
                        st.error(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Please provide text input")
    
    with tabs[1]:
        st.header("Skill Gap Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            resume_text = st.text_area(
                "Paste Resume Text:",
                height=300,
                placeholder="Paste your resume here..."
            )
        
        with col2:
            jd_text = st.text_area(
                "Paste Job Description:",
                height=300,
                placeholder="Paste the job description here..."
            )
        
        if st.button("📊 Analyze Skill Gap", type="primary", use_container_width=True):
            if resume_text and jd_text:
                with st.spinner("Analyzing skill gap..."):
                    result = st.session_state.analyzer.analyze_skill_gap(resume_text, jd_text)
                    
                    if result['success']:
                        st.session_state.analysis_result = result
                        
                        # Display overall score
                        st.subheader("📊 Overall Match Score")
                        score_color = "green" if result['overall_score'] >= 70 else "orange" if result['overall_score'] >= 50 else "red"
                        st.markdown(f"<h1 style='color: {score_color}; text-align: center;'>{result['overall_score']:.1f}%</h1>", unsafe_allow_html=True)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total JD Skills", result['statistics']['total_jd_skills'])
                        with col2:
                            st.metric("Matched Skills", result['statistics']['matched_count'])
                        with col3:
                            st.metric("Partial Matches", result['statistics']['partial_count'])
                        with col4:
                            st.metric("Missing Skills", result['statistics']['missing_count'])
                        
                        # Matched skills
                        if result['matched_skills']:
                            st.subheader("✅ Matched Skills")
                            for skill in result['matched_skills']:
                                st.markdown(f"""
                                <div class="skill-card matched-skill">
                                    <strong>{skill['jd_skill']}</strong> 
                                    <span style="float: right;">{skill['similarity']:.0%}</span>
                                    <br><small>Resume: {skill['resume_skill']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Partial matches
                        if result['partial_matches']:
                            st.subheader("⚠️ Partial Matches")
                            for skill in result['partial_matches']:
                                st.markdown(f"""
                                <div class="skill-card partial-skill">
                                    <strong>{skill['jd_skill']}</strong> 
                                    <span style="float: right;">{skill['similarity']:.0%}</span>
                                    <br><small>Resume: {skill['resume_skill']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Missing skills
                        if result['missing_skills']:
                            st.subheader("❌ Missing Skills")
                            for skill in result['missing_skills'][:10]:  # Show top 10
                                st.markdown(f"""
                                <div class="skill-card missing-skill">
                                    <strong>{skill['jd_skill']}</strong>
                                    {f"<br><small>Similar to: {skill['resume_skill']}</small>" if skill['resume_skill'] else ""}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Recommendations
                        if result['recommendations']:
                            st.subheader("💡 Recommendations")
                            for rec in result['recommendations']:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{rec['title']}</h4>
                                    <p>{rec['description']}</p>
                                    <ul>
                                        {"".join([f"<li>{skill}</li>" for skill in rec.get('skills', [])])}
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Please provide both resume and job description text")
    
    with tabs[2]:
        st.header("🧠 BERT Embeddings & Similarity Analysis")
        
        if not st.session_state.extraction_results:
            st.info("👆 Extract skills first to generate embeddings")
            return
        
        result = st.session_state.extraction_results
        skills = result['all_skills']
        
        # Initialize BERT embedder if needed
        if st.session_state.bert_embedder is None:
            st.session_state.bert_embedder = SentenceBERTEmbedder()
        
        if st.button("🚀 Generate BERT Embeddings", type="primary"):
            with st.spinner("Generating embeddings..."):
                embeddings = st.session_state.bert_embedder.encode_skills(skills)
                st.session_state.skill_embeddings = embeddings
                st.success(f"✅ Generated embeddings for {len(skills)} skills!")
        
        if st.session_state.skill_embeddings:
            st.subheader("🔍 Skill Similarity Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                skill1 = st.selectbox("Select first skill:", skills, key="sim_skill1")
            
            with col2:
                skill2 = st.selectbox("Select second skill:", skills, key="sim_skill2")
            
            if st.button("Calculate Similarity"):
                similarity = st.session_state.bert_embedder.compute_similarity(skill1, skill2)
                
                st.metric(
                    "Similarity Score",
                    f"{similarity:.2%}",
                    delta=f"{'High' if similarity > 0.7 else 'Medium' if similarity > 0.4 else 'Low'} similarity"
                )
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=similarity * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Similarity"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Find Similar Skills")
            
            target_skill = st.selectbox("Select target skill:", skills, key="target_skill")
            threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.05)
            
            if st.button("Find Similar Skills"):
                similar_skills = st.session_state.bert_embedder.find_similar_skills(
                    target_skill,
                    [s for s in skills if s != target_skill],
                    threshold=threshold,
                    top_k=10
                )
                
                if similar_skills:
                    st.success(f"Found {len(similar_skills)} similar skills:")
                    
                    df = pd.DataFrame(similar_skills, columns=['Skill', 'Similarity'])
                    df['Similarity'] = df['Similarity'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=[s[1] for s in similar_skills],
                        y=[s[0] for s in similar_skills],
                        orientation='h',
                        marker_color='lightblue'
                    )])
                    
                    fig.update_layout(
                        title=f"Skills Similar to '{target_skill}'",
                        xaxis_title="Similarity Score",
                        yaxis_title="Skill",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No skills found with similarity >= {threshold:.0%}")
            
            st.subheader("📊 Skill Similarity Matrix")
            
            if st.button("Generate Similarity Matrix"):
                with st.spinner("Computing similarities..."):
                    similarity_matrix = st.session_state.bert_embedder.compute_similarity_matrix(
                        skills[:20],
                        skills[:20]
                    )
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=similarity_matrix,
                        x=skills[:20],
                        y=skills[:20],
                        colorscale='Viridis',
                        text=similarity_matrix,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 8}
                    ))
                    
                    fig.update_layout(
                        title="Skill Similarity Heatmap (Top 20 Skills)",
                        height=700,
                        width=800
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.header("🏋️ Train Custom NER Model")
        
        st.markdown("""
        Train a custom spaCy NER model to detect skills in text.
        
        **Steps:**
        1. Load training data (use Annotate Data tab)
        2. Configure training parameters
        3. Train the model
        4. Test the model
        """)
        
        st.subheader("1️⃣ Load Training Data")
        
        training_source = st.radio(
            "Training data source:",
            ["Use Annotated Data", "Upload JSON File"],
            horizontal=True
        )
        
        training_data = None
        
        if training_source == "Use Annotated Data":
            if st.session_state.get('training_annotations'):
                st.success(f"✅ {len(st.session_state.training_annotations)} annotations available")
                training_data = annotator.prepare_training_data(
                    st.session_state.training_annotations
                )
            else:
                st.warning("⚠️ No annotations found. Use 'Annotate Data' tab first.")
        else:
            uploaded_file = st.file_uploader("Upload training data (JSON)", type=['json'])
            if uploaded_file:
                try:
                    annotations = json.load(uploaded_file)
                    training_data = annotator.prepare_training_data(annotations)
                    st.success(f"✅ Loaded {len(training_data)} training examples")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        if training_data:
            st.subheader("2️⃣ Configure Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_iterations = st.number_input(
                    "Number of iterations:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10
                )
            
            with col2:
                st.info(f"Training examples: {len(training_data)}")
            
            st.subheader("3️⃣ Train Model")
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model..."):
                    progress_bar = st.progress(0)
                    
                    try:
                        trainer = CustomSkillNERTrainer()
                        trainer.create_blank_model()
                        training_stats = trainer.train(training_data, n_iterations=n_iterations)
                        
                        st.session_state.trained_ner = trainer
                        st.session_state.training_stats = training_stats
                        
                        progress_bar.progress(100)
                        st.success("✅ Training complete!")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(training_stats['losses']) + 1)),
                            y=training_stats['losses'],
                            mode='lines+markers',
                            name='Training Loss'
                        ))
                        
                        fig.update_layout(
                            title="Training Loss Over Iterations",
                            xaxis_title="Iteration",
                            yaxis_title="Loss",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
            
            if st.session_state.get('trained_ner'):
                st.subheader("4️⃣ Test Model")
                
                test_text = st.text_area(
                    "Enter text to test model:",
                    placeholder="Example: I am proficient in Python, Java, and Machine Learning."
                )
                
                if st.button("🧪 Test"):
                    if test_text:
                        try:
                            predictions = st.session_state.trained_ner.predict(test_text)
                            
                            if predictions:
                                st.success(f"✅ Found {len(predictions)} skills:")
                                for skill, start, end in predictions:
                                    st.markdown(f"- **{skill}** (position {start}-{end})")
                            else:
                                st.warning("No skills detected")
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
    
    with tabs[4]:
        annotator.create_annotation_ui()
    
    with tabs[5]:
        if not st.session_state.extraction_results:
            st.info("👆 Please extract skills first in the 'Skill Extraction' tab")
            return
        
        result = st.session_state.extraction_results
        
        st.header("📈 Skill Analysis Visualizations")
        
        st.subheader("Skill Distribution by Category")
        fig_pie = visualizer.create_category_distribution_chart(result['categorized_skills'])
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("Top Skills by Confidence Score")
        top_n = st.slider("Number of top skills to display:", 5, 30, 15)
        fig_bar = visualizer.create_top_skills_chart(
            result['all_skills'],
            result['skill_confidence'],
            top_n
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("Extraction Methods Comparison")
        fig_methods = visualizer.create_extraction_methods_chart(result['extraction_methods'])
        st.plotly_chart(fig_methods, use_container_width=True)
        
        st.subheader("📋 Detailed Skill Table")
        
        detailed_data = []
        for skill in result['all_skills']:
            category = st.session_state.analyzer.skill_extractor.skill_db.get_category_for_skill(skill)
            confidence = result['skill_confidence'].get(skill, 0)
            
            detailed_data.append({
                'Skill': skill,
                'Category': category.replace('_', ' ').title(),
                'Confidence': f"{confidence:.0%}",
                'Confidence Score': confidence
            })
        
        df = pd.DataFrame(detailed_data)
        
        col1, col2 = st.columns(2)
        with col1:
            categories = ['All'] + sorted(df['Category'].unique().tolist())
            selected_category = st.selectbox("Filter by category:", categories)
        
        with col2:
            min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0, 0.1)
        
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        filtered_df = filtered_df[filtered_df['Confidence Score'] >= min_confidence]
        
        st.dataframe(
            filtered_df.sort_values('Confidence Score', ascending=False)[['Skill', 'Category', 'Confidence']],
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(df)} skills")
    
    with tabs[6]:
        if not st.session_state.extraction_results:
            st.info("👆 Please extract skills first in the 'Skill Extraction' tab")
            return
        
        result = st.session_state.extraction_results
        
        st.header("📥 Export Extracted Skills")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = export_manager.create_csv_export(result)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"extracted_skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = export_manager.create_json_export(result)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"extracted_skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            report_data = export_manager.create_text_report(result)
            st.download_button(
                label="📑 Download Report",
                data=report_data,
                file_name=f"skill_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Milestone 2 Complete")
        st.markdown("""
        **✅ All Features Implemented:**
        
        1. **Skill Extraction (NLP)**
           - spaCy pipeline
           - Multi-method extraction
           - 6 extraction techniques
        
        2. **BERT Embeddings**
           - Sentence-BERT
           - Similarity computation
           - Semantic matching
        
        3. **Custom NER Training**
           - Model training
           - Testing interface
           - Training visualization
        
        4. **Annotation Interface**
           - Training data creation
           - Export functionality
        
        5. **Visualizations**
           - Interactive charts
           - Distribution analysis
           - Similarity heatmaps
        
        6. **Export Options**
           - CSV, JSON, Text reports
        """)
        
        if st.session_state.get('extraction_results'):
            result = st.session_state.extraction_results
            st.header("📊 Current Stats")
            st.metric("Skills Found", result['statistics']['total_skills'])
            avg_conf = sum(result['skill_confidence'].values()) / len(result['skill_confidence']) if result['skill_confidence'] else 0
            st.metric("Avg Confidence", f"{avg_conf:.0%}")

if __name__ == "__main__":
    main()