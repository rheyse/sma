import os
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try importing sentence_transformers, but handle the error if it fails
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers could not be imported. Falling back to TF-IDF only.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Default embedding model for sentence transformers
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Try importing spaCy, but handle the error if it fails
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model, but don't fail if it's not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        SPACY_AVAILABLE = False
except ImportError:
    print("Warning: spaCy could not be imported. Some NLP features will be limited.")
    SPACY_AVAILABLE = False

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Domain-specific technology and skill patterns
TECH_PATTERNS = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "rust", "golang", "go", "ruby", "php", "swift",
        "kotlin", "scala", "r", "matlab", "perl", "shell", "bash", "powershell", "objective-c", "dart", "lua", "clojure",
        "haskell"
    ],
    "data_technologies": [
        "kafka", "spark", "hadoop", "flink", "elasticsearch", "mongodb", "postgresql", "mysql", "sql", "nosql", "redis",
        "database", "data lake", "data warehouse", "etl", "data processing", "stream processing", "batch processing",
        "real-time", "big data", "data engineering", "data science", "machine learning", "deep learning",
        "nifi", "airflow", "pandas", "numpy", "scikit-learn", "hive", "presto", "snowflake", "bigquery", "redshift",
        "databricks"
    ],
    "cloud_platforms": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", "digitalocean", "heroku",
        "openstack", "cloud foundry", "cloud computing", "cloud native", "serverless", "iaas", "paas", "saas",
        "containers", "docker", "kubernetes", "k8s"
    ],
    "frameworks": [
        "tensorflow", "pytorch", "keras", "fastapi", "sanic", "react", "angular", "vue", "svelte", "sveltekit", "next.js",
        "nuxt.js", "ember", "django", "flask", "spring", "spring boot", "hibernate", "node.js", "express", "fastify",
        "laravel", "symfony", "codeigniter", "rails", "asp.net", "quasar"
    ],
    "networking": [
        "routing", "bgp", "ospf", "nat", "vpn", "ip", "tcp/ip", "dns", "dhcp", "subnetting", "networking",
        "network security", "firewall", "load balancing", "cdn", "proxy", "reverse proxy", "http", "https",
        "ftp", "ssh", "ssl", "tls"
    ],
    "observability": [
        "prometheus", "grafana", "kibana", "datadog", "splunk", "logstash", "fluentd", "jaeger", "zipkin",
        "opentelemetry", "new relic", "logging", "monitoring", "alerting", "tracing", "metrics", "observability",
        "sre", "site reliability", "devops", "slo", "sli"
    ],
    "cybersecurity": [
        "security", "encryption", "cryptography", "authentication", "authorization", "oauth", "oidc", "jwt",
        "vulnerability", "penetration testing", "security audit", "compliance", "gdpr", "hipaa", "pci",
        "siem", "ids", "ips", "zero trust"
    ],
    "infrastructure": [
        "infrastructure", "automation", "terraform", "ansible", "puppet", "chef", "helm", "packer", "vagrant",
        "istio", "linkerd", "argo cd", "ci/cd", "jenkins", "gitlab", "github actions", "deployment",
        "configuration management", "infrastructure as code"
    ],
    "5g_technology": [
        "5g", "lte", "4g", "nr", "sub-6", "mmwave", "wireless", "telecommunications", "telco", "radio",
        "spectrum", "mimo", "beamforming"
    ],
    "fiber_technology": [
        "fiber", "ftx", "fttx", "ftth", "gpon", "xgs-pon", "passive optical", "optical", "ont",
        "fixed wireless", "wireless access", "broadband"
    ]
}

# Load the sentence transformer model
def get_sentence_transformer():
    """Load and return a sentence transformer model."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    model_name = os.getenv("SMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading sentence transformer model '{model_name}': {e}")
        return None

# Advanced text preprocessing
def advanced_preprocess(text):
    """
    Perform advanced preprocessing on text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Extract tech skills using predefined patterns
def extract_tech_skills(text):
    """Extract technical skills and technologies from text using predefined patterns"""
    text_lower = text.lower()
    
    # Find all matches
    matches = {}
    for category, terms in TECH_PATTERNS.items():
        matches[category] = []
        for term in terms:
            pattern = r'\b' + term + r'\b'
            if re.findall(pattern, text_lower):
                matches[category].append(term)
    
    # Convert matches to requirements
    requirements = []
    for category, terms in matches.items():
        if terms:
            if category == "programming_languages":
                requirements.append(f"Programming skills: {', '.join(terms)}")
            elif category == "data_technologies":
                requirements.append(f"Data engineering: {', '.join(terms)}")
            elif category == "cloud_platforms":
                requirements.append(f"Cloud platforms: {', '.join(terms)}")
            elif category == "frameworks":
                requirements.append(f"Frameworks: {', '.join(terms)}")
            elif category == "networking":
                requirements.append(f"Networking: {', '.join(terms)}")
            elif category == "observability":
                requirements.append(f"Observability: {', '.join(terms)}")
            elif category == "cybersecurity":
                requirements.append(f"Security: {', '.join(terms)}")
            elif category == "infrastructure":
                requirements.append(f"Infrastructure: {', '.join(terms)}")
            elif category == "5g_technology":
                requirements.append(f"5G Technology: {', '.join(terms)}")
            elif category == "fiber_technology":
                requirements.append(f"Fiber & Wireless: {', '.join(terms)}")
    
    return requirements

# Extract role-skill pairs from structured text
def extract_role_skill_pairs(text):
    """Extract role-skill pairs from structured text"""
    lines = text.split('\n')
    
    role_patterns = [
        r'^\s*(role|position|job title)\s*:\s*(.+)$',
        r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*:(.+)$',
        r'^\s*([A-Za-z\s]+)\s{2,}(.+)$'  # Role followed by multiple spaces then skill
    ]
    
    skill_patterns = [
        r'^\s*(skills?|requirements?|qualifications?)\s*:\s*(.+)$',
        r'^\s*[•\-*]\s*(.+)$'
    ]
    
    roles = []
    skills = []
    
    current_role = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for role patterns
        role_found = False
        for pattern in role_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                # Check if this is likely a role rather than a general term
                if candidate in ["role", "position", "job", "job title"]:
                    current_role = match.group(2).strip()
                else:
                    current_role = match.group(1).strip()
                
                if current_role and current_role not in roles:
                    roles.append(current_role)
                role_found = True
                
                # If there's content after the role, it might be a skill
                if match.group(2).strip():
                    skills.append((current_role, match.group(2).strip()))
                break
                
        # If no role pattern matches, check for skill patterns
        if not role_found and current_role:
            for pattern in skill_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if pattern == skill_patterns[0]:
                        # skill list pattern captures skill in group 2
                        skill = match.group(2).strip()
                    else:
                        # bullet pattern captures skill in group 1
                        skill = match.group(1).strip()
                    if skill:
                        skills.append((current_role, skill))
                    break
    
    # Convert to requirements
    requirements = []
    for role, skill in skills:
        requirements.append(f"{skill} (for {role})")
        
    return requirements

# Extract skills using spaCy's NER
def extract_skills_with_spacy(text):
    """Use spaCy for named entity recognition and dependency parsing"""
    if not SPACY_AVAILABLE:
        return []
    
    # Process the text
    doc = nlp(text)
    
    # Extract entities that might be technologies or skills
    tech_entities = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "GPE"]]
    
    # Extract noun phrases that might be skills
    skill_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
    
    # Combine and clean
    candidates = tech_entities + skill_phrases
    skills = list(set([item.lower() for item in candidates if len(item) > 3]))
    
    # Convert to requirements
    requirements = []
    for skill in skills:
        # Filter out common non-skill terms
        if skill not in ["the requirement", "the skill", "the role", "the position", "the company", "the team"]:
            requirements.append(f"Proficiency in {skill}")
    
    return requirements

# Extract sentences from text
def extract_sentences(text):
    """
    Extract and clean sentences from text
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Clean sentences
    clean_sentences = []
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 4:
            continue
            
        # Clean the sentence
        clean_sentence = advanced_preprocess(sentence)
        if clean_sentence:
            clean_sentences.append(clean_sentence)
    
    return clean_sentences

# Function to extract key sentences using TFIDF
def extract_key_sentences_tfidf(sentences, top_n=20):
    """
    Extract key sentences using TF-IDF vectorization
    """
    if len(sentences) <= top_n:
        return sentences
        
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence importance
    importance = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Get indices of top N important sentences
    top_indices = importance.argsort()[-top_n:][::-1]
    
    # Return the top sentences in original order
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    
    return top_sentences

# Function to cluster similar sentences using sentence embeddings
def cluster_requirements_with_transformers(requirements, threshold=0.75):
    """Cluster requirements using sentence transformers for better semantic understanding"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return requirements
        
    model = get_sentence_transformer()
    if not model:
        return requirements
    
    # Generate embeddings
    embeddings = model.encode(requirements)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Cluster similar requirements
    clusters = []
    used = set()
    
    for i in range(len(requirements)):
        if i in used:
            continue
            
        cluster = [i]
        used.add(i)
        
        for j in range(len(requirements)):
            if j not in used and similarity_matrix[i, j] > threshold:
                cluster.append(j)
                used.add(j)
                
        clusters.append([requirements[idx] for idx in cluster])
    
    # Generate representative requirements
    result = []
    for cluster in clusters:
        if len(cluster) == 1:
            result.append(cluster[0])
        else:
            # Find the most representative requirement (closest to centroid)
            if len(cluster) > 2:
                # Get the average embedding for this cluster
                cluster_indices = [requirements.index(req) for req in cluster]
                cluster_embeddings = [embeddings[idx] for idx in cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Find the requirement closest to the centroid
                distances = [np.linalg.norm(centroid - emb) for emb in cluster_embeddings]
                representative_idx = cluster_indices[np.argmin(distances)]
                representative = requirements[representative_idx]
                
                result.append(f"{representative} (including {len(cluster)-1} related requirements)")
            else:
                # For just 2 requirements, use the longer one
                result.append(max(cluster, key=len))
            
    return result

# Function to group similar requirements (fallback method)
def group_similar_requirements(requirements, similarity_threshold=0.6):
    """Group similar requirements together"""
    if len(requirements) <= 1:
        return requirements
        
    # Vectorize the requirements
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(requirements)
        
        # Calculate similarity between all pairs
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group similar requirements
        groups = []
        used_indices = set()
        
        for i in range(len(requirements)):
            if i in used_indices:
                continue
                
            group = [i]
            used_indices.add(i)
            
            for j in range(i+1, len(requirements)):
                if j not in used_indices and similarity_matrix[i, j] > similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
                    
            groups.append(group)
        
        # Take the first requirement from each group (or combine them)
        grouped_requirements = []
        for group in groups:
            if len(group) == 1:
                grouped_requirements.append(requirements[group[0]])
            else:
                # For multiple similar requirements, use the longest one as it's likely more informative
                main_req = max([requirements[idx] for idx in group], key=len)
                grouped_requirements.append(main_req)
        
        return grouped_requirements
    
    except:
        # If vectorization fails, just return the original requirements
        return requirements

# Main improved function to extract learning requirements
def extract_advanced_requirements(text, max_requirements=15):
    """
    Extract learning requirements using advanced NLP techniques
    """
    # 1. Try structured parsing first
    role_skill_requirements = extract_role_skill_pairs(text)
    
    # 2. Use technology pattern detection
    tech_requirements = extract_tech_skills(text)
    
    # 3. Use NER if available
    ner_requirements = extract_skills_with_spacy(text)
    
    # 4. Traditional sentence-based approach as fallback
    sentence_requirements = []
    if not (role_skill_requirements or tech_requirements or ner_requirements):
        sentences = extract_sentences(text)
        key_sentences = extract_key_sentences_tfidf(sentences)
        sentence_requirements = key_sentences
    
    # Combine all requirements
    all_requirements = role_skill_requirements + tech_requirements + ner_requirements + sentence_requirements
    
    # Remove duplicates while preserving order
    seen = set()
    unique_requirements = []
    for req in all_requirements:
        if req.lower() not in seen:
            seen.add(req.lower())
            unique_requirements.append(req)
    
    # 5. Cluster similar requirements
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            final_requirements = cluster_requirements_with_transformers(unique_requirements)
        else:
            final_requirements = group_similar_requirements(unique_requirements)
    except:
        # Fallback to original requirements if clustering fails
        final_requirements = unique_requirements
    
    # Limit the number of requirements
    return final_requirements[:max_requirements]

# Function to compute similarity between requirements and content
def compute_semantic_similarity(requirements, content_items):
    """
    Compute semantic similarity between requirements and content items
    """
    # Get the model
    model = get_sentence_transformer()
    if not model:
        return None
    
    # Generate embeddings
    requirement_embeddings = model.encode(requirements)
    content_embeddings = model.encode(content_items)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(requirement_embeddings, content_embeddings)
    
    return similarity_matrix
