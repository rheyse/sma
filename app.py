import os
import re
import sys
from io import BytesIO
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import nltk
import numpy as np
import math
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
import humanize
import requests
import traceback

# Constants
DATA_PATH = 'sample_programs.csv'
API_URL = 'https://api.udacity.com/api/unified-catalog'

# Setup paths and imports
sma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SMA')
if os.path.exists(sma_path):
    sys.path.insert(0, sma_path)

# Import advanced NLP module
try:
    from advanced_nlp import (
        extract_advanced_requirements,
        compute_semantic_similarity,
        get_sentence_transformer,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        SPACY_AVAILABLE,
        nlp,
    )
except ImportError as e:
    print(f"Error importing advanced NLP module: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SPACY_AVAILABLE = False
    nlp = None

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# ============== UTILITY FUNCTIONS ==============

def categorize_duration(duration_str):
    """Categorize duration strings into hour, day, week, or month categories"""
    duration_str = str(duration_str).lower()
    
    if 'hour' in duration_str or 'hr' in duration_str:
        return 'Hours'
    elif 'day' in duration_str:
        return 'Days'
    elif 'week' in duration_str:
        return 'Weeks'
    elif 'month' in duration_str:
        return 'Months'
    else:
        return 'Unknown'  # For durations that don't fit these categories

def convert_program_type(semantic_type):
    """Convert API program type to display type"""
    if semantic_type == 'Course':
        return 'FreeCourse'
    elif semantic_type == 'Degree':
        return 'Nanodegree'
    else:
        return 'PaidCourse'

def convert_slug_to_url(slug):
    """Convert slug to full URL"""
    return f"https://www.udacity.com/course/{slug}"

def convert_duration_mins_to_human_readable(duration_mins):
    """Convert duration minutes to readable format"""
    delta = pd.Timedelta(minutes=duration_mins)
    return humanize.naturaldelta(delta)

def identify_matching_terms(requirement, content, min_length=4):
    """Identify key terms that match between requirement and content."""

    req_lower = requirement.lower()
    content_lower = content.lower()

    # Use spaCy for lemmatization if available
    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(req_lower)
        tokens = [tok.lemma_ for tok in doc if tok.is_alpha]
    else:
        tokens = re.findall(r"\b\w+\b", req_lower)

    # Words meeting the minimum length
    req_words = [t for t in tokens if len(t) >= min_length]

    matches = []
    for word in req_words:
        synonyms = {word}
        try:
            for syn in wn.synsets(word):
                for lemma in syn.lemma_names():
                    synonyms.add(lemma.lower().replace("_", " "))
        except LookupError:
            pass

        if any(s in content_lower for s in synonyms):
            matches.append(word)

    # Phrase detection using lemmatized tokens
    req_phrases = []
    for i in range(len(tokens) - 1):
        if len(tokens[i]) >= min_length or len(tokens[i + 1]) >= min_length:
            phrase = f"{tokens[i]} {tokens[i + 1]}"
            if phrase in content_lower:
                req_phrases.append(phrase)

    all_matches = matches + req_phrases

    if all_matches:
        seen = set()
        ordered = []
        for m in all_matches:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return ", ".join(ordered)
    return ""

def search_programs(keyword, programs_df, exclude_keys=None, max_results=10):
    """Search programs_df for a keyword in title, summary, or skills"""
    if exclude_keys is None:
        exclude_keys = []

    keyword_lower = keyword.lower()

    def skill_match(skills):
        if isinstance(skills, (list, tuple)):
            text = " ".join(skills).lower()
        else:
            text = str(skills).lower()
        return keyword_lower in text

    mask = (
        programs_df['title'].str.lower().str.contains(keyword_lower, na=False) |
        programs_df['summary'].str.lower().str.contains(keyword_lower, na=False) |
        programs_df['skills'].apply(skill_match)
    )

    results = programs_df[mask]
    if exclude_keys:
        results = results[~results['key'].isin(exclude_keys)]

    return results.head(max_results)

def render_program_search(req, programs_df, exclude_keys=None):
    """Render a search UI allowing the user to pick a specific program"""
    if exclude_keys is None:
        exclude_keys = []

    with st.expander("Program Search"):
        search_kw = st.text_input(f"Keyword search for '{req}'", key=f"search_{req}")
        if search_kw:
            results = search_programs(search_kw, programs_df, exclude_keys)
            if results.empty:
                st.write("No matching programs found.")
            else:
                for s_idx, s_row in results.iterrows():
                    st.write(f"**{s_row['title']}**")
                    short_sum = s_row['summary'][:150] + ("..." if len(s_row['summary']) > 150 else "")
                    st.write(short_sum)
                    if st.button("Use this program", key=f"use_{req}_{s_idx}"):
                        selected = {
                            'program_key': s_row['key'],
                            'program_title': s_row['title'],
                            'program_type': s_row['program_type'],
                            'duration': s_row['duration'],
                            'difficulty': s_row['difficulty'],
                            'url': s_row['catalog_url'],
                            'summary': s_row.get('summary', ''),
                            'skills': s_row.get('skills', []),
                            'similarity_score': '',
                            'recommendation_reason': 'Selected via search'
                        }
                        st.session_state['current_recommendation'][req] = {'idx': s_idx, 'row': selected}
                        st.session_state['already_recommended_urls'].add(str(s_row.get('key', s_row.get('catalog_url', ''))))
                        st.success(f"Selected program for '{req}'")
                        st.rerun()

# ============== DATA LOADING & PROCESSING ==============

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded files (CSV, PDF, XLS)"""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(uploaded_file)
        return df.to_string()
    
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df = pd.read_excel(uploaded_file)
        return df.to_string()
    
    elif file_extension == '.pdf':
        pdf_text = ""
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
            for page_num in range(len(doc)):
                pdf_text += doc.load_page(page_num).get_text()
        return pdf_text
    
    else:
        return None

def extract_requirements_from_file(uploaded_file):
    """Extract learning requirements from CSV/XLS files (one per line)"""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == '.csv':
        # Read CSV file and extract requirements
        df = pd.read_csv(uploaded_file)
        # If there's only one column, use it directly
        if len(df.columns) == 1:
            requirements = df.iloc[:, 0].tolist()
        else:
            # If multiple columns, concatenate the first two columns assuming they're role/requirement
            # This handles formats like "Role, Requirement" common in skill matrices
            requirements = []
            for _, row in df.iterrows():
                if pd.notna(row[0]) and str(row[0]).strip():
                    if len(df.columns) > 1 and pd.notna(row[1]) and str(row[1]).strip():
                        req = f"{str(row[1]).strip()} (for {str(row[0]).strip()})"
                    else:
                        req = str(row[0]).strip()
                    requirements.append(req)
    
    elif file_extension == '.xlsx' or file_extension == '.xls':
        # Read Excel file and extract requirements
        df = pd.read_excel(uploaded_file)
        # If there's only one column, use it directly
        if len(df.columns) == 1:
            requirements = df.iloc[:, 0].tolist()
        else:
            # If multiple columns, handle similarly to CSV
            requirements = []
            for _, row in df.iterrows():
                if pd.notna(row[0]) and str(row[0]).strip():
                    if len(df.columns) > 1 and pd.notna(row[1]) and str(row[1]).strip():
                        req = f"{str(row[1]).strip()} (for {str(row[0]).strip()})"
                    else:
                        req = str(row[0]).strip()
                    requirements.append(req)
    
    else:
        return None
    
    # Clean requirements and remove empty entries
    cleaned_requirements = [req.strip() for req in requirements if req and str(req).strip()]
    return cleaned_requirements



# Add cache decorator to expensive operations
@st.cache_data(ttl=3600)
def prepare_programs_df():
    """Fetch catalog data from API and prepare programs DataFrame"""
    # Try to load from a sample CSV file first if it exists
    try:
        # Check if we have a local sample file
        if os.path.exists(DATA_PATH):
            # Use dtype=object to prevent automatic type inference
            programs_df = pd.read_csv(DATA_PATH, dtype=object)
            
            # Convert specific columns to appropriate types
            numeric_cols = ['similarity_score']
            for col in numeric_cols:
                if col in programs_df.columns:
                    # Convert to numeric safely
                    programs_df[col] = pd.to_numeric(programs_df[col], errors='coerce')
            
            # Handle skills column specifically
            if 'skills' in programs_df.columns:
                # First, ensure all skills values are strings
                programs_df['skills'] = programs_df['skills'].astype(str)
                
                # Then safely evaluate the string representations to lists
                def safe_eval_skills(skills_str):
                    try:
                        if skills_str.startswith('[') and skills_str.endswith(']'):
                            return eval(skills_str)
                        return []
                    except:
                        return []
                
                programs_df['skills'] = programs_df['skills'].apply(safe_eval_skills)
                programs_df['skills'] = programs_df['skills'].apply(lambda x: tuple(x))

            # Precompute semantic text and embeddings if possible
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                model = get_sentence_transformer()
                if model:
                    programs_df['semantic_text'] = programs_df.apply(
                        lambda row: f"{row['title']} {row['summary']} {' '.join(row['skills'])}",
                        axis=1,
                    )
                    try:
                        programs_df['embedding'] = list(model.encode(programs_df['semantic_text'].tolist()))
                        # Convert embeddings to tuples so caching can hash the dataframe
                        programs_df['embedding'] = programs_df['embedding'].apply(lambda x: tuple(map(float, x)))
                    except Exception as e:
                        st.warning(f"Embedding computation failed: {e}")

            # Add duration category for filtering
            if 'duration' in programs_df.columns:
                programs_df['duration_category'] = programs_df['duration'].apply(categorize_duration)

            st.success(f"Loaded {len(programs_df)} programs from local file")
            return programs_df
    except Exception as e:
        st.warning(f"Could not load from local file: {e}")
    
    # If no local file, try the API
    st.info("Fetching catalog data from Udacity API...")
    
    try:
        # Fetch data from API
        search_payload = {
            'PageSize': 1000,
            'SortBy': 'avgRating'
        }
        
        r = requests.post(f'{API_URL}/search', json=search_payload, timeout=10)
        if r.status_code != 200:
            raise Exception(f"API returned status code {r.status_code}")
        
        data = r.json()
        st.success(f"Successfully fetched catalog data: {data['searchResult']['nbHits']} hits")
        
        # Process catalog results
        catalog_results = []
        for catalog_result in data['searchResult']['hits']:
            if catalog_result.get('is_offered_to_public', False):
                catalog_results.append(catalog_result)
        
        # Convert catalog results to programs with strict type control
        programs = []
        for catalog_result in catalog_results:
            # Handle duration with strict type control
            duration_mins = 0
            try:
                if 'duration' in catalog_result:
                    duration_mins = float(catalog_result['duration'])
            except (ValueError, TypeError):
                duration_mins = 0
            
            duration_str = "Unknown"
            try:
                if duration_mins > 0:
                    duration_str = humanize.naturaldelta(pd.Timedelta(minutes=duration_mins))
                else:
                    duration_str = "Unknown duration"
            except:
                duration_str = "Unknown duration"
            
            # Create program dict with strict type control
            program = {
                'key': str(catalog_result.get('key', '')),
                'program_type': str(convert_program_type(catalog_result.get('semantic_type', ''))),
                'catalog_url': str(convert_slug_to_url(catalog_result.get('slug', ''))),
                'duration': str(duration_str),
                'difficulty': str(catalog_result.get('difficulty', 'Unknown')),
                'title': str(catalog_result.get('title', 'Untitled')),
                'summary': str(catalog_result.get('summary', '')),
                'skills': tuple(catalog_result.get('skill_names', []))
            }
            programs.append(program)
        
        # Create DataFrame - all columns will be object type initially
        programs_df = pd.DataFrame(programs)
        
        # Precompute semantic text and embeddings if possible
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            model = get_sentence_transformer()
            if model is not None:
                programs_df['semantic_text'] = programs_df.apply(
                    lambda row: f"{row['title']} {row['summary']} {' '.join(row['skills'])}",
                    axis=1,
                )
                try:
                    programs_df['embedding'] = list(model.encode(programs_df['semantic_text'].tolist()))
                    programs_df['embedding'] = programs_df['embedding'].apply(lambda x: tuple(map(float, x)))
                except Exception as e:
                    st.warning(f"Embedding computation failed: {e}")

        # Add duration category for filtering
        if 'duration' in programs_df.columns:
            programs_df['duration_category'] = programs_df['duration'].apply(categorize_duration)

        # Save to CSV for future use, with careful error handling
        try:
            programs_df.to_csv(DATA_PATH, index=False)
        except Exception as e:
            st.warning(f"Could not save to CSV: {str(e)}")
            
        st.success(f"Created programs DataFrame with {len(programs)} entries")
        return programs_df
    except Exception as e:
        st.error(f"Error fetching catalog data: {e}")
        raise Exception(f"Failed to fetch program data from the API: {e}")

# ============== RECOMMENDATION FUNCTIONS ==============

@st.cache_data
def recommend_content_tfidf(requirements, programs_df, top_n=3):
    """Return recommended content based on extracted learning requirements using TF-IDF"""
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = programs_df.copy()
    
    # Handle NaN values in each column
    for col in ['title', 'summary']:
        df_copy[col] = df_copy[col].fillna('').astype(str)
    
    # Handle skills column - convert to string with special handling for NaN/None
    df_copy['skills_str'] = df_copy['skills'].apply(
        lambda x: ' '.join(x) if isinstance(x, (list, tuple)) and len(x) > 0
                 else (str(x) if pd.notna(x) else '')
    )
    
    # Combine text fields into a single text field for vectorization
    # Skills often provide the clearest signal of course content, so we
    # weight them more heavily by repeating the skill terms. The
    # `SKILLS_WEIGHT` constant controls how many times the skills text is
    # repeated for TF-IDF vectorization.
    SKILLS_WEIGHT = 3
    df_copy['text_for_vectorization'] = (
        df_copy['title'] + ' ' +
        df_copy['summary'] + ' ' +
        ((df_copy['skills_str'] + ' ') * SKILLS_WEIGHT).str.strip()
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Handle NaN in requirements too
    clean_requirements = [req if pd.notna(req) else "" for req in requirements]
    clean_requirements = [req for req in clean_requirements if req.strip()]
    
    if not clean_requirements:
        st.warning("No valid requirements provided for TF-IDF search")
        return pd.DataFrame()  # Return empty dataframe if no valid requirements
    
    try:
        # Fit and transform the program texts
        program_vectors = vectorizer.fit_transform(df_copy['text_for_vectorization'])
        
        # Initialize a list to store recommendations
        all_recommendations = []
        
        # For each learning requirement, find similar programs
        for req in clean_requirements:
            # Transform the requirement text
            req_vector = vectorizer.transform([req])
            
            # Calculate similarity
            similarity_scores = cosine_similarity(req_vector, program_vectors).flatten()
            
            # Get indices of top N similar programs
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            
            # Get the recommended programs
            for idx in top_indices:
                if similarity_scores[idx] > 0.01:  # Only include if there's some similarity
                    recommendation = {
                        'requirement': req,
                        'program_key': df_copy.iloc[idx]['key'],
                        'program_title': df_copy.iloc[idx]['title'],
                        'program_type': df_copy.iloc[idx]['program_type'],
                        'duration': df_copy.iloc[idx]['duration'],
                        'difficulty': df_copy.iloc[idx]['difficulty'],
                        'similarity_score': float(similarity_scores[idx]),  # Ensure it's a float
                        'url': df_copy.iloc[idx]['catalog_url'],
                        'summary': df_copy.iloc[idx]['summary'],  # Add summary
                        'skills': df_copy.iloc[idx]['skills'],    # Add skills
                        'recommendation_reason': f"Similarity score: {similarity_scores[idx]:.4f}"
                    }
                    all_recommendations.append(recommendation)
        
        # Create DataFrame from recommendations
        recommendations_df = pd.DataFrame(all_recommendations)
        
        # Sort by similarity score
        if not recommendations_df.empty:
            recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
            st.success(f"TF-IDF search found {len(recommendations_df)} recommendations")
        else:
            st.warning("TF-IDF search returned no recommendations")
        
        if 'skills' in recommendations_df.columns:
            recommendations_df['skills'] = recommendations_df['skills'].apply(
                lambda x: tuple(x) if isinstance(x, (list, tuple)) else x
            )
        return recommendations_df
        
    except Exception as e:
        st.error(f"Error in TF-IDF processing: {str(e)}")
        st.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty dataframe on error

@st.cache_data
def recommend_content_semantic(requirements, programs_df, top_n=3):
    """Return recommended content based on extracted learning requirements using semantic search"""

    # Prepare content texts for comparison
    if 'semantic_text' in programs_df.columns:
        content_texts = programs_df['semantic_text'].tolist()
    else:
        content_texts = []
        for _, row in programs_df.iterrows():
            skills_text = ' '.join(row['skills']) if isinstance(row['skills'], (list, tuple)) else str(row['skills'])
            content_text = f"{row['title']} {row['summary']} {skills_text}"
            content_texts.append(content_text)

    # Calculate semantic similarity using cached embeddings if available
    if 'embedding' in programs_df.columns:
        content_embeddings = np.array(programs_df['embedding'].tolist())
        similarity_matrix = compute_semantic_similarity(
            requirements,
            content_items=None,
            content_embeddings=content_embeddings,
        )
    else:
        similarity_matrix = compute_semantic_similarity(requirements, content_texts)
    
    if similarity_matrix is None:
        # Fall back to TF-IDF if sentence transformers fails
        return recommend_content_tfidf(requirements, programs_df, top_n)
    
    # Initialize list to store recommendations
    all_recommendations = []
    
    # For each requirement, find similar programs
    for i, req in enumerate(requirements):
        # Get similarity scores for this requirement
        similarity_scores = similarity_matrix[i]
        
        # Get indices of top N similar programs
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        # Get the recommended programs
        for idx in top_indices:
            if similarity_scores[idx] > 0.2:  # Only include if there's decent similarity
                # Extract key terms that matched
                matching_terms = identify_matching_terms(req, content_texts[idx])
                
                recommendation = {
                    'requirement': req,
                    'program_key': programs_df.iloc[idx]['key'],
                    'program_title': programs_df.iloc[idx]['title'],
                    'program_type': programs_df.iloc[idx]['program_type'],
                    'duration': programs_df.iloc[idx]['duration'],
                    'difficulty': programs_df.iloc[idx]['difficulty'],
                    'similarity_score': float(similarity_scores[idx]),
                    'url': programs_df.iloc[idx]['catalog_url'],
                    'summary': programs_df.iloc[idx]['summary'],  # Add summary
                    'skills': programs_df.iloc[idx]['skills'],    # Add skills
                    'recommendation_reason': f"Matching concepts: {matching_terms}" if matching_terms else 
                                         f"Program covers relevant skills with {similarity_scores[idx]:.2f} similarity score"
                }
                all_recommendations.append(recommendation)
    
    # Create DataFrame from recommendations
    recommendations_df = pd.DataFrame(all_recommendations)
    
    # Sort by requirement and similarity score
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values(by=['requirement', 'similarity_score'], ascending=[True, False])
        
        # Only keep top N per requirement
        top_recommendations = []
        for req, group in recommendations_df.groupby('requirement'):
            top_recommendations.append(group.head(top_n))
        
        if top_recommendations:
            recommendations_df = pd.concat(top_recommendations)
    
    if 'skills' in recommendations_df.columns:
        recommendations_df['skills'] = recommendations_df['skills'].apply(
            lambda x: tuple(x) if isinstance(x, (list, tuple)) else x
        )
    return recommendations_df

# ============== UI COMPONENTS ==============

def display_recommendations_ui(filtered_df, all_requirements, programs_df):
    """Display the recommendations UI with interactive elements"""
    # Initialize session state variables if they don't exist
    if 'rejected_indices' not in st.session_state:
        st.session_state['rejected_indices'] = {}  # Dict to track rejected recommendations per requirement

    if 'accepted_recommendations' not in st.session_state:
        st.session_state['accepted_recommendations'] = {}  # Dict to store accepted recommendations

    if 'rejection_history' not in st.session_state:
        st.session_state['rejection_history'] = {}  # Track order of rejections per requirement

    if 'already_recommended_urls' not in st.session_state:
        st.session_state['already_recommended_urls'] = set()  # Set to track already recommended course URLs
    
    # Track the currently showing recommendation for each requirement
    if 'current_recommendation' not in st.session_state:
        st.session_state['current_recommendation'] = {}  # Dict to track current recommendation per requirement

    # Maintain ordered indices and pointer for faster navigation
    if 'recommendation_indices' not in st.session_state:
        st.session_state['recommendation_indices'] = {}
    if 'current_index' not in st.session_state:
        st.session_state['current_index'] = {}
        
    # Add debug switch to examine internal state
    if st.checkbox("Debug Mode", value=False, key="debug_mode"):
        st.write("Current rejected indices:")
        st.write(st.session_state['rejected_indices'])
        st.write("Current recommendations:")
        st.write(st.session_state['current_recommendation'])
        st.write("Current index pointers:")
        st.write(st.session_state.get('current_index', {}))

    # Check if filtered_df is empty
    if filtered_df.empty:
        st.warning("No recommendations match the current filters. Try adjusting your filter criteria.")
        return

    # Choose a unique identifier column
    available_columns = filtered_df.columns.tolist()
    id_column = 'key' if 'key' in available_columns else 'url' if 'url' in available_columns else 'program_title'
    
    st.subheader("Course Recommendations")
    st.write("For each requirement, the top recommendation is shown. Click 'Reject' to see the next best match.")
    
    # Check if we should show the final table
    if st.session_state.get('show_final_table', False):
        display_final_recommendations(all_requirements)
    else:
        # Process each requirement
        for req in all_requirements:
            st.markdown(f"### Requirement: {req}")
            
            # Get all recommendations for this requirement, sorted by similarity score
            req_recommendations = filtered_df[filtered_df['requirement'] == req].sort_values(
                by='similarity_score', ascending=False
            )
            
            # Skip if no recommendations for this requirement
            if req_recommendations.empty:
                st.warning(f"No matching courses found for: {req}")
                continue
            
            # Initialize state for this requirement if not already done
            if req not in st.session_state['rejected_indices']:
                st.session_state['rejected_indices'][req] = set()
            if req not in st.session_state['rejection_history']:
                st.session_state['rejection_history'][req] = []
            if req not in st.session_state['recommendation_indices']:
                st.session_state['recommendation_indices'][req] = req_recommendations.index.tolist()
            if req not in st.session_state['current_index']:
                st.session_state['current_index'][req] = 0
                
            # Debug: Show current rejected indices
            if st.session_state.get('debug_mode', False):
                st.write(f"Rejected indices for {req}: {st.session_state['rejected_indices'][req]}")
                st.write(f"Available indices: {req_recommendations.index.tolist()}")
            
            # Use pointer to find next non-rejected recommendation
            pointer = st.session_state['current_index'][req]
            indices = st.session_state['recommendation_indices'][req]
            rejected_set = {str(i) for i in st.session_state['rejected_indices'].get(req, set())}
            while pointer < len(indices) and str(indices[pointer]) in rejected_set:
                pointer += 1
            st.session_state['current_index'][req] = pointer

            # If we've exhausted all recommendations
            if pointer >= len(indices):
                st.info(f"All available recommendations for '{req}' have been reviewed. No more alternatives available.")
                # Allow user to search the catalog even when everything is rejected
                render_program_search(req, programs_df, req_recommendations['program_key'].tolist())
                # Reset rejection state to start over if user clicks
                if st.button(f"Start Over for '{req}'", key=f"start_over_{req}"):
                    st.session_state['rejected_indices'][req] = set()
                    st.session_state['rejection_history'][req] = []
                    st.session_state['current_index'][req] = 0
                    if req in st.session_state['current_recommendation']:
                        del st.session_state['current_recommendation'][req]
                    st.rerun()
                continue

            # Get the best non-rejected recommendation
            best_idx = indices[pointer]
            best_row = req_recommendations.loc[best_idx]
            
            # Store the current recommendation for this requirement
            st.session_state['current_recommendation'][req] = {
                'idx': best_idx,
                'row': best_row.to_dict() if hasattr(best_row, 'to_dict') else best_row
            }
            
            # Determine if this is a duplicate recommendation
            if hasattr(best_row, 'to_dict'):
                row_dict = best_row.to_dict()
            else:
                row_dict = best_row
                
            course_id = str(row_dict[id_column]) if id_column in row_dict else f"idx_{best_idx}"
            is_duplicate = course_id in st.session_state['already_recommended_urls']
            
            # Create recommendation card

            try:
                display_recommendation_card(best_row, req, best_idx, is_duplicate, req_recommendations, programs_df)
            except Exception as e:
                st.error(f"Error displaying recommendation: {str(e)}")
                st.write("Recommendation data:")
                st.write(best_row)

        
        # Move the "Accept Current Recommendations" button to the bottom of the page
        st.markdown("---")
        st.markdown("### Finalize Your Selection")
        st.write("Once you're satisfied with all recommendations shown above, click below to accept them:")
        
        # Add a button to accept all current recommendations - positioned at the bottom
        if st.button("Accept Current Recommendations", type="primary"):
            # Use the CURRENT showing recommendations (not filtering again)
            for req in all_requirements:
                if req in st.session_state['current_recommendation']:
                    # Use the recommendation that's currently being shown to the user
                    st.session_state['accepted_recommendations'][req] = st.session_state['current_recommendation'][req]['row']
                else:
                    # No current recommendation for this requirement
                    st.session_state['accepted_recommendations'][req] = None
            
            # Set a flag to show the final table
            st.session_state['show_final_table'] = True
            st.rerun()
        
        # Add a button to reset all rejections - also at the bottom
        if st.button("Reset All Rejections"):
            st.session_state['rejected_indices'] = {}
            st.session_state['rejection_history'] = {}
            st.session_state['current_recommendation'] = {}
            st.session_state['current_index'] = {}
            st.session_state['recommendation_indices'] = {}
            st.rerun()

def display_recommendation_card(row, req, best_idx, is_duplicate, req_recommendations, programs_df):
    """Display a card with recommendation details and action buttons"""
    # Ensure we're working with scalar values, not Series
    if hasattr(row, 'to_dict'):
        row_data = row.to_dict()
    else:
        row_data = row
        
    # Helper function to extract scalar value
    def get_scalar(value):
        if hasattr(value, 'iloc'):
            return value.iloc[0]
        elif isinstance(value, dict) and len(value) == 1:
            return next(iter(value.values()))
        return value
        
    # Determine course_id from the row data to uniquely identify this recommendation
    id_column = 'key' if 'key' in row_data else 'url' if 'url' in row_data else 'program_title'
    course_id_raw = row_data[id_column]
    course_id = str(get_scalar(course_id_raw))
    
    # Create a unique ID for this specific recommendation
    unique_id = f"{req}_{best_idx}_{hash(str(course_id)) % 10000}"
    
    with st.container():
        cols = st.columns([3, 1, 1])
        with cols[0]:
            # Extract title as a scalar value
            title_raw = row_data.get('program_title', "Untitled")
            title = get_scalar(title_raw)
            
            # Extract URL as a scalar value
            url_raw = row_data.get('url', "#")
            url = get_scalar(url_raw)
                
            # Display title and URL properly
            st.subheader(title)
            st.markdown(f"[View Course]({url})")
            
            # Add course summary in an expandable section
            with st.expander("Course Summary"):
                summary_raw = row_data.get('summary', "No summary available")
                summary = get_scalar(summary_raw)
                st.write(summary)
            
            # Display skills in an expandable section
            with st.expander("Skills Covered"):
                skills_raw = row_data.get('skills', [])
                skills = get_scalar(skills_raw)
                    
                if isinstance(skills, (list, tuple)) and skills:
                    for skill in skills:
                        st.write(f"• {skill}")
                elif isinstance(skills, str) and skills:
                    # Handle case where skills might be a string
                    if skills.startswith('[') and skills.endswith(']'):
                        try:
                            # Try to parse a string representation of a list
                            skills_list = eval(skills)
                            for skill in skills_list:
                                st.write(f"• {skill}")
                        except:
                            st.write(skills)
                    else:
                        st.write(skills)
                else:
                    st.write("No skills information available")
            
            # Basic info - handle each field as a potential Series
            duration_raw = row_data.get('duration', "Unknown")
            duration = get_scalar(duration_raw)
            st.write(f"**Duration:** {duration}")
            
            difficulty_raw = row_data.get('difficulty', "Unknown")
            difficulty = get_scalar(difficulty_raw)
            st.write(f"**Difficulty:** {difficulty}")
            
            program_type_raw = row_data.get('program_type', "Unknown")
            program_type = get_scalar(program_type_raw)
            st.write(f"**Type:** {program_type}")
            
            # Fix for the formatting error - ensure we're working with a scalar value
            try:
                similarity_score_raw = row_data.get('similarity_score', 0)
                # Get scalar value
                similarity_score = get_scalar(similarity_score_raw)
                score_value = float(similarity_score)
                st.write(f"**Relevance Score:** {score_value:.4f}")
            except Exception as e:
                # Fallback to display without formatting
                st.write(f"**Relevance Score:** {get_scalar(row_data.get('similarity_score', 'Unknown'))}")
            
            # Add recommendation source - also handle potential Series
            source_raw = row_data.get('recommendation_source', 'Unknown')
            source = get_scalar(source_raw)
            st.write(f"**Source:** {source}")
            
            reason_raw = row_data.get('recommendation_reason', "")
            reason = get_scalar(reason_raw)
            if reason:
                st.write(f"**Recommendation Reason:** {reason}")
            
            # Indicate if this is a duplicate course
            if is_duplicate:
                st.warning("Note: This course is already recommended for another requirement.")
        
        with cols[2]:
            # Calculate how many recommendations we've gone through
            current_idx = st.session_state['current_index'].get(req, 0)
            total_count = len(req_recommendations)

            st.write(f"Showing recommendation {current_idx + 1} of {total_count}")

            # Reject button - now with unique key
            if st.button(f"Reject & Show Next", key=f"reject_{unique_id}"):
                # Add this index to the rejected set for this requirement
                if req not in st.session_state['rejected_indices']:
                    st.session_state['rejected_indices'][req] = set()
                if req not in st.session_state['rejection_history']:
                    st.session_state['rejection_history'][req] = []

                # Add the string version of the index for consistent comparison
                st.session_state['rejected_indices'][req].add(str(best_idx))
                st.session_state['rejection_history'][req].append(str(best_idx))

                # Move pointer forward
                st.session_state['current_index'][req] = st.session_state['current_index'].get(req, 0) + 1

                # Remove from current recommendation if present
                if req in st.session_state['current_recommendation'] and \
                   str(st.session_state['current_recommendation'][req]['idx']) == str(best_idx):
                    del st.session_state['current_recommendation'][req]

                # Debug the rejection if in debug mode
                if st.session_state.get('debug_mode', False):
                    st.write(f"Added {best_idx} to rejected indices for {req}")
                    st.write(f"Rejected indices now: {st.session_state['rejected_indices'][req]}")

                st.rerun()

            # Show previous recommendation if available
            if st.button(f"Show Previous", key=f"show_prev_{unique_id}"):
                history = st.session_state['rejection_history'].get(req, [])
                if history:
                    last_idx = history.pop()
                    st.session_state['rejected_indices'][req].discard(last_idx)
                    st.session_state['current_index'][req] = max(st.session_state['current_index'].get(req, 1) - 1, 0)
                    if req in st.session_state['current_recommendation']:
                        del st.session_state['current_recommendation'][req]
                    st.rerun()

            # Accept button - now with unique key
            if st.button(f"Accept", key=f"accept_{unique_id}"):
                # Store the accepted recommendation
                st.session_state['already_recommended_urls'].add(course_id)
                st.success(f"Recommendation accepted for '{req}'")

            # Reject all remaining recommendations for this requirement
            if st.button(f"Reject All", key=f"reject_all_{unique_id}"):
                all_indices = {str(i) for i in req_recommendations.index}
                st.session_state['rejected_indices'][req] = all_indices
                st.session_state['rejection_history'][req] = list(all_indices)
                st.session_state['current_index'][req] = len(req_recommendations)
                if req in st.session_state['current_recommendation']:
                    del st.session_state['current_recommendation'][req]
                st.rerun()
        # Offer manual search after the action buttons
        render_program_search(req, programs_df, req_recommendations['program_key'].tolist())

    st.markdown("---")

def display_final_recommendations(all_requirements):
    """Display the final table of accepted recommendations"""
    st.success("Recommendations accepted! Here's your final selection:")
    
    # Construct the final table
    final_table = []
    for req in all_requirements:
        if req in st.session_state['accepted_recommendations']:
            rec = st.session_state['accepted_recommendations'][req]
            if rec is not None:
                # Create HTML link with correct title
                html_link = f'<a href="{rec["url"]}" target="_blank">{rec["program_title"]}</a>'
                
                # Prepare skills as a formatted string
                skills_text = ""
                skills = rec.get('skills', [])
                if isinstance(skills, (list, tuple)) and skills:
                    skills_text = ", ".join(skills[:5])
                    if len(skills) > 5:
                        skills_text += f" and {len(skills)-5} more"
                
                # Add summary preview
                summary = rec.get('summary', "")
                summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
                
                final_table.append([
                    req,
                    html_link,
                    rec['duration'],
                    f"{summary_preview}<br><b>Skills:</b> {skills_text}<br>{rec.get('recommendation_reason', '')}"
                ])
            else:
                # No recommendation for this requirement
                final_table.append([
                    req,
                    "No matching courses found",
                    "N/A",
                    "No recommendations could be generated for this requirement"
                ])
    
    # Create and display the DataFrame
    final_df = pd.DataFrame(
        final_table,
        columns=["Requirement", "Course Recommendation", "Duration", "Recommendation Reason"]
    )
    
    # Display the HTML table with styling
    html_table = final_df.to_html(escape=False, index=False)
    st.markdown("""
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        a {
            color: #4287f5;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Generate downloadable CSV
    download_data = generate_download_data(all_requirements)
    download_df = pd.DataFrame(download_data)
    download_csv = download_df.to_csv(index=False)

    # Add download button for CSV with final recommendations
    st.download_button(
        label="Download Final Recommendations as CSV",
        data=download_csv,
        file_name="final_recommendations.csv",
        mime="text/csv",
    )

    # Generate downloadable XLSX with hyperlinks
    download_xlsx = generate_download_excel(all_requirements)
    st.download_button(
        label="Download Final Recommendations as XLSX",
        data=download_xlsx,
        file_name="final_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
    # Add button to start over
    if st.button("Start New Selection"):
        # Reset the session state
        st.session_state['rejected_indices'] = {}
        st.session_state['accepted_recommendations'] = {}
        st.session_state['already_recommended_urls'] = set()
        st.session_state['show_final_table'] = False
        st.rerun()

def generate_download_data(all_requirements):
    """Generate data for CSV download"""
    download_data = []
    for req in all_requirements:
        if req in st.session_state['accepted_recommendations']:
            rec = st.session_state['accepted_recommendations'][req]
            if rec is not None:
                download_data.append({
                    'Requirement': req,
                    'Course': rec['program_title'],
                    'URL': rec['url'],
                    'Duration': rec['duration'],
                    'Difficulty': rec['difficulty'],
                    'Type': rec['program_type'],
                    'Summary': rec.get('summary', ''),
                    'Skills': ', '.join(rec.get('skills', [])) if isinstance(rec.get('skills', []), (list, tuple)) else rec.get('skills', ''),
                    'Similarity Score': rec.get('similarity_score', '')
                })
            else:
                download_data.append({
                    'Requirement': req,
                    'Course': 'No matching courses found',
                    'URL': '',
                    'Duration': '',
                    'Difficulty': '',
                    'Type': '',
                    'Summary': '',
                    'Skills': '',
                    'Similarity Score': ''
                })
    return download_data

def generate_download_excel(all_requirements):
    """Generate an XLSX file with embedded hyperlinks"""
    excel_rows = []
    for req in all_requirements:
        if req in st.session_state['accepted_recommendations']:
            rec = st.session_state['accepted_recommendations'][req]
            if rec is not None:
                skills_text = ""
                skills = rec.get('skills', [])
                if isinstance(skills, (list, tuple)) and skills:
                    skills_text = ", ".join(skills[:5])
                    if len(skills) > 5:
                        skills_text += f" and {len(skills)-5} more"

                summary = rec.get('summary', "")
                summary_preview = summary[:100] + "..." if len(summary) > 100 else summary

                excel_rows.append({
                    'Requirement': req,
                    'Course Recommendation': rec['program_title'],
                    'URL': rec['url'],
                    'Duration': rec['duration'],
                    'Recommendation Reason': f"{summary_preview} | Skills: {skills_text} | {rec.get('recommendation_reason', '')}",
                })
            else:
                excel_rows.append({
                    'Requirement': req,
                    'Course Recommendation': 'No matching courses found',
                    'URL': '',
                    'Duration': 'N/A',
                    'Recommendation Reason': 'No recommendations could be generated for this requirement',
                })

    df = pd.DataFrame(excel_rows)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.drop(columns=['URL']).to_excel(writer, index=False, sheet_name='Recommendations')
        ws = writer.sheets['Recommendations']
        for idx, url in enumerate(df['URL'], start=2):
            if url:
                cell = ws.cell(row=idx, column=2)
                cell.value = df.loc[idx-2, 'Course Recommendation']
                cell.hyperlink = url
                cell.style = 'Hyperlink'
    output.seek(0)
    return output.getvalue()

def main():
    """Main application function"""
    # Set up the Streamlit page
    st.set_page_config(page_title="Learning Content Recommender", 
                       page_icon="📚", 
                       layout="wide")
    
    st.title("Learning Content Recommender")
    st.markdown("""
    This application helps you find relevant learning content based on your requirements.
    Upload a file with learning requirements, and we'll recommend the most relevant courses from Udacity.
    """)
    
    # Load programs data
    try:
        programs_df = prepare_programs_df()
    except Exception as e:
        st.error(f"Failed to load program data: {str(e)}")
        st.stop()
    
    # ========== FILE UPLOAD SECTION ==========
    st.header("Upload Learning Requirements")
    st.markdown("""
    Upload a file containing your learning requirements. 
    Supported formats:
    - CSV files with requirements (one per row)
    - Excel files with requirements (one per row)
    
    Or you can input requirements directly in the text area below.
    """)
    
    uploaded_file = st.file_uploader("Upload a file with learning requirements", 
                                    type=["csv", "xlsx", "xls"])
    
    direct_input = st.text_area("Or enter learning requirements directly (one per line)",
                               height=150,
                               placeholder="Example:\n1. Data analysis with Python\n2. Machine learning basics\n3. Cloud deployment of AI models")
    
    # Extract learning requirements
    learning_requirements = []
    
    if uploaded_file is not None:
        st.info(f"Processing uploaded file: {uploaded_file.name}")
        
        # First try to extract as a structured requirements file
        requirements = extract_requirements_from_file(uploaded_file)
        
        if requirements:
            learning_requirements = requirements
            st.success(f"Successfully extracted {len(requirements)} requirements from file")
        else:
            # If that fails, try to extract as general text
            uploaded_file.seek(0)  # Reset file pointer
            text = extract_text_from_file(uploaded_file)
            
            if text:
                # Use the advanced NLP extraction if available
                learning_requirements = extract_advanced_requirements(text)
                st.success(f"Extracted {len(learning_requirements)} requirements from text content")
            else:
                st.error("Could not extract text from the uploaded file")
    
    elif direct_input:
        # Process direct text input
        lines = direct_input.strip().split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        if cleaned_lines:
            learning_requirements = cleaned_lines
            st.success(f"Processed {len(learning_requirements)} requirements from text input")
    
    # ========== RECOMMENDATION SECTION ==========
    if learning_requirements:
        st.header("Learning Requirements Summary")

        # Display a summary of the requirements
        st.markdown("The following learning requirements were identified:")
        for i, req in enumerate(learning_requirements, 1):
            st.markdown(f"{i}. {req}")

        # ========== FILTER SELECTION ==========
        st.header("Select Filters")
        cols = st.columns(3)

        program_types_all = programs_df['program_type'].dropna().unique().tolist()
        difficulties_all = [
            d for d in programs_df['difficulty'].dropna().unique().tolist()
            if str(d).strip() and not (isinstance(d, float) and math.isnan(d))
        ]
        duration_categories_all = programs_df['duration_category'].dropna().unique().tolist()

        if 'selected_program_types' not in st.session_state:
            st.session_state['selected_program_types'] = program_types_all
        if 'selected_difficulties' not in st.session_state:
            st.session_state['selected_difficulties'] = difficulties_all
        else:
            st.session_state['selected_difficulties'] = [
                d for d in st.session_state['selected_difficulties'] if d in difficulties_all
            ] or difficulties_all
        if 'selected_duration_categories' not in st.session_state:
            st.session_state['selected_duration_categories'] = duration_categories_all

        with cols[0]:
            st.subheader("Program Type")
            selected_program_types = st.multiselect(
                "Select Program Types",
                options=program_types_all,
                default=st.session_state['selected_program_types'],
                key="filter_program_type"
            )

        with cols[1]:
            st.subheader("Difficulty")
            selected_difficulties = st.multiselect(
                "Select Difficulty Levels",
                options=difficulties_all,
                default=st.session_state['selected_difficulties'],
                key="filter_difficulty"
            )

        with cols[2]:
            st.subheader("Duration")
            selected_duration_categories = st.multiselect(
                "Select Duration Ranges",
                options=duration_categories_all,
                default=st.session_state['selected_duration_categories'],
                key="filter_duration"
            )

        generate_clicked = st.button("Generate Recommendations", type="primary")

        if generate_clicked:
            st.session_state['selected_program_types'] = selected_program_types
            st.session_state['selected_difficulties'] = selected_difficulties
            st.session_state['selected_duration_categories'] = selected_duration_categories
            st.session_state['filters_confirmed'] = True

        if st.session_state.get('filters_confirmed'):
            filtered_programs_df = programs_df[
                programs_df['program_type'].isin(st.session_state['selected_program_types']) &
                programs_df['difficulty'].isin(st.session_state['selected_difficulties']) &
                programs_df['duration_category'].isin(st.session_state['selected_duration_categories'])
            ]

            if filtered_programs_df.empty:
                st.warning("No programs match the selected filters")
            else:
                st.header("Generating Recommendations...")

                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    st.info("Using semantic search for recommendations...")
                    semantic_recommendations = recommend_content_semantic(
                        learning_requirements, filtered_programs_df, top_n=5
                    )
                    st.success("Semantic recommendations generated")

                    if semantic_recommendations.empty:
                        st.info("Falling back to TF-IDF search...")
                        tfidf_recommendations = recommend_content_tfidf(
                            learning_requirements, filtered_programs_df, top_n=5
                        )
                        all_recommendations = tfidf_recommendations
                    else:
                        st.info("Adding TF-IDF recommendations for comparison...")
                        tfidf_recommendations = recommend_content_tfidf(
                            learning_requirements, filtered_programs_df, top_n=3
                        )

                        if not semantic_recommendations.empty:
                            semantic_recommendations['recommendation_source'] = 'Semantic Search'
                        if not tfidf_recommendations.empty:
                            tfidf_recommendations['recommendation_source'] = 'TF-IDF'

                        all_recommendations = pd.concat(
                            [semantic_recommendations, tfidf_recommendations]
                        ).drop_duplicates(subset=['requirement', 'program_key'])
                else:
                    st.info("Using TF-IDF for recommendations (semantic search not available)...")
                    all_recommendations = recommend_content_tfidf(
                        learning_requirements, filtered_programs_df, top_n=5
                    )
                    if not all_recommendations.empty:
                        all_recommendations['recommendation_source'] = 'TF-IDF'

                if all_recommendations.empty:
                    st.warning("No recommendations found for the given requirements")
                else:
                    st.success(f"Generated {len(all_recommendations)} recommendations")
                    display_recommendations_ui(all_recommendations, learning_requirements, filtered_programs_df)

    else:
        st.info("Upload a file or enter learning requirements to get started")

if __name__ == "__main__":
    main() 