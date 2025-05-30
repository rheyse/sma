# Content Recommendation Engine

This is a Streamlit-based content recommendation engine that processes natural language inputs from CSV, PDF, and XLS files, extracts learning requirements using NLP techniques, and queries a content database to provide personalized recommendations.

## Features

- **File Upload**: Upload CSV, PDF, or XLS files to analyze.
- **Text Extraction**: Automatically extract text content from uploaded files.
- **NLP Processing**: Process the extracted text to identify key learning requirements.
- **Content Matching**: Match identified requirements with available learning content.
- **Interactive Results**: View recommendations with relevance scores and direct links.
- **Downloadable Output**: Export final recommendations as CSV or XLSX with embedded hyperlinks.

## Installation

1. Clone this repository
2. Install the required packages (including the spaCy model):
   ```
   pip install -r requirements.txt
   ```
3. Download the required NLTK data (the application will attempt to do this automatically)

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Upload a file containing learning requirements (CSV, PDF, or XLS format)
3. View the extracted text and learning requirements
4. Explore the recommended content based on your learning needs

### Configuring the Embedding Model

The application loads a sentence transformer model to compute semantic
similarity. By default it uses `all-mpnet-base-v2`. Set the
`SMA_EMBEDDING_MODEL` environment variable to override this model name
when starting the application.

## Technical Details

### Components

1. **File Ingestion Module**: Processes uploaded files using Pandas for CSV/XLS and PyMuPDF for PDFs.
2. **NLP & Parsing Module**: Uses NLTK for text processing and learning requirement extraction.
3. **Content Data Store**: Uses a Pandas DataFrame to store and query available learning content.
4. **Recommendation Engine**: Uses TF-IDF vectorization and cosine similarity to match requirements with content. Skill terms are given extra weight in the TF-IDF input to better reflect course relevance.
5. **Streamlit Frontend**: Provides an interactive user interface for the entire process.

### Data Flow

1. User uploads a file
2. Application extracts text from the file
3. NLP processing identifies learning requirements
4. Recommendation engine compares requirements with available content
5. Results are displayed in an interactive format

## Limitations

- The NLP processing is simplified for this prototype; in a production environment, more sophisticated techniques would be used.
- The content database is either loaded from a notebook or generated as dummy data for demonstration.

## Future Improvements

- Enhance NLP processing with named entity recognition and transformer models
- Implement more advanced matching algorithms
- Add user feedback loop to improve recommendations
- Expand file format support

## Updating Program Data

Recommendations are derived from the Udacity API or the bundled `sample_programs.csv` file. To refresh the dataset with the latest Udacity catalog, run:

```bash
python update_program_data.py
```

This will download the current catalog and overwrite `sample_programs.csv`. You can extend the script to integrate additional learning content providers.
