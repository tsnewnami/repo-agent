import sqlite3
import os
import logging
from datasets import load_dataset, Dataset, Features, Value, Sequence
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "github_code.db")

# --- Database Schema ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS github_code_fts;
DROP TABLE IF EXISTS github_code;

CREATE TABLE github_code (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repository_name TEXT NOT NULL,
    func_path_in_repository TEXT NOT NULL,
    func_name TEXT NOT NULL,
    whole_func_string TEXT NOT NULL,
    language TEXT NOT NULL,
    func_code_string TEXT,
    func_code_tokens TEXT, -- JSON array stored as text
    func_documentation_string TEXT,
    func_documentation_tokens TEXT, -- JSON array stored as text
    split_name TEXT,
    func_code_url TEXT,
    
    UNIQUE(repository_name, func_path_in_repository, func_name)
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_github_code_repository_name ON github_code(repository_name);
CREATE INDEX idx_github_code_language ON github_code(language);
CREATE INDEX idx_github_code_func_name ON github_code(func_name);
CREATE INDEX idx_github_code_split_name ON github_code(split_name);
CREATE INDEX idx_github_code_repo_lang ON github_code(repository_name, language);
CREATE INDEX idx_github_code_repo_path_func ON github_code(repository_name, func_path_in_repository, func_name);

CREATE VIRTUAL TABLE github_code_fts USING fts5(
    func_name,
    whole_func_string,
    func_documentation_string,
    repository_name,
    language,
    content='github_code',
    content_rowid='id'
);

CREATE TRIGGER github_code_ai AFTER INSERT ON github_code BEGIN
    INSERT INTO github_code_fts (rowid, func_name, whole_func_string, func_documentation_string, repository_name, language)
    VALUES (new.id, new.func_name, new.whole_func_string, new.func_documentation_string, new.repository_name, new.language);
END;

CREATE TRIGGER github_code_ad AFTER DELETE ON github_code BEGIN
    DELETE FROM github_code_fts WHERE rowid=old.id;
END;

CREATE TRIGGER github_code_au AFTER UPDATE ON github_code BEGIN
    UPDATE github_code_fts SET 
        func_name=new.func_name,
        whole_func_string=new.whole_func_string,
        func_documentation_string=new.func_documentation_string,
        repository_name=new.repository_name,
        language=new.language
    WHERE rowid=old.id;
END;

INSERT INTO github_code_fts (rowid, func_name, whole_func_string, func_documentation_string, repository_name, language) 
SELECT id, func_name, whole_func_string, func_documentation_string, repository_name, language FROM github_code;
"""

# --- Functions ---
def load_github_issues_dataset(dataset_id: str):
    """
    Load the GitHub issues dataset from HuggingFace with proper schema validation.
    
    Returns:
        Dataset: The loaded dataset with validated schema
    """
    logging.info(f"Loading GitHub issues dataset from {dataset_id}")
    
    # Define the expected schema for validation
    features = Features({
        'repo_name': Value('string'),
        'topic': Value('string'), 
        'issue_number': Value('int64'),
        'title': Value('string'),
        'body': Value('string'),
        'state': Value('string'),
        'created_at': Value('string'),  # ISO 8601 timestamp
        'updated_at': Value('string'),  # ISO 8601 timestamp
        'url': Value('string'),
        'labels': Sequence(Value('string')),  # Array of label strings
        'user_login': Value('string'),
        'comments_count': Value('int64')
    })
    
    logging.info("Downloading dataset from HuggingFace...")
    dataset = load_dataset(
        dataset_id,
        features=features,
        cache_dir=os.path.join(BASE_DIR, ".cache")
    )
    
    # Validate dataset type
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset to be a dict, got {type(dataset)}")
    
    # Get the train split (main data)
    train_dataset = dataset['train']
    
    # Validate train dataset type
    if not isinstance(train_dataset, Dataset):
        raise TypeError(f"Expected train dataset to be a Dataset instance, got {type(train_dataset)}")
    
    logging.info(f"Successfully loaded dataset with {len(train_dataset)} records")
    
    return train_dataset

def create_database(db_path: str):
    """
    Create the SQLite database with tables, indexes, and triggers.
    """
    logging.info(f"Creating SQLite database and tables at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Database tables created successfully.")


def create_indexes_triggers(db_path: str):
    """
    Create indexes and triggers for the SQLite database.
    """
    logging.info(f"Creating indexes and triggers at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Indexes and triggers created successfully.")


def insert_dataset_to_db(dataset: Dataset, db_path: str):
    """
    Insert the HuggingFace dataset into the SQLite database.
    
    Maps dataset columns to database columns:
    - user_login -> user
    - comments_count -> comments  
    - state -> open (converted to boolean)
    """
    logging.info(f"Populating database {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # --- Performance Pragmas ---
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    
    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_issues = set()  # Track (repo_name, issue_number) tuples to dedupe
    
    conn.execute("BEGIN TRANSACTION;")  # Single transaction for bulk insert
    
    for issue_data in tqdm(dataset, desc="Inserting issues"):
        assert isinstance(issue_data, dict)
        
        repo_name = issue_data["repo_name"]
        topic = issue_data["topic"]
        issue_number = issue_data["issue_number"]
        title = issue_data["title"]
        body = issue_data["body"]
        state = issue_data["state"]
        created_at = issue_data["created_at"]
        updated_at = issue_data["updated_at"]
        url = issue_data["url"]
        labels = issue_data["labels"]
        user_login = issue_data["user_login"]
        comments_count = issue_data["comments_count"]
        
        # --- Data Cleaning and Filtering ---
        # Skip issues with extremely long bodies (>100k chars)
        if body and len(body) > 100000:
            logging.debug(f"Skipping issue {repo_name}#{issue_number}: Body length > 100k characters.")
            skipped_count += 1
            continue
        
        # Skip issues with missing critical fields
        if not repo_name or not title or issue_number <= 0:
            logging.debug(f"Skipping issue with missing critical fields: {repo_name}#{issue_number}")
            skipped_count += 1
            continue
        
        # --- Deduplication Check ---
        issue_key = (repo_name, issue_number)
        if issue_key in processed_issues:
            logging.debug(f"Skipping duplicate issue: {repo_name}#{issue_number}")
            duplicate_count += 1
            continue
        else:
            processed_issues.add(issue_key)
        
        # --- Data Transformation ---
        open_status = state == 'open'  # Convert string to boolean
        labels_str = str(labels) if labels else None  # Convert list to string
        
        cursor.execute(
            """
            INSERT INTO github_issues (
                repo_name, topic, issue_number, title, body, open, 
                created_at, updated_at, url, labels, user, comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (repo_name, topic, issue_number, title, body, open_status,
             created_at, updated_at, url, labels_str, user_login, comments_count)
        )
        record_count += 1
    
    conn.commit()
    conn.close()
    
    logging.info(f"Successfully inserted {record_count} issue records.")
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} issue records due to data quality issues.")
    if duplicate_count > 0:
        logging.info(f"Skipped {duplicate_count} duplicate issue records (based on repo_name, issue_number).")


def generate_database(overwrite: bool = False):
    """
    Generates the SQLite database from the specified Hugging Face dataset.
    Simplified version without extensive error handling.

    Args:
        overwrite: If True, any existing database file at DB_PATH will be removed.
    """
    logging.info(f"Starting database generation for repo '{DATASET_ID}' at '{DB_PATH}'")
    logging.info(f"Overwrite existing database: {overwrite}")

    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        logging.info(f"Creating data directory: {db_dir}")
        os.makedirs(db_dir)

    if overwrite and os.path.exists(DB_PATH):
        logging.warning(f"Removing existing database file: {DB_PATH}")
        os.remove(DB_PATH)
    elif not overwrite and os.path.exists(DB_PATH):
        # If not overwriting and file exists, subsequent steps might fail or behave unexpectedly.
        # We are removing the explicit error here as requested.
        logging.warning(
            f"Database file {DB_PATH} exists and overwrite is False. Assuming file is already generated."
        )
        return

    # 1. Download dataset
    dataset = load_github_issues_dataset(DATASET_ID)

    # 2. Create database schema (Tables only)
    # Note: This will fail if overwrite=False and the file exists with incompatible schema/data.
    create_database(DB_PATH)

    # 3. Populate database
    insert_dataset_to_db(dataset, DB_PATH)

    # 4. Create Indexes and Triggers
    create_indexes_triggers(DB_PATH)

    logging.info(f"Database generation process completed for {DB_PATH}.")


def print_repo_row_counts(db_path: str = DB_PATH):
    """
    Print the number of rows (issues) for each repository in the database.
    
    Args:
        db_path: Path to the SQLite database file. Defaults to DB_PATH.
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file does not exist: {db_path}")
        print(f"Error: Database file does not exist: {db_path}")
        return
    
    logging.info(f"Counting rows by repo_name in database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT repo_name, COUNT(*) as issue_count
            FROM github_issues
            GROUP BY repo_name
            ORDER BY issue_count DESC
        """)
        
        results = cursor.fetchall()
        
        if not results:
            print("No data found in the database.")
            return
        
        print(f"\nRepository Issue Counts:")
        print("-" * 50)
        print(f"{'Repository Name':<40} {'Issue Count':<10}")
        print("-" * 50)
        
        total_issues = 0
        for repo_name, count in results:
            print(f"{repo_name:<40} {count:<10}")
            total_issues += count
        
        print("-" * 50)
        print(f"{'Total Issues':<40} {total_issues:<10}")
        print(f"{'Total Repositories':<40} {len(results):<10}")
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        print(f"Database error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    print("Creating database...")
    generate_database(overwrite=False)
    print_repo_row_counts()