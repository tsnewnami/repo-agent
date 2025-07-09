import sqlite3
import os
import logging
from datasets import load_dataset, Dataset, Features, Value, Sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gh_issues.db")

DATASET_ID = "mlfoundations-dev/github-issues"

# --- Database Schema ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS github_issues_fts;
DROP TABLE IF EXISTS github_issues;

CREATE TABLE github_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_name TEXT NOT NULL,
    topic TEXT,
    issue_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    open BOOLEAN,
    created_at TEXT, -- Store as ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    updated_at TEXT, -- Store as ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    url TEXT,
    labels TEXT, -- JSON array stored as text
    user TEXT,
    comments INTEGER DEFAULT 0,
    
    UNIQUE(repo_name, issue_number)
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_github_issues_repo_name ON github_issues(repo_name);
CREATE INDEX idx_github_issues_topic ON github_issues(topic);
CREATE INDEX idx_github_issues_repo_issue ON github_issues(repo_name, issue_number);
CREATE INDEX idx_github_issues_open ON github_issues(open);
CREATE INDEX idx_github_issues_created_at ON github_issues(created_at);
CREATE INDEX idx_github_issues_user ON github_issues(user);
CREATE INDEX idx_github_issues_repo_topic_open ON github_issues(repo_name, topic, open);
CREATE INDEX idx_github_issues_topic_created ON github_issues(topic, created_at);

CREATE VIRTUAL TABLE github_issues_fts USING fts5(
    title,
    body,
    repo_name,
    user,
    content='github_issues',
    content_rowid='id'
);

CREATE TRIGGER github_issues_ai AFTER INSERT ON github_issues BEGIN
    INSERT INTO github_issues_fts (rowid, title, body, repo_name, user)
    VALUES (new.id, new.title, new.body, new.repo_name, new.user);
END;

CREATE TRIGGER github_issues_ad AFTER DELETE ON github_issues BEGIN
    DELETE FROM github_issues_fts WHERE rowid=old.id;
END;

CREATE TRIGGER github_issues_au AFTER UPDATE ON github_issues BEGIN
    UPDATE github_issues_fts SET 
        title=new.title, 
        body=new.body, 
        repo_name=new.repo_name, 
        user=new.user 
    WHERE rowid=old.id;
END;

INSERT INTO github_issues_fts (rowid, title, body, repo_name, user) 
SELECT id, title, body, repo_name, user FROM github_issues;
"""

# --- Functions ---
def load_github_issues_dataset():
    """
    Load the GitHub issues dataset from HuggingFace with proper schema validation.
    
    Returns:
        Dataset: The loaded dataset with validated schema
    """
    logging.info(f"Loading GitHub issues dataset from {DATASET_ID}")
    
    # Define the expected schema for validation
    features = Features({
        'repo_name': Value('string'),
        'topic': Value('string'), 
        'issue_number': Value('int64'),
        'title': Value('string'),
        'body': Value('string'),
        'open': Value('bool'),
        'created_at': Value('string'),  # ISO 8601 timestamp
        'updated_at': Value('string'),  # ISO 8601 timestamp
        'url': Value('string'),
        'labels': Sequence(Value('string')),  # Array of label strings
        'user': Value('string'),
        'comments': Value('int64')
    })
    
    try:
        logging.info("Downloading dataset from HuggingFace...")
        dataset = load_dataset(
            DATASET_ID,
            features=features,
            cache_dir=os.path.join(BASE_DIR, ".cache")
        )
        
        # Validate dataset type
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected dataset to be a dict, got {type(dataset)}")
        
        
        # Get the train split (main data)
        train_dataset = dataset['train']
        
        # Validate train dataset type
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"Expected train dataset to be a Dataset instance, got {type(train_dataset)}")
        
        logging.info(f"Successfully loaded dataset with {len(train_dataset)} records")
        
        return train_dataset
        
    except Exception as e:
        logging.error(f"Failed to load dataset {DATASET_ID}: {str(e)}")
        raise

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


if __name__ == "__main__":
    create_database(DB_PATH)