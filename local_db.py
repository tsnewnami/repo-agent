import sqlite3
import os
import logging
import json
from datasets import load_dataset
from tqdm import tqdm
from rich import print


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
    content='github_code',
    content_rowid='id'
);

CREATE TRIGGER github_code_ai AFTER INSERT ON github_code BEGIN
    INSERT INTO github_code_fts (rowid, func_name, whole_func_string, func_documentation_string, repository_name)
    VALUES (new.id, new.func_name, new.whole_func_string, new.func_documentation_string, new.repository_name);
END;

CREATE TRIGGER github_code_ad AFTER DELETE ON github_code BEGIN
    DELETE FROM github_code_fts WHERE rowid=old.id;
END;

CREATE TRIGGER github_code_au AFTER UPDATE ON github_code BEGIN
    UPDATE github_code_fts SET 
        func_name=new.func_name,
        whole_func_string=new.whole_func_string,
        func_documentation_string=new.func_documentation_string,
        repository_name=new.repository_name
    WHERE rowid=old.id;
END;

INSERT INTO github_code_fts (rowid, func_name, whole_func_string, func_documentation_string, repository_name) 
SELECT id, func_name, whole_func_string, func_documentation_string, repository_name FROM github_code;
"""

# --- Database Functions ---
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

def load_hf_dataset(language: str):
    """
    Load the CodeSearchNet dataset for the specified language from HuggingFace.

    Args:
        language (str): The programming language to load (e.g., 'python', 'javascript', etc.)
    Returns:
        DatasetDict: The loaded CodeSearchNet dataset splits for the given language
    """
    logging.info(f"Loading CodeSearchNet dataset for language: {language}")
    dataset = load_dataset(
        "code_search_net",
        language,
        cache_dir=os.path.join(BASE_DIR, ".cache")
    )
    logging.info(f"Successfully loaded CodeSearchNet dataset for {language} with splits: {list(dataset.keys())}")
    return dataset

def insert_dataset(dataset, db_path: str):
    """
    Insert all examples from a CodeSearchNet dataset (all splits) into the github_code table, with deduplication and bulk insert in a single transaction.

    Args:
        dataset: The loaded CodeSearchNet dataset (as returned by load_code_search_net_dataset)
        db_path: Path to the SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # --- Performance Pragmas ---
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    insert_sql = '''
        INSERT INTO github_code (
            repository_name,
            func_path_in_repository,
            func_name,
            whole_func_string,
            language,
            func_code_string,
            func_code_tokens,
            func_documentation_string,
            func_documentation_tokens,
            split_name,
            func_code_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_funcs = set()  # (repository_name, func_path_in_repository, func_name)

    conn.execute("BEGIN TRANSACTION;")
    for split_name in dataset.keys():
        for example in tqdm(dataset[split_name], desc=f"Inserting {split_name}"):
            if 'split_name' not in example or not example['split_name']:
                example = dict(example)
                example['split_name'] = split_name
            assert isinstance(example, dict)
            func_key = (
                example.get('repository_name', ''),
                example.get('func_path_in_repository', ''),
                example.get('func_name', ''),
            )
            if func_key in processed_funcs:
                duplicate_count += 1
                logging.info(f"Skipping duplicate function: {func_key}")
                continue
            func_docs = example.get('func_documentation_string', '')
            if not func_docs:
                skipped_count += 1
                logging.debug(f"Skipping function with no documentation: {func_key}")
                continue
            processed_funcs.add(func_key)
            try:
                cursor.execute(
                    insert_sql,
                    (
                        example.get('repository_name', ''),
                        example.get('func_path_in_repository', ''),
                        example.get('func_name', ''),
                        example.get('whole_func_string', ''),
                        example.get('language', ''),
                        example.get('func_code_string', ''),
                        json.dumps(example.get('func_code_tokens', [])),
                        example.get('func_documentation_string', ''),
                        json.dumps(example.get('func_documentation_tokens', [])),
                        example.get('split_name', ''),
                        example.get('func_code_url', ''),
                    )
                )
                record_count += 1
            except sqlite3.IntegrityError as e:
                skipped_count += 1
                logging.debug(f"Skipped row due to integrity error: {e}")
                continue
    conn.commit()
    conn.close()
    
    logging.info(f"Successfully inserted {record_count} function records.")
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} records due to integrity errors.")
    if duplicate_count > 0:
        logging.info(f"Skipped {duplicate_count} duplicate function records (based on repository_name, func_path_in_repository, func_name).")

def generate_database(languages: list[str], overwrite: bool = False, db_path: str = DB_PATH):
    """
    Generates the SQLite database from the CodeSearchNet dataset for the specified language.

    Args:
        language: The programming language to load from CodeSearchNet (default: 'python').
        overwrite: If True, any existing database file at db_path will be removed.
        db_path: The path where the SQLite database file should be created.
    """
    logging.info(
        f"Starting database generation at '{db_path}'"
    )
    logging.info(f"Overwrite existing database: {overwrite}")

    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        logging.info(f"Creating data directory: {db_dir}")
        os.makedirs(db_dir)

    if overwrite and os.path.exists(db_path):
        logging.warning(f"Removing existing database file: {db_path}")
        os.remove(db_path)
    elif not overwrite and os.path.exists(db_path):
        logging.warning(
            f"Database file {db_path} exists and overwrite is False. Assuming file is already generated."
        )
        return

    # 1. Create database schema (Tables only)
    create_database(db_path)

    # 2. Populate database
    for language in languages:
        dataset = load_hf_dataset(language)
        logging.info(f"Inserting CodeSearchNet dataset for language: {language}")
        insert_dataset(dataset, db_path)
        logging.info(f"Finished inserting CodeSearchNet dataset for language: {language}")

    # 3. Create Indexes and Triggers
    create_indexes_triggers(db_path)

    logging.info(f"Database generation process completed for {db_path}.")
    
def get_repository_counts(db_path: str):
    """
    Return a nested dictionary mapping each repository_name to a dict of split_name to count.
    Args:
        db_path: Path to the SQLite database
    Returns:
        dict: {repository_name: {split_name: count}}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT repository_name, split_name, COUNT(*)
        FROM github_code
        GROUP BY repository_name, split_name
    """)
    result = {}
    for repo, split, count in cursor.fetchall():
        if repo not in result:
            result[repo] = {}
        result[repo][split] = count
    conn.close()
    logging.info(f"Repository counts by split: {result}")
    return result

def get_first_n_from_repo(repo_name: str, n: int = 3, db_path: str = DB_PATH):
    """
    Get the first n entries from a specific repository in the github_code table.
    
    Args:
        repo_name: The repository name to search for
        n: Number of entries to return
        db_path: Path to the SQLite database
        
    Returns:
        list: List of dictionaries containing the entries, or empty list if no entries found
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM github_code 
        WHERE repository_name = ? 
        LIMIT ?
    """, (repo_name, n))
    
    results = cursor.fetchall()
    conn.close()
    
    ret = []
    if results:
        for row in results:
            # Convert sqlite3.Row to dict
            ret.append(dict(row))
            
    return ret

if __name__ == "__main__":
    # generate_database(languages=["python"], overwrite=True)
    print(get_first_n_from_repo("bcbio/bcbio-nextgen"))