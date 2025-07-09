import sqlite3
import os
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "gh_issues.db")

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
