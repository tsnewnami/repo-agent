from dataclasses import dataclass
import sqlite3
import logging
from typing import Optional, List, Tuple
from local_db import DB_PATH
from data_types import Function
from rich import print

conn = None

def get_conn():
    global conn
    if conn is None:
        conn = sqlite3.connect(
            f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


@dataclass
class SearchResult:
    repo_name: str
    func_name: str
    func_docs: str


def search_repo(
    repo_name: str,
    keywords: List[str],
    max_results: int = 10
) -> Optional[List[SearchResult]]:
    """
    Search GitHub code functions using full-text search and filtering parameters.
    
    Args:
        repo_name: Repository name to search within
        keywords: List of keywords to search for in function name, code, and documentation
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        List of SearchResult objects containing matching functions, or None if no results found
    """
    conn = get_conn()
    cursor = conn.cursor()
    
    try:
        # Build FTS query from keywords
        fts_query = " AND ".join(keywords) if keywords else "*"
        
        # Base query using FTS for keyword search
        query = """
        SELECT DISTINCT 
            gc.repository_name, 
            gc.func_name, 
            COALESCE(gc.func_documentation_string, '') as func_docs
        FROM github_code gc
        JOIN github_code_fts fts ON gc.id = fts.rowid
        WHERE github_code_fts MATCH ?
        AND gc.repository_name = ?
        """
        
        params = [fts_query, repo_name]
        
        # Order by relevance (FTS5 returns results in relevance order by default) and limit results
        query += " LIMIT ?"
        params.append(max_results)
        
        logging.info(f"Executing search query: repo_name={repo_name}, keywords={keywords}, max_results={max_results}")
        logging.debug(f"SQL query: {query}")
        logging.debug(f"SQL parameters: {params}")
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        logging.info(f"Search query executed successfully. Returned {len(results)} rows")
        
        # Convert to SearchResult objects
        search_results = []
        for row in results:
            repo_name_result, func_name, func_docs = row
            
            # Truncate func_docs to reasonable length
            if len(func_docs) > 200:
                func_docs = func_docs[:197] + "..."
            
            search_results.append(SearchResult(
                repo_name=repo_name_result,
                func_name=func_name,
                func_docs=func_docs
            ))
        
        if not search_results:
            logging.warning(f"No results found for search: repo_name={repo_name}, keywords={keywords}")
            return None
        
        logging.info(f"Search completed. Returning {len(search_results)} SearchResult objects")
        return search_results
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        logging.error("Make sure you've run generate_database() from local_db.py first to create the database and FTS tables.")
        return []


# def read_issue(repo_name: str, issue_id: int) -> Optional[Issue]:
#     """
#     Read a specific GitHub issue by repository name and issue number.
    
#     Args:
#         repo_name: Repository name (e.g., "pytorch/pytorch")
#         issue_id: Issue number/ID
    
#     Returns:
#         Issue object if found, None if not found
#     """
#     conn = get_conn()
#     cursor = conn.cursor()
    
#     try:
#         query = """
#         SELECT id, repo_name, topic, issue_number, title, body, open, 
#                created_at, updated_at, url, labels, user, comments
#         FROM github_issues
#         WHERE repo_name = ? AND issue_number = ?
#         """
        
#         logging.info(f"Reading issue: repo_name={repo_name}, issue_id={issue_id}")
#         logging.debug(f"SQL query: {query}")
#         logging.debug(f"SQL parameters: [{repo_name}, {issue_id}]")
        
#         cursor.execute(query, [repo_name, issue_id])
#         result: Optional[Tuple] = cursor.fetchone()
        
#         if result is None:
#             logging.warning(f"No issue found for repo_name={repo_name}, issue_id={issue_id}")
#             return None
        
#         logging.info(f"Issue found successfully: {repo_name}#{issue_id}")
#         logging.debug(f"Raw result: {result}")
        
#         # Destructure database row into named variables (matches SELECT column order)
#         (db_id, db_repo_name, db_topic, db_issue_number, db_title, db_body, db_open,
#          db_created_at, db_updated_at, db_url, db_labels, db_user, db_comments) = result
        
#         # Convert to Issue object with destructured variables
#         issue = Issue(
#             id=db_id,
#             repo_name=db_repo_name,
#             topic=db_topic,
#             issue_number=db_issue_number,
#             title=db_title,
#             body=db_body,
#             open=db_open,
#             created_at=db_created_at,
#             updated_at=db_updated_at,
#             url=db_url,
#             labels=db_labels,
#             user=db_user,
#             comments=db_comments
#         )
        
#         logging.info(f"Issue object created successfully for {repo_name}#{issue_id}")
#         return issue
        
#     except sqlite3.Error as e:
#         logging.error(f"Database error while reading issue: {e}")
#         logging.error("Make sure you've run generate_database() from local_db.py first to create the database and FTS tables.")
#         return None


if __name__ == "__main__":
    results = search_repo(
        repo_name="bcbio/bcbio-nextgen",
        keywords=["fastq"],
        max_results=5
    )
    print(results)
    # print(read_issue(results[0].repo_name, results[0].issue_number))