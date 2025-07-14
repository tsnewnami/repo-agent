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
    func_path: str
    func_name: str
    func_docs: str


def search_repo(
    repo_name: str,
    keywords: List[str],
    max_results: int = 15
) -> Optional[List[SearchResult]]:
    """
    Search GitHub code functions using full-text search and filtering parameters.
    
    Args:
        repo_name: Repository name to search within
        keywords: List of keywords to search for in function name, code, and documentation
        max_results: Maximum number of results to return
    
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
            gc.func_path_in_repository, 
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
            repo_name_result, func_path, func_name, func_docs = row
            
            # Truncate func_docs to reasonable length
            if len(func_docs) > 200:
                func_docs = func_docs[:197] + "..."
            
            search_results.append(SearchResult(
                repo_name=repo_name_result,
                func_path=func_path,
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


def read_repo_function(repo_name: str, func_path: str, func_name: str) -> Optional[Function]:
    """
    Read a specific GitHub function by repository name, function path, and function name.
    
    Args:
        repo_name: Repository name (e.g., "pytorch/pytorch")
        func_path: Function path in repository
        func_name: Function name
    
    Returns:
        Function object if found, None if not found
    """
    conn = get_conn()
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT repository_name, func_path_in_repository, func_name, whole_func_string, 
               language, func_documentation_string
        FROM github_code
        WHERE repository_name = ? AND func_path_in_repository = ? AND func_name = ?
        """
        
        logging.info(f"Reading function: repo_name={repo_name}, func_path={func_path}, func_name={func_name}")
        logging.debug(f"SQL query: {query}")
        logging.debug(f"SQL parameters: [{repo_name}, {func_path}, {func_name}]")
        
        cursor.execute(query, [repo_name, func_path, func_name])
        result: Optional[Tuple] = cursor.fetchone()
        
        if result is None:
            logging.warning(f"No function found for repo_name={repo_name}, func_path={func_path}, func_name={func_name}")
            return None
        
        logging.info(f"Function found successfully: {repo_name}#{func_path}#{func_name}")
        logging.debug(f"Raw result: {result}")
        
        # Destructure database row into named variables (matches SELECT column order)
        (db_repo_name, db_func_path, db_func_name, db_whole_func_string, 
         db_language, db_func_documentation_string) = result
        
        # Convert to Function object with destructured variables
        function = Function(
            repo_name=db_repo_name,
            func_path_in_repository=db_func_path,
            func_name=db_func_name,
            whole_func_string=db_whole_func_string,
            language=db_language,
            func_documentation_string=db_func_documentation_string
        )
        
        logging.info(f"Function object created successfully for {repo_name}#{func_path}#{func_name}")
        return function
        
    except sqlite3.Error as e:
        logging.error(f"Database error while reading function: {e}")
        logging.error("Make sure you've run generate_database() from local_db.py first to create the database and FTS tables.")
        return None


if __name__ == "__main__":
    results = search_repo(
        repo_name="bcbio/bcbio-nextgen",
        keywords=["fastq"],
        max_results=5
    )
    print(results)
    print(read_function(results[0].repo_name, results[0].func_path, results[0].func_name))