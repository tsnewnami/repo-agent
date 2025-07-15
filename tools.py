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
    func_snippet: str


def search_repo(
    repo_name: str,
    keywords: List[str],
    max_results: int = 20
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
        # Collect all search results from different methods
        all_results = []
        
        # Try FTS with original keywords
        fts_results = _search_with_fts(cursor, repo_name, keywords, max_results)
        all_results.extend(fts_results)
        
        # Try enhanced search with camelCase variations
        for keyword in keywords:
            enhanced_keywords = _generate_camelcase_variations([keyword])
            for enhanced_keyword in enhanced_keywords:
                enhanced_results = _search_with_fts(cursor, repo_name, [enhanced_keyword], max_results)
                all_results.extend(enhanced_results)
        
        # Try LIKE fallback for substring matching
        like_results = _search_with_like(cursor, repo_name, keywords, max_results)
        all_results.extend(like_results)
        
        # Remove duplicates while preserving order
        seen = set()
        search_results = []
        for result in all_results:
            # Create a unique key for each result
            key = (result.repo_name, result.func_path, result.func_name)
            if key not in seen:
                seen.add(key)
                search_results.append(result)
        
        # Limit to max_results
        search_results = search_results[:max_results]
        
        if not search_results:
            logging.warning(f"No results found for search: repo_name={repo_name}, keywords={keywords}")
            return None
        
        logging.info(f"Search completed. Returning {len(search_results)} SearchResult objects")
        return search_results
        
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        logging.error("Make sure you've run generate_database() from local_db.py first to create the database and FTS tables.")
        return []


def _generate_camelcase_variations(keywords: List[str]) -> List[str]:
    """Generate camelCase variations of keywords to improve search matching."""
    variations = []
    for keyword in keywords:
        # Original keyword
        variations.append(keyword)
        # Capitalized version
        variations.append(keyword.capitalize())
        # Common camelCase prefixes
        variations.append(f"Combine{keyword.capitalize()}")
        variations.append(f"Mix{keyword.capitalize()}")
        variations.append(f"Base{keyword.capitalize()}")
        # Prefix search for FTS (only wildcards at the end work in FTS5)
        variations.append(f"{keyword}*")
        variations.append(f"{keyword.capitalize()}*")
    return list(set(variations))  # Remove duplicates


def _search_with_fts(cursor, repo_name: str, keywords: List[str], max_results: int) -> List[SearchResult]:
    """Search using FTS5 full-text search."""
    # Build FTS query from keywords, escaping special characters
    escaped_keywords = []
    for keyword in keywords:
        # Escape FTS5 special characters by wrapping in double quotes
        if any(char in keyword for char in ['-', ' ', '"', '*']):
            # Escape any existing quotes and wrap in quotes
            escaped_keyword = '"' + keyword.replace('"', '""') + '"'
        else:
            escaped_keyword = keyword
        escaped_keywords.append(escaped_keyword)
    
    fts_query = " AND ".join(escaped_keywords) if escaped_keywords else "*"
    
    # Base query using FTS for keyword search
    query = """
    SELECT DISTINCT 
        gc.repository_name, 
        gc.func_path_in_repository, 
        gc.func_name,
        COALESCE(gc.func_documentation_string, '') as func_docs,
        COALESCE(gc.func_code_tokens, '') as func_code_tokens
    FROM github_code gc
    JOIN github_code_fts fts ON gc.id = fts.rowid
    WHERE github_code_fts MATCH ?
    AND gc.repository_name = ?
    LIMIT ?
    """
    
    params = [fts_query, repo_name, max_results]
    
    logging.info(f"FTS search: repo_name={repo_name}, keywords={keywords}")
    logging.debug(f"FTS query: {query}")
    logging.debug(f"FTS parameters: {params}")
    
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        logging.info(f"FTS search returned {len(results)} results")
        
        return _convert_to_search_results(results)
    except sqlite3.OperationalError as e:
        logging.warning(f"FTS search failed for keywords {keywords}: {e}")
        return []


def _search_with_like(cursor, repo_name: str, keywords: List[str], max_results: int) -> List[SearchResult]:
    """Fallback search using LIKE queries for substring matching."""
    if not keywords:
        return []
    
    # Build LIKE conditions for substring matching
    like_conditions = []
    params = []
    
    for keyword in keywords:
        like_conditions.append("""
            (gc.func_name LIKE ? OR 
             gc.func_documentation_string LIKE ? OR 
             gc.func_code_tokens LIKE ?)
        """)
        like_pattern = f"%{keyword}%"
        params.extend([like_pattern, like_pattern, like_pattern])
    
    # Combine conditions with AND
    where_clause = " AND ".join(like_conditions)
    
    query = f"""
    SELECT DISTINCT 
        gc.repository_name, 
        gc.func_path_in_repository, 
        gc.func_name,
        COALESCE(gc.func_documentation_string, '') as func_docs,
        COALESCE(gc.func_code_tokens, '') as func_code_tokens
    FROM github_code gc
    WHERE {where_clause}
    AND gc.repository_name = ?
    LIMIT ?
    """
    
    params.extend([repo_name, max_results])
    
    logging.info(f"LIKE search: repo_name={repo_name}, keywords={keywords}")
    logging.debug(f"LIKE query: {query}")
    logging.debug(f"LIKE parameters: {params}")
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    logging.info(f"LIKE search returned {len(results)} results")
    
    return _convert_to_search_results(results)


def _convert_to_search_results(results) -> List[SearchResult]:
    """Convert database results to SearchResult objects."""
    search_results = []
    for row in results:
        repo_name_result, func_path, func_name, func_docs, func_code_tokens = row
        
        # Truncate func_code_tokens to reasonable length
        tokens = func_code_tokens.split() if func_code_tokens else []
        func_snippet = ' '.join(tokens[:50])
        if len(tokens) > 50:
            func_snippet += "..."
        
        search_results.append(SearchResult(
            repo_name=repo_name_result,
            func_path=func_path,
            func_name=func_name,
            func_docs=func_docs,
            func_snippet=func_snippet
        ))
    
    return search_results


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
               language, func_documentation_string, func_code_tokens
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
        
        # Destructure database row into named variables (matches SELECT column order)
        (db_repo_name, db_func_path, db_func_name, db_whole_func_string, 
         db_language, db_func_documentation_string, db_func_code_tokens) = result
        
        # Convert to Function object with destructured variables
        function = Function(
            repo_name=db_repo_name,
            func_path_in_repository=db_func_path,
            func_name=db_func_name,
            whole_func_string=db_whole_func_string,
            language=db_language,
            func_documentation_string=db_func_documentation_string,
            code_tokens=db_func_code_tokens
        )
        
        return function
        
    except sqlite3.Error as e:
        logging.error(f"Database error while reading function: {e}")
        return None


if __name__ == "__main__":
    results = search_repo(
        repo_name="deepmind/sonnet",
        keywords=["curriculum"],
        max_results=5
    )
    print(results)
    print(read_repo_function(results[0].repo_name, results[0].func_path, results[0].func_name))