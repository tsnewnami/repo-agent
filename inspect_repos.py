import sqlite3
import os
from local_db import DB_PATH
from rich import print

def print_repos_with_over_n_functions(min_functions: int = 500, db_path: str = DB_PATH):
    """
    Print repositories that have over the specified number of functions stored in the database.
    Shows repository name and function count, ordered by count (highest first).
    
    Args:
        min_functions: Minimum number of functions required (default: 500)
        db_path: Path to the SQLite database
    """
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT repository_name, COUNT(*) as function_count
        FROM github_code
        GROUP BY repository_name
        HAVING COUNT(*) > ?
        ORDER BY function_count DESC
    """
    
    cursor.execute(query, (min_functions,))
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print(f"No repositories found with over {min_functions} functions.")
        return
    
    print(f"Found {len(results)} repositories with over {min_functions} functions:")
    print("-" * 60)
    for repo_name, count in results:
        print(f"{repo_name:<50} {count:>8} functions")


def get_functions_by_path(repo_name: str, min_functions_per_path: int = 2, db_path: str = DB_PATH) -> dict[str, int]:
    """
    Group functions by their path within a repository.
    Returns a dictionary mapping file paths to their function counts.
    Only includes paths that have at least min_functions_per_path functions.
    
    Args:
        repo_name: Name of the repository to analyze
        min_functions_per_path: Minimum number of functions required in a path to include it
        db_path: Path to the SQLite database
        
    Returns:
        Dict mapping file paths to their function counts
    """
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get paths and their function counts
    query = """
        SELECT func_path_in_repository, COUNT(*) as func_count
        FROM github_code 
        WHERE repository_name = ?
        GROUP BY func_path_in_repository
        HAVING COUNT(*) >= ?
        ORDER BY func_count DESC
    """
    
    cursor.execute(query, (repo_name, min_functions_per_path))
    results = cursor.fetchall()
    conn.close()
    
    # Build the return dictionary
    path_to_count: dict[str, int] = {}
    for path, count in results:
        path_to_count[path] = count
        
    return path_to_count


if __name__ == "__main__":
    print_repos_with_over_n_functions()
    # Example usage of new functions:
    print(get_functions_by_path("brocade/pynos", min_functions_per_path=3))