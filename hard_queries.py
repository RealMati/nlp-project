"""
Hard Text-to-SQL Test Queries
Level 1-5 difficulty for stress-testing your model
"""

HARD_QUERIES = [
    # === LEVEL 1: Basic Joins ===
    {
        "level": 1,
        "query": "List all employees with their department names",
        "expected_keywords": ["SELECT", "employees", "departments", "department_id", "dept_id", "JOIN"],
        "difficulty": "Easy"
    },
    {
        "level": 1,
        "query": "Show all projects along with the department managing them",
        "expected_keywords": ["SELECT", "projects", "departments", "JOIN"],
        "difficulty": "Easy"
    },

    # === LEVEL 2: Aggregations with Joins ===
    {
        "level": 2,
        "query": "Find the total salary budget per department",
        "expected_keywords": ["SELECT", "departments", "SUM", "salary", "GROUP BY"],
        "difficulty": "Medium"
    },
    {
        "level": 2,
        "query": "Count how many employees work in each department",
        "expected_keywords": ["SELECT", "departments", "COUNT", "employees", "GROUP BY"],
        "difficulty": "Medium"
    },
    {
        "level": 2,
        "query": "Calculate the average salary per department",
        "expected_keywords": ["SELECT", "AVG", "departments", "GROUP BY"],
        "difficulty": "Medium"
    },

    # === LEVEL 3: Complex Conditions ===
    {
        "level": 3,
        "query": "Find employees who earn more than the average salary in Engineering",
        "expected_keywords": ["SELECT", "employees", "salary", "AVG", "subquery"],
        "difficulty": "Hard"
    },
    {
        "level": 3,
        "query": "List projects that are over budget (actual assignments cost exceeds project budget)",
        "expected_keywords": ["SELECT", "projects", "project_assignments", "SUM", "hours_worked", ">"],
        "difficulty": "Hard"
    },
    {
        "level": 3,
        "query": "Find employees who have not been assigned to any project",
        "expected_keywords": ["SELECT", "employees", "LEFT JOIN", "NULL", "project_assignments"],
        "difficulty": "Hard"
    },

    # === LEVEL 4: Multi-Level Aggregation ===
    {
        "level": 4,
        "query": "For each project, show the total hours worked and the cost (hours × avg employee salary / 100)",
        "expected_keywords": ["SELECT", "projects", "project_assignments", "employees", "SUM", "AVG", "JOIN", "JOIN"],
        "difficulty": "Very Hard"
    },
    {
        "level": 4,
        "query": "Find the top 3 departments with highest total project hours",
        "expected_keywords": ["SELECT", "departments", "project_assignments", "SUM", "GROUP BY", "ORDER BY", "LIMIT"],
        "difficulty": "Very Hard"
    },
    {
        "level": 4,
        "query": "List clients who have placed orders totaling over $100,000",
        "expected_keywords": ["SELECT", "clients", "orders", "SUM", "GROUP BY", "HAVING"],
        "difficulty": "Very Hard"
    },

    # === LEVEL 5: Extreme Complexity ===
    {
        "level": 5,
        "query": "Find employees who are managers and whose team has total project hours exceeding 300 hours",
        "expected_keywords": ["SELECT", "employees", "manager_id", "project_assignments", "SUM", "GROUP BY", "HAVING"],
        "difficulty": "Extreme"
    },
    {
        "level": 5,
        "query": "Calculate the profit margin for each project: (order_amount - total_assignment_cost) / order_amount × 100",
        "expected_keywords": ["SELECT", "projects", "orders", "project_assignments", "ROUND", "(", "-", ")"],
        "difficulty": "Extreme"
    },
    {
        "level": 5,
        "query": "Find the department with the highest cost efficiency: (total billable hours / total salary cost)",
        "expected_keywords": ["SELECT", "departments", "project_assignments", "employees", "AVG", "SUM", "/"],
        "difficulty": "Extreme"
    },
    {
        "level": 5,
        "query": "List employees whose salary is in the top 20% but have below average project hours",
        "expected_keywords": ["SELECT", "employees", "salary", "PERCENTILE", "AVG", "project_assignments"],
        "difficulty": "Extreme"
    },
]


def print_queries():
    print("=" * 80)
    print("HARD TEXT-TO-SQL TEST QUERIES")
    print("=" * 80)
    print()

    for i, q in enumerate(HARD_QUERIES, 1):
        print(f"{'='*80}")
        print(f"Query #{i} | Level {q['level']} | {q['difficulty']}")
        print(f"{'='*80}")
        print(f"\nQuestion:\n  \"{q['query']}\"\n")
        print(f"Expected keywords: {', '.join(q['expected_keywords'])}")
        print()

if __name__ == "__main__":
    print_queries()
