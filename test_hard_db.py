"""
Hard Text-to-SQL Test Database
Creates a complex schema with multiple tables and relationships
"""
import sqlite3
import os

DB_PATH = "test_hard_database.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Companies and employees
c.execute('''CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL,
    manager_id INTEGER,
    location TEXT
)''')

c.execute('''CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    salary REAL,
    hire_date TEXT,
    manager_id INTEGER,
    status TEXT,
    FOREIGN KEY (department_id) REFERENCES departments(dept_id)
)''')

c.execute('''CREATE TABLE projects (
    project_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dept_id INTEGER,
    budget REAL,
    start_date TEXT,
    end_date TEXT,
    status TEXT,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
)''')

c.execute('''CREATE TABLE project_assignments (
    assignment_id INTEGER PRIMARY KEY,
    project_id INTEGER,
    emp_id INTEGER,
    hours_worked REAL,
    role TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
)''')

c.execute('''CREATE TABLE salaries (
    salary_id INTEGER PRIMARY KEY,
    emp_id INTEGER,
    amount REAL,
    effective_date TEXT,
    salary_type TEXT,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id)
)''')

c.execute('''CREATE TABLE clients (
    client_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    industry TEXT,
    contact_person TEXT,
    email TEXT
)''')

c.execute('''CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    client_id INTEGER,
    project_id INTEGER,
    order_date TEXT,
    amount REAL,
    status TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(client_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
)''')

# Insert data
dept_data = [
    (1, 'Engineering', 500000, 1, 'Building A'),
    (2, 'Marketing', 300000, 5, 'Building B'),
    (3, 'Sales', 400000, 9, 'Building C'),
    (4, 'HR', 150000, 12, 'Building A'),
    (5, 'Finance', 250000, 15, 'Building D'),
]

emp_data = [
    (1, 'Alice', 1, 120000, '2020-01-15', None, 'Active'),
    (2, 'Bob', 1, 95000, '2020-03-20', 1, 'Active'),
    (3, 'Charlie', 1, 88000, '2021-06-01', 1, 'Active'),
    (4, 'Diana', 1, 105000, '2019-11-10', 1, 'On Leave'),
    (5, 'Eve', 2, 98000, '2020-02-01', None, 'Active'),
    (6, 'Frank', 2, 75000, '2021-08-15', 5, 'Active'),
    (7, 'Grace', 3, 85000, '2020-07-01', 9, 'Active'),
    (8, 'Henry', 3, 72000, '2022-01-10', 9, 'Active'),
    (9, 'Ivan', 3, 110000, '2018-05-20', None, 'Active'),
    (10, 'Jane', 3, 68000, '2022-04-01', 9, 'Active'),
    (11, 'Kyle', 1, 92000, '2021-09-15', 1, 'Active'),
    (12, 'Laura', 4, 88000, '2019-12-01', None, 'Active'),
    (13, 'Mike', 4, 65000, '2021-03-20', 12, 'Active'),
    (14, 'Nina', 5, 92000, '2020-06-15', 15, 'Active'),
    (15, 'Oscar', 5, 115000, '2018-08-01', None, 'Active'),
    (16, 'Paula', 1, 78000, '2022-02-01', 1, 'Active'),
    (17, 'Quinn', 2, 82000, '2021-11-01', 5, 'Active'),
    (18, 'Rachel', 3, 95000, '2020-09-01', 9, 'Active'),
    (19, 'Steve', 1, 70000, '2023-01-15', 1, 'Active'),
    (20, 'Tina', 5, 78000, '2022-07-01', 15, 'Active'),
]

proj_data = [
    (1, 'Website Redesign', 1, 80000, '2024-01-01', '2024-06-30', 'In Progress'),
    (2, 'Mobile App', 1, 120000, '2024-02-01', '2024-08-31', 'In Progress'),
    (3, 'Q1 Campaign', 2, 50000, '2024-01-01', '2024-03-31', 'Completed'),
    (4, 'Q2 Campaign', 2, 60000, '2024-04-01', '2024-06-30', 'In Progress'),
    (5, 'Enterprise Sale', 3, 200000, '2024-01-01', '2024-12-31', 'In Progress'),
    (6, 'Recruitment System', 4, 30000, '2024-03-01', '2024-06-30', 'Completed'),
    (7, 'Budget Analysis', 5, 45000, '2024-01-01', '2024-03-31', 'Completed'),
    (8, 'Cloud Migration', 1, 150000, '2024-05-01', '2024-12-31', 'In Progress'),
]

assign_data = [
    (1, 1, 1, 40, 'Lead'),
    (2, 1, 2, 80, 'Developer'),
    (3, 1, 3, 80, 'Developer'),
    (4, 1, 11, 60, 'Developer'),
    (5, 2, 1, 30, 'Lead'),
    (6, 2, 4, 80, 'Developer'),
    (7, 2, 11, 80, 'Developer'),
    (8, 2, 16, 80, 'Developer'),
    (9, 3, 5, 20, 'Lead'),
    (10, 3, 6, 100, 'Marketer'),
    (11, 4, 5, 30, 'Lead'),
    (12, 4, 6, 80, 'Marketer'),
    (13, 4, 17, 60, 'Marketer'),
    (14, 5, 9, 20, 'Lead'),
    (15, 5, 7, 100, 'Sales'),
    (16, 5, 8, 100, 'Sales'),
    (17, 5, 10, 80, 'Sales'),
    (18, 5, 18, 80, 'Sales'),
    (19, 6, 12, 20, 'Lead'),
    (20, 6, 13, 100, 'HR'),
    (21, 7, 15, 40, 'Lead'),
    (22, 7, 14, 60, 'Analyst'),
    (23, 7, 20, 80, 'Analyst'),
    (24, 8, 2, 40, 'Developer'),
    (25, 8, 3, 60, 'Developer'),
    (26, 8, 16, 60, 'Developer'),
    (27, 8, 19, 100, 'Developer'),
]

salary_data = [
    (1, 1, 120000, '2024-01-01', 'Base'),
    (2, 2, 95000, '2024-01-01', 'Base'),
    (3, 3, 88000, '2024-01-01', 'Base'),
    (4, 4, 105000, '2024-01-01', 'Base'),
    (5, 5, 98000, '2024-01-01', 'Base'),
    (6, 6, 75000, '2024-01-01', 'Base'),
    (7, 7, 85000, '2024-01-01', 'Base'),
    (8, 8, 72000, '2024-01-01', 'Base'),
    (9, 9, 110000, '2024-01-01', 'Base'),
    (10, 10, 68000, '2024-01-01', 'Base'),
]

client_data = [
    (1, 'TechCorp', 'Technology', 'John Smith', 'john@techcorp.com'),
    (2, 'GlobalBank', 'Finance', 'Mary Johnson', 'mary@globalbank.com'),
    (3, 'HealthPlus', 'Healthcare', 'David Brown', 'david@healthplus.com'),
    (4, 'EduWorld', 'Education', 'Sarah Wilson', 'sarah@eduworld.com'),
    (5, 'RetailMax', 'Retail', 'Chris Lee', 'chris@retailmax.com'),
]

order_data = [
    (1, 1, 1, '2024-01-15', 50000, 'Completed'),
    (2, 2, 5, '2024-02-01', 100000, 'Completed'),
    (3, 3, 5, '2024-02-15', 75000, 'In Progress'),
    (4, 4, 5, '2024-03-01', 50000, 'In Progress'),
    (5, 5, 2, '2024-01-20', 25000, 'Completed'),
]

c.executemany('INSERT INTO departments VALUES (?,?,?,?,?)', dept_data)
c.executemany('INSERT INTO employees VALUES (?,?,?,?,?,?,?)', emp_data)
c.executemany('INSERT INTO projects VALUES (?,?,?,?,?,?,?)', proj_data)
c.executemany('INSERT INTO project_assignments VALUES (?,?,?,?,?)', assign_data)
c.executemany('INSERT INTO salaries VALUES (?,?,?,?,?)', salary_data)
c.executemany('INSERT INTO clients VALUES (?,?,?,?,?)', client_data)
c.executemany('INSERT INTO orders VALUES (?,?,?,?,?,?)', order_data)

conn.commit()
conn.close()

print(f"Created: {DB_PATH}")
print("Tables: departments, employees, projects, project_assignments, salaries, clients, orders")
