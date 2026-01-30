import json

# 1. Read the standard dev file
with open('data/dev.json', 'r') as f:
    data = json.load(f)

# 2. Extract just the SQL queries to a new file
with open('data/dev_gold.sql', 'w') as f:
    for item in data:
        # Write the query followed by a newline
        f.write(item['query'] + '\n')

print("✅ Created data/dev_gold.sql")