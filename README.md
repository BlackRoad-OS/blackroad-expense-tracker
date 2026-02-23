# BlackRoad Expense Tracker

> Expense management with keyword auto-categorization, budget alerts, recurring pattern detection, and monthly summaries.

Part of the [BlackRoad OS](https://github.com/BlackRoad-OS) platform.

## Features

- **Auto-categorization**: 100+ keywords across 10 categories (food, transport, utilities, etc.)
- **Monthly summaries**: Totals, per-category breakdown, daily average, top expense
- **Budget alerts**: Set per-category limits, get over-budget warnings
- **Recurring detection**: Fuzzy description matching finds subscription patterns
- **CSV export**: Full export with date range and category filters
- **Tags & receipts**: Attach tags and receipt paths to expenses

## Usage

```bash
# Add expense (auto-categorized)
python src/expense_tracker.py add 25.50 "Lunch at Subway"

# Add with explicit category
python src/expense_tracker.py add 15.99 "Netflix" --category utilities

# Monthly summary
python src/expense_tracker.py summary --year 2024 --month 3

# Set budget and check
python src/expense_tracker.py budget food 500 --year 2024 --month 3
python src/expense_tracker.py budget food 500 --check

# Detect recurring expenses
python src/expense_tracker.py recurring

# Export CSV
python src/expense_tracker.py export --start 2024-01-01 --end 2024-03-31
```

## Architecture

- `src/expense_tracker.py` — 626+ lines: `Expense`, `Budget`, `ExpenseDB`, `ExpenseTrackerService`
- `tests/` — 17 test functions
- SQLite: `expenses` + `categories` + `budgets` tables

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
