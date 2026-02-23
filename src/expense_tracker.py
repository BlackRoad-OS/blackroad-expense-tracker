"""
BlackRoad Expense Tracker
==========================
Production-quality expense management with auto-categorization,
budget alerts, recurring detection, monthly summaries, and SQLite persistence.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sqlite3
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

DB_PATH = Path.home() / ".blackroad" / "expenses.db"
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("expense_tracker")

PRECISION = Decimal("0.01")


# ─── Categorization Rules ─────────────────────────────────────────────────────
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "food": ["restaurant", "cafe", "coffee", "lunch", "dinner", "breakfast",
             "pizza", "burger", "sushi", "taco", "grocery", "supermarket",
             "whole foods", "trader joe", "costco", "doordash", "ubereats"],
    "transport": ["uber", "lyft", "taxi", "gas", "fuel", "parking", "toll",
                  "metro", "subway", "bus", "train", "airline", "flight",
                  "rental car", "zipcar"],
    "utilities": ["electric", "electricity", "water", "gas bill", "internet",
                  "wifi", "phone", "mobile", "cable", "netflix", "spotify",
                  "subscription", "hulu", "disney+"],
    "housing": ["rent", "mortgage", "hoa", "insurance", "repair", "maintenance",
                "plumber", "electrician", "home depot", "lowe's", "ikea"],
    "healthcare": ["doctor", "dentist", "pharmacy", "hospital", "clinic",
                   "medication", "prescription", "insurance premium", "copay",
                   "therapy", "gym", "fitness"],
    "entertainment": ["movie", "cinema", "concert", "theater", "museum",
                      "sport", "game", "bar", "alcohol", "wine", "beer",
                      "amusement", "vacation", "hotel", "airbnb"],
    "shopping": ["amazon", "target", "walmart", "clothes", "clothing",
                 "shoes", "electronics", "best buy", "apple store", "ebay"],
    "education": ["tuition", "course", "book", "udemy", "coursera", "school",
                  "university", "college", "textbook", "workshop", "seminar"],
    "business": ["office", "software", "saas", "hosting", "domain", "aws",
                 "google cloud", "github", "tools", "equipment", "conference"],
    "personal": ["haircut", "salon", "spa", "beauty", "grooming", "laundry",
                 "dry clean", "gift", "donation"],
}


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class Expense:
    id: str
    amount: Decimal
    category: str
    description: str
    date: date
    tags: List[str] = field(default_factory=list)
    receipt_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_recurring: bool = False
    recurrence_id: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        if isinstance(self.amount, (int, float, str)):
            self.amount = Decimal(str(self.amount))
        if isinstance(self.date, str):
            self.date = date.fromisoformat(self.date)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.tags, str):
            self.tags = [t.strip() for t in self.tags.split(",") if t.strip()]


@dataclass
class Budget:
    category: str
    monthly_limit: Decimal
    year: int
    month: int

    def __post_init__(self):
        if isinstance(self.monthly_limit, (int, float, str)):
            self.monthly_limit = Decimal(str(self.monthly_limit))


@dataclass
class MonthlyCategory:
    category: str
    total: Decimal
    count: int
    percentage: Decimal
    over_budget: bool = False
    budget_limit: Optional[Decimal] = None


@dataclass
class MonthlySummary:
    year: int
    month: int
    total: Decimal
    by_category: List[MonthlyCategory]
    expense_count: int
    daily_average: Decimal
    top_expense: Optional[Expense] = None


@dataclass
class RecurringPattern:
    description_key: str
    occurrences: int
    average_amount: Decimal
    last_date: date
    estimated_next: date
    category: str


# ─── Database Layer ────────────────────────────────────────────────────────────
class ExpenseDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self.transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS expenses (
                    id            TEXT PRIMARY KEY,
                    amount        TEXT NOT NULL,
                    category      TEXT NOT NULL,
                    description   TEXT NOT NULL,
                    date          TEXT NOT NULL,
                    tags          TEXT NOT NULL DEFAULT '',
                    receipt_path  TEXT,
                    created_at    TEXT NOT NULL,
                    is_recurring  INTEGER NOT NULL DEFAULT 0,
                    recurrence_id TEXT,
                    notes         TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS categories (
                    name        TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    color       TEXT NOT NULL DEFAULT '#888888',
                    icon        TEXT NOT NULL DEFAULT '💰',
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS budgets (
                    id          TEXT PRIMARY KEY,
                    category    TEXT NOT NULL,
                    monthly_limit TEXT NOT NULL,
                    year        INTEGER NOT NULL,
                    month       INTEGER NOT NULL,
                    created_at  TEXT NOT NULL,
                    UNIQUE(category, year, month)
                );

                CREATE INDEX IF NOT EXISTS idx_expenses_date
                    ON expenses(date, category);
                CREATE INDEX IF NOT EXISTS idx_expenses_category
                    ON expenses(category);
            """)
            # Seed default categories
            cats = [
                ("food", "Food & Dining", "#FF6B6B", "🍽️"),
                ("transport", "Transportation", "#4ECDC4", "🚗"),
                ("utilities", "Utilities", "#45B7D1", "💡"),
                ("housing", "Housing", "#96CEB4", "🏠"),
                ("healthcare", "Healthcare", "#FFEAA7", "🏥"),
                ("entertainment", "Entertainment", "#DDA0DD", "🎭"),
                ("shopping", "Shopping", "#98D8C8", "🛍️"),
                ("education", "Education", "#F7DC6F", "📚"),
                ("business", "Business", "#AED6F1", "💼"),
                ("personal", "Personal", "#F8BBD9", "👤"),
                ("other", "Other", "#D7CCC8", "📋"),
            ]
            for name, display, color, icon in cats:
                conn.execute(
                    "INSERT OR IGNORE INTO categories (name, display_name, color, icon, created_at) VALUES (?,?,?,?,?)",
                    (name, display, color, icon, datetime.utcnow().isoformat()),
                )

    def save_expense(self, expense: Expense) -> Expense:
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO expenses
                   (id, amount, category, description, date, tags,
                    receipt_path, created_at, is_recurring, recurrence_id, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    expense.id, str(expense.amount), expense.category,
                    expense.description, expense.date.isoformat(),
                    ",".join(expense.tags), expense.receipt_path,
                    expense.created_at.isoformat(),
                    1 if expense.is_recurring else 0,
                    expense.recurrence_id, expense.notes,
                ),
            )
        return expense

    def get_expense(self, expense_id: str) -> Optional[Expense]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM expenses WHERE id=?", (expense_id,)).fetchone()
            return self._row_to_expense(row) if row else None
        finally:
            conn.close()

    def delete_expense(self, expense_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
            return cur.rowcount > 0

    def list_expenses(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Expense]:
        conn = self._connect()
        try:
            query = "SELECT * FROM expenses WHERE 1=1"
            params: list = []
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())
            if category:
                query += " AND category=?"
                params.append(category)
            query += " ORDER BY date DESC, created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_expense(r) for r in rows]
        finally:
            conn.close()

    def get_budget(self, category: str, year: int, month: int) -> Optional[Budget]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM budgets WHERE category=? AND year=? AND month=?",
                (category, year, month),
            ).fetchone()
            if not row:
                return None
            return Budget(
                category=row["category"],
                monthly_limit=Decimal(row["monthly_limit"]),
                year=row["year"],
                month=row["month"],
            )
        finally:
            conn.close()

    def set_budget(self, budget: Budget):
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO budgets (id, category, monthly_limit, year, month, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), budget.category, str(budget.monthly_limit),
                    budget.year, budget.month, datetime.utcnow().isoformat(),
                ),
            )

    @staticmethod
    def _row_to_expense(row: sqlite3.Row) -> Expense:
        return Expense(
            id=row["id"],
            amount=Decimal(row["amount"]),
            category=row["category"],
            description=row["description"],
            date=date.fromisoformat(row["date"]),
            tags=[t for t in (row["tags"] or "").split(",") if t],
            receipt_path=row["receipt_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            is_recurring=bool(row["is_recurring"]),
            recurrence_id=row["recurrence_id"],
            notes=row["notes"] or "",
        )


# ─── Expense Service ───────────────────────────────────────────────────────────
class ExpenseTrackerService:
    def __init__(self, db: Optional[ExpenseDB] = None):
        self.db = db or ExpenseDB()

    def categorize(self, description: str) -> str:
        """Auto-categorize by keyword matching."""
        desc_lower = description.lower()
        scores: Dict[str, int] = defaultdict(int)
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in desc_lower:
                    scores[category] += len(kw)  # longer match = higher score
        return max(scores, key=scores.get) if scores else "other"

    def add_expense(
        self,
        amount: Decimal,
        description: str,
        category: Optional[str] = None,
        expense_date: Optional[date] = None,
        tags: Optional[List[str]] = None,
        receipt_path: Optional[str] = None,
        notes: str = "",
    ) -> Expense:
        """Add a new expense with optional auto-categorization."""
        amount = Decimal(str(amount)).quantize(PRECISION, rounding=ROUND_HALF_UP)
        if amount <= Decimal("0"):
            raise ValueError("Expense amount must be positive")

        auto_category = category or self.categorize(description)
        expense = Expense(
            id=str(uuid.uuid4()),
            amount=amount,
            category=auto_category,
            description=description,
            date=expense_date or date.today(),
            tags=tags or [],
            receipt_path=receipt_path,
            notes=notes,
        )
        self.db.save_expense(expense)
        logger.info("Expense added: %s — $%s [%s]", description[:40], amount, auto_category)
        return expense

    def monthly_summary(self, year: int, month: int) -> MonthlySummary:
        """Generate monthly expense summary with budget checks."""
        from calendar import monthrange
        _, days_in_month = monthrange(year, month)
        start = date(year, month, 1)
        end = date(year, month, days_in_month)

        expenses = self.db.list_expenses(start_date=start, end_date=end)
        total = sum(e.amount for e in expenses)

        # Aggregate by category
        cat_totals: Dict[str, Decimal] = defaultdict(Decimal)
        cat_counts: Dict[str, int] = defaultdict(int)
        for e in expenses:
            cat_totals[e.category] += e.amount
            cat_counts[e.category] += 1

        by_category: List[MonthlyCategory] = []
        for cat, cat_total in sorted(cat_totals.items(), key=lambda x: -x[1]):
            pct = (cat_total / total * 100).quantize(PRECISION) if total > 0 else Decimal("0")
            budget = self.db.get_budget(cat, year, month)
            over = budget is not None and cat_total > budget.monthly_limit
            by_category.append(MonthlyCategory(
                category=cat,
                total=cat_total,
                count=cat_counts[cat],
                percentage=pct,
                over_budget=over,
                budget_limit=budget.monthly_limit if budget else None,
            ))

        top = max(expenses, key=lambda e: e.amount) if expenses else None
        daily_avg = (total / days_in_month).quantize(PRECISION) if expenses else Decimal("0")

        return MonthlySummary(
            year=year, month=month, total=total,
            by_category=by_category,
            expense_count=len(expenses),
            daily_average=daily_avg,
            top_expense=top,
        )

    def budget_check(self, category: str, limit: Decimal, year: int, month: int) -> Dict:
        """Check current month's spending against budget."""
        from calendar import monthrange
        _, days_in_month = monthrange(year, month)
        start = date(year, month, 1)
        end = date(year, month, days_in_month)

        expenses = self.db.list_expenses(start_date=start, end_date=end, category=category)
        spent = sum(e.amount for e in expenses)
        remaining = limit - spent
        percentage_used = (spent / limit * 100).quantize(PRECISION) if limit > 0 else Decimal("0")

        return {
            "category": category,
            "limit": limit,
            "spent": spent,
            "remaining": remaining,
            "percentage_used": percentage_used,
            "over_budget": spent > limit,
            "transactions": len(expenses),
        }

    def recurring_detect(
        self,
        min_occurrences: int = 2,
        description_similarity_threshold: float = 0.8,
        lookback_days: int = 90,
    ) -> List[RecurringPattern]:
        """Detect recurring expense patterns using fuzzy description matching."""
        cutoff = date.today() - timedelta(days=lookback_days)
        expenses = self.db.list_expenses(start_date=cutoff)

        # Normalize description: lowercase, strip numbers/punctuation
        def normalize(desc: str) -> str:
            cleaned = re.sub(r"\d+", "", desc.lower())
            cleaned = re.sub(r"[^\w\s]", "", cleaned)
            return " ".join(cleaned.split())

        groups: Dict[str, List[Expense]] = defaultdict(list)
        for exp in expenses:
            key = normalize(exp.description)
            groups[key].append(exp)

        patterns: List[RecurringPattern] = []
        for key, group in groups.items():
            if len(group) < min_occurrences:
                continue
            sorted_group = sorted(group, key=lambda e: e.date)
            amounts = [e.amount for e in sorted_group]
            avg_amount = sum(amounts) / len(amounts)

            # Estimate recurrence interval
            if len(sorted_group) >= 2:
                intervals = [
                    (sorted_group[i + 1].date - sorted_group[i].date).days
                    for i in range(len(sorted_group) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                last_date = sorted_group[-1].date
                next_date = last_date + timedelta(days=int(avg_interval))
            else:
                next_date = sorted_group[-1].date + timedelta(days=30)

            patterns.append(RecurringPattern(
                description_key=sorted_group[-1].description,
                occurrences=len(group),
                average_amount=avg_amount.quantize(PRECISION),
                last_date=sorted_group[-1].date,
                estimated_next=next_date,
                category=sorted_group[-1].category,
            ))

        return sorted(patterns, key=lambda p: -p.occurrences)

    def export_csv(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        category: Optional[str] = None,
    ) -> str:
        """Export expenses to CSV."""
        expenses = self.db.list_expenses(start_date, end_date, category)
        expenses_sorted = sorted(expenses, key=lambda e: e.date)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Category", "Description", "Amount", "Tags", "Receipt", "Notes"])
        for e in expenses_sorted:
            writer.writerow([
                e.date.isoformat(), e.category, e.description,
                str(e.amount), ",".join(e.tags), e.receipt_path or "", e.notes,
            ])
        return output.getvalue()

    def set_budget(self, category: str, limit: Decimal, year: int, month: int):
        budget = Budget(category=category, monthly_limit=Decimal(str(limit)), year=year, month=month)
        self.db.set_budget(budget)


# ─── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expenses", description="BlackRoad Expense Tracker")
    parser.add_argument("--db", default=str(DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="Add an expense")
    p.add_argument("amount")
    p.add_argument("description")
    p.add_argument("--category", default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--tags", default="")
    p.add_argument("--receipt", default=None)
    p.add_argument("--notes", default="")

    p = sub.add_parser("list", help="List expenses")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("summary", help="Monthly summary")
    today = date.today()
    p.add_argument("--year", type=int, default=today.year)
    p.add_argument("--month", type=int, default=today.month)

    p = sub.add_parser("budget", help="Set/check budget")
    p.add_argument("category")
    p.add_argument("limit")
    p.add_argument("--year", type=int, default=date.today().year)
    p.add_argument("--month", type=int, default=date.today().month)
    p.add_argument("--check", action="store_true")

    sub.add_parser("recurring", help="Detect recurring expenses")

    p = sub.add_parser("export", help="Export to CSV")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--category", default=None)

    p = sub.add_parser("categorize", help="Categorize a description")
    p.add_argument("description")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    svc = ExpenseTrackerService(ExpenseDB(Path(args.db)))

    if args.command == "add":
        exp_date = date.fromisoformat(args.date) if args.date else None
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        exp = svc.add_expense(
            Decimal(args.amount), args.description,
            category=args.category, expense_date=exp_date,
            tags=tags, receipt_path=args.receipt, notes=args.notes,
        )
        print(f"✓ Added: {exp.id[:8]} — ${exp.amount} [{exp.category}] {exp.description}")

    elif args.command == "list":
        start = date.fromisoformat(args.start) if args.start else None
        end = date.fromisoformat(args.end) if args.end else None
        expenses = svc.db.list_expenses(start, end, args.category, limit=args.limit)
        print(f"{'Date':<12} {'Category':<15} {'Amount':>8}  Description")
        print("─" * 70)
        for e in expenses:
            print(f"{e.date!s:<12} {e.category:<15} ${e.amount:>8,.2f}  {e.description[:40]}")

    elif args.command == "summary":
        s = svc.monthly_summary(args.year, args.month)
        import calendar
        print(f"\n{'='*55}")
        print(f"  EXPENSE SUMMARY — {calendar.month_name[args.month]} {args.year}")
        print(f"{'='*55}")
        print(f"  Total Expenses:  ${s.total:>10,.2f} ({s.expense_count} transactions)")
        print(f"  Daily Average:   ${s.daily_average:>10,.2f}")
        if s.top_expense:
            print(f"  Top Expense:     ${s.top_expense.amount:>10,.2f} — {s.top_expense.description[:30]}")
        print(f"{'─'*55}")
        for cat in s.by_category:
            flag = " ⚠️ OVER BUDGET" if cat.over_budget else ""
            budget_str = f" / ${cat.budget_limit:,.2f}" if cat.budget_limit else ""
            print(
                f"  {cat.category:<18} ${cat.total:>9,.2f}{budget_str}"
                f"  {cat.percentage:>5.1f}%  ({cat.count}){flag}"
            )

    elif args.command == "budget":
        if args.check:
            result = svc.budget_check(args.category, Decimal(args.limit), args.year, args.month)
            flag = " ⚠️ OVER BUDGET" if result["over_budget"] else " ✓"
            print(f"Budget check for {result['category']} ({args.year}-{args.month:02d}){flag}")
            print(f"  Limit:   ${result['limit']:>10,.2f}")
            print(f"  Spent:   ${result['spent']:>10,.2f}  ({result['percentage_used']}%)")
            print(f"  Left:    ${result['remaining']:>10,.2f}")
        else:
            svc.set_budget(args.category, Decimal(args.limit), args.year, args.month)
            print(f"✓ Budget set: {args.category} = ${Decimal(args.limit):,.2f}")

    elif args.command == "recurring":
        patterns = svc.recurring_detect()
        print(f"Detected {len(patterns)} recurring patterns:")
        for p in patterns[:20]:
            print(f"  {p.occurrences}x  ${p.average_amount:>8,.2f}  next≈{p.estimated_next}  {p.description_key[:40]}")

    elif args.command == "export":
        start = date.fromisoformat(args.start) if args.start else None
        end = date.fromisoformat(args.end) if args.end else None
        print(svc.export_csv(start, end, args.category))

    elif args.command == "categorize":
        cat = svc.categorize(args.description)
        print(f"Category: {cat}")


if __name__ == "__main__":
    main()
