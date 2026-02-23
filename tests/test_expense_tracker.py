"""Tests for BlackRoad Expense Tracker."""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from expense_tracker import (
    ExpenseDB, ExpenseTrackerService, Expense, Budget
)


@pytest.fixture
def svc(tmp_path):
    db = ExpenseDB(tmp_path / "expenses.db")
    return ExpenseTrackerService(db)


def test_add_expense_basic(svc):
    exp = svc.add_expense(Decimal("25.50"), "Lunch at Subway")
    assert exp.amount == Decimal("25.50")
    assert exp.description == "Lunch at Subway"
    assert exp.category == "food"


def test_add_expense_custom_category(svc):
    exp = svc.add_expense(Decimal("100"), "Random", category="other")
    assert exp.category == "other"


def test_add_expense_with_date(svc):
    d = date(2024, 3, 15)
    exp = svc.add_expense(Decimal("50"), "Coffee", expense_date=d)
    assert exp.date == d


def test_add_expense_with_tags(svc):
    exp = svc.add_expense(Decimal("30"), "Spotify", tags=["subscription", "music"])
    assert "subscription" in exp.tags


def test_add_expense_negative_raises(svc):
    with pytest.raises(ValueError, match="positive"):
        svc.add_expense(Decimal("-10"), "Invalid")


def test_add_expense_zero_raises(svc):
    with pytest.raises(ValueError):
        svc.add_expense(Decimal("0"), "Zero")


def test_categorize_food(svc):
    assert svc.categorize("McDonalds lunch") == "food"
    assert svc.categorize("Whole Foods grocery") == "food"


def test_categorize_transport(svc):
    assert svc.categorize("Uber ride downtown") == "transport"


def test_categorize_utilities(svc):
    assert svc.categorize("Netflix subscription") == "utilities"


def test_categorize_unknown(svc):
    assert svc.categorize("abcxyz unknown thing") == "other"


def test_monthly_summary_totals(svc):
    svc.add_expense(Decimal("50"), "Restaurant", expense_date=date(2024, 3, 10))
    svc.add_expense(Decimal("30"), "Coffee shop", expense_date=date(2024, 3, 15))
    svc.add_expense(Decimal("200"), "Grocery store", expense_date=date(2024, 3, 20))
    # Different month — should not appear
    svc.add_expense(Decimal("100"), "Uber", expense_date=date(2024, 4, 1))

    summary = svc.monthly_summary(2024, 3)
    assert summary.total == Decimal("280.00")
    assert summary.expense_count == 3


def test_monthly_summary_top_expense(svc):
    svc.add_expense(Decimal("10"), "Coffee", expense_date=date(2024, 5, 1))
    svc.add_expense(Decimal("500"), "Laptop", expense_date=date(2024, 5, 5), category="shopping")
    summary = svc.monthly_summary(2024, 5)
    assert summary.top_expense.amount == Decimal("500.00")


def test_budget_check_under(svc):
    svc.add_expense(Decimal("50"), "Lunch", expense_date=date(2024, 6, 10))
    result = svc.budget_check("food", Decimal("200"), 2024, 6)
    assert result["spent"] == Decimal("50.00")
    assert result["remaining"] == Decimal("150.00")
    assert not result["over_budget"]


def test_budget_check_over(svc):
    svc.add_expense(Decimal("150"), "Dinner out", expense_date=date(2024, 7, 1))
    svc.add_expense(Decimal("100"), "Pizza", expense_date=date(2024, 7, 15))
    result = svc.budget_check("food", Decimal("200"), 2024, 7)
    assert result["over_budget"] is True
    assert result["spent"] == Decimal("250.00")


def test_set_budget(svc):
    svc.set_budget("food", Decimal("500"), 2024, 8)
    budget = svc.db.get_budget("food", 2024, 8)
    assert budget is not None
    assert budget.monthly_limit == Decimal("500.00")


def test_recurring_detect(svc):
    # Add same expense 3 times
    for i in range(3):
        svc.add_expense(
            Decimal("14.99"), "Netflix subscription",
            expense_date=date(2024, i + 1, 1)
        )
    patterns = svc.recurring_detect(min_occurrences=2)
    assert len(patterns) >= 1
    # Netflix should be detected
    found = any("netflix" in p.description_key.lower() for p in patterns)
    assert found


def test_export_csv(svc):
    svc.add_expense(Decimal("25"), "Lunch", expense_date=date(2024, 9, 1))
    svc.add_expense(Decimal("15"), "Coffee", expense_date=date(2024, 9, 2))
    csv_out = svc.export_csv()
    lines = csv_out.strip().split("\n")
    assert len(lines) == 3  # header + 2 expenses
    assert "Date" in lines[0]
    assert "Category" in lines[0]


def test_delete_expense(svc):
    exp = svc.add_expense(Decimal("30"), "Test expense")
    assert svc.db.get_expense(exp.id) is not None
    deleted = svc.db.delete_expense(exp.id)
    assert deleted is True
    assert svc.db.get_expense(exp.id) is None


def test_expense_precision(svc):
    exp = svc.add_expense(Decimal("19.999"), "Rounding test")
    assert exp.amount == Decimal("20.00")
