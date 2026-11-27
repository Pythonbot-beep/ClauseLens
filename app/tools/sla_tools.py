from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def calculate_sla_credit(
    monthly_spend_usd: float,
    downtime_minutes: float,
    credit_percentage: float,
    days_in_month: int = 30,
) -> dict:
    """Deterministic Python math for uptime % and exact SLA refund credits."""
    total_minutes = days_in_month * 24 * 60
    achieved_uptime_pct = max(0.0, ((total_minutes - downtime_minutes) / total_minutes) * 100.0)
    credit_amount_usd = round(monthly_spend_usd * (credit_percentage / 100.0), 2)

    return {
        "monthly_spend_usd": monthly_spend_usd,
        "downtime_minutes": downtime_minutes,
        "achieved_uptime_pct": round(achieved_uptime_pct, 4),
        "credit_percentage": credit_percentage,
        "credit_amount_usd": credit_amount_usd,
    }


async def query_sla_tiers(
    db: AsyncSession,
    vendor_name: str,
    service_name: str | None = None,
) -> list[dict]:
    """Parameterized read-only SQL lookup against structured SLA tiers."""
    result = await db.execute(
        text("""
            SELECT vendor_name, service_name, min_uptime_pct, max_uptime_pct,
                   credit_pct, claim_window_days
            FROM sla_tiers
            WHERE LOWER(vendor_name) = LOWER(:vendor_name)
              AND (:service_name IS NULL OR service_name ILIKE :service_pattern)
            ORDER BY min_uptime_pct DESC
        """),
        {
            "vendor_name": vendor_name,
            "service_name": service_name,
            "service_pattern": f"%{service_name}%" if service_name else None,
        },
    )
    return [dict(row._mapping) for row in result.fetchall()]