def restock(stock, item, qty):
    updated = dict(stock)
    updated[item] = updated.get(item, 0) + qty
    return updated


def apply_discount(price, percent):
    return price * percent / 100


def total_value(stock, prices):
    return sum(qty * prices[item] for item, qty in stock.items() if qty > 0)
