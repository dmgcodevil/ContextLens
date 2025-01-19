from core import Label, create_graph, load_entities, merge

if __name__ == '__main__':

    rows = load_entities([
        "./demo/transactions_entities.json",
        "./demo/log_entities.json",
        "./demo/inventory_entities.json"
    ])

    for row in rows:
        print(row)

    graph = create_graph(rows)
    print("BUCKETS:")
    for bucket in graph.buckets.values():
        print(bucket)
    columns = [
        Label.TRANSACTION_ID,
        Label.TRANSACTION_DATE,
        Label.TRANSACTION_AMOUNT,
        Label.USER,
        Label.USER_ID,
        Label.PRODUCT_MAKE,
        Label.PRODUCT_MODEL,
        Label.PRODUCT_PRICE

    ]
    graph.debug_info()

    result = merge(graph, columns)
    for row in result:
        record_str = []
        for label, tuples in row.values.items():
            value_str = ",".join(map(lambda t: f"({t.row_id},{t.source_value})", tuples))
            record_str.append(f'({label.name}: [{value_str}])')
        metadata = f'score={row.score}, row_id={row.id}'
        print(metadata + ",".join(record_str))
