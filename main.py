import argparse
from tabulate import tabulate
from core import create_graph, load_entities, merge, logger

if __name__ == '__main__':

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Merge Rows")

    # Add arguments
    parser.add_argument(
        "--entity-files",
        type=str,
        required=True,
        help="Comma-separated file paths for entity files."
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Comma-separated list of labels."
    )

    # Parse the arguments
    args = parser.parse_args()

    entity_files = args.entity_files.split(",")
    labels = args.labels.split(",")

    rows = load_entities(entity_files)

    for row in rows:
        logger.debug(row)

    graph = create_graph(rows)
    logger.debug("BUCKETS:")
    for bucket in graph.buckets.values():
        logger.debug(bucket)

    graph.debug_info()

    result = merge(graph, labels)
    logger.debug("RESULT\n\n")

    # Prepare data for the ASCII table
    table_data = []
    for row in result:
        row_data = {
            "Row ID": row.id,
            "Score": f"{row.score:.2f}"
        }
        for label, tuples in row.values.items():
            value_str = ",".join(map(lambda t: f"{t.source_value} ({t.row_id})", tuples))
            row_data[label] = value_str
        table_data.append(row_data)

    # Create a header and format table for output
    headers = ["Row ID", "Score"] + labels
    table = tabulate(table_data, headers="keys", tablefmt="grid")

    # Print the table
    print(table)
