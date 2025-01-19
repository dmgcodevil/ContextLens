import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import List, Dict, Set
import logging

logger = logging.getLogger("CustomLogger")
logger.setLevel(logging.DEBUG)

# constants
DEFAULT_TUPLE_MATCH_GAIN = 1.0
SELECTED_MATCH_GAIN = 1.5
PATH_PENALTY = 0.1

# type aliases
RowIdT = str
BucketIdT = str
HashcodeT = int


class Format(Enum):
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    TEXT = "text"


class Label(IntEnum):
    DATE = 1
    TRANSACTION_ID = 2
    TRANSACTION_AMOUNT = 3
    TRANSACTION_DATE = 4
    TRANSACTION_ACCOUNT = 5
    USER = 6
    USER_ID = 7
    PRODUCT_TYPE = 8
    PRODUCT_MAKE = 9
    PRODUCT_MODEL = 10
    PRODUCT_SCREEN_SIZE = 11
    PRODUCT_CPU = 12
    PRODUCT_RAM = 13
    PRODUCT_STORAGE_SIZE = 14
    PRODUCT_STORAGE_TYPE = 15
    PRODUCT_COLOR = 16
    PRODUCT_OS = 17
    PRODUCT_PRICE = 18
    ID = 19

    L1 = 20
    L2 = 21
    L3 = 22


class Type(Enum):
    INT = "int"
    DOUBLE = "double"
    BOOL = "bool"
    STRING = "string"
    DATE = "date"
    UNKNOWN = "unknown"


class Tokenizer:
    """
    Simple whitespace and punctuation tokenizer.
    """

    @staticmethod
    def tokenize(value: str) -> List[str]:
        return value.lower().split()


# tuples are entities extracted from the input
@dataclass(frozen=True)
class Tuple:
    row_id: str
    index: int
    path: str
    value: str
    source_value: str
    value_type: Type
    label: Label
    tokens: Dict[str, int]
    tokens_hashes: Set[int]
    weight: float

    @classmethod
    def create(cls, row_id: str, index: int, path: str, value: str,
               source_value: str, value_type: Type,
               label: Label,
               weight: float):
        tokens = set(Tokenizer.tokenize(value))  # dedup tokens
        hashed_tokens = {}
        tokens_hashes = set()
        for token in tokens:
            hashed_tokens[token] = hash(token)
            tokens_hashes.add(hashed_tokens[token])
        return cls(row_id, index, path, value, source_value, value_type,
                   label, hashed_tokens, tokens_hashes, weight)

    def __repr__(self):
        return (
            f"Tuple("
            f"row_id='{self.row_id}', "
            f"path='{self.path}', "
            f"value='{self.value}', "
            f"tokens='{self.tokens}', "
            f"source_value='{self.source_value}', "
            f"value_type={self.value_type.name}, "
            f"label={self.label.name}, "
            f"weight={self.weight}"
            f")"
        )


@dataclass
class Row:
    id: str
    tuples: List[Tuple]
    format: Format
    file_path: str

    def unique_labels(self) -> Set[Label]:
        labels = set()
        for t in self.tuples:
            labels.add(t.label)
        return labels

    def find_tuple_by_label(self, label: Label) -> None | Tuple:
        for t in self.tuples:
            if t.label == label:
                return t
        return None

    def __repr__(self):
        return (f"Row(id={self.id}, tuples={self.tuples}, format={self.format.name}, "
                f"file_path={self.file_path})")


@dataclass
class DataRow:
    id: str
    score: float
    values: Dict[Label, List[Tuple]]


@dataclass
class Node:
    row_id: str
    tuple_index: int
    # token_index: int
    token_hash: int
    value: str
    source_value: str


@dataclass
class Bucket:
    label: Label
    nodes: Dict[HashcodeT, List[Node]]
    connected: Dict[RowIdT, Dict[RowIdT, int]]  # row_id -> connected row + tuple match count

    def connect(self, row_id1: RowIdT, row_id2: RowIdT):
        if row_id1 != row_id2:
            if row_id1 not in self.connected:
                self.connected[row_id1] = defaultdict(int)
            if row_id2 not in self.connected:
                self.connected[row_id2] = defaultdict(int)
            self.connected[row_id1][row_id2] += 1
            self.connected[row_id2][row_id1] += 1


@dataclass
class SearchResult:
    score: float
    tuples: List[Tuple]


def strength_tuple(t1: Tuple, t2: Tuple, gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    """
    Calculates the similarity score between two tuples based on token overlap and tuple weights.

    The score is determined by:
        - The number of unique tokens shared between `t1` and `t2`.
        - A gain factor applied to the token overlap.
        - The sum of the weights of both tuples.

    Formula:
        score = (|tokens(t1) ∩ tokens(t2)| * gain) + (weight(t1) + weight(t2))

    Parameters:
        t1 (Tuple): The first tuple.
        t2 (Tuple): The second tuple.
        gain (float): A multiplier for token overlap (default: `DEFAULT_TUPLE_MATCH_GAIN`).

    Returns:
        float: The similarity score between `t1` and `t2`. Returns 0 if the labels of `t1` and `t2` do not match.

    Example:
        t1 = Tuple(label=L1, tokens={"a", "b"}, weight=0.5)
        t2 = Tuple(label=L1, tokens={"b", "c"}, weight=0.7)

        strength_tuple(t1, t2) == (1 * gain) + (0.5 + 0.7)
    """
    if t1.label != t2.label:
        return 0.0
    score = len(t1.tokens_hashes | t2.tokens_hashes) * gain
    if score > 0.0:
        score += (t1.weight + t2.weight)
    return score


def strength_tuples(tuples1: List[Tuple], tuples2: List[Tuple], gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    """
    Calculates the cumulative similarity score between two lists of tuples.

    For each tuple in `tuples1`, the score is computed with every tuple in `tuples2`
    that has the same label, using `strength_tuple`.

    Parameters:
        tuples1 (List[Tuple]): The first list of tuples.
        tuples2 (List[Tuple]): The second list of tuples.
        gain (float): A multiplier for token overlap (default: `DEFAULT_TUPLE_MATCH_GAIN`).

    Returns:
        float: The total similarity score across all tuple pairs.

    Example:
        tuples1 = [Tuple(label=L1, tokens={"a", "b"}, weight=0.5)]
        tuples2 = [Tuple(label=L1, tokens={"b", "c"}, weight=0.7),
                   Tuple(label=L2, tokens={"d"}, weight=0.4)]

        strength_tuples(tuples1, tuples2) == (1 * gain) + (0.5 + 0.7)
    """
    return sum(
        strength_tuple(t1, t2, gain)
        for t1 in tuples1
        for t2 in tuples2
        if t1.label == t2.label
    )


def best_strength_tuple(t1: Tuple, tuples2: List[Tuple], gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    """
    Calculates the maximum strength score for a single tuple against a list of tuples.

    Parameters:
        t1 (Tuple): The tuple to evaluate.
        tuples2 (List[Tuple]): The list of tuples to compare against.
        gain (float): A multiplier for token overlap (default: `DEFAULT_TUPLE_MATCH_GAIN`).

    Returns:
        float: The highest similarity score between `t1` and any tuple in `tuples2`.
               Returns 0 if `tuples2` is empty or no matching labels exist.

    Example:
        t1 = Tuple(label=L1, tokens={"a", "b"}, weight=0.5)
        tuples2 = [Tuple(label=L1, tokens={"b", "c"}, weight=0.7),
                   Tuple(label=L2, tokens={"d"}, weight=0.4)]

        best_strength_tuple(t1, tuples2) == (1 * gain) + (0.5 + 0.7)
    """
    return max(
        (strength_tuple(t1, t2, gain) for t2 in tuples2 if t1.label == t2.label),
        default=0.0
    )


def strength_selected(tuples: List[Tuple], selected: Dict[Label, List[Tuple]]) -> float:
    """
    Calculates the total strength of a list of tuples (`tuples`) with respect to a selected dictionary (`selected`).

    The strength is computed by iterating over each tuple `t1` in `tuples` and checking if its label exists in
    the keys of `selected`. For each matching label, the function computes the highest possible strength score
    between `t1` and all tuples associated with the label in `selected`, considering a specified gain factor.

    The highest score for a label is added to the total strength, ensuring that the most significant match
    contributes to the result.

    Formally, the total strength is given by:

        total_strength = Σ (max_strength(t1, selected[L])) for all t1 ∈ tuples, where:
            - L = t1.label
            - max_strength(t1, selected[L]) = max(best_strength_tuple(t1, t2, gain) for t2 ∈ selected[L])

    Parameters:
        tuples (List[Tuple]): A list of tuples to evaluate for strength.
        selected (Dict[Label, List[Tuple]]): A dictionary where keys are labels and values are lists of tuples
                                             representing pre-selected matches for the labels.

    Returns:
        float: The total strength score, which is the sum of the best-matching strength for each tuple in `tuples`.

    Notes:
        - This function assumes `best_strength_tuple` is defined and handles the scoring logic for individual tuples.
        - `SELECTED_MATCH_GAIN` is used as the gain factor when computing strength.

    """
    total_strength = 0.0
    for t1 in tuples:
        if t1.label in selected:
            total_strength += best_strength_tuple(t1, selected[t1.label], SELECTED_MATCH_GAIN)

    return total_strength


def is_perfect(tuples: List[Tuple], selected: Dict[Label, List[Tuple]]) -> bool:
    """
    Determines if a given list of tuples is "perfect" with respect to a selected dictionary.

    A list of tuples is considered "perfect" if, for every label `L` in the keys of `selected`,
    there exists a tuple `t1` in `tuples` such that:

        1. `t1.label == L` (matching label).
        2. `t1.value == t2.value` for some `t2` in `selected[L]` (strictly matching value).

    Formally, for each label `L` in `selected.keys()`:

        ∃ t1 ∈ tuples, ∃ t2 ∈ selected[L] such that:
            t1.label = L and t1.value = t2.value

    Returns:
        bool: True if `tuples` satisfies the above conditions for all labels in `selected`,
              otherwise False.

    Parameters:
        tuples (List[Tuple]): A list of tuples to check for perfection.
        selected (Dict[Label, List[Tuple]]): A dictionary where keys are labels and values
                                             are lists of tuples to match against.

    Example:
        tuples = [Tuple(label=L1, value="A"), Tuple(label=L2, value="B")]
        selected = {L1: [Tuple(label=L1, value="A")], L2: [Tuple(label=L2, value="B")]}

        is_perfect(tuples, selected)  # Returns True
    """
    matched_labels = set()
    for t1 in tuples:
        if t1.label in selected:
            for t2 in selected[t1.label]:
                if t1.value == t2.value:
                    matched_labels.add(t1.label)
                    break
    return len(selected.keys()) == len(matched_labels)


def strength(row1: Row, row2: Row, gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    return strength_tuples(row1.tuples, row2.tuples, gain)


class Graph:
    buckets: Dict[Label, Bucket] = dict()
    rows: Dict[RowIdT, Row] = dict()

    def get_nodes_by_label(self, label: Label) -> List[Node]:
        if label not in self.buckets:
            return []
        ids = []
        for nodes in self.buckets[label].nodes.values():
            ids += nodes
        return ids

    def get_rows_ids_by_label(self, label: Label) -> Set[str]:
        if label not in self.buckets:
            return set()
        ids = set()
        for nodes in self.buckets[label].nodes.values():
            for n in nodes:
                ids.add(n.row_id)
        return ids

    def get_directly_connected_rows_with_scores(self, row: Row) -> Dict[str, float]:
        res = {}
        ids = set()
        for label in row.unique_labels():
            bucket = self.buckets[label]
            if row.id in bucket.connected:
                ids.update(bucket.connected[row.id].keys())
        if row.id in ids:
            raise ValueError(f"row_id {row.id} connected to itself")
        for id in ids:
            res[id] = strength_tuples(row.tuples, self.rows[id].tuples)
        return res

    def add_tuple(self, row_id: str, t: Tuple, tuple_index: int):
        if t.label not in self.buckets:
            self.buckets[t.label] = Bucket(label=t.label, nodes=dict(), connected=dict())
        if len(t.tokens_hashes) > 2:
            print(f'multy-token tuple. row_id={row_id}, tuple_index={tuple_index}, tuple_value={t.value}')
        for token_index, token_hash in t.tokens.items():
            if token_hash not in self.buckets[t.label].nodes:
                self.buckets[t.label].nodes[token_hash] = []
            self.buckets[t.label].nodes[token_hash].append(
                Node(row_id=row_id, tuple_index=tuple_index,
                     token_hash=token_hash,
                     value=t.value,
                     source_value=t.source_value))

            bucket = self.buckets[t.label]
            for n in bucket.nodes[token_hash]:
                if n.row_id != row_id:
                    print(f'connect label={t.label.name} ,{n.row_id}->{row_id}')
                    bucket.connect(n.row_id, row_id)

    def add_row(self, row: Row):
        if row.id in self.rows:
            raise ValueError(f'duplicated row id={row.id}')
        self.rows[row.id] = row
        for index, value in enumerate(row.tuples):
            self.add_tuple(row.id, value, index)

    def node_to_tuple(self, n: Node) -> Tuple:
        return self.rows[n.row_id].tuples[n.tuple_index]

    def get_rows_ids_by_labels(self, labels: Set[Label]) -> Set[str]:
        ids = set()
        for label in labels:
            ids.update(self.get_rows_ids_by_label(label))
        return ids

    def _bfs_find_best(self, start_row: str, curr_row_id: str, label: Label, selected: Dict[Label, List[Tuple]],
                       initial_score) -> SearchResult | None:
        """
            Performs a weighted Breadth-First Search (BFS) to find the best tuple matching a given label.

            The BFS starts from the specified `curr_row_id`, exploring connected rows in descending order
            of their computed score. The score incorporates local strength, selected-match gain, and path penalty.

            Parameters:
                start_row (str): The ID of the starting row for the search.
                curr_row_id (str): The ID of the current row being evaluated.
                label (Label): The target label to find.
                selected (Dict[Label, List[Tuple]]): Previously selected label-value pairs to guide the search.
                initial_score (float): The starting score for the BFS.

            Returns:
                SearchResult | None: A `SearchResult` containing the highest-scoring tuple matching the label,
                                     or `None` if no such tuple is found.

            Notes:
                - Local strength is calculated using `strength_tuples`.
                - Selected-match gain is calculated using `strength_selected`.
                - Path penalty discourages longer paths unless they yield significantly higher scores.
            """
        pq = []
        visited = set()
        visited.add(start_row)
        heapq.heappush(pq, (-initial_score, curr_row_id, 0))  # Add path length as the third element

        while pq:
            neg_score, current_row_id, path_length = heapq.heappop(pq)
            current_score = -neg_score

            if current_row_id in visited:
                continue
            visited.add(current_row_id)

            # Check if current row contains the target label
            for t in self.rows[current_row_id].tuples:
                if t.label == label:
                    return SearchResult(score=current_score, tuples=[t])

            # Explore connected rows, i.e. rows that share the same labels
            candidates = self.get_directly_connected_rows_with_scores(self.rows[current_row_id])
            sorted_candidates = sorted(candidates, key=candidates.get, reverse=True)

            print(f'connected_row_ids={sorted_candidates}')
            for next_row_id in sorted_candidates:
                if next_row_id != curr_row_id:
                    local_score = candidates[next_row_id]

                    # Bonus for selected matches
                    selected_score = strength_selected(self.rows[next_row_id].tuples, selected)

                    # Apply path penalty
                    path_penalty = path_length * PATH_PENALTY

                    # Calculate total score
                    total_score = current_score + local_score + selected_score - path_penalty

                    heapq.heappush(pq, (-total_score, next_row_id, path_length + 1))
        return None

    def find_best_tuple(self, label: Label, source_row_id: str,
                        selected: Dict[Label, List[Tuple]]) -> None | SearchResult:
        """
            Finds the best tuple for a given label, starting from a source row.

            The function prioritizes "perfect" rows (those that fully match the `selected` dictionary) and
            then explores directly connected rows using `_bfs_find_best`. Scores for perfect rows and connected
            rows are computed using local strength, selected-match gain, and path penalty.

            Parameters:
                label (Label): The target label to find.
                source_row_id (str): The ID of the row to start the search from.
                selected (Dict[Label, List[Tuple]]): A dictionary of previously selected label-value pairs
                                                     to guide the search.

            Returns:
                SearchResult | None: The best tuple matching the label, or `None` if no match is found.

            Notes:
                - Perfect rows are initialized with a score equal to the number of selected label-value pairs.
                - If multiple tuples have the same highest score, all are included in the result.
        """
        print(f'Finding best value for label={label.name}, row_id={source_row_id}, selected={selected.values()}')
        if label not in self.buckets:
            return None

        best_result = SearchResult(score=0.0, tuples=[])
        # Step 1: Select starting rows based on selected
        # rows that fully match `selected` are called: perfect
        # We treat these perfect rows as high-priority BFS starts with an initial score of |selected|
        perfect_rows = {}  # id -> score
        if len(selected) > 0:
            for candidate in self.rows.values():
                if candidate.id != source_row_id and \
                        is_perfect(candidate.tuples, selected):
                    perfect_rows[candidate.id] = strength_selected(candidate.tuples, selected)

        # check perfect rows first
        for row_id in perfect_rows:
            res = self._bfs_find_best(source_row_id, row_id, label, selected, perfect_rows[row_id])
            if res.score > best_result.score:
                best_result = res
            elif res.score == best_result.score:
                best_result.tuples.extend(res.tuples)

        if len(best_result.tuples) > 0:
            # we found a tuple from perfect rows
            print(f'Found best value from perfect rows={perfect_rows}. label={label.name}, source row={source_row_id}: '
                  f'best result={best_result}')
            return best_result

        # Step2: try directly connected rows
        source_row = self.rows[source_row_id]
        # get all directly connected rows with strength scores
        candidates = self.get_directly_connected_rows_with_scores(source_row)
        # sort candidates by their scores in descending order
        sorted_candidates = sorted(candidates, key=candidates.get, reverse=True)
        print(f'candidates={sorted_candidates}')
        # we process rows in order: highest to lowest score
        for row_id in sorted_candidates:
            res = self._bfs_find_best(source_row_id, row_id, label, selected, candidates[row_id]
                                      + strength_selected(self.rows[row_id].tuples, selected)
                                      )
            if res.score > best_result.score:
                best_result = res
            elif res.score == best_result.score:
                best_result.tuples.extend(res.tuples)
        if len(best_result.tuples) > 0:
            print(
                f'Found best value for label={label.name}, row={source_row_id}: best_result{best_result}')
        else:
            print(f'No valid value found for label={label.name}, row={source_row_id}')

        return best_result

    def debug_info(self):
        for label, bucket in self.buckets.items():
            row_ids = set()
            for nodes in bucket.nodes.values():
                for n in nodes:
                    row_ids.add(n.row_id)
            print(f'bucket {label.name} has rows: {row_ids}')


def create_graph(rows: List[Row]) -> Graph:
    graph = Graph()
    for row in rows:
        graph.add_row(row)
    return graph


def merge(graph: Graph,
          columns: List[Label], **kwargs) -> List[DataRow]:
    """
        Merges rows in the graph into a cohesive table by resolving missing labels.

        For each row in the graph, the function attempts to find the best value for each label
        in `columns` by leveraging `find_best_tuple`. The resulting table includes only the rows
        matching the optional row filter and assigns scores to reflect the quality of the merges.

        Parameters:
            graph (Graph): The graph containing rows and their connections.
            columns (List[Label]): The list of labels to resolve for each row.
            **kwargs: Optional keyword arguments, including:
                - `row_filter` (Set[str]): A set of row IDs to include in the merge.

        Returns:
            List[DataRow]: A list of `DataRow` objects representing the merged rows with their scores.

        Notes:
            - If no value is found for a label, a placeholder tuple with `N/A` is used.
            - Rows are filtered based on the `row_filter` argument, if provided.
            - The function maintains a table of primary keys to ensure unique row outputs.
    """
    row_filter = set()
    if "row_filter" in kwargs:
        row_filter = kwargs["row_filter"]
    table = {}  # primary key ->? (score,row)
    for row in graph.rows.values():
        total_score = 0.0
        if len(row_filter) == 0 or row.id in row_filter:
            selected_rows = set()
            selected: Dict[Label, List[Tuple]] = {}
            for col in columns:
                if col not in selected:
                    res = graph.find_best_tuple(col, row.id, selected)
                    print(f'merge label={col} result={res}, selected_rows={selected_rows}')
                    print('=' * 100)
                    if res:
                        total_score += res.score
                        selected[col] = res.tuples
                    else:
                        # if we couldn't find a value use it from current row
                        existing = row.find_tuple_by_label(col)
                        selected[col] = [existing] if existing else [Tuple.create("N/A", 0, path="N/A", value="N/A",
                                                                                  source_value="N/A",
                                                                                  value_type=Type.UNKNOWN, label=col,
                                                                                  weight=0.0)]
            values = []
            for col in columns:
                col_values = []
                for t in selected[col]:
                    col_values.append(t.value)
                values.append(f"[{",".join(col_values)}]")
            key = "|".join(values)
            if key not in table or total_score > table[key].score:
                table[key] = DataRow(id=row.id, score=total_score, values=selected)
        print("\n".join(table.keys()))
    return list(table.values())


def _doc_to_rows(data) -> List[Row]:
    nodes = []
    for node_data in data["rows"]:
        row = Row(id=node_data["id"], tuples=[], format=Format(node_data["format"]),
                  file_path=node_data["file_path"])
        for tuple_index, tuple_data in enumerate(node_data["tuples"]):
            row.tuples.append(
                Tuple.create(
                    row_id=row.id,
                    index=tuple_index,
                    path=tuple_data["path"],
                    value=tuple_data["value"],
                    source_value=tuple_data["source_value"],
                    value_type=Type(tuple_data["value_type"]),
                    label=Label[tuple_data["label"]],
                    weight=0.0
                )
            )
        nodes.append(row)

    return nodes


def load_entities(file_paths: List[str]) -> List[Row]:
    nodes = []
    for file_path in file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            nodes += _doc_to_rows(data)
    return nodes


def _parse_rows(input_text: str, initial_weights={}) -> List[Row]:
    rows = []
    pattern = re.compile(r"(R\d+)=\[(.*?)\]")  # matches R1=[(L1,V1)]
    tuple_pattern = re.compile(r"\((L\d+),([^)]+)\)")  # matches (L1,V1)

    for match in pattern.finditer(input_text):
        row_id = match.group(1)  # extract row ID, e.g., R1
        tuple_str = match.group(2)  # extract tuples, e.g., (L1,V1), (L2,V2)

        tuples = []
        for index, tuple_match in enumerate(tuple_pattern.finditer(tuple_str)):
            label = tuple_match.group(1)  # extract label, e.g., L1
            value = tuple_match.group(2)  # extract value, e.g., V1
            weight = 0.0
            if row_id in initial_weights and index in initial_weights[row_id]:
                weight = initial_weights[row_id][index]
            tuples.append(Tuple.create(
                row_id=row_id,
                index=index,
                path="",
                value=value,
                source_value=value,
                value_type=Type.STRING,
                label=getattr(Label, label),
                weight=weight
            ))

        rows.append(Row(id=row_id, tuples=tuples, format=Format.TEXT, file_path=""))

    return rows


if __name__ == '__main__':
    # input = """
    # R1=[(L1,V1)]
    # R2=[(L1,V1), (L2,V2)]
    # R3=[(L1,V1), (L2,V3)]
    # """
    #
    # initial_weights = {
    #     "R3": {0: 0.1}
    # }
    #
    # rows = parse_rows(input, initial_weights)
    # graph = create_graph(rows)
    #
    # print(rows)
    # result = merge(graph, [Label.L1, Label.L2], row_filter={"R1"})
    #
    # print(result)

    labels_list = [label for label in Label]
    # extract_entities("./demo/transactions.json", labels_list)
    # print(nodes)
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
