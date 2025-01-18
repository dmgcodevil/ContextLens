import heapq
import json
import os
import time
import uuid
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import List, Dict, Set
from collections import defaultdict
from openai import OpenAI
import re

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


class Input(ABC):
    pass


@dataclass
class FileInput(Input):
    path: str


@dataclass
class RawData(Input):
    s: str
    format: Format


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
    score is calculated as |t1.tokens ∩ t2.tokens |
    we say
        T1(value="a", tokens="a b")
        T2(value="a", tokens="a b c")
        T3(value="a", tokens="a c")

    score(T1,T2) > score(T1,T3)

    each tuple has weight associated with it
    we add weights from both tuples to the final score
    """
    if t1.label != t2.label:
        return 0.0
    score = len(t1.tokens_hashes | t2.tokens_hashes) * gain
    if score > 0.0:
        score += (t1.weight + t2.weight)
    return score


def strength_tuples(tuples1: List[Tuple], tuples2: List[Tuple], gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    total_strength = 0.0
    for t1 in tuples1:
        for t2 in tuples2:
            if t1.label == t2.label:
                total_strength += strength_tuple(t1, t2, gain)
    return total_strength


def best_strength_tuple(t1: Tuple, tuples2: List[Tuple], gain: float = DEFAULT_TUPLE_MATCH_GAIN) -> float:
    best_score = 0.0
    for t2 in tuples2:
        best_score = max(best_score, strength_tuple(t1, t2, gain))
    return best_score


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
        # old approach that doesn't take into account tuple weight
        # for label in row.unique_labels():
        #     bucket = self.buckets[label]
        #     if row.id in bucket.connected:
        #         for id, match_count in bucket.connected[row.id].items():
        #             if id not in res:
        #                 res[id] = 0
        #             res[id] = res[id] + match_count
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

    @staticmethod
    def fully_match(row: Row, filter: List[Tuple]) -> bool:
        count = 0
        for ft in filter:
            for t in row.tuples:
                if t.label == ft.label and t.value == ft.value:
                    count = count + 1
                    break
        return count == len(filter)

    def find_best_tuple(self, label: Label, source_row_id: str,
                        selected: Dict[Label, List[Tuple]]) -> None | SearchResult:
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


def load_file(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def save_dict_as_json(file_path, data):
    """
    Save a dictionary as a JSON file. If the file doesn't exist, it will be created.
    If it exists, it will be overwritten.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def append_suffix_to_filename(file_path, suffix, separator="_"):
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}{separator}{suffix}"
    return os.path.join(dir_name, new_file_name)


def merge(graph: Graph,
          columns: List[Label], **kwargs) -> List[DataRow]:
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
                                                                                  value_type=Type.UNKNOWN, label=col)]
            key = "|".join([str(selected[col]) for col in columns])
            if key not in table or total_score > table[key].score:
                table[key] = DataRow(id=row.id, score=total_score, values=selected)
    # print("\n".join(table.keys()))
    return list(table.values())


def call_openai(prompt: str, model: str = "gpt-4o") -> dict:
    """
    Calls OpenAI's API with the provided prompt and returns the structured response.

    Args:
        prompt (str): The prompt to send to the OpenAI API.
        model (str): The model to use (default is "gpt-4").

    Returns:
        dict: The parsed response from the OpenAI API.
    """
    try:
        print(f'call openai prompt: {prompt}')
        # Ensure the API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )

        # Call OpenAI API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        # Extract the response text
        print(f'response={response}')
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        print(f"Error parsing response as JSON: {e}")
        return {"nodes": []}


def format_prompt(labels: List[Label], file_path: str, file_format: Format, data: str) -> str:
    labels_str = ", ".join(label.name for label in labels)
    types_str = "|".join(t.value for t in Type)
    json_format = f"""
    {{
      "nodes": [
        {{
          "id": "<uuid>",                 // Unique identifier for the node
          "format": "<{types_str}>", // Input file format
          "tuples": [                     // Extracted entities
            {{
              "path": "<path>",            // Precise value location (e.g., json_path, xpath, csv(row:column), span(line:start:end))
              "value": "<normalized_value>", // Normalized value
              "source_value": "<original_value>", // Original value from the input ass string
              "label": "<label>",          // Extracted label (e.g., TRANSACTION_ID, TRANSACTION_AMOUNT)
              "value_type": "<int|double|bool|string|date|unknown>" // Determined value type
            }}
          ],
          "file_path": "<file_path>"       // Path to the input file
        }}
      ]
    }}
    """

    return f"""
    Given the following labels: [{labels_str}], extract entities from the {file_format.name} input data below.
    Group related tuples under separate nodes. 
    Ensure that tuples are normalized as per the rules:
    1. Dates are converted to mm-dd-YYYY format.
    2. Values are normalized to lowercase where applicable.
    3. The value type is determined as {types_str}, or marked as 'unknown' if not determinable.
    
    Example Input:
    ```json
    [
      {{
        "transaction": {{
          "id": 11111,
          "amount": 999.0,
          "date": "01062025"
        }}
      }},
      {{
        "transaction": {{
          "id": 22222,
          "amount": 777.0,
          "date": "01072025"
        }}
      }}
    ]
    ```
    
    Example Expected Output:
    
    ```json
    {{
      "rows": [
        {{
          "id": "<uuid>",
          "format": "json",
          "tuples": [
            {{
              "path": "$[0].transaction.id",
              "value": "11111",
              "source_value": "11111",
              "label": "TRANSACTION_ID",
              "value_type": "int"
            }},
            {{
              "path": "$[0].transaction.amount",
              "value": "999.0",
              "source_value": "999.0",
              "label": "TRANSACTION_AMOUNT",
              "value_type": "double"
            }},
            {{
              "path": "$[0].transaction.date",
              "value": "01-06-2025",
              "source_value": "01062025",
              "label": "TRANSACTION_DATE",
              "value_type": "date"
            }}
          ],
          "file_path": "./demo/transactions.json"
        }},
        {{
          "id": "<uuid>",
          "format": "json",
          "tuples": [
            {{
              "path": "$[1].transaction.id",
              "value": "22222",
              "source_value": "22222",
              "label": "TRANSACTION_ID",
              "value_type": "int"
            }},
            {{
              "path": "$[1].transaction.amount",
              "value": "777.0",
              "source_value": "777.0",
              "label": "TRANSACTION_AMOUNT",
              "value_type": "double"
            }},
            {{
              "path": "$[1].transaction.date",
              "value": "01-07-2025",
              "source_value": "01072025",
              "label": "TRANSACTION_DATE",
              "value_type": "date"
            }}
          ],
          "file_path": "./demo/transactions.json"
        }}
      ]
    }}
    ```
    
    Input File Path: `{file_path}`

    Input Data:
    ```
    {data}
    ```

    Output Format:
    ```json
    {json_format}
    ```
    
    Important: return only the JSON result. Do not include explanations, commentary, or any other text.
    
    """


def doc_to_rows(data) -> List[Row]:
    nodes = []
    for node_data in data["rows"]:
        row = Row(id=node_data["id"], tuples=[], format=Format(node_data["format"]),
                  file_path=node_data["file_path"])
        for tuple_index, tuple_data in enumerate(node_data["tuples"]):
            row.tuples.append(
                Tuple(
                    row_id=row.id,
                    index=tuple_index,
                    path=tuple_data["path"],
                    value=tuple_data["value"],
                    source_value=tuple_data["source_value"],
                    value_type=Type(tuple_data["value_type"]),
                    label=Label[tuple_data["label"]]
                )
            )
        nodes.append(row)

    return nodes


def extract_entities(file_path: str, labels: List[Label]) -> List[Row]:
    print(f'extract_entities from file: {file_path}')
    raw_data = load_file(file_path)

    if file_path.endswith('.json'):
        file_format = Format.JSON
    elif file_path.endswith('.xml'):
        file_format = Format.XML
    elif file_path.endswith('.csv'):
        file_format = Format.CSV
    else:
        file_format = Format.TEXT

    prompt = format_prompt(labels, file_path, file_format, raw_data)

    start_time = time.time()
    extracted_data = call_openai(prompt)
    end_time = time.time()
    execution_time = end_time - start_time
    for row_data in extracted_data["rows"]:
        row_data["id"] = str(uuid.uuid4())
    print(f'file={file_path}, took={execution_time:.4f} seconds, extracted entities={extracted_data}')
    entities_file_path = append_suffix_to_filename(file_path, "entities") + ".json"
    save_dict_as_json(entities_file_path, extracted_data)

    rows = []
    for row_data in extracted_data["rows"]:
        row = Row(id=row_data["id"], tuples=[], format=file_format, file_path=file_path)
        for tuple_index, tuple_data in enumerate(row_data["tuples"]):
            row.tuples.append(
                Tuple(
                    row_id=row.id,
                    index=tuple_index,
                    path=tuple_data["path"],
                    value=tuple_data["value"],
                    source_value=tuple_data["source_value"],
                    value_type=Type(tuple_data["value_type"]),
                    label=Label[tuple_data["label"]]
                )
            )
        rows.append(row)

    return rows


def load_entities(file_paths: List[str]) -> List[Row]:
    nodes = []
    for file_path in file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            nodes += doc_to_rows(data)
    return nodes


def parse_rows(input_text: str, initial_weights={}) -> List[Row]:
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
    input = """
    R1=[(L1,V1)]
    R2=[(L1,V1), (L2,V2)]
    R3=[(L1,V1), (L2,V3)]
    """

    initial_weights = {
        "R3": {0: 0.1}
    }

    rows = parse_rows(input, initial_weights)
    graph = create_graph(rows)

    print(rows)
    result = merge(graph, [Label.L1, Label.L2], row_filter={"R1"})

    print(result)

    # labels_list = [label for label in Label]
    # # extract_entities("./demo/transactions.json", labels_list)
    # # print(nodes)
    # rows = load_entities([
    #     "./demo/transactions_entities.json",
    #     "./demo/log_entities.json",
    #     "./demo/inventory_entities.json"
    # ])
    #
    # for row in rows:
    #     print(row)
    #
    # graph = create_graph(rows)
    # print("BUCKETS:")
    # for bucket in graph.buckets.values():
    #     print(bucket)
    # columns = [
    #     Label.TRANSACTION_ID,
    #     Label.TRANSACTION_DATE,
    #     Label.TRANSACTION_AMOUNT,
    #     Label.USER,
    #     Label.USER_ID,
    #     Label.PRODUCT_MAKE,
    #     Label.PRODUCT_MODEL,
    #     Label.PRODUCT_PRICE
    #
    # ]
    # graph.debug_info()
    #
    # result = merge(graph, columns)
    # for row in result:
    #     record_str = []
    #     for v in row.values:
    #         record_str.append(f'({v})')
    #     metadata = f'score={row.score}, row_id={row.id}'
    #     print(metadata + ",".join(record_str))
