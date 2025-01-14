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

# constants
DEFAULT_MATCH_GAIN = 1.0
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

    def tokenized_values(self) -> List[str]:
        """
        Tokenize the value for more flexible matching.
        Simple whitespace and punctuation tokenizer.
        """
        tokens = self.value.lower().split()
        return tokens

    def token_hashes(self) -> List[int]:
        """
        Generate hashes for each token combined with the label.
        """
        return [hash((token, self.label)) for token in self.tokenized_values()]

    def __repr__(self):
        return (
            f"Tuple("
            f"path='{self.path}', "
            f"value='{self.value}', "
            f"source_value='{self.source_value}', "
            f"value_type={self.value_type.name}, "
            f"label={self.label.name}"
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
        return (f"Row(id={self.id}, tuples={self.tuples}, format={self.format.name}"
                f"file_path={self.file_path})")


@dataclass
class DataRow:
    score: float
    tuples: List[Tuple]


@dataclass
class Node:
    row_id: str
    tuple_index: int
    token_index: int
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
    row_id: str
    t: Tuple


def strength_tuples(tuples1: List[Tuple], tuples2: List[Tuple], gain: float = DEFAULT_MATCH_GAIN) -> float:
    total_strength = 0.0
    for t1 in tuples1:
        for t2 in tuples2:
            if t1.label == t2.label:
                if t1.value == t2.value:
                    total_strength += gain
    return total_strength


def strength(row1: Row, row2: Row, gain: float = DEFAULT_MATCH_GAIN) -> float:
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
        for label in row.unique_labels():
            bucket = self.buckets[label]
            if row.id in bucket.connected:
                for id, match_count in bucket.connected[row.id].items():
                    if id not in res:
                        res[id] = 0
                    res[id] = res[id] + match_count
        return res

    def add_tuple(self, row_id: str, t: Tuple, tuple_index: int):
        if t.label not in self.buckets:
            self.buckets[t.label] = Bucket(label=t.label, nodes=dict(), connected=dict())
        token_hashes = t.token_hashes()
        if len(token_hashes) > 2:
            print(f'multy-token tuple. row_id={row_id}, tuple_index={tuple_index}, tuple_value={t.value}')
        for token_index, token_hash in enumerate(t.token_hashes()):
            if token_hash not in self.buckets[t.label].nodes:
                self.buckets[t.label].nodes[token_hash] = []
            self.buckets[t.label].nodes[token_hash].append(
                Node(row_id=row_id, tuple_index=tuple_index, token_index=token_index,
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

    def _bfs_find_best(self, start_row: str, curr_row_id: str, label: Label, selected: Dict[Label, Tuple],
                       initial_score) -> SearchResult | None:

        """
            Perform a weighted BFS to find the row containing the specified label with the highest
            total score, starting from a candidate row and initial score.

            This method uses a max-heap (implemented by pushing negative scores) to prioritize paths
            with the greatest "score so far." Once a row that contains the target label is encountered,
            the search terminates, returning that row's tuple for the label.

            Args:
                start_row (str):
                    The ID of the row from which this BFS search is conceptually "originating."
                    Used to mark that row as visited initially (to avoid revisiting it).
                curr_row_id (str):
                    The row ID where we begin the actual BFS expansions. This is enqueued in the
                    priority queue with 'initial_score'.
                label (Label):
                    The target label we want to locate. Once we reach a row that has this label,
                    we return its tuple.
                selected (Dict[Label, Tuple]):
                    A dictionary of already chosen label-value pairs for the row we're trying to complete.
                    Rows that match these pairs receive an extra bonus (SELECTED_MATCH_GAIN).
                initial_score (float):
                    The initial cumulative score assigned to `curr_row_id` when pushed into the queue.
                    This can incorporate local strength from the source row or a bonus for perfect matches.

            Returns:
                Optional[SearchResult]:
                    A SearchResult object if we successfully find a row containing `label`, along with
                    the BFS-accumulated score and the tuple for that label. If no such row is found,
                    returns `None`.

            Algorithm Details:
                1. We store states in a priority queue (pq) as (-score, row_id, path_length).
                   This ensures that the highest score is always popped first (since we use negative scores).
                2. We mark 'start_row' as visited initially, so we don't circle back to it.
                3. For each row we pop:
                   - If it's already visited, skip it.
                   - Otherwise, mark visited and check if it contains the target label:
                     * If so, return the current SearchResult (with 'current_score').
                   - If not, compute local_strength and selected-match bonuses for each connected neighbor,
                     subtract a path penalty (PATH_PENALTY * path_length), and enqueue the neighbor state
                     with the updated cumulative score.
                4. If we exhaust the queue with no row containing the target label, return None.

            Complexity:
                - Let N be the number of rows. In the worst case, we may enqueue up to O(N) states.
                - Each BFS pop leads to checking neighbors, which can be up to O(N) in a fully connected graph.
                - Total worst-case time could be O(N^2) for dense graphs, though often sparser connectivity
                  or early termination leads to better practical performance.
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
                    return SearchResult(score=current_score, row_id=current_row_id, t=t)

            # Explore connected rows, i.e. rows that share the same labels
            candidates = self.get_directly_connected_rows_with_scores(self.rows[current_row_id])
            sorted_candidates = sorted(candidates, key=candidates.get, reverse=True)

            print(f'connected_row_ids={sorted_candidates}')
            for next_row_id in sorted_candidates:
                if next_row_id != curr_row_id:
                    local_score = candidates[next_row_id]

                    # Bonus for selected matches
                    selected_score = strength_tuples(self.rows[next_row_id].tuples, list(selected.values()),
                                                     SELECTED_MATCH_GAIN)

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

    def find_best_tuple(self, label: Label, source_row_id: str, selected: Dict[Label, Tuple]) -> None | SearchResult:
        """
            Attempt to find the best tuple for a given label within the graph, starting from the specified
            source row and taking into account already selected label-value pairs.

            This method implements a two-phase strategy:
              1. Identify "perfect rows," which are rows that fully match all label-values in `selected`.
                 These perfect rows are explored via BFS first, each with an initial score equal to
                 the size of `selected`.
              2. If no perfect row yields a valid result, we look at the rows directly connected to
                 the `source_row` (sorted by their local strength score), and run BFS from them,
                 giving each a combined initial score (local strength + selected-match bonus).

            Args:
                label (Label):
                    The target label (e.g., L1, L2, etc.) for which we want to find the best matching
                    row.
                source_row_id (str):
                    The ID of the row from which we conceptually initiate the search. We only want
                    to fill a missing label for this row.
                selected (Dict[Label, Tuple]):
                    A dictionary of label-value pairs that have already been chosen for the
                    `source_row`. Rows matching these pairs gain an extra bonus when BFS expansions
                    are scored.

            Returns:
                Optional[SearchResult]:
                    A SearchResult object if the BFS discovers a row that contains the requested
                    label, or None if no such row (with label `label`) is found. The SearchResult
                    includes:
                      - The final BFS-accumulated `score` for that path.
                      - The `row_id` of the discovered row containing `label`.
                      - The tuple `t` from that row matching the label.

            Algorithm Steps:
                1. **Perfect Rows**:
                   - We iterate over all rows, checking if each one fully matches every
                     label-value in `selected`.
                   - Those that match are considered "perfect," and BFS is called on each with
                     an initial score equal to `len(selected)`.
                   - We keep track of the highest-scoring BFS result from this set.

                2. **Directly Connected Rows**:
                   - If no satisfactory tuple is found from perfect rows, we compute local
                     strength scores between `source_row` and its directly connected neighbors.
                   - We sort these neighbors in descending order of local strength, and run BFS
                     from each, adding a bonus for how many items they match from `selected`.
                   - We again track the highest-scoring BFS result.

                3. We pick the best BFS outcome from either phase:
                   - If BFS finds a row that has `label`, we compare its final BFS `score` to any
                     existing best result. If it's higher, we store it.
                   - We return once all candidates are processed or we confirm a row with the best
                     final score.

            Example:
                Suppose `source_row_id` = "R2" is missing label L2, and we have `selected` =
                {L1: (L1, "V1", ...), L3: (L3, "V3", ...)}. First, we find rows that fully match
                both (L1, V1) and (L3, V3). If none match, we get all neighbors of R2, sort them,
                and BFS from each with their local strength + selected-match bonus. The BFS expansions
                incorporate path penalties and synergy scores. Ultimately, we pick the row whose BFS
                expansions yield the highest final score for label L2.

            Side Effects:
                - Debug prints show the BFS expansions and the best result found for the label.

            Complexity:
                - Phase 1 (finding perfect rows) is O(N) where N is the total number of rows.
                - Phase 2, for each directly connected candidate, we do a BFS. In the worst case,
                  BFS might visit many rows. However, BFS typically terminates early if it finds
                  a row containing `label`.
                - This approach is often efficient for smaller or moderately sized graphs, or if
                  the BFS expansions typically find matching rows quickly.

            Note:
                - This function internally calls `_bfs_find_best(...)` to perform the BFS with
                  an initial score. If both phases yield no solution (no row found with `label`),
                  it returns None.

            See Also:
                - `_bfs_find_best`: The BFS function that calculates scores and expansions.
                - `strength_tuples`: Used to check how many label-value pairs match between two
                  sets of tuples.
            """
        print(f'Finding best value for label={label.name}, row_id={source_row_id}, selected={selected.values()}')
        if label not in self.buckets:
            return None

        best_result: SearchResult | None = None
        # Step 1: Select starting rows based on selected
        # rows that fully match `selected` are called: perfect
        # We treat these perfect rows as high-priority BFS starts with an initial score of |selected|
        perfect_rows = set()
        if len(selected) > 0:
            for candidate in self.rows.values():
                if candidate.id != source_row_id and \
                        strength_tuples(candidate.tuples, list(selected.values())) == len(selected):
                    perfect_rows.add(candidate.id)

        # check perfect rows first
        for row_id in perfect_rows:
            res = self._bfs_find_best(source_row_id, row_id, label, selected, len(selected))
            if res:
                # all perfect rows have same initial score = len(selected)
                # thus we are looking for result with the highest score
                if not best_result or res.score > best_result.score:
                    best_result = res

        if best_result:
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
                                      + strength_tuples(self.rows[row_id].tuples, list(selected.values()),
                                                        SELECTED_MATCH_GAIN)
                                      )
            if res:
                if not best_result or res.score > best_result.score:
                    print(f'Found better value for label={label.name}, row={source_row_id}: old={best_result}, '
                          f'new: {res}')
                    best_result = res

        if best_result:
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
          primary_key: Set[Label],
          columns: List[Label], row_filter=None) -> List[DataRow]:
    if row_filter is None:
        row_filter = set()
    table = {}  # primary key: (score,row)
    unique = set()
    for row in graph.rows.values():
        total_score = 0.0
        if len(row_filter) == 0 or row.id in row_filter:
            selected_rows = set()
            selected: Dict[Label, Tuple] = {}
            for col in columns:
                if col not in selected:
                    current_t = row.find_tuple_by_label(col)
                    if current_t:
                        selected[col] = current_t
                        continue

                    res = graph.find_best_tuple(col, row.id, selected)
                    print(f'merge res={res}, selected_rows={selected_rows}')
                    print('=' * 100)
                    selected[col] = res.t if res else Tuple("N/A", 0, path="N/A", value="N/A", source_value="N/A",
                                                            value_type=Type.UNKNOWN, label=col)
                    if res:
                        total_score += res.score
                        selected_rows.add(res.row_id)
            key = "|".join([selected[col].value for col in primary_key])
            if key not in table or total_score > table[key].score:
                table[key] = DataRow(score=total_score, tuples=[selected[col] for col in columns])
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


if __name__ == '__main__':
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

    result = merge(graph, {Label.TRANSACTION_ID}, columns)
    for row in result:
        record_str = []
        for t in row.tuples:
            record_str.append(f'({t.label.name},{t.source_value})')
        print(f'score={row.score}' + ",".join(record_str))
