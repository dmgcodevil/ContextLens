import json
import os
import time
import uuid
from abc import ABC
from dataclasses import dataclass
from dataclasses import field
from difflib import SequenceMatcher
from enum import IntEnum, Enum
from typing import List, Dict, Set

from openai import OpenAI

bucket_id_gen = 0


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
class Node:
    id: str
    tuples: List[Tuple]
    format: Format
    file_path: str

    def __repr__(self):
        return (f"Node(id={self.id}, tuples={self.tuples}, format={self.format.name}"
                f"file_path={self.file_path})")


@dataclass
class Extractor:
    label: Label
    name: str
    selector: str
    format: Format


@dataclass
class Bucket:
    id: str
    hash: int
    nodes: Set[str]
    files: Set[str]
    label_node_tuple: Dict[Label, Dict[str, Tuple]] = field(default_factory=dict)  # label -> node_id -> tuple

    def __repr__(self):
        label_details = []
        for label, node_tuples in self.label_node_tuple.items():
            tuples_str = "; ".join([f"{node_id}: {tup}" for node_id, tup in node_tuples.items()])
            label_details.append(f"{label.name}: [{tuples_str}]")
        label_details_str = "\n    ".join(label_details)

        return (
            f"Bucket(\n"
            f"  ID: {self.id}\n"
            f"  Hash: {self.hash}\n"
            f"  Nodes: {', '.join(self.nodes) if self.nodes else 'None'}\n"
            f"  Files: {', '.join(self.files) if self.files else 'None'}\n"
            f"  Label-Node Tuples:\n    {label_details_str if label_details else 'None'}\n"
            f")"
        )


def similarity_score(value1: str, value2: str) -> float:
    return SequenceMatcher(None, value1.lower(), value2.lower()).ratio()


def calculate_score(tuple_to_select, reference_tuple):
    score = similarity_score(tuple_to_select.value, reference_tuple.value)
    # optionally add weights based on source priority
    return score


@dataclass
class Graph:
    buckets: Dict[int, Bucket]
    nodes: Dict[str, Set[str]]


def build_buckets(nodes: List[Node]) -> Graph:
    """
    Build buckets using token-based hashing from tuple values.
    Each token's hash is used as a key to group similar tuples.
    """
    buckets: Dict[int, Bucket] = {}
    node_buckets: Dict[str, Set[str]] = {}
    for node in nodes:
        for t in node.tuples:
            print(f'build_buckets: process tuple={t} node_id={node.id}')
            token_hashes = t.token_hashes()
            print(f'build_buckets: tuple={t} has following hashes={token_hashes}')
            for token_hash in token_hashes:
                if token_hash not in buckets:
                    global bucket_id_gen
                    bucket_id = bucket_id_gen
                    bucket_id_gen = bucket_id_gen + 1
                    buckets[token_hash] = Bucket(id='b-' + str(bucket_id), hash=token_hash, nodes=set(), files=set())
                    print(f'build_buckets: create new bucket id={buckets[token_hash].id}, hash={token_hash}')
                bucket = buckets[token_hash]
                bucket.nodes.add(node.id)
                bucket.files.add(node.file_path)
                if t.label not in bucket.label_node_tuple:
                    bucket.label_node_tuple[t.label] = dict()
                # todo what to do if there is already a tuple exist for the given node(row)
                # i.e. row 1 tuples: [ ("id": "1"), ("tx_id" : 1) ] should we store a list of tuples ?
                # should we peak more suitable tuple ?
                bucket.label_node_tuple[t.label][node.id] = t
                if node.id not in node_buckets:
                    node_buckets[node.id] = set()
                node_buckets[node.id].add(bucket.id)  # connect node to bucket
    return Graph(buckets=buckets, nodes=node_buckets)


"""
merges all buckets into a single table where columns are given labels
"""


def merge(buckets: List[Bucket],
          node_buckets: Dict[str, Set[str]],
          labels: List[Label]) -> List[List[Tuple]]:
    print(f'process {len(buckets)} buckets')
    visited_buckets = set()
    buckets_dict = dict()
    rows = list()
    for bucket in buckets:
        buckets_dict[bucket.id] = bucket
    for bucket in buckets:
        # idea: we visit buckets using bfs. buckets can be connected:
        # n - node
        # t - tuple
        # b - buket
        # n1: {t1, t2}
        # t1: {label:id}
        # t2: {label:product}
        # b1: {id: {n1.id: t1}}
        # b2: {product: {n1.id: t2}}
        # b1 and b2 are connected
        # we add unprocessed bucket to `grouped_tuples`
        grouped_tuples = dict()
        if bucket.id not in visited_buckets:
            print(f'process bucket={bucket}')
            # we found new connected component
            visited_buckets.add(bucket.id)
            connected = set()
            selected: Dict[Label, List[Tuple]] = dict()
            stack = list()
            stack.append(bucket)
            while stack:
                curr = stack.pop()
                connected.add(curr.id)
                for label, row_tuple in curr.label_node_tuple.items():
                    if label not in selected:
                        selected[label] = list()
                    for node_id, t in row_tuple.items():
                        selected[label].append(t)
                        print(f'append tuple={t} to {label.name}')
                        for neighbor in node_buckets[node_id]:
                            if neighbor not in visited_buckets:
                                n_bucket = buckets_dict[neighbor]
                                print(f'{curr.id} -> {n_bucket.id}')
                                visited_buckets.add(neighbor)
                                stack.append(n_bucket)
            # create a row from collected tuples
            print(f'bucket: {bucket.id} connected: {connected}')
            row: List[Tuple] = list()
            for label in labels:
                if label not in selected:
                    row.append(Tuple(path="N/A", value="NA", source_value="N/A", value_type=Type.UNKNOWN, label=label))
                else:
                    tuples = selected[label]
                    # simple solution sort be tuple value length
                    tuples.sort(key=lambda t: len(t.value), reverse=True)
                    best_tuple = tuples[0]
                    row.append(best_tuple)
            rows.append(row)
    return rows


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
      "nodes": [
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


def doc_to_entities(data) -> List[Node]:
    nodes = []
    for node_data in data["nodes"]:
        node = Node(id=node_data["id"], tuples=[], format=Format(node_data["format"]),
                    file_path=node_data["file_path"])
        for tuple_data in node_data["tuples"]:
            node.tuples.append(
                Tuple(
                    path=tuple_data["path"],
                    value=tuple_data["value"],
                    source_value=tuple_data["source_value"],
                    value_type=Type(tuple_data["value_type"]),
                    label=Label[tuple_data["label"]]
                )
            )
        nodes.append(node)

    return nodes


def extract_entities(file_path: str, labels: List[Label]) -> List[Node]:
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
    for node_data in extracted_data["nodes"]:
        node_data["id"] = str(uuid.uuid4())
    print(f'file={file_path}, took={execution_time:.4f} seconds, extracted entities={extracted_data}')
    entities_file_path = append_suffix_to_filename(file_path, "entities") + ".json"
    save_dict_as_json(entities_file_path, extracted_data)

    nodes = []
    for node_data in extracted_data["nodes"]:
        node = Node(id=node_data["id"], tuples=[], format=file_format, file_path=file_path)
        for tuple_data in node_data["tuples"]:
            node.tuples.append(
                Tuple(
                    path=tuple_data["path"],
                    value=tuple_data["value"],
                    source_value=tuple_data["source_value"],
                    value_type=Type(tuple_data["value_type"]),
                    label=Label[tuple_data["label"]]
                )
            )
        nodes.append(node)

    return nodes


def load_entities(file_paths: List[str]) -> List[Node]:
    nodes = []
    for file_path in file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            nodes += doc_to_entities(data)
    return nodes


if __name__ == '__main__':
    labels_list = [label for label in Label]
    # extract_entities("./demo/transactions.json", labels_list)
    # print(nodes)
    nodes = load_entities([
        "./demo/simple/transactions_entities.json",
        "./demo/simple/log_entities.json",
        "./demo/simple/inventory_entities.json"
    ])

    for node in nodes:
        print(node)

    graph = build_buckets(nodes)
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
    table = merge(buckets=list(graph.buckets.values()), node_buckets=graph.nodes, labels=columns)
    print("TABLE:")
    for row in table:
        row_simple = [(t.label.name, t.source_value) for t in row]
        print(row_simple)
