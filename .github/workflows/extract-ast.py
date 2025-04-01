import os
import sys
import json
from tree_sitter import Parser
from tree_sitter_language_pack import get_language

parser = Parser()
cpp_lang = get_language("cpp")
parser.set_language(cpp_lang)

def get_text(src, node):
    return src[node.start_byte:node.end_byte].decode("utf-8")

def extract_methods(class_node, source_code):
    methods = []
    for child in class_node.children:
        if child.type == "field_declaration_list":
            for decl in child.children:
                if decl.type == "function_definition":
                    method = {
                        "constructor": False,
                        "destructor": False,
                        "name": None,
                        "return_type": None,
                        "params": []
                    }
                    for sub in decl.children:
                        if sub.type == "type_descriptor":
                            method["return_type"] = get_text(source_code, sub).strip()
                        elif sub.type == "declarator":
                            idents = [n for n in sub.children if n.type == "identifier"]
                            if idents:
                                method["name"] = get_text(source_code, idents[0]).strip()
                            if '(' in get_text(source_code, sub):
                                param_list = [n for n in sub.children if n.type == "parameter_list"]
                                if param_list:
                                    for param in param_list[0].children:
                                        if param.type == "parameter_declaration":
                                            type_node = next((c for c in param.children if c.type != ','), None)
                                            if type_node:
                                                method["params"].append(get_text(source_code, type_node).strip())
                    if method["name"]:
                        if method["name"].startswith("~"):
                            method["destructor"] = True
                        elif method["name"] in class_names:
                            method["constructor"] = True
                        methods.append(method)
    return methods

def extract_class_info(path):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".cpp", ".h", ".cc", ".hpp")):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "rb") as f:
                        code = f.read()
                    tree = parser.parse(code)
                    root_node = tree.root_node
                    for node in root_node.children:
                        if node.type == "class_specifier":
                            class_name = None
                            bases = []
                            for child in node.children:
                                if child.type == "type_identifier":
                                    class_name = get_text(code, child)
                                    class_names.add(class_name)
                                elif child.type == "base_class_clause":
                                    bases = [get_text(code, c) for c in child.children if c.type == "type_identifier"]
                            if class_name:
                                methods = extract_methods(node, code)
                                print(json.dumps({
                                    "file": os.path.relpath(full_path, path),
                                    "class": class_name,
                                    "inherits": bases,
                                    "methods": methods
                                }))
                except Exception as e:
                    print(f"# ERROR parsing {full_path}: {e}", file=sys.stderr)

class_names = set()
with open("lammps-gpt-ast.jsonl", "w") as out:
    sys.stdout = out
    extract_class_info("lammps")
