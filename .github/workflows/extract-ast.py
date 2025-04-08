import os
import sys
import json
from tree_sitter import Parser
from tree_sitter_language_pack import get_parser

parser = get_parser('cpp') 

def get_text(src, node):
    return src[node.start_byte:node.end_byte].decode("utf-8")

def method_flags(decl_node):
    flags = {
        "static": False,
        "virtual": False,
        "inline": False,
        "pure_virtual": False,
        "const": False
    }
    for child in decl_node.children:
        t = child.type
        if t == "virtual":
            flags["virtual"] = True
        elif t == "static":
            flags["static"] = True
        elif t == "inline":
            flags["inline"] = True
        elif t == "function_definition":
            for fchild in child.children:
                if fchild.type == "declaration_specifiers":
                    if any(grandchild.type == "virtual" for grandchild in fchild.children):
                        flags["virtual"] = True
                if fchild.type == "function_declarator":
                    if get_text(code, fchild).strip().endswith("= 0"):
                        flags["pure_virtual"] = True
    return flags

def extract_methods(class_node, code):
    methods = []
    current_access = "private"

    for child in class_node.children:
        if child.type == "access_specifier":
            current_access = get_text(code, child).strip().rstrip(":")
        elif child.type == "field_declaration_list":
            for decl in child.children:
                if decl.type != "function_definition":
                    continue
                method = {
                    "constructor": False,
                    "destructor": False,
                    "name": None,
                    "return_type": None,
                    "params": [],
                    "const": False,
                    "access": current_access,
                    "flags": method_flags(decl)
                }

                for sub in decl.children:
                    if sub.type == "type_descriptor":
                        method["return_type"] = get_text(code, sub).strip()
                    elif sub.type == "declarator" or sub.type == "function_declarator":
                        idents = [n for n in sub.children if n.type == "identifier"]
                        if idents:
                            method["name"] = get_text(code, idents[0]).strip()

                        if get_text(code, sub).strip().endswith("const"):
                            method["const"] = True

                        param_list = [n for n in sub.children if n.type == "parameter_list"]
                        if param_list:
                            for param in param_list[0].children:
                                if param.type == "parameter_declaration":
                                    type_node = next((c for c in param.children if c.type != ','), None)
                                    if type_node:
                                        method["params"].append(get_text(code, type_node).strip())

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
            print(f"*** file {file}")
            if file.endswith((".cpp", ".cc", ".h", ".hpp")):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "rb") as f:
                        code = f.read()
                    print(f"*** code {code}")
                    tree = parser.parse(code)
                    root_node = tree.root_node
                    for node in root_node.children:
                        template_params = []
                        class_node = None

                        if node.type == "template_declaration":
                            class_node = next((c for c in node.children if c.type == "class_specifier"), None)
                            if not class_node:
                                continue
                            template_node = next((c for c in node.children if c.type == "template_parameter_list"), None)
                            if template_node:
                                template_params = [get_text(code, c) for c in template_node.children if c.type != ","]
                        elif node.type == "class_specifier":
                            class_node = node

                        if not class_node:
                            continue

                        class_name = None
                        bases = []
                        for child in class_node.children:
                            if child.type == "type_identifier":
                                class_name = get_text(code, child)
                                class_names.add(class_name)
                            elif child.type == "base_class_clause":
                                bases = [get_text(code, c) for c in child.children if c.type == "type_identifier"]

                        if class_name:
                            methods = extract_methods(class_node, code)
                            print(json.dumps({
                                "file": os.path.relpath(full_path, path),
                                "class": class_name,
                                "inherits": bases,
                                "templates": template_params,
                                "methods": methods
                            }))
                except Exception as e:
                    print(f"# ERROR parsing {full_path}: {e}", file=sys.stderr)

class_names = set()
with open("lammps-gpt-ast.jsonl.txt", "w") as out:
    sys.stdout = out
    extract_class_info("lammps/src")
