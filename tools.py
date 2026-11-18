import json


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression using numbers and + - * /.",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def calculate(expression: str):
    allowed = set("0123456789+-*/(). ")
    if not expression or any(character not in allowed for character in expression):
        return {"error": "Invalid expression"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as error:
        return {"error": str(error)}


def run_tool(name: str, arguments: dict):
    if name == "calculate":
        return calculate(arguments.get("expression", ""))
    return {"error": f"Unknown tool: {name}"}
